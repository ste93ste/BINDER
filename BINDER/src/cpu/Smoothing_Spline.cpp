#include "Smoothing_Spline.h"
#include <cmath>
#include <cstring>

Smoothing_Spline::Smoothing_Spline(const int spline_order, const int threads)
    : spline_order(spline_order), threads(threads) {

    // Precompute B-spline values
    bspline_values.resize(LOOKUP_RESOLUTION);
    for (int i = 0; i < LOOKUP_RESOLUTION; ++i) {
        const double x = static_cast<double>(i) / (LOOKUP_RESOLUTION * 0.5);
        bspline_values[i] = compute_bspline(x);
    }

    // Set Eigen thread count
    Eigen::setNbThreads(threads);
}

Smoothing_Spline::~Smoothing_Spline() = default;

double Smoothing_Spline::compute_bspline(const double x) const {
    if (spline_order == 0) {
        return (x < 0.5) + (x == 0.5) * 0.5;
    }
    if (spline_order == 1) {
        return (x < 1.0) ? (1.0 - x) : 0.0;
    }
    if (spline_order == 3) {
        if (x < 1.0) {
            const double x2 = x * x;
            const double x3 = x2 * x;
            return 2.0 / 3.0 - x2 + 0.5 * x3;
        }
        if (x < 2.0) {
            const double dx = 2.0 - x;
            const double dx2 = dx * dx;
            return dx2 * dx / 6.0;
        }
    }
    return 0.0;
}

inline double Smoothing_Spline::evaluate_B_spline(const double y, const double shift) const {
    const double x = std::abs(y - shift);
    if (x >= 2.0) return 0.0;

    const int index = static_cast<int>(x * (LOOKUP_RESOLUTION * 0.5));
    if (index >= LOOKUP_RESOLUTION) return 0.0;

    return bspline_values[index];
}

void Smoothing_Spline::perform_least_squares_splines_1D(
    const std::vector<double>& t,
    std::vector<double>& t_down,
    const int length,
    const int down_length,
    const int down_factor) const {

    const MatrixCacheKey key{length, down_length, down_factor};
    MatrixCacheEntry* entry_ptr = nullptr;

    // Get or create cached matrices
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto& entry = matrix_cache[key];
        if (entry.A.rows() == 0) {  // Not computed yet
            Eigen::MatrixXd A(length, down_length);
            #pragma omp parallel for collapse(2) if(length * down_length > 1000)
            for (int i = 0; i < length; ++i) {
                for (int j = 0; j < down_length; ++j) {
                    A(i, j) = evaluate_B_spline(static_cast<double>(i) / down_factor, j);
                }
            }
            entry = MatrixCacheEntry(A);
        }
        entry_ptr = &entry;
    }

    // Map input to Eigen vector
    const Eigen::Map<const Eigen::VectorXd> b(t.data(), length);

    // Solve system using cached decomposition
    Eigen::VectorXd coeffs = entry_ptr->ldlt.solve(entry_ptr->A.transpose() * b);

    // Compute final result
    #pragma omp parallel for if(down_length > 1000)
    for (int i = 0; i < down_length; ++i) {
        double sum = 0.0;
        for (int j = 0; j < down_length; ++j) {
            sum += entry_ptr->A(i * down_factor, j) * coeffs(j);
        }
        t_down[i] = sum;
    }
}

void Smoothing_Spline::perform_least_squares_splines_2D(
    const double* im,
    double* im_down,
    const int Nx, const int Ny,
    const int down_factor) const {

    const int new_y = static_cast<int>(std::ceil(static_cast<double>(Ny) / down_factor));
    const int new_x = static_cast<int>(std::ceil(static_cast<double>(Nx) / down_factor));

    std::vector<double> temp1(Nx * new_y);

    #pragma omp parallel
    {
        std::vector<double> y_line(Ny);
        std::vector<double> y_down_line(new_y);
        std::vector<double> x_line(Nx);
        std::vector<double> x_down_line(new_x);

        // Process y-direction
        #pragma omp for schedule(static)
        for (int x = 0; x < Nx; ++x) {
            std::memcpy(y_line.data(), &im[x * Ny], Ny * sizeof(double));
            perform_least_squares_splines_1D(y_line, y_down_line, Ny, new_y, down_factor);
            for (int y = 0; y < new_y; ++y) {
                temp1[x * new_y + y] = y_down_line[y];
            }
        }

        // Process x-direction
        #pragma omp for schedule(static)
        for (int y = 0; y < new_y; ++y) {
            for (int x = 0; x < Nx; ++x) {
                x_line[x] = temp1[x * new_y + y];
            }
            perform_least_squares_splines_1D(x_line, x_down_line, Nx, new_x, down_factor);
            for (int x = 0; x < new_x; ++x) {
                im_down[x * new_y + y] = x_down_line[x];
            }
        }
    }
}

void Smoothing_Spline::perform_least_squares_splines_3D(
    const double* im,
    double* im_down,
    const int Nx, const int Ny, const int Nz,
    const int down_factor) const {

    const int new_z = static_cast<int>(std::ceil(static_cast<double>(Nz) / down_factor));
    const int new_y = static_cast<int>(std::ceil(static_cast<double>(Ny) / down_factor));
    const int new_x = static_cast<int>(std::ceil(static_cast<double>(Nx) / down_factor));

    const int ny_nz = Ny * Nz;

    std::vector<double> temp1(Nx * Ny * new_z);
    std::vector<double> temp2(Nx * new_y * new_z);

    #pragma omp parallel
    {
        std::vector<double> z_line(Nz);
        std::vector<double> z_down_line(new_z);
        std::vector<double> y_line(Ny);
        std::vector<double> y_down_line(new_y);
        std::vector<double> x_line(Nx);
        std::vector<double> x_down_line(new_x);

        // Process z-direction
        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                const int base_idx = x * ny_nz + y * Nz;
                std::memcpy(z_line.data(), &im[base_idx], Nz * sizeof(double));

                perform_least_squares_splines_1D(z_line, z_down_line, Nz, new_z, down_factor);

                const int out_idx = (x * Ny + y) * new_z;
                std::memcpy(&temp1[out_idx], z_down_line.data(), new_z * sizeof(double));
            }
        }

        // Process y-direction
        #pragma omp for collapse(2) schedule(static)
        for (int z = 0; z < new_z; ++z) {
            for (int x = 0; x < Nx; ++x) {
                for (int y = 0; y < Ny; ++y) {
                    y_line[y] = temp1[(x * Ny + y) * new_z + z];
                }

                perform_least_squares_splines_1D(y_line, y_down_line, Ny, new_y, down_factor);

                for (int y = 0; y < new_y; ++y) {
                    temp2[(x * new_y + y) * new_z + z] = y_down_line[y];
                }
            }
        }

        // Process x-direction
        #pragma omp for collapse(2) schedule(static)
        for (int z = 0; z < new_z; ++z) {
            for (int y = 0; y < new_y; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    x_line[x] = temp2[(x * new_y + y) * new_z + z];
                }

                perform_least_squares_splines_1D(x_line, x_down_line, Nx, new_x, down_factor);

                for (int x = 0; x < new_x; ++x) {
                    im_down[(x * new_y + y) * new_z + z] = x_down_line[x];
                }
            }
        }
    }
}

void Smoothing_Spline::smooth_and_downsample_image(
    const double* im, double* im_down,
    const int Nx, const int Ny, const int Nz,
    const int down_factor) const {

    if (Nz == 1) {
        perform_least_squares_splines_2D(im, im_down, Nx, Ny, down_factor);
    } else {
        perform_least_squares_splines_3D(im, im_down, Nx, Ny, Nz, down_factor);
    }
}