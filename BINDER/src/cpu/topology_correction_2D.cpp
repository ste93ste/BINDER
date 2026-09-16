//
// Created by stce on 03/01/25.
//
#define _USE_MATH_DEFINES
#include <cmath>
#include "topology_correction_2D.h"
#include "FFT.h"
#include <iostream>
#include <vector>
#include <algorithm> // For std::min and std::max
#include <memory>    // For std::unique_ptr
#include <iomanip>
#include "utils.h"

topology_correction_2D::topology_correction_2D() = default;

double* topology_correction_2D::correct_topology(const double* deformedVoxelPositions, const double e1, const double e2,
    const int max_outer_it, const int max_inner_it, int threads, const double* voxel_sizes, int Nx_v, int Ny_v) {
    this->e1 = e1;
    this->e2 = e2;
    this->Nx_v = Nx_v;
    this->Ny_v = Ny_v;
    this->N = 2;
    this->deformedVoxelPositions = std::make_unique<double[]>(Nx_v * Ny_v * N);
    this->voxel_sizes[0] = voxel_sizes[0];
    this->voxel_sizes[1] = voxel_sizes[1];
    this->fft_x = std::make_unique<FFT>(Nx_v, Ny_v, 1, threads);
    this->fft_y = std::make_unique<FFT>(Nx_v, Ny_v, 1, threads);
    this->corners = 4;
    this->in_range = false;
    double mean_x;
    double mean_y;

    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            const int idx = x * (Ny_v * N) + y * N;
            this->deformedVoxelPositions[idx + 0] = deformedVoxelPositions[idx + 0];
            this->deformedVoxelPositions[idx + 1] = deformedVoxelPositions[idx + 1];
        }
    }

    // Start the algorithm

    // Go to physical space
    to_physical();

    std::cout << "Forcing Jacobians in range: [" << e1 << ", " << e2 << "]" << std::endl;
    std::vector<double> J = compute_jacobian_determinant_2D(this->deformedVoxelPositions.get(),
        Nx_v, Ny_v, voxel_sizes[0], voxel_sizes[1]);

    // Find the minimum and maximum Jacobian values
    double J_min = *std::min_element(J.begin(), J.end());
    double J_max = *std::max_element(J.begin(), J.end());
    std::cout << "Jacobians range: [" << std::scientific << std::setprecision(10) << J_min << ", " << J_max << "]" << std::endl;
    // Count how many are below e1 and above e2
    int count_below_e1 = std::count_if(J.begin(), J.end(), [e1](const double val) {
        return val < e1;
    });

    int count_above_e2 = std::count_if(J.begin(), J.end(), [e2](const double val) {
        return val > e2;
    });
    // Output the counts
    std::cout << "Jmin#: " << count_below_e1 << " Jmax#: " << count_above_e2 << std::endl;

    if (J_min >= e1 && J_max <= e2) {
        std::cout << "Jacobians already in specified range, exiting." << std::endl;
        to_grid();
        return std::move(this->deformedVoxelPositions).get();  // Transfer ownership to caller
    }

    // Initialize J_corner
    std::unique_ptr<double[]> J_corner(new double[(Nx_v - 1) * (Ny_v - 1) * corners]);
    // Initialize gradients
    std::unique_ptr<double[]> grad(new double[Nx_v * Ny_v * N * N]);

    int it_outer = 0;
    while ((J_min < e1 || J_max > e2) && it_outer < max_outer_it) {
        std::cout << "Outer iteration: " << it_outer + 1 << std::endl;

        // Compute corner Jacobians
        std::pair<int, int> jcorner_n = compute_corner_jacobians_2D(J_corner.get());
        int it_inner = 0;

        while ((jcorner_n.first > 0 || jcorner_n.second > 0) && it_inner < max_inner_it) {

            std::cout << "Inner iteration: " << it_inner + 1 << std::endl;

            // Compute gradient fields
            compute_gradients_2D(grad.get());

            // Limit gradient
            limit_gradients_2D(grad.get(), J_corner.get());

            // Compute mean coordinates
            compute_mean_coordinates(mean_x, mean_y);

            // Integrate gradient field
            integrate_gradient_field_2D(grad.get(), mean_x, mean_y);

            // Compute corner Jacobians
            jcorner_n = compute_corner_jacobians_2D(J_corner.get());

            it_inner++;
        }

        // Re-compute Jacobians
        J = compute_jacobian_determinant_2D(this->deformedVoxelPositions.get(),
            Nx_v, Ny_v, voxel_sizes[0], voxel_sizes[1]);
        J_min = *std::min_element(J.begin(), J.end());
        J_max = *std::max_element(J.begin(), J.end());
        std::cout << "Jacobians range: [" << std::scientific << std::setprecision(10) << J_min << ", " << J_max << "]" << std::endl;
        // Count how many are below e1 and above e2
        count_below_e1 = std::count_if(J.begin(), J.end(), [e1](const double val) {
            return val < e1;
        });

        count_above_e2 = std::count_if(J.begin(), J.end(), [e2](const double val) {
            return val > e2;
        });
        // Output the counts
        std::cout << "Jmin#: " << count_below_e1 << " Jmax#: " << count_above_e2 << std::endl;


        it_outer++;
    }

    // Go back to grid space
    to_grid();

    std::cout << "Done" << std::endl;

    return this->deformedVoxelPositions.get();
}

void topology_correction_2D::to_grid() const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            const int idx = x * (Ny_v * N) + y * N;
            deformedVoxelPositions[idx + 0] /= voxel_sizes[0];
            deformedVoxelPositions[idx + 1] /= voxel_sizes[1];
        }
    }
}

void topology_correction_2D::to_physical() const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            const int idx = x * (Ny_v * N) + y * N;
            deformedVoxelPositions[idx + 0] *= voxel_sizes[0];
            deformedVoxelPositions[idx + 1] *= voxel_sizes[1];
        }
    }
}

// Function to compute corner Jacobians
std::pair<int, int> topology_correction_2D::compute_corner_jacobians_2D(double *J_corner) const {
    int jmin_n = 0;
    int jmax_n = 0;
    double jmin = 1e20;
    double jmax = -1e20;
    const double vol_scale = 1.0 / (voxel_sizes[0] * voxel_sizes[1]);

    for (int x = 0; x < Nx_v - 1; ++x) {
        for (int y = 0; y < Ny_v - 1; ++y) {
            const int idx00 = x * Ny_v * N + y * N;
            const int idx01 = x * Ny_v * N + (y + 1) * N;
            const int idx10 = (x + 1) * Ny_v * N + y * N;
            const int idx11 = (x + 1) * Ny_v * N + (y + 1) * N;

            // Get coordinates
            const double wx00 = deformedVoxelPositions[idx00];
            const double wx01 = deformedVoxelPositions[idx01];
            const double wx10 = deformedVoxelPositions[idx10];
            const double wx11 = deformedVoxelPositions[idx11];
            const double wy00 = deformedVoxelPositions[idx00 + 1];
            const double wy01 = deformedVoxelPositions[idx01 + 1];
            const double wy10 = deformedVoxelPositions[idx10 + 1];
            const double wy11 = deformedVoxelPositions[idx11 + 1];

            // Compute Jacobians with volume scaling
            const double Jff = ((wx10 - wx00) * (wy01 - wy00) - (wx01 - wx00) * (wy10 - wy00)) * vol_scale;
            const double Jbf = ((wx10 - wx00) * (wy11 - wy10) - (wx11 - wx10) * (wy10 - wy00)) * vol_scale;
            const double Jfb = ((wx11 - wx01) * (wy01 - wy00) - (wx01 - wx00) * (wy11 - wy01)) * vol_scale;
            const double Jbb = ((wx11 - wx01) * (wy11 - wy10) - (wx11 - wx10) * (wy11 - wy01)) * vol_scale;

            // Store and count violations
            const int base_idx = x * (Ny_v - 1) * corners + y * corners;
            J_corner[base_idx + 0] = Jff;
            J_corner[base_idx + 1] = Jbf;
            J_corner[base_idx + 2] = Jfb;
            J_corner[base_idx + 3] = Jbb;

            if (Jff < e1 || Jbf < e1 || Jfb < e1 || Jbb < e1) jmin_n++;
            if (Jff > e2 || Jbf > e2 || Jfb > e2 || Jbb > e2) jmax_n++;

            jmin = std::min({jmin, Jff, Jbf, Jfb, Jbb});
            jmax = std::max({jmax, Jff, Jbf, Jfb, Jbb});
        }
    }

    std::cout << "Corner Jacobians\n"
              << "#Jmin: " << jmin_n << " #Jmax: " << jmax_n << " Tot: " << jmin_n + jmax_n << "\n"
              << "Range: [" << jmin << "," << jmax << "]" << std::endl;

    return {jmin_n, jmax_n};
}

// Placeholder for compute_gradients_2D
void topology_correction_2D::compute_gradients_2D(double * grad) const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            const int base_idx = (x * Ny_v + y) * N * N;
            if (x < Nx_v - 1 && y < Ny_v - 1) {
                // X component gradients
                grad[base_idx + 0] = (deformedVoxelPositions[(x + 1) * Ny_v * N + y * N + 0] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 0]) / voxel_sizes[0];
                grad[base_idx + 1] = (deformedVoxelPositions[x * Ny_v * N + (y + 1) * N + 0] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 0]) / voxel_sizes[0];
                // Y component gradients
                grad[base_idx + 2] = (deformedVoxelPositions[(x + 1) * Ny_v * N + y * N + 1] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 1]) / voxel_sizes[1];
                grad[base_idx + 3] = (deformedVoxelPositions[x * Ny_v * N + (y + 1) * N + 1] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 1]) / voxel_sizes[1];
            } else {
                // Handle boundary conditions with correct scaling
                int x2 = x + 1;
                int y2 = y + 1;

                if (x2 > Nx_v - 1) {
                    x2 -= Nx_v;  // Wrap around to 0
                }
                if (y2 > Ny_v - 1) {
                    y2 -= Ny_v;  // Wrap around to 0
                }

                grad[base_idx + 0] = (deformedVoxelPositions[x2 * Ny_v * N + y * N + 0] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 0]) / voxel_sizes[0];
                grad[base_idx + 1] = (deformedVoxelPositions[x * Ny_v * N + y2 * N + 0] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 0]) / voxel_sizes[0];
                grad[base_idx + 2] = (deformedVoxelPositions[x2 * Ny_v * N + y * N + 1] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 1]) / voxel_sizes[1];
                grad[base_idx + 3] = (deformedVoxelPositions[x * Ny_v * N + y2 * N + 1] -
                                    deformedVoxelPositions[x * Ny_v * N + y * N + 1]) / voxel_sizes[1];
            }
        }
    }
}

void topology_correction_2D::limit_gradients_2D(double* grad, const double* J_corner) const {
    // Initialize matrices
    std::vector<double> Jnew(N * N);
    std::vector J0(N * N, 0.0);  // Initialize to zero
    std::vector<double> J(N * N);
    double detJ = 0.0;
    std::vector<int> xoff(2), yoff(2);

    // Set up identity matrix for J0
    J0[0] = 1.0;
    J0[N + 1] = 1.0;

    for (int x = 0; x < Nx_v - 1; ++x) {
        for (int y = 0; y < Ny_v - 1; ++y) {
            for (int jacnum = 0; jacnum < corners; ++jacnum) {
                // Check if correction is needed for this corner
                if (const int idx = x * (Ny_v - 1) * corners + y * corners + jacnum;
                    J_corner[idx] < e1 || J_corner[idx] > e2) {

                    // Get jacobian matrix for the current corner
                    get_jac_off_2D(jacnum, xoff.data(), yoff.data());

                    // Build the Jacobian matrix from gradients
                    for (int n1 = 0; n1 < N; ++n1) {
                        for (int n2 = 0; n2 < N; ++n2) {
                            const int base_idx = (x + xoff[n2]) * (Ny_v * N * N) + (y + yoff[n2]) * N * N;
                            J[n1 * N + n2] = grad[base_idx + N * n1 + n2];
                        }
                    }

                    // Iteratively adjust alpha until Jacobian determinant is in range
                    double alpha = 0.0;
                    detJ = J[0] * J[N + 1] - J[1] * J[N];  // det of 2x2 matrix

                    while (detJ > e2 || detJ < e1) {
                        alpha += 0.1;
                        if (alpha > 1.0) {
                            alpha = 1.0;
                        }

                        // Compute new Jacobian matrix
                        for (int n1 = 0; n1 < N; ++n1) {
                            for (int n2 = 0; n2 < N; ++n2) {
                                Jnew[n1 * N + n2] = (1 - alpha) * J[n1 * N + n2] + alpha * J0[n1 * N + n2];
                            }
                        }

                        // Compute determinant of new Jacobian
                        detJ = Jnew[0] * Jnew[N + 1] - Jnew[1] * Jnew[N];
                    }

                    // Add a small additional adjustment
                    alpha += 0.1;
                    if (alpha > 1.0) {
                        alpha = 1.0;
                    }

                    // Update the gradient field with corrected values
                    for (int n1 = 0; n1 < N; ++n1) {
                        for (int n2 = 0; n2 < N; ++n2) {
                            const int base_idx = (x + xoff[n2]) * (Ny_v * N * N) + (y + yoff[n2]) * N * N;
                            grad[base_idx + N * n1 + n2] = (1 - alpha) * J[n1 * N + n2] + alpha * J0[n1 * N + n2];
                        }
                    }
                }
            }
        }
    }
}

void topology_correction_2D::integrate_gradient_field_2D(const double* grad, const double mean_x, const double mean_y) const {
    const double norm_factor = 1.0 / (Nx_v * Ny_v);

    for (int n = 0; n < N; ++n) {
        // Fill FFT input data - separate x and y components clearly
        for (int x = 0; x < Nx_v; ++x) {
            for (int y = 0; y < Ny_v; ++y) {
                const int idx = x * Ny_v + y;
                const int grad_idx = x * (Ny_v * N * N) + y * N * N + n * N;

                // Handle x component
                fft_x->fft_in[idx][0] = grad[grad_idx];
                fft_x->fft_in[idx][1] = 0.0;

                // Handle y component separately
                fft_y->fft_in[idx][0] = grad[grad_idx + 1];
                fft_y->fft_in[idx][1] = 0.0;
            }
        }

        // Transform both components
        fft_x->fft();
        fft_y->fft();

        // Process in frequency domain
        for (int x = 0; x < Nx_v; ++x) {
            for (int y = 0; y < Ny_v; ++y) {
                const int idx = x * Ny_v + y;
                const double cosy = cos(2.0 * M_PI * y / Ny_v);
                const double siny = sin(2.0 * M_PI * y / Ny_v);
                const double cosx = cos(2.0 * M_PI * x / Nx_v);
                const double sinx = sin(2.0 * M_PI * x / Nx_v);

                if (const double norm = 4.0 - 2.0 * cosy - 2.0 * cosx; norm > 1e-10) {
                    // Combine x and y components correctly
                    const double x_real = fft_x->fft_out[idx][0];
                    const double x_imag = fft_x->fft_out[idx][1];
                    const double y_real = fft_y->fft_out[idx][0];
                    const double y_imag = fft_y->fft_out[idx][1];

                    // Calculate combined effect
                    const double real_part = (x_real * (cosx - 1.0) + x_imag * sinx +
                                            y_real * (cosy - 1.0) + y_imag * siny) / norm;
                    const double imag_part = (x_imag * (cosx - 1.0) - x_real * sinx +
                                            y_imag * (cosy - 1.0) - y_real * siny) / norm;

                    // Store for inverse transform
                    fft_x->ifft_in[idx][0] = real_part;
                    fft_x->ifft_in[idx][1] = imag_part;
                } else {
                    fft_x->ifft_in[idx][0] = 0.0;
                    fft_x->ifft_in[idx][1] = 0.0;
                }
            }
        }

        // Inverse transform
        fft_x->ifft();

        // Update positions with proper scaling
        for (int x = 0; x < Nx_v; ++x) {
            for (int y = 0; y < Ny_v; ++y) {
                const int pos_idx = x * Ny_v * N + y * N + n;
                const int idx = x * Ny_v + y;

                // Apply scaling and add mean
                double val = fft_x->ifft_out[idx][0] * norm_factor;
                val *= voxel_sizes[n];
                val += (n == 0) ? mean_x : mean_y;

                deformedVoxelPositions[pos_idx] = val;
            }
        }
    }
}



void topology_correction_2D::get_jac_off_2D(const int jac_num, int* xoff, int* yoff) {
    if (jac_num == 0) {  // Jff
        xoff[0] = 0; xoff[1] = 0;
        yoff[0] = 0; yoff[1] = 0;
    } else if (jac_num == 1) {  // Jbf
        xoff[0] = 0; xoff[1] = 1;
        yoff[0] = 0; yoff[1] = 0;
    } else if (jac_num == 2) {  // Jfb
        xoff[0] = 0; xoff[1] = 0;
        yoff[0] = 1; yoff[1] = 0;
    } else if (jac_num == 3) {  // Jbb
        xoff[0] = 0; xoff[1] = 1;
        yoff[0] = 1; yoff[1] = 0;
    }
}


void topology_correction_2D::compute_mean_coordinates(double &mean_x, double &mean_y) const {
    double sum_x = 0.0;
    double sum_y = 0.0;

    // Sum all x and y coordinates from the deformed voxel positions
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            sum_x += deformedVoxelPositions[x * Ny_v * N + y * N + 0];    // x coordinate
            sum_y += deformedVoxelPositions[x * Ny_v * N + y * N + 1];    // y coordinate
        }
    }

    // Calculate the mean x and mean y coordinates
    mean_x = sum_x / (Nx_v * Ny_v);
    mean_y = sum_y / (Nx_v * Ny_v);
}
