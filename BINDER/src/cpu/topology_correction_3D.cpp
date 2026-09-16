//
// Created by stce on 03/01/25.
//
#define _USE_MATH_DEFINES
#include <cmath>
#include "topology_correction_3D.h"
#include "FFT.h"
#include <iostream>
#include <vector>
#include <algorithm> // For std::min and std::max
#include <memory>    // For std::unique_ptr
#include <iomanip>
#include "utils.h"

topology_correction_3D::topology_correction_3D() = default;

double* topology_correction_3D::correct_topology(const double* deformedVoxelPositions, const double e1, const double e2,
    const int max_outer_it, const int max_inner_it, int threads, const double* voxel_sizes, int Nx_v, int Ny_v, int Nz_v)
{
    this->e1 = e1;
    this->e2 = e2;
    this->Nx_v = Nx_v;
    this->Ny_v = Ny_v;
    this->Nz_v = Nz_v;
    this->N = 3;
    this->fft_x = std::make_unique<FFT>(Nx_v, Ny_v, Nz_v, threads);
    this->fft_y = std::make_unique<FFT>(Nx_v, Ny_v, Nz_v, threads);
    this->fft_z = std::make_unique<FFT>(Nx_v, Ny_v, Nz_v, threads);
    this->corners = 8;
    this->in_range = false;

    this->deformedVoxelPositions = std::make_unique<double[]>(Nx_v * Ny_v * Nz_v * N);
    this->voxel_sizes[0] = voxel_sizes[0];
    this->voxel_sizes[1] = voxel_sizes[1];
    this->voxel_sizes[2] = voxel_sizes[2];

    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int idx = x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N;
                this->deformedVoxelPositions[idx + 0] = deformedVoxelPositions[idx + 0];
                this->deformedVoxelPositions[idx + 1] = deformedVoxelPositions[idx + 1];
                this->deformedVoxelPositions[idx + 2] = deformedVoxelPositions[idx + 2];
            }
        }
    }

    // Go to physical space
    to_physical();

    // Start the algorithm
    std::cout << "Forcing Jacobians in range: [" << e1 << ", " << e2 << "]" << std::endl;
    std::vector<double> J = compute_jacobian_determinant_3D(this->deformedVoxelPositions.get(),
                                                             Nx_v, Ny_v, Nz_v,
                                                             voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]);

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
    const std::unique_ptr<double[]> J_corner(new double[(Nx_v - 1) * (Ny_v - 1) * (Nz_v - 1) * corners]);
    // Initialize gradients
    const std::unique_ptr<double[]> grad(new double[Nx_v * Ny_v * Nz_v * N * N]);

    int it_outer = 0;
    while ((J_min < e1 || J_max > e2) && it_outer < max_outer_it) {
        std::cout << "Outer iteration: " << it_outer + 1 << std::endl;

        // Compute corner Jacobians
        std::pair<int, int> jcorner_n = compute_corner_jacobians_3D(J_corner.get());

        int it_inner = 0;
        while ((jcorner_n.first > 0 || jcorner_n.second > 0) && it_inner < max_inner_it) {
            double mean_z = 0;
            double mean_y = 0;
            double mean_x = 0;
            std::cout << "Inner iteration: " << it_inner + 1 << std::endl;

            // Compute gradient fields
            compute_gradients_3D(grad.get());

            // Limit gradient
            limit_gradients_3D(grad.get(), J_corner.get());

            // Compute mean coordinates
            compute_mean_coordinates(mean_x, mean_y, mean_z);

            // Integrate gradient field
            integrate_gradient_field_3D(grad.get(), mean_x, mean_y, mean_z);

            // Compute corner Jacobians
            jcorner_n = compute_corner_jacobians_3D(J_corner.get());

            it_inner++;
        }

        // Re-compute Jacobians
        J = compute_jacobian_determinant_3D(this->deformedVoxelPositions.get(),
                                                                     Nx_v, Ny_v, Nz_v,
                                                                     voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]);
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

    return std::move(this->deformedVoxelPositions).get();  // Transfer ownership to caller
}

void topology_correction_3D::to_grid() const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int idx = (x * Ny_v * Nz_v + y * Nz_v + z) * N;
                deformedVoxelPositions[idx + 0] /= voxel_sizes[0];
                deformedVoxelPositions[idx + 1] /= voxel_sizes[1];
                deformedVoxelPositions[idx + 2] /= voxel_sizes[2];
            }
        }
    }
}

void topology_correction_3D::to_physical() const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int idx = (x * Ny_v * Nz_v + y * Nz_v + z) * N;
                deformedVoxelPositions[idx + 0] *= voxel_sizes[0];
                deformedVoxelPositions[idx + 1] *= voxel_sizes[1];
                deformedVoxelPositions[idx + 2] *= voxel_sizes[2];
            }
        }
    }
}


void topology_correction_3D::compute_gradients_3D(double* grad) const {

    // Constants for voxel sizes
    const double dx = voxel_sizes[0];
    const double dy = voxel_sizes[1];
    const double dz = voxel_sizes[2];

    // Iterate over all voxels in the 3D grid
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {

                // Index for accessing gradients
                const int base_idx = (x * Ny_v * Nz_v * N * N) + (y * Nz_v * N * N) + (z * N * N);

                if (x < Nx_v - 1 && y < Ny_v - 1 && z < Nz_v - 1) {
                    // Interior gradients
                    grad[base_idx + 0] = (deformedVoxelPositions[(x + 1) * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
                    grad[base_idx + 1] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + (y + 1) * Nz_v * N + z * N + 0] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
                    grad[base_idx + 2] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + (z + 1) * N + 0] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
                    grad[base_idx + 3] = (deformedVoxelPositions[(x + 1) * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
                    grad[base_idx + 4] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + (y + 1) * Nz_v * N + z * N + 1] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
                    grad[base_idx + 5] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + (z + 1) * N + 1] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
                    grad[base_idx + 6] = (deformedVoxelPositions[(x + 1) * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
                    grad[base_idx + 7] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + (y + 1) * Nz_v * N + z * N + 2] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
                    grad[base_idx + 8] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + (z + 1) * N + 2] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
                } else {
                    // Boundary handling (wrap-around for boundaries)
                    int x2 = x + 1;
                    int y2 = y + 1;
                    int z2 = z + 1;

                    if (x2 > Nx_v - 1) {
                        x2 -= Nx_v;  // Wrap around to 0
                    }
                    if (y2 > Ny_v - 1) {
                        y2 -= Ny_v;  // Wrap around to 0
                    }
                    if (z2 > Nz_v - 1) {
                        z2 -= Nz_v;  // Wrap around to 0
                    }

                    grad[base_idx + 0] = (deformedVoxelPositions[x2 * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
                    grad[base_idx + 1] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y2 * Nz_v * N + z * N + 0] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
                    grad[base_idx + 2] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z2 * N + 0] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
                    grad[base_idx + 3] = (deformedVoxelPositions[x2 * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
                    grad[base_idx + 4] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y2 * Nz_v * N + z * N + 1] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
                    grad[base_idx + 5] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z2 * N + 1] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
                    grad[base_idx + 6] = (deformedVoxelPositions[x2 * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
                    grad[base_idx + 7] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y2 * Nz_v * N + z * N + 2] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
                    grad[base_idx + 8] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z2 * N + 2] -
                                          deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
                }
            }
        }
    }
}

void topology_correction_3D::limit_gradients_3D(double * grad, const double * J_corner) const {

    // Allocate memory for matrices and offsets
    auto Jnew = std::vector(N * N, 0.0);
    auto J0 = std::vector(N * N, 0.0);
    auto J = std::vector(N * N, 0.0);

    std::vector<int> xoff(3), yoff(3), zoff(3);

    // Identity matrix J0
    J0[0] = 1.0;
    J0[1 * N + 1] = 1.0;
    J0[2 * N + 2] = 1.0;

    // Loop through all voxels
    for (int x = 0; x < Nx_v - 1; ++x) {
        for (int y = 0; y < Ny_v - 1; ++y) {
            for (int z = 0; z < Nz_v - 1; ++z) {
                for (int jacnum = 0; jacnum < corners; ++jacnum) {
                    const int idx = x * ((Ny_v - 1) * (Nz_v - 1) * corners) + y * ((Nz_v - 1) * corners) + z * corners + jacnum;
                    double jacobian_val = J_corner[idx];

                    if (jacobian_val < e1 || jacobian_val > e2) {
                        // Compute the Jacobian for the current voxel and corner
                        get_jac_off_3D(jacnum, xoff.data(), yoff.data(), zoff.data());
                        for (int n1 = 0; n1 < N; ++n1) {
                            for (int n2 = 0; n2 < N; ++n2) {
                                // Populate the J matrix with gradient values
                                const int grad_idx = (x + xoff[n2]) * (Ny_v * Nz_v * N * N) +
                                                     (y + yoff[n2]) * (Nz_v * N * N) +
                                                     (z + zoff[n2]) * N * N +
                                                      N * n1 + n2;
                                J[n1 * N + n2] = grad[grad_idx];
                            }
                        }

                        double alpha = 0.0;

                        // Compute the determinant of the Jacobian matrix
                        double detJ = J[0 * N + 0] * (J[1 * N + 1] * J[2 * N + 2] - J[1 * N + 2] * J[2 * N + 1]) -
                                      J[0 * N + 1] * (J[1 * N + 0] * J[2 * N + 2] - J[1 * N + 2] * J[2 * N + 0]) +
                                      J[0 * N + 2] * (J[1 * N + 0] * J[2 * N + 1] - J[1 * N + 1] * J[2 * N + 0]);

                        // Loop to adjust the gradient if the determinant is out of bounds
                        while (detJ > e2 || detJ < e1) {
                            alpha += 0.1;
                            if (alpha > 1.0) {
                                alpha = 1.0;
                            }

                            // Interpolate between J and J0 to adjust gradients
                            for (int n1 = 0; n1 < N; ++n1) {
                                for (int n2 = 0; n2 < N; ++n2) {
                                    Jnew[n1 * N + n2] = (1 - alpha) * J[n1 * N + n2] + alpha * J0[n1 * N + n2];
                                }
                            }

                            // Recompute the determinant of the updated Jacobian
                            detJ = Jnew[0 * N + 0] * (Jnew[1 * N + 1] * Jnew[2 * N + 2] - Jnew[1 * N + 2] * Jnew[2 * N + 1]) -
                                   Jnew[0 * N + 1] * (Jnew[1 * N + 0] * Jnew[2 * N + 2] - Jnew[1 * N + 2] * Jnew[2 * N + 0]) +
                                   Jnew[0 * N + 2] * (Jnew[1 * N + 0] * Jnew[2 * N + 1] - Jnew[1 * N + 1] * Jnew[2 * N + 0]);
                        }

                        // Adjust gradients one more time
                        alpha += 0.1;
                        if (alpha > 1.0) {
                            alpha = 1.0;
                        }

                        // Rescale the gradients as required
                        for (int n1 = 0; n1 < N; ++n1) {
                            for (int n2 = 0; n2 < N; ++n2) {
                                const int grad_idx = (x + xoff[n2]) * (Ny_v * Nz_v * N * N) +
                                                     (y + yoff[n2]) * (Nz_v * N * N) +
                                                     (z + zoff[n2]) * N * N +
                                                      N * n1 + n2;
                                grad[grad_idx] = (1 - alpha) * J[n1 * N + n2] + alpha * J0[n1 * N + n2];
                            }
                        }
                    }
                }
            }
        }
    }

}

void topology_correction_3D::compute_mean_coordinates(double &mean_x, double &mean_y, double &mean_z) const {

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;

    // Sum all x, y, and z coordinates from the deformed voxel positions
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                // Calculate the index in the deformed_voxel_positions array
                const int index = (x * Ny_v * Nz_v + y * Nz_v + z) * N;

                sum_x += deformedVoxelPositions[index + 0];  // x coordinate
                sum_y += deformedVoxelPositions[index + 1];  // y coordinate
                sum_z += deformedVoxelPositions[index + 2];  // z coordinate
            }
        }
    }

    // Calculate the mean x, y, and z coordinates
    mean_x = sum_x / (Nx_v * Ny_v * Nz_v);
    mean_y = sum_y / (Nx_v * Ny_v * Nz_v);
    mean_z = sum_z / (Nx_v * Ny_v * Nz_v);
}


void topology_correction_3D::integrate_gradient_field_3D(const double *grad,
                                                         double mean_x, double mean_y, double mean_z) const {
    const int Nx = Nx_v;
    const int Ny = Ny_v;
    const int Nz = Nz_v;

    for (int n = 0; n < N; ++n) {
        // Fill data_x, data_y, and data_z with gradient data
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                for (int z = 0; z < Nz; ++z) {
                    fft_x->fft_in[x * Ny * Nz + y * Nz + z][0] = grad[x * (Ny * Nz * N * N) + y * (Nz * N * N) + z * (N * N) + n * N + 0];
                    fft_x->fft_in[x * Ny * Nz + y * Nz + z][1] = 0.0;
                    fft_y->fft_in[x * Ny * Nz + y * Nz + z][0] = grad[x * (Ny * Nz * N * N) + y * (Nz * N * N) + z * (N * N) + n * N + 1];
                    fft_y->fft_in[x * Ny * Nz + y * Nz + z][1] = 0.0;
                    fft_z->fft_in[x * Ny * Nz + y * Nz + z][0] = grad[x * (Ny * Nz * N * N) + y * (Nz * N * N) + z * (N * N) + n * N + 2];
                    fft_z->fft_in[x * Ny * Nz + y * Nz + z][1] = 0.0;
                }
            }
        }

        fft_x->fft();
        fft_y->fft();
        fft_z->fft();

        // Apply Fourier-based integration
        for (int z = 0; z < Nz; ++z) {
            const double cosz = cos(2.0 * M_PI * z / (Nz * 1.0));
            const double sinz = sin(2.0 * M_PI * z / (Nz * 1.0));
            for (int y = 0; y < Ny; ++y) {
                const double cosy = cos(2.0 * M_PI * y / (Ny * 1.0));
                const double siny = sin(2.0 * M_PI * y / (Ny * 1.0));
                for (int x = 0; x < Nx; ++x) {
                    const double cosx = cos(2.0 * M_PI * x / (Nx * 1.0));
                    const double sinx = sin(2.0 * M_PI * x / (Nx * 1.0));

                    if (const double norm = 6.0 - 2.0 * cosz - 2.0 * cosx - 2.0 * cosy; std::fabs(norm) > 1e-10) {
                        double dotprodreal = fft_x->fft_out[x * Ny * Nz + y * Nz + z][0] * (cosx - 1) +
                                             fft_x->fft_out[x * Ny * Nz + y * Nz + z][1] * sinx +
                                             fft_y->fft_out[x * Ny * Nz + y * Nz + z][0] * (cosy - 1) +
                                             fft_y->fft_out[x * Ny * Nz + y * Nz + z][1] * siny +
                                             fft_z->fft_out[x * Ny * Nz + y * Nz + z][0] * (cosz - 1) +
                                             fft_z->fft_out[x * Ny * Nz + y * Nz + z][1] * sinz;
                        double dotprodimag = fft_x->fft_out[x * Ny * Nz + y * Nz + z][1] * (cosx - 1) -
                                             fft_x->fft_out[x * Ny * Nz + y * Nz + z][0] * sinx +
                                             fft_y->fft_out[x * Ny * Nz + y * Nz + z][1] * (cosy - 1) -
                                             fft_y->fft_out[x * Ny * Nz + y * Nz + z][0] * siny +
                                             fft_z->fft_out[x * Ny * Nz + y * Nz + z][1] * (cosz - 1) -
                                             fft_z->fft_out[x * Ny * Nz + y * Nz + z][0] * sinz;

                        fft_x->ifft_in[x * Ny * Nz + y * Nz + z][0] = dotprodreal / norm;
                        fft_x->ifft_in[x * Ny * Nz + y * Nz + z][1] = dotprodimag / norm;
                    } else {
                        fft_x->ifft_in[x * Ny * Nz + y * Nz + z][0] = 0;
                        fft_x->ifft_in[x * Ny * Nz + y * Nz + z][1] = 0;
                    }
                }
            }
        }

        fft_x->ifft();

        // Update deformed voxel positions with integrated data
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                for (int z = 0; z < Nz; ++z) {
                    deformedVoxelPositions[x * Ny * Nz * N + y * Nz * N + z * N + n] =
                        fft_x->ifft_out[x * Ny * Nz + y * Nz + z][0] / (Nx * Ny * Nz * 1.0);
                }
            }
        }

    }

    // Rescale new warp by the voxel dimensions and adjust mean
    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int z = 0; z < Nz; ++z) {
                deformedVoxelPositions[x * Ny * Nz * N + y * Nz * N + z * N + 0] *= voxel_sizes[0];
                deformedVoxelPositions[x * Ny * Nz * N + y * Nz * N + z * N + 0] += mean_x;
                deformedVoxelPositions[x * Ny * Nz * N + y * Nz * N + z * N + 1] *= voxel_sizes[1];
                deformedVoxelPositions[x * Ny * Nz * N + y * Nz * N + z * N + 1] += mean_y;
                deformedVoxelPositions[x * Ny * Nz * N + y * Nz * N + z * N + 2] *= voxel_sizes[2];
                deformedVoxelPositions[x * Ny * Nz * N + y * Nz * N + z * N + 2] += mean_z;
            }
        }
    }

}


void topology_correction_3D::get_jac_off_3D(const int jac_num, int* xoff, int* yoff, int* zoff) {
    if (jac_num == 0) {  // Jfff
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 1) {  // Jbff
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 2) {  // Jfbf
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 3) {  // Jffb
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    } else if (jac_num == 4) {  // Jfbb
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    } else if (jac_num == 5) {  // Jbfb
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    } else if (jac_num == 6) {  // Jbbf
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 7) {  // Jbbb
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    }
}

std::pair<int, int> topology_correction_3D::compute_corner_jacobians_3D(double *J_corner) const {
    int jmin_n = 0;
    int jmax_n = 0;
    double jmin = 1e20;
    double jmax = -1e20;

    const double vol_scale = 1.0 / (voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]);

    for (int x = 0; x < Nx_v - 1; ++x) {
        for (int y = 0; y < Ny_v - 1; ++y) {
            for (int z = 0; z < Nz_v - 1; ++z) {
                // Calculate indices for 1D array
                const int idx000 = (x * Ny_v * Nz_v + y * Nz_v + z) * N;
                const int idx001 = (x * Ny_v * Nz_v + y * Nz_v + (z + 1)) * N;
                const int idx010 = (x * Ny_v * Nz_v + (y + 1) * Nz_v + z) * N;
                const int idx011 = (x * Ny_v * Nz_v + (y + 1) * Nz_v + (z + 1)) * N;
                const int idx100 = ((x + 1) * Ny_v * Nz_v + y * Nz_v + z) * N;
                const int idx101 = ((x + 1) * Ny_v * Nz_v + y * Nz_v + (z + 1)) * N;
                const int idx110 = ((x + 1) * Ny_v * Nz_v + (y + 1) * Nz_v + z) * N;
                const int idx111 = ((x + 1) * Ny_v * Nz_v + (y + 1) * Nz_v + (z + 1)) * N;

                // X, Y, Z directions
                const double wx000 = deformedVoxelPositions[idx000];
                const double wx001 = deformedVoxelPositions[idx001];
                const double wx010 = deformedVoxelPositions[idx010];
                const double wx011 = deformedVoxelPositions[idx011];
                const double wx100 = deformedVoxelPositions[idx100];
                const double wx101 = deformedVoxelPositions[idx101];
                const double wx110 = deformedVoxelPositions[idx110];
                const double wx111 = deformedVoxelPositions[idx111];

                const double wy000 = deformedVoxelPositions[idx000 + 1];
                const double wy001 = deformedVoxelPositions[idx001 + 1];
                const double wy010 = deformedVoxelPositions[idx010 + 1];
                const double wy011 = deformedVoxelPositions[idx011 + 1];
                const double wy100 = deformedVoxelPositions[idx100 + 1];
                const double wy101 = deformedVoxelPositions[idx101 + 1];
                const double wy110 = deformedVoxelPositions[idx110 + 1];
                const double wy111 = deformedVoxelPositions[idx111 + 1];

                const double wz000 = deformedVoxelPositions[idx000 + 2];
                const double wz001 = deformedVoxelPositions[idx001 + 2];
                const double wz010 = deformedVoxelPositions[idx010 + 2];
                const double wz011 = deformedVoxelPositions[idx011 + 2];
                const double wz100 = deformedVoxelPositions[idx100 + 2];
                const double wz101 = deformedVoxelPositions[idx101 + 2];
                const double wz110 = deformedVoxelPositions[idx110 + 2];
                const double wz111 = deformedVoxelPositions[idx111 + 2];

                // Compute Jacobian values for the 8 corner Jacobians
                double Jfff = (wx100 - wx000) * ((wy010 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz010 - wz000)) -
                              (wx010 - wx000) * ((wy100 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz100 - wz000)) +
                              (wx001 - wx000) * ((wy100 - wy000) * (wz010 - wz000) - (wy010 - wy000) * (wz100 - wz000));
                Jfff *= vol_scale;
                double Jbff = (wx100 - wx000) * ((wy110 - wy100) * (wz101 - wz100) - (wy101 - wy100) * (wz110 - wz100)) -
                              (wx110 - wx100) * ((wy100 - wy000) * (wz101 - wz100) - (wy101 - wy100) * (wz100 - wz000)) +
                              (wx101 - wx100) * ((wy100 - wy000) * (wz110 - wz100) - (wy110 - wy100) * (wz100 - wz000));
                Jbff *= vol_scale;
                double Jfbf = (wx110 - wx010) * ((wy010 - wy000) * (wz011 - wz010) - (wy011 - wy010) * (wz010 - wz000)) - (wx010 - wx000) * \
                              ((wy110 - wy010) * (wz011 - wz010) - (wy011 - wy010) * (wz110 - wz010)) + (wx011 - wx010) * \
                              ((wy110 - wy010) * (wz010 - wz000) - (wy010 - wy000) * (wz110 - wz010));
                Jfbf *= vol_scale;
                double Jffb = (wx101 - wx001) * ((wy011 - wy001) * (wz001 - wz000) - (wy001 - wy000) * (wz011 - wz001)) - (wx011 - wx001) * \
                              ((wy101 - wy001) * (wz001 - wz000) - (wy001 - wy000) * (wz101 - wz001)) + (wx001 - wx000) * \
                              ((wy101 - wy001) * (wz011 - wz001) - (wy011 - wy001) * (wz101 - wz001));
                Jffb *= vol_scale;
                double Jfbb = (wx111 - wx011) * ((wy011 - wy001) * (wz011 - wz010) - (wy011 - wy010) * (wz011 - wz001)) - (wx011 - wx001) * \
                              ((wy111 - wy011) * (wz011 - wz010) - (wy011 - wy010) * (wz111 - wz011)) + (wx011 - wx010) * \
                              ((wy111 - wy011) * (wz011 - wz001) - (wy011 - wy001) * (wz111 - wz011));
                Jfbb *= vol_scale;
                double Jbfb = (wx101 - wx001) * ((wy111 - wy101) * (wz101 - wz100) - (wy101 - wy100) * (wz111 - wz101)) - (wx111 - wx101) * \
                              ((wy101 - wy001) * (wz101 - wz100) - (wy101 - wy100) * (wz101 - wz001)) + (wx101 - wx100) * \
                              ((wy101 - wy001) * (wz111 - wz101) - (wy111 - wy101) * (wz101 - wz001));
                Jbfb *= vol_scale;
                double Jbbf = (wx110 - wx010) * ((wy110 - wy100) * (wz111 - wz110) - (wy111 - wy110) * (wz110 - wz100)) - (wx110 - wx100) * \
                           ((wy110 - wy010) * (wz111 - wz110) - (wy111 - wy110) * (wz110 - wz010)) + (wx111 - wx110) * \
                           ((wy110 - wy010) * (wz110 - wz100) - (wy110 - wy100) * (wz110 - wz010));
                Jbbf *= vol_scale;
                double Jbbb = (wx111 - wx011) * ((wy111 - wy101) * (wz111 - wz110) - (wy111 - wy110) * (wz111 - wz101)) - (wx111 - wx101) * \
                              ((wy111 - wy011) * (wz111 - wz110) - (wy111 - wy110) * (wz111 - wz011)) + (wx111 - wx110) * \
                              ((wy111 - wy011) * (wz111 - wz101) - (wy111 - wy101) * (wz111 - wz011));
                Jbbb *= vol_scale;

                // Store corner Jacobians in J_corner
                const int idx = x * (Ny_v - 1) * (Nz_v - 1) * corners + y * (Nz_v - 1) * corners + z * corners;
                J_corner[idx + 0] = Jfff;
                J_corner[idx + 1] = Jbff;
                J_corner[idx + 2] = Jfbf;
                J_corner[idx + 3] = Jffb;
                J_corner[idx + 4] = Jfbb;
                J_corner[idx + 5] = Jbfb;
                J_corner[idx + 6] = Jbbf;
                J_corner[idx + 7] = Jbbb;

                // Count how many Jacobians are below e1 or above e2
                for (int i = 0; i < corners; ++i) {
                    double tmp = J_corner[(x * (Ny_v - 1) * (Nz_v - 1) * corners) + (y * (Nz_v - 1) * corners) + (z * corners) + i];
                    if (tmp < e1) {
                        jmin_n++;
                    }
                    if (tmp > e2) {
                        jmax_n++;
                    }

                    // Update min/max Jacobians
                    jmin = std::min(jmin, tmp);
                    jmax = std::max(jmax, tmp);
                }
            }
        }
    }

    // Output results
    std::cout << "Corner Jacobians" << std::endl;
    std::cout << "#J_min: " << jmin_n << " #J_max: " << jmax_n << std::endl;
    std::cout << "min_J: " << jmin << " max_J: " << jmax << std::endl;

    return {jmin_n, jmax_n};

}
