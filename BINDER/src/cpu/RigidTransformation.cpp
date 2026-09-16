#include "RigidTransformation.h"
#include <vector>
#include <iostream>
#include <cmath>

RigidTransformation::RigidTransformation(int Nx_v, int Ny_v, int Nz_v,
                                       int threads, double sigma_spline, double spline_offset,
                                       const double* R_init,
                                       const double* t_init,
                                       const double* voxel_pos)
    : Nx_v(Nx_v), Ny_v(Ny_v), Nz_v(Nz_v), threads(threads),
      sigma_spline(sigma_spline), spline_offset(spline_offset) {

    // Set dimensionality based on Nz
    N = (Nz_v == 1) ? 2 : 3;
    num_voxels = Nx_v * Ny_v * Nz_v;

    // Initialize rotation matrix and translation vector
    R.assign(R_init, R_init + N * N);
    t.assign(t_init, t_init + N);

    // Calculate means of voxel positions
    x_means.resize(N, 0.0);
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    const int idx = (x * Ny_v * Nz_v + y * Nz_v + z) * N + n;
                    x_means[n] += voxel_pos[idx];
                }
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        x_means[n] /= num_voxels;
    }

    // Calculate x_tilde (centered positions)
    x_tilde.resize(num_voxels * N);
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    const int voxel_idx = (x * Ny_v * Nz_v + y * Nz_v + z) * N + n;
                    const int tilde_idx = x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n;
                    x_tilde[tilde_idx] = voxel_pos[voxel_idx] - x_means[n];
                }
            }
        }
    }
}

RigidTransformation::~RigidTransformation() = default;


double RigidTransformation::constrain_transformation(const double* dream_locations,
                                                   const double* voxel_pos,
                                                   double* ds,
                                                   double* final_locations,
                                                   int* node_indices) {
    // Calculate means of dream locations
    std::vector<double> y_means(N, 0.0);
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    const int idx = (x * Ny_v * Nz_v + y * Nz_v + z) * N + n;
                    y_means[n] += dream_locations[idx];
                }
            }
        }
    }

    for (int n = 0; n < N; ++n) {
        y_means[n] /= num_voxels;
    }

    // Calculate H matrix using Eigen
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    const double y_tilde = dream_locations[(x * Ny_v * Nz_v + y * Nz_v + z) * N + n] - y_means[n];
                    for (int n2 = 0; n2 < N; ++n2) {
                        H(n, n2) += x_tilde[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n2] * y_tilde;
                    }
                }
            }
        }
    }

    // Perform SVD using Eigen
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    const Eigen::MatrixXd& V = svd.matrixV();

    // Calculate rotation matrix ensuring proper orientation
    Eigen::VectorXd tmp_diag = Eigen::VectorXd::Ones(N);
    tmp_diag(N-1) = (V * U.transpose()).determinant();

    Eigen::MatrixXd rotation = V * tmp_diag.asDiagonal() * U.transpose();

    // Update rotation matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            R[i * N + j] = rotation(i, j);
        }
    }

    // Calculate translation
    for (int n = 0; n < N; ++n) {
        double tmp_sum = y_means[n];
        for (int n2 = 0; n2 < N; ++n2) {
            tmp_sum -= R[n * N + n2] * x_means[n2];
        }
        t[n] = tmp_sum;
    }

    // Apply transformation to get final locations
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    double tmp_sum = t[n];
                    for (int n2 = 0; n2 < N; ++n2) {
                        tmp_sum += R[n * N + n2] * voxel_pos[(x * Ny_v * Nz_v + y * Nz_v + z) * N + n2];
                    }

                    const int idx = (x * Ny_v * Nz_v + y * Nz_v + z) * N + n;
                    node_indices[idx] = static_cast<int>(std::floor(tmp_sum + spline_offset));
                    ds[idx] = tmp_sum - std::floor(tmp_sum + spline_offset);
                    final_locations[idx] = tmp_sum;
                }
            }
        }
    }

    // Print results
    std::cout << "R: [";
    for (const auto& val : R) {
        std::cout << val << " ";
    }
    std::cout << "]\n";

    std::cout << "t: [";
    for (const auto& val : t) {
        std::cout << val << " ";
    }
    std::cout << "]\n";

    return 0.0;
}

void RigidTransformation::sample_transformation(const double* dream_locations,
                                              const double* voxel_pos,
                                              double* ds,
                                              double* final_locations,
                                              int* node_indices,
                                              int sample_gamma) {
    // TODO: Implement sampling logic
    constrain_transformation(dream_locations, voxel_pos, ds, final_locations, node_indices);
}

void RigidTransformation::set_generator(std::mt19937& gen) {
    this->gen = gen;
}