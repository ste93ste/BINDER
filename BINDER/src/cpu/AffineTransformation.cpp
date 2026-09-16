#include "AffineTransformation.h"
#include <vector>
#include <iostream>
#include <cmath>

AffineTransformation::AffineTransformation(const double* A, const double* t,
                                           const int Nx_v, const int Ny_v, const int Nz_v,
                                           const int threads, const double sigma_spline, const double spline_offset)
    : Nx_v(Nx_v), Ny_v(Ny_v), Nz_v(Nz_v), threads(threads),
      sigma_spline(sigma_spline), spline_offset(spline_offset) {

    // Set dimensionality based on Nz
    N = (Nz_v == 1) ? 2 : 3;
    num_voxels = Nx_v * Ny_v * Nz_v;

    // Initialize vectors with correct sizes
    X_hat.resize((N + 1) * num_voxels);
    this->A.assign(A, A + (N + 1) * (N + 1));
    this->t.assign(t, t + N);

}

AffineTransformation::~AffineTransformation() = default;

double AffineTransformation::constrain_transformation(const double* dream_locations,
                                                      const double* voxel_pos,
                                                      double* ds,
                                                      double* final_locations,
                                                      int* node_indices) {
    // Create Eigen matrices for computation
    Eigen::MatrixXd X(num_voxels, N + 1);
    X.col(0).setOnes();  // Set first column to 1 for homogeneous coordinates

    // Fill X matrix with voxel positions
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int idx = x * (Ny_v * Nz_v) + y * Nz_v + z;
                for (int n = 0; n < N; ++n) {
                    X(idx, n + 1) = voxel_pos[idx * N + n];
                }
            }
        }
    }

    // Calculate X_hat using Eigen
    Eigen::MatrixXd X_hat_eigen = (X.transpose() * X).inverse() * X.transpose();

    // Copy X_hat to our vector storage
    for (int i = 0; i < N + 1; ++i) {
        for (int j = 0; j < num_voxels; ++j) {
            X_hat[i * num_voxels + j] = X_hat_eigen(i, j);
        }
    }

    // Create dream locations matrix
    Eigen::MatrixXd Y(num_voxels, N);
    for (int i = 0; i < num_voxels; ++i) {
        for (int n = 0; n < N; ++n) {
            Y(i, n) = dream_locations[i * N + n];
        }
    }

    // Calculate transformation parameters
    Eigen::MatrixXd transform = X_hat_eigen * Y;

    // Extract translation and affine components
    for (int n = 0; n < N; ++n) {
        // First row contains translations
        t[n] = transform(0, n);

        // Remaining rows contain affine matrix
        for (int n2 = 0; n2 < N; ++n2) {
            A[n * (N + 1) + n2] = transform(n2 + 1, n);
        }
        // Set last column of affine matrix
        A[n * (N + 1) + N] = 0;
    }
    // Set bottom row of affine matrix to [0, 0, ..., 1]
    for (int n = 0; n < N; ++n) {
        A[N * (N + 1) + n] = 0;
    }
    A[(N + 1) * (N + 1) - 1] = 1;

    // Apply transformation to get final locations
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int idx = x * (Ny_v * Nz_v) + y * Nz_v + z;

                // Create homogeneous coordinate vector
                Eigen::VectorXd pos(N + 1);
                pos(0) = 1.0;
                for (int n = 0; n < N; ++n) {
                    pos(n + 1) = voxel_pos[idx * N + n];
                }

                // Apply transformation
                for (int n = 0; n < N; ++n) {
                    double tmp_sum = t[n];
                    for (int n2 = 0; n2 < N; ++n2) {
                        tmp_sum += A[n * (N + 1) + n2] * voxel_pos[idx * N + n2];
                    }

                    const int out_idx = idx * N + n;
                    node_indices[out_idx] = static_cast<int>(std::floor(tmp_sum + spline_offset));
                    ds[out_idx] = tmp_sum - std::floor(tmp_sum + spline_offset);
                    final_locations[out_idx] = tmp_sum;
                }
            }
        }
    }

    // Print results (matching Cython output)
    std::cout << "A: [";
    for (size_t i = 0; i < A.size(); ++i) {
        std::cout << A[i];
        if (i < A.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "t: [";
    for (size_t i = 0; i < t.size(); ++i) {
        std::cout << t[i];
        if (i < t.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    return 0.0;
}

void AffineTransformation::sample_transformation(const double* dream_locations,
                                               const double* voxel_pos,
                                               double* ds,
                                               double* final_locations,
                                               int* node_indices,
                                               int sample_gamma) {
    // TODO: Implement sampling logic
    constrain_transformation(dream_locations, voxel_pos, ds, final_locations, node_indices);
}

void AffineTransformation::set_generator(std::mt19937& gen) {
    this->gen = gen;
}