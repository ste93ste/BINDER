#include "MILikelihoodFilter.h"
#include <cmath>
#include <random>
#include <cstring>
#include <memory>
#include <vector>

MILikelihoodFilter::MILikelihoodFilter(const double alpha, const double* theta, const int threads,
                                        const int* binned_nodes, const int* binned_voxels,
                                        const int K, const int L, const int M,
                                        const int Nx_n, const int Ny_n, const int Nz_n,
                                        const int Nx_v, const int Ny_v, const int Nz_v)
    : alpha(alpha), K(K), L(L), KL(K*L), M(M), threads(threads),
      Nx_n(Nx_n), Ny_n(Ny_n), Nz_n(Nz_n),
      Nx_v(Nx_v), Ny_v(Ny_v), Nz_v(Nz_v),
      binned_nodes(binned_nodes), binned_voxels(binned_voxels),
      Ny_n_Nz_n(Ny_n * Nz_n) {

    // Allocate memory for theta and log_theta
    this->theta = std::make_unique<double[]>(M * KL);
    this->log_theta = std::make_unique<double[]>(M * KL);

    // Initialize theta and compute log_theta
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int l = 0; l < L; ++l) {
                const int idx = m * KL + k * L + l;
                this->theta[idx] = theta[idx];
                this->log_theta[idx] = std::log(theta[idx]);
            }
        }
    }

    // Allocate memory for temporary variables
    theta_tmp = std::make_unique<double[]>(M * KL * threads);
    initialize_parameters();
}

double MILikelihoodFilter::compute_cost() {
    double cost = 0.0;
    std::vector alpha_vec(L, alpha);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            cost += -dirichlet_logpdf(&theta[m * KL + k * L], alpha, L);
        }
    }
    return cost;
}

double MILikelihoodFilter::compute_likelihood(const int voxel_index, const int x_n, const int y_n, const int z_n) {
    double tmp = 1.0;

    if (x_n < 0 || y_n < 0 || z_n < 0 || x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n) {
        for (int m = 0; m < M; ++m) {
            tmp *= theta[m * KL + binned_voxels[voxel_index + m]];
        }
    } else {
        const int node_index = x_n * Ny_n_Nz_n + y_n * Nz_n + z_n;
        for (int m = 0; m < M; ++m) {
            tmp *= theta[m * KL + binned_nodes[node_index + m] * L + binned_voxels[voxel_index + m]];
        }
    }
    return tmp;
}

double MILikelihoodFilter::compute_log_likelihood(const int voxel_index,
                                                  const int x_n, const int y_n, const int z_n) {
    double tmp = 0.0;

    if (x_n < 0 || y_n < 0 || z_n < 0 || x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n) {
        for (int m = 0; m < M; ++m) {
            tmp += log_theta[m * KL + binned_voxels[voxel_index + m]];
        }
    } else {
        const int node_index = x_n * Ny_n_Nz_n + y_n * Nz_n + z_n;
        for (int m = 0; m < M; ++m) {
            tmp += log_theta[m * KL + binned_nodes[node_index + m] * L + binned_voxels[voxel_index + m]];
        }
    }
    return tmp;
}

void MILikelihoodFilter::update_parameters() {
    joint_histogram_thread_safe();

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            double theta_normalizer = 0.0;
            for (int l = 0; l < L; ++l) {
                theta_normalizer += (theta[m * KL + k * L + l] + alpha - 1.0);
            }
            for (int l = 0; l < L; ++l) {
                const int idx = m * KL + k * L + l;
                theta[idx] = (theta[idx] + alpha - 1.0) / theta_normalizer;
                log_theta[idx] = std::log(theta[idx]);
            }
        }
    }
}

void MILikelihoodFilter::accumulate_info(const int thread_id, const int voxel_index,
                                         const int x_n, const int y_n, const int z_n, const double posterior) {
    if (x_n < 0 || y_n < 0 || z_n < 0 || x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n) {
        for (int m = 0; m < M; ++m) {
            theta_tmp[thread_id * (M * KL) + m * KL + binned_voxels[voxel_index + m]] += posterior;
        }
    } else {
        const int node_index = x_n * Ny_n_Nz_n + y_n * Nz_n + z_n;
        for (int m = 0; m < M; ++m) {
            theta_tmp[thread_id * (M * KL) + m * KL + binned_nodes[node_index + m] * L +
                     binned_voxels[voxel_index + m]] += posterior;
        }
    }
}

void MILikelihoodFilter::initialize_parameters() {
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                for (int l = 0; l < L; ++l) {
                    theta_tmp[thread_id * (M * KL) + m * KL + k * L + l] = 0.0;
                }
            }
        }
    }
}

void MILikelihoodFilter::joint_histogram_thread_safe() const {
    // Reset theta
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int l = 0; l < L; ++l) {
                theta[m * KL + k * L + l] = 0.0;
            }
        }
    }

    // Accumulate from all threads
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                for (int l = 0; l < L; ++l) {
                    theta[m * KL + k * L + l] +=
                        theta_tmp[thread_id * (M * KL) + m * KL + k * L + l];
                }
            }
        }
    }
}

void MILikelihoodFilter::sample_parameters() {
    auto tmp = std::make_unique<double[]>(M * KL);
    joint_histogram_thread_safe();

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            double sum = 0.0;

            // Sample from gamma distributions
            for (int l = 0; l < L; ++l) {
                const double shape = theta[m * KL + k * L + l] + alpha;
                std::gamma_distribution dist(shape, 1.0);
                tmp[m * KL + k * L + l] = dist(gen);
                sum += tmp[m * KL + k * L + l];
            }

            // Normalize to get Dirichlet distribution
            for (int l = 0; l < L; ++l) {
                const int idx = m * KL + k * L + l;
                theta[idx] = tmp[idx] / sum;
                log_theta[idx] = std::log(theta[idx]);
            }
        }
    }
}

void MILikelihoodFilter::set_generator(std::mt19937 &gen) {
    this->gen = gen;
}

std::vector<double> MILikelihoodFilter::get_parameters() {
    return std::vector(theta.get(), theta.get() + M * KL);
}

// Destructor to clean up dynamically allocated memory
MILikelihoodFilter::~MILikelihoodFilter() = default;

// Function to compute log of gamma
double MILikelihoodFilter::log_gamma(const double x) {
    return std::lgamma(x);
}

// Dirichlet log PDF function using arrays
double MILikelihoodFilter::dirichlet_logpdf(const double* theta, const double alpha, const int size) {
    double sum_alpha = 0.0;
    double log_pdf = 0.0;

    // Compute sum of alpha
    for (int i = 0; i < size; ++i) {
        sum_alpha += alpha;
    }

    // First term: log(Γ(sum(α)))
    log_pdf += log_gamma(sum_alpha);

    // Second term: -sum(log(Γ(α_i)))
    for (int i = 0; i < size; ++i) {
        log_pdf -= log_gamma(alpha);
    }

    // Third term: sum((α_i - 1) * log(θ_i))
    for (int i = 0; i < size; ++i) {
        log_pdf += (alpha - 1.0) * std::log(theta[i]);
    }

    return log_pdf;
}