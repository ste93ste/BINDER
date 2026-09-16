#include "MILikelihood.h"
#include <cmath>
#include <random>
#include <cstring>
#include <memory>

// Constructor
MILikelihood::MILikelihood(const double alpha, const double* theta, const int threads,
                           const int* binned_nodes, const int* binned_voxels,
                           const int K, const int L,
                           const int Nx_n, const int Ny_n, const int Nz_n,
                           const int Nx_v, const int Ny_v, const int Nz_v)
    : alpha(alpha), K(K), L(L), KL(K*L), threads(threads), Nx_n(Nx_n),
      Ny_n(Ny_n), Nz_n(Nz_n),
      Nx_v(Nx_v), Ny_v(Ny_v), Nz_v(Nz_v),
      binned_nodes(binned_nodes), binned_voxels(binned_voxels),
      Ny_n_Nz_n(Ny_n * Nz_n) {

    // Allocate memory for temporary variables using smart pointers
    this->theta = std::make_unique<double[]>(KL);
    this->log_theta = std::make_unique<double[]>(KL);
    for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
            this->theta[k * L + l] = theta[k * L + l];
            this->log_theta[k * L + l] = std::log(theta[k * L + l]);
        }
    }
    theta_tmp = std::make_unique<double[]>(KL * threads);
    MILikelihood::initialize_parameters();
}

// Compute cost function (log prior)
double MILikelihood::compute_cost() {
    double cost = 0.0;
    for (int k = 0; k < K; ++k) {
        cost += (-dirichlet_logpdf(&theta[k * L], alpha, L));
    }
    return cost;
}

// Compute the likelihood (P(theta))
double MILikelihood::compute_likelihood(const int voxel_index,
                                        const int x_n, const int y_n, const int z_n) {

    // Check if the node index is out of bounds
    if (x_n < 0 || y_n < 0 || z_n < 0 || x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n) {
        return theta[binned_voxels[voxel_index]];
    }

    // Precompute node indices
    const int node_index = x_n * Ny_n_Nz_n + y_n * Nz_n + z_n;

    // Return the likelihood value
    return theta[binned_nodes[node_index] * L + binned_voxels[voxel_index]];
}

// Compute the log likelihood (log(P(theta)))
double MILikelihood::compute_log_likelihood(const int voxel_index,
                                            const int x_n, const int y_n, const int z_n) {
    // Check bounds for the voxel and node indices
    if (x_n < 0 || y_n < 0 || z_n < 0 || x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n) {
        return log_theta[0 * L + binned_voxels[voxel_index]];
    }

    return log_theta[binned_nodes[x_n * Ny_n * Nz_n + y_n * Nz_n + z_n] * L +
                     binned_voxels[voxel_index]];

}

// Update parameters based on the current information
void MILikelihood::update_parameters() {
    joint_histogram_thread_safe();

    for (int k = 0; k < K; ++k) {
        double theta_normalizer = 0.0;
        // Compute the normalizer by summing over all l
        for (int l = 0; l < L; ++l) {
            theta_normalizer += (theta[k * L + l] + alpha - 1.0);
        }
        // Normalize theta
        for (int l = 0; l < L; ++l) {
            theta[k * L + l] = (theta[k * L + l] + alpha - 1.0) / theta_normalizer;
            log_theta[k * L + l] = std::log(theta[k * L + l]);
        }
    }
}

// Accumulate information for posterior
void MILikelihood::accumulate_info(const int thread_id, const int voxel_index,
                                   const int x_n, const int y_n, const int z_n, const double posterior) {
    const int base_index = thread_id * KL + binned_voxels[voxel_index];
    if (x_n < 0 || y_n < 0 || z_n < 0 || x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n) {
        theta_tmp[base_index] += posterior;
    } else {
        const int node_index = binned_nodes[x_n * Ny_n_Nz_n + y_n * Nz_n + z_n] * L;
        theta_tmp[base_index + node_index] += posterior;
    }
}

void MILikelihood::initialize_parameters() {
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        for (int k = 0; k < K; ++k) {
            for (int l = 0; l < L; ++l) {
                theta_tmp[(thread_id * KL) + (k * L) + l] = 0.0;
            }
        }
    }
}

// Joint histogram computation
void MILikelihood::joint_histogram_thread_safe() const {
    for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
            theta[k * L + l] = 0.0;
        }
    }
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        for (int k = 0; k < K; ++k) {
            for (int l = 0; l < L; ++l) {
                theta[k * L + l] += theta_tmp[thread_id * KL + k * L + l];
            }
        }
    }
}

void MILikelihood::set_generator(std::mt19937 &gen) {
    this->gen = gen;
}


// Sample from Dirichlet distribution
void MILikelihood::sample_parameters() {
    // Temporary array to hold samples (K * L)
    auto tmp = std::make_unique<double[]>(K * L);

    // Compute joint histogram (thread-safe)
    joint_histogram_thread_safe();

    // Sample from Dirichlet distribution (theta_k | alpha_k) for all k
    for (int k = 0; k < K; ++k) {
        // Create alpha_k = (N_k_1, ..., N_k_L) + alpha
        auto alpha_k = std::make_unique<double[]>(L);
        double sum = 0.0;

        // First, sample from gamma distributions
        for (int l = 0; l < L; ++l) {
            // Set shape parameter (alpha) from theta + prior
            const double shape = theta[k * L + l] + alpha;
            // Use scale parameter = 1.0 as is standard for Dirichlet sampling
            std::gamma_distribution<double> dist(shape, 1.0);

            // Sample from gamma distribution using existing RNG
            tmp[k * L + l] = dist(gen);
            sum += tmp[k * L + l];
        }

        // Normalize to get Dirichlet distribution
        for (int l = 0; l < L; ++l) {
            tmp[k * L + l] /= sum;
        }
    }

    // Update theta with the sampled values
    for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
            theta[k * L + l] = tmp[k * L + l];
        }
    }
}

// Get parameters
std::vector<double> MILikelihood::get_parameters() {
    double* theta_data = theta.get();
    return std::vector(theta_data, theta_data + KL);
}

// Function to compute log of gamma
double MILikelihood::log_gamma(const double x) {
    return std::lgamma(x);
}

// Dirichlet log PDF function using arrays
double MILikelihood::dirichlet_logpdf(const double* theta, const double alpha, const int size) {
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
