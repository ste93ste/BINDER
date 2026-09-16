#define _USE_MATH_DEFINES
#include "SSDLikelihoodFilter.h"
#include <cmath>
#include <random>
#include <vector>
#include <memory>
#include <iostream>

#include "Likelihood.h"

SSDLikelihoodFilter::SSDLikelihoodFilter(const double* coeff, const int threads, const double* sigma,
                                       const double* nodes, const double* voxels,
                                       const int reg_across_features, const double alpha_0, const double beta_0,
                                       const int M, const int F,
                                       const int Nx_n, const int Ny_n, const int Nz_n,
                                       const int Nx_v, const int Ny_v, const int Nz_v)
    : M(M), F(F), threads(threads), reg_across_features(reg_across_features),
      alpha_0(alpha_0), beta_0(beta_0),
      sample(false), Nx_n(Nx_n), Ny_n(Ny_n),
      Nz_n(Nz_n), Nx_v(Nx_v), Ny_v(Ny_v),
      Nz_v(Nz_v), nodes(nodes), voxels(voxels) {

    this->coeff = std::vector<double>(M * F);
    for (int m = 0; m < M; ++m) {
        for (int f = 0; f < F; ++f) {
            this->coeff[m * F + f] = coeff[m * F + f];
        }
    }

    this->sigma = std::vector<double>(M);
    for (int m = 0; m < M; ++m) {
        this->sigma[m] = sigma[m];
    }

    initialize_parameters();
}

SSDLikelihoodFilter::~SSDLikelihoodFilter() = default;

void SSDLikelihoodFilter::initialize_parameters() {
    const int node_volume = Nx_n * Ny_n * Nz_n;

    sum_tmp.resize(F * F);
    sum_tmp_2.resize(M * F);
    gamma_image.resize(threads * node_volume);
    sigma_tmp_1.resize(threads * M * node_volume);
    sigma_tmp_2.resize(threads * M * node_volume);
    outside_gamma_image.resize(threads);
    outside_sigma_tmp_1.resize(threads * M);
    outside_sigma_tmp_2.resize(threads * M);
    sigma_num.resize(M);
    constant.resize(M);
    log_constant.resize(M);

    std::fill(sum_tmp.begin(), sum_tmp.end(), 0.0);
    std::fill(sum_tmp_2.begin(), sum_tmp_2.end(), 0.0);
    std::fill(gamma_image.begin(), gamma_image.end(), 0.0);
    std::fill(sigma_tmp_1.begin(), sigma_tmp_1.end(), 0.0);
    std::fill(sigma_tmp_2.begin(), sigma_tmp_2.end(), 0.0);
    std::fill(outside_gamma_image.begin(), outside_gamma_image.end(), 0.0);
    std::fill(outside_sigma_tmp_1.begin(), outside_sigma_tmp_1.end(), 0.0);
    std::fill(outside_sigma_tmp_2.begin(), outside_sigma_tmp_2.end(), 0.0);
    std::fill(sigma_num.begin(), sigma_num.end(), 0.0);

    effective_voxels = 0.0;

    for (int m = 0; m < M; ++m) {
        constant[m] = std::sqrt(2.0 * M_PI * sigma[m]);
        log_constant[m] = -0.5 * std::log(2.0 * M_PI * sigma[m]);
    }

}

double SSDLikelihoodFilter::compute_cost() {
    double cost = 0.0;

    for (int m = 0; m < M; ++m) {
        // Implement log pdf of inverse gamma distribution
        // P(x|a,b) = b^a/Gamma(a) * x^(-a-1) * exp(-b/x)
        // log P(x|a,b) = a*log(b) - logGamma(a) + (-a-1)*log(x) - b/x
        const double a = alpha_0;
        const double b = alpha_0 * beta_0;
        const double x = sigma[m];

        //
        cost += -(a * std::log(b) - std::lgamma(a) + (-a-1.0) * std::log(x) - b/x);

    }

    return cost;
}

double SSDLikelihoodFilter::compute_likelihood(const int voxel_index,
                                            const int x_n, const int y_n, const int z_n) {
    double likelihood = 1.0;
    const bool outside = x_n < 0 || y_n < 0 || z_n < 0 ||
                        x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n;

    if (reg_across_features == 0) {
        for (int m = 0; m < M; ++m) {
            double tmp = voxels[voxel_index * M + m];

            if (!outside) {
                const int node_base_idx = (x_n * Ny_n * Nz_n + y_n * Nz_n + z_n) * F;
                for (int f = 0; f < F; ++f) {
                    tmp -= coeff[m * F + f] * nodes[node_base_idx + f];
                }
            }

            likelihood *= std::exp((tmp * tmp) / (-2.0 * sigma[m])) / constant[m];
        }
    } else {
        if (outside) {
            for (int m = 0; m < M; ++m) {
                const double voxel_val = voxels[voxel_index * M + m];
                likelihood *= std::exp((voxel_val * voxel_val) / (-2.0 * sigma[m])) / constant[m];
            }
        } else {
            const int node_base_idx = (x_n * Ny_n * Nz_n + y_n * Nz_n + z_n) * M;
            for (int m = 0; m < M; ++m) {
                const double diff = voxels[voxel_index * M + m] - nodes[node_base_idx + m];
                likelihood *= std::exp((diff * diff) / (-2.0 * sigma[m])) / constant[m];
            }
        }
    }

    return likelihood;
}

double SSDLikelihoodFilter::compute_log_likelihood(const int voxel_index,
                                                   const int x_n, const int y_n, const int z_n) {
    double log_likelihood = 0.0;
    const bool outside = x_n < 0 || y_n < 0 || z_n < 0 ||
                        x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n;

    const int voxel_base_idx = voxel_index * M;

    if (reg_across_features == 0) {
        for (int m = 0; m < M; ++m) {
            double tmp = voxels[voxel_base_idx + m];

            if (!outside) {
                const int node_base_idx = (x_n * Ny_n * Nz_n + y_n * Nz_n + z_n) * F;
                for (int f = 0; f < F; ++f) {
                    tmp -= coeff[m * F + f] * nodes[node_base_idx + f];
                }
            }

            log_likelihood += (tmp * tmp) / (-2.0 * sigma[m]) + log_constant[m];
        }
    } else {
        if (outside) {
            for (int m = 0; m < M; ++m) {
                const double voxel_val = voxels[voxel_base_idx + m];
                log_likelihood += (voxel_val * voxel_val) / (-2.0 * sigma[m]) + log_constant[m];
            }
        } else {
            const int node_base_idx = (x_n * Ny_n * Nz_n + y_n * Nz_n + z_n) * M;
            for (int m = 0; m < M; ++m) {
                const double diff = voxels[voxel_base_idx + m] - nodes[node_base_idx + m];
                log_likelihood += (diff * diff) / (-2.0 * sigma[m]) + log_constant[m];
            }
        }
    }

    return log_likelihood;
}

void SSDLikelihoodFilter::accumulate_info(const int thread_id, const int voxel_index,
                                          const int x_n, const int y_n, const int z_n, const double posterior) {
    const bool outside = x_n < 0 || y_n < 0 || z_n < 0 ||
                        x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n;
    const int node_volume = Nx_n * Ny_n * Nz_n;

    if (!outside) {
        const int node_idx = x_n * Ny_n * Nz_n + y_n * Nz_n + z_n;
        const int base_idx = thread_id * node_volume + node_idx;
        gamma_image[base_idx] += posterior;

        for (int m = 0; m < M; ++m) {
            const int voxel_idx = voxel_index * M + m;
            const int sigma_idx = thread_id * M * node_volume + m * node_volume + node_idx;

            const double voxel_val = voxels[voxel_idx];
            sigma_tmp_1[sigma_idx] += posterior * voxel_val;
            sigma_tmp_2[sigma_idx] += posterior * voxel_val * voxel_val;
        }
    } else {
        outside_gamma_image[thread_id] += posterior;

        for (int m = 0; m < M; ++m) {
            const int voxel_idx = voxel_index * M + m;
            const double voxel_val = voxels[voxel_idx];

            outside_sigma_tmp_1[thread_id * M + m] += posterior * voxel_val;
            outside_sigma_tmp_2[thread_id * M + m] += posterior * voxel_val * voxel_val;
        }
    }
}

void SSDLikelihoodFilter::update_parameters() {
    update_coefficients();
    update_sigma();
}

void SSDLikelihoodFilter::update_coefficients() {
    if (reg_across_features == 1) {
        effective_voxels = 0.0;
        const int node_volume = Nx_n * Ny_n * Nz_n;

        for (int t = 0; t < threads; ++t) {
            for (int i = 0; i < node_volume; ++i) {
                effective_voxels += gamma_image[t * node_volume + i];
            }
            effective_voxels += outside_gamma_image[t];
        }
        return;
    }

    std::fill(sum_tmp.begin(), sum_tmp.end(), 0.0);
    std::fill(sum_tmp_2.begin(), sum_tmp_2.end(), 0.0);
    effective_voxels = 0.0;

    const int node_volume = Nx_n * Ny_n * Nz_n;

    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        for (int x_m = 0; x_m < Nx_n; ++x_m) {
            for (int y_m = 0; y_m < Ny_n; ++y_m) {
                for (int z_m = 0; z_m < Nz_n; ++z_m) {
                    const int node_idx = x_m * Ny_n * Nz_n + y_m * Nz_n + z_m;
                    const int node_base_idx = node_idx * F;
                    const double gamma_val = gamma_image[thread_id * node_volume + node_idx];

                    for (int f1 = 0; f1 < F; ++f1) {
                        for (int f2 = 0; f2 < F; ++f2) {
                            sum_tmp[F * f1 + f2] += nodes[node_base_idx + f1] *
                                                   nodes[node_base_idx + f2] *
                                                   gamma_val;
                        }
                    }
                    effective_voxels += gamma_val;
                }
            }
        }
        effective_voxels += outside_gamma_image[thread_id];
    }

    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        for (int m = 0; m < M; ++m) {
            for (int x_m = 0; x_m < Nx_n; ++x_m) {
                for (int y_m = 0; y_m < Ny_n; ++y_m) {
                    for (int z_m = 0; z_m < Nz_n; ++z_m) {
                        const int node_idx = x_m * Ny_n * Nz_n + y_m * Nz_n + z_m;
                        const int node_base_idx = node_idx * F;
                        const double sigma_val = sigma_tmp_1[thread_id * M * node_volume +
                                                         m * node_volume + node_idx];

                        for (int f1 = 0; f1 < F; ++f1) {
                            sum_tmp_2[m * F + f1] += sigma_val * nodes[node_base_idx + f1];
                        }
                    }
                }
            }
        }
    }

    std::vector A(F, std::vector<double>(F));
    std::vector b(M, std::vector<double>(F));

    for (int f1 = 0; f1 < F; ++f1) {
        for (int f2 = 0; f2 < F; ++f2) {
            A[f1][f2] = sum_tmp[F * f1 + f2];
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int f1 = 0; f1 < F; ++f1) {
            b[m][f1] = sum_tmp_2[m * F + f1];
        }
    }

    std::vector<double> x(F);
    for (int m = 0; m < M; ++m) {
        solve_linear_system(A, b[m], x);

        if (sample) {
            std::vector<double> sampled_coeff = sample_multivariate_normal(x, A, gen);
            for (int f = 0; f < F; ++f) {
                coeff[m * F + f] = sampled_coeff[f];
            }
        } else {
            for (int f = 0; f < F; ++f) {
                coeff[m * F + f] = x[f];
            }
        }
    }
}

void SSDLikelihoodFilter::update_sigma() {
    std::fill(sigma_num.begin(), sigma_num.end(), 0.0);
    const int node_volume = Nx_n * Ny_n * Nz_n;

    for (int x_m = 0; x_m < Nx_n; ++x_m) {
        for (int y_m = 0; y_m < Ny_n; ++y_m) {
            for (int z_m = 0; z_m < Nz_n; ++z_m) {
                const int node_idx = x_m * Ny_n * Nz_n + y_m * Nz_n + z_m;
                const int node_base_idx = node_idx * F;

                for (int m = 0; m < M; ++m) {
                    double tmp1 = 0.0;  // Coefficient sum
                    double tmp3 = 0.0;  // sigma_tmp_2 sum
                    double tmp4 = 0.0;  // sigma_tmp_1 sum
                    double tmp5 = 0.0;  // gamma_image sum

                    // Calculate coefficient sum with correct indexing for nodes array
                    if (reg_across_features == 0) {
                        for (int f1 = 0; f1 < F; ++f1) {
                            tmp1 += coeff[m * F + f1] * nodes[node_base_idx + f1];
                        }
                    } else {
                        tmp1 = nodes[node_idx * M + m];
                    }

                    // Sum up thread contributions
                    for (int thread_id = 0; thread_id < threads; ++thread_id) {
                        const int thread_offset = thread_id * node_volume;
                        const int sigma_offset = thread_id * M * node_volume + m * node_volume;

                        tmp3 += sigma_tmp_2[sigma_offset + node_idx];
                        tmp4 += sigma_tmp_1[sigma_offset + node_idx];
                        tmp5 += gamma_image[thread_offset + node_idx];
                    }

                    sigma_num[m] += tmp3 + (-2.0 * tmp4 * tmp1) + (tmp5 * tmp1 * tmp1);
                }
            }
        }
    }

    // Add contribution of voxels falling outside nodes boundaries
    for (int m = 0; m < M; ++m) {
        for (int thread_id = 0; thread_id < threads; ++thread_id) {
            sigma_num[m] += outside_sigma_tmp_2[thread_id * M + m];
        }
    }

    // Update sigma values
    for (int m = 0; m < M; ++m) {
        // Add prior contribution
        sigma_num[m] += 2.0 * alpha_0 * beta_0;
        const double tmp_den = effective_voxels + 2.0 * alpha_0;

        if (sample) {
            // Sample from inverse gamma distribution
            sigma[m] = sample_inverse_gamma(0.5 * tmp_den,
                                         0.5 * sigma_num[m],
                                         gen);
        } else {
            sigma[m] = sigma_num[m] / tmp_den;
        }
        std::cout << "σ[" << m << "]=" << sigma[m] << std::endl;
    }
}

// Helper functions for solving linear systems and sampling
void SSDLikelihoodFilter::solve_linear_system(const std::vector<std::vector<double>>& A,
                                              const std::vector<double>& b,
                                              std::vector<double>& x) {
    // Simple Gaussian elimination implementation
    const int n = static_cast<int>(A.size());
    std::vector aug(n, std::vector<double>(n + 1));

    // Create augmented matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j] = A[i][j];
        }
        aug[i][n] = b[i];
    }

    // Gaussian elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        int pivot = i;
        for (int j = i + 1; j < n; j++) {
            if (std::abs(aug[j][i]) > std::abs(aug[pivot][i])) {
                pivot = j;
            }
        }

        // Swap rows if necessary
        if (pivot != i) {
            std::swap(aug[i], aug[pivot]);
        }

        // Eliminate column
        for (int j = i + 1; j < n; j++) {
            const double factor = aug[j][i] / aug[i][i];
            for (int k = i; k <= n; k++) {
                aug[j][k] -= factor * aug[i][k];
            }
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += aug[i][j] * x[j];
        }
        x[i] = (aug[i][n] - sum) / aug[i][i];
    }
}

std::vector<double> SSDLikelihoodFilter::sample_multivariate_normal(
    const std::vector<double>& mean,
    const std::vector<std::vector<double>>& covariance,
    std::mt19937& gen) {

    const int n = static_cast<int>(mean.size());
    std::vector<double> result(n);
    std::normal_distribution normal(0.0, 1.0);

    // Compute Cholesky decomposition of covariance matrix
    std::vector L(n, std::vector(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += L[j][k] * L[j][k];
                }
                L[j][j] = std::sqrt(covariance[j][j] - sum);
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (covariance[i][j] - sum) / L[j][j];
            }
        }
    }

    // Generate random sample
    for (int i = 0; i < n; i++) {
        result[i] = mean[i];
        for (int j = 0; j <= i; j++) {
            result[i] += L[i][j] * normal(gen);
        }
    }

    return result;
}

double SSDLikelihoodFilter::sample_inverse_gamma(const double alpha, const double beta, std::mt19937& gen) {
    std::gamma_distribution gamma(alpha, 1.0/beta);
    return 1.0 / gamma(gen);
}

void SSDLikelihoodFilter::sample_parameters() {
    sample = true;
    update_parameters();
    sample = false;
}

void SSDLikelihoodFilter::set_generator(std::mt19937& gen) {
    this->gen = gen;
}

std::vector<double> SSDLikelihoodFilter::get_parameters() {
    std::vector<double> combined;
    combined.reserve(M * F);
    for (int m = 0; m < M; ++m) {
        for (int f = 0; f < F; ++f) {
            combined.push_back(coeff[m * F + f]);
        }
    }

    return combined;
}