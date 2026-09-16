#define _USE_MATH_DEFINES
#include "SSDLikelihood.h"
#include <cmath>
#include <random>
#include <iostream>

SSDLikelihood::SSDLikelihood(const int bins, const int threads, const double sigmaSq, const int shiftScale,
                             const double scale, const double shift,
                             const int* binned_nodes, const int* binned_voxels,
                             const double alpha_0, const double beta_0,
                             const int Nx_n, const int Ny_n, const int Nz_n,
                             const int Nx_v, const int Ny_v, const int Nz_v)
    : bins(bins), threads(threads), sigmaSq(sigmaSq), shiftScale(shiftScale), scale(scale), shift(shift),
      binned_nodes(binned_nodes), binned_voxels(binned_voxels),
      alpha_0(alpha_0), beta_0(beta_0),
      Nx_n(Nx_n), Ny_n(Ny_n), Nz_n(Nz_n), Nx_v(Nx_v), Ny_v(Ny_v), Nz_v(Nz_v),
      sample(0) {

    num_voxels = Nx_v * Ny_v * Nz_v;

    // Initialize vectors instead of malloc
    sigmaSumNum.resize(threads, 0.0);
    sigmaSumDen.resize(threads, 0.0);

    if (shiftScale == 1) {
        // Initialize arrays for shift-scale calculations
        const int node_size = Nx_n * Ny_n * Nz_n;
        gamma_image.resize(threads * node_size, 0.0);
        sigma_tmp_1.resize(threads * node_size, 0.0);
        sigma_tmp_2.resize(threads * node_size, 0.0);
        sum_A_tmp.resize(4, 0.0); // 2x2 matrix
        sum_b_tmp.resize(2, 0.0);
        outside_gamma_image.resize(threads, 0.0);
        outside_sigma_tmp_1.resize(threads, 0.0);
        outside_sigma_tmp_2.resize(threads, 0.0);
    }
}

void SSDLikelihood::initialize_parameters() {
    // Reset all vectors to zero
    std::fill(sigmaSumNum.begin(), sigmaSumNum.end(), 0.0);
    std::fill(sigmaSumDen.begin(), sigmaSumDen.end(), 0.0);

    if (shiftScale == 1) {
        std::fill(gamma_image.begin(), gamma_image.end(), 0.0);
        std::fill(sigma_tmp_1.begin(), sigma_tmp_1.end(), 0.0);
        std::fill(sigma_tmp_2.begin(), sigma_tmp_2.end(), 0.0);
        std::fill(sum_A_tmp.begin(), sum_A_tmp.end(), 0.0);
        std::fill(sum_b_tmp.begin(), sum_b_tmp.end(), 0.0);
        std::fill(outside_gamma_image.begin(), outside_gamma_image.end(), 0.0);
        std::fill(outside_sigma_tmp_1.begin(), outside_sigma_tmp_1.end(), 0.0);
        std::fill(outside_sigma_tmp_2.begin(), outside_sigma_tmp_2.end(), 0.0);
    }

    // Precompute constants
    constant = std::sqrt(2.0 * M_PI * sigmaSq);
    log_constant = -std::log(std::sqrt(2.0 * M_PI * sigmaSq));
}

double SSDLikelihood::compute_log_likelihood(const int voxel_index,
                                             const int x_n, const int y_n, const int z_n) {
    const bool outside = x_n < 0 || y_n < 0 || z_n < 0 ||
                         x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n;

    if (shiftScale == 0) {
        if (outside) {
            const double voxel_val = binned_voxels[voxel_index];
            return (voxel_val * voxel_val) / (-2.0 * sigmaSq) + log_constant;
        }
        {
            const double diff = binned_voxels[voxel_index] -
                          binned_nodes[x_n * Ny_n * Nz_n + y_n * Nz_n + z_n];
            return (diff * diff) / (-2.0 * sigmaSq) + log_constant;
        }
    }
    {
        const double voxel_val = binned_voxels[voxel_index];
        double adjusted_val;
        if (outside) {
            adjusted_val = voxel_val - shift;
        } else {
            adjusted_val = voxel_val - shift - scale * binned_nodes[x_n * Ny_n * Nz_n + y_n * Nz_n + z_n];
        }
        return (adjusted_val * adjusted_val) / (-2.0 * sigmaSq) + log_constant;
    }
}

double SSDLikelihood::compute_likelihood(const int voxel_index, const int x_n, const int y_n, const int z_n) {
    // Check if the node index is out of bounds
    const bool outside = x_n < 0 || y_n < 0 || z_n < 0 || x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n;

    if (shiftScale == 0) {
        if (outside) {
            const double voxel_val = binned_voxels[voxel_index];
            return exp((voxel_val * voxel_val) / (-2.0 * sigmaSq)) / constant;
        }
        {
            const int node_index = x_n * Ny_n * Nz_n + y_n * Nz_n + z_n;
            const double diff = binned_voxels[voxel_index] - binned_nodes[node_index];
            return exp((diff * diff) / (-2.0 * sigmaSq)) / constant;
        }
    }
    {
        const double voxel_val = binned_voxels[voxel_index];
        if (outside) {
            const double diff = voxel_val - shift;
            return exp((diff * diff) / (-2.0 * sigmaSq)) / constant;
        }
        {
            const int node_index = x_n * Ny_n * Nz_n + y_n * Nz_n + z_n;
            const double diff = voxel_val - shift - scale * binned_nodes[node_index];
            return exp((diff * diff) / (-2.0 * sigmaSq)) / constant;
        }
    }
}

void SSDLikelihood::accumulate_info(const int thread_id, const int voxel_index,
                                    const int x_n, const int y_n, const int z_n, const double posterior) {

    const bool outside = x_n < 0 || y_n < 0 || z_n < 0 ||
                         x_n >= Nx_n || y_n >= Ny_n || z_n >= Nz_n;

    if (shiftScale == 0) {
        if (outside) {
            const double voxel_val = binned_voxels[voxel_index];
            sigmaSumNum[thread_id] += posterior * voxel_val * voxel_val;
        } else {
            const double diff = binned_voxels[voxel_index] - binned_nodes[x_n * Ny_n * Nz_n + y_n * Nz_n + z_n];
            sigmaSumNum[thread_id] += posterior * diff * diff;
        }
        sigmaSumDen[thread_id] += posterior;
    } else {
        const int node_idx = thread_id * (Nx_n * Ny_n * Nz_n) + x_n * (Ny_n * Nz_n) + y_n * Nz_n + z_n;
        const double voxel_val = binned_voxels[voxel_index];

        if (outside) {
            outside_gamma_image[thread_id] += posterior;
            outside_sigma_tmp_1[thread_id] += posterior * voxel_val;
            outside_sigma_tmp_2[thread_id] += posterior * voxel_val * voxel_val;
        } else {
            gamma_image[node_idx] += posterior;
            sigma_tmp_1[node_idx] += posterior * voxel_val;
            sigma_tmp_2[node_idx] += posterior * voxel_val * voxel_val;
        }
    }
}

void SSDLikelihood::update_parameters() {
    if (shiftScale == 0) {
        double tmp_num = 0.0;
        double tmp_den = 0.0;

        for (int thread_id = 0; thread_id < threads; ++thread_id) {
            tmp_num += sigmaSumNum[thread_id];
            tmp_den += sigmaSumDen[thread_id];
        }

        tmp_num += alpha_0 * beta_0;
        tmp_den += alpha_0;

        sigmaSq = tmp_num / tmp_den;
        std::cout << "SigmaSq: " << sigmaSq << std::endl;
    } else {
        update_shift_and_scale();
        update_sigma();
    }
}

void SSDLikelihood::update_shift_and_scale() {
    Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
    Eigen::Vector2d b = Eigen::Vector2d::Zero();

    // Accumulate A and b info
    for (int x = 0; x < Nx_n; ++x) {
        for (int y = 0; y < Ny_n; ++y) {
            for (int z = 0; z < Nz_n; ++z) {
                const double node_val = binned_nodes[x * Ny_n * Nz_n + y * Nz_n + z];
                double gamma_sum = 0.0;
                double sigma_tmp_1_sum = 0.0;

                for (int thread_id = 0; thread_id < threads; ++thread_id) {
                    const int idx = thread_id * (Nx_n * Ny_n * Nz_n) + x * (Ny_n * Nz_n) + y * Nz_n + z;
                    gamma_sum += gamma_image[idx];
                    sigma_tmp_1_sum += sigma_tmp_1[idx];
                }

                A(0, 0) += gamma_sum;
                A(0, 1) += gamma_sum * node_val;
                A(1, 0) += gamma_sum * node_val;
                A(1, 1) += gamma_sum * node_val * node_val;

                b(0) += sigma_tmp_1_sum;
                b(1) += sigma_tmp_1_sum * node_val;
            }
        }
    }

    // Add outside contribution
    double outside_gamma_sum = 0.0;
    double outside_sigma_tmp_1_sum = 0.0;
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        outside_gamma_sum += outside_gamma_image[thread_id];
        outside_sigma_tmp_1_sum += outside_sigma_tmp_1[thread_id];
    }
    A(0, 0) += outside_gamma_sum;
    b(0) += outside_sigma_tmp_1_sum;

    // Solve using QR decomposition
    Eigen::Vector2d x = A.colPivHouseholderQr().solve(b);

    if (sample) {
        // Sample from multivariate normal
        Eigen::Matrix2d cov = A.inverse();
        std::normal_distribution normal(0.0, 1.0);

        Eigen::Vector2d z;
        z << normal(gen), normal(gen);

        // Cholesky decomposition for sampling
        const Eigen::LLT<Eigen::Matrix2d> llt(cov);
        Eigen::Vector2d sample = x + llt.matrixL() * z;

        shift = sample(0);
        scale = sample(1);

        std::cout << "Sampled shift: " << shift << std::endl;
        std::cout << "Sampled scale: " << scale << std::endl;
    } else {
        shift = x(0);
        scale = x(1);

        std::cout << "Shift: " << shift << std::endl;
        std::cout << "Scale: " << scale << std::endl;
    }
}

void SSDLikelihood::update_sigma() {
    double tmp_num = 0.0;
    double tmp_den = 0.0;

    for (int x = 0; x < Nx_n; ++x) {
        for (int y = 0; y < Ny_n; ++y) {
            for (int z = 0; z < Nz_n; ++z) {
                const double tmp_1 = shift + scale * binned_nodes[x * Ny_n * Nz_n + y * Nz_n + z];
                double tmp_3 = 0.0;
                double tmp_4 = 0.0;
                double tmp_5 = 0.0;

                for (int thread_id = 0; thread_id < threads; ++thread_id) {
                    const int idx = thread_id * (Nx_n * Ny_n * Nz_n) + x * (Ny_n * Nz_n) + y * Nz_n + z;
                    tmp_3 += sigma_tmp_2[idx];
                    tmp_4 += sigma_tmp_1[idx];
                    tmp_5 += gamma_image[idx];
                    tmp_den += gamma_image[idx];
                }

                tmp_num += tmp_3 + -2.0 * tmp_4 * tmp_1 + tmp_5 * (tmp_1 * tmp_1);
            }
        }
    }

    // Add outside contribution
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        tmp_num += outside_sigma_tmp_2[thread_id] -
                  2.0 * outside_sigma_tmp_1[thread_id] * shift +
                  outside_gamma_image[thread_id] * shift * shift;
        tmp_den += outside_gamma_image[thread_id];
    }

    tmp_num += alpha_0 * beta_0;
    tmp_den += alpha_0;

    if (!sample) {
        sigmaSq = tmp_num / tmp_den;
        std::cout << "SigmaSq: " << sigmaSq << std::endl;
    } else {
        // Sample from inverse gamma distribution
        std::gamma_distribution<double> gamma(tmp_den * 0.5, 2.0 * tmp_den / tmp_num);
        sigmaSq = 1.0 / gamma(gen);
        std::cout << "Sampled SigmaSq: " << sigmaSq << std::endl;
    }
}

void SSDLikelihood::sample_parameters() {
    sample = 1;

    if (shiftScale == 0) {
        double tmp_num = 0.0;
        double tmp_den = 0.0;

        for (int thread_id = 0; thread_id < threads; ++thread_id) {
            tmp_num += sigmaSumNum[thread_id];
            tmp_den += sigmaSumDen[thread_id];
        }

        tmp_num += alpha_0 * beta_0;
        tmp_den += alpha_0;

        // Sample from inverse gamma distribution
        std::gamma_distribution<double> gamma(tmp_den * 0.5, 2.0 * tmp_den / tmp_num);
        sigmaSq = 1.0 / gamma(gen);
        std::cout << "Sampled SigmaSq: " << sigmaSq << std::endl;
    } else {
        update_shift_and_scale();
        update_sigma();
    }
}


// Compute cost function (log prior)
double SSDLikelihood::compute_cost() {
    // Implement log pdf of inverse gamma distribution
    // P(x|a,b) = b^a/Gamma(a) * x^(-a-1) * exp(-b/x)
    // log P(x|a,b) = a*log(b) - logGamma(a) + (-a-1)*log(x) - b/x

    const double a = alpha_0;
    const double b = alpha_0 * beta_0;
    const double x = sigmaSq;

    //
    return -(a * std::log(b) - std::lgamma(a) + (-a-1.0) * std::log(x) - b/x);
}

void SSDLikelihood::set_generator(std::mt19937 &gen) {
    this->gen = gen;
}

std::vector<double> SSDLikelihood::get_parameters() {
    // Create a vector and add scale and shift
    std::vector<double> parameters;
    parameters.push_back(scale);
    parameters.push_back(shift);
    return parameters;
}
