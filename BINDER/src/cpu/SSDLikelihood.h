//
// Created by stce on 04/01/25.
//

#ifndef SSDLIKELIHOOD_H
#define SSDLIKELIHOOD_H
#include <random>
#include <Eigen/QR>
#include <Eigen/Dense>
#include "Likelihood.h"


class SSDLikelihood final : public Likelihood {
public:
    SSDLikelihood(int bins, int threads, double sigmaSq, int shiftScale, double scale, double shift,
                  const int* binned_nodes, const int* binned_voxels,
                  double alpha_0, double beta_0,
                  int Nx_n, int Ny_n, int Nz_n,
                  int Nx_v, int Ny_v, int Nz_v);

    void update_shift_and_scale();
    void update_sigma();

    // Parameters
    int bins;
    int threads;
    double sigmaSq;
    int shiftScale;
    double scale;
    double shift;
    const int* binned_nodes;
    const int* binned_voxels;
    double alpha_0;
    double beta_0;

    // Dimensions
    int Nx_n, Ny_n, Nz_n;  // Node dimensions
    int Nx_v, Ny_v, Nz_v;  // Voxel dimensions
    int num_voxels;

    // Precomputed constants
    double constant{};
    double log_constant{};

    // State
    int sample;

    // Random number generator
    std::mt19937 gen;

    // Vectors for computations
    std::vector<double> sigmaSumNum;
    std::vector<double> sigmaSumDen;

    // Vectors for shift-scale computations
    std::vector<double> gamma_image;
    std::vector<double> sigma_tmp_1;
    std::vector<double> sigma_tmp_2;
    std::vector<double> sum_A_tmp;
    std::vector<double> sum_b_tmp;
    std::vector<double> outside_gamma_image;
    std::vector<double> outside_sigma_tmp_1;
    std::vector<double> outside_sigma_tmp_2;

    // Compute cost function (log prior)
    double compute_cost() override;

    // Compute the likelihood (P(theta))
    double compute_likelihood(int voxel_index, int x_n, int y_n, int z_n) override;

    // Compute the log likelihood (log(P(theta)))
    double compute_log_likelihood(int voxel_index, int x_n, int y_n, int z_n) override;

    // Update parameters based on the current information
    void update_parameters() override;

    // Accumulate information for posterior
    void accumulate_info(int thread_id, int voxel_index, int x_n, int y_n, int z_n, double posterior) override;

    void set_generator(std::mt19937 &gen) override;

    std::vector<double> get_parameters() override;

    // Sample
    void sample_parameters() override;

    void initialize_parameters() override;
};



#endif //SSDLIKELIHOOD_H
