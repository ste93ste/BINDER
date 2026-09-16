// SSDLikelihoodFilter.h
#ifndef SSDLIKELIHOODFILTER_H
#define SSDLIKELIHOODFILTER_H
#include <vector>
#include <random>
#include <cmath>
#include <memory>

#include "Likelihood.h"
#include <Eigen/Dense>

class SSDLikelihoodFilter final : public Likelihood {
public:
    SSDLikelihoodFilter(const double* coeff, int threads, const double* sigma,
                        const double* nodes, const double* voxels,
                        int reg_across_features, double alpha_0, double beta_0,
                        int M, int F,
                        int Nx_n, int Ny_n, int Nz_n,
                        int Nx_v, int Ny_v, int Nz_v);

    ~SSDLikelihoodFilter() override;

    double compute_cost() override;
    double compute_likelihood(int voxel_index,
                              int x_n, int y_n, int z_n) override;
    double compute_log_likelihood(int voxel_index,
                                  int x_n, int y_n, int z_n) override;
    void update_parameters() override;
    void accumulate_info(int thread_id, int voxel_index,
                        int x_n, int y_n, int z_n, double posterior) override;
    void initialize_parameters() override;
    void sample_parameters() override;
    void set_generator(std::mt19937& gen) override;
    std::vector<double> get_parameters() override;

private:
    void update_coefficients();
    void update_sigma();

    // Core parameters
    int M, F;  // Number of measurements and features
    int threads;
    int reg_across_features;
    double alpha_0, beta_0;
    bool sample;

    // Data dimensions
    int Nx_n, Ny_n, Nz_n;  // Node dimensions
    int Nx_v, Ny_v, Nz_v;  // Voxel dimensions

    // Input data (pointers to external memory)
    const double* nodes;
    const double* voxels;
    std::vector<double> coeff;
    std::vector<double> sigma;

    // Internal storage
    std::vector<double> sum_tmp;
    std::vector<double> sum_tmp_2;
    std::vector<double> gamma_image;
    std::vector<double> sigma_tmp_1;
    std::vector<double> sigma_tmp_2;
    std::vector<double> outside_gamma_image;
    std::vector<double> outside_sigma_tmp_1;
    std::vector<double> outside_sigma_tmp_2;
    std::vector<double> sigma_num;
    std::vector<double> constant;
    std::vector<double> log_constant;
    double effective_voxels{};

    // Random number generator
    std::mt19937 gen;

    static double sample_inverse_gamma(double alpha, double beta, std::mt19937& gen);
    static void solve_linear_system(const std::vector<std::vector<double>>& A,
                             const std::vector<double>& b,
                             std::vector<double>& x);
    static std::vector<double> sample_multivariate_normal(const std::vector<double>& mean,
                                                   const std::vector<std::vector<double>>& covariance,
                                                   std::mt19937& gen);


};

#endif //SSDLIKELIHOODFILTER_H
