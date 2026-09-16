//
// Created by stce on 01/01/25.
//

#ifndef MILIKELIHOODFILTER_H
#define MILIKELIHOODFILTER_H

#include <memory>
#include <random>

#include "Likelihood.h"

class MILikelihoodFilter final : public Likelihood {
public:
    // Constructor
    MILikelihoodFilter(double alpha, const double* theta, int threads,
                        const int* binned_nodes,
                        const int* binned_voxels,
                        int K, int L, int M,
                        int Nx_n, int Ny_n, int Nz_n,
                        int Nx_v, int Ny_v, int Nz_v);

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

    // Joint histogram computation
    void joint_histogram_thread_safe() const;

    void set_generator(std::mt19937 &gen) override;

    // Sample from Dirichlet distribution
    void sample_parameters() override;

    void initialize_parameters() override;

    // Get parameters
    [[nodiscard]] std::vector<double> get_parameters() override;

    ~MILikelihoodFilter() override;

    static double log_gamma(double x);
    static double dirichlet_logpdf(const double *theta, double alpha, int size);

private:

    // Initialization function

    // Member variables
    double alpha;
    std::mt19937 gen;
    int K, L, KL, M, threads;
    int Nx_n, Ny_n, Nz_n;
    int Nx_v, Ny_v, Nz_v;
    const int *binned_nodes;
    const int *binned_voxels;
    std::unique_ptr<double[]> log_theta;
    std::unique_ptr<double[]> theta;
    std::unique_ptr<double[]> theta_tmp;
    int Ny_n_Nz_n;
};

#endif // MILIKELIHOODFILTER_H
