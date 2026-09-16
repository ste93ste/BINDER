//
// Created by stce on 01/01/25.
//

#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <vector>
#include <memory>
#include "Likelihood.h"
#include "MILikelihood.h"
#include "SSDLikelihood.h"
#include "MILikelihoodFilter.h"
#include "SSDLikelihoodFilter.h"
#include "Spline.h"
#include "Transformation.h"
#include "NonLinearTransformation.h"
#include "AffineTransformation.h"
#include "RigidTransformation.h"
#include <random>

class Registration {

    std::unique_ptr<double[]> ds;
    std::unique_ptr<int[]> node_indices;
    std::unique_ptr<double[]> dream_locations;
    int threads;
    int seed;
    std::mt19937 rng;
    const double *voxel_pos;
    const double * node_pos;
    int Nx_v, Ny_v, Nz_v, Nx_n, Ny_n, Nz_n, N;
    std::vector<double> mll_t;
    std::unique_ptr<double[]> final_locations;
    double num_voxels;
    double non_zero_voxels;
    int log_space{};
    int sample{};
    bool is2D;
    std::vector<std::mt19937> thread_rngs;
    std::unique_ptr<Likelihood> lkObj;
    std::unique_ptr<Spline> splineObj;
    std::unique_ptr<Transformation> transformationObj;

public:
    Registration(const double *final_locations, int threads, const double *voxel_pos, const double *node_pos,
                 int Nx_v, int Ny_v, int Nz_v, int Nx_n, int Ny_n, int Nz_n, int spline_order, double non_zero_voxels);

    ~Registration();

    void EM(int max_EM_iterations, double convergence_th,
            bool update_likelihood_parameters, bool debug);

    std::pair<std::vector<double>, std::vector<double>> sampler(int seed, int N_b, int N_s, int sample_gamma,
                                                                int sample_gamma_every, int sample_posteriors,
                                                                int sample_likelihood_parameters,
                                                                int sample_transformation_parameters);

    void setMILikelihood(double alpha,
                         const double* theta,
                         int K, int L,
                         int threads,
                         const int* binned_nodes,
                         const int* binned_voxels);

    void setMILikelihoodFilter(double alpha, const double* theta, int threads,
                               const int* binned_nodes, const int* binned_voxels,
                               int K, int L, int M);

    void setSSDLikelihood(int bins, int threads, double sigmaSq, int shiftScale,
                          double scale, double shift, const int* binned_nodes,
                          const int* binned_voxels,
                          double alpha_0, double beta_0);

    void setSSDLikelihoodFilter(const double *coeff, int threads, const double *sigma,
                                const double *features_nodes,
                                const double *features_voxels,
                                int reg_across_features, double alpha_0, double beta_0,
                                int M, int F);

    void setNonLinearTransformation(double gamma_x, double gamma_y, double gamma_z,
                                    double rho_x, double rho_y, double rho_z,
                                    double eta_x, double eta_y, double eta_z,
                                    const double* D);
    void setAffineTransformation(const double* A, const double* t);
    void setRigidTransformation(const double *R, const double *t);
    [[nodiscard]] std::vector<double> get_final_locations() const;

    std::vector<double> get_dream_locations() const;

    [[nodiscard]] std::vector<int> get_shape() const;

    void set_dream_locations(const double *dream_locations);

    std::vector<double> get_likelihood_parameters() const;

private:
    void initialize_distances_and_indices() const;

    void set_final_locations(const double *final_locations) const;

    void initialize_dream_locations() const;
    void estimate_posterior_and_dream_location(int x, int y, int z, int thread_id);

    void estimate_posterior_and_dream_location_log(int x, int y, int z, int thread_id);
};

#endif //REGISTRATION_H
