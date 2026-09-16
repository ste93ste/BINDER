//
// Created by stce on 01/01/25.
//

#include "Registration.h"
#include "Likelihood.h"
#include "MILikelihood.h"
#include "MILikelihoodFilter.h"
#include "SSDLikelihood.h"
#include "SSDLikelihoodFilter.h"
#include "Transformation.h"
#include "NonLinearTransformation.h"
#include "AffineTransformation.h"
#include "RigidTransformation.h"
#include <memory>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <random>


Registration::Registration(const double* final_locations,
                           const int threads,
                           const double* voxel_pos,
                           const double* node_pos,
                           const int Nx_v, const int Ny_v, const int Nz_v,
                           const int Nx_n, const int Ny_n, const int Nz_n,
                           const int spline_order, const double non_zero_voxels)
    : threads(threads),
      voxel_pos(voxel_pos),
      node_pos(node_pos),
      Nx_v(Nx_v),
      Ny_v(Ny_v),
      Nz_v(Nz_v),
      Nx_n(Nx_n),
      Ny_n(Ny_n),
      Nz_n(Nz_n),
      N(Nz_v == 1 ? 2 : 3),
      num_voxels(Nx_v * Ny_v * Nz_v),
      non_zero_voxels(non_zero_voxels),
      is2D(Nz_v == 1),
      splineObj(std::make_unique<Spline>(spline_order, is2D))
{
    mll_t.resize(threads, 0.0);

    //
    seed = 12345;  // will be set by sampler again
    log_space=false;
    thread_rngs.resize(threads);

    // Initialize the unique pointers
    // Initialize ds
    ds = std::make_unique<double[]>(Nx_v * Ny_v * Nz_v * N);
    // Initialize node indices
    node_indices = std::make_unique<int[]>(Nx_v * Ny_v * Nz_v * N);
    // Initialize dream locations
    dream_locations = std::make_unique<double[]>(Nx_v * Ny_v * Nz_v * N);

    // Initialize final_locations
    this->final_locations = std::make_unique<double[]>(Nx_v * Ny_v * Nz_v * N);

    // Copy final locations
    set_final_locations(final_locations);

    // Initialize distances and node indices
    initialize_distances_and_indices();
}

Registration::~Registration() = default;

void Registration::initialize_distances_and_indices() const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    const double tmp = final_locations[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n];
                    node_indices[x * (Ny_v * Nz_v * N) +
                                 y * (Nz_v * N) +
                                 z * N + n] = static_cast<int>(std::floor(tmp + splineObj->offset));
                    ds[x * (Ny_v * Nz_v * N) +
                       y * (Nz_v * N) +
                       z * N + n] = tmp - std::floor(tmp + splineObj->offset);
                }
            }
        }
    }
}

void Registration::set_final_locations(const double* final_locations) const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    this->final_locations[x * (Ny_v * Nz_v * N) +
                                          y * (Nz_v * N) +
                                          z * N + n] = final_locations[x * (Ny_v * Nz_v * N) +
                                                                       y * (Nz_v * N) +
                                                                       z * N + n];
                }
            }
        }
    }
}

void Registration::initialize_dream_locations() const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    dream_locations[x * (Ny_v * Nz_v * N) +
                                    y * (Nz_v * N) +
                                    z * N + n] = 0.0;
                }
            }
        }
    }
}

void Registration::EM(const int max_EM_iterations, const double convergence_th,
                      const bool update_likelihood_parameters, const bool debug) {

    int iteration = 0;
    bool converged = false;

    double mll = 0;
    double log_prior_transformation = 0;
    double log_prior_likelihood = 0;
    double mlp = 0;
    double mlp_old = 1e100;
    double t = 0;

    omp_set_num_threads(threads);

    // Start the EM iterations
    while (iteration < max_EM_iterations && !converged) {
        std::cout << "Iteration: " << iteration + 1 << std::endl;

        // Reset mll counter for each thread
        std::fill(mll_t.begin(), mll_t.end(), 0.0);

        // Initialize parameters
        lkObj->initialize_parameters();

        // Initialize dream locations
        initialize_dream_locations();

        t = omp_get_wtime();  // Start timing

        // Estimate posterior and update dream locations at each voxel
        #pragma omp parallel for schedule(static,1) num_threads(threads)
        for (int x = 0; x < Nx_v; ++x) {
            for (int y = 0; y < Ny_v; ++y) {
                for (int z = 0; z < Nz_v; ++z) {
                    const int thread_id = omp_get_thread_num();
                    if (log_space) {
                        estimate_posterior_and_dream_location_log(x, y, z, thread_id);
                    } else {
                        estimate_posterior_and_dream_location(x, y, z, thread_id);
                    }
                }
            }
        }

        // Re-initialize mll
        mll = 0;

        // Update likelihood parameters
        if (update_likelihood_parameters) {
            lkObj->update_parameters();
        }

        // Compute log prior likelihood
        log_prior_likelihood = lkObj->compute_cost();

        // Sum all the min-log likelihood from each thread (safe from race conditions)
        for (int thread_id = 0; thread_id < threads; ++thread_id) {
            mll += mll_t[thread_id];
        }

        t = omp_get_wtime() - t;  // End timing
        std::cout << "Time updating theta and dream locations: " << t << std::endl;

        // Constrain transformation (smoothing or affine/rigid transformation)
        t = omp_get_wtime();
        log_prior_transformation = transformationObj->constrain_transformation(dream_locations.get(),
                                                                               voxel_pos, ds.get(),
                                                                               final_locations.get(),
                                                                               node_indices.get());
        t = omp_get_wtime() - t;
        std::cout << "Time constraining transformation: " << t << std::endl;

        // Compute min log posterior
        mlp = mll + log_prior_transformation + log_prior_likelihood;
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Min Log-Likelihood: " << mll << std::endl;
        std::cout << "Min Log-Prior likelihood: " << log_prior_likelihood << std::endl;
        std::cout << "Min Log-Prior transformation: " << log_prior_transformation << std::endl;
        std::cout << "Min Log-Posterior: " << mlp << std::endl;
        std::cout << "Min Log-Posterior per nonzero voxel: " << mlp / non_zero_voxels << std::endl;

        if (mlp - mlp_old > 1e-8) {  // Avoid numerical error
            std::cout << "Error, mlp increased by: " << mlp - mlp_old << std::endl;
        }

        if ((mlp_old - mlp) / non_zero_voxels < convergence_th) {
            std::cout << "EM converged!" << std::endl;
            converged = true;
        }

        mlp_old = mlp;
        ++iteration;
    }

}

void Registration::estimate_posterior_and_dream_location(
    const int x, const int y, const int z, const int thread_id)
{
    const int XX = splineObj->neighbours;
    const int YY = splineObj->neighbours;
    const int ZZ = is2D ? 1 : splineObj->neighbours;
    const int YYZZ = YY * ZZ;

    std::vector<double> posteriors(XX * YY * ZZ);


    // Compute voxel pos index
    const int voxel_pos_index = x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N;
    // Compute voxel index
    const int voxel_index = x * (Ny_v * Nz_v) + y * (Nz_v) + z;
    // Compute spline interpolation for neighboring voxels
    const std::array<double, 4> x_direction = splineObj->spline(ds[voxel_pos_index + 0]);
    const std::array<double, 4> y_direction = splineObj->spline(ds[voxel_pos_index + 1]);
    const std::array<double, 4> z_direction = is2D ?
        std::array<double, 4>{1.0, 1.0, 1.0, 1.0} :
        splineObj->spline(ds[voxel_pos_index + 2]);

    // offsets
    const std::array<int, 4> off_x = splineObj->offx;
    const std::array<int, 4> off_y = splineObj->offy;
    const std::array<int, 4> off_z = is2D ? std::array<int, 4>{0, 0, 0, 0} : splineObj->offz;

    // Node indices
    const int idx_x = node_indices[voxel_pos_index + 0];
    const int idx_y = node_indices[voxel_pos_index + 1];
    const int idx_z = is2D ? 0 : node_indices[voxel_pos_index + 2];

    // Variables for normalization and posterior calculation
    double normalizer = 0.0;

    // Compute posterior probabilities
    for (int xx = 0; xx < XX; ++xx) {
        const int tmp_x = idx_x + off_x[xx];
        for (int yy = 0; yy < YY; ++yy) {
            const int tmp_y = idx_y + off_y[yy];
            const double xy_direction = x_direction[xx] * y_direction[yy];
            for (int zz = 0; zz < ZZ; ++zz) {
                const int tmp_z = idx_z + off_z[zz];

                const int posterior_index = (xx * YYZZ) + (yy * ZZ) + zz;
                const double posterior_value = xy_direction * z_direction[zz] *
                                               lkObj->compute_likelihood(voxel_index, tmp_x, tmp_y,
                                                                         tmp_z);

                posteriors[posterior_index] = posterior_value;
                normalizer += posterior_value;
            }
        }
    }

    // Normalize posteriors
    mll_t[thread_id] += - std::log(normalizer);

    // Sampling or updating dream locations
    if (sample) {
        std::uniform_real_distribution dis(0.0, 1.0);
        const double random_number = dis(thread_rngs[thread_id]);
        double cumulative_probability = 0.0;

        for (int xx = 0; xx < XX; ++xx) {
            const int tmp_x = idx_x + off_x[xx];
            for (int yy = 0; yy < YY; ++yy) {
                const int tmp_y = idx_y + off_y[yy];
                for (int zz = 0; zz < ZZ; ++zz) {
                    const int tmp_z = idx_z + off_z[zz];

                    const int posterior_index =  (xx * YYZZ) + (yy * ZZ) + zz;

                    posteriors[posterior_index] /= normalizer;

                    cumulative_probability += posteriors[posterior_index];

                    if (random_number <= cumulative_probability) {
                        lkObj->accumulate_info(thread_id, voxel_index, tmp_x, tmp_y, tmp_z, 1.0);

                        dream_locations[voxel_pos_index + 0] = static_cast<double>(tmp_x);
                        dream_locations[voxel_pos_index + 1] = static_cast<double>(tmp_y);
                        if (!is2D) dream_locations[voxel_pos_index + 2] = static_cast<double>(tmp_z);

                        return;
                    }
                }
            }
        }
    } else {
        for (int xx = 0; xx < XX; ++xx) {
            const int tmp_x = idx_x + off_x[xx];
            for (int yy = 0; yy < YY; ++yy) {
                const int tmp_y = idx_y + off_y[yy];
                for (int zz = 0; zz < ZZ; ++zz) {
                    const int tmp_z = idx_z + off_z[zz];

                    const int posterior_index = (xx * YYZZ) + (yy * ZZ) + zz;

                    posteriors[posterior_index] /= normalizer;

                    lkObj->accumulate_info(thread_id, voxel_index, tmp_x, tmp_y, tmp_z,
                                           posteriors[posterior_index]);

                    dream_locations[voxel_pos_index + 0] += posteriors[posterior_index] * static_cast<double>(tmp_x);
                    dream_locations[voxel_pos_index + 1] += posteriors[posterior_index] * static_cast<double>(tmp_y);
                    if (!is2D) dream_locations[voxel_pos_index + 2] += posteriors[posterior_index] * static_cast<double>(tmp_z);
                }
            }
        }
    }
}

void Registration::estimate_posterior_and_dream_location_log(
    const int x, const int y, const int z, const int thread_id)
{
    const int XX = splineObj->neighbours;
    const int YY = splineObj->neighbours;
    const int ZZ = is2D ? 1 : splineObj->neighbours;
    const int YYZZ = YY * ZZ;

    std::vector<double> posteriors(XX * YY * ZZ);

    // Compute indices
    const int voxel_pos_index = x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N;
    const int voxel_index = x * (Ny_v * Nz_v) + y * (Nz_v) + z;

    // Compute log spline interpolation
    const std::array<double, 4> log_x_direction = splineObj->log_spline(ds[voxel_pos_index + 0]);
    const std::array<double, 4> log_y_direction = splineObj->log_spline(ds[voxel_pos_index + 1]);
    const std::array<double, 4> log_z_direction = is2D ?
        std::array<double, 4>{0.0, 0.0, 0.0, 0.0} :  // log(1.0) = 0.0
        splineObj->log_spline(ds[voxel_pos_index + 2]);

    // offsets remain the same
    const std::array<int, 4> off_x = splineObj->offx;
    const std::array<int, 4> off_y = splineObj->offy;
    const std::array<int, 4> off_z = is2D ? std::array<int, 4>{0, 0, 0, 0} : splineObj->offz;

    // Node indices
    const int idx_x = node_indices[voxel_pos_index + 0];
    const int idx_y = node_indices[voxel_pos_index + 1];
    const int idx_z = is2D ? 0 : node_indices[voxel_pos_index + 2];

    // For log space calculations
    double max_log_posterior = -std::numeric_limits<double>::infinity();

    // Compute log posteriors
    for (int xx = 0; xx < XX; ++xx) {
        const int tmp_x = idx_x + off_x[xx];
        for (int yy = 0; yy < YY; ++yy) {
            const int tmp_y = idx_y + off_y[yy];
            const double log_xy_direction = log_x_direction[xx] + log_y_direction[yy];
            for (int zz = 0; zz < ZZ; ++zz) {
                const int tmp_z = idx_z + off_z[zz];

                const int posterior_index = (xx * YYZZ) + (yy * ZZ) + zz;

                // Sum logs instead of multiplying values
                const double log_posterior = log_xy_direction + log_z_direction[zz] +
                    lkObj->compute_log_likelihood(voxel_index, tmp_x, tmp_y, tmp_z);

                posteriors[posterior_index] = log_posterior;
                max_log_posterior = std::max(max_log_posterior, log_posterior);
            }
        }
    }

    // Compute normalizer in log space using the log-sum-exp trick
    double normalizer = 0.0;
    for (int i = 0; i < XX * YY * ZZ; ++i) {
        normalizer += std::exp(posteriors[i] - max_log_posterior);
    }
    const double log_normalizer = std::log(normalizer) + max_log_posterior;

    // Update mll (marginal log likelihood)
    mll_t[thread_id] += -log_normalizer;

    // Sampling or updating dream locations
    if (sample) {
        std::uniform_real_distribution dis(0.0, 1.0);
        const double random_number = dis(thread_rngs[thread_id]);
        double cumulative_probability = 0.0;

        for (int xx = 0; xx < XX; ++xx) {
            const int tmp_x = idx_x + off_x[xx];
            for (int yy = 0; yy < YY; ++yy) {
                const int tmp_y = idx_y + off_y[yy];
                for (int zz = 0; zz < ZZ; ++zz) {
                    const int tmp_z = idx_z + off_z[zz];

                    const int posterior_index = (xx * YYZZ) + (yy * ZZ) + zz;

                    // Convert from log space to probability
                    const double prob = std::exp(posteriors[posterior_index] - log_normalizer);
                    posteriors[posterior_index] = prob;  // Store normalized probability

                    cumulative_probability += prob;

                    if (random_number <= cumulative_probability) {
                        lkObj->accumulate_info(thread_id, voxel_index, tmp_x, tmp_y, tmp_z, 1.0);

                        dream_locations[voxel_pos_index + 0] = static_cast<double>(tmp_x);
                        dream_locations[voxel_pos_index + 1] = static_cast<double>(tmp_y);
                        if (!is2D) dream_locations[voxel_pos_index + 2] = static_cast<double>(tmp_z);

                        return;
                    }
                }
            }
        }
    } else {
        for (int xx = 0; xx < XX; ++xx) {
            const int tmp_x = idx_x + off_x[xx];
            for (int yy = 0; yy < YY; ++yy) {
                const int tmp_y = idx_y + off_y[yy];
                for (int zz = 0; zz < ZZ; ++zz) {
                    const int tmp_z = idx_z + off_z[zz];

                    const int posterior_index = (xx * YYZZ) + (yy * ZZ) + zz;

                    // Convert from log space to probability
                    const double prob = std::exp(posteriors[posterior_index] - log_normalizer);
                    posteriors[posterior_index] = prob;  // Store normalized probability

                    lkObj->accumulate_info(thread_id, voxel_index, tmp_x, tmp_y, tmp_z, prob);

                    dream_locations[voxel_pos_index + 0] += prob * static_cast<double>(tmp_x);
                    dream_locations[voxel_pos_index + 1] += prob * static_cast<double>(tmp_y);
                    if (!is2D) dream_locations[voxel_pos_index + 2] += prob * static_cast<double>(tmp_z);
                }
            }
        }
    }
}

void Registration::setMILikelihood(double alpha,
                                   const double* theta,
                                   int K,
                                   int L,
                                   int threads,
                                   const int* binned_nodes,
                                   const int* binned_voxels) {
    this->lkObj = std::make_unique<MILikelihood>(alpha, theta, threads, binned_nodes, binned_voxels, K, L,
                                                 Nx_n, Ny_n, Nz_n, Nx_v, Ny_v, Nz_v);
}

void Registration::setMILikelihoodFilter(double alpha, const double *theta, int threads,
                                         const int *binned_nodes, const int *binned_voxels, int K, int L, int M) {
    this->lkObj = std::make_unique<MILikelihoodFilter>(alpha, theta, threads, binned_nodes, binned_voxels,
                                                       K, L, M, Nx_n, Ny_n, Nz_n, Nx_v, Ny_v, Nz_v);
}

void Registration::setSSDLikelihood(int bins, int threads, double sigmaSq, int shiftScale, double scale,
                                    double shift, const int *binned_nodes, const int *binned_voxels,
                                    double alpha_0, double beta_0) {
    this->lkObj = std::make_unique<SSDLikelihood>(bins, threads, sigmaSq, shiftScale, scale, shift,
                                                  binned_nodes, binned_voxels, alpha_0, beta_0,
                                                  Nx_n, Ny_n, Nz_n, Nx_v, Ny_v, Nz_v);
    log_space = true;
}

void Registration::setSSDLikelihoodFilter(const double *coeff, int threads, const double *sigma,
                                          const double *features_nodes, const double *features_voxels,
                                          int reg_across_features, double alpha_0, double beta_0,
                                          int M, int F) {
    this->lkObj = std::make_unique<SSDLikelihoodFilter>(coeff, threads, sigma, features_nodes, features_voxels,
        reg_across_features, alpha_0, beta_0, M, F,Nx_n, Ny_n, Nz_n, Nx_v, Ny_v, Nz_v);
    log_space = true;
}



void Registration::setNonLinearTransformation(double gamma_x, double gamma_y, double gamma_z,
                                              const double rho_x, const double rho_y, double rho_z,
                                              const double eta_x, const double eta_y, double eta_z,
                                              const double* D) {
    this->transformationObj = std::make_unique<NonLinearTransformation>(gamma_x, gamma_y, gamma_z,
                                                                        rho_x, rho_y, rho_z,
                                                                        eta_x, eta_y, eta_z,
                                                                        D,
                                                                        Nx_v, Ny_v, Nz_v, threads,
                                                                        splineObj->sigma_spline, splineObj->offset);
}

void Registration::setAffineTransformation(const double* A, const double* t) {
    this->transformationObj = std::make_unique<AffineTransformation>(A, t, Nx_v, Ny_v, Nz_v, threads,
                                                                     splineObj->sigma_spline, splineObj->offset);
}

void Registration::setRigidTransformation(const double* R, const double* t) {
    this->transformationObj = std::make_unique<RigidTransformation>(Nx_v, Ny_v, Nz_v, threads,
                                                                    splineObj->sigma_spline, splineObj->offset,
                                                                    R, t, voxel_pos);
}

std::pair<std::vector<double>, std::vector<double>>  Registration::sampler(const int seed, const int N_b, const int N_s,
                                                    const int sample_gamma, const int sample_gamma_every,
                                                    const int sample_posteriors,
                                                    const int sample_likelihood_parameters,
                                                    const int sample_transformation_parameters) {

    sample = true;

    // Set RNG with new seed
    this->seed = seed;
    for(int i = 0; i < threads; ++i) {
        thread_rngs[i].seed(seed + i);  // Give each thread a different but deterministic seed
    }
    rng.seed(this->seed);
    lkObj->set_generator(rng);
    transformationObj->set_generator(rng);

    // Memory for averages and standard deviations
    auto avg_locations = std::make_unique<double[]>(Nx_v * Ny_v * Nz_v * N);
    std::fill_n(avg_locations.get(), Nx_v * Ny_v * Nz_v * N, 0.0);
    auto cov_d = std::make_unique<double[]>(Nx_v * Ny_v * Nz_v * N * N);
    std::fill_n(cov_d.get(), Nx_v * Ny_v * Nz_v * N * N, 0.0);

    std::vector out(Nx_v * Ny_v * Nz_v * N, 0.0);
    std::vector out_2(Nx_v * Ny_v * Nz_v * N * N, 0.0);

    const double inverse_N_s = 1.0 / N_s;


    //
    for (int sweep = 0; sweep < N_b + N_s; ++sweep) {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Sweep: " << sweep + 1 << std::endl;

        double t = omp_get_wtime();

        if (sample_posteriors) {
            lkObj->initialize_parameters();
            initialize_dream_locations();
        }

        // Estimate posterior and update dream location
        if (sample_posteriors) {
        #pragma omp parallel for schedule(static,1) num_threads(threads)
            for (int x = 0; x < Nx_v; ++x) {
                for (int y = 0; y < Ny_v; ++y) {
                    for (int z = 0; z < Nz_v; ++z) {
                        const int thread_id = omp_get_thread_num();
                        if (log_space) {
                            estimate_posterior_and_dream_location_log(x, y, z, thread_id);
                        } else {
                            estimate_posterior_and_dream_location(x, y, z, thread_id);
                        }
                    }
                }
            }
        }

        // Sample likelihood parameters if needed
        if (sample_likelihood_parameters) {
            lkObj->sample_parameters();
        }

        t = omp_get_wtime() - t;
        std::cout << "Time sampling theta and dream locations: " << t << std::endl;

        t = omp_get_wtime();
        // Sample deformation parameters if needed
        if (sample_transformation_parameters) {
            const bool sample_gamma_this_sweep = sample_gamma && (sweep % sample_gamma_every == 0);
            transformationObj->sample_transformation(dream_locations.get(), voxel_pos, ds.get(),
                                                     final_locations.get(),
                                                     node_indices.get(), sample_gamma_this_sweep);
        }

        t = omp_get_wtime() - t;
        std::cout << "Time sampling deformations: " << t << std::endl;

        if (sweep >= N_b) {
            const double n_new = static_cast<double>(sweep - N_b) + 1.0;
            const double inv_n = 1.0 / n_new;

            for (int x = 0; x < Nx_v; ++x) {
                for (int y = 0; y < Ny_v; ++y) {
                    for (int z = 0; z < Nz_v; ++z) {
                        const int base = x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N;
                        const int vox  = x * (Ny_v * Nz_v) + y * Nz_v + z;

                        double delta1[3], delta2[3], val[3];

                        for (int n = 0; n < N; ++n) {
                            val[n] = final_locations[base + n];
                            const double mean_old = avg_locations[base + n];
                            delta1[n] = val[n] - mean_old;
                            const double mean_new = mean_old + delta1[n] * inv_n;
                            avg_locations[base + n] = mean_new;
                            delta2[n] = val[n] - mean_new;
                        }

                        // M2 accumulator, N x N row-major per voxel
                        for (int r = 0; r < N; ++r) {
                            for (int c = 0; c < N; ++c) {
                                cov_d[vox * (N * N) + r * N + c] += delta1[r] * delta2[c];
                            }
                        }
                    }
                }
            }
        }

    }

    // mean
    for (int i = 0; i < Nx_v * Ny_v * Nz_v * N; ++i) {
        out[i] = avg_locations[i];
    }

    // covariance = M2 / (N_s - 1)
    const double inv_Ns_1 = 1.0 / (N_s - 1);
    for (int i = 0; i < Nx_v * Ny_v * Nz_v * N * N; ++i) {
        out_2[i] = cov_d[i] * inv_Ns_1;
    }

    std::cout << "Done sampling" << std::endl;

    return std::make_pair(std::move(out), std::move(out_2));

}


// Getter to return final_locations
std::vector<double> Registration::get_final_locations() const {
    // Create a vector view of the unique_ptr data
    return std::vector(final_locations.get(), final_locations.get() + Nx_v * Ny_v * Nz_v * N);
}

// Getter to return dream_locations
std::vector<double> Registration::get_dream_locations() const {
    // Create a vector view of the unique_ptr data
    return std::vector(dream_locations.get(), dream_locations.get() + Nx_v * Ny_v * Nz_v * N);
}

// Getter to return the shape of the final_locations array
std::vector<int> Registration::get_shape() const {
    return { Nx_v, Ny_v, Nz_v, N };
}

void Registration::set_dream_locations(const double * dream_locations) {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    this->dream_locations[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n] =
                        dream_locations[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n];
                }
            }
        }
    }
}

std::vector<double> Registration::get_likelihood_parameters() const {
    return lkObj->get_parameters();
}
