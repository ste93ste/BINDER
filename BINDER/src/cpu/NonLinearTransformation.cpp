//
// Created by stce on 01/01/25.
//

#include "NonLinearTransformation.h"
#include "DCT.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <utility>
#include <memory>

// Constructor
NonLinearTransformation::NonLinearTransformation(double gamma_x, double gamma_y, double gamma_z,
                                                 const double rho_x, const double rho_y, const double rho_z,
                                                 const double eta_x, const double eta_y, const double eta_z,
                                                 const double* D,
                                                 int Nx_v, int Ny_v, int Nz_v, int threads, const double sigma_spline,
                                                 const double spline_offset)
: threads(threads), sigma_spline(sigma_spline),
  spline_offset(spline_offset), Nx_v(Nx_v), Ny_v(Ny_v), Nz_v(Nz_v), N(Nz_v == 1 ? 2 : 3), D(D){
    if (Nz_v == 1){
        is2D=1;
    } else {
        is2D=0;
    }

    gamma[0] = gamma_x;
    gamma[1] = gamma_y;
    gamma[2] = gamma_z;
    rho[0] = rho_x;
    rho[1] = rho_y;
    rho[2] = rho_z;
    eta[0] = eta_x;
    eta[1] = eta_y;
    eta[2] = eta_z;

    num_voxels = Nx_v * Ny_v * Nz_v;

    alpha_0 = 1.0;
    beta_0 = 10000.0;
    nu = gamma[0] / (rho[0] * rho[1] * rho[2] * eta[0] * eta[0] * num_voxels);

    sample = 0;

    std::cout << "Setting up DCTs objects..." << std::endl;
    DCT::initializeWisdom("fftw", Nx_v, Ny_v, Nz_v, Nx_v, Ny_v, Nz_v);
    dcts.resize(N);
    for(int n = 0; n < N; n++) {
        dcts[n] = std::make_unique<DCT>(Nx_v, Ny_v, Nz_v, Nx_v, Ny_v, Nz_v, threads);
    }
    std::cout << "Done!" << std::endl;

    // Removed normalization_constant calculation
}

double NonLinearTransformation::constrain_transformation(const double* dream_locations,
                                                         const double* voxel_pos,
                                                         double* ds,
                                                         double* final_locations,
                                                         int* node_indices) {

    double log_prior_deformation = 0.0;
    #pragma omp parallel for num_threads(N) reduction(+:log_prior_deformation)
    for (int n=0; n < N; n++) {
        // Fill in data in dct object (dream_locations - voxel_pos)
        fill_in_data(n, dream_locations, voxel_pos);
        // Smooth and track cost
        log_prior_deformation += smooth(n);
        // Update final locations
        update_final_locations(n, voxel_pos, node_indices, ds, final_locations);
    }

    return log_prior_deformation;
}


void NonLinearTransformation::sample_transformation(const double* dream_locations,
                                                    const double* voxel_pos,
                                                    double* ds,
                                                    double* final_locations,
                                                    int* node_indices,
                                                    const int sample_gamma) {
    double total_raw_reg = 0.0;
    // Set sample flag
    sample = true;
    this->sample_gamma = sample_gamma;
    recompute_tmp_sigma = true;

    // Precompute samples from Gaussian distribution
    precompute_sample_from_gaussian_distribution();

    #pragma omp parallel for num_threads(N)
    for (int n = 0; n < N; ++n) {

        // Fill-in data for smoothing
        fill_in_data(n, dream_locations, voxel_pos);

        // Smooth dream locations
        smooth(n);

        // Compute regularization only if needed - uses coefficients from data_dct_out
        if (sample_gamma) {
            const double raw_reg_n = compute_raw_regularization(n);
            #pragma omp atomic
            total_raw_reg += raw_reg_n;
        }

        // Update final locations
        update_final_locations(n, voxel_pos, node_indices, ds, final_locations);

    }

    // Sample gamma parameter if required
    if (sample_gamma) {
        const double alpha_post = alpha_0 + 0.5 * static_cast<double>(N * num_voxels);
        const double beta_post = beta_0 + 0.5 * static_cast<double>(num_voxels) * total_raw_reg;

        std::gamma_distribution<double> gamma_dist(alpha_post, 1.0 / beta_post);
        nu = gamma_dist(gen);
        updateGammasFromNu();
        recompute_tmp_sigma = true;
    }

}

void NonLinearTransformation::fill_in_data(const int n, const double* dream_locations, const double* voxel_pos) const {
    for(int x=0; x < Nx_v; x++){
        for(int y=0; y < Ny_v; y++){
            for(int z=0; z < Nz_v; z++){
                const int index = x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + n;
                dcts[n]->data_dct_in[x * Ny_v * Nz_v + y * Nz_v + z] = dream_locations[index] - voxel_pos[index];
            }
        }
    }
}


void NonLinearTransformation::updateGammasFromNu() {
    const double rho_product = rho[0] * rho[1] * rho[2];
    gamma[0] = nu * rho_product * eta[0] * eta[0] * static_cast<double>(num_voxels);
    gamma[1] = nu * rho_product * eta[1] * eta[1] * static_cast<double>(num_voxels);
    gamma[2] = nu * rho_product * eta[2] * eta[2] * static_cast<double>(num_voxels);
    std::cout << "Gamma: x=" << gamma[0] << ", y=" << gamma[1] << ", z=" << gamma[2]
              << " (nu=" << nu << ")" << std::endl;
    recompute_tmp_sigma = true;
}


double NonLinearTransformation::compute_raw_regularization(const int n) const {
    double raw_reg = 0.0;
    const double rho_product = rho[0] * rho[1] * rho[2];

    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int i = x * Ny_v * Nz_v + y * Nz_v + z;

                // Get the smoothed DCT coefficient (before sampling noise)
                double coeff = dcts[n]->data_dct_out[i];

                // Compute contribution: η²_d * ρ * D[i] * coeff_d²
                const double contrib = eta[n] * eta[n] * rho_product * D[i] * coeff * coeff;
                raw_reg += contrib;
            }
        }
    }

    return raw_reg;
}

void NonLinearTransformation::update_final_locations(const int n, const double* voxel_pos,
                                                     int* node_indices, double* ds,
                                                     double* final_locations) const {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int index = x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + n;
                const double tmp = dcts[n]->data_idct_out[x * Ny_v * Nz_v + y * Nz_v + z] + voxel_pos[index];
                // Compute indices and distances
                node_indices[index] = static_cast<int>(std::floor(tmp + spline_offset));
                ds[index] = tmp - std::floor(tmp + spline_offset);
                final_locations[index] = tmp;
            }
        }
    }
}

double NonLinearTransformation::smooth(const int n) const {
    double cost = 0.0;

    // Perform DCT
    dcts[n]->dct();

    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                const int i = x * Ny_v * Nz_v + y * Nz_v + z;
                dcts[n]->data_idct_in[i] = (dcts[n]->data_dct_out[i]) *
                                 (1.0 / (1.0 + gamma[n] * sigma_spline * sigma_spline * D[i]));

                if (sample) {
                    dcts[n]->data_idct_in[i] += random_std_DCT[n * num_voxels + i];
                }

                if (sample && sample_gamma) {
                    // Store coefficients in dct_out for later usage for sampling gamma
                    dcts[n]->data_dct_out[i] = dcts[n]->data_idct_in[i];
                }

                // Accumulate cost
                cost += 0.5 * gamma[n] *
                        std::pow(std::sqrt(D[i]) * std::abs(dcts[n]->data_idct_in[i]), 2);
            }
        }
    }

    // Perform IDCT
    dcts[n]->idct();

    return cost;
}


std::pair<double, double *> NonLinearTransformation::smooth_deformation(const double *deformations) {

    // Create empty arrays for the rest
    double log_prior_deformation = 0;
    auto * dream_locations = new double[Nx_v * Ny_v * Nz_v * N];
    auto * voxel_pos = new double[Nx_v * Ny_v * Nz_v * N];
    auto * ds = new double[Nx_v * Ny_v * Nz_v * N];
    auto * final_locations = new double[Nx_v * Ny_v * Nz_v * N];
    auto * node_indices = new int[Nx_v * Ny_v * Nz_v * N];

    std::fill_n(final_locations, Nx_v * Ny_v * Nz_v * N, 0.0);
    std::fill_n(ds, Nx_v * Ny_v * Nz_v * N, 0.0);
    std::fill_n(node_indices, Nx_v * Ny_v * Nz_v * N, 0);
    std::fill_n(voxel_pos, Nx_v * Ny_v * Nz_v * N, 0);

    // Copy deformation field to dream location
    // Set rest to zero, as we are not interested in it
    for (int x=0; x < Nx_v; ++x) {
        for (int y=0; y < Ny_v; ++y) {
            for (int z=0; z < Nz_v; ++z) {
                for (int n=0; n < N; ++n) {
                    dream_locations[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + n] =
                        deformations[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + n];
                }
            }
        }
    }

    // Call the smoothing method integrated in the whole EM scheme
    log_prior_deformation = constrain_transformation(dream_locations, voxel_pos, ds, final_locations, node_indices);

    auto * new_deformations = new double[Nx_v * Ny_v * Nz_v * N];

    for (int x=0; x < Nx_v; ++x) {
        for (int y=0; y < Ny_v; ++y) {
            for (int z=0; z < Nz_v; ++z) {
                for (int n=0; n < N; ++n) {
                    new_deformations[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + n] =
                        final_locations[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + n];
                }
            }
        }
    }

    delete[] dream_locations;
    delete[] voxel_pos;
    delete[] ds;
    delete[] final_locations;
    delete[] node_indices;

    return std::make_pair(log_prior_deformation, new_deformations);

}

double * NonLinearTransformation::sampleDeformationField(const double* deformationField, const int number_of_samples) {

    // Declare output array with dimensions [number_of_samples, Nx, Ny, Nz, N]
    auto * out = new double[number_of_samples * Nx_v * Ny_v * Nz_v * N];

    //
    auto* dream_locations = new double[Nx_v * Ny_v * Nz_v * N];
    auto* voxel_pos = new double[Nx_v * Ny_v * Nz_v * N];
    auto* ds = new double[Nx_v * Ny_v * Nz_v * N];
    auto* final_locations = new double[Nx_v * Ny_v * Nz_v * N];
    auto* node_indices = new int[Nx_v * Ny_v * Nz_v * N];

    std::fill_n(dream_locations, Nx_v * Ny_v * Nz_v * N, 0.0);
    std::fill_n(final_locations, Nx_v * Ny_v * Nz_v * N, 0.0);
    std::fill_n(ds, Nx_v * Ny_v * Nz_v * N, 0.0);
    std::fill_n(node_indices, Nx_v * Ny_v * Nz_v * N, 0);
    std::fill_n(voxel_pos, Nx_v * Ny_v * Nz_v * N, 0);

    // Set sample flag
    sample = true;
    recompute_tmp_sigma = true;

    // Copy deformation field to dream location malloc allocation
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    dream_locations[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n] =
                        deformationField[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n];
                }
            }
        }
    }

    // Iterate for the number of samples
    for (int s = 0; s < number_of_samples; ++s) {
        // Sample transformation
        sample_transformation(dream_locations, voxel_pos, ds, final_locations, node_indices, 0);

        // Copy the final locations into the output
        int it = 0;
        for (int x = 0; x < Nx_v; ++x) {
            for (int y = 0; y < Ny_v; ++y) {
                for (int z = 0; z < Nz_v; ++z) {
                    for (int n = 0; n < N; ++n) {
                        out[s * num_voxels * N + it] = final_locations[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n];
                        it++;
                    }
                }
            }
        }
    }

    // Free dynamically allocated memory
    delete[] dream_locations;
    delete[] voxel_pos;
    delete[] ds;
    delete[] final_locations;
    delete[] node_indices;

    return out;
}


void NonLinearTransformation::compute_fourier_tmp_sigma_DCT() {

    // Allocate memory for fourier_tmp_sigma_DCT if not already allocated
    if (fourier_tmp_sigma_DCT == nullptr) {
        fourier_tmp_sigma_DCT = std::make_unique<double[]>(N * num_voxels);
    }

    // Loop through all the samples (n) and voxels (x, y, z)
    for (int n = 0; n < N; ++n) {
        int it = 0;
        for (int x = 0; x < Nx_v; ++x) {
            for (int y = 0; y < Ny_v; ++y) {
                for (int z = 0; z < Nz_v; ++z) {
                    fourier_tmp_sigma_DCT[n * num_voxels + it] = std::sqrt(sigma_spline * sigma_spline /
                        (1.0 + gamma[n] * std::pow(sigma_spline, 2) * D[it]));
                    it++;
                }
            }
        }
    }
}

std::vector<double> NonLinearTransformation::up_sample(const double *deformation_field,
                                                       int N_x, int N_y, int N_z,
                                                       int N_x_o, int N_y_o, int N_z_o, int threads) {

    // Output shape
    int N = 3;
    if (N_z_o == 1) {
        N = 2;
    }
    // create internal DCT, use fast search as we do only N times DCT operations
    const auto dct = std::make_unique<DCT>(N_x, N_y, N_z, N_x_o, N_y_o, N_z_o, threads, 1);
    // Output vector
    std::vector<double> out(N_x_o * N_y_o * N_z_o * N);

    // Fill in data
    for (int n = 0; n < N; ++n) {
        for (int x = 0; x < N_x; ++x) {
            for (int y = 0; y < N_y; ++y) {
                for (int z = 0; z < N_z; ++z) {
                    dct->data_dct_in[x * (N_y * N_z) + y * (N_z) + z] =
                        deformation_field[x * (N_y * N_z * N) + y * (N_z * N) + z * N + n];
                }
            }
        }

        // Perform DCT
        dct->dct();

        // Fill IDCT with all zeros
        for (int x = 0; x < N_x_o; ++x) {
            for (int y = 0; y < N_y_o; ++y) {
                for (int z = 0; z < N_z_o; ++z) {
                    dct->data_idct_in[x * (N_y_o * N_z_o) + y * (N_z_o) + z] = 0.0;
                }
            }
        }

        // Now fill the upper quadrant with results of dct (copying coefficients)
        int it = 0;
        for (int x = 0; x < N_x; ++x) {
            for (int y = 0; y < N_y; ++y) {
                for (int z = 0; z < N_z; ++z) {
                    dct->data_idct_in[x * (N_y_o * N_z_o) + y * (N_z_o) + z] = dct->data_dct_out[it];
                    ++it;
                }
            }
        }

        // Perform IDCT
        dct->idct();

        // Copy up sampled field to output
        for (int x = 0; x < N_x_o; ++x) {
            for (int y = 0; y < N_y_o; ++y) {
                for (int z = 0; z < N_z_o; ++z) {
                    out[x * (N_y_o * N_z_o * N) + y * (N_z_o * N) + z * N + n] =
                        dct->data_idct_out[x * (N_y_o * N_z_o) + y * (N_z_o) + z];
                }
            }
        }

    }

    return out;
}


void NonLinearTransformation::precompute_sample_from_gaussian_distribution() {

    // Compute sigma tmp, if not already computed
    if (recompute_tmp_sigma) {
        compute_fourier_tmp_sigma_DCT();
        recompute_tmp_sigma = false;
    }

    // Precompute random samples from Gaussian distribution
    std::normal_distribution dist(0.0, 1.0);

    // Allocate memory for random_std_DCT
    random_std_DCT = std::make_unique<double[]>(Nx_v * Ny_v * Nz_v * N);

    // Loop through and compute the standard deviations
    for (int it = 0; it < N * num_voxels; ++it) {
        random_std_DCT[it] = dist(gen) * fourier_tmp_sigma_DCT[it];
    }
}

// Getter to return the shape of the final_locations array
std::vector<int> NonLinearTransformation::get_shape() const {
    return {Nx_v, Ny_v, Nz_v, N};
}

void NonLinearTransformation::set_generator(std::mt19937 &gen) {
    this->gen = gen;
}

NonLinearTransformation::~NonLinearTransformation() = default;