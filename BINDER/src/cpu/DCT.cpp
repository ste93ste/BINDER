//
// Created by stce on 02/01/25.
//
#include "DCT.h"
#include <fftw3.h>
#include <cmath>

// *i refers to input and *o refers to output
// This separation is nice to have for up and down sampling of the deformation field
DCT::DCT(const int N_x_i, const int N_y_i, const int N_z_i,
         const int N_x_o, const int N_y_o, const int N_z_o,
         const int threads, const int fast_search){
        // Check if dimensions match the saved wisdom
        const bool dims_match = (N_x_i == saved_dims[0] && N_y_i == saved_dims[1] &&
                                 N_z_i == saved_dims[2] && N_x_o == saved_dims[3] &&
                                 N_y_o == saved_dims[4] && N_z_o == saved_dims[5]);
        if (!dims_match) {
            // If dimensions don't match, reset wisdom and update filename
            wisdom_loaded = false;
            if (!wisdom_file.empty()) {
                wisdom_file = getWisdomFilename(wisdom_file.substr(0, wisdom_file.find('_')),
                                              N_x_i, N_y_i, N_z_i,
                                              N_x_o, N_y_o, N_z_o);
            }
        }
    // if fast_search it means that we are not reusing this object often, so no need to do FFTW MEASURE
    int search = FFTW_MEASURE;
    if (fast_search == 1 && !wisdom_loaded) {
        search = FFTW_ESTIMATE;
    }
    this->N_x_i = N_x_i;
    this->N_y_i = N_y_i;
    this->N_z_i = N_z_i;
    this->N_x_o = N_x_o;
    this->N_y_o = N_y_o;
    this->N_z_o = N_z_o;
    const int num_voxels_i = N_x_i * N_y_i * N_z_i;
    const int num_voxels_o = N_x_o * N_y_o * N_z_o;
    this->data_dct_in = new double[num_voxels_i];
    this->data_dct_out = new double[num_voxels_i];
    this->data_idct_in = new double[num_voxels_o];
    this->data_idct_out = new double[num_voxels_o];
    this->threads = threads;

    fftw_init_threads();
    fftw_plan_with_nthreads(threads);
    if (this->N_z_i == 1) {
        // Create 2D plans
        this->plan_forward = fftw_plan_r2r_2d(
            N_x_i, N_y_i, this->data_dct_in, this->data_dct_out,
            FFTW_REDFT10, FFTW_REDFT10, search);
        this->plan_backward = fftw_plan_r2r_2d(
            N_x_o, N_y_o, this->data_idct_in, this->data_idct_out,
            FFTW_REDFT01, FFTW_REDFT01, search);
    } else {
        // Create 3D plans
        this->plan_forward = fftw_plan_r2r_3d(
            N_x_i, N_y_i, N_z_i, this->data_dct_in, this->data_dct_out,
            FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, search);
        this->plan_backward = fftw_plan_r2r_3d(
            N_x_o, N_y_o, N_z_o, this->data_idct_in, this->data_idct_out,
            FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, search);
    }
    // Save wisdom after creating plans, but only if dimensions changed or wisdom wasn't loaded
    if (!wisdom_loaded && !wisdom_file.empty()) {
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        // Update saved dimensions
        saved_dims[0] = N_x_i; saved_dims[1] = N_y_i; saved_dims[2] = N_z_i;
        saved_dims[3] = N_x_o; saved_dims[4] = N_y_o; saved_dims[5] = N_z_o;
        wisdom_loaded = true;
    }
}

DCT::~DCT(){
    fftw_destroy_plan(this->plan_forward);
    fftw_destroy_plan(this->plan_backward);
    delete[] this->data_dct_in;
    delete[] this->data_dct_out;
    delete[] this->data_idct_in;
    delete[] this->data_idct_out;
}

void DCT::dct() const {
    fftw_execute(this->plan_forward);

    // Apply orthogonal normalization
    if (N_z_i == 1) {
        // 2D case
        for (int x = 0; x < N_x_i; x++) {
            for (int y = 0; y < N_y_i; y++) {
                const int index = x * N_y_i + y;

                double factor = 1.0;
                if (x == 0) factor /= sqrt(2.0);
                if (y == 0) factor /= sqrt(2.0);

                const double norm = sqrt(1.0 / (2.0 * N_x_i)) * sqrt(1.0 / (2.0 * N_y_i));
                data_dct_out[index] *= norm * factor;
            }
        }
    } else {
        // 3D case
        for (int x = 0; x < N_x_i; x++) {
            for (int y = 0; y < N_y_i; y++) {
                for (int z = 0; z < N_z_i; z++) {
                    const int index = x * N_y_i * N_z_i + y * N_z_i + z;

                    double factor = 1.0;
                    if (x == 0) factor /= sqrt(2.0);
                    if (y == 0) factor /= sqrt(2.0);
                    if (z == 0) factor /= sqrt(2.0);

                    const double norm = sqrt(1.0 / (2.0 * N_x_i)) * sqrt(1.0 / (2.0 * N_y_i)) *
                                        sqrt(1.0 / (2.0 * N_z_i));
                    data_dct_out[index] *= norm * factor;
                }
            }
        }
    }
}

void DCT::idct() const {
    // Apply orthogonal normalization before inverse transform
    if (N_z_o == 1) {
        // 2D case
        for (int x = 0; x < N_x_o; x++) {
            for (int y = 0; y < N_y_o; y++) {
                const int index = x * N_y_o + y;

                double factor = 1.0;
                if (x == 0) factor /= sqrt(2.0);
                if (y == 0) factor /= sqrt(2.0);

                const double norm = sqrt(1.0 / (2.0 * N_x_o)) * sqrt(1.0 / (2.0 * N_y_o));
                data_idct_in[index] /= (norm * factor);
            }
        }
    } else {
        // 3D case
        for (int x = 0; x < N_x_o; x++) {
            for (int y = 0; y < N_y_o; y++) {
                for (int z = 0; z < N_z_o; z++) {
                    const int index = x * N_y_o * N_z_o + y * N_z_o + z;

                    double factor = 1.0;
                    if (x == 0) factor /= sqrt(2.0);
                    if (y == 0) factor /= sqrt(2.0);
                    if (z == 0) factor /= sqrt(2.0);

                    const double norm = sqrt(1.0 / (2.0 * N_x_o)) * sqrt(1.0 / (2.0 * N_y_o)) *
                                        sqrt(1.0 / (2.0 * N_z_o));
                    data_idct_in[index] /= (norm * factor);
                }
            }
        }
    }

    fftw_execute(this->plan_backward);

    // Apply final scaling to match CUDA implementation
    const int num_voxels = N_x_o * N_y_o * N_z_o;
    if (N_z_o == 1) {
        // 2D case: divide by 4.0 (2²) * Nx * Ny
        const double final_scale = 4.0 * N_x_o * N_y_o;
        for (int i = 0; i < num_voxels; i++) {
            data_idct_out[i] /= final_scale;
        }
    } else {
        // 3D case: divide by 8.0 (2³) * Nx * Ny * Nz
        const double final_scale = 8.0 * N_x_o * N_y_o * N_z_o;
        for (int i = 0; i < num_voxels; i++) {
            data_idct_out[i] /= final_scale;
        }
    }
}

// Initialize static members
bool DCT::wisdom_loaded = false;
std::string DCT::wisdom_file = "";
int DCT::saved_dims[6] = {0, 0, 0, 0, 0, 0};