//
// Created by stce on 2/18/25.
//

#ifndef NONLINEARTRANSFORMATION_CUH
#define NONLINEARTRANSFORMATION_CUH
#include "DCT3D.cuh"
#include <vector>
#include <tuple>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <curand_kernel.h>
#include <random>

class NonLinearTransformation {
public:
    // Constructor
    NonLinearTransformation(float gamma_x, float gamma_y, float gamma_z,
                            float rho_x, float rho_y, float rho_z,
                            float eta_x, float eta_y, float eta_z,
                            const float *D, int Nx_v, int Ny_v, int Nz_v,
                            int threads, float sigma_spline, float spline_offset);

    // Destructor
    ~NonLinearTransformation();

    float updateDeformation(float* d_ptr, const float* dream_location_ptr, const float* x_ptr, bool sampling,
                            bool sampling_gamma);

    std::vector<int> get_shape() const;

    std::pair<float, float *> smoothDeformation(const float *deformations);
    float * sampleDeformationField(const float* deformationField, int number_of_samples);

    void compute_fourier_tmp_sigma_DCT();
    void updateGammasFromNu();

    std::mt19937 rng;
    cublasHandle_t cublas_handle{};
    float alpha_0;
    float beta_0;
    float nu;
    float gamma_x;
    float gamma_y;
    float gamma_z;
    float rho_x;
    float rho_y;
    float rho_z;
    float eta_x;
    float eta_y;
    float eta_z;
    float spline_offset;
    int num_voxels;
    int Nx_v, Ny_v, Nz_v;
    int N;
    DCT3D* dct3d{};
    const float sigma_spline;
    curandStatePhilox4_32_10_t* d_states = nullptr;
    bool recompute_sigma = true;
    thrust::device_vector<float> D;
    float* D_ptr;
    thrust::device_vector<float> filter_sigma; // sigma of the filter
    float* filter_sigma_ptr;
    thrust::device_vector<float> d_prior; //for storing deformation prior cost
    float* d_prior_ptr;
    thrust::device_vector<float> tmp_x;
    thrust::device_vector<float> tmp_y;
    thrust::device_vector<float> tmp_z;
    float *tmp_x_ptr;
    float *tmp_y_ptr;
    float *tmp_z_ptr;
};



#endif //NONLINEARTRANSFORMATION_CUH
