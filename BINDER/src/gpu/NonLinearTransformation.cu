//
// Created by stce on 2/18/25.
//

#include "NonLinearTransformation.cuh"
#include "DCT3D.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <curand_kernel.h>

__global__ void init_curand_philox(
    curandStatePhilox4_32_10_t *states,
    unsigned long long seed,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // sequence = i gives independent streams
        curand_init(seed, i, 0, &states[i]);
    }
}

// computing deltai
__global__ void compute_displacement(float *x_direction, float *y_direction, float *z_direction,
                                     const float *dream_location, const float *x,
                                     const int N0, const int N1, const int N2)
{
    const int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int i0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 < N0 && i1 < N1 && i2 < N2)
    {

        // C-order linear index calculation
        const int i = i0 * N1 * N2 + i1 * N2 + i2;

        // Compute displacement for each dimension
        // deltai = Σwij*yj-xi
        x_direction[i] = dream_location[i * 3 + 0] - x[i * 3 + 0];
        y_direction[i] = dream_location[i * 3 + 1] - x[i * 3 + 1];
        z_direction[i] = dream_location[i * 3 + 2] - x[i * 3 + 2];
    }
}

__global__ void compute_fourier_sigma(
    float *filter_sigma,
    const float *D,
    const float sigma_spline,
    const float gamma_x,
    const float gamma_y,
    const float gamma_z,
    const int Nx_v,
    const int Ny_v,
    const int Nz_v,
    const int N)
{
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx_v && y < Ny_v && z < Nz_v)
    {
        const int voxel_idx = (x * Ny_v * Nz_v) + (y * Nz_v) + z;

        for (int n = 0; n < N; ++n)
        {
            const int filter_idx = voxel_idx * N + n;

            float current_gamma;
            if (n == 0)
                current_gamma = gamma_x;
            else if (n == 1)
                current_gamma = gamma_y;
            else
                current_gamma = gamma_z;

            // Base calculation for sigma
            const float base_sigma = sigma_spline * sigma_spline /
                                     (1.0f + current_gamma * powf(sigma_spline, 2) * D[voxel_idx]);

            filter_sigma[filter_idx] = sqrtf(base_sigma);
        }
    }
}

// update deformation in DCT: if sampling=true sample from deformation otherwise optimize
__global__ void smooth_deformation_no_sampling(
    float *x_component, float *y_component, float *z_component,
    const float *__restrict__ D,
    float gamma_x, float gamma_y, float gamma_z,
    float sigma_spline,
    int N0, int N1, int N2,
    float *__restrict__ d_prior)
{
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i0 = blockIdx.z * blockDim.z + threadIdx.z;
    if (i0 >= N0 || i1 >= N1 || i2 >= N2)
        return;

    int i = (i0 * N1 + i1) * N2 + i2;

    float Dval = D[i];
    float ss2 = sigma_spline * sigma_spline;

    // precompute factors
    float fx = 1.0f / (1.0f + gamma_x * ss2 * Dval);
    float fy = 1.0f / (1.0f + gamma_y * ss2 * Dval);
    float fz = 1.0f / (1.0f + gamma_z * ss2 * Dval);

    float x = x_component[i] * fx;
    float y = y_component[i] * fy;
    float z = z_component[i] * fz;

    x_component[i] = x;
    y_component[i] = y;
    z_component[i] = z;

    // prior: -0.5 * D * (gamma_x x^2 + gamma_y y^2 + gamma_z z^2)
    d_prior[i] = -0.5f * Dval * (gamma_x * x * x + gamma_y * y * y + gamma_z * z * z);
}

__global__ void smooth_deformation_sampling(
    curandStatePhilox4_32_10_t *__restrict__ states,
    float *x_component, float *y_component, float *z_component,
    const float *__restrict__ D,
    const float *__restrict__ filter_sigma,
    float gamma_x, float gamma_y, float gamma_z,
    float sigma_spline,
    int N0, int N1, int N2,
    float *__restrict__ d_prior)
{
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    int i0 = blockIdx.z * blockDim.z + threadIdx.z;
    if (i0 >= N0 || i1 >= N1 || i2 >= N2)
        return;

    int i = (i0 * N1 + i1) * N2 + i2;
    int base = 3 * i;

    float Dval = D[i];
    float ss2 = sigma_spline * sigma_spline;

    float fx = 1.0f / (1.0f + gamma_x * ss2 * Dval);
    float fy = 1.0f / (1.0f + gamma_y * ss2 * Dval);
    float fz = 1.0f / (1.0f + gamma_z * ss2 * Dval);

    float x = x_component[i] * fx;
    float y = y_component[i] * fy;
    float z = z_component[i] * fz;

    // prior
    d_prior[i] = -0.5f * Dval * (gamma_x * x * x + gamma_y * y * y + gamma_z * z * z);

    // RNG: 3 normals. Use normal4 to reduce overhead.
    curandStatePhilox4_32_10_t local = states[i]; // <-- recommend 1 state per voxel
    float4 r = curand_normal4(&local);            // 4 normals
    states[i] = local;

    x += r.x * filter_sigma[base + 0];
    y += r.y * filter_sigma[base + 1];
    z += r.z * filter_sigma[base + 2];

    x_component[i] = x;
    y_component[i] = y;
    z_component[i] = z;
}

NonLinearTransformation::NonLinearTransformation(float gamma_x, float gamma_y, float gamma_z,
                                                 float rho_x, float rho_y, float rho_z,
                                                 float eta_x, float eta_y, float eta_z,
                                                 const float *D,
                                                 const int Nx_v, const int Ny_v, const int Nz_v,
                                                 int threads, const float sigma_spline, const float spline_offset) : 
                                                 gamma_x(gamma_x), gamma_y(gamma_y), gamma_z(gamma_z),
                                                 rho_x(rho_x), rho_y(rho_y), rho_z(rho_z),
                                                 eta_x(eta_x), eta_y(eta_y), eta_z(eta_z),
                                                 spline_offset(spline_offset), Nx_v(Nx_v), Ny_v(Ny_v),
                                                 Nz_v(Nz_v), sigma_spline(sigma_spline)
{

    // Initialize CUDA handles
    cublasCreate(&cublas_handle);

    N = Nz_v == 1 ? 2 : 3;
    num_voxels = Nx_v * Ny_v * Nz_v;

    // Retrieve nu from gamma, rho and eta
    nu = gamma_x / (rho_x * rho_y * rho_z * eta_x * eta_x * num_voxels);
    // Set up hyperparameters on dct coeffiecients (gamma distribution)
    alpha_0 = 1.0f;
    beta_0 = 10000.0f;

    this->D.resize(num_voxels);
    copy(D, D + num_voxels, this->D.begin());

    // Initialize the temporary vectors once in the constructor
    const int I = Nx_v * Ny_v * Nz_v;
    tmp_x.resize(I);
    tmp_y.resize(I);
    tmp_z.resize(I);

    // Store raw pointers for quick access
    tmp_x_ptr = raw_pointer_cast(tmp_x.data());
    tmp_y_ptr = raw_pointer_cast(tmp_y.data());
    tmp_z_ptr = raw_pointer_cast(tmp_z.data());

    filter_sigma.resize(num_voxels * N);
    d_prior.resize(num_voxels);
    D_ptr = raw_pointer_cast(this->D.data());
    filter_sigma_ptr = raw_pointer_cast(filter_sigma.data());
    d_prior_ptr = raw_pointer_cast(d_prior.data());
    cudaMalloc(&d_states,
               num_voxels * sizeof(curandStatePhilox4_32_10_t));
    int thrd = 256;
    int blocks = (num_voxels + thrd - 1) / thrd;
    init_curand_philox<<<blocks, thrd>>>(d_states, 1234, num_voxels);
    cudaGetLastError();
    rng.seed(1234);
    dct3d = new DCT3D(Nx_v, Ny_v, Nz_v);
}

void NonLinearTransformation::updateGammasFromNu()
{
    const float rho_product = rho_x * rho_y * rho_z;
    gamma_x = nu * rho_product * eta_x * eta_x * static_cast<float>(num_voxels);
    gamma_y = nu * rho_product * eta_y * eta_y * static_cast<float>(num_voxels);
    gamma_z = nu * rho_product * eta_z * eta_z * static_cast<float>(num_voxels);
    printf("Gamma: x=%.6f, y=%.6f, z=%.6f (nu=%.10f)\n", gamma_x, gamma_y, gamma_z, nu);
    recompute_sigma = true;
}

NonLinearTransformation::~NonLinearTransformation()
{
    // Free CUDA-allocated memory
    if (d_states)
    {
        cudaFree(d_states);
        d_states = nullptr;
    }

    // Destroy CUBLAS handle
    if (cublas_handle)
    {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }

    // Delete the DCT3D object
    if (dct3d)
    {
        delete dct3d;
        dct3d = nullptr;
    }

    // Clear Thrust device vectors to release GPU memory
    D.clear();
    D.shrink_to_fit();

    filter_sigma.clear();
    filter_sigma.shrink_to_fit();

    d_prior.clear();
    d_prior.shrink_to_fit();

    // Reset raw pointers to avoid any potential double-free issues
    D_ptr = nullptr;
    filter_sigma_ptr = nullptr;
    d_prior_ptr = nullptr;

    // Ensure any remaining CUDA operations are complete
    cudaDeviceSynchronize();

    // Check for any CUDA errors during cleanup
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
    {
        std::cerr << "CUDA error during NonLinearTransformation destruction: "
                  << cudaGetErrorString(error) << std::endl;
    }
}

// Kernel to merge the separate components back into the interleaved deformation field
__global__ void merge_and_compute_locations(
    float *d_ptr,
    const float *x_component,
    const float *y_component,
    const float *z_component,
    const float *x_ptr,
    const int N0, const int N1, const int N2, const int N)
{
    const int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int i0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 < N0 && i1 < N1 && i2 < N2)
    {
        // Linear index calculation
        const int i = i0 * N1 * N2 + i1 * N2 + i2;
        const int base_idx = i * N;

        // Merge components and add initial positions in one step
        d_ptr[base_idx + 0] = x_component[i] + x_ptr[base_idx + 0];
        d_ptr[base_idx + 1] = y_component[i] + x_ptr[base_idx + 1];
        d_ptr[base_idx + 2] = z_component[i] + x_ptr[base_idx + 2];
    }
}

__global__ void compute_raw_regularization(
    const float *x_component, const float *y_component, const float *z_component,
    const float *D, const float rho_x, const float rho_y, const float rho_z,
    const float eta_x, const float eta_y, const float eta_z,
    const int num_voxels,
    float *raw_regularization)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_voxels)
    {
        // Compute each dimension's contribution: ηd² * ρ * D[i] * component[i]
        float rho = rho_x * rho_y * rho_z;
        float contrib_x = eta_x * eta_x * rho * D[i] * x_component[i] * x_component[i];
        float contrib_y = eta_y * eta_y * rho * D[i] * y_component[i] * y_component[i];
        float contrib_z = eta_z * eta_z * rho * D[i] * z_component[i] * z_component[i];

        raw_regularization[i] = (contrib_x + contrib_y + contrib_z);
    }
}

float NonLinearTransformation::updateDeformation(float *d_ptr,
                                                 const float *dream_location_ptr,
                                                 const float *x_ptr,
                                                 const bool sampling,
                                                 const bool sampling_gamma = false)
{
    // Define grid dimensions for CUDA kernels
    dim3 block_dim(32, 4, 2);
    dim3 grid_dim(
        (Nz_v + block_dim.x - 1) / block_dim.x,
        (Ny_v + block_dim.y - 1) / block_dim.y,
        (Nx_v + block_dim.z - 1) / block_dim.z);

    if (recompute_sigma and sampling)
    {
        compute_fourier_tmp_sigma_DCT();
    }

    // Compute displacements
    compute_displacement<<<grid_dim, block_dim>>>(tmp_x_ptr, tmp_y_ptr, tmp_z_ptr,
                                                  dream_location_ptr, x_ptr, Nx_v, Ny_v, Nz_v);

    // Forward DCT for each dimension (the DCT functions operate on device memory)
    dct3d->forward(tmp_x_ptr, tmp_x_ptr);
    dct3d->forward(tmp_y_ptr, tmp_y_ptr);
    dct3d->forward(tmp_z_ptr, tmp_z_ptr);

    // Smooth deformation (and accumulate cost) on the device
    if (sampling)
    {
        smooth_deformation_sampling<<<grid_dim, block_dim>>>(
            d_states, tmp_x_ptr, tmp_y_ptr, tmp_z_ptr, D_ptr, filter_sigma_ptr,
            gamma_x, gamma_y, gamma_z, sigma_spline,
            Nx_v, Ny_v, Nz_v, d_prior_ptr);
    }
    else
    {
        smooth_deformation_no_sampling<<<grid_dim, block_dim>>>(
            tmp_x_ptr, tmp_y_ptr, tmp_z_ptr, D_ptr,
            gamma_x, gamma_y, gamma_z, sigma_spline,
            Nx_v, Ny_v, Nz_v, d_prior_ptr);
    }

    // Compute deformation prior cost using Thrust reduction
    const float d_prior_cost = thrust::reduce(d_prior.begin(), d_prior.end(), 0.0f, thrust::plus<float>());

    if (sampling && sampling_gamma)
    {
        // Reuse d_prior for storing intermediate results
        compute_raw_regularization<<<(num_voxels + 255) / 256, 256>>>(
            tmp_x_ptr, tmp_y_ptr, tmp_z_ptr, D_ptr, rho_x, rho_y, rho_z, eta_x, eta_y, eta_z,
            num_voxels, d_prior_ptr);

        const float raw_reg_total = thrust::reduce(d_prior.begin(), d_prior.end(), 0.0f);

        const float alpha_post = alpha_0 + 0.5f * static_cast<float>(N * num_voxels);
        const float beta_post = beta_0 + 0.5f * static_cast<float>(num_voxels) * raw_reg_total;

        std::gamma_distribution<float> gamma_dist(alpha_post, 1.0f / beta_post);
        nu = gamma_dist(rng);
        updateGammasFromNu();
    }

    // Backward DCT for each dimension
    dct3d->backward(tmp_x_ptr, tmp_x_ptr);
    dct3d->backward(tmp_y_ptr, tmp_y_ptr);
    dct3d->backward(tmp_z_ptr, tmp_z_ptr);

    // Merge the backward-transformed components back into d_ptr
    merge_and_compute_locations<<<grid_dim, block_dim>>>(
        d_ptr,
        tmp_x_ptr,
        tmp_y_ptr,
        tmp_z_ptr,
        x_ptr,
        Nx_v, Ny_v, Nz_v, N);
    cudaDeviceSynchronize();

    return -d_prior_cost;
}

// Getter to return the shape of the final_locations array
std::vector<int> NonLinearTransformation::get_shape() const
{
    return {Nx_v, Ny_v, Nz_v, N};
}

std::pair<float, float *> NonLinearTransformation::smoothDeformation(const float *deformations)
{

    const int total_size = Nx_v * Ny_v * Nz_v * N;

    // Allocate device vectors
    thrust::device_vector<float> deformations_(total_size, 0.0f);
    thrust::device_vector<float> dream_locations(total_size);
    thrust::device_vector<float> voxel_pos(total_size, 0.0f);

    // Copy deformation field to dream_locations
    copy(deformations, deformations + total_size, dream_locations.begin());

    float *d_ptr = raw_pointer_cast(deformations_.data());
    const float *dream_locations_ptr = raw_pointer_cast(dream_locations.data());
    const float *voxel_pos_ptr = raw_pointer_cast(voxel_pos.data());

    // Call the smoothing method integrated in the whole EM scheme
    float log_prior_deformation = updateDeformation(d_ptr, dream_locations_ptr, voxel_pos_ptr, false, false);

    // Copy result back to host
    std::vector<float> deformations_host(total_size);
    thrust::copy(deformations_.begin(), deformations_.end(), deformations_host.begin());

    return std::make_pair(log_prior_deformation, deformations_host.data());
}

float *NonLinearTransformation::sampleDeformationField(const float *deformationField, const int number_of_samples)
{

    const int field_size = Nx_v * Ny_v * Nz_v * N;
    const int total_output_size = number_of_samples * field_size;

    // Allocate output array on host
    auto *out = new float[total_output_size];

    // Allocate device memory
    thrust::device_vector<float> deformations_(field_size, 0.0f);
    thrust::device_vector<float> dream_locations(field_size);
    thrust::device_vector<float> voxel_pos(field_size, 0.0f);

    // Allocate host vector for copying back results
    std::vector<float> deformations_host(field_size);

    // Optionally recompute sigma
    if (recompute_sigma)
    {
        compute_fourier_tmp_sigma_DCT();
    }

    // Copy deformationField to dream_locations on the device
    copy_n(deformationField, field_size, dream_locations.begin());

    // Get raw pointers for kernel calls
    float *d_ptr = raw_pointer_cast(deformations_.data());
    const float *dream_locations_ptr = raw_pointer_cast(dream_locations.data());
    const float *voxel_pos_ptr = raw_pointer_cast(voxel_pos.data());

    // Iterate for the number of samples
    for (int s = 0; s < number_of_samples; ++s)
    {
        // Sample transformation
        updateDeformation(d_ptr, dream_locations_ptr, voxel_pos_ptr, true, false);

        // Copy the result back to host
        thrust::copy(deformations_.begin(), deformations_.end(), deformations_host.begin());

        // Use std::copy to transfer sampled deformation field to the correct output slice
        std::copy(deformations_host.begin(), deformations_host.end(), out + s * field_size);
    }

    return out;
}

void NonLinearTransformation::compute_fourier_tmp_sigma_DCT()
{
    dim3 blockSize(32, 4, 2);
    dim3 gridSize(
        (Nz_v + blockSize.x - 1) / blockSize.x,
        (Ny_v + blockSize.y - 1) / blockSize.y,
        (Nx_v + blockSize.z - 1) / blockSize.z);

    compute_fourier_sigma<<<gridSize, blockSize>>>(
        filter_sigma_ptr, D_ptr, sigma_spline, gamma_x, gamma_y, gamma_z,
        Nx_v, Ny_v, Nz_v, N);
    recompute_sigma = false;
}
