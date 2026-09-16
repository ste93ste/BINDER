//
// Created by stce on 2/14/25.
//
#include "Registration.cuh"
#include "DCT3D.cuh"
#include "NonLinearTransformation.cuh"
#include <math_constants.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <curand_kernel.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <iomanip>
#include <cub/cub.cuh>




__device__ __forceinline__ float cubic_bspline_abs(float x) {
    // x >= 0
    if (x < 1.0f) {
        float x2 = x * x;
        return 0.6666666667f - x2 + 0.5f * x2 * x;
    }
    if (x < 2.0f) {
        float t = 2.0f - x;
        return (t * t * t) * 0.1666666667f;
    }
    return 0.0f;
}


__global__ void init_random_states(curandState *states, const unsigned long seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

//for sampling from dirichlet
__device__ float gamma_sample(const float alpha, curandState *state) {
    // Marsaglia and Tsang method (assuming alpha > 1)
    const float d = alpha - 1.0f / 3.0f;
    const float c = 1.0f / sqrtf(9.0f * d);
    while (true) {
        // Generate normally distributed random number
        const float x = curand_normal(state);
        float v = 1.0f + c * x;

        if (v <= 0) {
            continue;
        }

        v = v * v * v;  // cube of v

        // Apply acceptance condition
        if (const float u = curand_uniform(state);
            u < 1.0f - 0.0331f * (x * x) * (x * x) || logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) {
            return d * v;
        }
    }
}

//for w cost
struct log_functor {
    __host__ __device__
    float operator()(const float& x) const {
        return logf(x); // Using logf for float
    }
};



// Calculate Dirichlet log PDF for multiple probability vectors in parallel
__global__ void dirichlet_logpdf(
    const float* theta,     // Input probability vectors [K x L]
    const float alpha,      // Concentration parameter (same for all dimensions)
    float* output,         // Output log PDF values [K]
    const int K,           // Number of probability vectors
    const int L            // Dimension of each probability vector
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    // Get pointer to current probability vector
    const float* theta_k = theta + k * L;

    float log_pdf = 0.0f;

    // Calculate log(Γ(sum(α))) = log(Γ(L*α)) since all alphas are the same
    const float sum_alpha = static_cast<float>(L) * alpha;
    log_pdf += lgammaf(sum_alpha);

    // Calculate -sum(log(Γ(α))) = -L * log(Γ(α)) since all alphas are the same
    log_pdf -= static_cast<float>(L) * lgammaf(alpha);

    // Calculate sum((α-1) * log(θ_i))
    float sum_log_theta = 0.0f;
    bool valid = true;

    for (int i = 0; i < L; ++i) {
        if (theta_k[i] <= 0.0f) {
            valid = false;
            break;
        }
        sum_log_theta += logf(theta_k[i]);
    }

    if (!valid) {
        output[k] = -CUDART_INF_F;
        return;
    }

    // Add final term
    log_pdf += (alpha - 1.0f) * sum_log_theta;

    output[k] = log_pdf;
}

__global__ void compute_dream_location(
    float* __restrict__ w,              // output normalized weights, layout: IxK
    float* __restrict__ w_sum,              // output sum of weights
    const int* __restrict__ counts, // count for each l : size L
    const int* __restrict__ cumulative_counts, //cumulative count for each l : size L
    const int* __restrict__ indices, // indices for each voxel in u within each l : size I
    const float* __restrict__ d,
    const float* __restrict__ theta,
    const int* __restrict__ u,
    const int* __restrict__ v,
    float* __restrict__ dream_location,
    int N0_u, int N1_u, int N2_u,
    int N0_v, int N1_v, int N2_v,
    int L,int K)
{
    int i2 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i1 = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int i0 = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (i0 >= N0_u || i1 >= N1_u || i2 >= N2_u) return;

    const int i3 = (i0 * N1_u + i1) * N2_u + i2;

    int ui = u[i3];

    float x0 = d[i3 * 3 + 0];
    float x1 = d[i3 * 3 + 1];
    float x2 = d[i3 * 3 + 2];

    int base0 = __float2int_rd(x0);
    int base1 = __float2int_rd(x1);
    int base2 = __float2int_rd(x2);

    float bx[4], by[4], bz[4];
    int y0[4], y1[4], y2[4];

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int off = t - 1;
        y0[t] = base0 + off;
        y1[t] = base1 + off;
        y2[t] = base2 + off;

        bx[t] = cubic_bspline_abs(fabsf((float)y0[t] - x0));
        by[t] = cubic_bspline_abs(fabsf((float)y1[t] - x1));
        bz[t] = cubic_bspline_abs(fabsf((float)y2[t] - x2));
    }

    // First pass: compute unnormalized weights, sum, and dream location
    float sum_w = 0.0f;
    float dl0 = 0.0f, dl1 = 0.0f, dl2 = 0.0f;
   
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        int Y0 = y0[a];
        bool in0 = (unsigned)Y0 < (unsigned)N0_v;
        float b0 = bx[a];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int Y1 = y1[b];
            bool in1 = (unsigned)Y1 < (unsigned)N1_v;
            float b01 = b0 * by[b];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int Y2 = y2[c];
                bool in2 = (unsigned)Y2 < (unsigned)N2_v;

                int vj = 0;
                if (in0 && in1 && in2) {
                    int j = (Y0 * N1_v + Y1) * N2_v + Y2;
                    vj = v[j];
                }

                float theta_uivj = theta[ui + L * vj];
                
                float ww = (b01 * bz[c]) * theta_uivj;
              
                sum_w += ww;
                dl0 += ww * (float)Y0;
                dl1 += ww * (float)Y1;
                dl2 += ww * (float)Y2;
            }
        }
    }
    
    w_sum[i3] = sum_w;
    if (sum_w <= 0.0f) {
        dream_location[i3 * 3 + 0] = (float)base0;
        dream_location[i3 * 3 + 1] = (float)base1;
        dream_location[i3 * 3 + 2] = (float)base2;
        return;
    }

    float inv = 1.0f / sum_w;

    dream_location[i3 * 3 + 0] = dl0*inv;
    dream_location[i3 * 3 + 1] = dl1*inv;
    dream_location[i3 * 3 + 2] = dl2*inv;
    

    // Second pass: normalize weights and compute weights for each class k
    int prefix = (ui == 0 ? 0 : cumulative_counts[ui - 1]);  // exclusive prefix
    int cnt    = counts[ui];
    int idx_in = indices[i3];

    int base = prefix * K;  // start of label block in w  
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        int Y0 = y0[a];
        bool in0 = (unsigned)Y0 < (unsigned)N0_v;
        float b0 = bx[a];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int Y1 = y1[b];
            bool in1 = (unsigned)Y1 < (unsigned)N1_v;
            float b01 = b0 * by[b];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int Y2 = y2[c];
                bool in2 = (unsigned)Y2 < (unsigned)N2_v;

                int vj = 0;
                if (in0 && in1 && in2) {
                    int j = (Y0 * N1_v + Y1) * N2_v + Y2;
                    vj = v[j];
                }

                float theta_uivj = theta[ui + L * vj];
                
                float ww = (b01 * bz[c]) * theta_uivj;

                int index = base + vj * cnt + idx_in;
                w[index] += inv*ww;
            }
        }
    }

}



__device__ __forceinline__ int lower_bound_64(const float* __restrict__ cdf, float x)
{
    // returns smallest i in [0,63] with cdf[i] >= x, assuming cdf is nondecreasing
    int lo = 0, hi = 64;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        float v = cdf[mid];
        if (v < x) lo = mid + 1;
        else       hi = mid;
    }
    return (lo < 64) ? lo : 63;
}

__global__ void sample_latent_assignment(curandState* __restrict__ states,
                                         const float* __restrict__ d,
                                         const float* __restrict__ theta,
                                         const int* __restrict__ u,
                                         const int* __restrict__ v,
                                         float* __restrict__ dream_location,
                                         float* __restrict__ w_sum,
                                         uint16_t* __restrict__ theta_index,
                                         int N0_u, int N1_u, int N2_u,
                                         int N0_v, int N1_v, int N2_v,
                                         int L)
{
    int i2 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i1 = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int i0 = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (i0 >= N0_u || i1 >= N1_u || i2 >= N2_u) return;

    const int i3 = (i0 * N1_u + i1) * N2_u + i2;

    int ui = u[i3];

    float x0 = d[i3 * 3 + 0];
    float x1 = d[i3 * 3 + 1];
    float x2 = d[i3 * 3 + 2];

    // floor
    int base0 = __float2int_rd(x0);
    int base1 = __float2int_rd(x1);
    int base2 = __float2int_rd(x2);

    // Precompute axis weights and neighbor coordinates
    float bx[4], by[4], bz[4];
    int y0[4], y1[4], y2[4];

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int off = t - 1; // -1,0,1,2
        y0[t] = base0 + off;
        y1[t] = base1 + off;
        y2[t] = base2 + off;

        bx[t] = cubic_bspline_abs(fabsf((float)y0[t] - x0));
        by[t] = cubic_bspline_abs(fabsf((float)y1[t] - x1));
        bz[t] = cubic_bspline_abs(fabsf((float)y2[t] - x2));
    }

    float cdf[64];
    float sum_w = 0.0f;

    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        int Y0 = y0[a];
        bool in0 = (unsigned)Y0 < (unsigned)N0_v;
        float b0 = bx[a];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            int Y1 = y1[b];
            bool in1 = (unsigned)Y1 < (unsigned)N1_v;
            float b01 = b0 * by[b];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int Y2 = y2[c];
                bool in2 = (unsigned)Y2 < (unsigned)N2_v;

                int vj = 0;
                if (in0 && in1 && in2) {
                    int j = (Y0 * N1_v + Y1) * N2_v + Y2;
                    vj = v[j];
                }

                float theta_uivj = theta[ui + L * vj];

                float w = (b01 * bz[c]) * theta_uivj;

                int offset = (a << 4) + (b << 2) + c; // 0..63
                sum_w += w;
                cdf[offset] = sum_w;
            }
        }
    }

    w_sum[i3] = sum_w;
    if (sum_w <= 0.0f) return;

    // draw r in (0, sum_w]
    float r = curand_uniform(&states[i3]) * sum_w;

    int selected = lower_bound_64(cdf, r);

    // decode offsets
    int off0 = selected / 16 - 1;
    int off1 = (selected % 16) / 4 - 1;
    int off2 = (selected % 4) - 1;

    int yj0 = base0 + off0;
    int yj1 = base1 + off1;
    int yj2 = base2 + off2;

    dream_location[i3 * 3 + 0] = (float)yj0;
    dream_location[i3 * 3 + 1] = (float)yj1;
    dream_location[i3 * 3 + 2] = (float)yj2;

    int vj = 0;
    if ((unsigned)yj0 < (unsigned)N0_v &&
        (unsigned)yj1 < (unsigned)N1_v &&
        (unsigned)yj2 < (unsigned)N2_v)
    {
        int j = (yj0 * N1_v + yj1) * N2_v + yj2;
        vj = v[j];
    }

    theta_index[i3] = (uint16_t)(ui + L * vj);
}


//fill out the likelihood from the accumulated data
__global__ void fill_theta(const float* __restrict__ values,
                           float* __restrict__ theta,
                           int K, int L)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K*L) return;

    int l = idx / K;   // l-major
    int k = idx % K;

    // theta expects theta[l + L*k]
    theta[l + L*k] = values[l*K + k];
}

__global__ void update_likelihood(float* theta, const int K, const int L, const float alpha0) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k<K){
        float sum_theta = 0.0;
        for(int l=0; l<L; l++){
            const int index = l+k*L;
            theta[index] = theta[index]+alpha0-1;
            sum_theta += theta[index];
        }

        for(int l=0; l<L; l++){
            const int index = l+k*L;
            theta[index] /= sum_theta;
        }
    }
}

__global__ void sample_likelihood(curandState *states, float* theta, const int K, const int L, const float alpha0) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k<K){
        float sum_theta = 0.0;
        for(int l=0; l<L; l++){
            const int index = l+k*L;
            theta[index] = gamma_sample(theta[index]+alpha0,&states[index]);
            sum_theta += theta[index];
        }

        for(int l=0; l<L; l++){
            const int index = l+k*L;
            theta[index] /= sum_theta;
        }
    }
}

__global__ void deformation_stats_kernel(
    const float* __restrict__ new_d,   // [N0*N1*N2, 3]
    float* __restrict__ avg_d,         // [N0*N1*N2, 3]
    float* __restrict__ cov_d,         // [N0*N1*N2, 3, 3] accumulator M2
    float num_prev_samples,
    int N0,
    int N1,
    int N2
) {
    const int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int i0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 < N0 && i1 < N1 && i2 < N2) {
        const int i = (i0 * N1 + i1) * N2 + i2;

        const float n_new = num_prev_samples + 1.0f;
        const float inv_n = 1.0f / n_new;

        float x[3];
        float mean_old[3];
        float mean_new[3];
        float delta1[3];
        float delta2[3];

        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int idx = i * 3 + c;

            x[c] = new_d[idx];
            mean_old[c] = avg_d[idx];

            delta1[c] = x[c] - mean_old[c];
            mean_new[c] = mean_old[c] + delta1[c] * inv_n;

            avg_d[idx] = mean_new[c];

            delta2[c] = x[c] - mean_new[c];
        }

        // cov_d stores the Welford M2 accumulator, not yet divided.
        // Shape per voxel: 3x3, row-major.
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                const int cov_idx = i * 9 + r * 3 + c;
                cov_d[cov_idx] += delta1[r] * delta2[c];
            }
        }
    }
}


Registration::Registration(const float* final_locations,
                           const int threads,
                           const float* voxel_pos,
                           const float* node_pos,
                           const int Nx_v, const int Ny_v, const int Nz_v,
                           const int Nx_n, const int Ny_n, const int Nz_n,
                           int spline_order,
                           const float non_zero_voxels)
    : threads(threads),
      voxel_pos(voxel_pos),
      node_pos(node_pos),
      Nx_v(Nx_v), Ny_v(Ny_v), Nz_v(Nz_v),
      Nx_n(Nx_n), Ny_n(Ny_n), Nz_n(Nz_n),
      N(Nz_v == 1 ? 2 : 3),
      non_zero_voxels(non_zero_voxels),
      num_voxels(Nx_v * Ny_v * Nz_v),
      is2D(Nz_v == 1)
{
    std::cout << "Setting up Registration object..." << std::endl;


    // Initialize basic vectors without specific size yet
    // Sizes will be set in setMILikelihood or other initialization methods
    d.resize(N * num_voxels);
   
    w_sum.resize(num_voxels);
    x.resize(N * num_voxels);
    dream_location.resize(N * num_voxels);


    // Copy the final locations to device memory
    thrust::copy(voxel_pos, voxel_pos + N * num_voxels, x.begin());
    thrust::copy(final_locations, final_locations + N * num_voxels, d.begin());

    d_ptr = raw_pointer_cast(d.data());
    w_sum_ptr = raw_pointer_cast(w_sum.data());
    x_ptr = raw_pointer_cast(x.data());
    dream_location_ptr = raw_pointer_cast(dream_location.data());

    std::cout << "Done!" << std::endl;

}

Registration::~Registration() {
    // Free CUDA-allocated memory
    if (w_states) {
        cudaFree(w_states);
        w_states = nullptr;
    }

    if (theta_states) {
        cudaFree(theta_states);
        theta_states = nullptr;
    }

    // Clear Thrust device vectors to release GPU memory
    d.clear();
    d.shrink_to_fit();

    w.clear();
    w.shrink_to_fit();

    w_sum.clear();
    w_sum.shrink_to_fit();

    x.clear();
    x.shrink_to_fit();

    dream_location.clear();
    dream_location.shrink_to_fit();

    // Clear MI-specific vectors if they were initialized
    theta.clear();
    theta.shrink_to_fit();

    theta_logpdf.clear();
    theta_logpdf.shrink_to_fit();

    u.clear();
    u.shrink_to_fit();

    v.clear();
    v.shrink_to_fit();

    theta_index.clear();
    theta_index.shrink_to_fit();

    avg_d.clear();
    avg_d.shrink_to_fit();

    cov_d.clear();
    cov_d.shrink_to_fit();


    d_hist.clear();
    d_hist.shrink_to_fit();

    d_weighted_hist.clear();
    d_weighted_hist.shrink_to_fit();

    counts.clear();
    counts.shrink_to_fit();

    cum_counts.clear();
    cum_counts.shrink_to_fit();

    indices.clear();
    indices.shrink_to_fit();

    offsets.clear();
    offsets.shrink_to_fit();


    // Reset raw pointers to avoid any potential double-free issues
    d_ptr = nullptr;
    w_ptr = nullptr;
    w_sum_ptr = nullptr;
    x_ptr = nullptr;
    dream_location_ptr = nullptr;
    theta_ptr = nullptr;
    u_ptr = nullptr;
    v_ptr = nullptr;
    theta_logpdf_ptr = nullptr;
    theta_index_ptr = nullptr;
    avg_d_ptr = nullptr;
    cov_d_ptr = nullptr;
    d_hist_ptr = nullptr;
    d_weighted_hist_ptr = nullptr;
    counts_ptr = nullptr;
    cum_counts_ptr = nullptr;
    indices_ptr = nullptr;
    offsets_ptr = nullptr;

    if (d_temp_hist) {
        cudaFree(d_temp_hist);
        d_temp_hist = nullptr;
    }

    if (d_temp_seg) {
        cudaFree(d_temp_seg);
        d_temp_seg = nullptr;
    }

    // The transformationObj unique_ptr will automatically clean itself up
    // but we can explicitly reset it if needed
    transformationObj.reset();

    // Ensure any remaining CUDA operations are complete
    cudaDeviceSynchronize();
    
    // Check for any CUDA errors during cleanup
    if (cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        std::cerr << "CUDA error during destruction: " << cudaGetErrorString(error) << std::endl;
    }
}

void Registration::setMILikelihood(const float alpha,
                                 const float* theta_input,
                                 const int K, const int L,
                                 int threads,
                                 const int* binned_nodes,
                                 const int* binned_voxels)
{
    std::cout << "Setting up MI Likelihood..." << std::endl;

    // Set MI-specific parameters
    this->alpha0 = alpha;
    this->K = K;
    this->L = L;

    // Resize and initialize MI-specific vectors
    theta.resize(K * L);  // Use K and L consistently
    theta_logpdf.resize(K); // One per row of theta

    u.resize(Nx_v * Ny_v * Nz_v);
    v.resize(Nx_n * Ny_n * Nz_n);
    if(!sampling){
        counts.resize(L);
        cum_counts.resize(L);
        indices.resize(num_voxels);
        offsets.resize(K*L+1);
        std::vector<int> c(L,0);
        std::vector<int> idc(num_voxels);
        for(int i0=0; i0<Nx_v; i0++){
            for(int i1=0; i1<Ny_v; i1++){
                for(int i2=0; i2<Nz_v; i2++){
                    int idx = i2+i1*Nz_v+i0*Nz_v*Ny_v;
                    idc[idx] = c[binned_voxels[idx]]++;
                }
            }
        }
        std::vector<int> cumulative_counts(c.size());
        std::partial_sum(c.begin(), c.end(), cumulative_counts.begin());

        std::vector<int> boundaries;
        for(int bu=0; bu<L; bu++){
            int c1 = bu==0?0:cumulative_counts[bu-1];
            for(int bv=0; bv<K; bv++){
                boundaries.push_back(c1*K+bv*c[bu]);
            }
        }
        boundaries.push_back(num_voxels*K);
        thrust::copy(boundaries.begin(),boundaries.end(),offsets.begin());
        thrust::copy(c.begin(),c.end(),counts.begin());
        thrust::copy(cumulative_counts.begin(),cumulative_counts.end(),cum_counts.begin());
        thrust::copy(idc.begin(),idc.end(),indices.begin());
        counts_ptr = raw_pointer_cast(counts.data());
        cum_counts_ptr = raw_pointer_cast(cum_counts.data());
        indices_ptr = raw_pointer_cast(indices.data());
        offsets_ptr = raw_pointer_cast(offsets.data());
    }
    

    // Copy data to device
    thrust::copy(theta_input, theta_input + K * L, theta.begin());
    thrust::copy(binned_voxels, binned_voxels + Nx_v * Ny_v * Nz_v, u.begin());
    thrust::copy(binned_nodes, binned_nodes + Nx_n * Ny_n * Nz_n, v.begin());
    

    theta_ptr = raw_pointer_cast(theta.data());
    u_ptr = raw_pointer_cast(u.data());
    v_ptr = raw_pointer_cast(v.data());
    theta_logpdf_ptr = raw_pointer_cast(theta_logpdf.data());

    
    std::cout << "Done!" << std::endl;

}


void Registration::setNonLinearTransformation(float gamma_x, float gamma_y, float gamma_z,
                                              float rho_x, float rho_y, float rho_z,
                                              float eta_x, float eta_y, float eta_z,
                                              const float* D){
    moving_voxel_size_x = eta_x;
    moving_voxel_size_y = eta_y;
    moving_voxel_size_z = eta_z;
    std::cout << "Setting up NonLinear Transformation..." << std::endl;
    this->transformationObj = std::make_unique<NonLinearTransformation>(gamma_x, gamma_y, gamma_z,
                                                                        rho_x, rho_y, rho_z,
                                                                        eta_x, eta_y, eta_z,
                                                                        D,
                                                                        Nx_v, Ny_v, Nz_v, threads,
                                                                        0.6f, 0.0f);
    std::cout << "Done!" << std::endl;
}


__global__ void hist_to_float(const int* __restrict__ hist,
                              float* __restrict__ theta,
                              int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) theta[i] = (float)hist[i];
}

std::tuple<float, float> Registration::updateLikelihood(const int update_likelihood_parameters) {

    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(
        (Nz_v + block_dim.x - 1) / block_dim.x,
        (Ny_v + block_dim.y - 1) / block_dim.y,
        (Nx_v + block_dim.z - 1) / block_dim.z
    );
    

    if (sampling) {
        //sampling from latent assignments by computing the normalized cumulative sum
        //the latent assignments are stored in dream_location for memory efficiency
        sample_latent_assignment<<<grid_dim, block_dim>>>(
            w_states,
            d_ptr,
            theta_ptr,
            u_ptr,
            v_ptr,
            dream_location_ptr,
            w_sum_ptr,
            theta_index_ptr,
            Nx_v, Ny_v, Nz_v,
            Nx_n, Ny_n, Nz_n,
            L
        );
    } else {
        cudaMemset(w_ptr, 0, K * num_voxels * sizeof(float));
        //computing dream location and normalizing w
        compute_dream_location<<<grid_dim, block_dim>>>(
            w_ptr,
            w_sum_ptr,
            counts_ptr,
            cum_counts_ptr,
            indices_ptr,
            d_ptr,
            theta_ptr,
            u_ptr,
            v_ptr,
            dream_location_ptr,
            Nx_v, Ny_v, Nz_v,
            Nx_n, Ny_n, Nz_n,
            L,K
        );
    }
    cudaDeviceSynchronize();
    const float w_cost = thrust::transform_reduce(w_sum.begin(), w_sum.end(), log_functor(), 0.0f, thrust::plus<float>());

    // Update likelihood parameters
    // 2: likelihood
    // 2.1: accumulate likelihood
    if (update_likelihood_parameters) {

        //2.2: update/sample likelihood
        if (sampling) {
            cub::DeviceHistogram::HistogramEven(
                d_temp_hist, temp_hist_bytes,
                theta_index_ptr, d_hist_ptr,
                K*L+1, 0, K*L,
                num_voxels);
         
            hist_to_float<<<K * L, 1>>>(d_hist_ptr, theta_ptr, K*L);
            cudaDeviceSynchronize();
            sample_likelihood<<<K, 1>>>(theta_states, theta_ptr, K, L, alpha0);
            cudaDeviceSynchronize();
        } else {
            cub::DeviceSegmentedReduce::Sum(
                d_temp_seg, temp_seg_bytes,
                w_ptr, d_weighted_hist_ptr,
                K*L, offsets_ptr, offsets_ptr + 1);
            fill_theta<<<K*L,1>>>(d_weighted_hist_ptr,theta_ptr,K,L);
            cudaDeviceSynchronize();
            update_likelihood<<<K, 1>>>(theta_ptr, K, L, alpha0);
            cudaDeviceSynchronize();
        }

    }
    //theta prior cost
    dirichlet_logpdf<<<K, 1>>>(theta_ptr, alpha0, theta_logpdf_ptr, K, L);
    const float theta_prior_cost = thrust::reduce(theta_logpdf.begin(), theta_logpdf.end(), 0.0f, thrust::plus<float>());

    return {-w_cost, -theta_prior_cost};
}


void Registration::EM(const int max_number_of_iterations, const float convergence_th,
                      const int update_likelihood_parameters, int debug) {

    sampling = false;
    bool sample_gamma = false;
    int iteration = 0;
    bool converged = false;
    float mlp_old = 1e100f;
    float time_E_step_and_likelihood;
    float time_M_step_transformation;

    this->debug = debug;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    w.resize(K * num_voxels); 
    w_ptr = raw_pointer_cast(w.data());
    
    d_weighted_hist.resize(K*L);
    d_weighted_hist_ptr = raw_pointer_cast(d_weighted_hist.data());

    d_temp_seg = nullptr;
    temp_seg_bytes = 0;

    cub::DeviceSegmentedReduce::Sum(
        d_temp_seg, temp_seg_bytes,
        w_ptr, d_weighted_hist_ptr,
        K*L, offsets_ptr, offsets_ptr + 1);

    cudaMalloc(&d_temp_seg, temp_seg_bytes);

    while (iteration < max_number_of_iterations && !converged) {
        std::cout << "Iteration: " << iteration + 1 << std::endl;

        // Update likelihood
        cudaEventRecord(start);
        auto [mll, log_prior_likelihood] = updateLikelihood(update_likelihood_parameters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_E_step_and_likelihood, start, stop);
        std::cout << "Time estimating posterior and updating likelihood (ms): " << time_E_step_and_likelihood << std::endl;

        // Update deformation
        cudaEventRecord(start);
        const float log_prior_transformation = transformationObj->updateDeformation(d_ptr, dream_location_ptr, x_ptr,
                                                                                    sampling, sample_gamma);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_M_step_transformation, start, stop);
        std::cout << "Time updating transformation (ms): " << time_M_step_transformation << std::endl;

        const float mlp = mll + log_prior_transformation + log_prior_likelihood;
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Min Log-Likelihood: " << mll << std::endl;
        std::cout << "Min Log-Prior likelihood: " << log_prior_likelihood << std::endl;
        std::cout << "Min Log-Prior transformation: " << log_prior_transformation << std::endl;
        std::cout << "Min Log-Posterior: " << mlp << std::endl;
        std::cout << "Min Log-Posterior per nonzero voxel: " << mlp / non_zero_voxels << std::endl;
        std::cout << "Time elapsed (ms): " << time_E_step_and_likelihood + time_M_step_transformation << std::endl;

        // Check for numerical errors and convergence
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

std::pair<std::vector<float>, std::vector<float>> Registration::sampler(const int seed, const int N_b, const int N_s,
                                                                        int sample_gamma,
                                                                        int sample_gamma_every,
                                                                        int sample_posteriors,
                                                                        int sample_likelihood_parameters,
                                                                        int sample_transformation_parameters) {


    // Initialize CUDA random states if not already done
    if (!w_states) {
        cudaMalloc(&w_states, num_voxels * sizeof(curandState));
        cudaMalloc(&theta_states, K * L * sizeof(curandState));
        // Initialize random states
        init_random_states<<<num_voxels, 1>>>(w_states, seed);
        init_random_states<<<K * L, 1>>>(theta_states, seed);
    }

    float time_E_step_and_likelihood;
    float time_M_step_transformation;

    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(
        (Nz_v + block_dim.x - 1) / block_dim.x,
        (Ny_v + block_dim.y - 1) / block_dim.y,
        (Nx_v + block_dim.z - 1) / block_dim.z
    );

    this->seed = seed;

    avg_d.resize(N * num_voxels);
    fill(avg_d.begin(), avg_d.end(), 0.0f);
    avg_d_ptr = raw_pointer_cast(avg_d.data());

    cov_d.resize(N * N * num_voxels);
    fill(cov_d.begin(), cov_d.end(), 0.0f);
    cov_d_ptr = raw_pointer_cast(cov_d.data());


    // New size for theta index (no more 64 neighbours)
    theta_index.resize(num_voxels);
    theta_index_ptr = raw_pointer_cast(theta_index.data());

    d_hist.resize(K*L);
    d_hist_ptr = raw_pointer_cast(d_hist.data());
    
    d_temp_hist = nullptr;
    temp_hist_bytes = 0;

    cub::DeviceHistogram::HistogramEven(
    d_temp_hist, temp_hist_bytes,
    theta_index_ptr, d_hist_ptr,
    K*L+1, 0, K*L,
    num_voxels);

    cudaMalloc(&d_temp_hist, temp_hist_bytes);

    sampling = true;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector out(Nx_v * Ny_v * Nz_v * N, 0.0f);
    std::vector out_2(Nx_v * Ny_v * Nz_v * N * N, 0.0f);

    for (int sweep = 0; sweep < N_b + N_s; ++sweep) {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Sweep: " << sweep + 1 << std::endl;

        cudaEventRecord(start);
        // Sample likelihood
        if (sample_posteriors) {
            updateLikelihood(sample_likelihood_parameters);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_E_step_and_likelihood, start, stop);
        std::cout << "Time estimating posterior and updating likelihood (ms): " << time_E_step_and_likelihood << std::endl;

        // Sample deformation
        cudaEventRecord(start);
        bool sample_gamma_this_sweep = sample_gamma && (sweep % sample_gamma_every == 0);
        transformationObj->updateDeformation(d_ptr, dream_location_ptr, x_ptr, sampling, sample_gamma_this_sweep);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_M_step_transformation, start, stop);
        std::cout << "Time updating transformation (ms): " << time_M_step_transformation << std::endl;

        if (sweep >= N_b){
            deformation_stats_kernel<<<grid_dim,block_dim>>>(d_ptr,avg_d_ptr,cov_d_ptr,static_cast<float>(sweep - N_b),Nx_v,Ny_v,Nz_v);
            cudaDeviceSynchronize();
        }
    }

    thrust::transform(
        cov_d.begin(), cov_d.end(),   // input
        cov_d.begin(),               // output (in-place)
        [N_s] __device__ (float x) {
            return x / (N_s - 1);
        }
    );

    std::cout << "Time elapsed (ms): " << time_E_step_and_likelihood + time_M_step_transformation << std::endl;

    thrust::copy(avg_d.begin(), avg_d.end(), out.begin());
    thrust::copy(cov_d.begin(), cov_d.end(), out_2.begin());

    return std::make_pair(std::move(out), std::move(out_2));
}

std::vector<float> Registration::get_final_locations() const {
    std::vector<float> result(num_voxels * N);
    thrust::copy(d.begin(), d.begin() + num_voxels * N, result.begin());
    return result;
}

// Getter to return the shape of the final_locations array
std::vector<int> Registration::get_shape() const {
    return { Nx_v, Ny_v, Nz_v, N };
}

std::vector<float> Registration::get_likelihood_parameters() const {
    std::vector<float> result(K * L);
    thrust::copy(theta.begin(), theta.end(), result.begin());
    return result;
}

// Getter to return dream_locations
std::vector<float> Registration::get_dream_locations() const {
    std::vector<float> result(num_voxels * N);
    thrust::copy(dream_location.begin(), dream_location.end(), result.begin());
    return result;
}

// Getter for dream_locations
void Registration::set_dream_locations(const float * dream_locations) {
    for (int x = 0; x < Nx_v; ++x) {
        for (int y = 0; y < Ny_v; ++y) {
            for (int z = 0; z < Nz_v; ++z) {
                for (int n = 0; n < N; ++n) {
                    this->dream_location[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n] =
                        dream_locations[x * (Ny_v * Nz_v * N) + y * (Nz_v * N) + z * N + n];
                }
            }
        }
    }
}