#ifndef REGISTRATION_CUH
#define REGISTRATION_CUH

#include <vector>
#include <tuple>
#include <math_constants.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <curand_kernel.h>
#include "NonLinearTransformation.cuh"

class Registration {
public:
    void accumulateThetaStatistics(bool sampling);

    // Constructor
    Registration(const float* final_locations,
                 int threads,
                 const float* voxel_pos,
                 const float* node_pos,
                 int Nx_v, int Ny_v, int Nz_v,
                 int Nx_n, int Ny_n, int Nz_n,
                 int spline_order,
                 float non_zero_voxels);

    // Destructor
    ~Registration();

    // Main interface methods
    void setMILikelihood(float alpha,
                         const float* theta_input,
                         int K, int L,
                         int threads,
                         const int* binned_nodes,
                         const int* binned_voxels);

    void setNonLinearTransformation(float gamma_x, float gamma_y, float gamma_z,
                                    float rho_x, float rho_y, float rho_z,
                                    float nu_x, float nu_y, float nu_z,
                                    const float* D);
    void EM(int max_number_of_iterations, float convergence_th, int update_likelihood_parameters, int debug);
    std::pair<std::vector<float>, std::vector<float>> sampler(int seed, int N_b, int N_s, int sample_gamma,
                                                              int sample_gamma_every,
                                                              int sample_posteriors, int sample_likelihood_parameters,
                                                              int sample_transformation_parameters);
    // Getter methods
    std::vector<float> get_final_locations() const;
    std::vector<int> get_shape() const;
    std::vector<float> get_likelihood_parameters() const;

    std::vector<float> get_dream_locations() const;

    void set_dream_locations(const float *dream_locations);

private:
    // Helper methods
    void initializeFilter() const;
    std::tuple<float, float> updateLikelihood(int);

    // Member variables
    const int threads;
    const float* voxel_pos;
    const float* node_pos;
    const int Nx_v, Ny_v, Nz_v;
    const int Nx_n, Ny_n, Nz_n;
    const int N;
    const float non_zero_voxels;
    const int num_voxels;
    const bool is2D;
    bool debug = false;
    std::unique_ptr<NonLinearTransformation> transformationObj;
    float moving_voxel_size_x;
    float moving_voxel_size_y;
    float moving_voxel_size_z;
    // CUDA handles and states

    curandState* w_states = nullptr;
    curandState* theta_states = nullptr;

    // Configuration parameters
    float alpha0{};
    int K{};
    int L{};
    bool sampling = false;
    float gamma = 1.0f;
    unsigned long seed = 42;

    // Thrust device vectors and pointers
    thrust::device_vector<float> d; //deformation
    float* d_ptr;
    thrust::device_vector<float> avg_d;
    float* avg_d_ptr{};
    thrust::device_vector<float> cov_d;
    float* cov_d_ptr{};

    thrust::device_vector<float> w; //posterior
    float* w_ptr;
    thrust::device_vector<float> w_sum; //posterior sum of columns
    float* w_sum_ptr;
    thrust::device_vector<float> w_sum_log; //log posterior sum for cost
    float* w_sum_log_ptr;
    thrust::device_vector<float> theta; //likelihood
    float* theta_ptr{};
    thrust::device_vector<float> theta_logpdf; //likelihood log pdf for prior cost
    float* theta_logpdf_ptr{};
    thrust::device_vector<float> dream_location; //Σwij*yj
    float* dream_location_ptr;
    thrust::device_vector<uint16_t> theta_index; //ui+vj*bins for (ui,vj) pair
    uint16_t* theta_index_ptr{};
    thrust::device_vector<int> u; //fixed image
    int* u_ptr{};
    thrust::device_vector<int> v; //moving image
    int* v_ptr{};
    thrust::device_vector<float> x; //locations in u
    float* x_ptr;
    
    thrust::device_vector<int> d_hist;
    int* d_hist_ptr;

    thrust::device_vector<float> d_weighted_hist;
    float* d_weighted_hist_ptr;

    //for cub functions
    void* d_temp_hist = nullptr;
    size_t temp_hist_bytes = 0;

    void* d_temp_seg = nullptr;
    size_t temp_seg_bytes = 0;
    
    thrust::device_vector<int> offsets;
    int* offsets_ptr;
    thrust::device_vector<int> counts;
    int* counts_ptr;
    thrust::device_vector<int> cum_counts;
    int* cum_counts_ptr;
    thrust::device_vector<int> indices;
    int* indices_ptr;
};



#endif // REGISTRATION_CUH