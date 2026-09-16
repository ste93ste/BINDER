#ifndef TOPOLOGY_CORRECTION_3D_CUH
#define TOPOLOGY_CORRECTION_3D_CUH
#define _USE_MATH_DEFINES
#include <cmath>
#include <array>
#include <memory>
#include "FFT.cuh"
#include <thrust/device_vector.h>

class topology_correction_3D {
public:
    topology_correction_3D();
    ~topology_correction_3D();

    std::pair<int, int> compute_corner_jacobians_3D(float *J_corner) const;

    void compute_gradients_3D(float *grad) const;

    void limit_gradients_3D(float *grad, const float *J_corner) const;

    void compute_mean_coordinates(float &mean_x, float &mean_y, float &mean_z) const;

    void integrate_gradient_field_3D(const float *grad) const;

    static void get_jac_off_3D(int jac_num, int *xoff, int *yoff, int *zoff);

    float* correct_topology(const float* deformedVoxelPositions, float e1, float e2, int max_outer_it, int max_inner_it,
                            int threads, const float* voxel_sizes, int Nx_v, int Ny_v, int Nz_v);

    void to_grid() const;

    void to_physical() const;


private:
    std::unique_ptr<float[]> deformedVoxelPositions;
    float voxel_size_x;
    float voxel_size_y;
    float voxel_size_z;
    float e1{};
    float e2{};
    int max_outer_it{};
    int max_inner_it{};
    int Nx_v{};
    int Ny_v{};
    int Nz_v{};
    int N{};
    int threads{};
    std::unique_ptr<FFT> fft_x{};
    std::unique_ptr<FFT> fft_y{};
    std::unique_ptr<FFT> fft_z{};
    int corners{};
    bool in_range{};
    std::unique_ptr<float[]> J_corner;
    thrust::device_vector<float> d_positions;
    float* d_positions_ptr;
    thrust::device_vector<float> d_grad;
    float* d_grad_ptr;
    thrust::device_vector<float> d_corner;
    float* d_corner_ptr;

};



#endif //TOPOLOGY_CORRECTION_3D_CUH
