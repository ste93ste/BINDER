//
// Created by stce on 03/01/25.
//

#ifndef TOPOLOGY_CORRECTION_3D_H
#define TOPOLOGY_CORRECTION_3D_H
#define _USE_MATH_DEFINES
#include <cmath>
#include <array>
#include <memory>
#include "FFT.h"

class topology_correction_3D {
public:
    topology_correction_3D();

    std::pair<int, int> compute_corner_jacobians_3D(double *J_corner) const;

    void compute_gradients_3D(double *grad) const;

    void limit_gradients_3D(double *grad, const double *J_corner) const;

    void compute_mean_coordinates(double &mean_x, double &mean_y, double &mean_z) const;

    void integrate_gradient_field_3D(const double *grad, double mean_x, double mean_y, double mean_z) const;

    static void get_jac_off_3D(int jac_num, int *xoff, int *yoff, int *zoff);

    double* correct_topology(const double* deformedVoxelPositions, double e1, double e2, int max_outer_it, int max_inner_it,
                             int threads, const double* voxel_sizes, int Nx_v, int Ny_v, int Nz_v);

    void to_grid() const;

    void to_physical() const;


private:
    std::unique_ptr<double[]> deformedVoxelPositions;
    std::array<double, 3> voxel_sizes{};
    double e1{};
    double e2{};
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
    std::unique_ptr<double[]> J_corner;
};



#endif //TOPOLOGY_CORRECTION_3D_H
