//
// Created by stce on 03/01/25.
//

#ifndef TOPOLOGY_CORRECTION_2D_H
#define TOPOLOGY_CORRECTION_2D_H
#define _USE_MATH_DEFINES
#include <cmath>
#include <array>
#include <memory>
#include "FFT.h"

class topology_correction_2D {
public:
    topology_correction_2D();

    std::pair<int, int> compute_corner_jacobians_2D(double *J_corner) const;

    void compute_gradients_2D(double *grad) const;

    void limit_gradients_2D(double *grad, const double *J_corner) const;

    void integrate_gradient_field_2D(const double *grad, double mean_x, double mean_y) const;

    static void get_jac_off_2D(int jac_num, int *xoff, int *yoff);

    void compute_mean_coordinates(double &mean_x, double &mean_y) const;

    double* correct_topology(const double* deformedVoxelPositions, double e1, double e2,
                             int max_outer_it, int max_inner_it,
                             int threads, const double * voxel_sizes, int Nx_v, int Ny_v);

private:
    std::unique_ptr<double[]> deformedVoxelPositions{};
    double e1{};
    double e2{};
    int Nx_v{};
    int Ny_v{};
    int N{};
    int max_outer_it{};
    int max_inner_it{};
    int threads{};
    int corners{};
    bool in_range{};
    std::array<double, 2> voxel_sizes{};
    std::unique_ptr<FFT> fft_x{};
    std::unique_ptr<FFT> fft_y{};

    void to_grid() const;
    void to_physical() const;

};



#endif //TOPOLOGY_CORRECTION_2D_H
