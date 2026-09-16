//
// Created by stce on 04/01/25.
//

#ifndef AFFINETRANSFORMATION_H
#define AFFINETRANSFORMATION_H
#include "Transformation.h"
#include <random>
#include <vector>
#include <Eigen/Dense>


class AffineTransformation final : public Transformation{
public:
    AffineTransformation(const double * A, const double * t, int Nx_v, int Ny_v, int Nz_v,
                        int threads, double sigma_spline, double spline_offset);
    ~AffineTransformation() override;

    void compute_fourier_tmp_sigma_DCT();
    double constrain_transformation(const double *dream_locations,
                                    const double *voxel_pos,
                                    double *ds,
                                    double *final_locations,
                                    int *node_indices) override;
    void sample_transformation(const double *dream_locations,
                               const double *voxel_pos,
                               double *ds,
                               double *final_locations,
                               int *node_indices,
                               int sample_gamma) override;

    void set_generator(std::mt19937 &gen) override;


private:
    int Nx_v;
    int Ny_v;
    int Nz_v;
    int N;
    int threads;
    std::mt19937 gen;
    int num_voxels;
    double sigma_spline;
    double spline_offset;
    std::vector<double> X_hat;
    std::vector<double> A;
    std::vector<double> t;

};



#endif //AFFINETRANSFORMATION_H
