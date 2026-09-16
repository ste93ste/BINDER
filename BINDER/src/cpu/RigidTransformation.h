#ifndef RIGID_TRANSFORMATION_H
#define RIGID_TRANSFORMATION_H

#include <vector>
#include <random>
#include "Transformation.h"
#include <Eigen/Dense>

class RigidTransformation final : public Transformation{
public:
    RigidTransformation(int Nx_v, int Ny_v, int Nz_v,
                       int threads, double sigma_spline, double spline_offset,
                       const double* R_init,
                       const double* t_init,
                       const double* voxel_pos);
    
    ~RigidTransformation() override;

    double constrain_transformation(const double* dream_locations,
                                  const double* voxel_pos,
                                  double* ds,
                                  double* final_locations,
                                  int* node_indices) override;

    void sample_transformation(const double* dream_locations,
                             const double* voxel_pos,
                             double* ds,
                             double* final_locations,
                             int* node_indices,
                             int sample_gamma) override;

    void set_generator(std::mt19937& gen) override;

private:
    int Nx_v, Ny_v, Nz_v;
    int N;
    int threads;
    int num_voxels;
    double sigma_spline;
    double spline_offset;
    
    std::vector<double> R;        
    std::vector<double> t;       
    std::vector<double> x_means;  
    std::vector<double> x_tilde;  
    
    std::mt19937 gen;
};

#endif // RIGID_TRANSFORMATION_H