#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <array>

/**
 * Computes the Jacobian determinant of a 2D deformation field.
 * The input field should contain 2 components (x,y) per voxel.
 *
 * @param field Pointer to the deformation field data (x,y components interleaved)
 * @param Nx Number of voxels in x dimension
 * @param Ny Number of voxels in y dimension
 * @param voxel_size_x Physical size of voxel in x direction
 * @param voxel_size_y Physical size of voxel in y direction
 * @return std::vector<double> Computed Jacobian determinant field
 * @throws std::invalid_argument if field is null or dimensions are invalid
 */
std::vector<double> compute_jacobian_determinant_2D(
    const double* field,
    int Nx,
    int Ny,
    double voxel_size_x,
    double voxel_size_y
);

/**
 * Computes the Jacobian determinant of a 3D deformation field.
 * The input field should contain 3 components (x,y,z) per voxel.
 *
 * @param field Pointer to the deformation field data (x,y,z components interleaved)
 * @param Nx Number of voxels in x dimension
 * @param Ny Number of voxels in y dimension
 * @param Nz Number of voxels in z dimension
 * @param voxel_size_x Physical size of voxel in x direction
 * @param voxel_size_y Physical size of voxel in y direction
 * @param voxel_size_z Physical size of voxel in z dimension
 * @return std::vector<double> Computed Jacobian determinant field
 * @throws std::invalid_argument if field is null or dimensions are invalid
 */
std::vector<double> compute_jacobian_determinant_3D(
    const double* field,
    int Nx,
    int Ny,
    int Nz,
    double voxel_size_x,
    double voxel_size_y,
    double voxel_size_z
);

/**
* Compute inverse of a 3d deformation or transformation field.
* TODO: proper
*/
std::vector<double> computeGridInverse(
    const std::vector<double>& grid,
    const std::array<int, 3>& shape,
    bool isDeformation = true,
    double membrane = 0.1,
    bool extrapolate = true
);

#endif //UTILS_H