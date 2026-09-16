#include "topology_correction_3D.cuh"
#include "FFT.cuh"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/count.h>
#include <iomanip>

__global__ void computeJacobianDeterminantKernel(
    const float* field,
    float* jacobian_determinant,
    const int Nx,
    const int Ny,
    const int Nz,
    const float vol_scale
) {
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;

    constexpr int N = 3;
    const int idx = x * Ny * Nz + y * Nz + z;  // Output index

    // Helper function for field access
    auto field_idx = [Ny, Nz, N](const int x, const int y, const int z) -> int {
        return (x * Ny * Nz + y * Nz + z) * N;
    };

    const int curr_idx = field_idx(x, y, z);
    float J_total = 0.0f;
    int valid_corners = 0;

    // Get current point coordinates
    const float wx000 = field[curr_idx];
    const float wy000 = field[curr_idx + 1];
    const float wz000 = field[curr_idx + 2];

    // Forward-forward-forward corner (x+1, y+1, z+1)
    if (x < Nx - 1 && y < Ny - 1 && z < Nz - 1) {
        const int idx100 = field_idx(x + 1, y, z);
        const int idx010 = field_idx(x, y + 1, z);
        const int idx001 = field_idx(x, y, z + 1);

        const float wx100 = field[idx100];
        const float wy100 = field[idx100 + 1];
        const float wz100 = field[idx100 + 2];
        const float wx010 = field[idx010];
        const float wy010 = field[idx010 + 1];
        const float wz010 = field[idx010 + 2];
        const float wx001 = field[idx001];
        const float wy001 = field[idx001 + 1];
        const float wz001 = field[idx001 + 2];

        const float Jfff = (
            (wx100 - wx000) * ((wy010 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz010 - wz000)) -
            (wx010 - wx000) * ((wy100 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz100 - wz000)) +
            (wx001 - wx000) * ((wy100 - wy000) * (wz010 - wz000) - (wy010 - wy000) * (wz100 - wz000))
        ) * vol_scale;
        J_total += Jfff;
        valid_corners++;
    }

    // Backward-forward-forward corner (x-1, y+1, z+1)
    if (x > 0 && y < Ny - 1 && z < Nz - 1) {
        const int idxm100 = field_idx(x - 1, y, z);
        const int idx010 = field_idx(x, y + 1, z);
        const int idx001 = field_idx(x, y, z + 1);

        const float wxm100 = field[idxm100];
        const float wym100 = field[idxm100 + 1];
        const float wzm100 = field[idxm100 + 2];
        const float wx010 = field[idx010];
        const float wy010 = field[idx010 + 1];
        const float wz010 = field[idx010 + 2];
        const float wx001 = field[idx001];
        const float wy001 = field[idx001 + 1];
        const float wz001 = field[idx001 + 2];

        const float Jbff = (
            (wx000 - wxm100) * ((wy010 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz010 - wz000)) -
            (wx010 - wx000) * ((wy000 - wym100) * (wz001 - wz000) - (wy001 - wy000) * (wz000 - wzm100)) +
            (wx001 - wx000) * ((wy000 - wym100) * (wz010 - wz000) - (wy010 - wy000) * (wz000 - wzm100))
        ) * vol_scale;
        J_total += Jbff;
        valid_corners++;
    }

    // Forward-backward-forward corner (x+1, y-1, z+1)
    if (x < Nx - 1 && y > 0 && z < Nz - 1) {
        const int idx100 = field_idx(x + 1, y, z);
        const int idx0m10 = field_idx(x, y - 1, z);
        const int idx001 = field_idx(x, y, z + 1);

        const float wx100 = field[idx100];
        const float wy100 = field[idx100 + 1];
        const float wz100 = field[idx100 + 2];
        const float wx0m10 = field[idx0m10];
        const float wy0m10 = field[idx0m10 + 1];
        const float wz0m10 = field[idx0m10 + 2];
        const float wx001 = field[idx001];
        const float wy001 = field[idx001 + 1];
        const float wz001 = field[idx001 + 2];

        const float Jfbf = (
            (wx100 - wx000) * ((wy000 - wy0m10) * (wz001 - wz000) - (wy001 - wy000) * (wz000 - wz0m10)) -
            (wx000 - wx0m10) * ((wy100 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz100 - wz000)) +
            (wx001 - wx000) * ((wy100 - wy000) * (wz000 - wz0m10) - (wy000 - wy0m10) * (wz100 - wz000))
        ) * vol_scale;
        J_total += Jfbf;
        valid_corners++;
    }

    // Forward-forward-backward corner (x+1, y+1, z-1)
    if (x < Nx - 1 && y < Ny - 1 && z > 0) {
        const int idx100 = field_idx(x + 1, y, z);
        const int idx010 = field_idx(x, y + 1, z);
        const int idx00m1 = field_idx(x, y, z - 1);

        const float wx100 = field[idx100];
        const float wy100 = field[idx100 + 1];
        const float wz100 = field[idx100 + 2];
        const float wx010 = field[idx010];
        const float wy010 = field[idx010 + 1];
        const float wz010 = field[idx010 + 2];
        const float wx00m1 = field[idx00m1];
        const float wy00m1 = field[idx00m1 + 1];
        const float wz00m1 = field[idx00m1 + 2];

        const float Jffb = (
            (wx100 - wx000) * ((wy010 - wy000) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz010 - wz000)) -
            (wx010 - wx000) * ((wy100 - wy000) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz100 - wz000)) +
            (wx000 - wx00m1) * ((wy100 - wy000) * (wz010 - wz000) - (wy010 - wy000) * (wz100 - wz000))
        ) * vol_scale;
        J_total += Jffb;
        valid_corners++;
    }

    // Backward-backward-forward corner (x-1, y-1, z+1)
    if (x > 0 && y > 0 && z < Nz - 1) {
        const int idxm100 = field_idx(x - 1, y, z);
        const int idx0m10 = field_idx(x, y - 1, z);
        const int idx001 = field_idx(x, y, z + 1);

        const float wxm100 = field[idxm100];
        const float wym100 = field[idxm100 + 1];
        const float wzm100 = field[idxm100 + 2];
        const float wx0m10 = field[idx0m10];
        const float wy0m10 = field[idx0m10 + 1];
        const float wz0m10 = field[idx0m10 + 2];
        const float wx001 = field[idx001];
        const float wy001 = field[idx001 + 1];
        const float wz001 = field[idx001 + 2];

        const float Jbbf = (
            (wx000 - wxm100) * ((wy000 - wy0m10) * (wz001 - wz000) - (wy001 - wy000) * (wz000 - wz0m10)) -
            (wx000 - wx0m10) * ((wy000 - wym100) * (wz001 - wz000) - (wy001 - wy000) * (wz000 - wzm100)) +
            (wx001 - wx000) * ((wy000 - wym100) * (wz000 - wz0m10) - (wy000 - wy0m10) * (wz000 - wzm100))
        ) * vol_scale;
        J_total += Jbbf;
        valid_corners++;
    }

    // Backward-forward-backward corner (x-1, y+1, z-1)
    if (x > 0 && y < Ny - 1 && z > 0) {
        const int idxm100 = field_idx(x - 1, y, z);
        const int idx010 = field_idx(x, y + 1, z);
        const int idx00m1 = field_idx(x, y, z - 1);

        const float wxm100 = field[idxm100];
        const float wym100 = field[idxm100 + 1];
        const float wzm100 = field[idxm100 + 2];
        const float wx010 = field[idx010];
        const float wy010 = field[idx010 + 1];
        const float wz010 = field[idx010 + 2];
        const float wx00m1 = field[idx00m1];
        const float wy00m1 = field[idx00m1 + 1];
        const float wz00m1 = field[idx00m1 + 2];

        const float Jbfb = (
            (wx000 - wxm100) * ((wy010 - wy000) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz010 - wz000)) -
            (wx010 - wx000) * ((wy000 - wym100) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz000 - wzm100)) +
            (wx000 - wx00m1) * ((wy000 - wym100) * (wz010 - wz000) - (wy010 - wy000) * (wz000 - wzm100))
        ) * vol_scale;
        J_total += Jbfb;
        valid_corners++;
    }

    // Forward-backward-backward corner (x+1, y-1, z-1)
    if (x < Nx - 1 && y > 0 && z > 0) {
        const int idx100 = field_idx(x + 1, y, z);
        const int idx0m10 = field_idx(x, y - 1, z);
        const int idx00m1 = field_idx(x, y, z - 1);

        const float wx100 = field[idx100];
        const float wy100 = field[idx100 + 1];
        const float wz100 = field[idx100 + 2];
        const float wx0m10 = field[idx0m10];
        const float wy0m10 = field[idx0m10 + 1];
        const float wz0m10 = field[idx0m10 + 2];
        const float wx00m1 = field[idx00m1];
        const float wy00m1 = field[idx00m1 + 1];
        const float wz00m1 = field[idx00m1 + 2];

        const float Jfbb = (
            (wx100 - wx000) * ((wy000 - wy0m10) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz000 - wz0m10)) -
            (wx000 - wx0m10) * ((wy100 - wy000) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz100 - wz000)) +
            (wx000 - wx00m1) * ((wy100 - wy000) * (wz000 - wz0m10) - (wy000 - wy0m10) * (wz100 - wz000))
        ) * vol_scale;
        J_total += Jfbb;
        valid_corners++;
    }

    // Backward-backward-backward corner (x-1, y-1, z-1)
    if (x > 0 && y > 0 && z > 0) {
        const int idxm100 = field_idx(x - 1, y, z);
        const int idx0m10 = field_idx(x, y - 1, z);
        const int idx00m1 = field_idx(x, y, z - 1);

        const float wxm100 = field[idxm100];
        const float wym100 = field[idxm100 + 1];
        const float wzm100 = field[idxm100 + 2];
        const float wx0m10 = field[idx0m10];
        const float wy0m10 = field[idx0m10 + 1];
        const float wz0m10 = field[idx0m10 + 2];
        const float wx00m1 = field[idx00m1];
        const float wy00m1 = field[idx00m1 + 1];
        const float wz00m1 = field[idx00m1 + 2];

        const float Jbbb = (
            (wx000 - wxm100) * ((wy000 - wy0m10) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz000 - wz0m10)) -
            (wx000 - wx0m10) * ((wy000 - wym100) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz000 - wzm100)) +
            (wx000 - wx00m1) * ((wy000 - wym100) * (wz000 - wz0m10) - (wy000 - wy0m10) * (wz000 - wzm100))
        ) * vol_scale;
        J_total += Jbbb;
        valid_corners++;
    }

    // Average the valid corner Jacobians
    jacobian_determinant[idx] = valid_corners > 0 ? J_total / valid_corners : 1.0f;
}

__global__ void computeCornerJacobiansKernel(const float* deformedVoxelPositions, float* J_corner,
                                             int Nx_v, int Ny_v, int Nz_v,
                                             float dx, float dy, float dz) {
    const int z = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx_v - 1 || y >= Ny_v - 1 || z >= Nz_v - 1) return;

    const int N = 3;
    const int corners = 8;
    const float vol_scale = 1.0f / (dx * dy * dz);

    // Calculate indices for 1D array - directly matching CPU version
    const int idx000 = (x * Ny_v * Nz_v + y * Nz_v + z) * N;
    const int idx001 = (x * Ny_v * Nz_v + y * Nz_v + (z + 1)) * N;
    const int idx010 = (x * Ny_v * Nz_v + (y + 1) * Nz_v + z) * N;
    const int idx011 = (x * Ny_v * Nz_v + (y + 1) * Nz_v + (z + 1)) * N;
    const int idx100 = ((x + 1) * Ny_v * Nz_v + y * Nz_v + z) * N;
    const int idx101 = ((x + 1) * Ny_v * Nz_v + y * Nz_v + (z + 1)) * N;
    const int idx110 = ((x + 1) * Ny_v * Nz_v + (y + 1) * Nz_v + z) * N;
    const int idx111 = ((x + 1) * Ny_v * Nz_v + (y + 1) * Nz_v + (z + 1)) * N;

    // X, Y, Z directions - matching CPU component access pattern
    const float wx000 = deformedVoxelPositions[idx000];
    const float wx001 = deformedVoxelPositions[idx001];
    const float wx010 = deformedVoxelPositions[idx010];
    const float wx011 = deformedVoxelPositions[idx011];
    const float wx100 = deformedVoxelPositions[idx100];
    const float wx101 = deformedVoxelPositions[idx101];
    const float wx110 = deformedVoxelPositions[idx110];
    const float wx111 = deformedVoxelPositions[idx111];

    const float wy000 = deformedVoxelPositions[idx000 + 1];
    const float wy001 = deformedVoxelPositions[idx001 + 1];
    const float wy010 = deformedVoxelPositions[idx010 + 1];
    const float wy011 = deformedVoxelPositions[idx011 + 1];
    const float wy100 = deformedVoxelPositions[idx100 + 1];
    const float wy101 = deformedVoxelPositions[idx101 + 1];
    const float wy110 = deformedVoxelPositions[idx110 + 1];
    const float wy111 = deformedVoxelPositions[idx111 + 1];

    const float wz000 = deformedVoxelPositions[idx000 + 2];
    const float wz001 = deformedVoxelPositions[idx001 + 2];
    const float wz010 = deformedVoxelPositions[idx010 + 2];
    const float wz011 = deformedVoxelPositions[idx011 + 2];
    const float wz100 = deformedVoxelPositions[idx100 + 2];
    const float wz101 = deformedVoxelPositions[idx101 + 2];
    const float wz110 = deformedVoxelPositions[idx110 + 2];
    const float wz111 = deformedVoxelPositions[idx111 + 2];

    // Output index for corner Jacobians - matching CPU indexing
    const int idx = x * (Ny_v - 1) * (Nz_v - 1) * corners + y * (Nz_v - 1) * corners + z * corners;

    // Compute Jfff - Forward-forward-forward corner
    float Jfff = (wx100 - wx000) * ((wy010 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz010 - wz000)) -
                (wx010 - wx000) * ((wy100 - wy000) * (wz001 - wz000) - (wy001 - wy000) * (wz100 - wz000)) +
                (wx001 - wx000) * ((wy100 - wy000) * (wz010 - wz000) - (wy010 - wy000) * (wz100 - wz000));
    Jfff *= vol_scale;

    // Compute Jbff - Backward-forward-forward corner
    float Jbff = (wx100 - wx000) * ((wy110 - wy100) * (wz101 - wz100) - (wy101 - wy100) * (wz110 - wz100)) -
                (wx110 - wx100) * ((wy100 - wy000) * (wz101 - wz100) - (wy101 - wy100) * (wz100 - wz000)) +
                (wx101 - wx100) * ((wy100 - wy000) * (wz110 - wz100) - (wy110 - wy100) * (wz100 - wz000));
    Jbff *= vol_scale;

    // Compute Jfbf - Forward-backward-forward corner
    float Jfbf = (wx110 - wx010) * ((wy010 - wy000) * (wz011 - wz010) - (wy011 - wy010) * (wz010 - wz000)) -
                (wx010 - wx000) * ((wy110 - wy010) * (wz011 - wz010) - (wy011 - wy010) * (wz110 - wz010)) +
                (wx011 - wx010) * ((wy110 - wy010) * (wz010 - wz000) - (wy010 - wy000) * (wz110 - wz010));
    Jfbf *= vol_scale;

    // Compute Jffb - Forward-forward-backward corner
    float Jffb = (wx101 - wx001) * ((wy011 - wy001) * (wz001 - wz000) - (wy001 - wy000) * (wz011 - wz001)) -
                (wx011 - wx001) * ((wy101 - wy001) * (wz001 - wz000) - (wy001 - wy000) * (wz101 - wz001)) +
                (wx001 - wx000) * ((wy101 - wy001) * (wz011 - wz001) - (wy011 - wy001) * (wz101 - wz001));
    Jffb *= vol_scale;

    // Compute Jfbb - Forward-backward-backward corner
    float Jfbb = (wx111 - wx011) * ((wy011 - wy001) * (wz011 - wz010) - (wy011 - wy010) * (wz011 - wz001)) -
                (wx011 - wx001) * ((wy111 - wy011) * (wz011 - wz010) - (wy011 - wy010) * (wz111 - wz011)) +
                (wx011 - wx010) * ((wy111 - wy011) * (wz011 - wz001) - (wy011 - wy001) * (wz111 - wz011));
    Jfbb *= vol_scale;

    // Compute Jbfb - Backward-forward-backward corner
    float Jbfb = (wx101 - wx001) * ((wy111 - wy101) * (wz101 - wz100) - (wy101 - wy100) * (wz111 - wz101)) -
                (wx111 - wx101) * ((wy101 - wy001) * (wz101 - wz100) - (wy101 - wy100) * (wz101 - wz001)) +
                (wx101 - wx100) * ((wy101 - wy001) * (wz111 - wz101) - (wy111 - wy101) * (wz101 - wz001));
    Jbfb *= vol_scale;

    // Compute Jbbf - Backward-backward-forward corner
    float Jbbf = (wx110 - wx010) * ((wy110 - wy100) * (wz111 - wz110) - (wy111 - wy110) * (wz110 - wz100)) -
                (wx110 - wx100) * ((wy110 - wy010) * (wz111 - wz110) - (wy111 - wy110) * (wz110 - wz010)) +
                (wx111 - wx110) * ((wy110 - wy010) * (wz110 - wz100) - (wy110 - wy100) * (wz110 - wz010));
    Jbbf *= vol_scale;

    // Compute Jbbb - Backward-backward-backward corner
    float Jbbb = (wx111 - wx011) * ((wy111 - wy101) * (wz111 - wz110) - (wy111 - wy110) * (wz111 - wz101)) -
                (wx111 - wx101) * ((wy111 - wy011) * (wz111 - wz110) - (wy111 - wy110) * (wz111 - wz011)) +
                (wx111 - wx110) * ((wy111 - wy011) * (wz111 - wz101) - (wy111 - wy101) * (wz111 - wz011));
    Jbbb *= vol_scale;

    // Store corner Jacobians in J_corner
    J_corner[idx + 0] = Jfff;
    J_corner[idx + 1] = Jbff;
    J_corner[idx + 2] = Jfbf;
    J_corner[idx + 3] = Jffb;
    J_corner[idx + 4] = Jfbb;
    J_corner[idx + 5] = Jbfb;
    J_corner[idx + 6] = Jbbf;
    J_corner[idx + 7] = Jbbb;
}


// CUDA kernel for computing gradients
__global__ void computeGradientsKernel(const float* deformedVoxelPositions, float* grad,
                                       int Nx_v, int Ny_v, int Nz_v, float dx, float dy, float dz) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx_v || y >= Ny_v || z >= Nz_v) return;

    const int N = 3;
    const int base_idx = (x * Ny_v * Nz_v * N * N) + (y * Nz_v * N * N) + (z * N * N);

    if (x < Nx_v - 1 && y < Ny_v - 1 && z < Nz_v - 1) {
        grad[base_idx + 0] = (deformedVoxelPositions[(x + 1) * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
        grad[base_idx + 1] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + (y + 1) * Nz_v * N + z * N + 0] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
        grad[base_idx + 2] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + (z + 1) * N + 0] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
        grad[base_idx + 3] = (deformedVoxelPositions[(x + 1) * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
        grad[base_idx + 4] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + (y + 1) * Nz_v * N + z * N + 1] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
        grad[base_idx + 5] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + (z + 1) * N + 1] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
        grad[base_idx + 6] = (deformedVoxelPositions[(x + 1) * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
        grad[base_idx + 7] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + (y + 1) * Nz_v * N + z * N + 2] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
        grad[base_idx + 8] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + (z + 1) * N + 2] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
    } else {
        int x2 = x + 1;
        int y2 = y + 1;
        int z2 = z + 1;

        if (x2 > Nx_v - 1) {
            x2 -= Nx_v;  // Wrap around to 0
        }
        if (y2 > Ny_v - 1) {
            y2 -= Ny_v;  // Wrap around to 0
        }
        if (z2 > Nz_v - 1) {
            z2 -= Nz_v;  // Wrap around to 0
        }

        grad[base_idx + 0] = (deformedVoxelPositions[x2 * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
        grad[base_idx + 1] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y2 * Nz_v * N + z * N + 0] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
        grad[base_idx + 2] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z2 * N + 0] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 0]) / dx;
        grad[base_idx + 3] = (deformedVoxelPositions[x2 * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
        grad[base_idx + 4] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y2 * Nz_v * N + z * N + 1] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
        grad[base_idx + 5] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z2 * N + 1] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 1]) / dy;
        grad[base_idx + 6] = (deformedVoxelPositions[x2 * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
        grad[base_idx + 7] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y2 * Nz_v * N + z * N + 2] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
        grad[base_idx + 8] = (deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z2 * N + 2] -
                              deformedVoxelPositions[x * Ny_v * Nz_v * N + y * Nz_v * N + z * N + 2]) / dz;
    }
}

__device__ float computeDeterminant3x3(const float* J) {
    // For a 3x3 matrix:
    // | a b c |
    // | d e f |
    // | g h i |

    // The determinant is: a(ei - fh) - b(di - fg) + c(dh - eg)

    const float a = J[0], b = J[1], c = J[2];
    const float d = J[3], e = J[4], f = J[5];
    const float g = J[6], h = J[7], i = J[8];

    return a * (e * i - f * h) -
           b * (d * i - f * g) +
           c * (d * h - e * g);
}

__global__ void fillFFTInputsKernel(cufftDoubleComplex* d_fft_in_x_ptr,
                                   cufftDoubleComplex* d_fft_in_y_ptr,
                                   cufftDoubleComplex* d_fft_in_z_ptr,
                                   const float* d_grad_ptr,
                                   int Nx, int Ny, int Nz, int N, int n) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;

    const int idx = x * Ny * Nz + y * Nz + z;
    const int grad_idx = x * (Ny * Nz * N * N) + y * (Nz * N * N) + z * (N * N) + n * N;

    d_fft_in_x_ptr[idx].x = d_grad_ptr[grad_idx + 0];
    d_fft_in_x_ptr[idx].y = 0.0;

    d_fft_in_y_ptr[idx].x = d_grad_ptr[grad_idx + 1];
    d_fft_in_y_ptr[idx].y = 0.0;

    d_fft_in_z_ptr[idx].x = d_grad_ptr[grad_idx + 2];
    d_fft_in_z_ptr[idx].y = 0.0;
}

__global__ void computeIFFTInputKernel(
    const cufftDoubleComplex* d_fft_out_x_ptr,
    const cufftDoubleComplex* d_fft_out_y_ptr,
    const cufftDoubleComplex* d_fft_out_z_ptr,
    cufftDoubleComplex* d_ifft_in_ptr,
    int Nx, int Ny, int Nz) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;

    const int idx = x * Ny * Nz + y * Nz + z;

    const float cosz = cosf(2.0f * M_PI * z / static_cast<float>(Nz));
    const float sinz = sinf(2.0f * M_PI * z / static_cast<float>(Nz));
    const float cosy = cosf(2.0f * M_PI * y / static_cast<float>(Ny));
    const float siny = sinf(2.0f * M_PI * y / static_cast<float>(Ny));
    const float cosx = cosf(2.0f * M_PI * x / static_cast<float>(Nx));
    const float sinx = sinf(2.0f * M_PI * x / static_cast<float>(Nx));

    const float norm = 6.0f - 2.0f * cosz - 2.0f * cosx - 2.0f * cosy;

    if (fabsf(norm) > 1e-10f) {
        float dotprodreal = d_fft_out_x_ptr[idx].x * (cosx - 1.0f) +
                           d_fft_out_x_ptr[idx].y * sinx +
                           d_fft_out_y_ptr[idx].x * (cosy - 1.0f) +
                           d_fft_out_y_ptr[idx].y * siny +
                           d_fft_out_z_ptr[idx].x * (cosz - 1.0f) +
                           d_fft_out_z_ptr[idx].y * sinz;

        float dotprodimag = d_fft_out_x_ptr[idx].y * (cosx - 1.0f) -
                           d_fft_out_x_ptr[idx].x * sinx +
                           d_fft_out_y_ptr[idx].y * (cosy - 1.0f) -
                           d_fft_out_y_ptr[idx].x * siny +
                           d_fft_out_z_ptr[idx].y * (cosz - 1.0f) -
                           d_fft_out_z_ptr[idx].x * sinz;

        d_ifft_in_ptr[idx].x = dotprodreal / norm;
        d_ifft_in_ptr[idx].y = dotprodimag / norm;
    } else {
        d_ifft_in_ptr[idx].x = 0.0f;
        d_ifft_in_ptr[idx].y = 0.0f;
    }
}

// Helper function to get offsets for Jacobian computation
__device__ void get_jac_off_3D_device(const int jac_num, int* xoff, int* yoff, int* zoff) {
    if (jac_num == 0) {  // Jfff
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 1) {  // Jbff
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 2) {  // Jfbf
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 3) {  // Jffb
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    } else if (jac_num == 4) {  // Jfbb
        xoff[0] = 0; xoff[1] = 0; xoff[2] = 0;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    } else if (jac_num == 5) {  // Jbfb
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 0; yoff[1] = 0; yoff[2] = 0;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    } else if (jac_num == 6) {  // Jbbf
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 0; zoff[1] = 0; zoff[2] = 0;
    } else if (jac_num == 7) {  // Jbbb
        xoff[0] = 0; xoff[1] = 1; xoff[2] = 1;
        yoff[0] = 1; yoff[1] = 0; yoff[2] = 1;
        zoff[0] = 1; zoff[1] = 1; zoff[2] = 0;
    }
}

// CUDA kernel for limiting gradients
__global__ void limitGradientsKernel(float* grad, const float* J_corner,
                                     int Nx_v, int Ny_v, int Nz_v, float e1, float e2) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= Nx_v - 1 || y >= Ny_v - 1 || z >= Nz_v - 1) return;

    const int N = 3;
    const int corners = 8;
    const int idx = x * ((Ny_v - 1) * (Nz_v - 1) * corners) + y * ((Nz_v - 1) * corners) + z * corners;

    // Process each corner
    for (int jacnum = 0; jacnum < corners; jacnum++) {
        float jacobian_val = J_corner[idx + jacnum];
        if (jacobian_val < e1 || jacobian_val > e2) {
            // Get offsets for this corner
            int xoff[3], yoff[3], zoff[3];
            get_jac_off_3D_device(jacnum, xoff, yoff, zoff);

            float J[9] = {0};  // 3x3 matrix
            float J0[9] = {0}; // Identity matrix
            float Jnew[9] = {0}; // For intermediate calculations
            J0[0] = J0[4] = J0[8] = 1.0;

            // Fill Jacobian matrix
            for (int n1 = 0; n1 < N; n1++) {
                for (int n2 = 0; n2 < N; n2++) {
                    const int grad_idx = (x + xoff[n2]) * (Ny_v * Nz_v * N * N) +
                                         (y + yoff[n2]) * (Nz_v * N * N) +
                                         (z + zoff[n2]) * N * N +
                                         N * n1 + n2;
                    J[n1 * N + n2] = grad[grad_idx];
                }
            }

            // Calculate initial determinant
            float detJ = computeDeterminant3x3(J);

            // Iteratively adjust gradients
            float alpha = 0.0;

            while (detJ > e2 || detJ < e1) {
                alpha += 0.1;
                if (alpha > 1.0) alpha = 1.0;

                // Interpolate between J and J0
                for (int n1 = 0; n1 < N; n1++) {
                    for (int n2 = 0; n2 < N; n2++) {
                        Jnew[n1 * N + n2] = (1 - alpha) * J[n1 * N + n2] + alpha * J0[n1 * N + n2];
                    }
                }

                // Recompute determinant
                detJ = computeDeterminant3x3(Jnew);

                if (alpha >= 1.0) break; // Prevent infinite loop
            }

            // Final adjustment with a slight increase in alpha
            alpha += 0.1;
            if (alpha > 1.0) alpha = 1.0;

            // Apply the final corrections to gradients
            for (int n1 = 0; n1 < N; n1++) {
                for (int n2 = 0; n2 < N; n2++) {
                    const int grad_idx = (x + xoff[n2]) * (Ny_v * Nz_v * N * N) +
                                       (y + yoff[n2]) * (Nz_v * N * N) +
                                       (z + zoff[n2]) * N * N +
                                       N * n1 + n2;
                    grad[grad_idx] = (1 - alpha) * J[n1 * N + n2] + alpha * J0[n1 * N + n2];
                }
            }
        }
    }
}

__global__ void extractComponentsKernel(const float* d_positions_ptr,
                                       float* d_positions_x_ptr,
                                       float* d_positions_y_ptr,
                                       float* d_positions_z_ptr,
                                       int Nx, int Ny, int Nz) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;

    const int idx = x * (Ny * Nz) + y * Nz + z;
    const int pos_idx = idx * 3;

    d_positions_x_ptr[idx] = d_positions_ptr[pos_idx];
    d_positions_y_ptr[idx] = d_positions_ptr[pos_idx + 1];
    d_positions_z_ptr[idx] = d_positions_ptr[pos_idx + 2];
}

__global__ void updatePositionsKernel(
    float* d_positions_ptr,
    const cufftDoubleComplex* d_ifft_out_ptr,
    int Nx, int Ny, int Nz, int N, int n) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;

    const int pos_idx = x * Ny * Nz * N + y * Nz * N + z * N + n;
    const int fft_idx = x * Ny * Nz + y * Nz + z;

    // Update deformed voxel positions
    d_positions_ptr[pos_idx] = d_ifft_out_ptr[fft_idx].x / static_cast<float>(Nx * Ny * Nz);
}

// CUDA kernel for scaling positions
__global__ void scalePositionsKernel(float* positions,
                                     const float voxel_size_x, const float voxel_size_y, const float voxel_size_z,
                                     int Nx_v, int Ny_v, int Nz_v, bool multiply) {

    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx_v && y < Ny_v && z < Nz_v) {
        const int idx = (x * Ny_v * Nz_v + y * Nz_v + z) * 3;
        if (multiply) {
            positions[idx + 0] *= voxel_size_x;
            positions[idx + 1] *= voxel_size_y;
            positions[idx + 2] *= voxel_size_z;
        } else {
            positions[idx + 0] /= voxel_size_x;
            positions[idx + 1] /= voxel_size_y;
            positions[idx + 2] /= voxel_size_z;
        }
    }
}

__global__ void rescalePositionsKernel(float* d_positions_ptr,
                                      int Nx, int Ny, int Nz, int N,
                                      float voxel_size_x, float voxel_size_y, float voxel_size_z,
                                      float mean_x, float mean_y, float mean_z) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= Nx || y >= Ny || z >= Nz) return;

    const int base_idx = x * Ny * Nz * N + y * Nz * N + z * N;

    // Scale each coordinate and add the mean back, exactly matching the CPU implementation
    // X coordinate
    d_positions_ptr[base_idx + 0] *= voxel_size_x;
    d_positions_ptr[base_idx + 0] += mean_x;

    // Y coordinate
    d_positions_ptr[base_idx + 1] *= voxel_size_y;
    d_positions_ptr[base_idx + 1] += mean_y;

    // Z coordinate
    d_positions_ptr[base_idx + 2] *= voxel_size_z;
    d_positions_ptr[base_idx + 2] += mean_z;
}

void topology_correction_3D::to_grid() const {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (Nz_v + blockSize.x - 1) / blockSize.x,
        (Ny_v + blockSize.y - 1) / blockSize.y,
        (Nx_v + blockSize.z - 1) / blockSize.z
    );

    scalePositionsKernel<<<gridSize, blockSize>>>(
        d_positions_ptr,
        voxel_size_x, voxel_size_y, voxel_size_z,
        Nx_v, Ny_v, Nz_v, false);  // false for division
}

void topology_correction_3D::to_physical() const {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (Nz_v + blockSize.x - 1) / blockSize.x,
        (Ny_v + blockSize.y - 1) / blockSize.y,
        (Nx_v + blockSize.z - 1) / blockSize.z
    );

    scalePositionsKernel<<<gridSize, blockSize>>>(
        d_positions_ptr,
        voxel_size_x, voxel_size_y, voxel_size_z,
        Nx_v, Ny_v, Nz_v, true);  // true for multiplication
}

// Host function to launch the kernel
thrust::device_vector<float> compute_jacobian_determinant_3D(
    const float* field,
    const int Nx,
    const int Ny,
    const int Nz,
    const float voxel_size_x,
    const float voxel_size_y,
    const float voxel_size_z
) {
    // Allocate device memory for result
    thrust::device_vector<float> jacobian_determinant(Nx * Ny * Nz);

    // Calculate volume scale
    const float vol_scale = 1.0f / (voxel_size_x * voxel_size_y * voxel_size_z);

    // Set up grid and block dimensions
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (Nz + blockSize.x - 1) / blockSize.x,
        (Ny + blockSize.y - 1) / blockSize.y,
        (Nx + blockSize.z - 1) / blockSize.z
    );

    // Launch kernel
    computeJacobianDeterminantKernel<<<gridSize, blockSize>>>(
        field,
        thrust::raw_pointer_cast(jacobian_determinant.data()),
        Nx, Ny, Nz,
        vol_scale
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return jacobian_determinant;
}

std::pair<int, int> topology_correction_3D::compute_corner_jacobians_3D(float* J_corner) const {


    // Configure kernel launch parameters
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (Nz_v - 1 + blockSize.x - 1) / blockSize.x,
        (Ny_v - 1 + blockSize.y - 1) / blockSize.y,
        (Nx_v - 1 + blockSize.z - 1) / blockSize.z
    );

    // Launch kernel to compute corner Jacobians
    computeCornerJacobiansKernel<<<gridSize, blockSize>>>(d_positions_ptr, d_corner_ptr,
                                                          Nx_v, Ny_v, Nz_v,
                                                          voxel_size_x, voxel_size_y, voxel_size_z);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return {0, 0}; // Return empty result on error
    }

    // Synchronize to make sure kernel is finished
    cudaDeviceSynchronize();

    // Create Thrust device vector wrapper around the corner Jacobians
    const size_t num_corners = (Nx_v - 1) * (Ny_v - 1) * (Nz_v - 1) * corners;
    thrust::device_ptr<float> d_corner_thrust(d_corner_ptr);

    // Find min and max Jacobian values using Thrust
    auto minmax = thrust::minmax_element(d_corner_thrust, d_corner_thrust + num_corners);
    float jmin = *minmax.first;
    float jmax = *minmax.second;

    // Count Jacobians below e1 and above e2
    int jmin_n = thrust::count_if(d_corner_thrust, d_corner_thrust + num_corners,
                                 thrust::placeholders::_1 < e1);
    int jmax_n = thrust::count_if(d_corner_thrust, d_corner_thrust + num_corners,
                                 thrust::placeholders::_1 > e2);

    // Output results
    std::cout << "Corner Jacobians" << std::endl;
    std::cout << "#J_min: " << jmin_n << " #J_max: " << jmax_n << std::endl;
    std::cout << "min_J: " << jmin << " max_J: " << jmax << std::endl;

    return {jmin_n, jmax_n};
}

// Main implementation
topology_correction_3D::topology_correction_3D() = default;
topology_correction_3D::~topology_correction_3D() = default;

void topology_correction_3D::integrate_gradient_field_3D(const float* d_grad_ptr) const {
    const int Nx = Nx_v;
    const int Ny = Ny_v;
    const int Nz = Nz_v;
    const size_t size_complex = Nx * Ny * Nz;

    // Calculate mean values directly using thrust
    thrust::device_vector<float> positions_x(Nx * Ny * Nz);
    thrust::device_vector<float> positions_y(Nx * Ny * Nz);
    thrust::device_vector<float> positions_z(Nx * Ny * Nz);

    // Get raw pointers for device code
    float* d_positions_x_ptr = thrust::raw_pointer_cast(positions_x.data());
    float* d_positions_y_ptr = thrust::raw_pointer_cast(positions_y.data());
    float* d_positions_z_ptr = thrust::raw_pointer_cast(positions_z.data());

    // Extract components in a safer way - use a kernel instead of thrust::for_each
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (Nz + blockSize.x - 1) / blockSize.x,
        (Ny + blockSize.y - 1) / blockSize.y,
        (Nx + blockSize.z - 1) / blockSize.z
    );

    extractComponentsKernel<<<gridSize, blockSize>>>(
        d_positions_ptr, d_positions_x_ptr, d_positions_y_ptr, d_positions_z_ptr,
        Nx, Ny, Nz
    );
    cudaDeviceSynchronize();

    // Calculate means using thrust
    float mean_x = thrust::reduce(positions_x.begin(), positions_x.end(), 0.0f) / (Nx * Ny * Nz);
    float mean_y = thrust::reduce(positions_y.begin(), positions_y.end(), 0.0f) / (Nx * Ny * Nz);
    float mean_z = thrust::reduce(positions_z.begin(), positions_z.end(), 0.0f) / (Nx * Ny * Nz);

    // Allocate device memory for FFT operations
    thrust::device_vector<cufftDoubleComplex> d_fft_in_x(size_complex);
    thrust::device_vector<cufftDoubleComplex> d_fft_out_x(size_complex);
    thrust::device_vector<cufftDoubleComplex> d_fft_in_y(size_complex);
    thrust::device_vector<cufftDoubleComplex> d_fft_out_y(size_complex);
    thrust::device_vector<cufftDoubleComplex> d_fft_in_z(size_complex);
    thrust::device_vector<cufftDoubleComplex> d_fft_out_z(size_complex);
    thrust::device_vector<cufftDoubleComplex> d_ifft_in(size_complex);
    thrust::device_vector<cufftDoubleComplex> d_ifft_out(size_complex);

    // Get raw pointers for kernel calls
    cufftDoubleComplex* d_fft_in_x_ptr = thrust::raw_pointer_cast(d_fft_in_x.data());
    cufftDoubleComplex* d_fft_out_x_ptr = thrust::raw_pointer_cast(d_fft_out_x.data());
    cufftDoubleComplex* d_fft_in_y_ptr = thrust::raw_pointer_cast(d_fft_in_y.data());
    cufftDoubleComplex* d_fft_out_y_ptr = thrust::raw_pointer_cast(d_fft_out_y.data());
    cufftDoubleComplex* d_fft_in_z_ptr = thrust::raw_pointer_cast(d_fft_in_z.data());
    cufftDoubleComplex* d_fft_out_z_ptr = thrust::raw_pointer_cast(d_fft_out_z.data());
    cufftDoubleComplex* d_ifft_in_ptr = thrust::raw_pointer_cast(d_ifft_in.data());
    cufftDoubleComplex* d_ifft_out_ptr = thrust::raw_pointer_cast(d_ifft_out.data());

    // Process each coordinate (X, Y, Z)
    for (int n = 0; n < N; ++n) {
        // Fill in FFT input data from gradients
        fillFFTInputsKernel<<<gridSize, blockSize>>>(
            d_fft_in_x_ptr,
            d_fft_in_y_ptr,
            d_fft_in_z_ptr,
            d_grad_ptr,
            Nx, Ny, Nz, N, n
        );
        cudaDeviceSynchronize();

        // Execute FFTs - using raw pointers to match FFT function signatures
        fft_x->fft(d_fft_in_x_ptr, d_fft_out_x_ptr);
        fft_y->fft(d_fft_in_y_ptr, d_fft_out_y_ptr);
        fft_z->fft(d_fft_in_z_ptr, d_fft_out_z_ptr);

        // Apply Fourier-based integration
        computeIFFTInputKernel<<<gridSize, blockSize>>>(
            d_fft_out_x_ptr,
            d_fft_out_y_ptr,
            d_fft_out_z_ptr,
            d_ifft_in_ptr,
            Nx, Ny, Nz
        );
        cudaDeviceSynchronize();

        // Execute inverse FFT - using raw pointers
        fft_x->ifft(d_ifft_in_ptr, d_ifft_out_ptr);

        // Update deformed voxel positions with integrated data
        updatePositionsKernel<<<gridSize, blockSize>>>(
            d_positions_ptr,
            d_ifft_out_ptr,
            Nx, Ny, Nz, N, n
        );
        cudaDeviceSynchronize();
    }

	rescalePositionsKernel<<<gridSize, blockSize>>>(
    	d_positions_ptr,
    	Nx, Ny, Nz, N,
    	voxel_size_x, voxel_size_y, voxel_size_z,
    	mean_x, mean_y, mean_z
	);
    cudaDeviceSynchronize();
}

// Modified correct_topology method to avoid using pointers for mean values
float* topology_correction_3D::correct_topology(const float* deformedVoxelPositions, const float e1, const float e2,
    const int max_outer_it, const int max_inner_it, int threads, const float* voxel_sizes, int Nx_v, int Ny_v, int Nz_v) {

    this->e1 = e1;
    this->e2 = e2;
    this->Nx_v = Nx_v;
    this->Ny_v = Ny_v;
    this->Nz_v = Nz_v;
    this->N = 3;
    this->corners = 8;
    this->in_range = false;

    // Initialize FFT objects
    this->fft_x = std::make_unique<FFT>(Nx_v, Ny_v, Nz_v);
    this->fft_y = std::make_unique<FFT>(Nx_v, Ny_v, Nz_v);
    this->fft_z = std::make_unique<FFT>(Nx_v, Ny_v, Nz_v);

    // Copy voxel sizes
    this->voxel_size_x = voxel_sizes[0];
    this->voxel_size_y = voxel_sizes[1];
    this->voxel_size_z = voxel_sizes[2];

    // Allocate device vectors
    const size_t size_positions = Nx_v * Ny_v * Nz_v * N;
    const size_t size_grad = Nx_v * Ny_v * Nz_v * N * N;
    const size_t size_corner = (Nx_v - 1) * (Ny_v - 1) * (Nz_v - 1) * corners;

    d_positions.resize(size_positions);
    thrust::copy(deformedVoxelPositions, deformedVoxelPositions + size_positions, d_positions.begin());
    d_positions_ptr = thrust::raw_pointer_cast(d_positions.data());
    d_grad.resize(size_grad);
    d_grad_ptr = thrust::raw_pointer_cast(d_grad.data());
    d_corner.resize(size_corner);
    d_corner_ptr = thrust::raw_pointer_cast(d_corner.data());

    // Go to physical space
    to_physical();
    cudaDeviceSynchronize();

    // Start the algorithm
    std::cout << "Forcing Jacobians in range: [" << e1 << ", " << e2 << "]" << std::endl;

    // Compute initial Jacobians
    thrust::device_vector<float> J = compute_jacobian_determinant_3D(
        d_positions_ptr,
        Nx_v, Ny_v, Nz_v, voxel_size_x, voxel_size_y, voxel_size_z);
    cudaDeviceSynchronize();

    // Find the minimum and maximum Jacobian values
    auto minmax = thrust::minmax_element(J.begin(), J.end());
    float J_min = *minmax.first;
    float J_max = *minmax.second;
    cudaDeviceSynchronize();

    std::cout << "Jacobians range: [" << std::scientific << std::setprecision(10)
              << J_min << ", " << J_max << "]" << std::endl;

    // Count elements below e1 and above e2
    int count_below_e1 = thrust::count_if(J.begin(), J.end(),
        thrust::placeholders::_1 < e1);
    int count_above_e2 = thrust::count_if(J.begin(), J.end(),
        thrust::placeholders::_1 > e2);
    cudaDeviceSynchronize();

    std::cout << "Jmin#: " << count_below_e1 << " Jmax#: " << count_above_e2 << std::endl;

    if (J_min >= e1 && J_max <= e2) {
        std::cout << "Jacobians already in specified range, exiting." << std::endl;
        to_grid();
        cudaDeviceSynchronize();
        float* result = new float[size_positions];
        thrust::copy(d_positions.begin(), d_positions.end(), result);
        return result;
    }

    // Configure kernel launch parameters
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (Nz_v + blockSize.x - 1) / blockSize.x,
        (Ny_v + blockSize.y - 1) / blockSize.y,
        (Nx_v + blockSize.z - 1) / blockSize.z
    );

    int it_outer = 0;
    while ((J_min < e1 || J_max > e2) && it_outer < max_outer_it) {
        std::cout << "Outer iteration: " << it_outer + 1 << std::endl;

        // Compute corner Jacobians
        auto jcorner_n = compute_corner_jacobians_3D(d_corner_ptr);
        cudaDeviceSynchronize();

        int it_inner = 0;
        while ((jcorner_n.first > 0 || jcorner_n.second > 0) && it_inner < max_inner_it) {
            std::cout << "Inner iteration: " << it_inner + 1 << std::endl;

            // Compute gradients
            computeGradientsKernel<<<gridSize, blockSize>>>(d_positions_ptr, d_grad_ptr, Nx_v, Ny_v, Nz_v,
                                                            voxel_size_x, voxel_size_y, voxel_size_z);
            cudaDeviceSynchronize();

            // Limit gradients
            limitGradientsKernel<<<gridSize, blockSize>>>(d_grad_ptr, d_corner_ptr, Nx_v, Ny_v, Nz_v, e1, e2);
            cudaDeviceSynchronize();

            // Integrate gradient field
            integrate_gradient_field_3D(d_grad_ptr);
            cudaDeviceSynchronize();

            // Recompute corner Jacobians
            jcorner_n = compute_corner_jacobians_3D(d_corner_ptr);
            cudaDeviceSynchronize();

            it_inner++;
        }

        // Re-compute Jacobians
        J = compute_jacobian_determinant_3D(d_positions_ptr, Nx_v, Ny_v, Nz_v,
                                            voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]);
        cudaDeviceSynchronize();

        // Update min/max values
        minmax = thrust::minmax_element(J.begin(), J.end());
        J_min = *minmax.first;
        J_max = *minmax.second;
        cudaDeviceSynchronize();

        std::cout << "Jacobians range: [" << std::scientific << std::setprecision(10)
                  << J_min << ", " << J_max << "]" << std::endl;

        // Update counts
        count_below_e1 = thrust::count_if(J.begin(), J.end(),
            thrust::placeholders::_1 < e1);
        count_above_e2 = thrust::count_if(J.begin(), J.end(),
            thrust::placeholders::_1 > e2);
        cudaDeviceSynchronize();

        std::cout << "Jmin#: " << count_below_e1 << " Jmax#: " << count_above_e2 << std::endl;

        it_outer++;
    }

    // Go back to grid space
    to_grid();
    cudaDeviceSynchronize();

    std::cout << "Done" << std::endl;

    // Copy result back to host
    float* result = new float[size_positions];
    thrust::copy(d_positions.begin(), d_positions.end(), result);
    return result;
}