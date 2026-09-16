#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <array>
#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<double> compute_jacobian_determinant_2D(
    const double* field,
    const int Nx,
    const int Ny,
    const double voxel_size_x,
    const double voxel_size_y
) {
    if (field == nullptr) {
        throw std::invalid_argument("Field pointer cannot be null");
    }
    if (Nx <= 0 || Ny <= 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }

    constexpr int N = 2;
    std::vector<double> jacobian_determinant(Nx * Ny);
    const double vol_scale = 1.0 / (voxel_size_x * voxel_size_y);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            const int idx = x * Ny + y;

            // Calculate all four corner Jacobians for this cell
            double J_total = 0.0;
            int valid_corners = 0;

            // Get current point coordinates (bottom-left)
            const double wx00 = field[idx * N];
            const double wy00 = field[idx * N + 1];

            // Forward-forward (bottom-right corner)
            if (x < Nx - 1 && y < Ny - 1) {
                const double wx10 = field[(x + 1) * Ny * N + y * N];
                const double wy10 = field[(x + 1) * Ny * N + y * N + 1];
                const double wx01 = field[x * Ny * N + (y + 1) * N];
                const double wy01 = field[x * Ny * N + (y + 1) * N + 1];

                const double Jff = ((wx10 - wx00) * (wy01 - wy00) -
                                  (wx01 - wx00) * (wy10 - wy00)) * vol_scale;
                J_total += Jff;
                valid_corners++;
            }

            // Backward-forward (top-right corner)
            if (x > 0 && y < Ny - 1) {
                const double wx_prev = field[(x - 1) * Ny * N + y * N];
                const double wy_prev = field[(x - 1) * Ny * N + y * N + 1];
                const double wx_next = field[x * Ny * N + (y + 1) * N];
                const double wy_next = field[x * Ny * N + (y + 1) * N + 1];

                const double Jbf = ((wx00 - wx_prev) * (wy_next - wy00) -
                                  (wx_next - wx00) * (wy00 - wy_prev)) * vol_scale;
                J_total += Jbf;
                valid_corners++;
            }

            // Forward-backward (bottom-left corner)
            if (x < Nx - 1 && y > 0) {
                const double wx_next = field[(x + 1) * Ny * N + y * N];
                const double wy_next = field[(x + 1) * Ny * N + y * N + 1];
                const double wx_prev = field[x * Ny * N + (y - 1) * N];
                const double wy_prev = field[x * Ny * N + (y - 1) * N + 1];

                const double Jfb = ((wx_next - wx00) * (wy00 - wy_prev) -
                                  (wx00 - wx_prev) * (wy_next - wy00)) * vol_scale;
                J_total += Jfb;
                valid_corners++;
            }

            // Backward-backward (top-left corner)
            if (x > 0 && y > 0) {
                const double wx_prevX = field[(x - 1) * Ny * N + y * N];
                const double wy_prevX = field[(x - 1) * Ny * N + y * N + 1];
                const double wx_prevY = field[x * Ny * N + (y - 1) * N];
                const double wy_prevY = field[x * Ny * N + (y - 1) * N + 1];

                const double Jbb = ((wx00 - wx_prevX) * (wy00 - wy_prevY) -
                                  (wx00 - wx_prevY) * (wy00 - wy_prevX)) * vol_scale;
                J_total += Jbb;
                valid_corners++;
            }

            // Average the valid corner Jacobians
            jacobian_determinant[idx] = valid_corners > 0 ? J_total / valid_corners : 1.0;
        }
    }

    return jacobian_determinant;
}

std::vector<double> compute_jacobian_determinant_3D(
    const double* field,
    const int Nx,
    const int Ny,
    const int Nz,
    const double voxel_size_x,
    const double voxel_size_y,
    const double voxel_size_z
) {
    if (field == nullptr) {
        throw std::invalid_argument("Field pointer cannot be null");
    }
    if (Nx <= 0 || Ny <= 0 || Nz <= 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }

    constexpr int N = 3;
    std::vector<double> jacobian_determinant(Nx * Ny * Nz);
    const double vol_scale = 1.0 / (voxel_size_x * voxel_size_y * voxel_size_z);

    // Helper function to compute linear index for field access
    auto field_idx = [Ny, Nz, N](const int x, const int y, const int z) -> int {
        return (x * Ny * Nz + y * Nz + z) * N;
    };

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int z = 0; z < Nz; ++z) {
                const int idx = x * Ny * Nz + y * Nz + z;  // Index for output array
                const int curr_idx = field_idx(x, y, z);   // Index for field access

                // Calculate all eight corner Jacobians for this cell
                double J_total = 0.0;
                int valid_corners = 0;

                // Get current point coordinates
                const double wx000 = field[curr_idx];
                const double wy000 = field[curr_idx + 1];
                const double wz000 = field[curr_idx + 2];

                // Forward-forward-forward corner (x+1, y+1, z+1)
                if (x < Nx - 1 && y < Ny - 1 && z < Nz - 1) {
                    const int idx100 = field_idx(x + 1, y, z);
                    const int idx010 = field_idx(x, y + 1, z);
                    const int idx001 = field_idx(x, y, z + 1);

                    const double wx100 = field[idx100];
                    const double wy100 = field[idx100 + 1];
                    const double wz100 = field[idx100 + 2];
                    const double wx010 = field[idx010];
                    const double wy010 = field[idx010 + 1];
                    const double wz010 = field[idx010 + 2];
                    const double wx001 = field[idx001];
                    const double wy001 = field[idx001 + 1];
                    const double wz001 = field[idx001 + 2];

                    const double Jfff = (
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

                    const double wxm100 = field[idxm100];
                    const double wym100 = field[idxm100 + 1];
                    const double wzm100 = field[idxm100 + 2];
                    const double wx010 = field[idx010];
                    const double wy010 = field[idx010 + 1];
                    const double wz010 = field[idx010 + 2];
                    const double wx001 = field[idx001];
                    const double wy001 = field[idx001 + 1];
                    const double wz001 = field[idx001 + 2];

                    const double Jbff = (
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

                    const double wx100 = field[idx100];
                    const double wy100 = field[idx100 + 1];
                    const double wz100 = field[idx100 + 2];
                    const double wx0m10 = field[idx0m10];
                    const double wy0m10 = field[idx0m10 + 1];
                    const double wz0m10 = field[idx0m10 + 2];
                    const double wx001 = field[idx001];
                    const double wy001 = field[idx001 + 1];
                    const double wz001 = field[idx001 + 2];

                    const double Jfbf = (
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

                    const double wx100 = field[idx100];
                    const double wy100 = field[idx100 + 1];
                    const double wz100 = field[idx100 + 2];
                    const double wx010 = field[idx010];
                    const double wy010 = field[idx010 + 1];
                    const double wz010 = field[idx010 + 2];
                    const double wx00m1 = field[idx00m1];
                    const double wy00m1 = field[idx00m1 + 1];
                    const double wz00m1 = field[idx00m1 + 2];

                    const double Jffb = (
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

                    const double wxm100 = field[idxm100];
                    const double wym100 = field[idxm100 + 1];
                    const double wzm100 = field[idxm100 + 2];
                    const double wx0m10 = field[idx0m10];
                    const double wy0m10 = field[idx0m10 + 1];
                    const double wz0m10 = field[idx0m10 + 2];
                    const double wx001 = field[idx001];
                    const double wy001 = field[idx001 + 1];
                    const double wz001 = field[idx001 + 2];

                    const double Jbbf = (
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

                    const double wxm100 = field[idxm100];
                    const double wym100 = field[idxm100 + 1];
                    const double wzm100 = field[idxm100 + 2];
                    const double wx010 = field[idx010];
                    const double wy010 = field[idx010 + 1];
                    const double wz010 = field[idx010 + 2];
                    const double wx00m1 = field[idx00m1];
                    const double wy00m1 = field[idx00m1 + 1];
                    const double wz00m1 = field[idx00m1 + 2];

                    const double Jbfb = (
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

                    const double wx100 = field[idx100];
                    const double wy100 = field[idx100 + 1];
                    const double wz100 = field[idx100 + 2];
                    const double wx0m10 = field[idx0m10];
                    const double wy0m10 = field[idx0m10 + 1];
                    const double wz0m10 = field[idx0m10 + 2];
                    const double wx00m1 = field[idx00m1];
                    const double wy00m1 = field[idx00m1 + 1];
                    const double wz00m1 = field[idx00m1 + 2];

                    const double Jfbb = (
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

                    const double wxm100 = field[idxm100];
                    const double wym100 = field[idxm100 + 1];
                    const double wzm100 = field[idxm100 + 2];
                    const double wx0m10 = field[idx0m10];
                    const double wy0m10 = field[idx0m10 + 1];
                    const double wz0m10 = field[idx0m10 + 2];
                    const double wx00m1 = field[idx00m1];
                    const double wy00m1 = field[idx00m1 + 1];
                    const double wz00m1 = field[idx00m1 + 2];

                    const double Jbbb = (
                        (wx000 - wxm100) * ((wy000 - wy0m10) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz000 - wz0m10)) -
                        (wx000 - wx0m10) * ((wy000 - wym100) * (wz000 - wz00m1) - (wy000 - wy00m1) * (wz000 - wzm100)) +
                        (wx000 - wx00m1) * ((wy000 - wym100) * (wz000 - wz0m10) - (wy000 - wy0m10) * (wz000 - wzm100))
                    ) * vol_scale;
                    J_total += Jbbb;
                    valid_corners++;
                }

                // Average the valid corner Jacobians
                jacobian_determinant[idx] = valid_corners > 0 ? J_total / valid_corners : 1.0;
            }
        }
    }
    return jacobian_determinant;
}