#include "Registration.h"
#include "topology_correction_3D.h"
#include "Smoothing_Spline.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    // Seed for random number generation
    std::srand(std::time(0));

    int threads = 1;
    int spline_order = 3;
    int Nx_v = 25;
    int Ny_v = 56;
    int Nz_v = 7;
    int Nx_n = 25;
    int Ny_n = 21;
    int Nz_n = 10;
    int N = 3;
    double non_zero_voxels = std::floor(Nx_v * Ny_v * Nz_v * 0.88);

    // Initialize example data for voxel_pos and node_pos
    auto* final_locations = new double[Nx_v * Ny_v * Nz_v * N];
    auto* voxel_pos = new double[Nx_v * Ny_v * Nz_v * N];
    auto* node_pos = new double[Nx_n * Ny_n * Nz_n * N];

    //
    for (int i = 0; i < Nx_v * Ny_v * Nz_v; ++i) {
        int x = i % Nx_v;
        int y = (i / Nx_v) % Ny_v;
        int z = (i / (Nx_v * Ny_v)) % Nz_v;
        final_locations[N*i] = x + static_cast<double>(std::rand()) / RAND_MAX;
        final_locations[N*i + 1] = y + static_cast<double>(std::rand()) / RAND_MAX;
        final_locations[N*i + 2] = z + static_cast<double>(std::rand()) / RAND_MAX;
    }

    for (int i = 0; i < Nx_n * Ny_n * Nz_n; ++i) {
        int x = i % Nx_n;
        int y = (i / Nx_n) % Ny_n;
        int z = (i / (Nx_n * Ny_n)) % Nz_n;
        node_pos[i] = x;
        node_pos[i + 1] = y;
        node_pos[i + 2] = z;
    }
    for (int i = 0; i < Nx_v * Ny_v * Nz_v; ++i) {
        int x = i % Nx_v;
        int y = (i / Nx_v) % Ny_v;
        int z = (i / (Nx_v * Ny_v)) % Nz_v;
        voxel_pos[i] = x;
        voxel_pos[i + 1] = y;
        voxel_pos[i + 2] = z;
    }

    // Create Registration object with the parameters
    Registration* registrator = new Registration(final_locations, threads, voxel_pos, node_pos,
                                                 Nx_v, Ny_v, Nz_v, Nx_n, Ny_n, Nz_n, spline_order, non_zero_voxels);

    // Set the MILikelihood parameters
    int K = 64;
    int L = 64;
    double alpha = 2.0;
    auto * theta = new double[K * L];
    auto * binned_nodes = new int[Nx_n * Ny_n * Nz_n];
    auto * binned_voxels = new int[Nx_v * Ny_v * Nz_v];

    // Fill theta with random values
    for (int i = 0; i < K * L; ++i) {
        theta[i] = 1.0 / (K * L);
    }

    for (int i = 0; i < Nx_n * Ny_n * Nz_n; ++i) {
        binned_nodes[i] = std::rand() % K;
    }
    for (int i = 0; i < Nx_v * Ny_v * Nz_v; ++i) {
        binned_voxels[i] = std::rand() % L;
    }

    // Set the MILikelihood for the Registration object
    registrator->setMILikelihood(alpha, theta, K, L, threads, binned_nodes, binned_voxels);

    // Set nonlinear transformation parameters
    double gamma_x = 2.5;
    double gamma_y = 2.1;
    double gamma_z = 1.8;
    double rho_x = 1.0f;
    double rho_y = 1.0f;
    double rho_z = 1.0f;
    double eta_x = 1.0f;
    double eta_y = 1.0f;
    double eta_z = 1.0f;
    auto * D = new double[Nx_v * Ny_v * Nz_v];
    for (int i = 0; i < Nx_v * Ny_v * Nz_v; ++i) {
        final_locations[i] = static_cast<double>(std::rand()) / RAND_MAX;
        D[i] = static_cast<double>(std::rand()) / RAND_MAX  + static_cast<double>(i)* 0.01;
    }

    registrator->setNonLinearTransformation(gamma_x, gamma_y, gamma_z, rho_x, rho_y, rho_z, eta_x, eta_y, eta_z, D);
    // Call the EM function
    int max_EM_iterations = 10;
    double convergence_th = 1e-15;

    registrator->EM(max_EM_iterations, convergence_th, true, true);

    std::cout << "EM function executed successfully!" << std::endl;

    int N_x_o = Nx_v * 2;
    int N_y_o = Ny_v * 2;
    int N_z_o = Nz_v * 2;
    if (N == 2) {
        N_z_o = 1;
    }
    std::vector<double> up_sampled = NonLinearTransformation::up_sample(final_locations, Nx_v, Ny_v, Nz_v,
                                                                        N_x_o, N_y_o, N_z_o, threads);

    std::cout << "Upsample function executed successfully!" << std::endl;

    auto * im = new double[20 * 20 * 20];
    auto * im_down = new double[10 * 10 * 10];
    auto* smooth_obj = new Smoothing_Spline(spline_order, threads);
    smooth_obj->smooth_and_downsample_image(im, im_down, 20, 20, 20,2);

    std::cout << "Smoothing function executed successfully!" << std::endl;

    auto * topology_correction_obj = new topology_correction_3D();

    auto * voxel_sizes = new double[3];
    voxel_sizes[0] = 1;
    voxel_sizes[1] = 1.2;
    voxel_sizes[2] = 0.7;
    topology_correction_obj->correct_topology(final_locations,
        0.01, 100, 10, 10, threads, voxel_sizes, Nx_v, Ny_v, Nz_v);

    std::cout << "Topology correction function executed successfully!" << std::endl;

    int seed = 12345;
    int N_s = 100;
    int N_b = 100;
    registrator->sampler(seed, N_b, N_s, false, 100, true, true, true);

    std::cout << "Sampler function executed successfully!" << std::endl;

    return 0;
}
