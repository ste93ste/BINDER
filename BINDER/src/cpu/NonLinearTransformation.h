//
// Created by stce on 01/01/25.
//

#ifndef NONLINEARTRANSFORMATION_H
#define NONLINEARTRANSFORMATION_H

#include "Transformation.h"
#include "DCT.h"
#include <vector>
#include <memory>

class NonLinearTransformation final : public Transformation {
public:
    NonLinearTransformation(double gamma_x, double gamma_y, double gamma_z,
                            double rho_x, double rho_y, double rho_z,
                            double eta_x, double eta_y, double eta_z,
                            const double* D,
                            int Nx_v, int Ny_v, int Nz_v,
                            int threads, double sigma_spline, double spline_offset);
    ~NonLinearTransformation() override;

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

    void fill_in_data(int n, const double *dream_locations, const double *voxel_pos) const;

    void updateGammasFromNu();

    double compute_raw_regularization(int n) const;

    void update_final_locations(int n, const double *voxel_pos, int *node_indices, double *ds,
                                double *final_locations) const;

    double smooth(int n) const;

    std::pair<double, double *> smooth_deformation(const double *deformations);
    double * sampleDeformationField(const double* deformationField, int number_of_samples);

    void precompute_sample_from_gaussian_distribution();
    double smooth_using_DCT(int input);
    void fill_in_data(double* input, int n, double* output);
    void compute_final_locations(int n, double* input, double* output, double* params, int* status);
    [[nodiscard]] std::vector<int> get_shape() const;

    void set_generator(std::mt19937 &gen) override;

    [[nodiscard]] static std::vector<double> up_sample(const double* deformation_field, int N_x, int N_y, int N_z,
                                                       int N_x_o, int N_y_o, int N_z_o, int threads);
    [[nodiscard]] std::vector<double> down_sample(double* deformation_field, int factor);

private:

    double gamma[3]{};
    double rho[3]{};
    double eta[3]{};
    double alpha_0;
    double beta_0;
    double nu;
    int sample;
    int sample_gamma;
    int is2D;
    int threads;
    std::mt19937 gen;
    int num_voxels;
    double normalization_constant;
    double sigma_spline;
    double spline_offset;
    int Nx_v;
    int Ny_v;
    int Nz_v;
    int N;
    const double *D;
    std::unique_ptr<double[]> random_std_DCT;
    std::unique_ptr<double[]> fourier_tmp_sigma_DCT;
    bool recompute_tmp_sigma{};
    std::vector<std::unique_ptr<DCT>> dcts;
};

#endif // NONLINEARTRANSFORMATION_H
