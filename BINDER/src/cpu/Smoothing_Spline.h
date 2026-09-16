// Smoothing_Spline.h
#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <Eigen/Dense>

class Smoothing_Spline {
public:
    Smoothing_Spline(int spline_order, int threads);
    ~Smoothing_Spline();

    void smooth_and_downsample_image(
        const double* im, double* im_down,
        int Nx, int Ny, int Nz,
        int down_factor) const;

private:
    static constexpr int LOOKUP_RESOLUTION = 50000;
    const int spline_order;
    const int threads;
    std::vector<double> bspline_values;

    // Cache structures
    struct MatrixCacheKey {
        int length;
        int down_length;
        int down_factor;

        bool operator==(const MatrixCacheKey& other) const {
            return length == other.length &&
                   down_length == other.down_length &&
                   down_factor == other.down_factor;
        }
    };

    struct MatrixCacheKeyHash {
        std::size_t operator()(const MatrixCacheKey& k) const {
            return std::hash<int>()(k.length) ^
                   (std::hash<int>()(k.down_length) << 1) ^
                   (std::hash<int>()(k.down_factor) << 2);
        }
    };

    struct MatrixCacheEntry {
        Eigen::MatrixXd A;
        Eigen::MatrixXd ATA;
        Eigen::LDLT<Eigen::MatrixXd> ldlt;

        MatrixCacheEntry() = default;
        explicit MatrixCacheEntry(Eigen::MatrixXd  A_) : A(std::move(A_)) {
            ATA = A.transpose() * A;
            ldlt.compute(ATA);
        }
    };

    mutable std::unordered_map<MatrixCacheKey, MatrixCacheEntry, MatrixCacheKeyHash> matrix_cache;
    mutable std::mutex cache_mutex;

    // Core methods
    double compute_bspline(double x) const;
    inline double evaluate_B_spline(double y, double shift) const;

    void perform_least_squares_splines_1D(
        const std::vector<double>& t,
        std::vector<double>& t_down,
        int length,
        int down_length,
        int down_factor) const;

    void perform_least_squares_splines_2D(
        const double* im,
        double* im_down,
        int Nx, int Ny,
        int down_factor) const;

    void perform_least_squares_splines_3D(
        const double* im,
        double* im_down,
        int Nx, int Ny, int Nz,
        int down_factor) const;
};