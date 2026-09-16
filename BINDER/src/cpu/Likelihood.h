//
// Created by stce on 01/01/25.
//

#ifndef LIKELIHOOD_H
#define LIKELIHOOD_H
#include <random>

class Likelihood {
public:
    // Constructor (can be implemented if needed)
    Likelihood();

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Likelihood();

    // Pure virtual functions to enforce implementation in derived classes
    virtual double compute_likelihood(int voxel_index, int x_n, int y_n, int z_n) = 0;
    virtual double compute_log_likelihood(int voxel_index, int x_n, int y_n, int z_n) = 0;

    virtual void initialize_parameters() = 0;
    virtual double compute_cost() = 0;
    virtual void set_generator(std::mt19937 &gen) = 0;
    virtual void update_parameters() = 0;
    virtual void sample_parameters() = 0;
    virtual void accumulate_info(int thread_id, int voxel_index,
                                 int x_n, int y_n, int z_n, double posterior) = 0;
    virtual std::vector<double> get_parameters() = 0;
};

#endif // LIKELIHOOD_H
