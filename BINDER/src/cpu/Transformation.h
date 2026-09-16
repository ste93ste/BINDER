//
// Created by stce on 01/01/25.
//

#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H
#include <random>


class Transformation {
public:
    // Constructor
    Transformation();

    // Virtual destructor to allow proper cleanup for derived classes
    virtual ~Transformation() = default;

    // Abstract method that is meant to be overridden by derived classes
    virtual double constrain_transformation(const double *dream_locations,
                                            const double *voxel_pos,
                                            double *ds,
                                            double *final_locations,
                                            int *node_indices) = 0;

    // Virtual method that can be overridden or left unimplemented by derived classes
    virtual void sample_transformation(const double *dream_locations,
                                       const double *voxel_pos,
                                       double *ds,
                                       double *final_locations,
                                       int *node_indices,
                                       int sample_gamma) = 0;

    virtual void set_generator(std::mt19937 &gen) = 0;


};

#endif //TRANSFORMATION_H
