//
// Created by stce on 01/01/25.
//

#ifndef SPLINE_H
#define SPLINE_H

#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

class Spline {
public:
    Spline(int order, bool is2D);

    std::array<double, 4> spline(double d) const;
    std::array<double, 4> log_spline(double d) const;

    int neighbours;
    int order;
    bool is2D;
    double sigma_spline{};
    double offset{};
    std::array<int, 4> offx;
    std::array<int, 4> offy;
    std::array<int, 4> offz;

    void init_order_3();
    void init_order_2();
    void init_order_1();

};

#endif //SPLINE_H