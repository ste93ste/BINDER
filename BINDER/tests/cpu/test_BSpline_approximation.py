# Test for checking Gaussian approximation of b-spline of order 2 and order 3
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as stats


@pytest.mark.skip(reason="Don't test plotting functions")
def plotSplineAndGaussian(spline, mu, sigma_gaussian, values, order):
    splineValues = []
    gaussianValues = []
    for x in values:
        splineValues.append(spline(x))
        gaussianValues.append(stats.norm.pdf(x, mu, sigma_gaussian))
    plt.title("Spline order: " + str(order))
    plt.plot(values, splineValues, label='B-Spline: order ' + str(order))
    plt.plot(values, gaussianValues, label='Gaussian, sigma: ' + str(sigma_gaussian))
    plt.grid()
    plt.legend()
    plt.show()


@pytest.mark.skip(reason="Don't test this, more like a sanity check")
def check_spline_approximation():

    visualizer = True

    orders = [1, 2, 3]

    for order in orders:

        print("Order: " + str(order))

        if order == 1:
            # Check Gaussian approximation for spline order 1
            # We are fixing the interval to be between -1.5 and 1.5
            spline_interval = np.array([-1.5, 0, 1.5])
            # This is for visualization purposes
            data_plot = np.arange(-2.5, 2.5, 0.01)
            sigma_gaussian = np.sqrt(0.4)
            mu = 0
            # Gaussian distribution with mean at the center of the support
            y = 1 / (sigma_gaussian * np.sqrt(2 * np.pi)) * np.exp(-(spline_interval - mu) ** 2 / (2 * sigma_gaussian ** 2))
        elif order == 2:
            # Check Gaussian approximation for spline order 2
            # We are fixing the interval to be between -2.0 and 2.0
            spline_interval = np.arange(-2.0, 2.0 + 0.5, 0.5)
            # This is for visualization purposes
            data_plot = np.arange(-2.5, 2.5, 0.01)

            sigma_gaussian = np.sqrt(8 / (9 * np.pi))
            mu = 0
            # Gaussian distribution with mean at the center of the support
            y = 1 / (sigma_gaussian * np.sqrt(2 * np.pi)) * np.exp(-(spline_interval - mu) ** 2 / (2 * sigma_gaussian ** 2))
        elif order == 3:
            # Check Gaussian approximation for spline order 3
            # We are fixing the interval to be between -2.5 and 2.5
            spline_interval = np.arange(-2.5, 2.5 + 0.5, 0.5)
            # This is for visualization purposes
            data_plot = np.arange(-2.5, 2.5, 0.01)
            sigma_gaussian = np.sqrt(9 / (8 * np.pi))
            mu = 0
            # Gaussian distribution with mean at the center of the support
            y = 1 / (sigma_gaussian * np.sqrt(2 * np.pi)) * np.exp(-(spline_interval - mu) ** 2 / (2 * sigma_gaussian ** 2))
        else:
            print("Order not implemented, exit")

        t, c, k = interpolate.splrep(spline_interval, y, s=0, k=order)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        if visualizer:
            plotSplineAndGaussian(spline, mu, sigma_gaussian, data_plot, order)
