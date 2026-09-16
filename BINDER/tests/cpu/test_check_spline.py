# Test for checking b-spline of order 1, 2 and 3 implemented in Cython for both 2D and 3D
import os
import numpy as np
import BINDER

def test_spline():
    print("Spline tests")

    orders = [1, 2, 3]
    dimensions = [2, 3]
    for dimension in dimensions:

        print("Dimension: " + str(dimension))

        for order in orders:

            print("Order: " + str(order))

            if dimension == 2:
                splineObj = BINDER.Registration.Spline(order, 1)
            else:
                splineObj = BINDER.Registration.Spline(order, 0)

            # Test values inside range for all spline orders
            if dimension == 2:
                distances = np.array([0.4, 0.3], dtype=float)
            else:
                distances = np.array([0.2, 0.4, 0.1], dtype=float)
            # No log space
            x, y = np.array(splineObj.spline(distances[0]))[:order + 1], np.array(splineObj.spline(distances[1]))[:order + 1]
            if dimension > 2:
                z = np.array(splineObj.spline(distances[2]))[:order + 1]
            # Log space
            x_l, y_l = np.array(splineObj.log_spline(distances[0]))[:order + 1], np.array(splineObj.log_spline(distances[1]))[:order + 1]
            if dimension > 2:
                z_l = np.array(splineObj.log_spline(distances[2]))[:order + 1]

            # Check that our Spline probabilities sum to 1
            if dimension == 2:
                assert (np.allclose(1.0, np.sum(x[:, None, None] * y[None, :, None])))
            else:
                assert (np.allclose(1.0, np.sum(x[:, None, None] * y[None, :, None] * z[None, None, :])))

            # Check that spline and log spline lead to the same outcome
            assert (np.allclose(x, np.exp(x_l)))
            assert (np.allclose(y, np.exp(y_l)))
            if dimension == 3:
                assert (np.allclose(z, np.exp(z_l)))

            # Test values purposely outside the range (at least in one of the directions)
            # We want to make sure that no nans or inf are computed
            if dimension == 2:
                distances = np.array([0.0, 1.0], dtype=float)
            else:
                distances = np.array([0.0, 1.0, -0.3], dtype=float)

            # No log space
            x, y = np.array(splineObj.spline(distances[0]))[:order + 1], np.array(splineObj.spline(distances[1]))[:order + 1]
            if dimension > 2:
                z = np.array(splineObj.spline(distances[2]))[:order + 1]
            # Log space
            x_l, y_l = np.array(splineObj.log_spline(distances[0]))[:order + 1], np.array(splineObj.log_spline(distances[1]))[:order + 1]
            if dimension > 2:
                z_l = np.array(splineObj.log_spline(distances[2]))[:order + 1]

            # Still check that our Spline probabilities sum to 1
            if dimension == 2:
                assert (np.allclose(1.0, np.sum(x[:, None, None] * y[None, :, None])))
            else:
                assert (np.allclose(1.0, np.sum(x[:, None, None] * y[None, :, None] * z[None, None, :])))

            # Check log spline does not have nans
            assert (np.allclose(False, np.isnan(x_l)))
            assert (np.allclose(False, np.isnan(y_l)))
            if dimension == 3:
                assert (np.allclose(False, np.isnan(z_l)))

    print("Done, all tests passed")
