# Test for checking b-spline of order 1, 2 and 3 implemented in Cython for both 2D and 3D
import os
import numpy as np
import BINDER

def evaluate_B_spline( y, shift=0, order=0 ):
    """
    Evaluates the uniform B-spline of order "order", centered around "shift",
    at the locations in vector "y"
    """
    x = np.abs( y - shift )
    if order == 0:
        # Zeroth order "constant" B-spline
        kernel = ( x < .5 ) + ( x == .5 ) / 2
    elif order == 1:
        # First order "linear" B-spline
        kernel = ( 1 - x ) * ( x < 1 )
    elif order == 3:
        # Third order "cubic" B-spline
        kernel = ( 2 / 3 - x**2 + x**3 / 2 ) * ( x < 1 ) + \
                 ( 2 - x )**3 / 6 * ( x >= 1 ) * ( x < 2 )  # Eq (6) in UnserSPM1999
    else:
        raise ValueError( f"order {order} is not implemented" )

    return kernel

def perform_least_squares_splines_1D( t, down_factor=2, order=3 ):
    x = np.arange( 0, t.shape[0] ).reshape( -1, 1 )
    A = evaluate_B_spline( x / down_factor, shift=x[::down_factor].T / down_factor, order=order )
    A_solved = np.linalg.solve( A.T @ A, A.T @ t )
    A_downsampled = A[::down_factor, :]
    t_down = A_downsampled @ A_solved
    return t_down

def perform_least_squares_splines_3D(im, down_factor=2, order=3):
    """
    Perform least squares spline smoothing and downsampling in 3D.
    The process is applied sequentially along the three axes: z, y, x.
    """
    # Downsample along the z-axis (depth)
    z, y, x = im.shape
    new_z, new_y, new_x = int(np.ceil(z / down_factor)), int(np.ceil(y / down_factor)), int(np.ceil(x / down_factor))
    im_down = np.zeros((new_z, y, x))  # Reduced depth
    for i in range(y):  # y-axis
        for j in range(x):  # x-axis
            im_down[:, i, j] = perform_least_squares_splines_1D(im[:, i, j], down_factor, order)

    # Downsample along the y-axis (height)
    im_down2 = np.zeros((new_z, new_y, x))  # Reduced height
    for i in range(new_z):  # Reduced depth
        for j in range(x):  # x-axis
            im_down2[i, :, j] = perform_least_squares_splines_1D(im_down[i, :, j], down_factor, order)

    # Downsample along the x-axis (width)
    im_down3 = np.zeros((new_z, new_y, new_x))  # Reduced width
    for i in range(new_z):  # Reduced depth
        for j in range(new_y):  # Reduced height
            im_down3[i, j, :] = perform_least_squares_splines_1D(im_down2[i, j, :], down_factor, order)

    return im_down3


def test_smoothing_spline():
    print("Spline tests")

    threads = 1
    orders = [1, 3]
    dimensions = [2, 3]
    for dimension in dimensions:

        print("Dimension: " + str(dimension))

        if dimension == 2:
            shapes = [30, 16, 1]
            T_down_shapes = [int(np.ceil(shapes[0] // 2)), int(np.ceil(shapes[1] // 2)), 1]
            T = np.random.random(shapes)
            T_down = np.zeros(T_down_shapes)
        else:
            shapes = [10, 6, 8]
            T_down_shapes = [int(np.ceil(shapes[0] // 2)), int(np.ceil(shapes[1] // 2)), int(np.ceil(shapes[2] // 2))]
            T = np.random.random(shapes)
            T_down = np.zeros(T_down_shapes)

        print("T_down shapes: " + str(T_down_shapes))

        T = np.ascontiguousarray(T)
        T_down = np.ascontiguousarray(T_down)

        for order in orders:

            print("Order: " + str(order))

            smooth_spline_object = BINDER.Registration.Smoothing_Spline(order, threads)

            smooth_spline_object.smooth_and_downsample_image(T.ravel(order="C"),
                                                             T_down.ravel(order="C"),
                                                             shapes[0],
                                                             shapes[1],
                                                             shapes[2],
                                                             2)

            T_down_python = perform_least_squares_splines_3D(T, 2, order=order)

            assert(np.allclose(T_down, T_down_python))

    print("Done, all tests passed")
