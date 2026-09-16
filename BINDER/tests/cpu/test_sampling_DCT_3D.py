# Test 3D DCT sampling and compare it with ground truth covariance matrix
import os
import numpy as np
from matplotlib import pyplot as plt
from BINDER.utils import initialize_fourier_domain_smoothing
import BINDER
def test():

    visualizer = False

    seed = 12345
    np.random.seed(seed)

    # gamma here is as in Fast Nonparametric Mutual-Information-based Registration and Uncertainty Estimation
    # gamma_p is as in PrimerOfSmoothing, i.e., gamma_p = gamma * sigma**2,
    # where sigma is the standard deviation of the spline approximation
    gamma = 5.0
    sigma = .34
    gamma_p = gamma * sigma ** 2
    Nx, Ny, Nz = 3, 5, 4
    numberOfSamples = 10 ** 5

    #
    Gamma1cx = np.diag(np.ones(Nx), 0) - np.diag(np.ones(Nx - 1), -1)
    Gamma1cx = Gamma1cx[1:, :]
    Gamma1cy = np.diag(np.ones(Ny), 0) - np.diag(np.ones(Ny - 1), -1)
    Gamma1cy = Gamma1cy[1:, :]
    Gamma1cz = np.diag(np.ones(Nz), 0) - np.diag(np.ones(Nz - 1), -1)
    Gamma1cz = Gamma1cz[1:, :]
    Gamma2cx = Gamma1cx.T @ Gamma1cx
    Gamma2cy = Gamma1cy.T @ Gamma1cy
    Gamma2cz = Gamma1cz.T @ Gamma1cz

    N, D = Nx * Ny * Nz, 3  # 3D implementation
    for order in [1, 2]:
        # Let's set up the ground truth covariance matrix
        if order == 1:
            GammaRowDirection = np.kron(np.eye(Nz), np.kron(np.eye(Ny), Gamma1cx))
            GammaColumnDirection = np.kron(np.eye(Nz), np.kron(Gamma1cy, np.eye(Nx)))
            GammaDepthDirection = np.kron(Gamma1cz, np.kron(np.eye(Nx), np.eye(Ny)))
            tmp = np.eye(N) + gamma_p * GammaRowDirection.T @ GammaRowDirection + \
                              gamma_p * GammaColumnDirection.T @ GammaColumnDirection + \
                              gamma_p * GammaDepthDirection.T @ GammaDepthDirection
        else:
            Gamma = np.kron(np.eye(Nz), np.kron(np.eye(Ny), Gamma2cx)) + \
                    np.kron(np.eye(Nz), np.kron(Gamma2cy, np.eye(Nx))) + \
                    np.kron(Gamma2cz, np.kron(np.eye(Nx), np.eye(Ny)))
            tmp = np.eye(N) + gamma_p * Gamma.T @ Gamma

        covariance = sigma ** 2 * np.linalg.inv(tmp)

        #
        if visualizer:
            plt.figure()
            plt.subplot(3, 2, 1)
            plt.imshow(covariance)
            plt.title(f"covariance [order: {order}]")
            plt.colorbar()
            plt.subplot(3, 2, 2)
            plt.plot(covariance[np.round(N / 2).astype(int), :])
            plt.title(f"middle row of covariance [order: {order}]")

        # Let's sample using a naive implementation to make sure we get all the details right
        samples = np.random.multivariate_normal(np.zeros(N), covariance, numberOfSamples)
        empiricalCovariance = samples.T @ samples / numberOfSamples
        if visualizer:
            plt.subplot(3, 2, 3)
            plt.imshow(empiricalCovariance)
            plt.title(f"naive empirical covariance [order: {order}]")
            plt.colorbar()
            plt.subplot(3, 2, 4)
            plt.plot(empiricalCovariance[np.round(N / 2).astype(int), :])
            plt.title(f"middle row of naive empirical covariance [order: {order}]")

        # Weak test on similarity of the covariance matrices
        assert (np.allclose(empiricalCovariance, covariance, rtol=1e-3, atol=1e-3))

        # Do the same for our implementation
        T_tmp = np.zeros([Nx, Ny, Nz, 3])  # Empty signal, we are only interested in adding structured noise here

        Delta = initialize_fourier_domain_smoothing([Nx, Ny, Nz], DCT=True, finite_difference_order=order,
                                                    number_of_dimensions=3,
                                                    return_full_matrix=False)

        # Ours
        Delta = np.ascontiguousarray(Delta)
        T_tmp = np.ascontiguousarray(T_tmp)
        deformationObj = BINDER.Registration.NonLinearTransformation(gamma,
                                                              gamma,
                                                              gamma,
                                                              1.0, 1.0, 1.0, # Voxels voxel sizes
                                                              1.0, 1.0, 1.0, # Nodes voxel sizes
                                                              Delta.ravel(order="C"),
                                                              Nx,
                                                              Ny,
                                                              Nz,
                                                              1,
                                                              sigma, 0.0,  # 3D order Spline stuff (std and offset)
                                                              )

        our_samples = deformationObj.sampleDeformationField(T_tmp.ravel(order="C"), number_of_samples=numberOfSamples)

        our_samples = our_samples[:, :, :, :, 0].reshape(-1, Nx * Ny * Nz, order='F')
        empiricalCovarianceOur = our_samples.T @ our_samples / numberOfSamples

        # Weak test on similarity of the covariance matrices
        assert (np.allclose(empiricalCovarianceOur, covariance, rtol=1e-3, atol=1e-3))

        if visualizer:
            plt.subplot(3, 2, 5)
            plt.imshow(empiricalCovarianceOur)
            plt.title(f"Ours empirical covariance [order: {order}]")
            plt.colorbar()
            plt.subplot(3, 2, 6)
            plt.plot(empiricalCovarianceOur[np.round(N / 2).astype(int), :])
            plt.title(f"middle row of ours empirical covariance [order: {order}]")
            plt.show()

    print("Test passed!")
