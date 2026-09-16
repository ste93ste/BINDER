# Test 3D DCT sampling and compare it with ground truth covariance matrix
import os
import numpy as np
np.set_printoptions(suppress=True)
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
    Nx, Ny = 8, 7
    numberOfSamples = 10 ** 5

    #
    Gamma1cx = np.diag(np.ones(Nx), 0) - np.diag(np.ones(Nx - 1), -1)
    Gamma1cx = Gamma1cx[1:, :]
    Gamma1cy = np.diag(np.ones(Ny), 0) - np.diag(np.ones(Ny - 1), -1)
    Gamma1cy = Gamma1cy[1:, :]
    Gamma2cx = Gamma1cx.T @ Gamma1cx
    Gamma2cy = Gamma1cy.T @ Gamma1cy

    N, D = Nx * Ny, 2  # 2D implementation
    for order in [1, 2]:
        # Let's set up the ground truth covariance matrix
        if order == 1:
            GammaRowDirection = np.kron(np.eye(Ny), Gamma1cx)
            GammaColumnDirection = np.kron(Gamma1cy, np.eye(Nx))
            tmp = np.eye(N) + gamma_p * GammaRowDirection.T @ GammaRowDirection + \
                  gamma_p * GammaColumnDirection.T @ GammaColumnDirection
        else:
            Gamma = np.kron(np.eye(Ny), Gamma2cx) + np.kron(Gamma2cy, np.eye(Nx))
            tmp = np.eye(N) + gamma_p * Gamma.T @ Gamma
        covariance = sigma ** 2 * np.linalg.inv(tmp)

        #
        if visualizer:
            plt.figure()
            plt.subplot(4, 2, 1)
            plt.imshow(covariance)
            plt.title(f"covariance [order: {order}]")
            plt.colorbar()
            plt.subplot(4, 2, 2)
            plt.plot(covariance[np.round(N / 2).astype(int), :])
            plt.title(f"middle row of covariance [order: {order}]")

        # Let's sample using a naive implementation to make sure we get all the details right
        samples = np.random.multivariate_normal(np.zeros(N), covariance, numberOfSamples)
        empiricalCovariance = samples.T @ samples / numberOfSamples

        if visualizer:
            plt.subplot(4, 2, 3)
            plt.imshow(empiricalCovariance)
            plt.title(f"naive empirical covariance [order: {order}]")
            plt.colorbar()
            plt.subplot(4, 2, 4)
            plt.plot(empiricalCovariance[np.round(N / 2).astype(int), :])
            plt.title(f"middle row of naive empirical covariance [order: {order}]")

        assert (np.allclose(empiricalCovariance, covariance, rtol=1e-3, atol=1e-3))

        # Now using DCT decomposition
        nxs = np.arange(Nx).reshape(-1, 1)
        kxs = np.arange(Nx).reshape(1, -1)
        Cx = np.cos(np.pi / Nx * (nxs + .5) @ kxs)
        Cx[:, 0] /= np.sqrt(2)

        #
        nys = np.arange(Ny).reshape(-1, 1)
        kys = np.arange(Ny).reshape(1, -1)
        Cy = np.cos(np.pi / Ny * (nys + .5) @ kys)
        Cy[:, 0] /= np.sqrt(2)

        #
        Kx, Ky = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
        Delta1 = (2.0 - 2 * np.cos(np.pi / Nx * Kx)) + \
                 (2.0 - 2 * np.cos(np.pi / Ny * Ky))
        Delta2 = Delta1 ** 2
        if order == 1:
            Delta = Delta1
        else:
            Delta = Delta2

        #
        samples = np.zeros((numberOfSamples, N))
        for sampleNumber in range(numberOfSamples):
            P = (sigma ** 2 * 2 ** D / N / (1 + gamma_p * Delta)) ** (1 / 2) * np.random.standard_normal((Nx, Ny))
            sample = Cx @ P @ Cy.T  # This should use a 2D IDCT implementation in real applications
            samples[sampleNumber, :] = sample.reshape(1, -1, order='F')

        empiricalCovarianceDCT = samples.T @ samples / numberOfSamples

        if visualizer:
            plt.subplot(4, 2, 5)
            plt.imshow(empiricalCovarianceDCT)
            plt.title(f"DCT-based empirical covariance [order: {order}]")
            plt.colorbar()
            plt.subplot(4, 2, 6)
            plt.plot(empiricalCovarianceDCT[np.round(N / 2).astype(int), :])
            plt.title(f"middle row of DCT-based empirical covariance [order: {order}]")

        # Do the same for our implementation
        T_tmp = np.zeros([Nx, Ny, 1, 2])  # Empty signal, we are only interested in adding structured noise here
        Delta = initialize_fourier_domain_smoothing([Nx, Ny], DCT=True, finite_difference_order=order,
                                                    number_of_dimensions=2,
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
                                                              1,
                                                              1,
                                                              sigma, 0.0,  # 3D order Spline stuff (std and offset)
                                                              )

        our_samples = deformationObj.sampleDeformationField(T_tmp.ravel(order="C"), numberOfSamples)
        our_samples = our_samples[:, :, :, 0, 0].reshape(-1, Nx * Ny, order='F')
        empiricalCovarianceOur = our_samples.T @ our_samples / numberOfSamples

        if visualizer:
            plt.subplot(4, 2, 7)
            plt.imshow(empiricalCovarianceOur)
            plt.title(f"Ours empirical covariance [order: {order}]")
            plt.colorbar()
            plt.subplot(4, 2, 8)
            plt.plot(empiricalCovarianceOur[np.round(N / 2).astype(int), :])
            plt.title(f"middle row of Ours empirical covariance  [order: {order}]")

        # Weak test on similarity of the covariance matrices
        assert (np.allclose(empiricalCovarianceOur, covariance, rtol=1e-3, atol=1e-3))

        if visualizer:
            # Finally, also show some samples
            # DCT-based samples
            plt.figure()
            for i in range(5 ** 2):
                plt.subplot(5, 5, i + 1)
                sample = samples[i, :].reshape((Nx, Ny), order='F')
                plt.imshow(sample)
                plt.colorbar()
                plt.title(f"DCT-based sample number {i} [order: {order}]")

            # Ours DCT-based samples
            plt.figure()
            for i in range(5 ** 2):
                plt.subplot(5, 5, i + 1)
                sample = our_samples[i, :].reshape((Nx, Ny), order='F')
                plt.imshow(sample)
                plt.colorbar()
                plt.title(f"Ours DCT-based sample number {i} [order: {order}]")

    print("Test passed!")
