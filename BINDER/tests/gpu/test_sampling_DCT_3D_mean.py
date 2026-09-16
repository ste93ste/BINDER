# Test 3D DCT sampling by looking at "closeness" of mean sample for spatial, DCT scipy, and ours.
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
    sigma = np.sqrt(9.0 / (8.0 * np.pi))  # sigma of B-spline order 3
    gamma_p = gamma * sigma ** 2
    Nx, Ny, Nz = 4, 3, 5
    numberOfSamples = 10 ** 5
    # For visualization purposes
    slice = Nz // 2
    X, Y, Z = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny),
                          np.linspace(-1, 1, Nz), indexing='ij')
    mask = ((X/.5)**2 + (Y/.7)**2 + + (Z/.4)**2) < 1
    T = mask + np.random.standard_normal((Nx, Ny, Nz)) * 100
    # Round so that we simulate dream location sampling
    T = np.round(T)
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

        # Let's sample using a naive implementation to make sure we get all the details right
        t = T.reshape(-1, 1, order='F')
        mean = 1 / sigma**2 * covariance @ t
        samples = np.random.multivariate_normal(mean.ravel(), covariance, numberOfSamples)
        mean_sample = np.mean(samples, axis=0).reshape((Nx, Ny, Nz), order='F')

        #
        if visualizer:
            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(T[:, :, slice])
            plt.title("Original signal")
            plt.subplot(1, 4, 2)
            plt.imshow(mean_sample[:, :, slice])
            plt.title("Mean sample spatial")

        # Do the same for our implementation
        T_tmp = np.zeros([Nx, Ny, Nz, 3])  # Empty signal, we are only interested in adding structured noise here
        T_tmp[:, :, :, 0] = T
        T_tmp[:, :, :, 1] = 0
        T_tmp[:, :, :, 2] = 0

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

        our_samples = deformationObj.sampleDeformationField(T_tmp.ravel(order="C"), numberOfSamples)

        our_samples = our_samples[:, :, :, :, 0].reshape(-1, Nx * Ny * Nz, order='F')

        mean_sample_ours = np.mean(our_samples, axis=0).reshape((Nx, Ny, Nz), order='F')

        # Now check that if we run the whole sampler object, sample from the deformation only,
        # and keep everything else fixed, we get the same results.
        emptyPositions = np.zeros([Nx, Ny, Nz, 3])
        emptyPositions = np.ascontiguousarray(emptyPositions)
        Delta = np.ascontiguousarray(Delta)
        emptyBinnedNodes = np.zeros([Nx, Ny, Nz])
        emptyBinnedNodes = np.ascontiguousarray(emptyBinnedNodes)
        emptyTheta = np.zeros([32, 32])
        emptyTheta = np.ascontiguousarray(emptyTheta)
        registratorObj = BINDER.Registration.Registration(final_locations=emptyPositions.ravel(order="C"),
                                                           threads=1,
                                                           voxel_pos=emptyPositions.ravel(order="C"),
                                                           node_pos=emptyPositions.ravel(order="C"),
                                                           Nx_v=Nx, Ny_v=Ny, Nz_v=Nz, Nx_n=Nx, Ny_n=Ny, Nz_n=Nz,
                                                           spline_order=3,
                                                           non_zero_voxels=Nx * Ny * Nz)
        registratorObj.setNonLinearTransformation(gamma_x=gamma, gamma_y=gamma, gamma_z=gamma,
                                                  voxels_size_x =1.0, voxels_size_y=1.0, voxels_size_z=1.0,
                                                  nodes_size_x =1.0, nodes_size_y=1.0, nodes_size_z=1.0,
                                                  D=Delta.ravel(order="C"))
        registratorObj.setMILikelihood(alpha=2, theta=emptyTheta.ravel(order="C"), K=32, L=32, threads=1,
                                       binned_nodes=emptyBinnedNodes.ravel(order="C"),
                                       binned_voxels=emptyBinnedNodes.ravel(order="C"))
        registratorObj.set_dream_locations(T_tmp.ravel(order="C"))
        meanDefVoxelPos, _ = registratorObj.sampler(seed=seed, N_b=0, N_s=numberOfSamples,
                                                    sample_gamma=0,
                                                    sample_gamma_every=100,
                                                    sample_posteriors=0,
                                                    sample_likelihood_parameters=0,
                                                    sample_transformation_parameters=1)
        mean_sample_ours_pipeline = meanDefVoxelPos[:, :, :, 0]

        if visualizer:
            plt.subplot(1, 4, 3)
            plt.imshow(mean_sample_ours[:, :, slice])
            plt.title("Mean sample ours")
            plt.subplot(1, 4, 4)
            plt.imshow(mean_sample_ours_pipeline[:, :, slice])
            plt.title("Mean sample ours pipeline")
            plt.show()

        # Check that mean sample is very similar
        assert (np.allclose(mean_sample, mean_sample_ours, rtol=1e-3, atol=1e-2))
        assert (np.allclose(mean_sample, mean_sample_ours_pipeline, rtol=1e-3, atol=1e-2))

    print("Test passed!")
