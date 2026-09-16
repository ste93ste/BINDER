#
# DCT implementation vs. naively computing log cost
import os
import numpy as np
from BINDER.utils import initialize_fourier_domain_smoothing
import BINDER

def test_DCT_log_cost():

    # Create dream location corrupted by random noise
    sizes_exp = [[5, 10], [11, 13]]  # Try both even and odd for last dimension, 2D and 3D

    threads = 1
    # gamma here is as in Fast Nonparametric Mutual-Information-based Registration and Uncertainty Estimation
    # gamma_p is as in PrimerOfSmoothing, i.e., gamma_p = gamma * sigma**2,
    # where sigma is the standard deviation of the spline approximation
    gamma = 5
    sigma = 0.34
    gamma_p = 1 * sigma ** 2
    for order in [1, 2]:

        print("Order: " + str(order))

        for sizes in sizes_exp:

            print("Sizes: " + str(sizes))

            if len(sizes) == 2:
                deformationField = np.random.standard_normal((sizes[0], sizes[1], 1, 2))
                z_size = 1
            else:
                deformationField = np.random.standard_normal((sizes[0], sizes[1], sizes[2], 3))
                z_size = sizes[2]

            #
            D, D_full = initialize_fourier_domain_smoothing(sizes, DCT=True, finite_difference_order=order,
                                                            number_of_dimensions=len(sizes),
                                                            return_full_matrix=True)

            #
            D = np.ascontiguousarray(D)
            deformationField = np.ascontiguousarray(deformationField)
            deformationObj = BINDER.Registration.NonLinearTransformation(gamma,
                                                                  gamma,
                                                                  gamma,
                                                                  1.0, 1.0, 1.0, # Voxels voxel sizes
                                                                  1.0, 1.0, 1.0, # Nodes voxel sizes
                                                                  D.ravel(order="C"),
                                                                  sizes[0],
                                                                  sizes[1],
                                                                  z_size,
                                                                  threads,
                                                                  sigma, 0.0,  # 3D order Spline stuff (std and offset)
                                                                  )

            log_prior_deformation, deformationField = deformationObj.smooth_deformation(deformationField.ravel(order="C"))

            # Spatial (naive) way of smoothing
            Nx = sizes[0]
            Ny = sizes[1]
            Gamma1cx = np.diag(np.ones(Nx), 0) - np.diag(np.ones(Nx - 1), -1);
            Gamma1cx = Gamma1cx[1:, :]
            Gamma1cy = np.diag(np.ones(Ny), 0) - np.diag(np.ones(Ny - 1), -1);
            Gamma1cy = Gamma1cy[1:, :]
            Gamma2cx = Gamma1cx.T @ Gamma1cx
            Gamma2cy = Gamma1cy.T @ Gamma1cy

            #
            nxs = np.arange(Nx).reshape(-1, 1)
            kxs = np.arange(Nx).reshape(1, -1)
            Cx = np.cos(np.pi / Nx * (nxs + .5) @ kxs)
            Cx[:, 0] /= np.sqrt(2)

            #
            nys = np.arange(Ny).reshape(-1, 1)
            kys = np.arange(Ny).reshape(1, -1)
            Cy = np.cos(np.pi / Ny * (nys + .5) @ kys)
            Cy[:, 0] /= np.sqrt(2)

            cost_naive = 0
            for n in range(len(sizes)):
                if order == 1:
                    filteredInRowDirection = Gamma1cx @ deformationField[..., 0, n]
                    filteredInColumnDirection = deformationField[..., 0, n] @ Gamma1cy.T
                    cost_naive += 0.5 * gamma * ((filteredInRowDirection ** 2).ravel().sum() +
                                                 (filteredInColumnDirection ** 2).ravel().sum())
                else:
                    filtered = Gamma2cx @ deformationField[..., 0, n] + deformationField[..., 0, n] @ Gamma2cy.T
                    cost_naive += 0.5 * gamma * (filtered ** 2).ravel().sum()

            print(f"log_prior_deformation Ours: {log_prior_deformation}")
            print(f"log_prior_deformation Naive: {cost_naive}")

            assert (np.allclose(cost_naive, log_prior_deformation))

    print("Test passed!")
