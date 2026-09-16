#
import os
import numpy as np
from matplotlib import pyplot as plt
from BINDER.utils import initialize_fourier_domain_smoothing
import BINDER
plt.ion()


def test():

    visualizer = False

    # Get some data
    Nx, Ny, Nz = 10, 11, 12
    # For visualization purposes
    slice = Nz // 2
    X, Y, Z = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny),
                          np.linspace(-1, 1, Nz), indexing='ij')
    mask = ((X/.5)**2 + (Y/.7)**2 + + (Z/.4)**2) < 1
    T = mask + np.random.standard_normal((Nx, Ny, Nz)) * .1

    # gamma here is as in Fast Nonparametric Mutual-Information-based Registration and Uncertainty Estimation
    # gamma_p is as in PrimerOfSmoothing, i.e., gamma_p = gamma * sigma**2,
    # where sigma is the standard deviation of the spline approximation
    gamma = 5.0
    sigma = .34
    gamma_p = gamma * sigma ** 2
    Gamma1cx = np.diag(np.ones(Nx), 0) - np.diag(np.ones(Nx - 1), -1); Gamma1cx = Gamma1cx[1:, :]
    Gamma1cy = np.diag(np.ones(Ny), 0) - np.diag(np.ones(Ny - 1), -1); Gamma1cy = Gamma1cy[1:, :]
    Gamma1cz = np.diag(np.ones(Nz), 0) - np.diag(np.ones(Nz - 1), -1); Gamma1cz = Gamma1cz[1:, :]
    Gamma2cx = Gamma1cx.T @ Gamma1cx
    Gamma2cy = Gamma1cy.T @ Gamma1cy
    Gamma2cz = Gamma1cz.T @ Gamma1cz

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

    #
    nzs = np.arange(Nz).reshape(-1, 1)
    kzs = np.arange(Nz).reshape(1, -1)
    Cz = np.cos(np.pi / Nz * (nzs + .5) @ kzs)
    Cz[:, 0] /= np.sqrt(2)

    #
    Kx, Ky, Kz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    Delta1 = (2.0 - 2 * np.cos(np.pi / Nx * Kx)) + \
             (2.0 - 2 * np.cos(np.pi / Ny * Ky)) + \
             (2.0 - 2 * np.cos(np.pi / Nz * Kz))
    Delta2 = Delta1**2

    # First things first: make sure our equations for decomposing the prior precision matrices are correct
    #
    # first order==1
    C = np.kron(Cz, np.kron(Cy, Cx))
    GammaRowDirection = np.kron(np.eye(Nz), np.kron(np.eye(Ny), Gamma1cx))
    GammaColumnDirection = np.kron(np.eye(Nz), np.kron(Gamma1cy, np.eye(Nx)))
    GammaDepthDirection = np.kron(Gamma1cz, np.kron(np.eye(Nx), np.eye(Ny)))
    xxx1 = GammaRowDirection.T @ GammaRowDirection + \
           GammaColumnDirection.T @ GammaColumnDirection + \
           GammaDepthDirection.T @ GammaDepthDirection
    yyy1 = C @ np.diag(Delta1.ravel(order='F')) @ np.linalg.inv(C)
    assert (np.allclose(xxx1, yyy1))

    # now order==2
    Gamma = np.kron(np.eye(Nz), np.kron(np.eye(Ny), Gamma2cx)) + \
            np.kron(np.eye(Nz), np.kron(Gamma2cy, np.eye(Nx))) + \
            np.kron(Gamma2cz, np.kron(np.eye(Nx), np.eye(Ny)))
    xxx2 = Gamma.T @ Gamma
    yyy2 = C @ np.diag(Delta2.ravel(order='F')) @ np.linalg.inv(C)
    assert (np.allclose(xxx2, yyy2))

    #
    for order in [2]:

        print('order = {}'.format(order))

        #
        if order == 1:
            Delta = Delta1
        else:
            Delta = Delta2

        # First a naive implementation: vectorize everything and use the same equations as in 1D
        N, D = Nx * Ny * Nz, 3  # 3D case
        C = np.kron(Cz, np.kron(Cy, Cx))
        delta = Delta.reshape(-1, 1, order='F')
        t = T.reshape(-1, 1, order='F')
        p_bar = 2**D / N * C.T @ t  # coefficients: inv( C ) @ t
        assert (np.allclose(C @ p_bar, t))
        p = p_bar / (1 + gamma_p * delta)  # Reduce high frequencies
        t_hat_naive = C @ p  # Go back to spatial domain
        T_hat_naive = t_hat_naive.reshape((Nx, Ny, Nz), order='F')

        # Even more professional implementation, using 2D DCTs. Note that the various rescaling is done
        # and then simply undone; normally you can skip these stips but here we compare against directly
        # against other ways of computing things.
        from scipy.fft import dctn, idctn
        P_bar_dct = 1/N * dctn(T)  # coefficients
        # DC component is scaled differently
        P_bar_dct[0, :, :] /= np.sqrt(2); P_bar_dct[:, 0, :] /= np.sqrt(2); P_bar_dct[:, :, 0] /= np.sqrt(2)
        P_dct = P_bar_dct / (1 + gamma_p * Delta)  # Reduce high frequencies
        # DC component is scaled differently
        tmp = P_dct.copy(); tmp[0, :, :] *= np.sqrt(2); tmp[:, 0, :] *= np.sqrt(2); tmp[:, :, 0] *= np.sqrt(2)
        T_hat_dct = idctn(tmp) * N  # Careful with the normalizing N!

        assert (np.allclose(T_hat_naive, T_hat_dct))
        assert (np.allclose(p_bar, P_bar_dct.reshape(-1, 1, order='F')))
        assert (np.allclose(p, P_dct.reshape(-1, 1, order='F')))

        # Last but not least: compute the regularizer
        # First the naive implementation: filter in the spatial domain
        if order == 1:
            # First order
            filteredInRowDirection = np.einsum('ij, jkl -> ikl', Gamma1cx, T_hat_dct)
            filteredInColumnDirection = np.einsum('jkl, ki -> jil', T_hat_dct, Gamma1cy.T)
            filteredInDepthDirection = np.einsum('jkl, li -> jki', T_hat_dct, Gamma1cz.T)
            cost_naive = 0.5 * gamma * ((filteredInRowDirection**2).ravel().sum() + \
                                        (filteredInColumnDirection**2).ravel().sum() + \
                                        (filteredInDepthDirection**2).ravel().sum())
            print(f"cost_naive: {cost_naive}")
            if visualizer:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(filteredInRowDirection[:, :, slice], cmap=plt.cm.gray)
                plt.title(f'filteredInRowDirection [order: {order}]')
                plt.subplot(1, 3, 2)
                plt.imshow(filteredInColumnDirection[:, :, slice], cmap=plt.cm.gray)
                plt.title(f'filteredInColumnDirection [order: {order}]')
                plt.subplot(1, 3, 3)
                plt.imshow(filteredInDepthDirection[Nx // 2, :, :], cmap=plt.cm.gray)
                plt.title(f'filteredInDepthDirection [order: {order}]')
        else:
            # Second order
            filtered = np.einsum('ij, jkl -> ikl', Gamma2cx, T_hat_dct) + \
                       np.einsum('jkl, ki -> jil', T_hat_dct, Gamma2cy.T) + \
                       np.einsum('jkl, li -> jki', T_hat_dct, Gamma2cz.T)
            cost_naive = 0.5 * gamma * (filtered**2).ravel().sum()
            print(f"cost_naive: {cost_naive}")
            if visualizer:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(filtered[..., slice], cmap=plt.cm.gray)
                plt.title(f'filtered [order: {order}]')

            # Let's also have a look at the 3D kernel we're really using
            dirac = np.zeros((Nx, Ny, Nz))
            dirac[round(Nx / 2), round(Ny / 2), round(Nz / 2)] = 1
            filteredDirac = np.einsum('ij, jkl -> ikl', Gamma2cx, dirac) + \
                            np.einsum('jkl, ki -> jil', dirac, Gamma2cy.T) + \
                            np.einsum('jkl, li -> jki', dirac, Gamma2cz.T)
            kernel = filteredDirac[round(Nx / 2) - 1: round(Nx / 2) + 2,
                                   round(Ny / 2) - 1: round(Ny / 2) + 2,
                                   round(Nz / 2) - 1: round(Nz / 2) + 2]
            print("kernel: ", kernel)
            if visualizer:
                plt.figure()
                plt.subplot(1, 2, 2)
                plt.title(f'kernel [order: {order}]')
                plt.subplot(1, 3, 1)
                plt.imshow(kernel[..., 0], cmap=plt.cm.gray)
                plt.subplot(1, 3, 2)
                plt.imshow(kernel[..., 1], cmap=plt.cm.gray)
                plt.subplot(1, 3, 3)
                plt.imshow(kernel[..., 2], cmap=plt.cm.gray)

        # More professional implementation: use the DCT coefficients
        cost_dct = 0.5 * gamma * (N / 2**D * ((Delta**0.5 * P_dct)**2).ravel().sum())
        print(f"cost_dct: {cost_dct}")
        assert(np.allclose(cost_naive, cost_dct))

        # Our implementation
        T_tmp = np.zeros([Nx, Ny, Nz, 3])
        T_tmp[:, :, :, 0] = T.copy()
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

        ours_cost, T_tmp = deformationObj.smooth_deformation(T_tmp.ravel(order="C"))

        if visualizer:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(T[..., slice], cmap=plt.cm.gray, vmin=T[..., slice].min(), vmax=T[..., slice].max())
            plt.title('T: slice {}'.format(slice))

            plt.subplot(2, 2, 2)
            plt.imshow(T_hat_naive[..., slice], cmap=plt.cm.gray, vmin=T[..., slice].min(), vmax=T[..., slice].max())
            plt.title(f'T_hat_naive [order: {order}]')

            plt.subplot(2, 2, 3)
            plt.imshow(T_hat_dct[..., slice], cmap=plt.cm.gray, vmin=T[..., slice].min(), vmax=T[..., slice].max())
            plt.title(f'T_hat_dct [order: {order}]')

            plt.subplot(2, 2, 4)
            plt.imshow(T_tmp[..., 0][..., slice], cmap=plt.cm.gray, vmin=T[..., slice].min(), vmax=T[..., slice].max())
            plt.title(f'T_hat_ours [order: {order}]')

        print(f"ours_cost: {ours_cost}")
        assert (np.allclose(ours_cost, cost_dct))
        assert (np.allclose(T_tmp[:, :, :, 0], T_hat_naive, rtol=1e-5, atol=1e-6))
