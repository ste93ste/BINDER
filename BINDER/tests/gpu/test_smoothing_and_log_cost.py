#
# Test DFT implementation vs. Numpy DFT implementation and check log cost
# Test 2D/3D and even/odd sizes
# Also test DCT implementation vs. naively computing smoothing and log cost
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct  # Only 1D versions!
from numpy.fft import fftn, ifftn
from BINDER.utils import initialize_fourier_domain_smoothing
import BINDER

def dctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = dct(x, axis=i, norm=norm)
    return x


def idctn(x, norm="ortho"):
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=norm)
    return x


def test_smoothing_and_log_cost():

    visualizer = False

    # Create dream location corrupted by random noise
    sizes_exp = [[11, 20, 7]]

    threads = 1
    gamma = 5.0

    for sizes in sizes_exp:

        print("Sizes: " + str(sizes))

        deformationField = np.random.standard_normal((sizes[0], sizes[1], sizes[2], 3))
        z_size = sizes[2]
        #
        D_bar, D_bar_full = initialize_fourier_domain_smoothing(sizes, DCT=True, finite_difference_order=2,
                                                                number_of_dimensions=len(sizes),
                                                                return_full_matrix=True)

        #
        D_bar = np.ascontiguousarray(D_bar)
        deformationFieldIn = np.ascontiguousarray(deformationField)
        deformationObj = BINDER.Registration.NonLinearTransformation(gamma,
                                                              gamma,
                                                              gamma,
                                                              1.0, 1.0, 1.0, # Voxels voxel sizes
                                                              1.0, 1.0, 1.0, # Nodes voxel sizes
                                                              D_bar.ravel(order="C").astype(np.float32),
                                                              sizes[0],
                                                              sizes[1],
                                                              z_size,
                                                              threads,
                                                              1, 0.0,  # 3D order Spline stuff (std and offset)
                                                              )

        log_prior_deformation, deformationFieldOut = deformationObj.smooth_deformation(deformationFieldIn.ravel(order="C").astype(np.float32))

        # Scipy DCT/IDCT
        T_hat_dct = []
        T_hat_fft = []
        cost_fft = 0

        for n in range(len(sizes)):
            if len(sizes) == 3:
                data = deformationField[:, :, :, n]
            else:
                data = deformationField[:, :, 0, n]

            C_bar_dct = 1 / (np.prod(sizes) ** len(sizes)) * dctn(data)  # Fourier coefficients

            C_dct = C_bar_dct / (1 + gamma * D_bar)  # Reduce high frequencies

            T_hat_dct.append(idctn(C_dct).real * (np.prod(sizes) ** len(sizes)))

            # Check that DCT is the same as DFT with double the size (in each dimension)
            mirrored_data = np.vstack((data, np.flip(data, axis=0)))
            mirrored_data = np.hstack((mirrored_data, np.flip(mirrored_data, axis=1)))
            if len(sizes) == 3:
                mirrored_data = np.concatenate((mirrored_data, np.flip(mirrored_data, axis=2)), axis=-1)

            C_bar_fft = 1 / np.prod(mirrored_data.shape) * fftn(mirrored_data)  # Fourier coefficients
            C_fft = C_bar_fft / (1 + gamma * D_bar_full)  # Reduce high frequencies
            tmp = 0.5 * gamma * np.prod(mirrored_data.shape) *\
                  ((D_bar_full ** 0.5 * np.absolute(C_fft)) ** 2).ravel().sum()
            cost_fft += tmp / 2**len(sizes)
            t_tmp = ifftn(C_fft).real * np.prod(mirrored_data.shape)

            if len(sizes) == 2:
                T_hat_fft.append(t_tmp[:sizes[0], :sizes[1]])
            else:
                T_hat_fft.append(t_tmp[:sizes[0], :sizes[1], :sizes[2]])

        if visualizer:
            # Plot before vs. after smoothing
            f, ax = plt.subplots(nrows=len(sizes), ncols=5)
            for n in range(len(sizes)):
                dct_smoothed = T_hat_dct[n]
                fft_smoothed = T_hat_fft[n]
                ours_smoothed = deformationFieldOut[..., n]
                if len(sizes) == 3:
                    dct_smoothed = dct_smoothed[:, :, 0]
                    fft_smoothed = fft_smoothed[:, :, 0]
                ours_smoothed = ours_smoothed[:, :, 0]

                ax[n][0].imshow(deformationField[:, :, 0, n])
                ax[n][0].set_title("Signal")
                ax[n][1].imshow(ours_smoothed)
                ax[n][1].set_title("Smoothed: Ours-DCT")
                ax[n][2].imshow(dct_smoothed)
                ax[n][2].set_title("Smoothed: Scipy-DCT")
                ax[n][3].imshow(fft_smoothed)
                ax[n][3].set_title("Smoothed: Numpy-DFT\n double the size")
                im = ax[n][4].imshow(dct_smoothed - ours_smoothed)
                ax[n][4].set_title("Ours - Scipy")

            f.colorbar(im, ax=ax.ravel().tolist())

            plt.show()

        # Check that ours, Numpy fft on double signal and Numpy dct give the same results
        for n in range(len(sizes)):
            assert (np.allclose(T_hat_dct[n], deformationFieldOut[:, :, :, n], atol=1e-6))
            assert (np.allclose(T_hat_fft[n], deformationFieldOut[:, :, :, n], atol=1e-6))

        # Check that costs are correct
        print(f"log_prior_deformation Ours: {log_prior_deformation}")
        print(f"log_prior_deformation Numpy dft on double the signal: {cost_fft}")

        assert (np.allclose(cost_fft, log_prior_deformation))

    print("Test passed!")
