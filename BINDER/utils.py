import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage
from skimage.transform import warp
#


# Bin input signal
def bin_signal(signal, number_of_bins):
    min_signal = np.min(signal)
    max_signal = np.max(signal)
    bin_width = (max_signal - min_signal) / (number_of_bins - 1)
    return np.round((signal - min_signal) / bin_width).astype(np.int32)


# Bin input signal
def bin_signal_robust(signal, number_of_bins, lower_percentile=0.01, higher_percentile=0.99, intensity_bins=None):
    """
    Bin signal data into discrete intervals, either automatically calculated or using provided bins.

    Parameters:
    -----------
    signal : array-like
        Input signal to be binned
    number_of_bins : int
        Number of bins to use (ignored if intensity_bins is provided)
    lower_percentile : float, optional
        Lower percentile for robust minimum calculation
    higher_percentile : float, optional
        Higher percentile for robust maximum calculation
    intensity_bins : array-like, optional
        Custom intensity bin edges. If provided, overrides automatic bin calculation

    Returns:
    --------
    tuple:
        - binned_signal: array of binned values
        - bin_edges: array of bin edges including start and end points
    """
    if intensity_bins is not None:
        # Use provided intensity bins
        bin_edges = intensity_bins
        binned_signal = np.digitize(signal, bin_edges).astype(np.int32) - 1
        # Handle min/max bins explicitly
        binned_signal[signal <= bin_edges[0]] = 0
        binned_signal[signal >= bin_edges[-1]] = number_of_bins - 1
        return binned_signal, bin_edges

    # Calculate robust min/max using internal bins
    internal_bins = 1000
    histogram, binCenters = np.histogram(signal, internal_bins)
    cumulativePdf = np.cumsum(histogram / np.sum(histogram))

    # Find robust minimum
    tmp = np.where(cumulativePdf <= lower_percentile)
    add = 0.01
    while len(tmp[0]) == 0:
        tmp = np.where(cumulativePdf <= lower_percentile + add)
        add = add + 0.01
    robustMinimum = binCenters[tmp[0][-1]]

    # Find robust maximum
    tmp = np.where(cumulativePdf <= higher_percentile)
    add = 0.01
    while len(tmp[0]) == 0:
        tmp = np.where(cumulativePdf <= higher_percentile + add)
        add = add + 0.01
    robustMaximum = binCenters[tmp[0][-1]]

    # Calculate bin width and edges
    bin_width = (robustMaximum - robustMinimum) / (number_of_bins - 1)
    bin_edges = np.linspace(robustMinimum, robustMaximum, number_of_bins + 1)

    # Bin the signal
    binned_signal = np.floor((signal - robustMinimum) / bin_width).astype(np.int32)
    binned_signal[signal < robustMinimum] = 0
    binned_signal[signal > robustMaximum] = number_of_bins - 1

    return binned_signal, bin_edges

def hist_norm(source, template):
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    interp_t_values = interp_t_values.astype(olddtype)

    return interp_t_values[bin_idx].reshape(oldshape)


# Copied from stack overflow
def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def plot_signals_and_warp(binned_voxels, binned_nodes, deformedVoxelPositions, voxels_shape, voxels_pos,
                          number_of_dimensions):

    if number_of_dimensions != 2:
        # TODO: Probably better to have an interactive plot, for now take a slice
        slice = int(voxels_shape[1] // 2)
        voxels_shape = [voxels_shape[0], voxels_shape[2]]
        binned_voxels = binned_voxels[:, slice, :]
        voxels_pos = voxels_pos[:, slice, :, 0:2]
        warped = warp(binned_nodes[:, slice, :], np.moveaxis(deformedVoxelPositions, 3, 0)[[0, 2], :, slice, :],
                      order=0, output_shape=voxels_shape, preserve_range=True)
        binned_nodes = binned_nodes[:, slice, :]
        deformedVoxelPositions = deformedVoxelPositions[:, slice, :, 0:2]
    else:
        # Remove third dimension of size 1
        binned_voxels = binned_voxels[:, :, 0]
        binned_nodes = binned_nodes[:, :, 0]
        deformedVoxelPositions = deformedVoxelPositions[:, :, 0, 0:2]
        voxels_pos = voxels_pos[:, :, 0, 0:2]
        voxels_shape = voxels_shape[0:2]
        warped = warp(binned_nodes, np.moveaxis(deformedVoxelPositions, 2, 0),
                      order=0, output_shape=voxels_shape, preserve_range=True)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 5), sharex=True, sharey=True)

    ax = axs[0, 0]
    ax.set_title("voxels ('fixed')")
    xxx = ax.imshow(binned_voxels, cmap='gray')
    plt.colorbar(xxx, ax=ax, shrink=1.0)
    ax.grid()

    ax = axs[0, 1]
    ax.set_title("nodes ('moving')")
    xxx = ax.imshow(binned_nodes, cmap='gray')
    plt.colorbar(xxx, ax=ax, shrink=1.0)
    ax.grid()

    ax = axs[0, 2]
    ax.set_title("voxels - nodes")
    max_size = np.minimum(binned_voxels.shape, binned_nodes.shape)
    xxx = ax.imshow((binned_voxels[:max_size[0], :max_size[1]] - binned_nodes[:max_size[0], :max_size[1]]), cmap='gray',
                    vmin=(binned_voxels[:max_size[0], :max_size[1]] - binned_nodes[:max_size[0], :max_size[1]]).min(),
                    vmax=(binned_voxels[:max_size[0], :max_size[1]] - binned_nodes[:max_size[0], :max_size[1]]).max())
    plt.colorbar(xxx, ax=ax, shrink=1.0)
    ax.grid()

    ax = axs[1, 0]
    ax.set_title("voxels ('fixed') with deformation overlaid")
    xxx = ax.imshow(binned_voxels, cmap='gray')
    plt.colorbar(xxx, ax=ax, shrink=1.0)
    tmp = deformedVoxelPositions - voxels_pos
    X, Y = np.indices(voxels_shape)
    ax.quiver(Y, X, tmp[:, :, 1], -tmp[:, :, 0], color='blue')
    ax.grid()

    ax = axs[1, 1]
    ax.set_title("warped (resampled) nodes")
    xxx = ax.imshow(warped, cmap='gray')
    plt.colorbar(xxx, ax=ax, shrink=1.0)
    ax.grid()

    ax = axs[1, 2]
    ax.set_title("voxels - warped nodes")
    xxx = ax.imshow((binned_voxels - warped), cmap='gray',
                    vmin=(binned_voxels[:max_size[0], :max_size[1]] - binned_nodes[:max_size[0], :max_size[1]]).min(),
                    vmax=(binned_voxels[:max_size[0], :max_size[1]] - binned_nodes[:max_size[0], :max_size[1]]).max())
    plt.set_cmap('gray')
    plt.colorbar(xxx, ax=ax, shrink=1.0)
    ax.grid()

    plt.draw()


def plot_theta(theta):
    plt.figure()
    plt.grid()
    plt.title("Theta")
    plt.imshow(theta, cmap="Reds")
    plt.xlabel("Fixed")
    plt.ylabel("Moving")
    plt.show()


def plot_coeff(coeff):
    plt.figure()
    plt.grid()
    plt.title("Coeff")
    plt.imshow(coeff, cmap="Reds")
    plt.xlabel("F features")
    plt.ylabel("M features")
    plt.colorbar()
    plt.show()


def plot_fourier_smoothing(fourier_smoothing, number_of_dimensions):
    if number_of_dimensions == 2:
        plt.figure()
        plt.grid()
        plt.title("Smoothing matrix")
        plt.imshow(fourier_smoothing, cmap="Reds")
        plt.show()
    else:
        # Show a random slice
        slice = int(fourier_smoothing.shape[1] // 2)
        plt.figure()
        plt.title("Smoothing matrix")
        plt.imshow(fourier_smoothing[:, slice, :])
        plt.show()


def save_warped_image(deformed_position, movingImage, outputDir, order=1, affine=np.eye(4), outputName="warped.nii.gz"):
    img = warp(movingImage, np.moveaxis(deformed_position, 3, 0), order=order, output_shape=deformed_position.shape,
               preserve_range=True)
    tmp = nib.Nifti1Image(img, affine)
    nib.save(tmp, os.path.join(outputDir, outputName))
    print("Warped image saved")


def save_warp(deformations, outputDir, affine=np.eye(4), outputName="warp.nii.gz"):
    tmp = nib.Nifti1Image(deformations, affine)
    nib.save(tmp, os.path.join(outputDir, outputName))
    print("Warp saved")


def save_image(generalImage, outputDir, outputName, affine=np.eye(4)):
    tmp = nib.Nifti1Image(generalImage, affine)
    nib.save(tmp, os.path.join(outputDir, outputName))
    print("Image saved")


def maskImage(image, th_value=0, use_robust_minimum=False):

    #
    if use_robust_minimum:
        internal_bins = 1000
        histogram, binCenters = np.histogram(image.ravel(), internal_bins)
        cumulativePdf = np.cumsum(histogram / np.sum(histogram))
        tmp = np.where(cumulativePdf <= 0.1)
        add = 0.01
        while len(tmp[0]) == 0:
            # print("Lower percentile is too low for this image, increasing to: " + str(lower_percentile + add))
            tmp = np.where(cumulativePdf <= 0.1 + add)
            add = add + 0.01
        th_value = binCenters[tmp[0][-1]]

    # Threshold for background
    backgroundMask = np.ma.greater(image, th_value)

    # Remove noise, if any, by applying a closing operation (erosion + dilation)
    mask = ndimage.binary_closing(backgroundMask, iterations=1)

    # Fill holes
    mask = ndimage.binary_fill_holes(mask)

    return mask.astype(np.int32)


# Initializing Fourier domain smoothing
def initialize_fourier_domain_smoothing(size, DCT=True, finite_difference_order=2, number_of_dimensions=3,
                                        return_full_matrix=False, voxel_size=np.ones(3)):

    if number_of_dimensions == 1:
        if DCT:
            Kx = np.arange(size[0] * 2)
            D = (2.0 - 2 * np.cos(np.pi * Kx / size[0])) / voxel_size[0]**2
            D = D ** finite_difference_order
            if return_full_matrix:
                return D[:size[0]], D
            else:
                return D[:size[0]]
        else:
            Kx = np.arange(size[0])
            D = (2.0 - 2 * np.cos(2 * np.pi * Kx / (size[0]))) / voxel_size[0]**2
            half = int(np.floor(len(D) / 2)) + 1
            D = D ** finite_difference_order
            if return_full_matrix:
                return D[:half], D
            else:
                return D[:half]

    if number_of_dimensions == 2:
        X, Y = size[0], size[1]

        if DCT:
            if return_full_matrix:
                Kx, Ky = np.meshgrid(np.arange(X * 2),
                                     np.arange(Y * 2),
                                     indexing='ij')
            else:
                Kx, Ky = np.meshgrid(np.arange(X * 2),
                                     np.arange(Y * 2),
                                     indexing='ij')

            D = (2.0 - 2 * np.cos(np.pi * Kx / X)) / voxel_size[0]**2 + \
                (2.0 - 2 * np.cos(np.pi * Ky / Y)) / voxel_size[1]**2
            D = D ** finite_difference_order
            if return_full_matrix:
                return D[:X, :Y], D
            else:
                return D[:X, :Y]

        else:  # FFT, real to complex

            Kx, Ky = np.meshgrid(np.arange(X),
                                 np.arange(Y), indexing='ij')
            D = (2.0 - 2 * np.cos(2 * np.pi * Kx / X)) / voxel_size[0]**2 + \
                (2.0 - 2 * np.cos(2 * np.pi * Ky / Y)) / voxel_size[1]**2
            D = D ** finite_difference_order

            half = int(np.floor(D.shape[number_of_dimensions - 1] / 2)) + 1
            if return_full_matrix:
                return D[:, :half], D
            else:
                return D[:, :half]

    else:

        X, Y, Z = size[0], size[1], size[2]

        if DCT:
            if return_full_matrix:
                Kx, Ky, Kz = np.meshgrid(np.arange(X * 2),
                                         np.arange(Y * 2),
                                         np.arange(Z * 2), indexing='ij')
            else:
                Kx, Ky, Kz = np.meshgrid(np.arange(X),
                                         np.arange(Y),
                                         np.arange(Z), indexing='ij')

            D = (2.0 - 2 * np.cos(np.pi * Kx / X)) / voxel_size[0]**2 + \
                (2.0 - 2 * np.cos(np.pi * Ky / Y)) / voxel_size[1]**2 + \
                (2.0 - 2 * np.cos(np.pi * Kz / Z)) / voxel_size[2]**2
            D = D ** finite_difference_order

            if return_full_matrix:
                return D[:X, :Y, :Z], D
            else:
                return D[:X, :Y, :Z]

        else:  # FFT, real to complex

            Kx, Ky, Kz = np.meshgrid(np.arange(X),
                                     np.arange(Y),
                                     np.arange(Z), indexing='ij')
            D = (2.0 - 2 * np.cos(2 * np.pi * Kx / X)) / voxel_size[0]**2 + \
                (2.0 - 2 * np.cos(2 * np.pi * Ky / Y)) / voxel_size[1]**2 + \
                (2.0 - 2 * np.cos(2 * np.pi * Kz / Z)) / voxel_size[2]**2
            D = D ** finite_difference_order

            half = int(np.floor(D.shape[number_of_dimensions - 1] / 2)) + 1
            if return_full_matrix:
                return D[..., :half], D
            else:
                return D[..., :half]


def compute_features(signal, max_features, smooth_features=False, number_of_dimensions=3,
                     include_image_as_feature=False, smooth_sigma=1.0):

    #
    filtered = np.zeros([signal.shape[0], signal.shape[1], signal.shape[2], max_features])

    if include_image_as_feature:
        # Add signal as first feature
        filtered[..., 0] = signal
        f = 1
    else:
        f = 0

    if not f >= max_features:
        if number_of_dimensions == 2:
            filtered[..., 0, f] = ndimage.sobel(signal[:, :, 0])
        else:
            filtered[..., f] = ndimage.sobel(signal)
        f = f + 1
    if not f >= max_features:
        if number_of_dimensions == 2:
            filtered[..., 0, f] = ndimage.prewitt(signal[:, :, 0])
        else:
            filtered[..., f] = ndimage.prewitt(signal)
        f = f + 1
    if not f >= max_features:
        if number_of_dimensions == 2:
            filtered[..., 0, f] = ndimage.laplace(signal[:, :, 0])
        else:
            filtered[..., f] = ndimage.laplace(signal)
        f = f + 1

    # TODO: which filter to use? What if we convolve two times with different filters
    f1_3x3 = np.array([1, 0, -1])
    f2_3x3 = np.array([1, -2, 1])
    f3_3x3 = np.array([1, 2, 1])

    filters_3x3 = np.vstack((f1_3x3, f2_3x3, f3_3x3))

    # Just create kernel from a combination of 1x3 filters
    if number_of_dimensions == 2:
        for f_1 in filters_3x3:
            for f_2 in filters_3x3:
                if f >= max_features:
                    continue
                kernel = (f_1[:, None].T * f_2[:, None])
                print("Convolving...")
                filtered[:, :, 0, f] = ndimage.convolve(signal[:, :, 0], kernel, mode='constant')
                if smooth_features:
                    # Smooth features
                    filtered[:, :, 0, f] = ndimage.gaussian_filter(filtered[:, :, 0, f],
                                                                   sigma=(smooth_sigma, smooth_sigma),
                                                                   order=0)
                f = f + 1
    else:
        for f_1 in filters_3x3:
            for f_2 in filters_3x3:
                for f_3 in filters_3x3:
                    if f >= max_features:
                        continue
                    kernel = (f_1[:, None].T * f_2[:, None]) * f_3[:, None, None]
                    print("Convolving...")
                    filtered[..., f] = ndimage.convolve(signal, kernel, mode='constant')
                    if smooth_features:
                        # Smooth features
                        filtered[..., f] = ndimage.gaussian_filter(filtered[..., f],
                                                                   sigma=(smooth_sigma, smooth_sigma, smooth_sigma),
                                                                   order=0)

                    f = f + 1

    return filtered


# Save likelihood parameters
def save_likelihood_parameters(registrator, outputDir):

    if registrator.metric == 'MI':
        parameters = {
            'metric': registrator.metric,
            'theta': registrator.theta,
            'alpha': registrator.alpha
        }
    elif registrator.metric == 'SSD' and not (registrator.feature_moving or registrator.feature_fixed):
        parameters = {
            'metric': registrator.metric,
            'sigma_sq': registrator.sigma_sq,
            'scale_shift': registrator.scale_shift,
            'scale': registrator.scale,
            'shift': registrator.shift,
            'alpha_0': registrator.alpha_0,
            'beta_0': registrator.beta_0,
            'number_of_bins': registrator.num_voxel_intensity_levels,
        }
    else:
        parameters = {
            'metric': 'Features',
            'sigma_sq': registrator.sigma_sq,
            'alpha_0': registrator.alpha_0,
            'beta_0': registrator.beta_0,
            'coeff': registrator.coeff,
            'include_image_as_feature': registrator.include_image_as_feature
        }

    np.savez(os.path.join(outputDir, 'likelihood_parameters.npz'), **parameters)
