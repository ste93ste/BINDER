import os
import numpy as np
from BINDER import FastRegistration
import time
from skimage.transform import warp

def test_2D_MI_registration():

    # Initialize two images with different sizes
    fixed = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 2, 3, 1, 0, 0],
             [0, 0, 1, 1, 1, 2, 3, 1, 0, 0],
             [0, 0, 1, 1, 1, 2, 3, 1, 0, 0],
             [0, 0, 1, 1, 1, 2, 3, 1, 0, 0],
             [0, 0, 1, 1, 1, 2, 3, 1, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    moving = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 2, 3, 1, 0],
              [0, 0, 1, 1, 1, 2, 3, 1, 0],
              [0, 0, 1, 1, 1, 2, 3, 1, 0],
              [0, 0, 1, 1, 1, 2, 3, 1, 0],
              [0, 0, 1, 1, 1, 2, 3, 1, 0],
              [0, 0, 1, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    num_bins = 4

    fixed = np.array(fixed)
    moving = np.array(moving)
    resolutions = [1]
    seed = 12345
    n_iterations = 100
    n_burnins = 0
    n_samples = 100
    gamma = 100
    visualizer = False
    threads = 1

    registrator = FastRegistration(voxels=fixed,
                                   nodes=moving,
                                   num_voxel_intesity_levels=num_bins,
                                   num_node_intensity_levels=num_bins,
                                   gamma=gamma,
                                   visualizer=visualizer,
                                   threads=threads,
                                   metric='MI',
                                   seed=seed,
                                   alpha=1.1,
                                   bin_robust=False,
                                   topology_correction=False
                                   )

    print("Start")
    t = time.time()
    _ = registrator.run(resolutions=resolutions, max_number_of_iterations=n_iterations)
    elapsed = time.time() - t
    print("Time for EM: " + str(elapsed))

    warped_EM = warp(moving, np.moveaxis(registrator.deformedVoxelPositions, 3, 0)[0:2, :, :, 0], order=0,
                     output_shape=fixed.shape, preserve_range=True)

    # We assume that the method can perfectly register the two images
    assert (np.allclose(warped_EM, fixed))

    meanDefVoxelPos, _, = registrator.sample_param(n_burnins, n_samples)

    warped_MCMC = warp(moving, np.moveaxis(meanDefVoxelPos, 3, 0)[0:2, :, :, 0], order=0, output_shape=fixed.shape,
                       preserve_range=True)

    # Same when using MCMC
    assert (np.allclose(warped_MCMC, fixed))
