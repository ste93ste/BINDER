import os
from BINDER import FastRegistration, utils
import pytest
import time
import nibabel as nib
import os
import numpy as np
from BINDER import BINDERDIR


@pytest.fixture
def tmppath(tmpdir):
    return str(tmpdir)


@pytest.fixture(scope='module')
def test_T1_1():
    fn = nib.load(os.path.join(BINDERDIR, "_internal_resources", "testing_files", "3DTestImages", "T1w_1_8mm.nii.gz"))
    return fn


@pytest.fixture(scope='module')
def test_T1_2():
    fn = nib.load(os.path.join(BINDERDIR, "_internal_resources", "testing_files", "3DTestImages", "T1w_2_8mm.nii.gz"))
    return fn


def test_3D_MI_registration(test_T1_1, test_T1_2, tmppath):

    fixed = test_T1_1.get_fdata()
    fixed_affine = test_T1_1.affine
    moving = test_T1_2.get_fdata()
    moving_affine = test_T1_2.affine

    threads = 1
    spline_order = 3

    resolutions = [1]
    n_iterations = 3
    n_burnins = 0
    n_samples = 20
    gamma = 1e-4
    outputDir = tmppath
    visualizer = False
    sampler = True

    registrator = FastRegistration(voxels=fixed,
                                   nodes=moving,
                                   voxels_matrix=fixed_affine,
                                   nodes_matrix=moving_affine,
                                   num_voxel_intesity_levels=32,
                                   num_node_intensity_levels=32,
                                   gamma=gamma,
                                   metric='MI',
                                   visualizer=visualizer,
                                   threads=threads,
                                   spline_order=spline_order,
                                   topology_correction=False)

    print("Start")
    t = time.time()
    _ = registrator.run(resolutions=resolutions, max_number_of_iterations=n_iterations)
    elapsed = time.time() - t
    print("Time for EM: " + str(elapsed))

    utils.save_warped_image(deformed_position=registrator.deformedVoxelPositions, movingImage=moving,
                            outputDir=outputDir, affine=fixed_affine)
    utils.save_warp(deformations=registrator.deformedVoxelPositions - registrator.voxels_pos,
                    outputDir=outputDir, affine=fixed_affine)

    # Load warped
    warped = nib.load(os.path.join(outputDir, "warped.nii.gz")).get_fdata()

    # Check that we are actually registering the two images
    # Instead of looking at MI, we are looking at SSD here (the two images are CTs)
    starting_SSD = np.sum((moving - fixed)**2)
    final_SSD_EM = np.sum((warped - fixed)**2)
    assert (final_SSD_EM < starting_SSD)

    if sampler:
        mean_samples, cov_field = registrator.sample_param(burnin=n_burnins,
                                                           samples=n_samples,
                                                           sample_gamma=False)

        utils.save_warped_image(deformed_position=mean_samples, movingImage=moving,
                                outputDir=outputDir, affine=fixed_affine, outputName='mean_warped.nii.gz')

        # Load warped
        warped = nib.load(os.path.join(outputDir, "mean_warped.nii.gz")).get_fdata()

        # Check that we are actually registering the two images
        # Instead of looking at MI, we are looking at SSD here (the two images are CTs)
        starting_SSD = np.sum((moving - fixed)**2)
        final_SSD_MCMC = np.sum((warped - fixed)**2)
        assert (final_SSD_MCMC < starting_SSD)
        # Also check that sampling is bringing us closer since we use few iterations in the EM
        assert (final_SSD_MCMC < final_SSD_EM)

@pytest.mark.skip(reason="Skipping MI filter for now")
def test_3D_MI_registration_Filters(test_T1_1, test_T1_2, tmppath):

    fixed = test_T1_1.get_fdata()
    fixed_affine = test_T1_1.affine
    moving = test_T1_2.get_fdata()
    moving_affine = test_T1_2.affine

    threads = 1
    spline_order = 2

    resolutions = [1]
    n_iterations = 3
    n_burnins = 0
    n_samples = 20
    gamma = 1e-4
    visualizer = False
    sampler = True
    outputDir = tmppath
    number_of_filters_fixed = 2
    number_of_filters_moving = 2
    feature_fixed = True
    feature_moving = True

    registrator = FastRegistration(voxels=fixed,
                                   nodes=moving,
                                   voxels_matrix=fixed_affine,
                                   nodes_matrix=moving_affine,
                                   num_voxel_intesity_levels=32,
                                   num_node_intensity_levels=32,
                                   gamma=gamma,
                                   metric='MI',
                                   visualizer=visualizer,
                                   threads=threads,
                                   spline_order=spline_order,
                                   topology_correction=False,
                                   number_of_filters_fixed=number_of_filters_fixed,
                                   number_of_filters_moving=number_of_filters_moving,
                                   feature_fixed=feature_fixed,
                                   feature_moving=feature_moving)

    print("Start")
    t = time.time()
    _ = registrator.run(resolutions=resolutions, max_number_of_iterations=n_iterations)
    elapsed = time.time() - t
    print("Time for EM: " + str(elapsed))

    utils.save_warped_image(deformed_position=registrator.deformedVoxelPositions, movingImage=moving,
                            outputDir=outputDir, affine=fixed_affine)
    utils.save_warp(deformations=registrator.deformedVoxelPositions - registrator.voxels_pos,
                    outputDir=outputDir, affine=fixed_affine)

    # Load warped
    warped = nib.load(os.path.join(outputDir, "warped.nii.gz")).get_fdata()

    # Check that we are actually registering the two images
    # Instead of looking at MI, we are looking at SSD here (the two images are CTs)
    starting_SSD = np.sum((moving - fixed)**2)
    final_SSD_EM = np.sum((warped - fixed)**2)
    assert (final_SSD_EM < starting_SSD)

    if sampler:
        mean_samples, cov_field = registrator.sample_param(burnin=n_burnins,
                                                           samples=n_samples)

        utils.save_warped_image(deformed_position=mean_samples, movingImage=moving,
                                outputDir=outputDir, affine=fixed_affine, outputName='mean_warped.nii.gz')

        # Load warped
        warped = nib.load(os.path.join(outputDir, "mean_warped.nii.gz")).get_fdata()

        # Check that we are actually registering the two images
        # Instead of looking at MI, we are looking at SSD here (the two images are CTs)
        starting_SSD = np.sum((moving - fixed)**2)
        final_SSD_MCMC = np.sum((warped - fixed)**2)
        assert (final_SSD_MCMC < starting_SSD)
        # Also check that sampling is bringing us closer since we use few iterations in the EM
        assert (final_SSD_MCMC < final_SSD_EM)
