from BINDER import FastRegistration, utils
import time
import nibabel as nib
import os
from BINDER import BINDERDIR

test_T1 = nib.load(os.path.join(BINDERDIR, "_internal_resources", "testing_files", "3DTestImages", "T1w_1_2mm.nii.gz"))
test_T1_2 = nib.load(os.path.join(BINDERDIR, "_internal_resources", "testing_files", "3DTestImages", "T1w_2_2mm.nii.gz"))

fixed = test_T1.get_fdata()
fixed_affine = test_T1.affine
moving = test_T1_2.get_fdata()
moving_affine = test_T1_2.affine

threads = 8
spline_order = 3

resolutions = [4, 2, 1]
n_iterations = 200
n_burnins = 0
n_samples = 10000
gamma = 5e-7
outputDir = ""
visualizer = True
sampler = True

registrator = FastRegistration(voxels=fixed,
                               nodes=moving,
                               voxels_matrix=fixed_affine,
                               nodes_matrix=moving_affine,
                               num_voxel_intesity_levels=64,
                               num_node_intensity_levels=64,
                               gamma=gamma,
                               metric='MI',
                               visualizer=visualizer,
                               threads=threads,
                               spline_order=spline_order,
                               topology_correction=True,
                               smooth=False,
                               finite_difference_order=2,
                               debug=0)

print("Start")
t = time.time()
_ = registrator.run(resolutions=resolutions, max_number_of_iterations=n_iterations)
elapsed = time.time() - t
print("Time for EM: " + str(elapsed))

registrator.move_resolution(1)

utils.save_warped_image(deformed_position=registrator.deformedVoxelPositions, movingImage=moving,
                        outputDir=outputDir, affine=fixed_affine)
utils.save_warp(deformations=registrator.deformedVoxelPositions - registrator.voxels_pos,
                outputDir=outputDir, affine=fixed_affine)

if sampler:
    mean_samples, cov_field = registrator.sample_param(burnin=n_burnins,
                                                                   samples=n_samples,
                                                                   sample_posteriors=1,
                                                                   sample_likelihood_parameters=1,
                                                                   sample_transformation_parameters=1,
                                                                   sample_gamma=True,
                                                                   sample_gamma_every=1)

    utils.save_warped_image(deformed_position=mean_samples, movingImage=moving,
                            outputDir=outputDir, affine=fixed_affine, outputName='mean_warped.nii.gz')
