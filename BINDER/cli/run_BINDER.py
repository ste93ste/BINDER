import argparse
from skimage.transform import resize
import nibabel as nib
import os
import numpy as np
import time
from BINDER import FastRegistration, utils

def parseArguments():

    # ------ Parse Command Line Arguments ------

    parser = argparse.ArgumentParser()

    default_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    # required
    parser.add_argument('-o', '--output', metavar='DIR', help='Output directory.', required=True)
    parser.add_argument('-f', '--fixed', dest='fixed', help='Fixed image.', required=True)
    parser.add_argument('-m', '--moving', dest='moving', help='Moving image.', required=True)
    # optional processing options
    # masking options
    parser.add_argument('--mask-fixed', type=str, help='Mask to apply to the fixed image.')
    parser.add_argument('--mask-moving', type=str, help='Mask to apply to the moving image.')
    parser.add_argument('--automatic-mask', action='store_true', default=False, help='Estimate mask for both images (see mask-below).')
    parser.add_argument('--mask-below', type=float, default=0, help='Mask values below a threshold.')
    parser.add_argument('--save-masks', action='store_true', default=False, help='Save masks.')
    #
    parser.add_argument('--iterations', type=int, default=1000, help='Maximum number of iterations per resolution.')
    parser.add_argument('--convergence-th', type=float, default=2.5e-5, help='Convergence threshold.')
    parser.add_argument('--do-not-increase-gamma', action='store_true', default=False, help='Do not increase gamma at each DS by DS^D, with DS downsampling factor and D dimension.')
    parser.add_argument('--save-warped-image', action='store_true', default=False, help='Save warped image.')
    parser.add_argument('--save-binned-images', action='store_true', default=False, help='Save binned images.')
    parser.add_argument('--save-jacobian-determinant', action='store_true', default=False, help='Save jacobian determinant image.')
    parser.add_argument('--output-interpolation', type=int, default=3, help='Interpolation order for warped image.')
    parser.add_argument('--initial-transformation', type=str, help='Initial transformation (.mat file).')
    parser.add_argument('--do-not-initialize-center-of-mass', action='store_true', default=False, help='Do not use center of mass initialization if image headers are different.')
    # MI options
    parser.add_argument('--MI', action='store_true', default=True, help='Enable mutual information metric.')
    parser.add_argument('--bins-fixed', type=int, default=64, help='Number of bins for the fixed image.')
    parser.add_argument('--bins-moving', type=int, default=64, help='Number of bins for the fixed image.')
    parser.add_argument('--do-not-bin-robust', action='store_true', default=False, help='Do not bin within 1- and 99-percentile.')
    parser.add_argument('--alpha', type=float, default=2.0, help='Prior on theta (for each k).')
    # SSD options
    # parser.add_argument('--SSD', action='store_true', default=False, help='Enable sum of squares difference metric.')
    # parser.add_argument('--SSD-bins', type=int, default=512, help='Number of bins for SSD metric.')
    # parser.add_argument('--alpha-zero', type=float, default=1000, help='Prior on sigma, (shape) IG(sigma^2 | alpha_0, beta_0.')
    # parser.add_argument('--beta-zero', type=float, default=1, help='Prior on sigma, (scale) IG(sigma^2 | alpha_0, beta_0.')
    # parser.add_argument('--histogram-matching', action='store_true', default=False, help='Pre-process data with histogram matching.')
    # parser.add_argument('--no-estimate-scale-and-shift', action='store_true', default=False, help='Do not estimate scale and shift parameters.')
    # Filter/Features options
    # parser.add_argument('--features-moving', action='store_true', default=False, help='Moving image is a linear combination of image features.')
    # parser.add_argument('--features-fixed', action='store_true', default=False, help='Fixed image is a linear combination of image features.')
    # parser.add_argument('--precomputed-features-moving', help='Precomputed filters for moving image.')
    # parser.add_argument('--precomputed-features-fixed', help='Precomputed filters for fixed image.')
    # parser.add_argument('--number-of-filters-moving', type=int, default=12, help='Number of filters for moving image.')
    # parser.add_argument('--number-of-filters-fixed', type=int, default=12, help='Number of filters for fixed image.')
    # parser.add_argument('--do-not-include-image-as-feature', action='store_true', default=False, help='Do not include image as feature.')
    # parser.add_argument('--fixed-features-reg', action='store_true', default=False, help='Perform registration across features. Coefficients are fixed.')
    # parser.add_argument('--save-warped-features-images', action='store_true', default=False, help='Save warped features images.')
    # Linear registration options
    # parser.add_argument('--affine', action='store_true', default=False, help='Perform affine registration.')
    # parser.add_argument('--rigid', action='store_true', default=False, help='Perform rigid registration.')
    # Topology correction options
    parser.add_argument('--no-topology-correction', action='store_true', default=False, help='Perform topology-correction.')
    parser.add_argument('--e1', type=float, default=0.01, help='Mininum value for Jacobian determinant.')
    parser.add_argument('--e2', type=float, default=100, help='Maximum value for Jacobian determinant.')
    parser.add_argument('--max_outer_it', type=int, default=10, help='Maximum number of outer iterations.')
    parser.add_argument('--max_inner_it', type=int, default=10, help='Maximum number of inner iterations.')
    #
    parser.add_argument('--resolutions', nargs='+', type=int, default=[8, 4, 2, 1], help='Resolution levels.')
    parser.add_argument('--gamma', type=float, default=1e-6, help='Smoothing factor per voxel.')
    parser.add_argument('--bspline-order', type=int, default=3, help='B-spline order.')
    parser.add_argument('--finite-difference-order', type=int, default=2, help='Finite difference order.')
    parser.add_argument('--smooth', action='store_true', default=False, help='Smooth images.')
    parser.add_argument('--upsample-in-frequency-domain', action='store_true', default=False, help='Upsample deformation field in frequency domain')
    parser.add_argument('--threads', type=int, default=default_threads, help='Number of threads to use.')
    parser.add_argument('--save-likelihood-parameters', action='store_true', default=False, help='Save likelihood parameters.')
    parser.add_argument('--visualizer', action='store_true', default=False, help='Turn visualizer on.')
    # sampler options
    parser.add_argument('--sampler', action='store_true', default=False, help='Enable sampler mode.')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
    parser.add_argument('--number-of-burnin', type=int, default=50, help='Number of burn-in.')
    parser.add_argument('--number-of-samples', type=int, default=50, help='Number of samples.')
    parser.add_argument('--sample-gamma', action='store_true', default=False, help='Sample also gamma parameter.')
    parser.add_argument('--sample-gamma-every', type=int, default=1, help='Sample gamma parameter every X sweeps.')
    parser.add_argument('--do-not-sample-posteriors', action='store_true', default=False, help='Do not sample posteriors. Posterior will be kept constant.')
    parser.add_argument('--do-not-sample-likelihood-parameters', action='store_true', default=False, help='Do not sample likelihood parameters. Likelihood parameters will be kept constant.')
    parser.add_argument('--do-not-sample-deformation-parameters', action='store_true', default=False, help='Do not sample deformation parameters. Deformation parameters will be kept constant.')
    args = parser.parse_args()

    return args


def main():

    args = parseArguments()

    # ------ Initial Setup ------
    # Create the output folder
    os.makedirs(args.output, exist_ok=True)

    voxels = nib.load(args.fixed).get_fdata()
    nodes = nib.load(args.moving).get_fdata()
    fixed_affine = nib.load(args.fixed).affine
    moving_affine = nib.load(args.moving).affine

    if args.mask_fixed is not None:
        print("Loading mask for the fixed image")
        mask_fixed = nib.load(args.mask_fixed).get_fdata()
        print("Done")
    else:
        mask_fixed = None

    if args.mask_moving is not None:
        print("Loading mask for the moving image")
        mask_moving = nib.load(args.mask_moving).get_fdata()
        print("Done")
    else:
        mask_moving = None

    #if args.SSD:
    #    metric = 'SSD'
    #    bins_fixed = args.SSD_bins
    #    bins_moving = args.SSD_bins

    #else:
        metric = 'MI'
        bins_fixed = args.bins_fixed
        bins_moving = args.bins_moving

    #if args.precomputed_features_moving is not None:
    #    print("Loading precomputed features for the moving image")
    #    precomputed_features_moving = nib.load(args.precomputed_features_moving).get_fdata()
    #    print("Done")
    #else:
    #    precomputed_features_moving = None

    #if args.precomputed_features_fixed is not None:
    #    print("Loading precomputed features for the fixed image")
    #    precomputed_features_fixed = nib.load(args.precomputed_features_fixed).get_fdata()
    #    print("Done")
    #else:
    #    precomputed_features_fixed = None

    if args.initial_transformation is not None:
        import surfa as sf
        initial_transformation = sf.load_affine(args.initial_transformation)
    else:
        initial_transformation = None

    # Initialize
    registrator = FastRegistration(voxels=voxels,
                                   nodes=nodes,
                                   voxels_matrix=fixed_affine,
                                   nodes_matrix=moving_affine,
                                   mask_voxels=mask_fixed,
                                   mask_nodes=mask_moving,
                                   use_masks=args.automatic_mask,
                                   mask_threshold=args.mask_below,
                                   num_voxel_intesity_levels=bins_fixed,
                                   num_node_intensity_levels=bins_moving,
                                   bin_robust=not args.do_not_bin_robust,
                                   gamma=args.gamma,
                                   metric=metric,
                                   scale_shift=not args.no_estimate_scale_and_shift,
                                   histogram_matching=args.histogram_matching,
                                   spline_order=args.bspline_order,
                                   topology_correction=not args.no_topology_correction,
                                   finite_difference_order=args.finite_difference_order,
                                   alpha=args.alpha,
                                   alpha_0=args.alpha_zero,
                                   beta_0=args.beta_zero,
                                   visualizer=args.visualizer,
                                   threads=args.threads,
                                   convergence_th=args.convergence_th,
                                   #feature_moving=args.features_moving,
                                   #feature_fixed=args.features_fixed,
                                   number_of_filters_fixed=args.number_of_filters_fixed,
                                   number_of_filters_moving=args.number_of_filters_moving,
                                   #fixed_features_reg=args.fixed_features_reg,
                                   seed=args.seed,
                                   affine=args.affine,
                                   rigid=args.rigid,
                                   e1=args.e1,
                                   e2=args.e2,
                                   max_outer_it=args.max_outer_it,
                                   max_inner_it=args.max_inner_it,
                                   smooth=args.smooth,
                                   upsample_in_frequency_domain=args.upsample_in_frequency_domain,
                                   #precomputed_features_moving=precomputed_features_moving,
                                   #precomputed_features_fixed=precomputed_features_fixed,
                                   initial_transformation=initial_transformation,
                                   include_image_as_feature=not args.do_not_include_image_as_feature,
                                   initialize_with_center_of_mass=not args.do_not_initialize_center_of_mass,
                                   increase_gamma_at_DS_resolution=not args.do_not_increase_gamma
                                   )

    # Run registration
    start = time.time()
    registrator.run(resolutions=args.resolutions, max_number_of_iterations=args.iterations)
    end = time.time()
    print("Time for EM: " + str(end - start) + " seconds")

    # Perform sampling, if enabled
    if args.sampler:

        print("Running Gibbs sampler")
        start = time.time()
        mean_warp, cov_field = registrator.sample_param(args.number_of_burnin,
                                                                 args.number_of_samples,
                                                                 int(args.sample_gamma),
                                                                 args.sample_gamma_every,
                                                                 not args.do_not_sample_posteriors,
                                                                 not args.do_not_sample_likelihood_parameters,
                                                                 not args.do_not_sample_deformation_parameters,
                                                                 )
        end = time.time()
        print("Time for Gibbs sampler: " + str(end - start) + " seconds")

    # Make sure we are at resolution 1 (native)
    if registrator.current_resolution != 1:

        if args.sampler:
            # Resize also mean and std warps
            mean_warp = resize(mean_warp - registrator.voxels_pos, registrator.orig_voxels_shape,
                               order=1) * registrator.current_resolution
            mean_warp += registrator.full_voxels_pos
            #
            cov_field = resize(cov_field, registrator.orig_voxels_shape,
                                        order=1) * registrator.current_resolution

    #if args.save_warped_features_images:
    #    if args.features_fixed == 0:
    #        number_of_filters_fixed = 1
    #    else:
    #        number_of_filters_fixed = args.number_of_filters_fixed
    #    for feature_fixed in range(number_of_filters_fixed):
    #        # Save warped feature moving image
    #        tmp = np.sum(registrator.binned_nodes * registrator.coeff[feature_fixed, :], axis=-1)
    #        utils.save_warped_image(deformed_position=registrator.deformedVoxelPositions,
    #                                movingImage=tmp,
    #                                order=args.output_interpolation,
    #                                affine=fixed_affine,
    #                                outputDir=args.output,
    #                                outputName='warped_fixed_feature_' + str(feature_fixed + 1) + '.nii.gz')
    #        # Save warped feature fixed image
    #        utils.save_image(generalImage=registrator.binned_voxels[:, :, :, feature_fixed], affine=fixed_affine,
    #                         outputDir=args.output, outputName='fixed_feature_' + str(feature_fixed + 1) + '.nii.gz')

    # Save deformation displacement
    utils.save_warp(deformations=registrator.deformedVoxelPositions - registrator.voxels_pos,
                    outputDir=args.output, affine=fixed_affine)

    # Save deformation field
    utils.save_image(generalImage=registrator.deformedVoxelPositions, outputDir=args.output, affine=fixed_affine,
                     outputName='deformation_field.nii.gz')

    # Save jacobian determinant image, if requested
    if args.save_jacobian_determinant:
        from BINDER.Registration import compute_jacobian_determinant_3D
        tmp = registrator.deformedVoxelPositions
        tmp = np.ascontiguousarray(tmp)
        # go to physical domain
        tmp = tmp / registrator.voxels_voxel_size[None, None, None, :]
        jac_det = compute_jacobian_determinant_3D(tmp.ravel(order="C"), tmp.shape[0], tmp.shape[1], tmp.shape[2],
                                                  registrator.voxels_voxel_size[0], registrator.voxels_voxel_size[1],
                                                  registrator.voxels_voxel_size[2])
        utils.save_image(generalImage=jac_det, outputDir=args.output, affine=fixed_affine,
                         outputName='jacobian_determinant.nii.gz')

    # Save warped image, if requested
    if args.save_warped_image:
        utils.save_warped_image(deformed_position=registrator.deformedVoxelPositions,
                                movingImage=nodes,
                                order=args.output_interpolation,
                                affine=fixed_affine,
                                outputDir=args.output)

    # Save binned images, if requested
    if args.save_binned_images:
        # Moving
        utils.save_warped_image(deformed_position=registrator.deformedVoxelPositions,
                                movingImage=registrator.full_binned_nodes,
                                order=0,
                                affine=fixed_affine,
                                outputDir=args.output,
                                outputName='binned_warped.nii.gz')
        # Fixed
        utils.save_image(generalImage=registrator.full_binned_voxels, outputDir=args.output, affine=fixed_affine,
                         outputName='binned_fixed.nii.gz')

    if args.sampler:
        utils.save_warp(deformations=mean_warp - registrator.voxels_pos,
                        outputDir=args.output, affine=fixed_affine, outputName='mean_warp_sample.nii.gz')
        utils.save_image(generalImage=mean_warp, outputDir=args.output, affine=fixed_affine,
                         outputName='mean_deformation_field_sample.nii.gz')
        utils.save_image(generalImage=cov_field, outputDir=args.output, affine=fixed_affine,
                         outputName='covariance_field_sample.nii.gz')

        if args.save_warped_image:
            utils.save_warped_image(deformed_position=mean_warp,
                                    movingImage=nodes,
                                    order=args.output_interpolation,
                                    affine=fixed_affine,
                                    outputDir=args.output,
                                    outputName='warped_sample.nii.gz')

    if args.save_masks:
        # Fixed
        utils.save_image(generalImage=registrator.full_mask_voxels, outputDir=args.output, affine=fixed_affine,
                         outputName='mask_fixed.nii.gz')
        # Moving
        utils.save_image(generalImage=registrator.full_mask_nodes, outputDir=args.output, affine=fixed_affine,
                         outputName='mask_moving.nii.gz')

    if args.save_likelihood_parameters:
        utils.save_likelihood_parameters(registrator, args.output)

    print("Done! FASTMIRegistration exit WITHOUT ERRORS.")


if __name__ == '__main__':
    main()
