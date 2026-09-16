import argparse
import nibabel as nib
import numpy as np
import os
import sys
from BINDER import utils

def parseArguments():
    parser = argparse.ArgumentParser()

    default_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    # required
    parser.add_argument('-o', '--outputDir', metavar='DIR', help='Output directory.', required=True)
    parser.add_argument('-m', '--moving', dest='moving', help='Moving image.', required=True)

    parser.add_argument('--disp', help='Displacement field.')
    parser.add_argument('--transf', help='Transformation field.')
    parser.add_argument('--compute-jacobian-determinant', default=False, action='store_true', help='Compute jacobian determinant')
    parser.add_argument('--name', default='warped.nii.gz', help='Output file name')
    parser.add_argument('--order', type=int, default=3, help="Interpolation order.")
    parser.add_argument('--initial-affine', help="Initial affine transformation (only LTA from FreeSurfer supported atm.")

    # Topology correction
    parser.add_argument('--topology-correction', default=False, action='store_true', help="Perform topology correction.")
    parser.add_argument('--e1', type=float, default=0.01, help='Mininum value for Jacobian determinant.')
    parser.add_argument('--e2', type=float, default=100, help='Maximum value for Jacobian determinant.')
    parser.add_argument('--max_outer_it', type=int, default=10, help='Maximum number of outer iterations.')
    parser.add_argument('--max_inner_it', type=int, default=10, help='Maximum number of inner iterations.')

    parser.add_argument('--threads', type=int, default=default_threads, help='Number of threads to use.')

    args = parser.parse_args()

    return args

def main():
    args = parseArguments()

    if not args.disp and not args.transf:
        sys.exit("Error: You must provide a displacement or transformation field.")

    if args.compute_jacobian_determinant and not (args.disp or args.transf):
        sys.exit("Error: You must provide either a displacement or transformation field to compute the jacobian determinant.")

    if args.topology_correction and not (args.disp or args.transf):
        sys.exit("Error: You must provide either a displacement or transformation field for the topology correction algorithm.")

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir, exist_ok=True)

    # Load moving image
    moving_image = nib.load(args.moving).get_fdata()

    if args.initial_affine is not None:
        import surfa as sf
        initial_affine = sf.load_affine(args.initial_affine)

    # Load and prepare transformation field
    if args.transf:
        deformed_positions = nib.load(args.transf).get_fdata()
        fixed_affine = nib.load(args.transf).affine
    elif args.disp:
        warp = nib.load(args.disp).get_fdata()
        indices = np.moveaxis(np.indices(warp.shape[0:3]), 0, 3)
        fixed_affine = nib.load(args.disp).affine
        deformed_positions = warp + indices

        # Avoid resampling twice so concatenate linear + nonlinear deformation
        if args.initial_affine is not None:
            # Assuming FreeSurfer RAS to RAS format for initial_affine
            deformed_positions = nib.affines.apply_affine(np.linalg.inv(nib.load(args.moving).affine)
                                                          @ np.linalg.inv(initial_affine)
                                                          @ fixed_affine, deformed_positions)

    # Save warped image
    utils.save_warped_image(deformed_position=deformed_positions, outputDir=args.outputDir,
                            movingImage=moving_image, affine=fixed_affine, outputName=args.name, order=args.order)

    # Compute Jacobian determinant if requested
    if args.compute_jacobian_determinant:
        from Reg.Registration import compute_jacobian_determinant_3D
        voxel_size = np.sum(fixed_affine[0:3, 0:3] ** 2, axis=0) ** (1 / 2)
        field_for_jacobian = np.ascontiguousarray(deformed_positions)
        # go to physical domain
        field_for_jacobian = field_for_jacobian / voxel_size[None, None, None, :]
        J = compute_jacobian_determinant_3D(field_for_jacobian.ravel(order="C"),
                                            field_for_jacobian.shape[0],
                                            field_for_jacobian.shape[1],
                                            field_for_jacobian.shape[2],
                                            voxel_size[0], voxel_size[1], voxel_size[2])
        utils.save_image(generalImage=J, outputDir=args.outputDir, affine=fixed_affine, outputName='Jacobian.nii.gz')
        print("Jacobian determinant image saved")

    # Perform topology correction of the displacement or transformation field, if requested
    if args.topology_correction:
        from Reg.Registration import topology_correction_3D
        field = np.ascontiguousarray(deformed_positions)
        top_cor_obj = topology_correction_3D()
        if args.transf:
            out_name = "top_cor_transf.nii.gz"
        else:
            out_name = "top_cor_warp.nii.gz"

        voxel_size = np.sum(fixed_affine[0:3, 0:3] ** 2, axis=0) ** (1 / 2)
        field_corrected = top_cor_obj.correct_topology(field.ravel(order="C"), args.e1, args.e2,
                                                       args.max_outer_it, args.max_inner_it,
                                                       args.threads, voxel_size,
                                                       field.shape[0], field.shape[1], field.shape[2])
        if not args.transf:
            indices = np.moveaxis(np.indices(field.shape[0:3]), 0, 3)
            field_corrected = field_corrected - indices

        utils.save_warp(field_corrected, args.outputDir, fixed_affine, out_name)
        print("Topology corrected field saved")

if __name__ == '__main__':
    main()
