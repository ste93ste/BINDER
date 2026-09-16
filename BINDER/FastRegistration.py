import sys
import numpy as np
import nibabel as nib
from skimage.transform import resize
import os
from . import utils

def get_registration_module():
    import importlib

    try:
        return importlib.import_module("BINDER._core")
    except ImportError as exc:
        raise ImportError(
            "Could not import the Reg registration extension. "
            "Make sure the package was built successfully."
        ) from exc


Registration = get_registration_module()


##############################
# "voxels" image is registered into another image "nodes"
##############################
class FastRegistration:
    def __init__(self,
                 voxels,
                 nodes,
                 voxels_matrix=np.eye(4),
                 nodes_matrix=np.eye(4),
                 mask_voxels=None,
                 mask_nodes=None,
                 use_masks=False,
                 mask_threshold=0,
                 num_voxel_intesity_levels=64,
                 num_node_intensity_levels=64,
                 alpha=2.0,
                 alpha_0=1,
                 beta_0=1,
                 gamma=5e-7,
                 metric='MI',
                 histogram_matching=False,
                 scale_shift=True,
                 scale=1,
                 shift=0,
                 spline_order=3,
                 finite_difference_order=2,
                 topology_correction=True,
                 visualizer=False,
                 feature_moving=False,
                 feature_fixed=False,
                 precomputed_features_moving=None,
                 precomputed_features_fixed=None,
                 number_of_filters_moving=1,
                 number_of_filters_fixed=1,
                 fixed_features_reg=False,
                 affine=False,
                 rigid=False,
                 smooth=False,
                 upsample_in_frequency_domain=False,
                 convergence_th=5e-5,
                 e1=0.01,
                 e2=100,
                 max_outer_it=10,
                 max_inner_it=10,
                 threads=1,
                 seed=12345,
                 debug=0,
                 update_likelihood_parameters=1,
                 initial_transformation=None,
                 initialize_with_center_of_mass=True,
                 bin_robust=True,
                 include_image_as_feature=True,
                 increase_gamma_at_DS_resolution=False,
                 ):

        self.registratorObj = None

        def __init__(self, *args, **kwargs):
            self._registration = None
            try:
                Registration = get_registration_module(prefer_gpu=kwargs.get('prefer_gpu', True))
                self._registration = Registration(*args, **kwargs)
            except ImportError as e:
                raise RuntimeError(f"Failed to initialize FastRegistration: {e}")

        def __getattr__(self, name):
            if self._registration is None:
                raise AttributeError("No registration module available")
            return getattr(self._registration, name)

        self.D = None
        self.number_of_dimensions = len(nodes.shape)  # Assuming nodes and voxels has the same number of dimension

        self.convergence_th = convergence_th

        self.include_image_as_feature = include_image_as_feature
        self.affine = affine
        self.rigid = rigid
        self.debug = debug
        self.update_likelihood_parameters = update_likelihood_parameters
        self.upsample_in_frequency_domain = upsample_in_frequency_domain
        self.increase_gamma_at_DS_resolution = increase_gamma_at_DS_resolution

        if self.affine:
            self.A = np.eye(3)
            self.t = np.zeros(3)
        if self.rigid:
            self.R = np.eye(3)
            self.t = np.zeros(3)

        self.rigid = rigid
        self.threads = threads
        self.seed = seed

        if topology_correction:
            if not getattr(Registration, "has_topology_correction",
                           hasattr(Registration, "topology_correction_3D")):
                raise RuntimeError(
                    "topology_correction=True was requested, but this build of "
                    "BINDER was compiled without the topology correction algorithm "
                    "(BINDER_DISABLE_TOPOLOGY_CORRECTION=ON). Rebuild without that "
                    "flag to enable it, or pass topology_correction=False."
                )
            if self.number_of_dimensions == 2:
                self.top_cor_obj = Registration.topology_correction_2D()
            else:
                self.top_cor_obj = Registration.topology_correction_3D()
            self.e1 = e1
            self.e2 = e2
            self.max_inner_it = max_inner_it
            self.max_outer_it = max_outer_it
        else:
            self.top_cor_obj = 0

        if smooth:
            self.smooth = True
            self.smooth_spline_object = Registration.Smoothing_Spline(spline_order, threads)
        else:
            self.smooth = False
            self.smooth_spline_object = 0

        self.visualizer = visualizer

        # Create extra fake dimension if 2D images
        if self.number_of_dimensions == 2:
            nodes = nodes[:, :, None]
            voxels = voxels[:, :, None]

        self.voxels = voxels
        self.nodes = nodes

        self.voxels_voxel_size = np.sum(voxels_matrix[0:3, 0:3] ** 2, axis=0) ** (1 / 2)
        self.nodes_voxel_size = np.sum(nodes_matrix[0:3, 0:3] ** 2, axis=0) ** (1 / 2)
        self.current_voxels_voxel_size = self.voxels_voxel_size
        self.current_nodes_voxel_size = self.nodes_voxel_size
        # Gamma per voxel at 1mm resolution
        self.gamma_init = gamma

        if histogram_matching:
            print("Perform histogram matching")
            nodes = utils.hist_norm(nodes, voxels)

        if mask_nodes is None:
            if use_masks:
                self.full_mask_nodes = utils.maskImage(nodes, th_value=mask_threshold)
            else:
                self.full_mask_nodes = np.ones(nodes.shape)
        else:
            self.full_mask_nodes = np.array(mask_nodes)

        if mask_voxels is None:
            if use_masks:
                self.full_mask_voxels = utils.maskImage(voxels, th_value=mask_threshold)
            else:
                self.full_mask_voxels = np.ones(voxels.shape)
        else:
            self.full_mask_voxels = np.array(mask_voxels)

        self.feature_moving = feature_moving
        self.feature_fixed = feature_fixed
        self.number_of_filters_moving = number_of_filters_moving
        self.number_of_filters_fixed = number_of_filters_fixed
        self.number_of_filters_fixed = number_of_filters_fixed

        if precomputed_features_fixed is not None:
            self.number_of_filters_fixed = precomputed_features_fixed.shape[-1]
            if include_image_as_feature:
                self.number_of_filters_fixed += 1  # 1 for original image
        if precomputed_features_moving is not None:
            self.number_of_filters_moving = precomputed_features_moving.shape[-1]
            if include_image_as_feature:
                self.number_of_filters_moving += 1  # 1 for original image
        self.fixed_features_reg = int(fixed_features_reg)

        self.finite_difference_order = finite_difference_order
        self.orig_voxels_shape = np.shape(voxels)
        self.orig_nodes_shape = np.shape(nodes)

        self.num_voxel_intensity_levels = num_voxel_intesity_levels
        self.num_node_intensity_levels = num_node_intensity_levels

        self.metric = metric

        if self.metric == 'MI' and not (self.feature_moving or self.feature_fixed):
            if bin_robust:
                self.full_binned_voxels, self.bin_edges_voxels = utils.bin_signal_robust(voxels * self.full_mask_voxels,
                                                                                         self.num_voxel_intensity_levels)
            else:
                self.full_binned_voxels = utils.bin_signal(voxels * self.full_mask_voxels,
                                                           self.num_voxel_intensity_levels)
            self.full_binned_voxels[np.logical_not(self.full_mask_voxels)] = 0
            if bin_robust:
                self.full_binned_nodes, self.bin_edges_nodes = utils.bin_signal_robust(nodes * self.full_mask_nodes,
                                                                                       self.num_node_intensity_levels)
            else:
                self.full_binned_nodes = utils.bin_signal(nodes * self.full_mask_nodes,
                                                          self.num_node_intensity_levels)
            self.full_binned_nodes[np.logical_not(self.full_mask_nodes)] = 0

            self.alpha = alpha
            # Initialize theta as in paper: 1/L \forall k,l,
            self.theta = np.ones([self.num_node_intensity_levels,
                                  self.num_voxel_intensity_levels]) / self.num_voxel_intensity_levels
        elif self.metric == 'SSD' and not (self.feature_moving or self.feature_fixed):
            if bin_robust:
                self.full_binned_voxels, self.bin_edges_voxels = utils.bin_signal_robust(voxels * self.full_mask_voxels,
                                                                  self.num_voxel_intensity_levels)
            else:
                self.full_binned_voxels = utils.bin_signal(voxels * self.full_mask_voxels,
                                                           self.num_voxel_intensity_levels)
            self.full_binned_voxels[np.logical_not(self.full_mask_voxels)] = 0
            if bin_robust:
                self.full_binned_nodes, self.bin_edges_nodes = utils.bin_signal_robust(nodes * self.full_mask_nodes,
                                                                 self.num_node_intensity_levels)
            else:
                self.full_binned_nodes = utils.bin_signal(nodes * self.full_mask_nodes,
                                                          self.num_node_intensity_levels)
            self.full_binned_nodes[np.logical_not(self.full_mask_nodes)] = 0
            # Initialize sigma_sq by just taking the mean voxel-wise squared difference within
            # the maximum common area of the two images
            tmp = np.minimum(self.orig_voxels_shape, self.orig_nodes_shape)
            self.sigma_sq = np.sum((self.full_binned_nodes[:tmp[0], :tmp[1], :tmp[2]] -
                                    self.full_binned_voxels[:tmp[0], :tmp[1], :tmp[2]]) ** 2) / np.prod(tmp)
            print("Initializing sigma sq with value: " + str(self.sigma_sq))

            self.alpha_0 = alpha_0
            self.beta_0 = beta_0

            if scale_shift:
                print("Estimate scale and shift")
                self.scale_shift = 1
                self.scale = scale
                self.shift = shift
            else:
                self.scale_shift = 0
                self.scale = 1
                self.shift = 0

        if (self.feature_moving or self.feature_fixed
            or precomputed_features_fixed is not None or precomputed_features_moving is not None) and metric == 'SSD':

            if (self.feature_moving and self.feature_fixed) or\
                    (precomputed_features_fixed is not None and precomputed_features_moving is not None):
                if self.fixed_features_reg:
                    if self.number_of_filters_fixed == self.number_of_filters_moving:
                        self.coeff = np.eye(self.number_of_filters_fixed)
                    else:
                        print("Number of filters in fixed and moving image need to be the same!")
                        sys.exit()
                else:
                    self.coeff = np.ones([self.number_of_filters_fixed, self.number_of_filters_moving])

            elif self.feature_moving:
                self.coeff = np.ones([1, self.number_of_filters_moving])
            elif self.feature_fixed:
                self.coeff = np.ones([self.number_of_filters_fixed, 1])

            if self.feature_moving and not precomputed_features_moving is not None:
                print("Computing features for moving image")
                self.full_binned_nodes = utils.compute_features(self.nodes * self.full_mask_nodes,
                                                                self.number_of_filters_moving,
                                                                number_of_dimensions=self.number_of_dimensions,
                                                                include_image_as_feature=self.include_image_as_feature)
            elif precomputed_features_moving is not None:
                if include_image_as_feature:
                    self.full_binned_nodes = np.concatenate((np.expand_dims(nodes, -1), precomputed_features_moving),
                                                            axis=-1)
                else:
                    self.full_binned_nodes = precomputed_features_moving
            #
            if self.feature_fixed and not precomputed_features_fixed is not None:
                print("Computing features for fixed image")
                self.full_binned_voxels = utils.compute_features(self.voxels * self.full_mask_voxels,
                                                                 self.number_of_filters_fixed,
                                                                 number_of_dimensions=self.number_of_dimensions,
                                                                 include_image_as_feature=self.include_image_as_feature)
            elif precomputed_features_fixed is not None:
                if include_image_as_feature:
                    self.full_binned_voxels = np.concatenate((np.expand_dims(voxels, -1), precomputed_features_fixed),
                                                             axis=-1)
                else:
                    self.full_binned_voxels = precomputed_features_fixed

            # Add extra dimension corresponding to one feature if we are using features for only 1 of the two images
            if self.feature_fixed and not self.feature_moving:
                self.full_binned_nodes = np.expand_dims(self.nodes, -1).astype(np.float64)
            if self.feature_moving and not self.feature_fixed:
                self.full_binned_voxels = np.expand_dims(self.voxels, -1).astype(np.float64)

            # Mask images
            self.full_binned_voxels[np.logical_not(self.full_mask_voxels)] = 0
            self.full_binned_nodes[np.logical_not(self.full_mask_nodes)] = 0

            # Initialize sigma_sq by just taking the mean voxel-wise squared difference within
            # the maximum common area of the two images
            max_size = np.minimum(self.orig_voxels_shape, self.orig_nodes_shape)

            if not self.feature_fixed:
                self.sigma_sq = np.sum((self.full_binned_nodes[:max_size[0], :max_size[1], :max_size[2]] -
                                        self.full_binned_voxels[:max_size[0], :max_size[1], :max_size[2]])
                                       ** 2) / np.prod(max_size)
                print("Initializing sigma sq with value: " + str(self.sigma_sq))

                if self.feature_moving:
                    self.sigma_sq = np.atleast_1d(self.sigma_sq)
            else:
                self.sigma_sq = np.zeros(self.number_of_filters_fixed)
                for f in range(self.number_of_filters_fixed):
                    self.sigma_sq[f] = np.sum((self.full_binned_nodes[:max_size[0], :max_size[1], :max_size[2], f] -
                                               self.full_binned_voxels[:max_size[0], :max_size[1], :max_size[2], f])
                                              ** 2) / np.prod(max_size)
                    print("Initializing sigma sq with value: " + str(self.sigma_sq[f]))

            self.alpha_0 = alpha_0
            self.beta_0 = beta_0

        elif (self.feature_moving or self.feature_fixed) and metric == 'MI':

            print("Computing features for moving image")
            self.full_binned_nodes = utils.compute_features(self.nodes, self.number_of_filters_moving,
                                                            number_of_dimensions=self.number_of_dimensions,
                                                            include_image_as_feature=self.include_image_as_feature)
            # Bin signal if MI features
            if bin_robust:
                self.full_binned_nodes, self.bin_edges_nodes = utils.bin_signal_robust(self.full_binned_nodes,
                                                                 self.num_node_intensity_levels)
            else:
                self.full_binned_nodes = utils.bin_signal(self.full_binned_nodes, self.num_node_intensity_levels)
            print("Computing features for fixed image")
            self.full_binned_voxels = utils.compute_features(self.voxels, self.number_of_filters_fixed,
                                                             number_of_dimensions=self.number_of_dimensions,
                                                             include_image_as_feature=self.include_image_as_feature)
            if bin_robust:
                self.full_binned_voxels, self.bin_edges_voxels = utils.bin_signal_robust(self.full_binned_voxels,
                                                                        self.num_voxel_intensity_levels)
            else:
                self.full_binned_nodes = utils.bin_signal(self.full_binned_nodes, self.num_voxel_intensity_levels)

            self.alpha = alpha
            # Initialize theta as in paper: 1/L \forall k,l,
            self.theta = np.ones([self.number_of_filters_fixed, self.num_node_intensity_levels,
                                  self.num_voxel_intensity_levels])
            for m in range(self.number_of_filters_fixed):
                self.theta[m, :, :] /= self.num_voxel_intensity_levels

        if self.number_of_dimensions == 2:
            # In image (voxel) space
            self.full_voxels_pos = np.stack(np.indices(np.shape(voxels), dtype=np.float64),
                                            axis=self.number_of_dimensions + 1)[:, :, :, 0:2]
            self.full_nodes_pos = np.stack(np.indices(np.shape(nodes), dtype=np.float64),
                                           axis=self.number_of_dimensions + 1)[:, :, :, 0:2]
        else:
            # In image (voxel) space
            self.full_voxels_pos = np.stack(np.indices(np.shape(voxels), dtype=np.float64), axis=self.number_of_dimensions)
            self.full_nodes_pos = np.stack(np.indices(np.shape(nodes), dtype=np.float64), axis=self.number_of_dimensions)

        # Initialized voxel pos so that they take into account an affine transformation bringing them into node space
        if initial_transformation is not None:
            self.full_voxels_pos = nib.affines.apply_affine(initial_transformation, self.full_voxels_pos)
        else:
            # If the headers are the same, skip this step
            if not np.allclose(nodes_matrix, voxels_matrix):
                self.full_voxels_pos = nib.affines.apply_affine(np.linalg.inv(nodes_matrix) @ voxels_matrix,
                                                                self.full_voxels_pos)
                if initialize_with_center_of_mass:
                    # Shift by center of mass too (we don't actually trust the header too much!)
                    from scipy.ndimage import center_of_mass
                    center_of_mass_nodes = np.array(center_of_mass(self.full_binned_nodes > 0))
                    center_of_mass_voxels_in_nodes = nib.affines.apply_affine(np.linalg.inv(nodes_matrix) @
                                                                              voxels_matrix,
                                                                              np.array(center_of_mass(
                                                                                  self.full_binned_voxels > 0)))
                    self.full_voxels_pos += center_of_mass_nodes - center_of_mass_voxels_in_nodes

        # Initialize current resolution variables at resolution 1 (i.e., "full")
        self.current_resolution = 1
        self.voxels_pos = self.full_voxels_pos
        self.nodes_pos = self.full_nodes_pos
        self.binned_voxels = self.full_binned_voxels
        self.binned_nodes = self.full_binned_nodes
        self.nodes_shape = self.orig_nodes_shape
        self.voxels_shape = self.orig_voxels_shape
        self.non_zero_voxels = np.count_nonzero(self.binned_voxels)
        self.deformedVoxelPositions = self.full_voxels_pos
        self.dreamLocations = np.zeros_like(self.deformedVoxelPositions)

        self.spline_order = spline_order

        # Scale gamma by number of voxels and resolution
        self.gamma_x = (self.gamma_init * np.prod(self.voxels_voxel_size) * np.prod(self.voxels_shape) *
                        self.nodes_voxel_size[0] ** 2)
        self.gamma_y = (self.gamma_init * np.prod(self.voxels_voxel_size) * np.prod(self.voxels_shape) *
                        self.nodes_voxel_size[1] ** 2)
        self.gamma_z = (self.gamma_init * np.prod(self.voxels_voxel_size) * np.prod(self.voxels_shape) *
                        self.nodes_voxel_size[2] ** 2)

        print("Gamma at full resolution, x direction: " + str(self.gamma_x))
        print("Gamma at full resolution, y direction: " + str(self.gamma_y))
        if self.number_of_dimensions > 2:
            print("Gamma at full resolution, z direction: " + str(self.gamma_z))

        if spline_order > 3:
            print("Spline order " + str(spline_order) + " not implemented, exit")
            sys.exit()

    def run(self, resolutions, max_number_of_iterations):

        for r, resolution in enumerate(resolutions):
            print("Resolution: " + str(resolution))
            # Change resolution for signals and deformation
            self.move_resolution(resolution)

            #
            self.deformedVoxelPositions = np.ascontiguousarray(self.deformedVoxelPositions)
            self.voxels_pos = np.ascontiguousarray(self.voxels_pos)
            self.nodes_pos = np.ascontiguousarray(self.nodes_pos)
            self.binned_voxels = np.ascontiguousarray(self.binned_voxels)
            self.binned_nodes = np.ascontiguousarray(self.binned_nodes)

            # Initialize registrator object
            self.registratorObj = Registration.Registration(self.deformedVoxelPositions.ravel(order="C"),
                                                            self.threads,
                                                            self.voxels_pos.ravel(order="C"),
                                                            self.nodes_pos.ravel(order="C"),
                                                            self.voxels_shape[0], self.voxels_shape[1], self.voxels_shape[2],
                                                            self.nodes_shape[0], self.nodes_shape[1], self.nodes_shape[2],
                                                            self.spline_order,
                                                            self.non_zero_voxels)

            # Set-up transformation model
            if not (self.affine or self.rigid):
                self.D = np.ascontiguousarray(self.D)
                self.registratorObj.setNonLinearTransformation(self.gamma_x, self.gamma_y, self.gamma_z,
                                                               self.current_voxels_voxel_size[0],
                                                               self.current_voxels_voxel_size[1],
                                                               self.current_voxels_voxel_size[2],
                                                               self.current_nodes_voxel_size[0],
                                                               self.current_nodes_voxel_size[1],
                                                               self.current_nodes_voxel_size[2],
                                                               self.D.ravel(order="C"))
            elif self.affine:
                self.A = np.ascontiguousarray(self.A)
                self.t = np.ascontiguousarray(self.t)
                self.registratorObj.setAffineTransformation(self.A.ravel(order="C"),
                                                            self.t.ravel(order="C"))
            else:
                self.R = np.ascontiguousarray(self.R)
                self.t = np.ascontiguousarray(self.t)
                self.registratorObj.setRigidTransformation(self.R.ravel(order="C"),
                                                           self.t.ravel(order="C"))

            # Set-up likelihood model
            if self.metric == 'MI' and not (self.feature_moving or self.feature_fixed):
                self.theta = np.ascontiguousarray(self.theta)
                self.registratorObj.setMILikelihood(self.alpha, self.theta.ravel(order="C"),
                                                    self.num_node_intensity_levels,
                                                    self.num_voxel_intensity_levels, self.threads,
                                                    self.binned_nodes.ravel(order="C"),
                                                    self.binned_voxels.ravel(order="C"))
            elif self.metric == 'SSD' and not (self.feature_moving or self.feature_fixed):
                self.registratorObj.setSSDLikelihood(self.num_voxel_intensity_levels, self.threads, self.sigma_sq,
                                                     self.scale_shift, self.scale,
                                                     self.shift, self.binned_nodes.ravel(order="C"),
                                                     self.binned_voxels.ravel(order="C"),
                                                     self.alpha_0, self.beta_0)
            elif self.metric == 'SSD' and (self.feature_moving or self.feature_fixed):
                self.coeff = np.ascontiguousarray(self.coeff)
                self.sigma_sq = np.ascontiguousarray(self.sigma_sq)
                self.registratorObj.setSSDLikelihoodFilter(self.coeff.ravel(order="C"),
                                                           self.threads,
                                                           self.sigma_sq.ravel(order="C"),
                                                           self.binned_nodes.ravel(order="C"),
                                                           self.binned_voxels.ravel(order="C"),
                                                           self.fixed_features_reg,
                                                           self.alpha_0, self.beta_0,
                                                           self.number_of_filters_fixed,
                                                           self.number_of_filters_moving)


            else:
                self.theta = np.ascontiguousarray(self.theta)
                self.registratorObj.setMILikelihoodFilter(self.alpha, self.theta.ravel(order="C"), self.threads,
                                                          self.binned_nodes.ravel(order="C"),
                                                          self.binned_voxels.ravel(order="C"),
                                                          self.num_node_intensity_levels,
                                                          self.num_voxel_intensity_levels,
                                                          self.number_of_filters_moving)

            self.registratorObj.EM(max_number_of_iterations * self.current_resolution,
                                   self.convergence_th,
                                   update_likelihood_parameters=self.update_likelihood_parameters,
                                   debug=self.debug)
            self.deformedVoxelPositions = self.registratorObj.get_final_locations_as_numpy_array()

            if (self.feature_moving or self.feature_fixed) and self.metric == 'SSD':
                self.coeff = np.array(self.registratorObj.get_likelihood_parameters()).reshape(self.number_of_filters_fixed,
                                                                                               self.number_of_filters_moving)
            elif self.metric == 'SSD':
                self.shift, self.scale = self.registratorObj.get_likelihood_parameters()
            elif self.metric == 'MI' and not self.smooth:  # Do not pass estimated theta to new iteration if smoothing
                self.theta = np.array(self.registratorObj.get_likelihood_parameters()).reshape(self.num_node_intensity_levels,
                                                                                               self.num_voxel_intensity_levels)
                self.theta = np.ascontiguousarray(self.theta)

            if self.top_cor_obj:
                print("Performing topology correction")
                if self.number_of_dimensions == 2:
                    self.deformedVoxelPositions[:, :, 0] = self.top_cor_obj.correct_topology(self.deformedVoxelPositions[:, :, 0].ravel(order="C"),
                                                                                             self.e1, self.e2,
                                                                                             self.max_outer_it,
                                                                                             self.max_inner_it,
                                                                                             self.threads,
                                                                                             self.current_voxels_voxel_size,
                                                                                             self.voxels_shape[0],
                                                                                             self.voxels_shape[1])
                else:
                    self.deformedVoxelPositions = self.top_cor_obj.correct_topology(self.deformedVoxelPositions.ravel(order="C"),
                                                                                    self.e1, self.e2,
                                                                                    self.max_outer_it,
                                                                                    self.max_inner_it, self.threads,
                                                                                    self.current_voxels_voxel_size,
                                                                                    self.voxels_shape[0],
                                                                                    self.voxels_shape[1],
                                                                                    self.voxels_shape[2])

            self.visualize_output(positions=self.deformedVoxelPositions)

    def move_resolution(self, resolution):
        #
        self.current_voxels_voxel_size = self.voxels_voxel_size * resolution
        self.current_nodes_voxel_size = self.nodes_voxel_size * resolution

        self.voxels_shape = tuple(np.int32(np.ceil(x / resolution)) for x in self.orig_voxels_shape)
        self.nodes_shape = np.array([np.int32(np.ceil(x / resolution)) for x in self.orig_nodes_shape])

        self.voxels_pos = self.full_voxels_pos[::resolution, ::resolution, ::resolution, :] / resolution
        self.nodes_pos = self.full_nodes_pos[::resolution, ::resolution, ::resolution, :] / resolution

        # Change resolution of deformation (we need to adjust its length)
        # Linear interpolation, note the self.current_resolution / resolution factor!
        if not self.upsample_in_frequency_domain or self.current_resolution - resolution < 0:
            self.deformedVoxelPositions = resize(self.deformedVoxelPositions, self.voxels_shape,
                                                 order=1) * (self.current_resolution / resolution)
        elif self.upsample_in_frequency_domain and resolution != self.current_resolution:
            print("Up sampling deformation field in the frequency domain")
            delta = np.ascontiguousarray(self.deformedVoxelPositions -
                                         self.full_voxels_pos[::self.current_resolution,
                                                              ::self.current_resolution,
                                                              ::self.current_resolution, :] / self.current_resolution)
            tmp = Registration.NonLinearTransformation.up_sample(delta.ravel(order="C"),
                                                                 self.deformedVoxelPositions.shape[0],
                                                                 self.deformedVoxelPositions.shape[1],
                                                                 self.deformedVoxelPositions.shape[2],
                                                                 self.voxels_shape[0],
                                                                 self.voxels_shape[1],
                                                                 self.voxels_shape[2],
                                                                 self.threads)
            self.deformedVoxelPositions = (tmp * self.current_resolution / resolution) + self.voxels_pos


        self.binned_voxels = self.full_binned_voxels[::resolution, ::resolution, ::resolution]
        self.binned_nodes = self.full_binned_nodes[::resolution, ::resolution, ::resolution]

        if self.smooth and resolution != 1:
            print("Smoothing images")
            tmp = np.zeros(self.binned_nodes.shape)
            tmp = np.ascontiguousarray(tmp)
            self.nodes = np.ascontiguousarray(self.nodes)
            self.smooth_spline_object.smooth_and_downsample_image(self.nodes.ravel(order="C"),
                                                                  tmp.ravel(order="C"),
                                                                  self.orig_nodes_shape[0],
                                                                  self.orig_nodes_shape[1],
                                                                  self.orig_nodes_shape[2],
                                                                  resolution)
            self.binned_nodes, _ = utils.bin_signal_robust(tmp *
                                                           self.full_mask_nodes[::resolution, ::resolution, ::resolution],
                                                           self.num_node_intensity_levels)
            tmp = np.zeros(self.binned_voxels.shape)
            tmp = np.ascontiguousarray(tmp)
            self.voxels = np.ascontiguousarray(self.voxels)
            self.smooth_spline_object.smooth_and_downsample_image(self.voxels.ravel(order="C"),
                                                                  tmp.ravel(order="C"),
                                                                  self.orig_voxels_shape[0],
                                                                  self.orig_voxels_shape[1],
                                                                  self.orig_voxels_shape[2],
                                                                  resolution)
            self.binned_voxels, _ = utils.bin_signal_robust(tmp *
                                                            self.full_mask_voxels[::resolution, ::resolution, ::resolution],
                                                            self.num_voxel_intensity_levels)
            print("Done!")

        # Compute fourier smoothing
        self.D = utils.initialize_fourier_domain_smoothing(self.voxels_shape, True,
                                                           self.finite_difference_order,
                                                           self.number_of_dimensions,
                                                           voxel_size=self.current_voxels_voxel_size)

        #
        self.non_zero_voxels = np.count_nonzero(self.binned_voxels)

        # Scale gamma appropriately by taking into consideration node voxel size
        if not self.increase_gamma_at_DS_resolution:
            self.gamma_x = (self.gamma_init * np.prod(self.current_voxels_voxel_size) * np.prod(self.voxels_shape) *
                            self.current_nodes_voxel_size[0] ** 2)
            self.gamma_y = (self.gamma_init * np.prod(self.current_voxels_voxel_size) * np.prod(self.voxels_shape) *
                            self.current_nodes_voxel_size[1] ** 2)
            self.gamma_z = (self.gamma_init * np.prod(self.current_voxels_voxel_size) * np.prod(self.voxels_shape) *
                            self.current_nodes_voxel_size[2] ** 2)
        else:
            # Here we do increase gamma at each lower level. For DS 2 and 3D, this means a 64 increase
            self.gamma_x = (self.gamma_init * np.prod(self.current_voxels_voxel_size) * np.prod(self.orig_voxels_shape) *
                            self.current_nodes_voxel_size[0] ** 2)
            self.gamma_y = (self.gamma_init * np.prod(self.current_voxels_voxel_size) * np.prod(self.orig_voxels_shape) *
                            self.current_nodes_voxel_size[1] ** 2)
            self.gamma_z = (self.gamma_init * np.prod(self.current_voxels_voxel_size) * np.prod(self.orig_voxels_shape) *
                            self.current_nodes_voxel_size[2] ** 2)

        # Update current resolution
        self.current_resolution = resolution

        # Print some info
        print("Number of voxels: " + str(np.prod(self.voxels_shape)))
        print("gamma_x: " + str(self.gamma_x))
        print("gamma_y: " + str(self.gamma_y))
        if self.number_of_dimensions > 2:
            print("gamma_z: " + str(self.gamma_z))
        print("Fixed image voxel size: " + str(self.current_voxels_voxel_size[0:self.number_of_dimensions]))
        print("Moving image voxel size: " + str(self.current_nodes_voxel_size[0:self.number_of_dimensions]))

    def sample_param(self, burnin, samples, sample_gamma=0, sample_gamma_every=1,
                     sample_posteriors=1, sample_likelihood_parameters=1,
                     sample_transformation_parameters=1):

        #
        self.deformedVoxelPositions = np.ascontiguousarray(self.deformedVoxelPositions)
        self.voxels_pos = np.ascontiguousarray(self.voxels_pos)
        self.nodes_pos = np.ascontiguousarray(self.nodes_pos)
        self.binned_voxels = np.ascontiguousarray(self.binned_voxels)
        self.binned_nodes = np.ascontiguousarray(self.binned_nodes)
        if sample_posteriors == 0:
            # Set precomputed dreamLocations in new registration object
            self.dreamLocations = self.registratorObj.get_dream_locations()
            self.dreamLocations = np.ascontiguousarray(self.dreamLocations)
            
        self.registratorObj = Registration.Registration(self.deformedVoxelPositions.ravel(order="C"),
                                                        self.threads,
                                                        self.voxels_pos.ravel(order="C"),
                                                        self.nodes_pos.ravel(order="C"),
                                                        self.voxels_pos.shape[0], self.voxels_pos.shape[1], self.voxels_pos.shape[2],
                                                        self.nodes_pos.shape[0], self.nodes_pos.shape[1], self.nodes_pos.shape[2],
                                                        self.spline_order,
                                                        self.non_zero_voxels)


        if sample_posteriors == 0:
            self.registratorObj.set_dream_locations(self.dreamLocations.ravel(order="C"))

        # TODO: do this properly, accounting for different likelihood functions
        if self.metric == 'MI':
            if sample_likelihood_parameters == 1 and sample_posteriors == 0:
                # Set precomputed theta_tmp (joint histogram filling) in new registration object
                #TODO: _, _, theta_tmp = self.registratorObj.get_likelihood_parameters()
                theta = np.ascontiguousarray(self.theta.copy())
                theta_as_theta_tmp = 1
                #theta = theta_tmp
            else:
                theta = np.ascontiguousarray(self.theta.copy())
                theta_as_theta_tmp = 0

        if self.affine:
            self.registratorObj.setAffineTransfomation(self.A.ravel(order="C"), self.t.ravel(order="C"))
        elif self.rigid:
            self.registratorObj.setRigidTransformation(self.R.ravel(order="C"), self.t.ravel(order="C"))
        else:
            self.D = np.ascontiguousarray(self.D)
            self.registratorObj.setNonLinearTransformation(self.gamma_x, self.gamma_y, self.gamma_z,
                                                           self.current_voxels_voxel_size[0],
                                                           self.current_voxels_voxel_size[1],
                                                           self.current_voxels_voxel_size[2],
                                                           self.current_nodes_voxel_size[0],
                                                           self.current_nodes_voxel_size[1],
                                                           self.current_nodes_voxel_size[2],
                                                           self.D.ravel(order="C"))

        if self.metric == 'MI' and not (self.feature_fixed or self.feature_moving):
            theta = np.ascontiguousarray(theta)
            self.registratorObj.setMILikelihood(self.alpha, theta.ravel(order="C"),
                                           self.num_node_intensity_levels,
                                           self.num_voxel_intensity_levels,
                                           self.threads,
                                           self.binned_nodes.ravel(order="C"),
                                           self.binned_voxels.ravel(order="C"))
        elif self.metric == 'SSD' and not (self.feature_fixed or self.feature_moving):
            self.registratorObj.setSSDLikelihood(self.num_voxel_intensity_levels, self.threads, self.sigma_sq,
                                            self.scale_shift, self.scale,
                                            self.shift, self.binned_nodes, self.binned_voxels,
                                            self.alpha_0, self.beta_0)
        elif self.metric == 'SSD' and (self.feature_moving or self.feature_fixed):
            self.coeff = np.ascontiguousarray(self.coeff)
            self.sigma_sq = np.ascontiguousarray(self.sigma_sq)
            self.registratorObj.setSSDLikelihoodFilter(self.coeff.ravel(order="C"),
                                                  self.threads,
                                                  self.sigma_sq.ravel(order="C"),
                                                  self.binned_nodes.ravel(order="C"),
                                                  self.binned_voxels.ravel(order="C"),
                                                  self.fixed_features_reg,
                                                  self.alpha_0, self.beta_0,
                                                  self.number_of_filters_fixed,
                                                  self.number_of_filters_moving)
        else:
            self.theta = np.ascontiguousarray(self.theta)
            self.registratorObj.setMILikelihoodFilter(self.alpha, self.theta.ravel(order="C"), self.threads,
                                                 self.binned_nodes.ravel(order="C"),
                                                 self.binned_voxels.ravel(order="C"),
                                                 self.num_node_intensity_levels,
                                                 self.num_voxel_intensity_levels,
                                                 self.number_of_filters_moving)

        meanDefVoxelPos, covField = self.registratorObj.sampler(self.seed, burnin, samples,
                                                               sample_gamma,
                                                               sample_gamma_every,
                                                               sample_posteriors,
                                                               sample_likelihood_parameters,
                                                               sample_transformation_parameters)

        self.visualize_output(positions=meanDefVoxelPos)

        return meanDefVoxelPos, covField

    def visualize_output(self, positions):

        if self.visualizer:
            if not self.feature_moving and not self.feature_fixed:
                if self.metric == 'SSD' and self.scale_shift:
                    # Display scaled and shifted signal
                    tmp_binned = self.shift + self.binned_nodes * self.scale
                else:
                    tmp_binned = self.binned_nodes
                utils.plot_signals_and_warp(self.binned_voxels, tmp_binned, positions,
                                            self.voxels_shape, self.voxels_pos, self.number_of_dimensions)
            else:
                if self.feature_moving:
                    for m in range(self.number_of_filters_fixed):
                        if self.metric == 'SSD':
                            tmp_binned = self.binned_nodes @ self.coeff[m, :]
                        else:
                            tmp_binned = self.binned_nodes[..., m]
                        utils.plot_signals_and_warp(self.binned_voxels[:, :, :, m], tmp_binned, positions,
                                                    self.voxels_shape, self.voxels_pos, self.number_of_dimensions)

            utils.plot_fourier_smoothing(self.D, self.number_of_dimensions)

            if self.metric == 'MI':
                if self.feature_moving:
                    for m in range(self.number_of_filters_fixed):
                        utils.plot_theta(self.theta[m])
                else:
                    utils.plot_theta(self.theta)
            if self.metric == 'SSD' and (self.feature_moving or self.feature_fixed):
                utils.plot_coeff(self.coeff)
