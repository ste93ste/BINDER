//
// Created by stce on 01/01/25.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Registration.h"
#include "NonLinearTransformation.h"
#include "Spline.h"
#include "topology_correction_2D.h"
#include "topology_correction_3D.h"
#include "Smoothing_Spline.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ module for Registration with pybind11";
    m.attr("backend") = "cpu";
    // Define the Registration class binding
	py::class_<Registration, std::unique_ptr<Registration>>(m, "Registration")
        .def(py::init([](py::array_t<const double, py::array::c_style | py::array::forcecast> final_locations,
                         const int threads,
                         py::array_t<const double, py::array::c_style | py::array::forcecast> voxel_pos,
                         py::array_t<const double, py::array::c_style | py::array::forcecast> node_pos,
                         const int Nx_v, const int Ny_v, const int Nz_v, const int Nx_n, const int Ny_n, const int Nz_n,
                         const int spline_order, const double non_zero_voxels) {
            // Get the raw pointer to final_locations array
			const double* ptr_final_locations = static_cast<const double*>(final_locations.data());

            // Get the raw pointer to voxel_pos array
            const double* ptr_voxel = static_cast<const double*>(voxel_pos.data());

            // Get the raw pointer to node_pos array
            const double* ptr_node = static_cast<const double*>(node_pos.data());

            // Create and return a new Registration object
            auto reg =  std::make_unique<Registration>(ptr_final_locations,
                                            		   threads,
                                    			  	   ptr_voxel,
                                    			       ptr_node,
                                    			       Nx_v, Ny_v, Nz_v, Nx_n, Ny_n, Nz_n,
                                    			       spline_order, non_zero_voxels);
			return reg;
        }),
        py::arg("final_locations"),
        py::arg("threads"),
        py::arg("voxel_pos"),
        py::arg("node_pos"),
        py::arg("Nx_v"),
        py::arg("Ny_v"),
        py::arg("Nz_v"),
        py::arg("Nx_n"),
        py::arg("Ny_n"),
        py::arg("Nz_n"),
        py::arg("spline_order"),
        py::arg("non_zero_voxels"),
        py::keep_alive<0, 1>())
        .def("setMILikelihood", [](Registration& self,
                                   const double alpha,
                                   py::array_t<const double, py::array::c_style | py::array::forcecast> theta,
                                   const int K, const int L,
                                   const int threads,
                                   py::array_t<const int, py::array::c_style | py::array::forcecast> binned_nodes,
                                   py::array_t<const int, py::array::c_style | py::array::forcecast> binned_voxels) {

            // Get raw pointer to theta
            const double* ptr_theta = static_cast<const double*>(theta.data());

            // Get raw pointer to binned_nodes
            const int* ptr_binned_nodes = static_cast<const int*>(binned_nodes.data());

            // Get raw pointer to binned_voxels
            const int* ptr_binned_voxels = static_cast<const int*>(binned_voxels.data());

            // Call the `setMILikelihood` method
            self.setMILikelihood(alpha, ptr_theta, K, L, threads, ptr_binned_nodes,
                                 ptr_binned_voxels);
        },
        py::arg("alpha"),
        py::arg("theta"),
        py::arg("K"),
        py::arg("L"),
        py::arg("threads"),
        py::arg("binned_nodes"),
        py::arg("binned_voxels"),
        py::keep_alive<0, 1>()
		)
        .def("setMILikelihoodFilter", [](Registration& self,
                                    double alpha,
	                                py::array_t<const double, py::array::c_style | py::array::forcecast> theta,
                                    int threads,
                                    py::array_t<const int, py::array::c_style | py::array::forcecast> binned_nodes,
                                    py::array_t<const int, py::array::c_style | py::array::forcecast> binned_voxels,
                                    int K, int L, int M) {

            // Get raw pointer to theta
            const double* ptr_binned_theta = static_cast<const double*>(theta.data());

            // Get raw pointer to binned_nodes
            const int* ptr_binned_nodes = static_cast<const int*>(binned_nodes.data());

            // Get raw pointer to binned_voxels
            const int* ptr_binned_voxels = static_cast<const int*>(binned_voxels.data());

            // Call the `setMILikelihoodFilter` method
            self.setMILikelihoodFilter(alpha, ptr_binned_theta, threads, ptr_binned_nodes, ptr_binned_voxels, K, L, M);
        },
        py::arg("alpha"),
        py::arg("theta"),
        py::arg("threads"),
        py::arg("binned_nodes"),
        py::arg("binned_voxels"),
        py::arg("K"),
        py::arg("L"),
        py::arg("M"),
        py::keep_alive<0, 1>()
		)
        .def("setSSDLikelihood", [](Registration& self,
                                   int bins,
                                   int threads,
                                   double sigma_sq,
                                   int scale_shift,
                                   double scale,
                                   double shift,
                                   py::array_t<const int, py::array::c_style | py::array::forcecast> binned_nodes,
                                   py::array_t<const int, py::array::c_style | py::array::forcecast> binned_voxels,
                                   double alpha_0,
                                   double beta_0) {

            // Get raw pointer to binned_nodes
            const int* ptr_binned_nodes = static_cast<const int*>(binned_nodes.data());

            // Get raw pointer to binned_voxels
            const int* ptr_binned_voxels = static_cast<const int*>(binned_voxels.data());

            // Call the `setSSDLikelihood` method
            self.setSSDLikelihood(bins, threads, sigma_sq, scale_shift, scale, shift,
                                  ptr_binned_nodes, ptr_binned_voxels, alpha_0, beta_0);
        },
        py::arg("bins"),
        py::arg("threads"),
        py::arg("sigma_sq"),
        py::arg("scale_shift"),
        py::arg("scale"),
        py::arg("shift"),
        py::arg("binned_nodes"),
        py::arg("binned_voxels"),
        py::arg("alpha_0"),
        py::arg("beta_0"),
        py::keep_alive<0, 1>()
		)
        .def("setSSDLikelihoodFilter", [](Registration& self,
                                          py::array_t<const double, py::array::c_style | py::array::forcecast> coeff,
                                          int threads,
                                          py::array_t<const double, py::array::c_style | py::array::forcecast> sigma,
                                          py::array_t<const double, py::array::c_style | py::array::forcecast> features_nodes,
                                          py::array_t<const double, py::array::c_style | py::array::forcecast> features_voxels,
                                          int reg_across_features,
                                          double alpha_0,
                                          double beta_0,
                                          int M,
                                          int F) {

            // Get raw pointer to coeff
            const double* ptr_coeff = static_cast<const double*>(coeff.data());
            // Get raw pointer to sigma
            const double* ptr_sigma = static_cast<const double*>(sigma.data());
            // Get raw pointer to features_nodes
            const double* ptr_features_nodes = static_cast<const double*>(features_nodes.data());
            // Get raw pointer to features_voxels
            const double* ptr_features_voxels = static_cast<const double*>(features_voxels.data());

            // Call the `setSSDLikelihoodFilter` method
            self.setSSDLikelihoodFilter(ptr_coeff, threads, ptr_sigma, ptr_features_nodes, ptr_features_voxels, reg_across_features,
                                  		alpha_0, beta_0, M, F);
        },
        py::arg("coeff"),
        py::arg("threads"),
        py::arg("sigma"),
        py::arg("features_nodes"),
        py::arg("features_voxels"),
        py::arg("reg_across_features"),
        py::arg("alpha_0"),
        py::arg("beta_0"),
        py::arg("M"),
        py::arg("F"),
        py::keep_alive<0, 1>()
		)
        .def("setNonLinearTransformation", [](Registration& self,
                                              double gamma_x,
                                              double gamma_y,
                                              double gamma_z,
                                              const double voxels_size_x,
                                              const double voxels_size_y,
                                              const double voxels_size_z,
                                              const double nodes_size_x,
                                              const double nodes_size_y,
                                              const double nodes_size_z,
                                              py::array_t<const double, py::array::c_style | py::array::forcecast> D) {
            // Get raw pointer to D
            const double* ptr_D = static_cast<const double*>(D.data());

            // Call the `setNonLinearTransformation` method
            self.setNonLinearTransformation(gamma_x, gamma_y, gamma_z,
            							voxels_size_x, voxels_size_y, voxels_size_z,
            							nodes_size_x, nodes_size_y, nodes_size_z,
            							ptr_D);
        },
        py::arg("gamma_x"),
        py::arg("gamma_y"),
        py::arg("gamma_z"),
        py::arg("voxels_size_x"),
		py::arg("voxels_size_y"),
		py::arg("voxels_size_z"),
		py::arg("nodes_size_x"),
		py::arg("nodes_size_y"),
		py::arg("nodes_size_z"),
        py::arg("D"),
        py::keep_alive<0, 1>()
		)
        .def("setAffineTransformation", [](Registration& self,
                                           py::array_t<const double, py::array::c_style | py::array::forcecast> A,
                                           py::array_t<const double, py::array::c_style | py::array::forcecast> t) {
            // Get raw pointer to A
            const double* ptr_A = static_cast<const double*>(A.data());
            // Get raw pointer to t
            const double* ptr_t = static_cast<const double*>(t.data());

            // Call the `setAffineTransformation` method
            self.setAffineTransformation(ptr_A, ptr_t);
        },
        py::arg("A"),
        py::arg("t"))
        .def("setRigidTransformation", [](Registration& self,
                                           py::array_t<const double, py::array::c_style | py::array::forcecast> R,
                                           py::array_t<const double, py::array::c_style | py::array::forcecast> t) {
            // Get raw pointer to R
            const double* ptr_R = static_cast<const double*>(R.data());
            // Get raw pointer to t
            const double* ptr_t = static_cast<const double*>(t.data());

            // Call the `setRigidTransformation` method
            self.setRigidTransformation(ptr_R, ptr_t);
        },
        py::arg("R"),
        py::arg("t"))
        .def("EM", &Registration::EM,
             py::arg("max_EM_iterations"),
             py::arg("convergence_th"),
             py::arg("update_likelihood_parameters"),
             py::arg("debug"),
             py::keep_alive<0, 1>()
			 )
        .def("sampler", [](Registration& self, int seed, int N_b, int N_s,
             								   int sample_gamma, int sample_gamma_every,
             								   int sample_posteriors,
             								   int sample_likelihood_parameters,
             								   int sample_transformation_parameters){

        auto [meanDefField, covField] = self.sampler(seed, N_b, N_s, sample_gamma,
                                                     sample_gamma_every, sample_posteriors,
                                                     sample_likelihood_parameters,
                                                     sample_transformation_parameters);

        std::vector<int> shape = self.get_shape();
        // mean deformation field: [Nx, Ny, Nz, N]
        py::array_t<double> result_array_1(shape, meanDefField.data());

        // covariance field: [Nx, Ny, Nz, N, N]
        std::vector<int> shape_cov = {shape[0], shape[1], shape[2], shape[3], shape[3]};
        py::array_t<double> result_array_2(shape_cov, covField.data());

        return py::make_tuple(result_array_1, result_array_2);
        },
        py::arg("seed"),
        py::arg("N_b"),
        py::arg("N_s"),
        py::arg("sample_gamma"),
        py::arg("sample_gamma_every"),
        py::arg("sample_posteriors"),
        py::arg("sample_likelihood_parameters"),
        py::arg("sample_transformation_parameters"))
        .def("set_dream_locations", [](Registration& self,
                                       py::array_t<const double, py::array::c_style | py::array::forcecast> dream_locations){
          	// Get raw pointer to dream_locations
            const double* ptr_dream_locations = static_cast<const double*>(dream_locations.data());
            self.set_dream_locations(ptr_dream_locations);
        })
        .def("get_final_locations", &Registration::get_final_locations, py::return_value_policy::copy)
		.def("get_dream_locations", &Registration::get_dream_locations, py::return_value_policy::copy)
		.def("get_likelihood_parameters", &Registration::get_likelihood_parameters, py::return_value_policy::copy)
        .def("get_shape", &Registration::get_shape, py::return_value_policy::copy)
	.def("get_final_locations_as_numpy_array", [](Registration& self) {
    	auto locations = self.get_final_locations();  // Get copy of data as vector
    	std::vector<int> shape = self.get_shape();

	    // Create numpy array from the vector data
    	return py::array_t<double>(
        	shape,                     // shape
        	{                         // strides
            	sizeof(double) * shape[1] * shape[2] * shape[3],
            	sizeof(double) * shape[2] * shape[3],
            	sizeof(double) * shape[3],
            	sizeof(double)
	        },
    	    locations.data()          // data pointer
    	);
	});

    // Define the NonLinearTransformation class binding
    py::class_<NonLinearTransformation, std::unique_ptr<NonLinearTransformation>>(m, "NonLinearTransformation")
        .def(py::init([](double gamma_x, double gamma_y, double gamma_z,
        	             const double voxels_size_x, const double voxels_size_y, const double voxels_size_z,
        	             const double nodes_size_x, const double nodes_size_y, const double nodes_size_z,
        	             py::array_t<const double, py::array::c_style | py::array::forcecast> D,
                         int Nx_v, int Ny_v, int Nz_v,
                         int threads,
                         double sigma_spline,
                         double spline_offset) {

            // Get raw pointer to D
            const double* ptr_D = static_cast<const double*>(D.data());
            // Create and return a new Registration object
            auto obj =  std::make_unique<NonLinearTransformation>(gamma_x, gamma_y, gamma_z,
                                                                  voxels_size_x, voxels_size_y, voxels_size_z,
                                                                  nodes_size_x, nodes_size_y, nodes_size_z,
                                               					  ptr_D,
                                               					  Nx_v, Ny_v, Nz_v,
                                               					  threads,
                                               					  sigma_spline,
                                               					  spline_offset);
			return obj;
        }),
        py::arg("gamma_x"),
        py::arg("gamma_y"),
        py::arg("gamma_z"),
        py::arg("voxels_size_x"),
        py::arg("voxels_size_y"),
        py::arg("voxels_size_z"),
        py::arg("nodes_size_x"),
        py::arg("nodes_size_y"),
        py::arg("nodes_size_z"),
        py::arg("D"),
        py::arg("Nx_v"),
        py::arg("Ny_v"),
        py::arg("Nz_v"),
        py::arg("threads"),
        py::arg("sigma_spline"),
        py::arg("spline_offset"),
        py::keep_alive<0, 1>())
        .def("get_shape", &NonLinearTransformation::get_shape)
		.def("smooth_deformation", [](NonLinearTransformation& self,
        		                      py::array_t<const double, py::array::c_style | py::array::forcecast> deformation) {
		    const double* ptr_deformation = static_cast<const double*>(deformation.data());

		    auto [log_prior_deformation, new_deformation] = self.smooth_deformation(ptr_deformation);
    		std::vector<int> shape = self.get_shape();

		    auto result_array = py::array(
        		py::buffer_info(
            	new_deformation,
            	sizeof(double),
            	py::format_descriptor<double>::format(),
            	shape.size(),
            	shape,
            	{
                	sizeof(double) * shape[1] * shape[2] * shape[3],
                	sizeof(double) * shape[2] * shape[3],
                	sizeof(double) * shape[3],
                	sizeof(double)
            	}
        		));
    		return py::make_tuple(log_prior_deformation, result_array);
			},
			py::arg("deformation"))
		.def("sampleDeformationField",
    		[](NonLinearTransformation& self,
                    py::array_t<double, py::array::c_style | py::array::forcecast> deformationField,
                    int number_of_samples) {
            const double* ptr_deformation = static_cast<const double*>(deformationField.data());

	        double* raw_data = self.sampleDeformationField(ptr_deformation, number_of_samples);

    	    std::vector<int> shape_tmp = self.get_shape();
        	std::vector<int> shape = {number_of_samples, shape_tmp[0], shape_tmp[1], shape_tmp[2], shape_tmp[3]};

	        auto result_array = py::array(py::buffer_info(
    	        raw_data,
        	    sizeof(double),
            	py::format_descriptor<double>::format(),
	            shape.size(),
    	        shape,
        	    {
            	    sizeof(double) * shape[1] * shape[2] * shape[3] * shape[4],
                	sizeof(double) * shape[2] * shape[3] * shape[4],
    	            sizeof(double) * shape[3] * shape[4],
        	        sizeof(double) * shape[4],
            	    sizeof(double)
            	}
        	));

	        return result_array;
    	},
    	py::arg("deformationField"),
    	py::arg("number_of_samples"))
     	.def_static("up_sample", [](py::array_t<const double, py::array::c_style | py::array::forcecast> deformation_field,
                                 int N_x, int N_y, int N_z, int N_x_o, int N_y_o, int N_z_o, int threads) {
            // Ensure the input is contiguous
            const double* ptr_deformation = static_cast<const double*>(deformation_field.data());
     		int N = 3;
            if (N_z_o == 1){
                N = 2;
			}

            // Call the C++ static method
            std::vector<double> result = NonLinearTransformation::up_sample(ptr_deformation, N_x, N_y, N_z,
                                                                                             N_x_o, N_y_o, N_z_o,
                                                                                             threads);

            // Return the result as a NumPy array
            return py::array_t<double>(
                {N_x_o, N_y_o, N_z_o, N},  // Shape
                {sizeof(double) * (N_y_o) * N_z_o * N,  // Stride for X
                 sizeof(double) * N_z_o * N,            // Stride for Y
                 sizeof(double) * N,                    // Stride for Z
                 sizeof(double)},
                result.data()  // Pointer to the data
            );
        }, py::arg("deformation_field"), py::arg("N_x"), py::arg("N_y"), py::arg("N_z"),
				                         py::arg("N_x_o"), py::arg("N_y_o"), py::arg("N_z_o"), py::arg("threads"),
           "Up-sample a 3D deformation field");

    // Define the Spline class binding
    py::class_<Spline>(m, "Spline")
    .def(py::init([](int order, bool is2D) {
        return new Spline(order, is2D);
    }),
    py::arg("order"),
    py::arg("is2D"))
    .def("spline", &Spline::spline,
         py::arg("d"))
    .def("log_spline", &Spline::log_spline,
         py::arg("d"));

#ifdef BINDER_WITH_TOPOLOGY_CORRECTION
    // Define the TopologyCorrection2D class binding
    py::class_<topology_correction_2D>(m, "topology_correction_2D")
    .def(py::init([]() {
        return new topology_correction_2D();
    }))
    .def("correct_topology", [](topology_correction_2D& self,
                                py::array_t<const double, py::array::c_style | py::array::forcecast> h_init,
                                double e1, double e2, int max_outer_it, int max_it,
                                int threads,
                                py::array_t<const double, py::array::c_style | py::array::forcecast> voxel_sizes,
                                int Nx_v, int Ny_v) {
        // Get raw pointer to h_init
        const double* ptr_h_init = static_cast<const double*>(h_init.data());
        // Get raw pointer to voxel_sizes
        const double* ptr_voxel_sizes = static_cast<const double*>(voxel_sizes.data());
        double* raw_data = self.correct_topology(ptr_h_init, e1, e2, max_outer_it, max_it, threads, ptr_voxel_sizes,
                                                 Nx_v, Ny_v);
        std::vector<int> shape = {Nx_v, Ny_v, 2};

        return py::array(py::buffer_info(
            raw_data,
            sizeof(double),
            py::format_descriptor<double>::format(),
            shape.size(),
            shape,
            {sizeof(double) * shape[1] * shape[2],
             sizeof(double) * shape[2],
             sizeof(double)}
        ));
    });

    // Define the TopologyCorrection3D class binding
    py::class_<topology_correction_3D>(m, "topology_correction_3D")
    .def(py::init([]() {
        return new topology_correction_3D();
    }))
    .def("correct_topology", [](topology_correction_3D& self,
                                py::array_t<double const, py::array::c_style | py::array::forcecast> h_init,
                                double e1, double e2, int max_outer_it, int max_it,
                                int threads, py::array_t<double const, py::array::c_style | py::array::forcecast> voxel_sizes,
                                int Nx_v, int Ny_v, int Nz_v) {
        // Get raw pointer to h_init
        const double* ptr_h_init = static_cast<const double*>(h_init.data());
        // Get raw pointer to voxel_sizes
        const double* ptr_voxel_sizes = static_cast<const double*>(voxel_sizes.data());
        double* raw_data = self.correct_topology(ptr_h_init, e1, e2, max_outer_it, max_it, threads, ptr_voxel_sizes,
                                                 Nx_v, Ny_v, Nz_v);
        std::vector<int> shape = {Nx_v, Ny_v, Nz_v, 3};

        return py::array(py::buffer_info(
            raw_data,
            sizeof(double),
            py::format_descriptor<double>::format(),
            shape.size(),
            shape,
            {sizeof(double) * shape[1] * shape[2] * shape[3],
             sizeof(double) * shape[2] * shape[3],
             sizeof(double) * shape[3],
             sizeof(double)}
        ));
    });
#endif // BINDER_WITH_TOPOLOGY_CORRECTION

    // Define the Smoothing Spline class binding
    py::class_<Smoothing_Spline>(m, "Smoothing_Spline")
    .def(py::init([](int spline_order, int threads) {
        return new Smoothing_Spline(spline_order, threads);
    }))
    .def("smooth_and_downsample_image", [](Smoothing_Spline& self,
                                py::array_t<double const, py::array::c_style | py::array::forcecast> im,
                                py::array_t<double, py::array::c_style | py::array::forcecast> im_down,
                                int z, int y, int x, int down_factor) {

        // Get raw pointer to im
        const double* ptr_im = static_cast<const double*>(im.data());
        // Ensure the output image buffer is writable
        auto buf_im_down = im_down.request();
        if (buf_im_down.ptr == nullptr) {
            throw std::runtime_error("Output image buffer is null!");
        }
        if (buf_im_down.readonly) {
            throw std::runtime_error("Output image buffer is not writable!");
        }
        double* ptr_im_down = static_cast<double*>(buf_im_down.ptr);
        self.smooth_and_downsample_image(ptr_im, ptr_im_down, z, y, x, down_factor);
    });

	m.def("compute_jacobian_determinant_2D",
		  [](py::array_t<double const, py::array::c_style | py::array::forcecast> field,
			 int Nx, int Ny,
			 double voxel_size_x, double voxel_size_y) {

            // Get raw pointer to the deformation field
			const double* ptr_field = static_cast<const double*>(field.data());

			// Call the C++ function
			std::vector<double> raw_1D_data = compute_jacobian_determinant_2D(ptr_field, Nx, Ny, voxel_size_x, voxel_size_y);

			std::vector<int> shape = {Nx, Ny};

			return py::array(py::buffer_info(
							 raw_1D_data.data(),
							 sizeof(double),
  							 py::format_descriptor<double>::format(),
							 shape.size(),
							 shape,
  							 {sizeof(double) * Ny,
							  sizeof(double)}
 			));
          },
      	  py::arg("field"),
	      py::arg("Nx"),
	      py::arg("Ny"),
	      py::arg("voxel_size_x"),
	      py::arg("voxel_size_y"));

	m.def("compute_jacobian_determinant_3D",
		  [](py::array_t<double const, py::array::c_style | py::array::forcecast> field,
			 int Nx, int Ny, int Nz,
			 double voxel_size_x, double voxel_size_y, double voxel_size_z) {
			// Get raw pointer to the deformation field
			const double* ptr_field = static_cast<const double*>(field.data());
			// Call the C++ function
			std::vector<double> raw_1D_data = compute_jacobian_determinant_3D(ptr_field, Nx, Ny, Nz,
                                                                                          voxel_size_x,
                                                                                          voxel_size_y,
                                                                                          voxel_size_z);
			std::vector<int> shape = {Nx, Ny, Nz};
			return py::array(py::buffer_info(
							 raw_1D_data.data(),
							 sizeof(double),
							   py::format_descriptor<double>::format(),
							 shape.size(),
							 shape,
							  {sizeof(double) * Ny * Nz,
							  sizeof(double) * Nz,
							  sizeof(double)}
			 ));
		  },
		  py::arg("field"),
		  py::arg("Nx"),
		  py::arg("Ny"),
          py::arg("Nz"),
		  py::arg("voxel_size_x"),
		  py::arg("voxel_size_y"),
          py::arg("voxel_size_z"));
}
