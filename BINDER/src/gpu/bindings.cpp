//
// Created by stce on 2/14/25.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Registration.cuh"
#include "NonLinearTransformation.cuh"
#include "topology_correction_3D.cuh"
#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "CUDA-accelerated Registration module with pybind11 bindings";
	m.attr("backend") = "cuda";
    py::class_<Registration>(m, "Registration")
        .def(py::init([](py::array_t<const float, py::array::c_style | py::array::forcecast> final_locations,
                        int threads,
                        py::array_t<const float, py::array::c_style | py::array::forcecast> voxel_pos,
                        py::array_t<const float, py::array::c_style | py::array::forcecast> node_pos,
                        int Nx_v, int Ny_v, int Nz_v,
                        int Nx_n, int Ny_n, int Nz_n,
                        int spline_order,
                        float non_zero_voxels) {
            // Get buffer info
            const float* ptr_final_locations = static_cast<const float*>(final_locations.data());
            const float* ptr_final_voxel_pos = static_cast<const float*>(voxel_pos.data());
            const float* ptr_final_node_pos = static_cast<const float*>(node_pos.data());

            return std::make_unique<Registration>(ptr_final_locations, threads, ptr_final_voxel_pos, ptr_final_node_pos,
                                                  Nx_v, Ny_v, Nz_v,
                                                  Nx_n, Ny_n, Nz_n,
                                                  spline_order, non_zero_voxels);
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
        py::arg("non_zero_voxels"))

        .def("setMILikelihood", [](Registration& self,
                                  float alpha,
                                  py::array_t<const float, py::array::c_style | py::array::forcecast> theta_input,
                                  int K, int L,
                                  int threads,
                                  py::array_t<const int, py::array::c_style | py::array::forcecast> binned_nodes,
                                  py::array_t<const int, py::array::c_style | py::array::forcecast> binned_voxels) {
            //
            const float* ptr_theta = static_cast<const float*>(theta_input.data());
            const int* ptr_binned_nodes = static_cast<const int*>(binned_nodes.data());
            const int* ptr_binned_voxels = static_cast<const int*>(binned_voxels.data());

            // Call the actual method
            self.setMILikelihood(alpha, ptr_theta, K, L, threads, ptr_binned_nodes, ptr_binned_voxels);
        },
        py::arg("alpha"),
        py::arg("theta"),
        py::arg("K"),
        py::arg("L"),
        py::arg("threads"),
        py::arg("binned_nodes"),
        py::arg("binned_voxels"))

        .def("setNonLinearTransformation", [](Registration& self,
                                              float gamma_x, float gamma_y, float gamma_z,
                                              const float voxels_size_x, const float voxels_size_y, const float voxels_size_z,
											  const float nodes_size_x, const float nodes_size_y, const float nodes_size_z,
                                              py::array_t<const float, py::array::c_style | py::array::forcecast> D){
            //
            const float* ptr_D = static_cast<const float*>(D.data());

            // Call the actual method
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
        py::arg("D"))

        .def("get_likelihood_parameters", &Registration::get_likelihood_parameters, py::return_value_policy::copy)

        .def("EM", &Registration::EM,
             py::arg("max_EM_iterations"),
             py::arg("convergence_th"),
             py::arg("update_likelihood_parameters"),
             py::arg("debug"))

        .def("sampler", [](Registration& self, int seed, int N_b, int N_s,
             								   int sample_gamma, int sample_gamma_every, int sample_posteriors,
             								   int sample_likelihood_parameters,
             								   int sample_transformation_parameters){

          	auto [meanDefField, covDefField] = self.sampler(seed, N_b, N_s, sample_gamma, sample_gamma_every,
															 sample_posteriors,
                       	 					     			 sample_likelihood_parameters,
                                                             sample_transformation_parameters);

            std::vector<int> shape = self.get_shape();
    		// Create numpy arrays directly from vectors
    		py::array_t<float> result_array_1(shape, meanDefField.data());

		    std::vector<int> shape_cov = {shape[0], shape[1], shape[2], shape[3], shape[3]};
    		py::array_t<float> result_array_2(shape_cov, covDefField.data());

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

        .def("get_dream_locations", &Registration::get_dream_locations, py::return_value_policy::copy)

        .def("set_dream_locations", [](Registration& self,
                                       py::array_t<const float, py::array::c_style | py::array::forcecast> dream_locations){
          	// Get raw pointer to dream_locations
            const float* ptr_dream_locations = static_cast<const float*>(dream_locations.data());
            self.set_dream_locations(ptr_dream_locations);
        })

		.def("get_final_locations_as_numpy_array", [](Registration& self) {
    		auto locations = self.get_final_locations();  // Get copy of data as vector
    		std::vector<int> shape = self.get_shape();

		    // Create numpy array from the vector data
    		return py::array_t<float>(
        		shape,                     // shape
        		{                         // strides
            		sizeof(float) * shape[1] * shape[2] * shape[3],
            		sizeof(float) * shape[2] * shape[3],
            		sizeof(float) * shape[3],
            		sizeof(float)
		        },
    		    locations.data()          // data pointer
    		);
		});

	// Define the NonLinearTransformation class binding
	py::class_<NonLinearTransformation, std::unique_ptr<NonLinearTransformation>>(m, "NonLinearTransformation")
		.def(py::init([](float gamma_x, float gamma_y, float gamma_z,
				  		 const float voxels_size_x, const float voxels_size_y, const float voxels_size_z,
		                 const float nodes_size_x, const float nodes_size_y, const float nodes_size_z,
						 py::array_t<const float, py::array::c_style | py::array::forcecast> D,
						 int Nx_v, int Ny_v, int Nz_v,
						 int threads,
						 float sigma_spline,
						 float spline_offset) {

			// Get raw pointer to D
			const float* ptr_D = static_cast<const float*>(D.data());
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
		py::arg("spline_offset"))
		.def("get_shape", &NonLinearTransformation::get_shape)

        .def("smooth_deformation", [](NonLinearTransformation& self,
        		                      py::array_t<const float, py::array::c_style | py::array::forcecast> deformation) {
		    const float* ptr_deformation = static_cast<const float*>(deformation.data());

		    auto [log_prior_deformation, new_deformation] = self.smoothDeformation(ptr_deformation);
    		std::vector<int> shape = self.get_shape();

		    auto result_array = py::array(
        		py::buffer_info(
            	new_deformation,
            	sizeof(float),
            	py::format_descriptor<float>::format(),
            	shape.size(),
            	shape,
            	{
                	sizeof(float) * shape[1] * shape[2] * shape[3],
                	sizeof(float) * shape[2] * shape[3],
                	sizeof(float) * shape[3],
                	sizeof(float)
            	}
        		));
    		return py::make_tuple(log_prior_deformation, result_array);
			},
			py::arg("deformation"))

   		.def("sampleDeformationField",
    		[](NonLinearTransformation& self,
                    py::array_t<float, py::array::c_style | py::array::forcecast> deformationField,
                    int number_of_samples) {
            const float* ptr_deformation = static_cast<const float*>(deformationField.data());

	        float* raw_data = self.sampleDeformationField(ptr_deformation, number_of_samples);

    	    std::vector<int> shape_tmp = self.get_shape();
        	std::vector<int> shape = {number_of_samples, shape_tmp[0], shape_tmp[1], shape_tmp[2], shape_tmp[3]};

	        auto result_array = py::array(py::buffer_info(
    	        raw_data,
        	    sizeof(float),
            	py::format_descriptor<float>::format(),
	            shape.size(),
    	        shape,
        	    {
            	    sizeof(float) * shape[1] * shape[2] * shape[3] * shape[4],
                	sizeof(float) * shape[2] * shape[3] * shape[4],
    	            sizeof(float) * shape[3] * shape[4],
        	        sizeof(float) * shape[4],
            	    sizeof(float)
            	}
        	));

	        return result_array;
    	},
    	py::arg("deformationField"),
    	py::arg("number_of_samples"));

#ifdef BINDER_WITH_TOPOLOGY_CORRECTION
	// Define the TopologyCorrection3D class binding
	py::class_<topology_correction_3D>(m, "topology_correction_3D")
	.def(py::init([]() {
		return new topology_correction_3D();
	}))
	.def("correct_topology", [](topology_correction_3D& self,
								py::array_t<float const, py::array::c_style | py::array::forcecast> h_init,
								float e1, float e2, int max_outer_it, int max_it,
								int threads, py::array_t<float const, py::array::c_style | py::array::forcecast> voxel_sizes,
								int Nx_v, int Ny_v, int Nz_v) {
		// Get raw pointer to h_init
		const float* ptr_h_init = static_cast<const float*>(h_init.data());
		// Get raw pointer to voxel_sizes
		const float* ptr_voxel_sizes = static_cast<const float*>(voxel_sizes.data());
		float* raw_data = self.correct_topology(ptr_h_init, e1, e2, max_outer_it, max_it, threads, ptr_voxel_sizes,
												 Nx_v, Ny_v, Nz_v);
		std::vector<int> shape = {Nx_v, Ny_v, Nz_v, 3};

		return py::array(py::buffer_info(
			raw_data,
			sizeof(float),
			py::format_descriptor<float>::format(),
			shape.size(),
			shape,
			{sizeof(float) * shape[1] * shape[2] * shape[3],
			 sizeof(float) * shape[2] * shape[3],
			 sizeof(float) * shape[3],
			 sizeof(float)}
		));
	});
#endif // BINDER_WITH_TOPOLOGY_CORRECTION

}