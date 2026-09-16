#include "FFT.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

FFT::FFT(const int N_x, const int N_y, const int N_z)
    : N_x(N_x), N_y(N_y), N_z(N_z) {
    
    // Create cuFFT plans
    if (N_z == 1) {
        // Create 2D plans
        if (cufftPlan2d(&plan_forward, N_x, N_y, CUFFT_Z2Z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create forward 2D FFT plan");
        }
        if (cufftPlan2d(&plan_backward, N_x, N_y, CUFFT_Z2Z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create backward 2D FFT plan");
        }
    } else {
        // Create 3D plans
        if (cufftPlan3d(&plan_forward, N_x, N_y, N_z, CUFFT_Z2Z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create forward 3D FFT plan");
        }
        if (cufftPlan3d(&plan_backward, N_x, N_y, N_z, CUFFT_Z2Z) != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create backward 3D FFT plan");
        }
    }
}

FFT::~FFT() {
    cufftDestroy(plan_forward);
    cufftDestroy(plan_backward);
}

void FFT::fft(cufftDoubleComplex* d_input, cufftDoubleComplex* d_output) const {
    if (const cufftResult result = cufftExecZ2Z(plan_forward, d_input, d_output, CUFFT_FORWARD);
        result != CUFFT_SUCCESS) {
        throw std::runtime_error("Forward FFT execution failed");
    }
}

void FFT::ifft(cufftDoubleComplex* d_input, cufftDoubleComplex* d_output) const {
    if (const cufftResult result = cufftExecZ2Z(plan_backward, d_input, d_output, CUFFT_INVERSE);
        result != CUFFT_SUCCESS) {
        throw std::runtime_error("Inverse FFT execution failed");
    }
}

void FFT::fft(const thrust::device_ptr<cufftDoubleComplex> d_input,
              const thrust::device_ptr<cufftDoubleComplex> d_output) const {
    fft(raw_pointer_cast(d_input),
        raw_pointer_cast(d_output));
}

void FFT::ifft(const thrust::device_ptr<cufftDoubleComplex> d_input,
               const thrust::device_ptr<cufftDoubleComplex> d_output) const {
    ifft(raw_pointer_cast(d_input),
         raw_pointer_cast(d_output));
}
