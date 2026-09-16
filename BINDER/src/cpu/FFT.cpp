//
// Created by stce on 02/01/25.
//

#include "FFT.h"
#include <complex.h>
#include <fftw3.h>

FFT::FFT(const int N_x, const int N_y, const int N_z, const int threads){

    const int num_voxels = N_x * N_y * N_z;
    this->fft_in = new fftw_complex [num_voxels];
    this->fft_out = new fftw_complex [num_voxels];
    this->ifft_in = new fftw_complex [num_voxels];
    this->ifft_out = new fftw_complex [num_voxels];
    this->N_x = N_x;
    this->N_y = N_y;
    this->N_z = N_z;
    this->threads = threads;
    fftw_init_threads();
    fftw_plan_with_nthreads(threads);
    if (this->N_z == 1) {
        // Create 2D plans
        this->plan_forward = fftw_plan_dft_2d(
            N_x, N_y, this->fft_in, this->fft_out, FFTW_FORWARD, FFTW_MEASURE);
        this->plan_backward = fftw_plan_dft_2d(
            N_x, N_y, this->ifft_in, this->ifft_out, FFTW_BACKWARD, FFTW_MEASURE);
    } else {
        // Create 3D plans
        this->plan_forward = fftw_plan_dft_3d(
            N_x, N_y, N_z, this->fft_in, this->fft_out, FFTW_FORWARD, FFTW_MEASURE);
        this->plan_backward = fftw_plan_dft_3d(
            N_x, N_y, N_z, this->ifft_in, this->ifft_out, FFTW_BACKWARD, FFTW_MEASURE);
    }
}

FFT::~FFT(){
    fftw_destroy_plan(this->plan_forward);
    fftw_destroy_plan(this->plan_backward);
    delete[] this->fft_in;
    delete[] this->fft_out;
    delete[] this->ifft_in;
    delete[] this->ifft_out;
}

void FFT::fft() const {
    fftw_execute(this->plan_forward);
}

void FFT::ifft() const {
    fftw_execute(this->plan_backward);
}
