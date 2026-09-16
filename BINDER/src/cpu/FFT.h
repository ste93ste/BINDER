//
// Created by stce on 02/01/25.
//

#ifndef FFT_H
#define FFT_H
#include <complex.h>
#include <fftw3.h>

class FFT{

public:
    FFT(int N_x, int N_y, int N_z, int threads);
    ~FFT();
    void fft() const;
    void ifft() const;
    fftw_complex* fft_in;
    fftw_complex* fft_out;
    fftw_complex* ifft_in;
    fftw_complex* ifft_out;

private:
    fftw_plan plan_forward;
    fftw_plan plan_backward;
    int threads;
    int N_x;
    int N_y;
    int N_z;
};



#endif //FFT_H
