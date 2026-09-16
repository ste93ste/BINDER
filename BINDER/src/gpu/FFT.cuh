#ifndef FFT_CUH
#define FFT_CUH

#include <cufft.h>
#include <thrust/device_ptr.h>

class FFT {
    cufftHandle plan_forward{};
    cufftHandle plan_backward{};
    int N_x, N_y, N_z;

public:
    // Constructor
    FFT(int N_x, int N_y, int N_z);
    
    // Destructor
    ~FFT();
    
    // FFT methods that work directly with device pointers
    void fft(cufftDoubleComplex* d_input, cufftDoubleComplex* d_output) const;
    void ifft(cufftDoubleComplex* d_input, cufftDoubleComplex* d_output) const;
    
    // Optional methods for working with thrust device pointers
    void fft(thrust::device_ptr<cufftDoubleComplex> d_input,
             thrust::device_ptr<cufftDoubleComplex> d_output) const;
    void ifft(thrust::device_ptr<cufftDoubleComplex> d_input,
              thrust::device_ptr<cufftDoubleComplex> d_output) const;
              
    // Getter methods for dimensions
    int get_Nx() const { return N_x; }
    int get_Ny() const { return N_y; }
    int get_Nz() const { return N_z; }
};

#endif // FFT_CUH