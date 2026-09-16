#ifndef DCT3D_CUH
#define DCT3D_CUH

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(x)                                           \
    do                                                          \
    {                                                           \
        cudaError_t _err = (x);                                 \
        if (_err != cudaSuccess)                                \
            throw std::runtime_error(cudaGetErrorString(_err)); \
    } while (0)
#define CHECK_CUFFT(x)                               \
    do                                               \
    {                                                \
        cufftResult _e = (x);                        \
        if (_e != CUFFT_SUCCESS)                     \
            throw std::runtime_error("cuFFT error"); \
    } while (0)
    
class DCT3D
{
private:
    int N1, N2, N3, Nh;
    bool ortho = false;
    bool timing_enabled = false;

    cudaStream_t stream{};
    cufftHandle plan_r2c{}, plan_c2r{};

    // device buffers
    float *d_v = nullptr;
    cufftComplex *d_Vh = nullptr;

    // trig tables
    float *d_c1 = nullptr, *d_s1 = nullptr;
    float *d_c2 = nullptr, *d_s2 = nullptr;
    float *d_c3 = nullptr, *d_s3 = nullptr;

    float* d_a1 = nullptr;
    float* d_a2 = nullptr;
    float* d_a3 = nullptr;

    void build_trig_tables()
    {
        std::vector<float> hc1(N1), hs1(N1), hc2(N2), hs2(N2), hc3(N3), hs3(N3);
        for (int k = 0; k < N1; k++)
        {
            float th = (float)M_PI * k / (2.0f * N1);
            hc1[k] = cosf(th);
            hs1[k] = sinf(th);
        }
        for (int k = 0; k < N2; k++)
        {
            float th = (float)M_PI * k / (2.0f * N2);
            hc2[k] = cosf(th);
            hs2[k] = sinf(th);
        }
        for (int k = 0; k < N3; k++)
        {
            float th = (float)M_PI * k / (2.0f * N3);
            hc3[k] = cosf(th);
            hs3[k] = sinf(th);
        }
        CHECK_CUDA(cudaMemcpy(d_c1, hc1.data(), (size_t)N1 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_s1, hs1.data(), (size_t)N1 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_c2, hc2.data(), (size_t)N2 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_s2, hs2.data(), (size_t)N2 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_c3, hc3.data(), (size_t)N3 * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_s3, hs3.data(), (size_t)N3 * sizeof(float), cudaMemcpyHostToDevice));
    }

public:
    DCT3D(int n1, int n2, int n3, bool ortho = true);
    ~DCT3D();

    void forward(const float *input, float *output);
    void backward(const float *input, float *output);
};

#endif // DCT3D_CUH