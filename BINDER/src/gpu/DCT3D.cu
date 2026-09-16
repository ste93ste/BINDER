#include "DCT3D.cuh"

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

// ---------------- device helpers ----------------
__device__ __forceinline__ int idx3(int i1, int i2, int i3, int N2, int N3)
{
    return (i1 * N2 + i2) * N3 + i3;
}
__device__ __forceinline__ int idx3h(int k1, int k2, int k3, int N2, int Nh)
{
    return (k1 * N2 + k2) * Nh + k3;
}
__device__ __forceinline__ int makhoul_map(int i, int N)
{
    int half = (N + 1) >> 1;
    return (i < half) ? (2 * i) : (2 * (N - 1 - i) + 1);
}
__device__ __forceinline__ int makhoul_inv_map(int j, int N)
{
    return ((j & 1) == 0) ? (j >> 1) : (N - 1 - ((j - 1) >> 1));
}
__device__ __forceinline__ int flip0(int k, int N)
{
    return (k == 0) ? 0 : (N - k);
}
__device__ __forceinline__ int mod_neg(int k, int N)
{
    return (k == 0) ? 0 : (N - k);
}

// ---------------- kernels ----------------
__global__ void k_reorder_to_fft_in(const float *__restrict__ x,
                                    float *__restrict__ v,
                                    int N1, int N2, int N3)
{
    int i3 = blockIdx.x * blockDim.x + threadIdx.x; // fastest / contiguous
    int i2 = blockIdx.y * blockDim.y + threadIdx.y;
    int i1 = blockIdx.z * blockDim.z + threadIdx.z;
    if (i1 >= N1 || i2 >= N2 || i3 >= N3)
        return;

    int j1 = makhoul_map(i1, N1);
    int j2 = makhoul_map(i2, N2);
    int j3 = makhoul_map(i3, N3);

    v[idx3(i1, i2, i3, N2, N3)] = x[idx3(j1, j2, j3, N2, N3)];
}

__global__ void k_unreorder_from_fft_out_scaled(const float *__restrict__ v,
                                         float *__restrict__ x,
                                         int N1, int N2, int N3,
                                         float scale)
{
    int j3 = blockIdx.x * blockDim.x + threadIdx.x;
    int j2 = blockIdx.y * blockDim.y + threadIdx.y;
    int j1 = blockIdx.z * blockDim.z + threadIdx.z;
    if (j1 >= N1 || j2 >= N2 || j3 >= N3)
        return;

    int i1 = makhoul_inv_map(j1, N1);
    int i2 = makhoul_inv_map(j2, N2);
    int i3 = makhoul_inv_map(j3, N3);

    x[idx3(j1, j2, j3, N2, N3)] = v[idx3(i1, i2, i3, N2, N3)]* scale;
}


// Forward merged recombination: Vh (R2C half-spectrum) -> C (full real).
// Uses precomputed cos/sin tables for θ1, θ2, θ3 and computes cos/sin(θ1±θ2±θ3)
// using addition identities (no trig in the kernel).
// Ortho scaling (if enabled) is computed inlin.
__global__ void k_dct3_forward_merge_tables(const cufftComplex *__restrict__ Vh,
                                            float *__restrict__ C,
                                            const float *__restrict__ c1,
                                            const float *__restrict__ s1,
                                            const float *__restrict__ c2,
                                            const float *__restrict__ s2,
                                            const float *__restrict__ c3,
                                            const float *__restrict__ s3,
                                            int N1, int N2, int N3,
                                            int ortho_flag)
{
    int k3 = blockIdx.x * blockDim.x + threadIdx.x;
    int k2 = blockIdx.y * blockDim.y + threadIdx.y;
    int k1 = blockIdx.z * blockDim.z + threadIdx.z;
    if (k1 >= N1 || k2 >= N2 || k3 >= N3)
        return;

    int Nh = N3 / 2 + 1;

    // get full-spectrum V(k1,k2,k3_full) from half-spectrum via 3D Hermitian symmetry
    auto getV = [&](int a1, int a2, int a3) -> cufftComplex
    {
        if (a3 < Nh)
        {
            return Vh[idx3h(a1, a2, a3, N2, Nh)];
        }
        else
        {
            int b3 = N3 - a3;
            int b1 = mod_neg(a1, N1);
            int b2 = mod_neg(a2, N2);
            cufftComplex t = Vh[idx3h(b1, b2, b3, N2, Nh)];
            t.y = -t.y; // conj
            return t;
        }
    };

    int k2b = flip0(k2, N2);
    int k3b = flip0(k3, N3);

    cufftComplex V0 = getV(k1, k2, k3);
    cufftComplex Vf3 = getV(k1, k2, k3b);
    cufftComplex Vf2 = getV(k1, k2b, k3);
    cufftComplex Vf23 = getV(k1, k2b, k3b);

    float ca = c1[k1], sa = s1[k1];
    float cb = c2[k2], sb = s2[k2];
    float cc = c3[k3], sc = s3[k3];

    // compute cos/sin(a + sB*b + sC*c)
    auto cos_sin_abc = [&](int sB, int sC, float &co, float &si)
    {
        float cos_ab = ca * cb - (float)sB * sa * sb;
        float sin_ab = sa * cb + (float)sB * ca * sb;
        co = cos_ab * cc - (float)sC * sin_ab * sc;
        si = sin_ab * cc + (float)sC * cos_ab * sc;
    };

    float c_p1, s_p1, c_p2, s_p2, c_p3, s_p3, c_p4, s_p4;
    cos_sin_abc(+1, +1, c_p1, s_p1); // θ1 + θ2 + θ3
    cos_sin_abc(+1, -1, c_p2, s_p2); // θ1 + θ2 - θ3
    cos_sin_abc(-1, +1, c_p3, s_p3); // θ1 - θ2 + θ3
    cos_sin_abc(-1, -1, c_p4, s_p4); // θ1 - θ2 - θ3

    // Re(e^{-jp}Z) = cos(p)*Re(Z) + sin(p)*Im(Z)
    float out =
        c_p1 * V0.x + s_p1 * V0.y +
        c_p2 * Vf3.x + s_p2 * Vf3.y +
        c_p3 * Vf2.x + s_p3 * Vf2.y +
        c_p4 * Vf23.x + s_p4 * Vf23.y;

    float val = 2.0f * out;

    // Ortho scaling inline:
    // C_ortho = (alpha1*alpha2*alpha3)/8 * C_un
    if (ortho_flag)
    {
        float a1 = (k1 == 0) ? rsqrtf((float)N1) : sqrtf(2.0f / (float)N1);
        float a2 = (k2 == 0) ? rsqrtf((float)N2) : sqrtf(2.0f / (float)N2);
        float a3 = (k3 == 0) ? rsqrtf((float)N3) : sqrtf(2.0f / (float)N3);
        val *= (a1 * a2 * a3) * 0.125f; // /8
    }

    C[idx3(k1, k2, k3, N2, N3)] = val;
}

// Inverse: build Vh (half-spectrum) directly from C using packed merged inverse.
__device__ __forceinline__ int idxC(int i1, int i2, int i3, int N2, int N3)
{
    return (i1 * N2 + i2) * N3 + i3;
}
__device__ __forceinline__ int idxH(int k1, int k2, int k3, int N2, int Nh)
{
    return (k1 * N2 + k2) * Nh + k3;
}

template <bool ORTHO>
__global__ void k_idct3_build_Vh_tables_fast(
    const float* __restrict__ C,
    cufftComplex* __restrict__ Vh,
    const float* __restrict__ c1, const float* __restrict__ s1,
    const float* __restrict__ c2, const float* __restrict__ s2,
    const float* __restrict__ c3, const float* __restrict__ s3,
    const float* __restrict__ a1,
    const float* __restrict__ a2,
    const float* __restrict__ a3,
    int N1, int N2, int N3)
{
    int k3 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int k2 = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int k1 = (int)(blockIdx.z * blockDim.z + threadIdx.z);

    int Nh = N3 / 2 + 1;
    if (k1 >= N1 || k2 >= N2 || k3 >= Nh) return;

    int k3b = (k3 == 0) ? 0 : (N3 - k3);

    // helper: load C_un at (i1,i2,i3)
    auto load_Cun = [&](int i1, int i2, int i3) {
        float v = C[idxC(i1, i2, i3, N2, N3)];
        if constexpr (ORTHO) {
            // undo forward scaling: * 8 / (a1*a2*a3)
            v *= 8.0f / (a1[i1] * a2[i2] * a3[i3]);
        }
        return v;
    };

    // Chat_at(i1,i2) = C_un(i1,i2,k3) - j*C_un(i1,i2,k3b)
    auto load_Chat = [&](int i1, int i2) {
        float re = load_Cun(i1, i2, k3);
        float im = 0.0f;
        if (k3 != 0) im = -load_Cun(i1, i2, k3b);
        return cufftComplex{re, im};
    };

    cufftComplex Chat = load_Chat(k1, k2);

    cufftComplex A = Chat;
    cufftComplex B{0.0f, 0.0f};

    if (k1 > 0 && k2 > 0) {
        cufftComplex t = load_Chat(N1 - k1, N2 - k2);
        A.x -= t.x; A.y -= t.y;
    }
    if (k1 > 0) {
        cufftComplex t = load_Chat(N1 - k1, k2);
        B.x += t.x; B.y += t.y;
    }
    if (k2 > 0) {
        cufftComplex t = load_Chat(k1, N2 - k2);
        B.x += t.x; B.y += t.y;
    }

    float Xr = A.x + B.y;
    float Xi = A.y - B.x;

    float ca = c1[k1], sa = s1[k1];
    float cb = c2[k2], sb = s2[k2];
    float cc = c3[k3], sc = s3[k3];

    float cos_ab = ca * cb - sa * sb;
    float sin_ab = sa * cb + ca * sb;
    float cphi = cos_ab * cc - sin_ab * sc;
    float sphi = sin_ab * cc + cos_ab * sc;

    cufftComplex V;
    V.x = 0.125f * (cphi * Xr - sphi * Xi);
    V.y = 0.125f * (sphi * Xr + cphi * Xi);

    Vh[idxH(k1, k2, k3, N2, Nh)] = V;
}


// ---------------- DCT3D class ----------------

DCT3D::DCT3D(int n1, int n2, int n3, bool ortho)
    : N1(n1), N2(n2), N3(n3), ortho(ortho)
{
    if (N1 <= 0 || N2 <= 0 || N3 <= 0)
        throw std::runtime_error("Invalid sizes");
    Nh = N3 / 2 + 1;
    stream = 0;

    // allocate core buffers
    size_t real_bytes = (size_t)N1 * N2 * N3 * sizeof(float);
    size_t half_bytes = (size_t)N1 * N2 * Nh * sizeof(cufftComplex);

    CHECK_CUDA(cudaMalloc(&d_v, real_bytes));   // FFT real in/out
    CHECK_CUDA(cudaMalloc(&d_Vh, half_bytes));  // half-spectrum

    // trig per-axis (k3 full range 0..N3-1 for forward merge)
    CHECK_CUDA(cudaMalloc(&d_c1, (size_t)N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s1, (size_t)N1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c2, (size_t)N2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s2, (size_t)N2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c3, (size_t)N3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s3, (size_t)N3 * sizeof(float)));

    build_trig_tables();

    CHECK_CUDA(cudaMalloc(&d_a1, N1*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a2, N2*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a3, N3*sizeof(float)));

    // ---- build ortho scaling factors on CPU ----
    std::vector<float> h_a1(N1);
    std::vector<float> h_a2(N2);
    std::vector<float> h_a3(N3);

    for (int k = 0; k < N1; ++k)
    {
        if (k == 0)
            h_a1[k] = 1.0f / std::sqrt((float)N1);
        else
            h_a1[k] = std::sqrt(2.0f / (float)N1);
    }

    for (int k = 0; k < N2; ++k)
    {
        if (k == 0)
            h_a2[k] = 1.0f / std::sqrt((float)N2);
        else
            h_a2[k] = std::sqrt(2.0f / (float)N2);
    }

    for (int k = 0; k < N3; ++k)
    {
        if (k == 0)
            h_a3[k] = 1.0f / std::sqrt((float)N3);
        else
            h_a3[k] = std::sqrt(2.0f / (float)N3);
    }

    // ---- copy to device ----
    CHECK_CUDA(cudaMemcpy(d_a1, h_a1.data(), N1*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a2, h_a2.data(), N2*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_a3, h_a3.data(), N3*sizeof(float), cudaMemcpyHostToDevice));
    
    // cuFFT plans
    CHECK_CUFFT(cufftPlan3d(&plan_r2c, N1, N2, N3, CUFFT_R2C));
    CHECK_CUFFT(cufftPlan3d(&plan_c2r, N1, N2, N3, CUFFT_C2R));
    CHECK_CUFFT(cufftSetStream(plan_r2c, stream));
    CHECK_CUFFT(cufftSetStream(plan_c2r, stream));
}

DCT3D::~DCT3D()
{
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    cudaFree(d_v);
    cudaFree(d_Vh);
    
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_a3);

    cudaFree(d_c1);
    cudaFree(d_s1);
    cudaFree(d_c2);
    cudaFree(d_s2);
    cudaFree(d_c3);
    cudaFree(d_s3);
}

void DCT3D::forward(const float *d_in, float *d_out)
{

    dim3 block(32, 4, 2);
    dim3 grid(div_up(N3, block.x), div_up(N2, block.y), div_up(N1, block.z));

    // reorder
    k_reorder_to_fft_in<<<grid, block, 0, stream>>>(d_in, d_v, N1, N2, N3);
    CHECK_CUDA(cudaGetLastError());

    // R2C FFT
    CHECK_CUFFT(cufftExecR2C(plan_r2c, (cufftReal *)d_v, (cufftComplex *)d_Vh));

    // merge (ortho inline)
    k_dct3_forward_merge_tables<<<grid, block, 0, stream>>>(
        (cufftComplex *)d_Vh, d_out,
        d_c1, d_s1, d_c2, d_s2, d_c3, d_s3,
        N1, N2, N3,
        ortho ? 1 : 0);
    CHECK_CUDA(cudaGetLastError());

}

void DCT3D::backward(const float *d_in, float *d_out)
{

    dim3 block(32, 4, 2);
    dim3 grid_half(div_up(Nh, block.x), div_up(N2, block.y), div_up(N1, block.z));
    dim3 grid_full(div_up(N3, block.x), div_up(N2, block.y), div_up(N1, block.z));

    // build Vh (phi + ortho computed inline)
    if (ortho) {
        k_idct3_build_Vh_tables_fast<true><<<grid_half, block, 0, stream>>>(
            d_in, (cufftComplex*)d_Vh,
            d_c1, d_s1, d_c2, d_s2, d_c3, d_s3,
            d_a1, d_a2, d_a3,
            N1, N2, N3);
    } else {
        k_idct3_build_Vh_tables_fast<false><<<grid_half, block, 0, stream>>>(
            d_in, (cufftComplex*)d_Vh,
            d_c1, d_s1, d_c2, d_s2, d_c3, d_s3,
            nullptr, nullptr, nullptr,
            N1, N2, N3);
    }
    CHECK_CUDA(cudaGetLastError());

    // C2R IFFT (unnormalized)
    CHECK_CUFFT(cufftExecC2R(plan_c2r, (cufftComplex *)d_Vh, (cufftReal *)d_v));

    float scale = 1.0f / (float)(N1 * N2 * N3);
    // unreorder
    k_unreorder_from_fft_out_scaled<<<grid_full, block, 0, stream>>>(d_v, d_out, N1, N2, N3, scale);
    CHECK_CUDA(cudaGetLastError());

}
