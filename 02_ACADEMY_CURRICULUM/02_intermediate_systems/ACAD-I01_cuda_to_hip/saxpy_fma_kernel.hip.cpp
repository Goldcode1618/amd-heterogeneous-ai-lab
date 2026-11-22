// ACAD-I01: Native AMD HIP C++ SAXPY + Fused Multiply-Add Kernel
// Compile on ROCm (CDNA 3 MI300X or RDNA 3):
//   hipcc -O3 --offload-arch=gfx942,gfx1103 saxpy_fma_kernel.hip.cpp -o saxpy_fma
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = (call);                                             \
        if (err != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                         hipGetErrorString(err));                            \
            return 1;                                                        \
        }                                                                    \
    } while (0)

__global__ void saxpy_fma_kernel(int n, float alpha, const float* __restrict__ x,
                                 float* __restrict__ y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = __fmaf_rn(alpha, x[idx], y[idx]);
    }
}

int main() {
    constexpr int N = 1 << 22;  // 4,194,304 elements (16 MB per vector)
    constexpr float alpha = 2.5f;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_x(N, 1.5f);
    std::vector<float> h_y(N, 2.0f);

    float *d_x = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), bytes, hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    constexpr int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(saxpy_fma_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       N, alpha, d_x, d_y);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, bytes, hipMemcpyDeviceToHost));

    // Read x, Read y, Write y -> 3 * bytes transferred
    const double gb_per_sec = (3.0 * static_cast<double>(bytes)) / (ms * 1e6);
    std::printf("SAXPY FMA N=%d | Elapsed=%.4f ms | Effective BW=%.2f GB/s | y[0]=%.2f\n",
                N, ms, gb_per_sec, h_y[0]);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return (std::fabs(h_y[0] - 5.75f) < 1e-5f) ? 0 : 2;
}
