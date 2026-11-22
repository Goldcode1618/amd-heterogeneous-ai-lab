// ACAD-A05: Bank-Conflict-Free Tiled GEMM Using AMD Local Data Share (LDS)
// Pads LDS tile stride (TILE + 1) to eliminate 32-bank LDS conflicts on RDNA 3 / CDNA 3.
//
// Compile on ROCm:
//   hipcc -O3 --offload-arch=gfx942,gfx1103 tiled_gemm_lds.hip.cpp -o tiled_gemm_lds
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int TILE = 16;

__global__ void tiled_gemm_lds_kernel(int n, const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C) {
    // +1 column padding prevents 32-bank LDS conflicts during column-wise reads
    __shared__ float tileA[TILE][TILE + 1];
    __shared__ float tileB[TILE][TILE + 1];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc = __fmaf_rn(tileA[threadIdx.y][k], tileB[k][threadIdx.x], acc);
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

int main() {
    constexpr int N = 256;
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    std::vector<float> h_A(N * N, 1.0f), h_B(N * N, 2.0f), h_C(N * N, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);
    hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    hipLaunchKernelGGL(tiled_gemm_lds_kernel, grid, block, 0, 0, N, d_A, d_B, d_C);
    hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost);

    const float expected = static_cast<float>(N) * 2.0f;
    std::printf("Tiled LDS GEMM (%dx%d) C[0]=%.1f (expected=%.1f)\n", N, N, h_C[0], expected);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    return (std::fabs(h_C[0] - expected) < 1e-3f) ? 0 : 1;
}
