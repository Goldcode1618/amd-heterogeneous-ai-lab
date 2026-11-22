// ACAD-A04: Cross-Architecture HIP Parallel Reduction (Wavefront32 vs Wavefront64)
// Demonstrates portable shuffle reduction across:
//   - AMD RDNA 3 (Radeon 760M gfx1103): __AMDGCN_WAVEFRONT_SIZE == 32
//   - AMD CDNA 3 (Instinct MI300X gfx942): __AMDGCN_WAVEFRONT_SIZE == 64
//
// Compile for both targets on ROCm:
//   hipcc -O3 --offload-arch=gfx1103,gfx942 wavefront_reduction.hip.cpp -o wf_reduce
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#ifndef __AMDGCN_WAVEFRONT_SIZE
#define __AMDGCN_WAVEFRONT_SIZE 64
#endif

constexpr int WF_SIZE = __AMDGCN_WAVEFRONT_SIZE;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WF_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, WF_SIZE);
    }
    return val;
}

__global__ void block_reduce_sum_kernel(int n, const float* __restrict__ in,
                                        float* __restrict__ block_out) {
    __shared__ float smem[32];  // Supports up to 1024 threads / 32 = 32 wavefronts per block
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < n) ? in[gid] : 0.0f;
    val = warp_reduce_sum(val);

    const int lane = tid & (WF_SIZE - 1);
    const int wid = tid / WF_SIZE;
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    const int num_wfs = (blockDim.x + WF_SIZE - 1) / WF_SIZE;
    val = (tid < num_wfs) ? smem[tid] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
        if (tid == 0) {
            block_out[blockIdx.x] = val;
        }
    }
}

int main() {
    constexpr int N = 1 << 20;
    constexpr int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_partial(GRID, 0.0f);

    float *d_in = nullptr, *d_partial = nullptr;
    hipMalloc(&d_in, N * sizeof(float));
    hipMalloc(&d_partial, GRID * sizeof(float));
    hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(block_reduce_sum_kernel, dim3(GRID), dim3(BLOCK), 0, 0,
                       N, d_in, d_partial);
    hipMemcpy(h_partial.data(), d_partial, GRID * sizeof(float), hipMemcpyDeviceToHost);

    double total = 0.0;
    for (float v : h_partial) total += static_cast<double>(v);
    std::printf("WavefrontSize=%d | Reduced Sum=%.1f (expected=%d)\n", WF_SIZE, total, N);

    hipFree(d_in);
    hipFree(d_partial);
    return (static_cast<int>(total) == N) ? 0 : 1;
}
