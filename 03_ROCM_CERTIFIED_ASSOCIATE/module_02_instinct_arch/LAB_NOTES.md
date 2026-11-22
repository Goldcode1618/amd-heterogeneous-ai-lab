# Module 02: AMD Instinct CDNA 3 Architecture & HIP Programming — Lab Notes

## Objectives
- Understand CDNA 3 Matrix Cores (`MFMA` instructions), 8-chiplet XCD topology, and 192 GB HBM3 unified memory bandwidth (`5.3 TB/s`) on **AMD Instinct MI300X (`gfx942`)**.
- Compile and execute native HIP kernels (`hipcc`) with explicit `hipEventElapsedTime` synchronization.
- Write wavefront-size-portable kernels (`__AMDGCN_WAVEFRONT_SIZE`: `32` on RDNA 3 vs `64` on CDNA 3).

## Key Commands & Recipes
```bash
# Inspect compiler version and target offload architectures
hipcc --version

# Compile SAXPY + FMA kernel for both MI300X (gfx942) and RDNA 3 (gfx1103)
hipcc -O3 --offload-arch=gfx942,gfx1103 \
  ../../02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I01_cuda_to_hip/saxpy_fma_kernel.hip.cpp \
  -o saxpy_fma && ./saxpy_fma

# Compile wavefront-portable shuffle reduction (WF32 vs WF64)
hipcc -O3 --offload-arch=gfx942,gfx1103 \
  ../../02_ACADEMY_CURRICULUM/03_advanced_rdna_hip/ACAD-A04_wavefront32_vs_64/wavefront_reduction.hip.cpp \
  -o wf_reduce && ./wf_reduce
```

## Notes & Observations
- **Wavefront Scheduling:** Hardcoding `32` in warp-shuffle masks breaks CDNA 3 reductions (`64` threads/wavefront). Always guard shuffle loops with `__AMDGCN_WAVEFRONT_SIZE` (see [`wavefront_reduction.hip.cpp`](../../02_ACADEMY_CURRICULUM/03_advanced_rdna_hip/ACAD-A04_wavefront32_vs_64/wavefront_reduction.hip.cpp)).
- **LDS Bank Conflicts:** Local Data Share (64 KB/CU) uses 32 banks; padding shared memory tile widths by `+1` (`__shared__ float tile[16][17]`) eliminates bank conflicts in [`tiled_gemm_lds.hip.cpp`](../../02_ACADEMY_CURRICULUM/03_advanced_rdna_hip/ACAD-A05_lds_shared_memory/tiled_gemm_lds.hip.cpp).
