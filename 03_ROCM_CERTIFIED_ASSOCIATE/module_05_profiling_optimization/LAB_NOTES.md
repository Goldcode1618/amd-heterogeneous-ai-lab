# Module 05: ROCm Profiling (`rocprofv2` / `omniperf`) & Production Deployment — Lab Notes

## Objectives
- Profile HIP kernels, HBM3 memory bandwidth, and wave occupancy using `rocprofv2` and `omniperf`.
- Classify kernels into **Memory-Bandwidth Bound** vs **Compute Bound** using Roofline analysis ($I = \text{FLOPs}/\text{Byte}$ vs Hardware Ridge Point).
- Enforce cloud credit safety (`auto_shutdown_guard.sh`) and cryptographic provenance (`scripts/log_experiment.py`).

## Key `rocprofv2` Counters & Roofline Workflow

| Hardware Counter / Metric | Subsystem | Diagnostic Meaning |
|---|---|---|
| `SQ_WAVES`, `SQ_INSTS_VALU` | Shader / Compute Unit | Wavefront scheduling & VALU/MFMA instruction throughput |
| `TCC_HIT_sum`, `TCC_MISS_sum` | L2 Cache (TCC) | L2 cache hit ratio and global HBM3 traffic amplification |
| `FETCH_SIZE`, `WRITE_SIZE` | Memory Controller | Total bytes read/written from HBM3 for Arithmetic Intensity |
| `SQ_LDS_BANK_CONFLICT` | Local Data Share | Bank conflicts inside tiled kernels (`+1` stride padding target = 0) |

```bash
# Capture HIP + kernel trace on MI300X
rocprofv2 --hip-trace --kernel-trace -o trace_out ./saxpy_fma

# Run cross-architecture Roofline Analyzer (MI300X vs Radeon 760M vs XDNA 1 NPU)
python ../../02_ACADEMY_CURRICULUM/03_advanced_rdna_hip/ACAD-A02_rocprofv2_roofline/roofline_analyzer.py
```

## Notes & Observations
- **Ridge Point Contrast:** MI300X BF16 (`1307.4 TFLOPS` / `5.3 TB/s`) has a high ridge point of **`246.68 Ops/Byte`**, meaning small-batch LLM decode (`I < 10`) is strictly memory-bandwidth bound while large-batch prefill / dense GEMM (`8192x8192`, `I = 1365.3 Ops/B`) reaches the **Compute Bound** roof.
- **Credit Protection:** Always launch [`auto_shutdown_guard.sh`](../../04_CLOUD_AND_API_LABS/mi300x_dev_cloud/auto_shutdown_guard.sh) during interactive MI300X sessions (`$1.99/hr`, `$100` grant).
