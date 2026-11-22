# Module 01: ROCm Orientation & System Topology — Lab Notes

## Objectives
- Understand the ROCm software stack architecture (`HIP`, `rocBLAS`, `MIOpen`, `RCCL`, `hipBLASLt`, `Triton on ROCm`).
- Inspect GPU topology, NUMA affinity, and HBM3/DDR5 telemetry via `amd-smi`, `rocm-smi`, and `rocminfo`.
- Verify reproducible container workflows (`rocm/pytorch:latest`) and host-level WSL2 isolation boundaries.

## Key Commands & Exercises
```bash
# Static hardware & ISA inspection
amd-smi static
rocm-smi --showproductname --showmeminfo vram --showtopo
rocminfo | grep -E "Name:|Marketing Name:|gfx|Wavefront"

# Verify container runtime (Linux bare-metal / MI300X VM)
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  rocm/pytorch:latest python3 -c "import torch; print(torch.__version__, torch.version.hip)"
```

## Engineering Notes & Observations (`GOLDENRATIO1618` vs MI300X)
- **Host Isolation (`GOLDENRATIO1618`):** Windows 11 Pro host preserves NVIDIA RTX 3050 CUDA drivers and Ryzen AI `1.9.0` (`XDNA 1`) in separate environments; WSL2 does not expose `/dev/accel` for Linux XRT `amdxdna`, so bare-metal Ubuntu 24.04 is required for Linux NPU tests, while `rocm/pytorch` containers run cleanly on AMD Developer Cloud MI300X VMs.
- **Target ISA Identifiers:**
  - `gfx1103` — AMD Radeon 760M (RDNA 3 iGPU, Wavefront32 native).
  - `gfx942` — AMD Instinct MI300X (CDNA 3, 304 CUs, 192 GB HBM3 @ 5.3 TB/s, Wavefront64).
- **Automated Readiness Check:** [`check_rocm_lab_readiness.ps1`](../check_rocm_lab_readiness.ps1) validates Docker daemon, WSL2 kernel, and local GPU isolation prior to cloud provisioning.
