# Module 01: ROCm Fundamentals — Lab Notes

## Objectives
- Understand ROCm software stack architecture (HIP, rocBLAS, MIOpen, RCCL, Triton on ROCm).
- Inspect GPU topology and telemetry (`rocm-smi`, `amd-smi`, `rocminfo`).
- Verify container runtime workflow (`rocm/pytorch` Docker image).

## Key Commands & Exercises
```bash
amd-smi static
rocm-smi --showtopo
rocminfo | grep -E "Name:|Marketing Name:|gfx"
```

## Notes & Observations
- _Record lab outputs and key takeaways here._
