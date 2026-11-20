# Module 03: PyTorch & ONNX on ROCm — Lab Notes

## Objectives
- Run PyTorch workloads transparently using ROCm HIP backend (`torch.cuda.is_available()`, `torch.version.hip`).
- Configure `ROCMExecutionProvider` / `MIGraphXExecutionProvider` in ONNX Runtime.

## Key Commands & Exercises
```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
```

## Notes & Observations
- _Record PyTorch training/inference benchmarks and mixed precision notes here._
