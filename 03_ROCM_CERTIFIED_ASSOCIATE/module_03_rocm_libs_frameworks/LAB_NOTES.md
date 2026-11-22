# Module 03: ROCm Libraries, PyTorch, Triton & ONNX Execution Providers — Lab Notes

## Objectives
- Execute PyTorch workloads transparently over ROCm HIP (`torch.cuda.is_available()`, `torch.version.hip`) using `rocBLAS` / `hipBLASLt` and `MIOpen`.
- Compare edge ONNX Runtime execution providers (`VitisAIExecutionProvider` `'target': 'X1'` and `DmlExecutionProvider`) against datacenter `ROCMExecutionProvider` / `MIGraphXExecutionProvider`.
- Author fused memory-bound kernels with **Triton on ROCm** and **FP8/BF16 mixed precision**.

## Key Commands & Reference Scripts
```bash
# Verify PyTorch ROCm HIP stack and run MI300X BF16/FP16 GEMM benchmark
python ../../04_CLOUD_AND_API_LABS/mi300x_dev_cloud/smoke_test_rocm.py

# Cross-tier PyTorch/CPU tensor probe (logs to MASTER_EXPERIMENT_LOG.jsonl)
python ../../02_ACADEMY_CURRICULUM/01_beginner_basics/ACAD-B01_pytorch_rocm/tensor_device_probe.py

# Triton Fused RMSNorm kernel reference & memory traffic reduction benchmark
python ../../02_ACADEMY_CURRICULUM/02_intermediate_systems/ACAD-I05_triton_rocm/fused_rmsnorm_reference.py
```

## Notes & Observations
- **PyTorch HIP Namespace Compatibility:** ROCm PyTorch intentionally preserves `torch.cuda.*` APIs at the Python layer so existing CUDA scripts run unmodified once `torch.version.hip` is active.
- **FP8 E4M3FN vs BF16 (`ACAD-I09`):** MI300X doubles Matrix Core throughput from `1307.4 TFLOPS` (BF16) to `2614.9 TFLOPS` (FP8), making per-tensor dynamic amax scaling essential for outlier-heavy LLM activations.
