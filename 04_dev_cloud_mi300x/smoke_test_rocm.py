"""Smoke test & quick BF16/FP16 GEMM benchmark for AMD Developer Cloud MI300X (ROCm)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time


def run_smi_check() -> None:
    """Print concise rocm-smi or amd-smi summary if available."""
    for tool in ("rocm-smi", "amd-smi"):
        exe = shutil.which(tool)
        if exe:
            print(f"=== Telemetry ({tool}) ===")
            args = [exe, "--showproductname", "--showmeminfo", "vram"] if tool == "rocm-smi" else [exe, "metric", "-m"]
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
            print(proc.stdout.strip() or proc.stderr.strip())
            return
    print("=== Telemetry ===\nrocm-smi / amd-smi not found in PATH.")


def benchmark_matmul(torch_mod, dtype, size: int = 8192, warmup: int = 3, iters: int = 10) -> float:
    """Run square GEMM benchmark and return estimated TFLOPS."""
    device = torch_mod.device("cuda:0")
    a = torch_mod.randn((size, size), device=device, dtype=dtype)
    b = torch_mod.randn((size, size), device=device, dtype=dtype)

    for _ in range(warmup):
        _ = a @ b
    torch_mod.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch_mod.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    flops_per_iter = 2.0 * (size ** 3)
    tflops = (flops_per_iter * iters) / elapsed / 1e12
    return tflops


def main() -> int:
    run_smi_check()

    try:
        import torch
    except ImportError as exc:
        print(f"ERROR: PyTorch is not installed ({exc}).", file=sys.stderr)
        return 1

    print("\n=== PyTorch ROCm Status ===")
    print(f"torch.__version__      : {torch.__version__}")
    print(f"torch.version.hip      : {getattr(torch.version, 'hip', None)}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: No ROCm/CUDA GPU device visible to PyTorch.", file=sys.stderr)
        return 2

    count = torch.cuda.device_count()
    print(f"Visible GPU count      : {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        vram_gib = props.total_memory / (1024 ** 3)
        print(f"  Device {i}: {props.name} | Total VRAM: {vram_gib:.2f} GiB | MultiProcessors: {props.multi_processor_count}")

    print("\n=== GEMM Inference / Compute Benchmark (8192 x 8192) ===")
    for label, dt in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        try:
            tflops = benchmark_matmul(torch, dt)
            print(f"  {label:8s}: {tflops:8.2f} TFLOPS")
        except Exception as exc:
            print(f"  {label:8s}: FAILED ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
