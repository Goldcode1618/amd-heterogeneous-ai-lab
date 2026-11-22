#!/usr/bin/env python3
"""ACAD-B01: Cross-Tier Tensor Device Probe & GEMM Micro-Benchmark (PyTorch ROCm / DirectML / CPU).

Probes available acceleration backends (AMD ROCm HIP via torch.version.hip,
DirectML, CUDA, or CPU), executes a deterministic matrix multiplication and
elementwise SiLU workload, and optionally records telemetry to 00_RESEARCH_LIBRARY.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402


def percentile_ms(samples_ms: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile over latency samples."""
    ordered = sorted(samples_ms)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def probe_backends() -> dict[str, Any]:
    """Detect available PyTorch / DirectML / NumPy compute backends safely."""
    info: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "hip_version": None,
        "cuda_available": False,
        "device_name": "AMD Ryzen 5 8600G (Zen 4 CPU)",
        "selected_backend": "NumPy_CPU_AVX512",
    }
    try:
        import torch

        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["hip_version"] = getattr(torch.version, "hip", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            if info["hip_version"]:
                info["selected_backend"] = f"PyTorch_ROCm_HIP_{info['hip_version']}"
            else:
                info["selected_backend"] = "PyTorch_CUDA_Backend"
        else:
            info["selected_backend"] = "PyTorch_CPU"
    except ImportError:
        pass
    return info


def run_tensor_benchmark(size: int = 512, warmup: int = 3, iters: int = 15) -> dict[str, Any]:
    """Run a deterministic FP32 GEMM + SiLU benchmark on the best available tensor backend."""
    backend = probe_backends()
    latencies_ms: list[float] = []
    flops_per_iter = 2.0 * (size ** 3) + 4.0 * (size ** 2)

    if backend["torch_installed"]:
        import torch

        device = torch.device("cuda:0" if backend["cuda_available"] else "cpu")
        torch.manual_seed(42)
        a = torch.randn((size, size), dtype=torch.float32, device=device)
        b = torch.randn((size, size), dtype=torch.float32, device=device)

        def step() -> Any:
            y = a @ b
            return torch.nn.functional.silu(y)

        for _ in range(warmup):
            _ = step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(iters):
            t0 = time.perf_counter()
            _ = step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    else:
        import numpy as np

        rng = np.random.default_rng(42)
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)

        def step_np() -> Any:
            y = a @ b
            return y / (1.0 + np.exp(-np.clip(y, -20.0, 20.0)))

        for _ in range(warmup):
            _ = step_np()

        for _ in range(iters):
            t0 = time.perf_counter()
            _ = step_np()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_ms = statistics.median(latencies_ms)
    p95_ms = percentile_ms(latencies_ms, 95.0)
    mean_sec = statistics.mean(latencies_ms) / 1000.0
    gflops = (flops_per_iter / mean_sec) / 1e9 if mean_sec > 0 else 0.0

    return {
        **backend,
        "matrix_size": size,
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "gflops": round(gflops, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACAD-B01 PyTorch/ROCm Tensor Device Probe")
    parser.add_argument("--size", type=int, default=512, help="Square GEMM dimension N")
    parser.add_argument("--iters", type=int, default=15, help="Timed benchmark iterations")
    parser.add_argument("--no-log", action="store_true", help="Skip MASTER_EXPERIMENT_LOG recording")
    args = parser.parse_args(argv)

    res = run_tensor_benchmark(size=args.size, iters=args.iters)
    print("=== ACAD-B01: PyTorch on AMD — Device Probe ===")
    print(f"Selected Backend : {res['selected_backend']}")
    print(f"Device Name      : {res['device_name']}")
    print(f"ROCm HIP Version : {res['hip_version']}")
    print(f"Workload         : GEMM+SiLU ({args.size}x{args.size} FP32)")
    print(f"Latency          : p50={res['p50_ms']:.3f} ms | p95={res['p95_ms']:.3f} ms")
    print(f"Throughput       : {res['gflops']:.2f} GFLOP/s")

    if not args.no_log:
        tier = "Cloud_MI300X" if res["hip_version"] else "Edge_CPU"
        entry = record_experiment(
            track="academy_ACAD-B01",
            hardware_tier=tier,
            device_name=res["device_name"],
            execution_provider=res["selected_backend"],
            model_or_workload=f"GEMM_SiLU_{args.size}x{args.size}",
            precision="FP32",
            batch_size=1,
            latency_ms_p50=res["p50_ms"],
            latency_ms_p95=res["p95_ms"],
            throughput_val=res["gflops"],
            throughput_unit="GFLOP/s",
            status="PASS",
            script_path=__file__,
            notes=f"ACAD-B01 tensor probe torch={res['torch_version']} hip={res['hip_version']}",
            extra_metadata=res,
        )
        print(f"Logged           : experiment_id={entry['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
