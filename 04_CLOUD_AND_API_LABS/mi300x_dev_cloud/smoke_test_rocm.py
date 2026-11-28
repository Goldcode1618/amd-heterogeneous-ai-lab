"""Smoke test & BF16/FP16 GEMM benchmark for AMD Developer Cloud MI300X (ROCm).

Automatically logs results via scripts/log_experiment.py when executed inside
the cloned workspace, or writes a standalone JSONL fallback file alongside the
script when copied directly to an ephemeral MI300X VM.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_resolved_parents = Path(__file__).resolve().parents
WORKSPACE_ROOT = (
    _resolved_parents[2] if len(_resolved_parents) > 2 else _resolved_parents[-1]
)
if (WORKSPACE_ROOT / "scripts" / "log_experiment.py").is_file():
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    from scripts.log_experiment import record_experiment
else:
    record_experiment = None


def run_smi_check() -> None:
    """Print concise rocm-smi / amd-smi and rocminfo summaries if available."""
    found_smi = False
    for tool in ("rocm-smi", "amd-smi"):
        exe = shutil.which(tool)
        if exe:
            print(f"=== Telemetry ({tool}) ===")
            args = (
                [exe, "--showproductname", "--showmeminfo", "vram"]
                if tool == "rocm-smi"
                else [exe, "metric", "-m"]
            )
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
            print(proc.stdout.strip() or proc.stderr.strip())
            found_smi = True
            break
    if not found_smi:
        print("=== Telemetry ===\nrocm-smi / amd-smi not found in PATH.")

    rocminfo_exe = shutil.which("rocminfo")
    if rocminfo_exe:
        proc = subprocess.run([rocminfo_exe], capture_output=True, text=True, check=False)
        agent_lines = [
            line.strip()
            for line in proc.stdout.splitlines()
            if "Name:" in line or "Marketing Name:" in line
        ][:12]
        if agent_lines:
            print("=== ROCm Agents (rocminfo) ===")
            print("\n".join(agent_lines))


def percentile_ms(samples_ms: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile over sample latencies."""
    ordered = sorted(samples_ms)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def benchmark_matmul(
    torch_mod, dtype, size: int = 8192, warmup: int = 3, iters: int = 10
) -> tuple[float, float, float]:
    """Run square GEMM benchmark and return (tflops, p50_ms, p95_ms)."""
    device = torch_mod.device("cuda:0")
    a = torch_mod.randn((size, size), device=device, dtype=dtype)
    b = torch_mod.randn((size, size), device=device, dtype=dtype)

    for _ in range(warmup):
        _ = a @ b
    torch_mod.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = a @ b
        torch_mod.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_sec = sum(latencies_ms) / 1000.0
    flops_per_iter = 2.0 * (size ** 3)
    tflops = (flops_per_iter * iters) / total_sec / 1e12
    return tflops, statistics.median(latencies_ms), percentile_ms(latencies_ms, 95.0)


def log_result(
    *,
    device_name: str,
    precision: str,
    tflops: float,
    p50_ms: float,
    p95_ms: float,
    vram_mb: float,
    hip_version: str | None,
) -> None:
    """Persist benchmark run to 00_RESEARCH_LIBRARY or standalone local JSONL."""
    if record_experiment is not None:
        entry = record_experiment(
            track="04_cloud_mi300x_gemm",
            hardware_tier="Cloud_MI300X",
            device_name=device_name,
            execution_provider=f"PyTorch_ROCm_HIP_{hip_version or 'unknown'}",
            model_or_workload="GEMM_8192x8192",
            precision=precision,
            batch_size=1,
            latency_ms_p50=p50_ms,
            latency_ms_p95=p95_ms,
            throughput_val=tflops,
            throughput_unit="TFLOPS",
            memory_mb=vram_mb,
            status="PASS",
            script_path=__file__,
            notes="AMD Developer Cloud MI300X smoke benchmark",
        )
        print(f"    Logged -> experiment_id={entry['experiment_id']}")
    else:
        fallback = Path(__file__).resolve().parent / "mi300x_remote_results.jsonl"
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device_name": device_name,
            "precision": precision,
            "tflops": round(tflops, 3),
            "latency_ms_p50": round(p50_ms, 3),
            "latency_ms_p95": round(p95_ms, 3),
            "vram_mb": round(vram_mb, 2),
            "hip_version": hip_version,
        }
        with fallback.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload) + "\n")
        print(f"    Saved standalone result -> {fallback}")


def main() -> int:
    run_smi_check()

    try:
        import torch
    except ImportError as exc:
        print(f"ERROR: PyTorch is not installed ({exc}).", file=sys.stderr)
        return 1

    hip_version = getattr(torch.version, "hip", None)
    print("\n=== PyTorch ROCm Status ===")
    print(f"torch.__version__      : {torch.__version__}")
    print(f"torch.version.hip      : {hip_version}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: No ROCm/CUDA GPU device visible to PyTorch.", file=sys.stderr)
        return 2

    count = torch.cuda.device_count()
    print(f"Visible GPU count      : {count}")
    primary_name = "AMD Instinct MI300X"
    primary_vram_mb = 0.0
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        vram_gib = props.total_memory / (1024 ** 3)
        if i == 0:
            primary_name = props.name
            primary_vram_mb = props.total_memory / (1024 ** 2)
        print(
            f"  Device {i}: {props.name} | Total VRAM: {vram_gib:.2f} GiB | "
            f"MultiProcessors: {props.multi_processor_count}"
        )

    print("\n=== GEMM Inference / Compute Benchmark (8192 x 8192) ===")
    for label, dt in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        try:
            tflops, p50_ms, p95_ms = benchmark_matmul(torch, dt)
            print(
                f"  {label:8s}: {tflops:8.2f} TFLOPS | "
                f"p50={p50_ms:.3f} ms | p95={p95_ms:.3f} ms"
            )
            log_result(
                device_name=primary_name,
                precision=label.upper(),
                tflops=tflops,
                p50_ms=p50_ms,
                p95_ms=p95_ms,
                vram_mb=primary_vram_mb,
                hip_version=hip_version,
            )
        except Exception as exc:
            print(f"  {label:8s}: FAILED ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
