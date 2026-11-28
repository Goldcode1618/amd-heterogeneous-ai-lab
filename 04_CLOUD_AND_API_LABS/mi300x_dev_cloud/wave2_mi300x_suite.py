#!/usr/bin/env python3
"""Wave 2 MI300X (gfx942) hardware scaling, FP8, HBM3 copy, and SDPA benchmark suite.

Designed to execute standalone inside `/work/` (depth-1 container mount) without relying
on `Path(__file__).resolve().parents[3]`. Wraps every sub-benchmark in an independent
`try/except` block and writes results incrementally after each sub-test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_OUT_DIR = Path(__file__).resolve().parent


def utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def percentile_ms(samples_ms: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile over sample latencies in ms."""
    ordered = sorted(samples_ms)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def detect_fp8_dtype(torch_mod: Any) -> tuple[Any | None, str]:
    """Prefer native CDNA 3 (gfx942) float8_e4m3fnuz, falling back to float8_e4m3fn."""
    for attr in ("float8_e4m3fnuz", "float8_e4m3fn"):
        dtype = getattr(torch_mod, attr, None)
        if dtype is not None:
            return dtype, attr
    return None, "UNAVAILABLE"


def append_jsonl_record(jsonl_path: Path, record: dict[str, Any]) -> None:
    """Append a single benchmark record immediately to the JSONL log."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def write_json_summary(
    json_path: Path, metadata: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    """Persist structured JSON summary atomically after each sub-test."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in records if r.get("status") == "PASS")
    failed = sum(1 for r in records if r.get("status") == "ERROR")
    skipped = sum(1 for r in records if r.get("status") == "SKIP")
    payload = {
        "metadata": metadata,
        "summary": {
            "total": len(records),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "results": records,
    }
    tmp_path = json_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(json_path)


def run_subtest(
    name: str,
    fn: Callable[[], dict[str, Any]],
    *,
    jsonl_path: Path,
    json_path: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run one sub-benchmark inside an isolated try/except and persist immediately."""
    t0 = time.perf_counter()
    try:
        metrics = fn()
        status = str(metrics.pop("status", "PASS"))
        record: dict[str, Any] = {
            "subtest": name,
            "status": status,
            "timestamp_utc": utc_now_iso(),
            "elapsed_s": round(time.perf_counter() - t0, 3),
            **metrics,
        }
    except Exception as exc:
        record = {
            "subtest": name,
            "status": "ERROR",
            "timestamp_utc": utc_now_iso(),
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }

    records.append(record)
    append_jsonl_record(jsonl_path, record)
    write_json_summary(json_path, metadata, records)
    print(f"[{record['status']}] {name}: {json.dumps(record)}")
    return record


def _time_cuda_op(torch_mod: Any, op: Callable[[], Any], warmup: int, iters: int) -> list[float]:
    """Warm up and measure synchronized GPU execution latencies in ms."""
    for _ in range(warmup):
        op()
    torch_mod.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        op()
        torch_mod.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms


def run_dense_gemm(
    torch_mod: Any, precision_label: str, dtype: Any, size: int, warmup: int = 3, iters: int = 10
) -> dict[str, Any]:
    """Run square dense GEMM benchmark for FP32/BF16/FP16."""
    device = torch_mod.device("cuda:0")
    torch_mod.cuda.reset_peak_memory_stats(device)
    a = b = None
    try:
        a = torch_mod.randn((size, size), device=device, dtype=dtype)
        b = torch_mod.randn((size, size), device=device, dtype=dtype)
        latencies_ms = _time_cuda_op(torch_mod, lambda: a @ b, warmup=warmup, iters=iters)
        p50_ms = statistics.median(latencies_ms)
        p95_ms = percentile_ms(latencies_ms, 95.0)
        total_sec = sum(latencies_ms) / 1000.0
        tflops = ((2.0 * (size ** 3)) * iters) / total_sec / 1e12
        peak_vram_mb = torch_mod.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {
            "category": "gemm",
            "precision": precision_label,
            "size": size,
            "tflops": round(tflops, 3),
            "latency_ms_p50": round(p50_ms, 3),
            "latency_ms_p95": round(p95_ms, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
    finally:
        del a, b
        torch_mod.cuda.empty_cache()


def run_fp8_scaled_mm(
    torch_mod: Any, size: int, warmup: int = 3, iters: int = 10
) -> dict[str, Any]:
    """Run FP8 GEMM via torch._scaled_mm preferring gfx942 float8_e4m3fnuz."""
    fp8_dtype, fp8_name = detect_fp8_dtype(torch_mod)
    if fp8_dtype is None:
        return {
            "status": "SKIP",
            "category": "gemm_fp8",
            "size": size,
            "fp8_dtype": fp8_name,
            "reason": "Neither float8_e4m3fnuz nor float8_e4m3fn exists in this torch build.",
        }
    if not hasattr(torch_mod, "_scaled_mm"):
        return {
            "status": "SKIP",
            "category": "gemm_fp8",
            "size": size,
            "fp8_dtype": fp8_name,
            "reason": "torch._scaled_mm is unavailable in this torch build.",
        }

    device = torch_mod.device("cuda:0")
    torch_mod.cuda.reset_peak_memory_stats(device)
    a = b_t = scale_a = scale_b = None
    try:
        a = torch_mod.randn((size, size), device=device, dtype=torch_mod.bfloat16).to(fp8_dtype)
        b_t = torch_mod.randn((size, size), device=device, dtype=torch_mod.bfloat16).to(fp8_dtype).t()
        scale_a = torch_mod.tensor(1.0, device=device, dtype=torch_mod.float32)
        scale_b = torch_mod.tensor(1.0, device=device, dtype=torch_mod.float32)

        def _mm() -> Any:
            out = torch_mod._scaled_mm(
                a, b_t, scale_a=scale_a, scale_b=scale_b, out_dtype=torch_mod.bfloat16
            )
            return out[0] if isinstance(out, tuple) else out

        latencies_ms = _time_cuda_op(torch_mod, _mm, warmup=warmup, iters=iters)
        p50_ms = statistics.median(latencies_ms)
        p95_ms = percentile_ms(latencies_ms, 95.0)
        total_sec = sum(latencies_ms) / 1000.0
        tflops = ((2.0 * (size ** 3)) * iters) / total_sec / 1e12
        peak_vram_mb = torch_mod.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {
            "category": "gemm_fp8",
            "precision": fp8_name,
            "size": size,
            "tflops": round(tflops, 3),
            "latency_ms_p50": round(p50_ms, 3),
            "latency_ms_p95": round(p95_ms, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
    finally:
        del a, b_t, scale_a, scale_b
        torch_mod.cuda.empty_cache()


def run_hbm3_copy(torch_mod: Any, size_mb: int, warmup: int = 3, iters: int = 10) -> dict[str, Any]:
    """Measure effective HBM3 device-to-device copy bandwidth (GB/s)."""
    device = torch_mod.device("cuda:0")
    n_bytes = size_mb * 1024 * 1024
    n_elems = n_bytes // 4
    src = dst = None
    try:
        src = torch_mod.ones((n_elems,), device=device, dtype=torch_mod.float32)
        dst = torch_mod.empty((n_elems,), device=device, dtype=torch_mod.float32)
        latencies_ms = _time_cuda_op(torch_mod, lambda: dst.copy_(src), warmup=warmup, iters=iters)
        p50_ms = statistics.median(latencies_ms)
        p95_ms = percentile_ms(latencies_ms, 95.0)
        # Read src + write dst => 2 * n_bytes transferred on HBM3
        bw_gbps = (2.0 * n_bytes) / ((p50_ms / 1000.0) * 1e9)
        return {
            "category": "hbm3_d2d_copy",
            "size_mb": size_mb,
            "bandwidth_gbps": round(bw_gbps, 2),
            "latency_ms_p50": round(p50_ms, 3),
            "latency_ms_p95": round(p95_ms, 3),
        }
    finally:
        del src, dst
        torch_mod.cuda.empty_cache()


def run_sdpa_bf16(
    torch_mod: Any,
    seq_len: int,
    *,
    batch: int = 4,
    heads: int = 32,
    head_dim: int = 128,
    warmup: int = 2,
    iters: int = 6,
) -> dict[str, Any]:
    """Benchmark BF16 causal scaled_dot_product_attention across sequence lengths."""
    device = torch_mod.device("cuda:0")
    torch_mod.cuda.reset_peak_memory_stats(device)
    q = k = v = None
    try:
        shape = (batch, heads, seq_len, head_dim)
        q = torch_mod.randn(shape, device=device, dtype=torch_mod.bfloat16)
        k = torch_mod.randn(shape, device=device, dtype=torch_mod.bfloat16)
        v = torch_mod.randn(shape, device=device, dtype=torch_mod.bfloat16)
        sdpa = torch_mod.nn.functional.scaled_dot_product_attention
        latencies_ms = _time_cuda_op(
            torch_mod, lambda: sdpa(q, k, v, is_causal=True), warmup=warmup, iters=iters
        )
        p50_ms = statistics.median(latencies_ms)
        p95_ms = percentile_ms(latencies_ms, 95.0)
        tokens_per_sec = (batch * seq_len) / (p50_ms / 1000.0)
        # Causal attention FLOPs ~ 2 * B * H * S^2 * D (halved from 4 * B * H * S^2 * D)
        tflops = (2.0 * batch * heads * (seq_len ** 2) * head_dim) / ((p50_ms / 1000.0) * 1e12)
        peak_vram_mb = torch_mod.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {
            "category": "sdpa_bf16",
            "batch": batch,
            "heads": heads,
            "head_dim": head_dim,
            "seq_len": seq_len,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "approx_tflops": round(tflops, 2),
            "latency_ms_p50": round(p50_ms, 3),
            "latency_ms_p95": round(p95_ms, 3),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
    finally:
        del q, k, v
        torch_mod.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wave 2 MI300X Hardware Benchmark Suite")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for wave2_mi300x_suite.jsonl and wave2_mi300x_suite.json",
    )
    parser.add_argument("--gemm-sizes", default="2048,4096,8192,16384")
    parser.add_argument("--fp8-sizes", default="4096,8192,16384")
    parser.add_argument("--copy-sizes-mb", default="256,1024,4096")
    parser.add_argument("--sdpa-seqs", default="512,1024,2048,4096,8192")
    args = parser.parse_args(argv)

    import torch  # Imported inside main so helpers can be unit tested locally without torch

    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm/CUDA GPU visible to PyTorch.")

    props = torch.cuda.get_device_properties(0)
    fp8_dtype, fp8_name = detect_fp8_dtype(torch)
    metadata = {
        "started_utc": utc_now_iso(),
        "device_name": props.name,
        "vram_gib": round(props.total_memory / (1024 ** 3), 2),
        "multiprocessors": props.multi_processor_count,
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "detected_fp8_dtype": fp8_name if fp8_dtype is not None else "UNAVAILABLE",
    }

    out_dir = args.out_dir.resolve()
    jsonl_path = out_dir / "wave2_mi300x_suite.jsonl"
    json_path = out_dir / "wave2_mi300x_suite.json"
    if jsonl_path.exists():
        jsonl_path.unlink()

    records: list[dict[str, Any]] = []
    gemm_sizes = [int(x) for x in args.gemm_sizes.split(",") if x.strip()]
    fp8_sizes = [int(x) for x in args.fp8_sizes.split(",") if x.strip()]
    copy_sizes = [int(x) for x in args.copy_sizes_mb.split(",") if x.strip()]
    sdpa_seqs = [int(x) for x in args.sdpa_seqs.split(",") if x.strip()]

    for label, dt in (("FP32", torch.float32), ("BF16", torch.bfloat16), ("FP16", torch.float16)):
        for n in gemm_sizes:
            run_subtest(
                f"gemm_{label.lower()}_{n}",
                lambda l=label, d=dt, s=n: run_dense_gemm(torch, l, d, s),
                jsonl_path=jsonl_path,
                json_path=json_path,
                metadata=metadata,
                records=records,
            )

    for n in fp8_sizes:
        run_subtest(
            f"gemm_fp8_scaled_mm_{n}",
            lambda s=n: run_fp8_scaled_mm(torch, s),
            jsonl_path=jsonl_path,
            json_path=json_path,
            metadata=metadata,
            records=records,
        )

    for mb in copy_sizes:
        run_subtest(
            f"hbm3_d2d_copy_{mb}mb",
            lambda sz=mb: run_hbm3_copy(torch, sz),
            jsonl_path=jsonl_path,
            json_path=json_path,
            metadata=metadata,
            records=records,
        )

    for seq in sdpa_seqs:
        run_subtest(
            f"sdpa_bf16_seq{seq}",
            lambda s=seq: run_sdpa_bf16(torch, s),
            jsonl_path=jsonl_path,
            json_path=json_path,
            metadata=metadata,
            records=records,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
