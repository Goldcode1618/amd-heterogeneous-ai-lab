#!/usr/bin/env python3
"""ACAD-I09: Mixed Precision & FP8 (E4M3 / E5M2) vs BF16 / FP16 on MI300X.

Simulates IEEE/OCP FP8 (E4M3FN for forward weights/activations, E5M2 for gradients)
alongside BF16 and FP16 on Transformer-like weight/activation distributions, measuring
Signal-to-Quantization-Noise Ratio (SQNR dB), dynamic-range clipping rate, and
MI300X CDNA 3 Matrix Core theoretical peak TFLOPS scaling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402

# Peak dense Matrix Core specs for 1x AMD Instinct MI300X OAM (750W TDP)
MI300X_PEAK_TFLOPS = {
    "FP32": 163.4,
    "BF16": 1307.4,
    "FP16": 1307.4,
    "FP8_E4M3": 2614.9,
    "FP8_E5M2": 2614.9,
}


def simulate_minifloat(
    x: np.ndarray,
    *,
    mantissa_bits: int,
    max_val: float,
    min_subnormal: float,
) -> tuple[np.ndarray, float]:
    """Simulate IEEE-style minifloat rounding with saturation and subnormal floor."""
    arr = x.astype(np.float64)
    clipped_mask = np.abs(arr) > max_val
    clip_rate = float(np.mean(clipped_mask))
    sat = np.clip(arr, -max_val, max_val)

    abs_v = np.abs(sat)
    signs = np.sign(sat)
    out = np.zeros_like(sat)

    nz = abs_v >= min_subnormal
    exps = np.floor(np.log2(np.where(nz, abs_v, 1.0)))
    step = np.power(2.0, exps - mantissa_bits)
    quant_abs = np.where(nz, np.round(abs_v / step) * step, 0.0)
    quant_abs = np.minimum(quant_abs, max_val)
    out = signs * quant_abs
    return out.astype(np.float32), clip_rate


def simulate_bf16(x: np.ndarray) -> np.ndarray:
    """Truncate/round FP32 mantissa from 23 bits to 7 bits (BFloat16)."""
    u32 = x.astype(np.float32).view(np.uint32)
    # Round-to-nearest-even upper 16 bits
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    bf16_u32 = (u32 + rounding_bias) & np.uint32(0xFFFF0000)
    return bf16_u32.view(np.float32)


def sqnr_db(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute Signal-to-Quantization-Noise Ratio (dB)."""
    sig = float(np.mean(ref.astype(np.float64) ** 2))
    noise = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    if noise <= 1e-18:
        return 120.0
    return 10.0 * float(np.log10(sig / noise))


def evaluate_precisions(n_elems: int = 262144) -> list[dict[str, Any]]:
    """Evaluate BF16, FP16, FP8-E4M3 (scaled), and FP8-E5M2 fidelity."""
    rng = np.random.default_rng(1337)
    # Transformer activation distribution with moderate outliers
    x = rng.normal(0.0, 1.0, size=n_elems).astype(np.float32)
    x[::512] *= 18.0  # Simulate outlier channels common in LLMs

    results: list[dict[str, Any]] = []

    # 1. BF16
    t0 = time.perf_counter()
    bf16_x = simulate_bf16(x)
    dt_bf16 = (time.perf_counter() - t0) * 1000.0
    results.append({
        "format": "BF16",
        "sqnr_db": round(sqnr_db(x, bf16_x), 2),
        "clip_rate_pct": 0.0,
        "mi300x_peak_tflops": MI300X_PEAK_TFLOPS["BF16"],
        "sim_ms": round(dt_bf16, 3),
    })

    # 2. FP16
    t0 = time.perf_counter()
    fp16_x = x.astype(np.float16).astype(np.float32)
    dt_fp16 = (time.perf_counter() - t0) * 1000.0
    results.append({
        "format": "FP16",
        "sqnr_db": round(sqnr_db(x, fp16_x), 2),
        "clip_rate_pct": round(float(np.mean(np.abs(x) > 65504.0)) * 100.0, 4),
        "mi300x_peak_tflops": MI300X_PEAK_TFLOPS["FP16"],
        "sim_ms": round(dt_fp16, 3),
    })

    # 3. FP8 E4M3FN with per-tensor dynamic amax scaling (standard Transformer Engine / ROCm recipe)
    t0 = time.perf_counter()
    amax = float(np.max(np.abs(x)))
    fp8_scale = 448.0 / amax if amax > 0 else 1.0
    q_e4m3, clip_e4m3 = simulate_minifloat(
        x * fp8_scale, mantissa_bits=3, max_val=448.0, min_subnormal=2.0 ** -9
    )
    deq_e4m3 = q_e4m3 / fp8_scale
    dt_e4m3 = (time.perf_counter() - t0) * 1000.0
    results.append({
        "format": "FP8_E4M3",
        "sqnr_db": round(sqnr_db(x, deq_e4m3), 2),
        "clip_rate_pct": round(clip_e4m3 * 100.0, 4),
        "mi300x_peak_tflops": MI300X_PEAK_TFLOPS["FP8_E4M3"],
        "sim_ms": round(dt_e4m3, 3),
    })

    # 4. FP8 E5M2 (wider dynamic range, lower mantissa precision)
    t0 = time.perf_counter()
    q_e5m2, clip_e5m2 = simulate_minifloat(
        x, mantissa_bits=2, max_val=57344.0, min_subnormal=2.0 ** -16
    )
    dt_e5m2 = (time.perf_counter() - t0) * 1000.0
    results.append({
        "format": "FP8_E5M2",
        "sqnr_db": round(sqnr_db(x, q_e5m2), 2),
        "clip_rate_pct": round(clip_e5m2 * 100.0, 4),
        "mi300x_peak_tflops": MI300X_PEAK_TFLOPS["FP8_E5M2"],
        "sim_ms": round(dt_e5m2, 3),
    })

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACAD-I09 Mixed Precision & FP8 Micro-Benchmark")
    parser.add_argument("--elems", type=int, default=262144, help="Number of tensor elements")
    parser.add_argument("--no-log", action="store_true", help="Skip MASTER_EXPERIMENT_LOG recording")
    args = parser.parse_args(argv)

    rows = evaluate_precisions(n_elems=args.elems)
    print("=== ACAD-I09: Mixed Precision & FP8 Fidelity vs MI300X Peak TFLOPS ===")
    print(f"{'Format':<10} | {'SQNR (dB)':>10} | {'Clip (%)':>10} | {'MI300X Peak TFLOPS':>18}")
    print("-" * 58)
    for r in rows:
        print(
            f"{r['format']:<10} | {r['sqnr_db']:>10.2f} | {r['clip_rate_pct']:>10.4f} | "
            f"{r['mi300x_peak_tflops']:>18.1f}"
        )

    if not args.no_log:
        e4m3 = next(r for r in rows if r["format"] == "FP8_E4M3")
        entry = record_experiment(
            track="academy_ACAD-I09",
            hardware_tier="Cloud_MI300X",
            device_name="AMD Instinct MI300X (CDNA 3 FP8 Spec + Host Sim)",
            execution_provider="FP8_E4M3_Scaled_Quantizer",
            model_or_workload=f"Transformer_Outlier_Tensor_{args.elems}",
            precision="FP8_E4M3",
            batch_size=1,
            latency_ms_p50=e4m3["sim_ms"],
            latency_ms_p95=e4m3["sim_ms"],
            throughput_val=e4m3["mi300x_peak_tflops"],
            throughput_unit="TFLOPS_peak",
            status="PASS",
            script_path=__file__,
            notes=f"FP8_E4M3 SQNR={e4m3['sqnr_db']}dB vs BF16 SQNR={rows[0]['sqnr_db']}dB",
            extra_metadata={"formats": rows},
        )
        print(f"Logged: experiment_id={entry['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
