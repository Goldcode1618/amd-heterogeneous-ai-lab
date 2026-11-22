#!/usr/bin/env python3
"""ACAD-B02: Model Quantization Fundamentals (FP32 -> Symmetric/Asymmetric INT8 QDQ).

Demonstrates Quantize-Dequantize (QDQ) scale/zero-point calibration required for
AMD Ryzen AI XDNA 1 NPU (AIE2 INT8/UINT8 execution) and evaluates Signal-to-Quantization-Noise
Ratio (SQNR dB), RMSE, cosine similarity, and 4x memory footprint compression.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402


def quantize_symmetric_int8(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize FP32 weights symmetrically into [-127, 127] INT8."""
    max_abs = float(np.max(np.abs(tensor)))
    scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
    q = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_symmetric_int8(q: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize symmetric INT8 tensor back to FP32."""
    return q.astype(np.float32) * np.float32(scale)


def quantize_asymmetric_uint8(tensor: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Quantize skewed activations (e.g., ReLU/SiLU) asymmetrically into [0, 255] UINT8."""
    t_min = float(np.min(tensor))
    t_max = float(np.max(tensor))
    scale = (t_max - t_min) / 255.0 if (t_max - t_min) > 1e-12 else 1.0
    zero_point = int(np.clip(round(-t_min / scale), 0, 255))
    q = np.clip(np.round(tensor / scale) + zero_point, 0, 255).astype(np.uint8)
    return q, scale, zero_point


def dequantize_asymmetric_uint8(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize asymmetric UINT8 tensor back to FP32."""
    return (q.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


def compute_sqnr_db(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in decibels."""
    signal_power = float(np.mean(reference.astype(np.float64) ** 2))
    noise_power = float(np.mean((reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if noise_power <= 1e-18:
        return 120.0
    return 10.0 * float(np.log10(signal_power / noise_power))


def evaluate_qdq_pipeline(dim: int = 1024, runs: int = 20) -> dict[str, Any]:
    """Benchmark QDQ calibration accuracy and INT8 GEMM vs FP32 memory footprint."""
    rng = np.random.default_rng(1618)
    weights_fp32 = rng.normal(0.0, 0.08, size=(dim, dim)).astype(np.float32)
    # Asymmetric post-GELU/SiLU activation distribution
    raw_act = rng.normal(0.1, 0.7, size=(64, dim)).astype(np.float32)
    act_fp32 = np.maximum(raw_act, -0.17)

    latencies_ms: list[float] = []
    q_w, w_scale = quantize_symmetric_int8(weights_fp32)
    q_a, a_scale, a_zp = quantize_asymmetric_uint8(act_fp32)

    for _ in range(runs):
        t0 = time.perf_counter()
        q_w, w_scale = quantize_symmetric_int8(weights_fp32)
        q_a, a_scale, a_zp = quantize_asymmetric_uint8(act_fp32)
        _ = dequantize_symmetric_int8(q_w, w_scale)
        _ = dequantize_asymmetric_uint8(q_a, a_scale, a_zp)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    w_hat = dequantize_symmetric_int8(q_w, w_scale)
    a_hat = dequantize_asymmetric_uint8(q_a, a_scale, a_zp)

    fp32_out = act_fp32 @ weights_fp32
    qdq_out = a_hat @ w_hat

    sqnr_w = compute_sqnr_db(weights_fp32, w_hat)
    sqnr_a = compute_sqnr_db(act_fp32, a_hat)
    sqnr_out = compute_sqnr_db(fp32_out, qdq_out)
    rmse_out = float(np.sqrt(np.mean((fp32_out - qdq_out) ** 2)))
    cos_sim = float(
        np.sum(fp32_out * qdq_out)
        / (np.linalg.norm(fp32_out) * np.linalg.norm(qdq_out) + 1e-12)
    )

    fp32_bytes = weights_fp32.nbytes + act_fp32.nbytes
    int8_bytes = q_w.nbytes + q_a.nbytes
    compression_ratio = fp32_bytes / int8_bytes

    p50_ms = statistics.median(latencies_ms)
    p95_ms = sorted(latencies_ms)[min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))]
    ops_per_sec = 1000.0 / statistics.mean(latencies_ms)

    return {
        "dim": dim,
        "weight_sqnr_db": round(sqnr_w, 2),
        "activation_sqnr_db": round(sqnr_a, 2),
        "matmul_output_sqnr_db": round(sqnr_out, 2),
        "output_rmse": round(rmse_out, 5),
        "cosine_similarity": round(cos_sim, 6),
        "compression_ratio": round(compression_ratio, 2),
        "fp32_mb": round(fp32_bytes / (1024 ** 2), 3),
        "int8_mb": round(int8_bytes / (1024 ** 2), 3),
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "throughput_ops_s": round(ops_per_sec, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACAD-B02 INT8 QDQ Quantization Reference Lab")
    parser.add_argument("--dim", type=int, default=1024, help="Weight matrix square dimension")
    parser.add_argument("--runs", type=int, default=20, help="Calibration timing runs")
    parser.add_argument("--no-log", action="store_true", help="Skip MASTER_EXPERIMENT_LOG recording")
    args = parser.parse_args(argv)

    res = evaluate_qdq_pipeline(dim=args.dim, runs=args.runs)
    print("=== ACAD-B02: Model Quantization Fundamentals (INT8 QDQ) ===")
    print(f"Weight Symmetric INT8 SQNR : {res['weight_sqnr_db']} dB")
    print(f"Act Asymmetric UINT8 SQNR  : {res['activation_sqnr_db']} dB")
    print(f"End-to-End GEMM Output SQNR: {res['matmul_output_sqnr_db']} dB (cos_sim={res['cosine_similarity']})")
    print(f"Memory Footprint           : {res['fp32_mb']} MB (FP32) -> {res['int8_mb']} MB (INT8) [{res['compression_ratio']}x]")
    print(f"QDQ Calibration Latency    : p50={res['p50_ms']} ms | p95={res['p95_ms']} ms")

    if not args.no_log:
        entry = record_experiment(
            track="academy_ACAD-B02",
            hardware_tier="Edge_CPU",
            device_name="AMD Ryzen 5 8600G (Zen 4 CPU)",
            execution_provider="NumPy_QDQ_INT8_Reference",
            model_or_workload=f"QDQ_MatMul_{args.dim}x{args.dim}",
            precision="INT8_QDQ",
            batch_size=64,
            latency_ms_p50=res["p50_ms"],
            latency_ms_p95=res["p95_ms"],
            throughput_val=res["throughput_ops_s"],
            throughput_unit="calibrations/s",
            memory_mb=res["int8_mb"],
            status="PASS",
            script_path=__file__,
            notes=(
                f"out_sqnr={res['matmul_output_sqnr_db']}dB "
                f"cos_sim={res['cosine_similarity']} ratio={res['compression_ratio']}x"
            ),
            extra_metadata=res,
        )
        print(f"Logged                     : experiment_id={entry['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
