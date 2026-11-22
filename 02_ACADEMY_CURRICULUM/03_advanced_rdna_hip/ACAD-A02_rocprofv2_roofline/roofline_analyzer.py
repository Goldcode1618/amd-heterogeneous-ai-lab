#!/usr/bin/env python3
"""ACAD-A02: Cross-Architecture Roofline Analyzer (`rocprofv2` & Edge NPU/iGPU).

Computes Arithmetic Intensity (I = FLOPs / Byte), hardware ridge points, and
bandwidth-bound vs compute-bound regime classifications across three AMD hardware
tiers:
1. AMD Instinct MI300X (CDNA 3, 5.3 TB/s HBM3, 1307.4 TFLOPS BF16)
2. AMD Radeon 760M (RDNA 3 iGPU, ~89.6 GB/s DDR5-5600 dual-channel, ~8.6 TFLOPS FP16)
3. AMD Ryzen 5 8600G XDNA 1 NPU (Phoenix, ~89.6 GB/s shared, 16 TOPS INT8)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402


@dataclass(frozen=True)
class HardwareRoof:
    name: str
    peak_gops: float      # Peak GFLOP/s or GIOP/s
    peak_bw_gb_s: float   # Peak memory bandwidth in GB/s

    @property
    def ridge_point(self) -> float:
        """Minimum arithmetic intensity (Ops/Byte) needed to saturate compute."""
        return self.peak_gops / self.peak_bw_gb_s

    def attainable_gops(self, arithmetic_intensity: float) -> tuple[float, str]:
        """Return (attainable_GOPS, bottleneck_regime)."""
        bw_bound = arithmetic_intensity * self.peak_bw_gb_s
        if bw_bound < self.peak_gops:
            return bw_bound, "Memory-Bandwidth Bound"
        return self.peak_gops, "Compute Bound"


AMD_HARDWARE_PROFILES: dict[str, HardwareRoof] = {
    "MI300X_BF16": HardwareRoof("AMD Instinct MI300X (CDNA 3 BF16)", 1_307_400.0, 5300.0),
    "MI300X_FP8": HardwareRoof("AMD Instinct MI300X (CDNA 3 FP8)", 2_614_900.0, 5300.0),
    "Radeon_760M_FP16": HardwareRoof("AMD Radeon 760M (RDNA 3 iGPU FP16)", 8_600.0, 89.6),
    "XDNA1_NPU_INT8": HardwareRoof("AMD Ryzen 5 8600G NPU (XDNA 1 INT8)", 16_000.0, 89.6),
}


def analyze_kernel_roofline(
    kernel_name: str,
    flops: float,
    dram_bytes: float,
) -> list[dict[str, Any]]:
    """Evaluate a kernel's arithmetic intensity across all AMD hardware roofs."""
    intensity = flops / dram_bytes if dram_bytes > 0 else 0.0
    rows: list[dict[str, Any]] = []
    for key, roof in AMD_HARDWARE_PROFILES.items():
        attainable, regime = roof.attainable_gops(intensity)
        rows.append({
            "profile": key,
            "device": roof.name,
            "kernel": kernel_name,
            "arithmetic_intensity": round(intensity, 3),
            "ridge_point": round(roof.ridge_point, 2),
            "attainable_gops": round(attainable, 2),
            "regime": regime,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACAD-A02 AMD Cross-Tier Roofline Analyzer")
    parser.add_argument("--kernel", default="GEMM_8192x8192_BF16", help="Kernel label")
    parser.add_argument("--flops", type=float, default=2.0 * (8192 ** 3), help="Total FLOPs/Ops")
    parser.add_argument("--bytes", type=float, default=3.0 * (8192 ** 2) * 2.0, help="DRAM bytes moved")
    parser.add_argument("--no-log", action="store_true", help="Skip MASTER_EXPERIMENT_LOG recording")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    results = analyze_kernel_roofline(args.kernel, args.flops, args.bytes)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"=== ACAD-A02: Roofline Analysis for {args.kernel} (I = {results[0]['arithmetic_intensity']} Ops/B) ===")
    print(f"{'Profile':<20} | {'Ridge (Ops/B)':>14} | {'Attainable GOPS':>16} | {'Regime'}")
    print("-" * 82)
    for r in results:
        print(
            f"{r['profile']:<20} | {r['ridge_point']:>14.2f} | {r['attainable_gops']:>16.2f} | {r['regime']}"
        )

    if not args.no_log:
        mi300x = results[0]
        entry = record_experiment(
            track="academy_ACAD-A02",
            hardware_tier="Cloud_MI300X",
            device_name="AMD Instinct MI300X (Roofline Model)",
            execution_provider="rocprofv2_Roofline_Engine",
            model_or_workload=args.kernel,
            precision="BF16",
            batch_size=1,
            latency_ms_p50=round(elapsed_ms, 3),
            latency_ms_p95=round(elapsed_ms, 3),
            throughput_val=round(mi300x["attainable_gops"] / 1000.0, 2),
            throughput_unit="TFLOPS_ceil",
            status="PASS",
            script_path=__file__,
            notes=(
                f"I={mi300x['arithmetic_intensity']} Ops/B "
                f"ridge={mi300x['ridge_point']} regime={mi300x['regime']}"
            ),
            extra_metadata={"roofline_profiles": results},
        )
        print(f"Logged: experiment_id={entry['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
