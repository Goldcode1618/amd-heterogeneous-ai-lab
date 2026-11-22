#!/usr/bin/env python3
"""ACAD-I01: Automated CUDA-to-HIP Source Porting & Compatibility Auditor.

Demonstrates deterministic CUDA-to-HIP API translation rules (mirroring AMD
`hipify-perl` / `hipify-clang`), converts CUDA runtime/library calls and triple-angle
kernel launches into portable HIP syntax, validates translation invariants, and
optionally logs verification metrics to 00_RESEARCH_LIBRARY.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402

CUDA_TO_HIP_MAP: list[tuple[str, str]] = [
    ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
    ("#include <cublas_v2.h>", "#include <rocblas/rocblas.h>"),
    ("cudaError_t", "hipError_t"),
    ("cudaSuccess", "hipSuccess"),
    ("cudaGetErrorString", "hipGetErrorString"),
    ("cudaMalloc", "hipMalloc"),
    ("cudaFree", "hipFree"),
    ("cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
    ("cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
    ("cudaMemcpy", "hipMemcpy"),
    ("cudaEvent_t", "hipEvent_t"),
    ("cudaEventCreate", "hipEventCreate"),
    ("cudaEventRecord", "hipEventRecord"),
    ("cudaEventSynchronize", "hipEventSynchronize"),
    ("cudaEventElapsedTime", "hipEventElapsedTime"),
    ("cudaEventDestroy", "hipEventDestroy"),
    ("cudaDeviceSynchronize", "hipDeviceSynchronize"),
    ("cublasHandle_t", "rocblas_handle"),
    ("cublasCreate", "rocblas_create_handle"),
    ("cublasSgemm", "rocblas_sgemm"),
]

SAMPLE_CUDA_SOURCE = """#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void vector_scale(int n, float s, float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

int launch_scale(int n, float* h_x) {
    float* d_x = nullptr;
    cudaError_t err = cudaMalloc(&d_x, n * sizeof(float));
    if (err != cudaSuccess) return 1;
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    vector_scale<<<(n + 255) / 256, 256>>>(n, 2.0f, d_x);
    cudaDeviceSynchronize();
    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    return 0;
}
"""

KERNEL_LAUNCH_RE = re.compile(
    r"([A-Za-z_]\w*)\s*<<<\s*([^,>]+?)\s*,\s*([^>]+?)\s*>>>\s*\((.*?)\)\s*;",
    re.DOTALL,
)


def translate_cuda_to_hip(cuda_code: str) -> tuple[str, dict[str, int]]:
    """Translate a CUDA C++ snippet to portable AMD HIP C++ and return conversion counts."""
    hip_code = cuda_code
    counts: dict[str, int] = {"api_replacements": 0, "kernel_launches": 0}

    for cuda_tok, hip_tok in CUDA_TO_HIP_MAP:
        occurrences = hip_code.count(cuda_tok)
        if occurrences > 0:
            hip_code = hip_code.replace(cuda_tok, hip_tok)
            counts["api_replacements"] += occurrences

    def _replace_launch(match: re.Match[str]) -> str:
        counts["kernel_launches"] += 1
        kernel, grid, block, args = (g.strip() for g in match.groups())
        arg_suffix = f", {args}" if args else ""
        return f"hipLaunchKernelGGL({kernel}, dim3({grid}), dim3({block}), 0, 0{arg_suffix});"

    hip_code = KERNEL_LAUNCH_RE.sub(_replace_launch, hip_code)
    return hip_code, counts


def audit_portability(hip_code: str) -> dict[str, Any]:
    """Verify zero leftover CUDA tokens remain after source-to-source translation."""
    leftover_cuda = re.findall(r"\bcuda[A-Z]\w*|\bcublas[A-Z]\w*|<<<|>>>", hip_code)
    return {
        "clean_port": len(leftover_cuda) == 0,
        "leftover_tokens": leftover_cuda,
        "has_hip_header": "#include <hip/hip_runtime.h>" in hip_code,
        "has_hip_launch": "hipLaunchKernelGGL(" in hip_code,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACAD-I01 CUDA-to-HIP Porting Auditor")
    parser.add_argument("--no-log", action="store_true", help="Skip MASTER_EXPERIMENT_LOG recording")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    hip_code, counts = translate_cuda_to_hip(SAMPLE_CUDA_SOURCE)
    audit = audit_portability(hip_code)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print("=== ACAD-I01: CUDA-to-HIP Migration & Portability Auditor ===")
    print(f"API Replacements : {counts['api_replacements']}")
    print(f"Kernel Launches  : {counts['kernel_launches']}")
    print(f"Clean HIP Port   : {audit['clean_port']} (leftovers={audit['leftover_tokens']})")
    print(f"Audit Latency    : {elapsed_ms:.3f} ms")

    if not audit["clean_port"]:
        return 2

    if not args.no_log:
        entry = record_experiment(
            track="academy_ACAD-I01",
            hardware_tier="Edge_CPU",
            device_name="AMD Ryzen 5 8600G (Zen 4 Host)",
            execution_provider="HIPIFY_Static_Analyzer",
            model_or_workload="CUDA_to_HIP_SAXPY_Port",
            precision="Source_AST",
            batch_size=1,
            latency_ms_p50=elapsed_ms,
            latency_ms_p95=elapsed_ms,
            throughput_val=float(counts["api_replacements"] + counts["kernel_launches"]),
            throughput_unit="conversions",
            status="PASS",
            script_path=__file__,
            notes=(
                f"api_replacements={counts['api_replacements']} "
                f"kernel_launches={counts['kernel_launches']} clean_port=True"
            ),
            extra_metadata={**counts, **audit},
        )
        print(f"Logged           : experiment_id={entry['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
