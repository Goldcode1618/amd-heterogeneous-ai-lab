"""Head-to-head ONNX Runtime benchmark: XDNA 1 NPU vs Radeon 760M DirectML vs Zen 4 CPU.

Run inside the isolated Ryzen AI 1.9.0 environment:
  C:\\Program Files\\RyzenAI\\ryzen-ai-1.9.0\\Scripts\\activate.bat
  python benchmark_directml_vs_npu.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402


def resolve_test_model() -> Path:
    """Locate Ryzen AI quicktest ONNX model."""
    candidates = [
        Path(os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")) / "quicktest" / "test_model.onnx",
        Path(r"C:\Program Files\RyzenAI\1.9.0\1.9.0\quicktest\test_model.onnx"),
        Path(r"C:\Program Files\RyzenAI\1.7.1\quicktest\test_model.onnx"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not locate Ryzen AI quicktest/test_model.onnx")


def resolve_phoenix_xclbin() -> Path:
    """Locate Phoenix/Hawk Point 4x4.xclbin overlay."""
    candidates = [
        Path(os.environ.get("RYZEN_AI_INSTALLATION_PATH", ""))
        / "voe-4.0-win_amd64"
        / "xclbins"
        / "phoenix"
        / "4x4.xclbin",
        Path(r"C:\Program Files\RyzenAI\1.9.0\1.9.0\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin"),
        Path(r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not locate Phoenix 4x4.xclbin for XDNA 1 NPU")


def percentile_ms(samples_ms: list[float], pct: float) -> float:
    """Compute linear-interpolated percentile from sample latencies."""
    ordered = sorted(samples_ms)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def benchmark_provider(
    ort,
    np,
    model_path: Path,
    *,
    hardware_tier: str,
    device_name: str,
    provider_name: str,
    provider_options: list[dict] | None,
    precision: str,
    warmup: int = 5,
    runs: int = 30,
) -> dict:
    """Create an InferenceSession for a single provider, benchmark latency, and record results."""
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3

    kwargs = {"sess_options": sess_opts, "providers": [provider_name]}
    if provider_options is not None:
        kwargs["provider_options"] = provider_options

    t_init_0 = time.perf_counter()
    session = ort.InferenceSession(str(model_path), **kwargs)
    init_ms = (time.perf_counter() - t_init_0) * 1000.0

    input_meta = session.get_inputs()[0]
    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
    rng = np.random.default_rng(42)
    dummy_input = rng.random(shape, dtype=np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_meta.name: dummy_input})

    latencies_ms: list[float] = []
    output_shape = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = session.run(None, {input_meta.name: dummy_input})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        output_shape = out[0].shape

    p50_ms = statistics.median(latencies_ms)
    p95_ms = percentile_ms(latencies_ms, 95.0)
    mean_ms = statistics.mean(latencies_ms)
    ips = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    logged = record_experiment(
        track="01_local_edge_head_to_head",
        hardware_tier=hardware_tier,
        device_name=device_name,
        execution_provider=provider_name,
        model_or_workload=model_path.name,
        precision=precision,
        batch_size=1,
        latency_ms_p50=p50_ms,
        latency_ms_p95=p95_ms,
        throughput_val=ips,
        throughput_unit="inf/s",
        status="PASS",
        script_path=__file__,
        notes=f"warmup={warmup} runs={runs} init_ms={init_ms:.1f} out={output_shape} ort={ort.__version__}",
        extra_metadata={"mean_ms": round(mean_ms, 3), "init_ms": round(init_ms, 2)},
    )

    return {
        "provider": provider_name,
        "tier": hardware_tier,
        "device": device_name,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "mean_ms": mean_ms,
        "ips": ips,
        "init_ms": init_ms,
        "experiment_id": logged["experiment_id"],
    }


def main() -> int:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(
            f"ERROR: Missing dependency ({exc}). Run inside Ryzen AI 1.9.0 environment.",
            file=sys.stderr,
        )
        return 1

    model_path = resolve_test_model()
    xclbin_path = resolve_phoenix_xclbin()
    available = ort.get_available_providers()
    print(f"ONNX Runtime : {ort.__version__}")
    print(f"Available EPs: {available}")
    print(f"Model        : {model_path}")
    print(f"NPU xclbin   : {xclbin_path}\n")

    specs = [
        {
            "hardware_tier": "Edge_NPU",
            "device_name": "AMD Ryzen AI NPU Phoenix (XDNA 1)",
            "provider_name": "VitisAIExecutionProvider",
            "provider_options": [
                {
                    "target": "X1",
                    "xlnx_enable_py3_round": 0,
                    "xclbin": str(xclbin_path),
                }
            ],
            "precision": "INT8/AIE2_4x4",
        },
        {
            "hardware_tier": "Edge_iGPU",
            "device_name": "AMD Radeon 760M (RDNA 3 iGPU)",
            "provider_name": "DmlExecutionProvider",
            "provider_options": None,
            "precision": "FP32/DirectML",
        },
        {
            "hardware_tier": "Edge_CPU",
            "device_name": "AMD Ryzen 5 8600G (Zen 4 CPU)",
            "provider_name": "CPUExecutionProvider",
            "provider_options": None,
            "precision": "FP32/AVX512",
        },
    ]

    results: list[dict] = []
    for spec in specs:
        pname = spec["provider_name"]
        if pname not in available:
            print(f"SKIP: {pname} not available in this ONNX Runtime build.")
            continue
        print(f"Running {pname} ({spec['device_name']}) ...")
        try:
            res = benchmark_provider(ort, np, model_path, **spec)
            results.append(res)
            print(
                f"  -> p50={res['p50_ms']:.3f} ms | p95={res['p95_ms']:.3f} ms | "
                f"throughput={res['ips']:.2f} inf/s | init={res['init_ms']:.1f} ms"
            )
        except Exception as exc:
            print(f"  -> FAILED: {exc}", file=sys.stderr)
            record_experiment(
                track="01_local_edge_head_to_head",
                hardware_tier=spec["hardware_tier"],
                device_name=spec["device_name"],
                execution_provider=pname,
                model_or_workload=model_path.name,
                precision=spec["precision"],
                status="FAIL",
                script_path=__file__,
                notes=f"Error: {exc}",
            )
            return 2

    print("\n=== Head-to-Head Summary ===")
    print(f"{'Tier':<12} | {'Provider':<25} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'inf/s':>10}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['tier']:<12} | {r['provider']:<25} | {r['p50_ms']:>10.3f} | "
            f"{r['p95_ms']:>10.3f} | {r['ips']:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
