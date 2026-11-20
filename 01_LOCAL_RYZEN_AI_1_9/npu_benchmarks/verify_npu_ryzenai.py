"""Diagnostic & latency verification for AMD Ryzen AI NPU (Hawk Point / Ryzen 5 8600G).

Run inside the isolated Ryzen AI 1.9.0 virtual environment:
  C:\\Program Files\\RyzenAI\\ryzen-ai-1.9.0\\Scripts\\activate.bat
  python verify_npu_ryzenai.py
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.log_experiment import record_experiment  # noqa: E402

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0


def run_hidden(cmd: list[str]) -> tuple[int, str, str]:
    """Execute a CLI utility without spawning visible console windows on Windows."""
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=NO_WINDOW,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def detect_hawk_point_npu() -> tuple[str, str]:
    """Detect NPU PCI device type via pnputil."""
    rc, stdout, _ = run_hidden(["pnputil", "/enum-devices", "/bus", "PCI", "/deviceids"])
    if rc != 0:
        return "UNKNOWN", ""
    if "PCI\\VEN_1022&DEV_1502" in stdout:
        return "HPT", "PCI\\VEN_1022&DEV_1502 (Hawk Point / Phoenix XDNA 1)"
    if "PCI\\VEN_1022&DEV_17F0" in stdout:
        return "STX", "PCI\\VEN_1022&DEV_17F0 (Strix / Krackan XDNA 2)"
    return "UNKNOWN", ""


def find_ryzen_ai_root() -> Path | None:
    """Resolve active Ryzen AI installation root, preferring env var or 1.9.0."""
    env_path = os.environ.get("RYZEN_AI_INSTALLATION_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path(r"C:\Program Files\RyzenAI\1.9.0\1.9.0"),
            Path(r"C:\Program Files\RyzenAI\1.9.0"),
            Path(r"C:\Program Files\RyzenAI\ryzen-ai-1.9.0"),
            Path(r"C:\Program Files\RyzenAI\1.7.1"),
        ]
    )
    for path in candidates:
        if path.exists() and (path / "voe-4.0-win_amd64").exists():
            return path
    for path in candidates:
        if path.exists():
            return path
    return None


def find_hawk_point_xclbin(install_dir: Path) -> Path | None:
    """Locate the Phoenix/Hawk Point xclbin inside the Ryzen AI installation."""
    preferred = [
        install_dir / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin",
        install_dir / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "1x4.xclbin",
        Path(r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin"),
        Path(r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\xclbins\phoenix\1x4.xclbin"),
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    for match in install_dir.rglob("*.xclbin"):
        if "phoenix" in str(match).lower() or "4x4" in match.name.lower():
            return match
    return None


def check_xrt_smi() -> None:
    """Invoke xrt-smi examine from PATH or default System32\\AMD location."""
    exe = "xrt-smi"
    fallback = Path(r"C:\Windows\System32\AMD\xrt-smi.exe")
    if fallback.exists():
        exe = str(fallback)
    rc, stdout, stderr = run_hidden([exe, "examine"])
    print("=== [1] xrt-smi examine ===")
    print(stdout.strip() or stderr.strip())
    if rc != 0:
        print(f"WARNING: xrt-smi exited with code {rc}")


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


def main() -> int:
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.splitlines()[0]}")
    check_xrt_smi()

    npu_arch, npu_desc = detect_hawk_point_npu()
    print("\n=== [2] NPU PCIe Architecture ===")
    print(f"Detected NPU Arch : {npu_arch} ({npu_desc or 'not matched'})")

    install_dir = find_ryzen_ai_root()
    print(f"Ryzen AI Root     : {install_dir or 'NOT FOUND'}")
    xclbin_file = find_hawk_point_xclbin(install_dir) if install_dir else None
    vaip_config = (install_dir / "voe-4.0-win_amd64" / "vaip_config.json") if install_dir else None
    phoenix_overlay_ref = (
        (install_dir / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin")
        if install_dir
        else None
    )
    if xclbin_file:
        print(f"Target xclbin     : {xclbin_file}")
    elif vaip_config and vaip_config.exists():
        print(f"Target xclbin     : Dynamic Dispatch / AMD_AIE2_Nx4_Overlay ({phoenix_overlay_ref})")
    else:
        print("Target xclbin     : NOT FOUND")

    print("\n=== [3] ONNX Runtime Providers ===")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        print(
            f"ERROR: Missing dependency ({exc}).\n"
            "Activate the isolated Ryzen AI 1.9.0 environment first:\n"
            r'  "C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"',
            file=sys.stderr,
        )
        return 1

    available = ort.get_available_providers()
    print(f"onnxruntime version : {ort.__version__}")
    print(f"Available providers : {available}")
    has_vitis = "VitisAIExecutionProvider" in available
    has_dml = "DmlExecutionProvider" in available
    print(f"VitisAIExecutionProvider : {'AVAILABLE' if has_vitis else 'MISSING'}")
    print(f"DmlExecutionProvider     : {'AVAILABLE' if has_dml else 'MISSING'}")

    if not has_vitis:
        print(
            "WARNING: VitisAIExecutionProvider is not active in this Python environment.\n"
            "Ensure you activated C:\\Program Files\\RyzenAI\\ryzen-ai-1.9.0\\Scripts\\activate.bat",
            file=sys.stderr,
        )
        return 2

    if not install_dir:
        print("ERROR: Could not locate Ryzen AI installation directory.", file=sys.stderr)
        return 3

    test_model = install_dir / "quicktest" / "test_model.onnx"
    if not test_model.exists():
        matches = list(install_dir.rglob("test_model.onnx"))
        test_model = matches[0] if matches else test_model

    if not test_model.exists():
        print(f"WARNING: quicktest model not found at {test_model}; skipping live session run.")
        return 0

    xclbin_ref = xclbin_file or phoenix_overlay_ref
    provider_options = [
        {
            "target": "X1",
            "xlnx_enable_py3_round": 0,
            "xclbin": str(xclbin_ref),
        }
    ]
    print("\n=== [4] Live VitisAIExecutionProvider Inference Benchmark (Hawk Point 'target': 'X1') ===")
    print(f"Model            : {test_model}")
    print(f"Provider options : {provider_options}")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    session = ort.InferenceSession(
        str(test_model),
        sess_options=session_options,
        providers=["VitisAIExecutionProvider"],
        provider_options=provider_options,
    )
    input_meta = session.get_inputs()[0]
    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
    dummy_input = np.random.rand(*shape).astype(np.float32)

    warmup_runs = 3
    bench_runs = 20
    for _ in range(warmup_runs):
        _ = session.run(None, {input_meta.name: dummy_input})

    latencies_ms: list[float] = []
    outputs = None
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_meta.name: dummy_input})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_ms = statistics.median(latencies_ms)
    p95_ms = percentile_ms(latencies_ms, 95.0)
    mean_ms = statistics.mean(latencies_ms)
    ips = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"Inference PASSED -> Output shape: {outputs[0].shape}")
    print(f"Latency (N={bench_runs}) -> mean={mean_ms:.3f} ms | p50={p50_ms:.3f} ms | p95={p95_ms:.3f} ms | {ips:.2f} inf/s")

    entry = record_experiment(
        track="01_local_ryzen_ai_npu",
        hardware_tier="Edge_NPU",
        device_name="AMD Ryzen 5 8600G NPU Phoenix (XDNA 1)",
        execution_provider="VitisAIExecutionProvider",
        model_or_workload=test_model.name,
        precision="INT8/AIE2_4x4",
        batch_size=1,
        latency_ms_p50=p50_ms,
        latency_ms_p95=p95_ms,
        throughput_val=ips,
        throughput_unit="inf/s",
        status="PASS",
        script_path=__file__,
        notes=f"target=X1 warmup={warmup_runs} runs={bench_runs} ort={ort.__version__}",
        extra_metadata={"mean_ms": round(mean_ms, 3), "xclbin": str(xclbin_ref)},
    )
    print(f"Logged to 00_RESEARCH_LIBRARY -> experiment_id={entry['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
