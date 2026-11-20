#!/usr/bin/env bash
# Quick bootstrap script for AMD Developer Cloud MI300X instances
# Credit Budget Reference: $100.00 @ ~$1.99/hr => ~50.25 compute hours
set -euo pipefail

echo "============================================================"
echo " AMD Developer Cloud MI300X - Environment Bootstrap"
echo " Budget Reminder: \$100 credit (~50.25 hrs @ \$1.99/hr)"
echo " Remember to STOP/TERMINATE instance when idle!"
echo "============================================================"

echo "[1/4] Host & ROCm Driver Verification"
if command -v amd-smi >/dev/null 2>&1; then
    amd-smi static | head -n 40
elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi
else
    echo "WARNING: Neither amd-smi nor rocm-smi found in PATH."
fi

if [ -f /opt/rocm/.info/version ]; then
    echo "ROCm Version: $(cat /opt/rocm/.info/version)"
fi

echo "[2/4] Python & PyTorch ROCm Check"
python3 --version
python3 - <<'PY'
import sys
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")
    print(f"cuda/hip available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"  GPU[{idx}]: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GiB")
except ImportError as exc:
    print(f"PyTorch not installed in active Python ({sys.executable}): {exc}")
PY

echo "[3/4] Optional vLLM / Transformers Check"
python3 - <<'PY'
for pkg in ("vllm", "transformers", "accelerate", "onnxruntime"):
    try:
        mod = __import__(pkg)
        print(f"  {pkg}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  {pkg}: NOT installed")
PY

echo "[4/4] Running smoke_test_rocm.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/smoke_test_rocm.py"
