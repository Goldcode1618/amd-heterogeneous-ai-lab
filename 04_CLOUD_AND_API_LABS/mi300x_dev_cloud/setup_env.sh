#!/usr/bin/env bash
# Deterministic Wave 2 container bootstrap for AMD Developer Cloud MI300X instances.
# Credit Budget Reference: $100.00 @ ~$1.99/hr => ~50.25 compute hours (expires 2026-12-27)
set -euo pipefail

ROCM_IMAGE="${ROCM_IMAGE:-rocm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-mi300x_wave2}"
PYTORCH_ROCM_INDEX_URL="${PYTORCH_ROCM_INDEX_URL:-${TORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.14}}"
CLEANUP_CONTAINER="${CLEANUP_CONTAINER:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rm-container)
            CLEANUP_CONTAINER=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " AMD Developer Cloud MI300X - Wave 2 Deterministic Bootstrap"
echo " Image: ${ROCM_IMAGE} | Container: ${CONTAINER_NAME} | Torch: ${PYTORCH_ROCM_INDEX_URL}"
echo " Budget Reminder: \$100 credit (~50.25 hrs @ \$1.99/hr)"
echo " CRITICAL: Powering off STILL BILLS (\$1.99/hr)! You MUST DESTROY the GPU Droplet in devcloud.amd.com when done! (Credit \$100.00 expires 2026-12-27)"
echo "============================================================"

echo "[1/4] Host & ROCm Driver Verification"
if command -v amd-smi >/dev/null 2>&1; then
    amd-smi static | head -n 40
elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi
else
    echo "WARNING: Neither amd-smi nor rocm-smi found in PATH."
fi

if command -v rocminfo >/dev/null 2>&1; then
    echo "ROCm Agents (rocminfo):"
    rocminfo 2>/dev/null | grep -E "Name:|Marketing Name:" | head -n 12 || true
fi

if command -v hipcc >/dev/null 2>&1; then
    hipcc --version 2>/dev/null | head -n 2 || true
fi

if [ -f /opt/rocm/.info/version ]; then
    echo "ROCm Version: $(cat /opt/rocm/.info/version)"
fi

if command -v rocprofv3 >/dev/null 2>&1; then
    echo "Profiler detected: $(command -v rocprofv3)"
elif command -v rocprof >/dev/null 2>&1; then
    echo "Profiler detected: $(command -v rocprof)"
else
    echo "Profiler detected: none on host (Track A will fall back cleanly)"
fi

echo "[2/4] Deterministic Container & PyTorch ROCm Setup (${ROCM_IMAGE})"
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not in PATH." >&2
    exit 1
fi
if ! docker image inspect "${ROCM_IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: Pre-pulled image '${ROCM_IMAGE}' not found locally on droplet." >&2
    exit 1
fi

IS_RUNNING="$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
if [[ "${IS_RUNNING}" != "true" ]]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    GROUP_FLAGS=(--group-add video)
    KFD_GID="$(stat -c '%g' /dev/kfd 2>/dev/null || true)"
    if [[ -n "${KFD_GID}" && "${KFD_GID}" != "0" ]]; then
        GROUP_FLAGS+=(--group-add "${KFD_GID}")
    elif getent group render >/dev/null 2>&1; then
        GROUP_FLAGS+=(--group-add "$(getent group render | cut -d: -f3)")
    fi
    docker run -d --name "${CONTAINER_NAME}" \
        --device=/dev/kfd --device=/dev/dri "${GROUP_FLAGS[@]}" \
        --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        -v "${SCRIPT_DIR}:/work" \
        "${ROCM_IMAGE}" sleep infinity >/dev/null
    echo "Started container: ${CONTAINER_NAME} (${GROUP_FLAGS[*]})"
else
    echo "Reusing running container: ${CONTAINER_NAME}"
fi

if ! docker exec "${CONTAINER_NAME}" python3 -c "import torch" >/dev/null 2>&1; then
    echo "Installing PyTorch ROCm deterministically from ${PYTORCH_ROCM_INDEX_URL}..."
    docker exec "${CONTAINER_NAME}" pip install --no-cache-dir --index-url "${PYTORCH_ROCM_INDEX_URL}" torch
else
    echo "PyTorch already installed inside ${CONTAINER_NAME}; skipping pip install."
fi

echo "[3/4] Container PyTorch ROCm & Optional Stack Check"
docker exec "${CONTAINER_NAME}" python3 - <<'PY'
import torch
print(f"python: {__import__('sys').version.split()[0]}")
print(f"torch: {torch.__version__}")
print(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")
print(f"cuda/hip available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(f"  GPU[{idx}]: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GiB")
for pkg in ("vllm", "transformers", "accelerate", "onnxruntime"):
    try:
        mod = __import__(pkg)
        print(f"  {pkg}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  {pkg}: NOT installed")
PY

echo "[4/4] Running smoke_test_rocm.py inside ${CONTAINER_NAME}"
docker exec "${CONTAINER_NAME}" python3 /work/smoke_test_rocm.py

if [[ "${CLEANUP_CONTAINER}" -eq 1 ]]; then
    echo "Cleaning up container ${CONTAINER_NAME} (--rm-container / CLEANUP_CONTAINER=1)..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
else
    echo "Container ${CONTAINER_NAME} kept warm for Wave 2: docker exec ${CONTAINER_NAME} python3 /work/<script>.py"
fi
