#!/usr/bin/env bash
# Wave 2 MI300X multi-track execution runner (Track A HIP/rocprofv3, Track B PyTorch suite, Track C guarded vLLM).
# NOTE: Stopping charges on devcloud.amd.com requires permanent DESTROY in the portal, NEVER VM poweroff!
set -euo pipefail

START_EPOCH="$(date +%s)"
CUTOFF_MIN=38
TRACK_C_MIN_REMAINING=15
CONTAINER_NAME="${CONTAINER_NAME:-mi300x_wave2}"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-epoch)
            START_EPOCH="$2"
            shift 2
            ;;
        --cutoff-min)
            CUTOFF_MIN="$2"
            shift 2
            ;;
        --track-c-min-remaining)
            TRACK_C_MIN_REMAINING="$2"
            shift 2
            ;;
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

HIP_LOG="${WORK_DIR}/hip_kernels_gfx942.log"
SUITE_LOG="${WORK_DIR}/wave2_mi300x_suite.log"
VLLM_JSON="${WORK_DIR}/vllm_live_probe.json"
: > "${HIP_LOG}"

profile_binary() {
    local name="$1"
    local bin="$2"

    if command -v rocprofv3 >/dev/null 2>&1; then
        echo "--- Profiling ${name} with rocprofv3 ---" | tee -a "${HIP_LOG}"
        if rocprofv3 --kernel-trace --stats -d "${WORK_DIR}/rocprofv3_${name}" -- "${bin}" >>"${HIP_LOG}" 2>&1; then
            echo "PROFILER_OK: rocprofv3 (${name})" | tee -a "${HIP_LOG}"
            return 0
        fi
        echo "PROFILER_WARN: rocprofv3 failed for ${name}; attempting rocprof fallback..." | tee -a "${HIP_LOG}"
    fi

    if command -v rocprof >/dev/null 2>&1; then
        echo "--- Profiling ${name} with rocprof ---" | tee -a "${HIP_LOG}"
        if rocprof --stats -o "${WORK_DIR}/rocprof_${name}.csv" "${bin}" >>"${HIP_LOG}" 2>&1; then
            echo "PROFILER_OK: rocprof (${name})" | tee -a "${HIP_LOG}"
            return 0
        fi
        echo "PROFILER_WARN: rocprof failed for ${name}." | tee -a "${HIP_LOG}"
    fi

    echo "PROFILER_FALLBACK: continuing without GPU profiler for ${name}" | tee -a "${HIP_LOG}"
    return 0
}

echo "============================================================"
echo " Wave 2 MI300X Multi-Track Runner"
echo " Start Epoch: ${START_EPOCH} | Cutoff: ${CUTOFF_MIN}m | Track C Guard: >${TRACK_C_MIN_REMAINING}m left"
echo " WARNING: Permanent DESTROY in devcloud.amd.com is REQUIRED after artifact pull!"
echo "============================================================"

echo "[Track A] Native HIP C++ Kernels (gfx942) + Profiler Fallback" | tee -a "${HIP_LOG}"
HIP_SOURCES=(
    "saxpy_fma:${WORK_DIR}/saxpy_fma_kernel.hip.cpp"
    "wavefront_reduction:${WORK_DIR}/wavefront_reduction.hip.cpp"
    "tiled_gemm_lds:${WORK_DIR}/tiled_gemm_lds.hip.cpp"
)

for entry in "${HIP_SOURCES[@]}"; do
    name="${entry%%:*}"
    src="${entry#*:}"
    bin="${WORK_DIR}/${name}.out"
    if [[ ! -f "${src}" ]]; then
        echo "[Track A SKIP] Missing source: ${src}" | tee -a "${HIP_LOG}"
        continue
    fi
    if ! command -v hipcc >/dev/null 2>&1; then
        echo "[Track A SKIP] hipcc not found in PATH" | tee -a "${HIP_LOG}"
        break
    fi

    echo "--- Compiling ${name} (${src}) ---" | tee -a "${HIP_LOG}"
    if hipcc -O3 --offload-arch=gfx942 "${src}" -o "${bin}" >>"${HIP_LOG}" 2>&1; then
        echo "--- Running ${name} ---" | tee -a "${HIP_LOG}"
        "${bin}" >>"${HIP_LOG}" 2>&1 || echo "[Track A WARN] ${name} exited non-zero" | tee -a "${HIP_LOG}"
        profile_binary "${name}" "${bin}"
    else
        echo "[Track A WARN] hipcc compilation failed for ${name}; continuing" | tee -a "${HIP_LOG}"
    fi
done

echo "[Track B] PyTorch ROCm Hardware Suite inside ${CONTAINER_NAME}"
if [[ -f "${WORK_DIR}/wave2_mi300x_suite.py" ]]; then
    docker exec "${CONTAINER_NAME}" python3 /work/wave2_mi300x_suite.py --out-dir /work 2>&1 | tee "${SUITE_LOG}" || true
else
    echo "[Track B SKIP] wave2_mi300x_suite.py not found in ${WORK_DIR}" | tee "${SUITE_LOG}"
fi

NOW_EPOCH="$(date +%s)"
ELAPSED_MIN=$(( (NOW_EPOCH - START_EPOCH) / 60 ))
REM_MIN=$(( CUTOFF_MIN - ELAPSED_MIN ))
HAS_VLLM=0
if docker exec "${CONTAINER_NAME}" python3 -c "import vllm" >/dev/null 2>&1; then
    HAS_VLLM=1
fi

echo "[Track C] Guard Check: elapsed=${ELAPSED_MIN}m cutoff=${CUTOFF_MIN}m remaining=${REM_MIN}m has_vllm=${HAS_VLLM}"
if [[ "${HAS_VLLM}" -eq 1 && "${REM_MIN}" -ge 5 ]] || [[ "${REM_MIN}" -gt "${TRACK_C_MIN_REMAINING}" ]]; then
    echo "[Track C] Condition met (has_vllm=${HAS_VLLM}, rem_min=${REM_MIN}m). Recording runtime probe..."
    docker exec -i "${CONTAINER_NAME}" python3 - <<'PY' > "${VLLM_JSON}" || true
import json
try:
    import vllm
    payload = {"status": "VLLM_PRESENT", "vllm_version": getattr(vllm, "__version__", "unknown")}
except ImportError:
    payload = {"status": "VLLM_NOT_PREINSTALLED", "note": "Skipped heavy download to protect Wave 2 credit budget."}
print(json.dumps(payload, indent=2))
PY
else
    echo "[Track C SKIPPED] elapsed=${ELAPSED_MIN}m, remaining_before_cutoff=${REM_MIN}m (requires has_vllm=1 with >=5m OR >${TRACK_C_MIN_REMAINING}m left)."
    printf '{\n  "status": "SKIPPED",\n  "elapsed_min": %s,\n  "remaining_before_cutoff_min": %s,\n  "has_vllm": %s\n}\n' \
        "${ELAPSED_MIN}" "${REM_MIN}" "${HAS_VLLM}" > "${VLLM_JSON}"
fi

echo "============================================================"
echo " Wave 2 Execution Complete. NEXT REQUIRED STEPS:"
echo " 1) Pull logs & JSON/JSONL artifacts to local disk."
echo " 2) Permanently DESTROY this GPU Droplet in devcloud.amd.com!"
echo "    (Do NOT rely on VM poweroff; powered-off GPU droplets STILL BILL \$1.99/hr.)"
echo "============================================================"
