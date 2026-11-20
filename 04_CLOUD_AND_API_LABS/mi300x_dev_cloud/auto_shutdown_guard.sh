#!/usr/bin/env bash
# Auto-shutdown & credit-budget watchdog for AMD Developer Cloud MI300X ($1.99/hr, $100 credit cap).
# Usage:
#   bash auto_shutdown_guard.sh [--max-minutes 180] [--idle-minutes 20] [--enforce]

set -euo pipefail

MAX_MINUTES=180
IDLE_LIMIT_MINUTES=20
POLL_INTERVAL_SEC=60
ENFORCE_SHUTDOWN=0
HOURLY_RATE_USD="1.99"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-minutes)
            MAX_MINUTES="$2"
            shift 2
            ;;
        --idle-minutes)
            IDLE_LIMIT_MINUTES="$2"
            shift 2
            ;;
        --poll-seconds)
            POLL_INTERVAL_SEC="$2"
            shift 2
            ;;
        --enforce)
            ENFORCE_SHUTDOWN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

START_EPOCH=$(date +%s)
IDLE_SECONDS=0
LOG_FILE="${HOME}/mi300x_credit_guard.log"

get_gpu_util() {
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showuse 2>/dev/null | awk '/GPU use \(%\)/ {print int($NF); exit}' || echo "0"
    elif command -v amd-smi >/dev/null 2>&1; then
        amd-smi metric -u 2>/dev/null | awk '/GFX_ACTIVITY/ {print int($NF); exit}' || echo "0"
    else
        echo "0"
    fi
}

echo "============================================================" | tee -a "${LOG_FILE}"
echo " MI300X Credit Watchdog Started (UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "${LOG_FILE}"
echo " Session Cap: ${MAX_MINUTES} min | Idle Cap: ${IDLE_LIMIT_MINUTES} min | Enforce: ${ENFORCE_SHUTDOWN}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

while true; do
    NOW_EPOCH=$(date +%s)
    ELAPSED_SEC=$((NOW_EPOCH - START_EPOCH))
    ELAPSED_MIN=$((ELAPSED_SEC / 60))
    EST_COST=$(awk -v sec="${ELAPSED_SEC}" -v rate="${HOURLY_RATE_USD}" 'BEGIN { printf "%.2f", (sec / 3600.0) * rate }')

    GPU_UTIL=$(get_gpu_util)
    GPU_UTIL="${GPU_UTIL:-0}"

    if [[ "${GPU_UTIL}" -lt 5 ]]; then
        IDLE_SECONDS=$((IDLE_SECONDS + POLL_INTERVAL_SEC))
    else
        IDLE_SECONDS=0
    fi
    IDLE_MIN=$((IDLE_SECONDS / 60))

    MSG="[$(date -u +%H:%M:%SZ)] elapsed=${ELAPSED_MIN}m est_cost=\$${EST_COST} gpu_util=${GPU_UTIL}% idle=${IDLE_MIN}m"
    echo "${MSG}" | tee -a "${LOG_FILE}"

    if [[ "${ELAPSED_MIN}" -ge "${MAX_MINUTES}" ]]; then
        echo "ALERT: Session cap (${MAX_MINUTES} min, ~\$${EST_COST}) reached!" | tee -a "${LOG_FILE}"
        if [[ "${ENFORCE_SHUTDOWN}" -eq 1 ]]; then
            echo "Executing sudo poweroff to preserve AMD Developer Cloud credits..." | tee -a "${LOG_FILE}"
            sudo poweroff || true
            exit 0
        fi
    fi

    if [[ "${IDLE_MIN}" -ge "${IDLE_LIMIT_MINUTES}" ]]; then
        echo "ALERT: MI300X GPU idle for ${IDLE_MIN} min (limit ${IDLE_LIMIT_MINUTES} min)!" | tee -a "${LOG_FILE}"
        if [[ "${ENFORCE_SHUTDOWN}" -eq 1 ]]; then
            echo "Executing sudo poweroff due to sustained GPU idle..." | tee -a "${LOG_FILE}"
            sudo poweroff || true
            exit 0
        fi
    fi

    sleep "${POLL_INTERVAL_SEC}"
done
