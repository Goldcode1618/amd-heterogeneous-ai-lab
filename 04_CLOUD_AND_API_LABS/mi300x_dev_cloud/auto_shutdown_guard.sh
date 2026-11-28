#!/usr/bin/env bash
# Auto-destroy & credit-budget watchdog for AMD Developer Cloud MI300X ($1.99/hr, $100 credit cap, expires 2026-12-27).
# IMPORTANT: DigitalOcean GPU Droplets STILL BILL when powered off; you MUST DESTROY instances to stop billing.
# Usage:
#   bash auto_shutdown_guard.sh [--max-minutes 60] [--idle-minutes 15] [--credit-usd 100.00] [--enforce-destroy]

set -euo pipefail

MAX_MINUTES=60
IDLE_LIMIT_MINUTES=15
POLL_INTERVAL_SEC=60
CREDIT_USD="100.00"
HOURLY_RATE_USD="1.99"
DROPLET_ID="${DROPLET_ID:-}"
ENFORCE_DESTROY=0

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
        --credit-usd)
            CREDIT_USD="$2"
            shift 2
            ;;
        --hourly-rate)
            HOURLY_RATE_USD="$2"
            shift 2
            ;;
        --droplet-id)
            DROPLET_ID="$2"
            shift 2
            ;;
        --enforce|--enforce-destroy)
            ENFORCE_DESTROY=1
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
MAX_SAFE_HOURS=$(awk -v c="${CREDIT_USD}" -v r="${HOURLY_RATE_USD}" 'BEGIN { printf "%.2f", c / r }')

get_gpu_util() {
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showuse 2>/dev/null | awk '/GPU use \(%\)/ {print int($NF); exit}' || echo "0"
    elif command -v amd-smi >/dev/null 2>&1; then
        amd-smi metric -u 2>/dev/null | awk '/GFX_ACTIVITY/ {print int($NF); exit}' || echo "0"
    else
        echo "0"
    fi
}

emit_critical_alert() {
    local reason="$1"
    local est_cost="$2"
    local banner="CRITICAL: MANUAL_DESTROY_REQUIRED (${reason}, est_cost=\$${est_cost}) — Powering off STILL BILLS \$${HOURLY_RATE_USD}/hr! Destroy this droplet in devcloud.amd.com!"
    echo "${banner}" | tee -a "${LOG_FILE}"
    if command -v wall >/dev/null 2>&1; then
        printf '%s\n' "${banner}" | wall 2>/dev/null || true
    fi
}

attempt_api_destroy() {
    local reason="$1"
    local est_cost="$2"
    emit_critical_alert "${reason}" "${est_cost}"

    if [[ "${ENFORCE_DESTROY}" -ne 1 ]]; then
        return 0
    fi

    if [[ -z "${DO_API_TOKEN:-}" ]]; then
        echo "NOTICE: DO_API_TOKEN is not set. Intentionally NOT running 'sudo poweroff' because powered-off GPU droplets STILL BILL \$${HOURLY_RATE_USD}/hr. Destroy manually in devcloud.amd.com!" | tee -a "${LOG_FILE}"
        return 1
    fi

    local target_id="${DROPLET_ID}"
    if [[ -z "${target_id}" ]] && command -v curl >/dev/null 2>&1; then
        target_id=$(curl -fsS --max-time 3 "http://169.254.169.254/metadata/v1/id" 2>/dev/null || true)
    fi

    if [[ -z "${target_id}" ]]; then
        echo "ERROR: Cannot resolve DROPLET_ID for API Destroy. Intentionally NOT running 'sudo poweroff' — destroy manually in devcloud.amd.com!" | tee -a "${LOG_FILE}"
        return 1
    fi

    echo "Calling DigitalOcean API DELETE for droplet ${target_id} to DESTROY instance and stop billing..." | tee -a "${LOG_FILE}"
    if curl -fsS -X DELETE -H "Authorization: Bearer ${DO_API_TOKEN}" "https://api.digitalocean.com/v2/droplets/${target_id}"; then
        echo "SUCCESS: Droplet ${target_id} destroy request accepted by API." | tee -a "${LOG_FILE}"
        exit 0
    else
        echo "ERROR: API destroy failed for droplet ${target_id}. Intentionally NOT running 'sudo poweroff' — destroy manually in devcloud.amd.com!" | tee -a "${LOG_FILE}"
        return 1
    fi
}

echo "============================================================" | tee -a "${LOG_FILE}"
echo " MI300X Credit Watchdog Started (UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "${LOG_FILE}"
echo " Credit: \$${CREDIT_USD} (exp 2026-12-27, ~${MAX_SAFE_HOURS} hrs @ \$${HOURLY_RATE_USD}/hr)" | tee -a "${LOG_FILE}"
echo " Session Cap: ${MAX_MINUTES} min | Idle Cap: ${IDLE_LIMIT_MINUTES} min | Enforce Destroy: ${ENFORCE_DESTROY}" | tee -a "${LOG_FILE}"
echo " WARNING: Powered-off GPU Droplets STILL BILL — only DESTROY stops charges!" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

while true; do
    NOW_EPOCH=$(date +%s)
    ELAPSED_SEC=$((NOW_EPOCH - START_EPOCH))
    ELAPSED_MIN=$((ELAPSED_SEC / 60))
    EST_COST=$(awk -v sec="${ELAPSED_SEC}" -v rate="${HOURLY_RATE_USD}" 'BEGIN { printf "%.2f", (sec / 3600.0) * rate }')
    REM_USD=$(awk -v c="${CREDIT_USD}" -v e="${EST_COST}" 'BEGIN { rem = c - e; if (rem < 0) rem = 0; printf "%.2f", rem }')
    REM_HRS=$(awk -v rem="${REM_USD}" -v rate="${HOURLY_RATE_USD}" 'BEGIN { printf "%.2f", rem / rate }')
    BUDGET_ALERT=$(awk -v e="${EST_COST}" -v c="${CREDIT_USD}" 'BEGIN { print (e >= c * 0.90) ? 1 : 0 }')

    GPU_UTIL=$(get_gpu_util)
    GPU_UTIL="${GPU_UTIL:-0}"

    if [[ "${GPU_UTIL}" -lt 5 ]]; then
        IDLE_SECONDS=$((IDLE_SECONDS + POLL_INTERVAL_SEC))
    else
        IDLE_SECONDS=0
    fi
    IDLE_MIN=$((IDLE_SECONDS / 60))

    MSG="[$(date -u +%H:%M:%SZ)] elapsed=${ELAPSED_MIN}m est_cost=\$${EST_COST} remaining=\$${REM_USD} (${REM_HRS}h left) gpu_util=${GPU_UTIL}% idle=${IDLE_MIN}m exp=2026-12-27"
    echo "${MSG}" | tee -a "${LOG_FILE}"

    if [[ "${BUDGET_ALERT}" -eq 1 ]]; then
        attempt_api_destroy "est_cost >= 90% of \$${CREDIT_USD} budget" "${EST_COST}" || true
    elif [[ "${ELAPSED_MIN}" -ge "${MAX_MINUTES}" ]]; then
        attempt_api_destroy "session cap ${MAX_MINUTES} min reached" "${EST_COST}" || true
    elif [[ "${IDLE_MIN}" -ge "${IDLE_LIMIT_MINUTES}" ]]; then
        attempt_api_destroy "GPU idle ${IDLE_MIN} min (limit ${IDLE_LIMIT_MINUTES} min)" "${EST_COST}" || true
    fi

    sleep "${POLL_INTERVAL_SEC}"
done
