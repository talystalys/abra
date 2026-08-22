#!/usr/bin/env bash
set -euo pipefail

GPU_UUID="${1:-GPU-5078520c-bf92-4143-0d48-b02fd9d250dd}"
OUTPUT_CSV="${2:-telemetry.csv}"
INTERVAL_SEC="${3:-0.25}"

echo "Starting GPU telemetry sampler for UUID: ${GPU_UUID} -> ${OUTPUT_CSV} (interval: ${INTERVAL_SEC}s)"

# Ensure header is written
echo "timestamp,gpu_uuid,pci_bus_id,temperature_gpu_c,power_draw_w,clocks_sm_mhz,clocks_mem_mhz,utilization_gpu_pct,utilization_mem_pct,memory_used_mib,memory_total_mib,pstate" > "${OUTPUT_CSV}"

# Run loop until killed
while true; do
    TIMESTAMP="$(date -u '+%Y-%m-%d %H:%M:%S.%3N')"
    SAMPLE="$(nvidia-smi --id="${GPU_UUID}" --query-gpu=uuid,pci.bus_id,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,memory.used,memory.total,pstate --format=csv,noheader,nounits 2>/dev/null || true)"
    
    if [[ -n "${SAMPLE}" ]]; then
        # Format: timestamp, clean csv items
        CLEAN_SAMPLE="$(echo "${SAMPLE}" | tr -d ' ')"
        echo "${TIMESTAMP},${CLEAN_SAMPLE}" >> "${OUTPUT_CSV}"
    fi
    sleep "${INTERVAL_SEC}"
done
