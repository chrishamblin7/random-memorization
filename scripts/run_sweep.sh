#!/usr/bin/env bash
# GPU job scheduler: runs up to 4 jobs concurrently (one per GPU),
# launching the next config whenever a GPU becomes free.
#
# Usage:
#   cd ~/projects/random-memorization
#   bash scripts/run_sweep.sh experiments/configs/mlp_N50k_*.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
NUM_GPUS=4
POLL_INTERVAL=30

configs=("$@")
n_configs=${#configs[@]}

if [ "$n_configs" -eq 0 ]; then
    echo "Usage: $0 <config1.yaml> [config2.yaml ...]"
    exit 1
fi

echo "=== Sweep: $n_configs configs across $NUM_GPUS GPUs ==="
for c in "${configs[@]}"; do echo "  $c"; done
echo ""

declare -A gpu_pid
declare -A gpu_config
for g in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_pid[$g]=0
    gpu_config[$g]=""
done

next_idx=0
running=0

launch_on_gpu() {
    local gpu=$1
    local cfg=$2
    local logfile="/tmp/rm_sweep_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON" experiments/train_random.py --config "$cfg" > "$logfile" 2>&1 &
    local pid=$!
    gpu_pid[$gpu]=$pid
    gpu_config[$gpu]="$cfg"
    echo "[$(date +%H:%M:%S)] GPU $gpu: launched PID=$pid  config=$(basename "$cfg")"
    running=$((running + 1))
}

gpu_is_free() {
    local gpu=$1
    local pid=${gpu_pid[$gpu]}
    if [ "$pid" -eq 0 ]; then
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        echo "[$(date +%H:%M:%S)] GPU $gpu: finished  config=$(basename "${gpu_config[$gpu]}")"
        gpu_pid[$gpu]=0
        gpu_config[$gpu]=""
        running=$((running - 1))
        return 0
    fi
    return 1
}

while [ "$next_idx" -lt "$n_configs" ] || [ "$running" -gt 0 ]; do
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        if gpu_is_free "$g" && [ "$next_idx" -lt "$n_configs" ]; then
            launch_on_gpu "$g" "${configs[$next_idx]}"
            next_idx=$((next_idx + 1))
        fi
    done

    if [ "$next_idx" -lt "$n_configs" ] || [ "$running" -gt 0 ]; then
        sleep "$POLL_INTERVAL"
    fi
done

echo ""
echo "=== Sweep complete: $n_configs experiments finished ==="
