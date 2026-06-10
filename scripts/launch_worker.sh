#!/usr/bin/env bash
# One DiLoCo replica group. Usage:
#   scripts/launch_worker.sh <replica_id> <num_replicas> <train-config> [--set k=v ...]
# Env: TORCHFT_LIGHTHOUSE (default http://worker1:29510)
set -euo pipefail
cd "$(dirname "$0")/.."
export REPLICA_GROUP_ID="$1"; shift
export NUM_REPLICA_GROUPS="$1"; shift
CONFIG="$1"; shift
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://worker1:29510}"
# torchft Manager expects torchrun-style env even for a single-process replica
# group (it spins up a TCPStore per group) — distinct port per replica on one host.
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-$((29600 + REPLICA_GROUP_ID))}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
exec .venv/bin/python -m ftdiloco.train --config "$CONFIG" "$@"
