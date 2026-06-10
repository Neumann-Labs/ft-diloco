#!/usr/bin/env bash
# M0.5 smoke: 2 CPU replicas (micro model, synthetic data) against the worker1
# lighthouse. Run ON worker4. Each replica goes in its own tmux session so the
# chaos side (kill -9, relaunch) can target them independently.
#   scripts/run_m05_smoke.sh [run_id]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-m05-smoke}"
LH="${TORCHFT_LIGHTHOUSE:-http://worker1.attlocal.net:29510}"
mkdir -p "experiments/$RUN"

launch_replica() {
  local R="$1"
  tmux kill-session -t "ftdsmoke$R" 2>/dev/null || true
  tmux new -d -s "ftdsmoke$R" \
    "cd /srv/fpga/ft-diloco && MASTER_ADDR=localhost MASTER_PORT=$((29600 + R)) RANK=0 WORLD_SIZE=1 \
     REPLICA_GROUP_ID=$R NUM_REPLICA_GROUPS=2 TORCHFT_LIGHTHOUSE=$LH \
     .venv/bin/python -m ftdiloco.train --config configs/train/m1_diloco.yaml \
       --set run_id=$RUN --set model=micro --set data_dir=/tmp/ftd-smoke-data \
       --set device=cpu --set dtype=float32 --set batch_size=4 --set max_steps=600 \
       --set sync_every=20 --set eval_every=100 --set eval_batches=2 --set log_every=20 \
       >> experiments/$RUN/worker$R.log 2>&1"
}

case "${2:-both}" in
  both) launch_replica 0; launch_replica 1 ;;
  r0) launch_replica 0 ;;
  r1) launch_replica 1 ;;
esac
echo "SMOKE_LAUNCHED ${2:-both}"
