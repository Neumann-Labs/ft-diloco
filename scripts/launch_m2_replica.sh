#!/usr/bin/env bash
# Launch ONE GPU DiLoCo replica for an M2 chaos run, inside its netns, in tmux.
# Usage: launch_m2_replica.sh <run_id> <replica_idx>   (idempotent per replica)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
R="${2:?replica idx}"
STEPS="${STEPS:-3000}"
H="${H:-50}"
mkdir -p "experiments/$RUN"
tmux kill-session -t "ftdm2r$R" 2>/dev/null || true
tmux new -d -s "ftdm2r$R" \
  "cd /srv/fpga/ft-diloco && sudo ip netns exec ftd$R sudo -u claude env \
   MASTER_ADDR=10.77.0.1$R MASTER_PORT=$((29600 + R)) RANK=0 WORLD_SIZE=1 \
   GLOO_SOCKET_IFNAME=eth0 REPLICA_GROUP_ID=$R NUM_REPLICA_GROUPS=2 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   TORCHFT_LIGHTHOUSE=http://192.168.1.104:29510 FTD_ADVERTISE_HOST=10.77.0.1$R \
   /srv/fpga/ft-diloco/.venv/bin/python -m ftdiloco.train \
     --config configs/train/m1_diloco.yaml \
     --set run_id=$RUN --set sync_every=$H --set max_steps=$STEPS \
     >> experiments/$RUN/worker$R.log 2>&1"
echo "replica $R launched for $RUN"
