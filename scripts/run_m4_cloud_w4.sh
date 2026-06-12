#!/usr/bin/env bash
# worker4's replica for a cloud-hybrid run: advertises its TAILSCALE address so cloud
# peers can reach the store/gloo/recovery endpoints across the real internet.
#   bash scripts/run_m4_cloud_w4.sh <run_id> <n_replicas> [steps]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
N="${2:?n_replicas}"
STEPS="${3:-400}"
H="${H:-100}"
TS_IP=$(tailscale ip -4)
mkdir -p "experiments/$RUN"
tmux kill-session -t ftdcloud0 2>/dev/null || true
tmux new -d -s ftdcloud0 \
  "cd /srv/fpga/ft-diloco && MASTER_ADDR=$TS_IP MASTER_PORT=29600 RANK=0 WORLD_SIZE=1 \
   GLOO_SOCKET_IFNAME=tailscale0 REPLICA_GROUP_ID=0 NUM_REPLICA_GROUPS=$N \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   TORCHFT_LIGHTHOUSE=http://100.86.208.63:29510 FTD_ADVERTISE_HOST=$TS_IP \
   .venv/bin/python -m ftdiloco.train --config configs/train/m1_diloco.yaml \
     --set run_id=$RUN --set sync_every=$H --set max_steps=$STEPS \
     --set ckpt_every_syncs=0 --set quorum_timeout_s=600 --set pg_timeout_s=600 \
     >> experiments/$RUN/worker0.log 2>&1"
echo "W4_CLOUD_REPLICA_LAUNCHED $RUN ts=$TS_IP"
