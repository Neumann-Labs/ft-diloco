#!/usr/bin/env bash
# M4 two-physical-machine datapoint: worker4 GPU replica + worker1 CPU replica over
# real gigabit ethernet (no netns). worker1 runs batch 1 (measured 805ms/step ≈ GPU's
# 850ms/step) — the "weak volunteer node" config, equal H, honest caveats documented.
# Run ON worker4 inside tmux:  bash scripts/run_m4_twomachine.sh [steps]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="m4-twomachine"
STEPS="${1:-400}"
H="${H:-100}"
W4_IP=192.168.1.66
W1_IP=192.168.1.104

bash scripts/kill_run.sh "$RUN"
rm -rf "experiments/$RUN"; mkdir -p "experiments/$RUN"

# replica 0: GPU on this host (direct LAN, no namespace)
tmux kill-session -t ftd2m0 2>/dev/null || true
tmux new -d -s ftd2m0 \
  "cd /srv/fpga/ft-diloco && MASTER_ADDR=$W4_IP MASTER_PORT=29600 RANK=0 WORLD_SIZE=1 \
   GLOO_SOCKET_IFNAME=enp7s0 REPLICA_GROUP_ID=0 NUM_REPLICA_GROUPS=2 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   TORCHFT_LIGHTHOUSE=http://$W1_IP:29510 FTD_ADVERTISE_HOST=$W4_IP \
   .venv/bin/python -m ftdiloco.train --config configs/train/m1_diloco.yaml \
     --set run_id=$RUN --set sync_every=$H --set max_steps=$STEPS \
     --set ckpt_every_syncs=0 --set quorum_timeout_s=300 --set pg_timeout_s=300 \
     >> experiments/$RUN/worker0.log 2>&1"

# replica 1: CPU on worker1, batch 1 (step-time-matched)
ssh -o ConnectTimeout=10 "claude@$W1_IP" "cd ~/ft-diloco && mkdir -p experiments/$RUN && \
  tmux kill-session -t ftd2m1 2>/dev/null; tmux new -d -s ftd2m1 \
  'cd ~/ft-diloco && MASTER_ADDR=$W1_IP MASTER_PORT=29600 RANK=0 WORLD_SIZE=1 \
   GLOO_SOCKET_IFNAME=eno1 REPLICA_GROUP_ID=1 NUM_REPLICA_GROUPS=2 \
   TORCHFT_LIGHTHOUSE=http://$W1_IP:29510 FTD_ADVERTISE_HOST=$W1_IP \
   .venv/bin/python -m ftdiloco.train --config configs/train/m1_diloco.yaml \
     --set run_id=$RUN --set sync_every=$H --set max_steps=$STEPS \
     --set device=cpu --set dtype=float32 --set batch_size=1 --set grad_accum=1 \
     --set ckpt_every_syncs=0 --set quorum_timeout_s=300 --set pg_timeout_s=300 \
     >> experiments/$RUN/worker1.log 2>&1'"
echo "TWOMACHINE_LAUNCHED $RUN steps=$STEPS H=$H $(date -Is)"
