#!/usr/bin/env bash
# M4 WAN sweep: throttle the netns link with netem and measure throughput for
# DiLoCo (H=100) vs per-step sync (H=1, the DDP communication pattern).
# Our documented profiles: rate in {1000,100,50,10} Mbps, 10ms delay each way (20ms RTT).
# Step budgets are adaptive — at 10 Mbps a single H=1 "step" costs ~3 min of sync.
# Run inside tmux on worker4:
#   tmux new -d -s m4 'bash scripts/run_m4_wan.sh 2>&1 | tee /tmp/m4.log'
set -euo pipefail
cd "$(dirname "$0")/.."
sudo bash scripts/netns_cluster.sh up 2 || true

# profile table: label rate_mbit h steps pg_timeout
PROFILES=(
  "1000 1   60  120"
  "1000 100 300 120"
  "100  1   30  240"
  "100  100 300 240"
  "50   1   20  300"
  "50   100 300 300"
  "10   1   6   900"
  "10   100 220 900"
)

for prof in "${PROFILES[@]}"; do
  read -r RATE H STEPS PGT <<<"$prof"
  RUN="m4-wan-${RATE}mbit-h${H}"
  echo "=== $RUN (rate=${RATE}mbit RTT=20ms H=$H steps=$STEPS) $(date -Is)"
  bash scripts/kill_run.sh "$RUN"; rm -rf "experiments/$RUN"; mkdir -p "experiments/$RUN"
  for i in 0 1; do
    sudo bash scripts/netns_cluster.sh netem $i rate "${RATE}mbit" delay 10ms
  done
  .venv/bin/python -m ftdiloco.netmon --ifaces vftd0 vftd1 \
    --out "experiments/$RUN/netmon.jsonl" --interval 0.5 &
  NETMON=$!
  pids=()
  for R in 0 1; do
    sudo ip netns exec "ftd$R" sudo -u claude env \
      MASTER_ADDR="10.77.0.1$R" MASTER_PORT=$((29600 + R)) RANK=0 WORLD_SIZE=1 \
      GLOO_SOCKET_IFNAME=eth0 REPLICA_GROUP_ID=$R NUM_REPLICA_GROUPS=2 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      TORCHFT_LIGHTHOUSE=http://192.168.1.104:29510 FTD_ADVERTISE_HOST="10.77.0.1$R" \
      "$PWD/.venv/bin/python" -m ftdiloco.train --config configs/train/m1_diloco.yaml \
      --set run_id="$RUN" --set sync_every="$H" --set max_steps="$STEPS" \
      --set ckpt_every_syncs=0 --set eval_batches=25 \
      --set pg_timeout_s="$PGT" --set quorum_timeout_s="$PGT" \
      > "experiments/$RUN/worker$R.log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  kill "$NETMON" 2>/dev/null || true
  for i in 0 1; do sudo bash scripts/netns_cluster.sh clear-netem $i; done
  echo "=== $RUN done rc=$rc $(date -Is)"
done
echo "MARK wan_sweep_complete $(date -Is)"
