#!/usr/bin/env bash
# M3 failure storm: 2 GPU replicas + supervisors + Poisson chaos for a fixed window,
# then teardown. Run ON worker4 inside tmux:
#   tmux new -d -s m3 'bash scripts/run_m3_storm.sh m3-storm-k120 configs/chaos/m3_storm_k120.yaml 2700 2>&1 | tee /tmp/m3.log'
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
SCHEDULE="${2:?chaos schedule yaml}"
DURATION="${3:-2700}"
H="${H:-50}"

echo "=== M3 storm $RUN: ${DURATION}s, schedule $SCHEDULE $(date -Is)"
bash scripts/kill_run.sh "$RUN"
rm -rf "experiments/$RUN"
mkdir -p "experiments/$RUN"
sudo bash scripts/netns_cluster.sh up 2 || true

tmux kill-session -t ftdm2mon 2>/dev/null || true
tmux new -d -s ftdm2mon \
  ".venv/bin/python -m ftdiloco.netmon --ifaces vftd0 vftd1 --out experiments/$RUN/netmon.jsonl --interval 0.5"

# workers (max_steps high — the storm window is wall-clock bounded)
for R in 0 1; do STEPS=99999 H=$H bash scripts/launch_m2_replica.sh "$RUN" "$R"; done

# supervisors
for R in 0 1; do
  tmux kill-session -t "ftdsup$R" 2>/dev/null || true
  tmux new -d -s "ftdsup$R" \
    "cd /srv/fpga/ft-diloco && bash scripts/supervisor.sh $RUN $R 15 >> experiments/$RUN/supervisor$R.log 2>&1"
done

sleep 90  # warmup before the first fault
echo "=== chaos begins $(date -Is)"
.venv/bin/python -m chaos.controller --schedule "$SCHEDULE" --run-dir "experiments/$RUN"
echo "=== chaos schedule done $(date -Is); cooldown 120s"
sleep 120

echo "=== teardown $(date -Is)"
for R in 0 1; do tmux kill-session -t "ftdsup$R" 2>/dev/null || true; done
bash scripts/kill_run.sh "$RUN"
tmux kill-session -t ftdm2mon 2>/dev/null || true
echo "MARK storm_complete $RUN $(date -Is)"
