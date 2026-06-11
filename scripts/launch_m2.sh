#!/usr/bin/env bash
# Start a full M2 chaos run: netns up, netmon, both replicas. Controller runs separately.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-m2-kill-rejoin}"
sudo bash scripts/netns_cluster.sh up 2 || true
mkdir -p "experiments/$RUN"
tmux kill-session -t ftdm2mon 2>/dev/null || true
tmux new -d -s ftdm2mon \
  ".venv/bin/python -m ftdiloco.netmon --ifaces vftd0 vftd1 --out experiments/$RUN/netmon.jsonl --interval 0.5"
for R in 0 1; do bash scripts/launch_m2_replica.sh "$RUN" "$R"; done
echo "M2_RUN_STARTED $RUN $(date -Is)"
