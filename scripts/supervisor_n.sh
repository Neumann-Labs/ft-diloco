#!/usr/bin/env bash
# Restart-on-death supervisor for an N-way storm. One loop watches all N replica
# tmux sessions (ftdr0..ftdr<N-1>) — the session dies with the training process —
# and relaunches the dead one after a fixed delay. The delay is part of the measured
# recovery cost (documented, not hidden). A SIGSTOP'd straggler keeps its session
# alive, so the supervisor correctly leaves it alone; only SIGKILL/exit triggers
# relaunch. Generalized from supervisor.sh (which watched a single replica).
#   scripts/supervisor_n.sh <run_id> <num_replicas> [restart_delay_s] [config]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
N="${2:?num replicas}"
DELAY="${3:-15}"
CONFIG="${4:-configs/train/storm_micro.yaml}"
echo "supervisor: watching ftdr0..ftdr$((N-1)) (restart delay ${DELAY}s)"
while true; do
  for R in $(seq 0 $((N - 1))); do
    if ! tmux has-session -t "ftdr$R" 2>/dev/null; then
      echo "supervisor: replica $R down at $(date -Is); restarting in ${DELAY}s"
      sleep "$DELAY"
      STEPS="${STEPS:-1000000}" bash scripts/launch_storm_replica.sh "$RUN" "$R" "$N" "$CONFIG" \
        || echo "supervisor: relaunch $R failed"
    fi
  done
  sleep 3
done
