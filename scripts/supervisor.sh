#!/usr/bin/env bash
# Restart-on-death supervisor for one replica. Polls the replica's tmux session
# (the session dies with the training process) and relaunches after a fixed delay.
# The delay is part of the measured recovery cost — documented, not hidden.
#   scripts/supervisor.sh <run_id> <replica_idx> [restart_delay_s]
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_id}"
R="${2:?replica idx}"
DELAY="${3:-15}"
echo "supervisor[$R]: watching ftdm2r$R (restart delay ${DELAY}s)"
while true; do
  if ! tmux has-session -t "ftdm2r$R" 2>/dev/null; then
    echo "supervisor[$R]: replica down at $(date -Is); restarting in ${DELAY}s"
    sleep "$DELAY"
    bash scripts/launch_m2_replica.sh "$RUN" "$R" || echo "supervisor[$R]: relaunch failed"
  fi
  sleep 3
done
