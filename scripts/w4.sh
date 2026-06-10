#!/usr/bin/env bash
# Run a command on worker4, retrying across Tailscale link flaps until the
# sentinel "W4_DONE" comes back. Usage: scripts/w4.sh '<remote command>'
# The remote command runs under bash -c on worker4 from /srv/fpga/ft-diloco.
set -uo pipefail
CMD="${1:?remote command}"
TRIES="${TRIES:-30}"
for _ in $(seq "$TRIES"); do
  out=$(ssh -o ConnectTimeout=15 worker4 "cd /srv/fpga/ft-diloco && { $CMD ; } && echo W4_DONE" 2>/dev/null)
  if printf '%s' "$out" | grep -q "W4_DONE"; then
    printf '%s\n' "$out" | grep -v "^W4_DONE$"
    exit 0
  fi
  sleep 45
done
echo "W4_GAVE_UP after $TRIES tries" >&2
exit 1
