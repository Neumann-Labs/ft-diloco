"""Chaos controller: executes a YAML fault schedule against a live run and logs every
action (with pre-resolved PIDs and monotonic+wall timestamps) to chaos.jsonl — the
ground-truth fault timeline that analysis/parse_logs.py fuses with training telemetry.

Schedule format (configs/chaos/*.yaml):

  run_id: m2-kill-rejoin
  launch_cmd: "bash scripts/launch_m2_replica.sh m2-kill-rejoin {R}"
  events:
    - {at: 300, fault: kill, target: 1}
    - {at: 420, fault: relaunch, target: 1}
    - {at: 600, fault: partition, target: 1}
    - {at: 630, fault: heal, target: 1}
    - {at: 700, fault: throttle, target: 0, netem: "rate 10mbit delay 50ms"}
    - {at: 760, fault: unthrottle, target: 0}

`at` is seconds from controller start. Run it when the training run is already warm:

  python -m chaos.controller --schedule configs/chaos/m2_kill_rejoin.yaml \
      --run-dir experiments/m2-kill-rejoin
"""

import argparse
import json
import time
from pathlib import Path

import yaml

from . import faults

FAULTS_NEEDING_RUN_DIR = {"kill", "stop", "cont"}


def execute(schedule: dict, run_dir: Path, dry_run: bool = False) -> None:
    events = sorted(schedule["events"], key=lambda e: e["at"])
    launch_cmd = schedule.get("launch_cmd", "")
    log_path = run_dir / "chaos.jsonl"
    t0_mono = time.monotonic()
    t0_wall = time.time()

    def log(rec: dict) -> None:
        with open(log_path, "a", buffering=1) as f:
            f.write(json.dumps(rec) + "\n")

    log({"event": "chaos_start", "ts": t0_wall, "mono": t0_mono, "schedule": events})
    print(f"chaos: {len(events)} events over {events[-1]['at']}s", flush=True)

    for ev in events:
        delay = ev["at"] - (time.monotonic() - t0_mono)
        if delay > 0:
            time.sleep(delay)
        fault = ev["fault"]
        target = ev.get("target")
        rec = {
            "event": "fault",
            "fault": fault,
            "target": target,
            "at": ev["at"],
            "ts": time.time(),
            "mono": time.monotonic(),
        }
        try:
            if dry_run:
                rec["result"] = {"dry_run": True}
            elif fault in FAULTS_NEEDING_RUN_DIR:
                rec["result"] = getattr(faults, fault)(run_dir, target)
            elif fault == "relaunch":
                rec["result"] = faults.relaunch(launch_cmd, target)
            elif fault == "throttle":
                rec["result"] = faults.throttle(target, ev["netem"])
            elif fault in ("partition", "heal", "unthrottle"):
                rec["result"] = getattr(faults, fault)(target)
            else:
                raise ValueError(f"unknown fault {fault}")
            rec["ok"] = True
        except Exception as e:  # log and continue — chaos must not stop on a miss
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        log(rec)
        print(f"[{ev['at']:>5}s] {fault} target={target} ok={rec['ok']}", flush=True)

    log({"event": "chaos_end", "ts": time.time(), "mono": time.monotonic()})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    schedule = yaml.safe_load(Path(args.schedule).read_text())
    execute(schedule, Path(args.run_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
