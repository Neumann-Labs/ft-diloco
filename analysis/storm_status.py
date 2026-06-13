"""Compact one-shot status of a running storm — aggregate, not per-node, so it can be
polled cheaply over SSH without streaming N worker logs. Reads replica*.jsonl +
chaos.jsonl and prints ~12 lines: liveness, quorum size, commit throughput, eval band,
and faults executed/skipped so far.

  python analysis/storm_status.py --run experiments/storm-n32 [--window 60]
"""

import argparse
import json
import time
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--window", type=float, default=60.0, help="recent-throughput window (s)")
    args = p.parse_args()
    run = Path(args.run)
    now = time.time()
    W = args.window

    reps = sorted(run.glob("replica*.jsonl"), key=lambda f: int(f.stem.replace("replica", "")))
    n = len(reps)
    total_syncs = total_commits = recent_commits = 0
    last_part = []          # most recent num_participants per replica
    last_seen = []          # seconds since each replica's last event (liveness)
    eval_losses = []
    last_outer = []
    for f in reps:
        rid = int(f.stem.replace("replica", ""))
        evs = load_jsonl(f)
        if not evs:
            continue
        syncs = [e for e in evs if e.get("event") == "outer_sync"]
        commits = [e for e in syncs if e.get("committed")]
        total_syncs += len(syncs)
        total_commits += len(commits)
        recent_commits += sum(1 for e in commits if now - e["ts"] <= W)
        if commits:
            last_part.append(commits[-1]["num_participants"])
            last_outer.append(commits[-1]["outer_step"])
        evals = [e for e in evs if e.get("event") == "eval"]
        if evals:
            eval_losses.append(evals[-1].get("loss"))
        last_seen.append(now - evs[-1]["ts"])

    alive = sum(1 for s in last_seen if s < 20)  # event within 20s ~ running (not killed/stopped)
    chaos = load_jsonl(run / "chaos.jsonl")
    faults = [e for e in chaos if e.get("event") == "fault"]
    executed = [e for e in faults if e.get("ok") and not (isinstance(e.get("result"), dict) and "skipped" in e["result"])]
    skipped = [e for e in faults if isinstance(e.get("result"), dict) and "skipped" in e["result"]]
    kinds: dict[str, int] = {}
    for e in executed:
        kinds[e["fault"]] = kinds.get(e["fault"], 0) + 1

    print(f"run={run.name}  replicas={n}  log-active(<20s)={alive}/{n}")
    if last_part:
        print(f"quorum size (last commit/replica): min={min(last_part)} max={max(last_part)} "
              f"median={sorted(last_part)[len(last_part)//2]}")
    if last_outer:
        print(f"outer_step: min={min(last_outer)} max={max(last_outer)} spread={max(last_outer)-min(last_outer)}")
    print(f"committed syncs: total={total_commits}  (of {total_syncs} attempted)  "
          f"last {int(W)}s={recent_commits}  ~{recent_commits/W:.2f}/s")
    if eval_losses:
        el = [x for x in eval_losses if x is not None]
        if el:
            print(f"eval loss (latest/replica): min={min(el):.3f} max={max(el):.3f}")
    print(f"faults: executed={len(executed)} {kinds}  skipped(no-healthy-donor)={len(skipped)}")
    if last_seen:
        stale = sum(1 for s in last_seen if s >= 20)
        print(f"stale/down replicas (no event >=20s): {stale}")


if __name__ == "__main__":
    main()
