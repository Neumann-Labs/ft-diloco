"""Fuse a chaos run's telemetry into one annotated timeline + recovery latencies.

Inputs per run dir: replica<k>.jsonl (training), chaos.jsonl (faults, ground truth),
worker<k>.log (torchft manager INFO lines, used for heal evidence).

Latency definitions (docs/architecture.md):
  T_resume  kill ts -> survivor's next committed outer_sync       (per kill)
  T_detect  kill ts -> survivor's first sync with fewer participants (upper bound:
            sync-boundary granularity; manager-internal detection is faster)
  T_rejoin  relaunch ts -> rejoined replica's first committed outer_sync

Usage:
  python analysis/parse_logs.py --run experiments/m2-kill-rejoin [--json out.json]
"""

import argparse
import json
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


def fuse(run_dir: Path) -> dict:
    replicas: dict[int, list[dict]] = {}
    for f in sorted(run_dir.glob("replica*.jsonl")):
        rid = int(f.stem.replace("replica", ""))
        replicas[rid] = load_jsonl(f)
    chaos = [e for e in load_jsonl(run_dir / "chaos.jsonl") if e.get("event") == "fault" and e.get("ok")]
    for c in chaos:
        if isinstance(c.get("result"), dict) and "skipped" in c["result"]:
            c["skipped"] = True

    syncs = {
        rid: sorted([e for e in evs if e["event"] == "outer_sync"], key=lambda e: e["ts"])
        for rid, evs in replicas.items()
    }

    starts = {
        rid: sorted([e["ts"] for e in evs if e.get("phase") == "start"])
        for rid, evs in replicas.items()
    }
    faults_out = []
    for f in chaos:
        rec = dict(fault=f["fault"], target=f["target"], ts=f["ts"], at=f["at"])
        if f["fault"] in ("kill", "kill_safe") and "skipped" not in (f.get("result") or {}):
            survivor = [r for r in syncs if r != f["target"]]
            if survivor:
                s = survivor[0]
                after = [e for e in syncs[s] if e["ts"] > f["ts"] and e["committed"]]
                if after:
                    rec["t_resume_s"] = after[0]["ts"] - f["ts"]
                pre = [e for e in syncs[s] if e["ts"] <= f["ts"]]
                pre_participants = pre[-1]["num_participants"] if pre else None
                drop = [
                    e for e in syncs[s]
                    if e["ts"] > f["ts"] and pre_participants and e["num_participants"] < pre_participants
                ]
                if drop:
                    rec["t_detect_s"] = drop[0]["ts"] - f["ts"]
            # supervisor-restart path (storms have no explicit relaunch events):
            t = f["target"]
            restart = [ts for ts in starts.get(t, []) if ts > f["ts"]]
            if restart:
                rec["t_restart_s"] = restart[0] - f["ts"]
                back = [e for e in syncs.get(t, []) if e["ts"] > restart[0] and e["committed"]]
                if back:
                    rec["t_back_s"] = back[0]["ts"] - f["ts"]
        elif f["fault"] == "relaunch":
            t = f["target"]
            after = [e for e in syncs.get(t, []) if e["ts"] > f["ts"] and e["committed"]]
            if after:
                rec["t_rejoin_s"] = after[0]["ts"] - f["ts"]
        faults_out.append(rec)

    # digest agreement after the last relaunch (momentum-recovery proof)
    digest_check = None
    relaunches = [f for f in chaos if f["fault"] == "relaunch"]
    if relaunches and len(replicas) >= 2:
        t_rej = relaunches[-1]["ts"]
        digs: dict[tuple, dict[int, str]] = {}
        for rid, evs in replicas.items():
            for e in evs:
                if e["event"] == "digest" and e["ts"] > t_rej:
                    digs.setdefault((e["outer_step"], e["kind"]), {})[rid] = e["sha256_16"]
        common = {k: v for k, v in digs.items() if len(v) >= 2}
        matches = sum(1 for v in common.values() if len(set(v.values())) == 1)
        digest_check = {"common_points": len(common), "matches": matches}

    return {
        "run": run_dir.name,
        "faults": faults_out,
        "digest_check": digest_check,
        "replicas": sorted(replicas),
        "total_syncs": {rid: len(s) for rid, s in syncs.items()},
        "committed_syncs": {rid: sum(1 for e in s if e["committed"]) for rid, s in syncs.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--json", help="write summary JSON here")
    args = p.parse_args()
    summary = fuse(Path(args.run))
    out = json.dumps(summary, indent=2)
    print(out)
    if args.json:
        Path(args.json).write_text(out)


if __name__ == "__main__":
    main()
