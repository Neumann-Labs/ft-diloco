"""Eval-loss convergence plot from run JSONL.

Usage:
    python analysis/plot_convergence.py --runs experiments/m0-tiny-s* --out plots/m0_tiny.png

Curves are grouped by run_id with a trailing `-s<seed>` stripped; groups with
multiple seeds get a min/max band around the mean. X axis: tokens (default) or hours.
In diloco runs, per-replica token counts are summed at matching eval steps so the
x-axis is total tokens across the cluster.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_evals(run_dir: Path) -> list[dict]:
    events = []
    for f in sorted(run_dir.glob("replica*.jsonl")):
        for line in f.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "eval":
                events.append(rec)
    return events


def run_curve(run_dir: Path, x_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """One (x, eval_loss) curve per run: replica-0 losses (identical post-sync for
    diloco), x = total tokens across replicas at that step, or wall hours."""
    events = load_evals(run_dir)
    if not events:
        raise SystemExit(f"no eval events in {run_dir}")
    by_step: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_step[e["step"]].append(e)
    t0 = min(e["ts"] for e in events)
    xs, ys = [], []
    for step in sorted(by_step):
        recs = by_step[step]
        r0 = min(recs, key=lambda r: r["replica_id"])
        if x_mode == "tokens":
            xs.append(sum(r["tokens"] for r in {r["replica_id"]: r for r in recs}.values()))
        else:
            xs.append((max(r["ts"] for r in recs) - t0) / 3600)
        ys.append(r0["eval_loss"])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--x", choices=["tokens", "hours"], default="tokens")
    p.add_argument("--title", default="Eval loss")
    args = p.parse_args()

    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for r in args.runs:
        run_dir = Path(r)
        label = re.sub(r"-s\d+$", "", run_dir.name)
        groups[label].append(run_curve(run_dir, args.x))

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, curves in sorted(groups.items()):
        if len(curves) == 1:
            x, y = curves[0]
            ax.plot(x, y, label=label)
        else:
            # interpolate every seed onto the first seed's x grid for the band
            x0 = curves[0][0]
            ys = np.stack([np.interp(x0, x, y) for x, y in curves])
            ax.plot(x0, ys.mean(0), label=f"{label} (n={len(curves)})")
            ax.fill_between(x0, ys.min(0), ys.max(0), alpha=0.2)
    ax.set_xlabel("total tokens" if args.x == "tokens" else "wall-clock (h)")
    ax.set_ylabel("eval loss")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
