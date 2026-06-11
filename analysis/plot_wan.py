"""M4 WAN plot: training throughput vs link bandwidth, DiLoCo H=100 vs per-step sync.

  python analysis/plot_wan.py --glob 'experiments/m4-wan-*' --out plots/m4_wan.png

Throughput = aggregate tokens/sec across both replicas, measured from step-event
timestamps after the first committed sync (steady state, excludes startup).
"""

import argparse
import glob as globmod
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parse_logs import load_jsonl


def steady_tokens_per_sec(run_dir: Path) -> float:
    total = 0.0
    for f in run_dir.glob("replica*.jsonl"):
        evs = load_jsonl(f)
        syncs = [e for e in evs if e["event"] == "outer_sync" and e["committed"]]
        steps = [e for e in evs if e["event"] == "step"]
        if not syncs or len(steps) < 3:
            continue
        t_start = syncs[0]["ts"]
        sel = [e for e in steps if e["ts"] >= t_start]
        if len(sel) < 2:
            sel = steps
        d_tok = sel[-1]["tokens"] - sel[0]["tokens"]
        d_t = sel[-1]["ts"] - sel[0]["ts"]
        if d_t > 0:
            total += d_tok / d_t
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="experiments/m4-wan-*")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    series: dict[int, list[tuple[int, float]]] = {}
    for d in sorted(globmod.glob(args.glob)):
        run = Path(d)
        m = re.match(r"m4-wan-(\d+)mbit-h(\d+)", run.name)
        if not m:
            continue
        rate, h = int(m.group(1)), int(m.group(2))
        tps = steady_tokens_per_sec(run)
        series.setdefault(h, []).append((rate, tps))
        print(f"{run.name}: {tps:,.0f} tok/s")

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {1: ("tab:red", "per-step sync (DDP comm pattern, H=1)"),
              100: ("tab:blue", "DiLoCo (H=100)")}
    for h, pts in sorted(series.items()):
        pts.sort()
        xs = [r for r, _ in pts]
        ys = [t / 1000 for _, t in pts]
        color, label = styles.get(h, ("gray", f"H={h}"))
        ax.plot(xs, ys, "o-", color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("link bandwidth (Mbps), 20 ms RTT")
    ax.set_ylabel("aggregate throughput (k tokens/s)")
    ax.set_title("Throughput vs link speed: DiLoCo holds, per-step sync collapses")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
