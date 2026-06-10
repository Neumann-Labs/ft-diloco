"""Communication-volume plot: measured veth bytes vs analytic DiLoCo vs analytic DDP.

Usage:
    python analysis/plot_comm.py --runs experiments/m1-h25-s1337 experiments/m1-h100-s1337 ... \
        --out plots/m1_comm.png

Per run:
  analytic DiLoCo bytes/replica  = payload * 2(N-1)/N * num_committed_syncs
  analytic DDP bytes/replica     = payload * 2(N-1)/N * num_inner_steps  (what per-step
                                   sync of the same model would have cost; not a real run)
  measured bytes/replica         = sum over replicas of (rx+tx)/2 deltas on vftd<i>
                                   from netmon.jsonl, averaged per replica
payload = fp32 param bytes, logged by ft.py as bytes_analytic on outer_sync events.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run_stats(run_dir: Path) -> dict:
    replicas = sorted(run_dir.glob("replica*.jsonl"))
    events = [e for f in replicas for e in load_jsonl(f)]
    syncs = [e for e in events if e["event"] == "outer_sync"]
    steps = [e for e in events if e["event"] == "step"]
    start = next(e for e in events if e["event"] == "lifecycle" and e.get("phase") == "start")
    cfg = start["config"]
    n_rep = max(e["replica_id"] for e in events) + 1
    payload = syncs[0]["bytes_analytic"] if syncs else 4 * 0
    ring = 2 * (n_rep - 1) / n_rep if n_rep > 1 else 0.0
    committed = {}
    inner_steps = {}
    for e in syncs:
        if e.get("committed"):
            committed[(e["replica_id"], e["outer_step"])] = 1
    for e in steps:
        inner_steps[e["replica_id"]] = max(inner_steps.get(e["replica_id"], 0), e["step"])
    syncs_per_replica = len(committed) / max(n_rep, 1)
    steps_per_replica = sum(inner_steps.values()) / max(len(inner_steps), 1)

    measured = None
    nm = run_dir / "netmon.jsonl"
    if nm.exists():
        recs = load_jsonl(nm)
        per_iface = {}
        for iface in [k for k in recs[0] if k.startswith("vftd")]:
            vals = [r[iface] for r in recs if r.get(iface)]
            if len(vals) >= 2:
                per_iface[iface] = (
                    vals[-1]["rx_bytes"] - vals[0]["rx_bytes"] + vals[-1]["tx_bytes"] - vals[0]["tx_bytes"]
                ) / 2  # rx+tx double-counts each transferred byte on a veth pair end
        if per_iface:
            measured = sum(per_iface.values()) / len(per_iface)

    return {
        "label": run_dir.name,
        "H": cfg.get("sync_every"),
        "n_replicas": n_rep,
        "analytic_diloco": payload * ring * syncs_per_replica,
        "analytic_ddp": payload * ring * steps_per_replica,
        "measured": measured,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    stats = sorted((run_stats(Path(r)) for r in args.runs), key=lambda s: s["H"] or 0)
    hs = [s["H"] for s in stats]
    x = np.arange(len(stats))
    w = 0.28

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, [s["analytic_ddp"] / 1e9 for s in stats], w, label="DDP (analytic, same model)")
    ax.bar(x, [s["analytic_diloco"] / 1e9 for s in stats], w, label="DiLoCo (analytic)")
    meas = [s["measured"] / 1e9 if s["measured"] else np.nan for s in stats]
    ax.bar(x + w, meas, w, label="DiLoCo (measured, veth)")
    for xi, s in zip(x, stats):
        red = s["analytic_ddp"] / max(s["analytic_diloco"], 1)
        ax.text(xi, s["analytic_diloco"] / 1e9, f"{red:.0f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"H={h}" for h in hs])
    ax.set_yscale("log")
    ax.set_ylabel("GB per replica (whole run)")
    ax.set_title("Sync communication volume: DiLoCo vs per-step DP")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for s in stats:
        print(s)


if __name__ == "__main__":
    main()
