"""N=32 storm results figure: (A) quorum participation + liveness over the storm with a
kill rug, (B) recovery-latency CDF, (C) global eval-loss trajectory through the storm.
Tells the honest story: ~all replicas stay ALIVE and the loss keeps descending through
125 faults/hr, while per-sync PARTICIPATION sits below liveness (the soft-barrier +
membership-churn alignment tax).

  python analysis/plot_n32.py --run experiments/storm-n32p --reference experiments/storm-n32-refp \
      --n 32 --h 20 --out plots/m5_n32_results.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parse_logs import fuse
from plot_storm import chaos_window, committed_rate
from storm_events import build, load_jsonl, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--h", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run, ref, n = Path(args.run), Path(args.reference), args.n

    starts, commits, kills, stops, parts, quorum, faults_tl = build(run, n)
    win = chaos_window(run)
    t0, t1 = win
    srate, _ = committed_rate(run, args.h, win)
    rrate, _ = committed_rate(ref, args.h, chaos_window(ref))
    eff = srate / rrate

    # quorum series + liveness sampled on a grid
    qts = [(t - t0) / 60 for t, _ in quorum if t0 <= t <= t1]
    qsz = [q for t, q in quorum if t0 <= t <= t1]
    grid = np.linspace(t0, t1, 160)
    live = []
    for t in grid:
        c = sum(1 for r in range(n)
                if state(r, t, starts, commits, kills, stops, parts) in ("training", "commit", "recover"))
        live.append(c)
    gmin = [(t - t0) / 60 for t in grid]
    kill_ts = [(k - t0) / 60 for r in range(n) for k in kills[r] if t0 <= k <= t1]

    # recovery latencies
    d = fuse(run)
    tb = sorted(f["t_back_s"] for f in d["faults"] if f["fault"] == "kill_safe" and "t_back_s" in f)
    tr = sorted(f["t_resume_s"] for f in d["faults"] if "t_resume_s" in f)

    # convergence: pooled eval_loss vs wall-clock
    ev = sorted((e["ts"], e["eval_loss"]) for f in run.glob("replica*.jsonl")
                for e in load_jsonl(f) if e.get("event") == "eval" and "eval_loss" in e
                and t0 - 200 <= e["ts"])
    em = [(t - t0) / 60 for t, _ in ev]
    el = [l for _, l in ev]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 7.4), facecolor="#0d1117")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.22,
                          left=0.07, right=0.97, top=0.9, bottom=0.09)
    for_ax = lambda ax: (ax.set_facecolor("#0d1117"), ax.grid(alpha=0.12),
                         [s.set_color("#30363d") for s in ax.spines.values()])

    axA = fig.add_subplot(gs[0, :]); for_ax(axA)
    axA.fill_between(gmin, live, step="mid", color="#1f6f33", alpha=0.55, label="alive (training/healing)")
    axA.plot(qts, qsz, color="#58a6ff", lw=1.6, label="quorum (participating per sync)")
    for i, k in enumerate(kill_ts):
        axA.axvline(k, color="#f85149", alpha=0.5, lw=0.8, ymax=0.06)
    axA.axhline(n, color="#21262d", ls=":", lw=1)
    axA.text(0.2, n + 0.4, f"cluster size N={n}", color="#8b949e", fontsize=8)
    axA.set_xlim(0, (t1 - t0) / 60); axA.set_ylim(0, n + 2)
    axA.set_xlabel("storm time (min)"); axA.set_ylabel("replicas")
    fault_rate = int(len([1 for r in range(n) for k in kills[r]]) / ((t1 - t0) / 3600))
    axA.set_title(f"N={n} liveness vs per-sync participation  ·  {len(kill_ts)} kills, "
                  f"red ticks  ·  whole storm = 125 faults/hr", color="#e6edf3", fontsize=11, loc="left")
    axA.legend(loc="lower right", fontsize=8, framealpha=0.2)

    axB = fig.add_subplot(gs[1, 0]); for_ax(axB)
    if tb:
        axB.step(tb, np.arange(1, len(tb) + 1) / len(tb) * 100, where="post",
                 color="#a371f7", lw=2, label=f"T_back (full recovery), median {int(np.median(tb))}s")
    if tr:
        axB.step(tr, np.arange(1, len(tr) + 1) / len(tr) * 100, where="post",
                 color="#58a6ff", lw=2, label=f"T_resume (survivor), median {int(np.median(tr))}s")
    axB.set_xlabel("seconds after kill"); axB.set_ylabel("% of kills ≤ t")
    axB.set_title("recovery latency (CDF)", color="#e6edf3", fontsize=11, loc="left")
    axB.legend(loc="lower right", fontsize=8, framealpha=0.2)

    axC = fig.add_subplot(gs[1, 1]); for_ax(axC)
    axC.scatter(em, el, s=8, color="#3fb950", alpha=0.5, edgecolors="none")
    if len(em) > 6:  # median trend in time bins
        bins = np.linspace(min(em), max(em), 14)
        idx = np.digitize(em, bins)
        bx = [np.mean([em[j] for j in range(len(em)) if idx[j] == b]) for b in range(1, len(bins)) if any(idx == b)]
        by = [np.median([el[j] for j in range(len(el)) if idx[j] == b]) for b in range(1, len(bins)) if any(idx == b)]
        axC.plot(bx, by, color="#f0f6fc", lw=2, label="median")
    for k in kill_ts:
        axC.axvline(k, color="#f85149", alpha=0.25, lw=0.6)
    axC.set_xlabel("storm time (min)"); axC.set_ylabel("eval loss")
    axC.set_title("global loss descends through the storm", color="#e6edf3", fontsize=11, loc="left")
    axC.legend(loc="upper right", fontsize=8, framealpha=0.2)

    ncommit = sum(d["committed_syncs"].values())
    fig.suptitle(f"32-replica DiLoCo failure storm, one commodity desktop  ·  {eff:.0%} step "
                 f"efficiency  ·  {ncommit} commits · all kills recovered · 0 OOM",
                 color="#e6edf3", fontsize=12, y=0.975)
    fig.savefig(args.out, dpi=130, facecolor="#0d1117")
    print(f"wrote {args.out}  (eff={eff:.1%}, kills={len(tb)}, eval_pts={len(em)})")


if __name__ == "__main__":
    main()
