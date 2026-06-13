"""Reconstruct an N-replica failure storm as an animated GIF, entirely from the run's
JSONL telemetry (chaos.jsonl faults + replica*.jsonl commits/lifecycle). No live
recording needed — the timestamps are the ground truth — so the time axis is ours to
compress (30 min -> ~25 s) and the encoding ours to choose. A grid of replica cells
(one per group) shows each replica's state; a top strip tracks quorum size, cumulative
commits, and the active fault. This is the scale analogue of the 2-worker demo.gif:
the close-up shows the recovery mechanism, this wide shot shows a swarm absorbing chaos.

  python analysis/storm_gif.py --run experiments/storm-n32p --n 32 --out plots/m5_storm_n32.gif \
      --seconds 26 --fps 18
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from storm_events import build, load_jsonl, state

# Two palettes so the GIF can match the blog's light/dark toggle (a light variant + a
# dark variant are rendered, and the post swaps them with the theme). Dark = GitHub-dark
# telemetry look; light = tuned for contrast on the site's warm near-white surface.
THEMES = {
    "dark": dict(
        bg="#0d1117", fg="#c9d1d9", grid="#30363d", ceil="#21262d", axtick="#6e7681",
        qfill="#1f6feb", qmark="#58a6ff", lscatter="#26492e", lline="#3fb950",
        mark="#f0f6fc", tick="#f85149", caption="#8b949e", edge="#0d1117",
        label_dark="#0d1117", label_light="#f0f6fc", label_dark_states={"commit", "training"},
        states={"training": "#3fb950", "commit": "#f0f6fc", "down": "#5a1e1e",
                "stopped": "#d29922", "partition": "#388bfd", "recover": "#a371f7"}),
    "light": dict(
        bg="#fbf7f1", fg="#1c1c19", grid="#ddd5c9", ceil="#cfc8bc", axtick="#9b958c",
        qfill="#b06a4e", qmark="#94452b", lscatter="#a9c6ad", lline="#2e7d3f",
        mark="#1c1c19", tick="#c0392b", caption="#6f6a62", edge="#fbf7f1",
        label_dark="#1c1c19", label_light="#ffffff", label_dark_states={"down"},
        states={"training": "#3a9d4e", "commit": "#1c1c19", "down": "#d8b3b3",
                "stopped": "#c6850f", "partition": "#2f6fd0", "recover": "#7a52c9"}),
}
T = THEMES["dark"]  # selected in main() via --theme


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=26.0)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--pulse", type=float, default=4.0, help="wall-seconds a commit flash lasts")
    p.add_argument("--theme", choices=("dark", "light"), default="dark")
    args = p.parse_args()
    global T
    T = THEMES[args.theme]
    run = Path(args.run)
    n = args.n
    starts, commits, kills, stops, parts, quorum, faults_tl = build(run, n)

    # time span: chaos window padded, fall back to commit span
    chaos = load_jsonl(run / "chaos.jsonl")
    cs = [e["ts"] for e in chaos if e.get("event") == "chaos_start"]
    ce = [e["ts"] for e in chaos if e.get("event") == "chaos_end"]
    all_c = [c for r in range(n) for c in commits[r]]
    t0 = (cs[0] - 15) if cs else (min(all_c) if all_c else 0)
    t1 = (ce[0] + 8) if ce else (max(all_c) if all_c else 1)
    nframes = int(args.seconds * args.fps)
    qts = [q[0] for q in quorum]
    qsz = [q[1] for q in quorum]

    # eval-loss series (pooled across replicas) + a binned-median trend for a clean curve
    ev = sorted((e["ts"], e["eval_loss"]) for f in run.glob("replica*.jsonl")
                for e in load_jsonl(f) if e.get("event") == "eval" and "eval_loss" in e
                and t0 - 1 <= e["ts"] <= t1 + 1)
    evt = [a for a, _ in ev]
    evl = [b for _, b in ev]
    trt, trl = [], []
    if evt:
        edges = np.linspace(t0, t1, 46)
        idx = np.digitize(evt, edges)
        for b in range(1, len(edges)):
            sel = [j for j in range(len(evt)) if idx[j] == b]
            if sel:
                trt.append(float(np.mean([evt[j] for j in sel])))
                trl.append(float(np.median([evl[j] for j in sel])))
    lo = (min(evl) - 0.4) if evl else 0
    hi = (max(evl) + 0.4) if evl else 1
    kill_ts = [k for r in range(n) for k in kills[r] if t0 <= k <= t1]

    cols = args.cols
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(9.6, 7.0), facecolor=T["bg"])
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.5], width_ratios=[1, 1],
                          hspace=0.32, wspace=0.18, left=0.07, right=0.96, top=0.86, bottom=0.13)
    axq = fig.add_subplot(gs[0, 0]); axl = fig.add_subplot(gs[0, 1]); axg = fig.add_subplot(gs[1, :])
    for ax in (axq, axl, axg):
        ax.set_facecolor(T["bg"])
        for s in ax.spines.values():
            s.set_visible(False)

    title = fig.text(0.07, 0.945, "", color=T["fg"], fontsize=12, ha="left", fontweight="bold")
    fig.text(0.07, 0.905, "Despite constant interruptions, the cluster never lost quorum "
             "and the loss kept falling.", color=T["caption"], fontsize=9.5, ha="left", style="italic")

    # quorum panel (faint full line + animated fill/marker + kill ticks)
    axq.plot(qts, qsz, color=T["grid"], lw=1.0)
    for k in kill_ts:
        axq.axvline(k, color=T["tick"], alpha=0.35, lw=0.7, ymax=0.08)
    axq.set_xlim(t0, t1); axq.set_ylim(0, n + 1); axq.set_xticks([])
    axq.set_ylabel("quorum  /32", color=T["fg"], fontsize=9)
    axq.tick_params(colors=T["axtick"], labelsize=7)
    axq.axhline(n, color=T["ceil"], lw=0.8, ls=":")
    fill = axq.fill_between([t0, t0], [0, 0], color=T["qfill"], alpha=0.25)
    qmark, = axq.plot([], [], "o", color=T["qmark"], ms=5)

    # loss panel (faint full scatter + animated descending trend + kill ticks)
    axl.scatter(evt, evl, s=6, color=T["lscatter"], alpha=0.7, edgecolors="none")
    for k in kill_ts:
        axl.axvline(k, color=T["tick"], alpha=0.3, lw=0.7, ymax=0.08)
    axl.set_xlim(t0, t1); axl.set_ylim(lo, hi); axl.set_xticks([])
    axl.set_ylabel("eval loss", color=T["fg"], fontsize=9)
    axl.tick_params(colors=T["axtick"], labelsize=7)
    lline, = axl.plot([], [], color=T["lline"], lw=2.2)
    lmark, = axl.plot([], [], "o", color=T["mark"], ms=5)

    # grid of replica cells
    axg.set_xlim(0, cols); axg.set_ylim(0, rows); axg.invert_yaxis()
    axg.set_xticks([]); axg.set_yticks([]); axg.set_aspect("equal")
    rects, labels = [], []
    for r in range(n):
        cx, cy = r % cols, r // cols
        rect = Rectangle((cx + 0.06, cy + 0.06), 0.88, 0.88, facecolor=T["states"]["down"],
                         edgecolor=T["edge"], lw=1.5)
        axg.add_patch(rect); rects.append(rect)
        labels.append(axg.text(cx + 0.5, cy + 0.5, str(r), ha="center", va="center",
                               color=T["label_dark"], fontsize=8, fontweight="bold"))
    ticker = axg.text(0.0, -0.26, "", color=T["tick"], fontsize=11,
                      fontweight="bold", va="bottom")
    leg = [("training", "training"), ("commit", "commit"), ("stopped", "straggler"),
           ("partition", "partition"), ("recover", "healing"), ("down", "down")]
    for i, (key, label) in enumerate(leg):
        fig.text(0.115 + i * 0.135, 0.035, "■ " + label, color=T["states"][key],
                 fontsize=8.5, ha="left", va="bottom", fontweight="bold")

    def update(i):
        t = t0 + (t1 - t0) * i / max(1, nframes - 1)
        ncommit = sum(1 for r in range(n) for c in commits[r] if c <= t)
        live_q = [s for s, _ in quorum if s <= t]
        cur_q = qsz[len(live_q) - 1] if live_q else 0
        for r in range(n):
            st = state(r, t, starts, commits, kills, stops, parts)
            if st == "training" and any(t - args.pulse <= c <= t for c in commits[r]):
                st = "commit"
            rects[r].set_facecolor(T["states"][st])
            labels[r].set_color(T["label_dark"] if st in T["label_dark_states"] else T["label_light"])
        nonlocal fill
        fill.remove()
        seg_t = [s for s in qts if s <= t] or [t0]
        seg_q = qsz[:len(seg_t)] or [0]
        fill = axq.fill_between(seg_t, seg_q, color=T["qfill"], alpha=0.22)
        qmark.set_data([t], [cur_q])
        k = len([x for x in trt if x <= t])
        lline.set_data(trt[:k], trl[:k])
        cur_l = trl[k - 1] if k else None
        lmark.set_data([t], [cur_l]) if cur_l is not None else lmark.set_data([], [])
        mins = (t - t0) / 60.0
        speed = (t1 - t0) / args.seconds
        lstr = f"loss {cur_l:.1f}" if cur_l is not None else "loss —"
        title.set_text(f"32-replica DiLoCo failure storm   ·   t+{mins:4.1f} min   ·   "
                       f"quorum {cur_q}/{n}   ·   {lstr}   ·   {speed:.0f}×")
        recent = [lab for (ts, lab) in faults_tl if t - args.pulse * 1.5 <= ts <= t]
        ticker.set_text("⚡ " + "   ".join(recent[-3:]) if recent else "")
        return rects + labels + [qmark, lline, lmark, title, ticker]

    anim = FuncAnimation(fig, update, frames=nframes, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}  ({nframes} frames, {args.seconds}s @ {args.fps}fps)")


if __name__ == "__main__":
    main()
