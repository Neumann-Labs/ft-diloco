"""The money plot: per-replica training loss vs wall-clock with fault events overlaid.

  python analysis/plot_recovery.py --run experiments/m2-kill-rejoin --out plots/m2_recovery.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from parse_logs import fuse, load_jsonl  # same directory

FAULT_STYLE = {
    "kill": ("x", "tab:red", "kill -9"),
    "relaunch": ("^", "tab:green", "relaunch"),
    "partition": ("v", "tab:orange", "partition (link down)"),
    "heal": ("o", "tab:cyan", "heal (link up)"),
    "stop": ("s", "tab:purple", "SIGSTOP"),
    "cont": ("D", "tab:olive", "SIGCONT"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run_dir = Path(args.run)

    summary = fuse(run_dir)
    chaos = summary["faults"]
    t0 = None

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for f in sorted(run_dir.glob("replica*.jsonl")):
        rid = int(f.stem.replace("replica", ""))
        steps = [e for e in load_jsonl(f) if e["event"] == "step"]
        if not steps:
            continue
        t0 = min(t0 or steps[0]["ts"], steps[0]["ts"])
    for f in sorted(run_dir.glob("replica*.jsonl")):
        rid = int(f.stem.replace("replica", ""))
        steps = [e for e in load_jsonl(f) if e["event"] == "step"]
        if not steps:
            continue
        xs = [(e["ts"] - t0) / 60 for e in steps]
        ys = [e["loss"] for e in steps]
        # break the line across gaps (death periods) so the kill is visible
        gap_x, gap_y = [], []
        last = None
        for x, y in zip(xs, ys):
            if last is not None and (x - last) * 60 > 30:
                gap_x.append(float("nan"))
                gap_y.append(float("nan"))
            gap_x.append(x)
            gap_y.append(y)
            last = x
        ax.plot(gap_x, gap_y, label=f"worker {rid}", linewidth=1.2)

    seen = set()
    for fev in chaos:
        marker, color, label = FAULT_STYLE.get(fev["fault"], ("|", "k", fev["fault"]))
        x = (fev["ts"] - t0) / 60
        ax.axvline(x, color=color, alpha=0.35, linestyle="--", linewidth=1)
        lbl = label if label not in seen else None
        seen.add(label)
        ax.plot([x], [ax.get_ylim()[1] * 0.97], marker=marker, color=color, label=lbl, markersize=9, clip_on=False)

    lat = []
    for fev in chaos:
        for k in ("t_resume_s", "t_detect_s", "t_rejoin_s"):
            if k in fev:
                lat.append(f"{fev['fault']}@{fev['at']}s: {k.replace('_s', '')}={fev[k]:.1f}s")
    if lat:
        ax.text(
            0.99, 0.98, "\n".join(lat), transform=ax.transAxes, fontsize=8,
            va="top", ha="right", bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        )

    ax.set_xlabel("wall-clock (min)")
    ax.set_ylabel("training loss")
    ax.set_title(f"{run_dir.name}: training through real faults")
    ax.legend(loc="upper center", ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
