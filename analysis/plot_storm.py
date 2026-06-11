"""M3 storm analysis: step efficiency vs failure rate + recovery-latency CDF.

  python analysis/plot_storm.py --storms experiments/m3-storm-k120 experiments/m3-storm-k60 \
      --reference experiments/m1-h50-s1337 --h 50 --out plots/m3_storm.png

Step efficiency = (committed outer syncs x H) / wall-second within the chaos window,
relative to the fault-free reference run at the same config. The torchft Llama blog's
"82.3% step efficiency through ~1,100 failures" is the large-scale analogue.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from parse_logs import fuse, load_jsonl


def committed_rate(run_dir: Path, h: int, window: tuple[float, float] | None = None) -> tuple[float, int]:
    """(committed inner steps per second, committed syncs) within window."""
    syncs = []
    for f in run_dir.glob("replica*.jsonl"):
        for e in load_jsonl(f):
            if e["event"] == "outer_sync" and e["committed"]:
                syncs.append(e["ts"])
    if window:
        lo, hi = window
        syncs = [t for t in syncs if lo <= t <= hi]
        wall = hi - lo
    else:
        wall = max(syncs) - min(syncs)
    return len(syncs) * h / wall, len(syncs)


def chaos_window(run_dir: Path) -> tuple[float, float]:
    recs = load_jsonl(run_dir / "chaos.jsonl")
    t0 = next(r["ts"] for r in recs if r["event"] == "chaos_start")
    t1 = next(r["ts"] for r in recs if r["event"] == "chaos_end")
    return t0, t1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--storms", nargs="+", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--h", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ref_rate, _ = committed_rate(Path(args.reference), args.h)

    rows = [{"label": "fault-free", "rate_hr": 0.0, "eff": 1.0, "summary": None}]
    cdf_data = {}
    for s in args.storms:
        run = Path(s)
        win = chaos_window(run)
        rate, n = committed_rate(run, args.h, win)
        summary = fuse(run)
        kills = [f for f in summary["faults"] if f["fault"] in ("kill", "kill_safe") and "t_back_s" in f]
        raw = [e for e in __import__("parse_logs").load_jsonl(run / "chaos.jsonl")
               if e.get("event") == "fault" and e.get("ok")]
        n_faults = len([e for e in raw if not (isinstance(e.get("result"), dict) and "skipped" in e["result"])])
        wall_hr = (win[1] - win[0]) / 3600
        rows.append({
            "label": run.name.replace("m3-storm-", ""),
            "rate_hr": n_faults / wall_hr,
            "eff": rate / ref_rate,
            "summary": summary,
        })
        cdf_data[run.name] = {
            "t_back": sorted(f["t_back_s"] for f in kills),
            "t_resume": sorted(f["t_resume_s"] for f in summary["faults"] if "t_resume_s" in f),
        }
        print(f"{run.name}: {n_faults} faults, {n} committed syncs in window, "
              f"eff={rate / ref_rate:.1%}, kills with full recovery: {len(kills)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    xs = [r["rate_hr"] for r in rows]
    ys = [r["eff"] * 100 for r in rows]
    ax1.plot(xs, ys, "o-", color="tab:blue")
    for r in rows:
        ax1.annotate(r["label"], (r["rate_hr"], r["eff"] * 100),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax1.axhline(82.3, color="gray", linestyle=":", linewidth=1)
    ax1.text(0.02, 82.8, "torchft Llama-3 1B blog: 82.3% @ 1 failure/min (300 GPUs)",
             fontsize=7, color="gray", transform=ax1.get_yaxis_transform())
    ax1.set_xlabel("injected faults per hour")
    ax1.set_ylabel("step efficiency vs fault-free (%)")
    ax1.set_title("Training step efficiency under failure storms")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 105)

    colors = ["tab:red", "tab:orange", "tab:purple"]
    for i, (name, d) in enumerate(cdf_data.items()):
        for key, style in (("t_back", "-"), ("t_resume", "--")):
            vals = d[key]
            if not vals:
                continue
            ys_cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax2.step(vals, ys_cdf, style, color=colors[i % len(colors)],
                     label=f"{name.replace('m3-storm-', '')} {key}")
    ax2.set_xlabel("seconds after kill")
    ax2.set_ylabel("CDF")
    ax2.set_title("Recovery latency (t_resume: survivor commits; t_back: victim recovered)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
