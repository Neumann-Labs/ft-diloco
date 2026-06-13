"""Emit theme-adaptive inline SVG charts for the ft-diloco blog post. Colors are the
blog's CSS variables (with light-mode hex fallbacks) so figures flip with the site's
light/dark toggle; `currentColor` (axes/labels) tracks body text. Numbers come from the
SAME loaders the matplotlib plots use, so every chart is ground-truth, not eyeballed.

  python analysis/svg.py --fig 4  --out fig4.svg     # comm volume vs H
  python analysis/svg.py --fig 5  --out fig5.svg     # storm step-efficiency
  python analysis/svg.py --fig 6  --out fig6.svg     # eval regression -> fix
  python analysis/svg.py --fig 7  --out fig7.svg     # WAN throughput sweep
  python analysis/svg.py --fig 10 --out fig10.svg    # liveness vs participation
  python analysis/svg.py --fig 11 --out fig11.svg    # recovery-latency CDF
"""

import argparse
import math
import statistics as st
from pathlib import Path

from parse_logs import fuse, load_jsonl
from plot_comm import run_stats
from plot_storm import chaos_window, committed_rate
from plot_wan import steady_tokens_per_sec
from storm_events import build, state

EXP = Path("experiments")
PRIMARY = "var(--primary,#94452b)"
ERROR = "var(--error,#a64542)"
PRIMARY_C = "var(--primary-container,#fceee9)"
SURF = "var(--surface-container,#f3f0eb)"
AX = "currentColor"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Chart:
    """Minimal SVG chart canvas: data->pixel mapping + axis/label/shape primitives."""

    def __init__(self, w=720, h=380, ml=64, mr=22, mt=34, mb=52):
        self.w, self.h = w, h
        self.x0, self.x1 = ml, w - mr
        self.y0, self.y1 = mt, h - mb  # y0 top, y1 bottom (pixels)
        self.els = []
        self.xlog = self.ylog = False
        self.xdom = (0, 1)
        self.ydom = (0, 1)

    # --- scales ---
    def setx(self, lo, hi, log=False):
        self.xlog, self.xdom = log, (lo, hi)

    def sety(self, lo, hi, log=False):
        self.ylog, self.ydom = log, (lo, hi)

    def px(self, x):
        lo, hi = self.xdom
        if self.xlog:
            x, lo, hi = math.log10(x), math.log10(lo), math.log10(hi)
        return self.x0 + (x - lo) / (hi - lo) * (self.x1 - self.x0)

    def py(self, y):
        lo, hi = self.ydom
        if self.ylog:
            y, lo, hi = math.log10(max(y, 1e-9)), math.log10(lo), math.log10(hi)
        return self.y1 - (y - lo) / (hi - lo) * (self.y1 - self.y0)

    # --- primitives ---
    def line(self, x1, y1, x2, y2, color=AX, w=1.0, op=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.els.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{color}" stroke-width="{w}" opacity="{op}"{d}/>')

    def rect(self, x, y, w, h, fill, op=1.0, rx=2):
        self.els.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                        f'rx="{rx}" fill="{fill}" opacity="{op}"/>')

    def poly(self, pts, color, w=2.0, fill="none", op=1.0):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.els.append(f'<polyline points="{p}" fill="{fill}" stroke="{color}" '
                        f'stroke-width="{w}" opacity="{op}" stroke-linejoin="round" stroke-linecap="round"/>')

    def area(self, pts, fill, op=0.2):
        if not pts:
            return
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        p = f"{pts[0][0]:.1f},{self.y1:.1f} " + p + f" {pts[-1][0]:.1f},{self.y1:.1f}"
        self.els.append(f'<polygon points="{p}" fill="{fill}" opacity="{op}"/>')

    def dot(self, x, y, color, r=3.2):
        self.els.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>')

    def text(self, x, y, s, size=12, color=AX, anchor="middle", op=1.0, weight=400, italic=False):
        st_ = ' font-style="italic"' if italic else ""
        self.els.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                        f'text-anchor="{anchor}" opacity="{op}" font-weight="{weight}"{st_}>{esc(s)}</text>')

    # --- axes ---
    def yticks(self, vals, fmt=lambda v: f"{v:g}", grid=True, label=None):
        for v in vals:
            yp = self.py(v)
            if grid:
                self.line(self.x0, yp, self.x1, yp, AX, 0.8, 0.12)
            self.text(self.x0 - 8, yp + 4, fmt(v), 11, AX, "end", 0.65)
        if label:
            self.text(16, (self.y0 + self.y1) / 2, label, 12, AX, "middle", 0.8,
                      )  # rotated below
            self.els[-1] = self.els[-1].replace("<text ",
                f'<text transform="rotate(-90 16 {(self.y0+self.y1)/2:.0f})" ', 1)

    def xticklabels(self, xs, labels):
        for x, lab in zip(xs, labels):
            self.text(self.px(x), self.y1 + 18, lab, 11, AX, "middle", 0.7)

    def xlabel(self, s):
        self.text((self.x0 + self.x1) / 2, self.h - 10, s, 12, AX, "middle", 0.8)

    def title(self, s):
        self.text(self.x0, 18, s, 13, AX, "start", 1.0, 600)

    def svg(self):
        body = "\n  ".join(self.els)
        return (f'<svg viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg" '
                f'style="font-family:\'Inter\',sans-serif; max-width:100%; height:auto;">\n  '
                f'{body}\n</svg>\n')


# ----------------------------- figures -----------------------------

def fig4_comm():
    hs = [25, 50, 100, 200, 500]
    stats = [run_stats(EXP / f"m1-h{h}-s1337") for h in hs]
    c = Chart(760, 400, mb=58)
    c.title("Communication volume per replica — DiLoCo vs syncing every step")
    ymax = max(s["analytic_ddp"] for s in stats) / 1e9
    c.setx(0, len(hs), False); c.sety(0.8, ymax * 1.6, log=True)
    c.yticks([1, 3, 10, 30, 100, 300], lambda v: f"{v:g} GB", label="GB / replica (whole run)")
    gw = (c.x1 - c.x0) / len(hs)
    bw = gw / 4
    series = [("analytic_ddp", ERROR, "sync every step (DDP)"),
              ("analytic_diloco", PRIMARY, "DiLoCo"),
              ("measured", AX, "DiLoCo measured (veth)")]
    for i, s in enumerate(stats):
        cx = c.x0 + gw * i + gw / 2
        for j, (key, col, _) in enumerate(series):
            v = s[key]
            if not v:
                continue
            gb = v / 1e9
            x = cx + (j - 1) * bw - bw / 2
            op = 0.85 if col == AX else 1.0
            c.rect(x, c.py(gb), bw, c.y1 - c.py(gb), col, op)
        red = s["analytic_ddp"] / max(s["analytic_diloco"], 1)
        c.text(cx, c.py(s["analytic_diloco"] / 1e9) - 6, f"{red:.0f}×", 11, PRIMARY, "middle", 1, 600)
    c.xticklabels([i + 0.5 for i in range(len(hs))], [f"H={h}" for h in hs])
    c.xlabel("inner steps between syncs (H)")
    # legend
    lx = c.x0 + 8
    for key, col, lab in series:
        c.rect(lx, c.y0 + 2, 11, 11, col, 0.9 if col != AX else 0.7, 2)
        c.text(lx + 16, c.y0 + 11, lab, 10.5, AX, "start", 0.8)
        lx += 16 + len(lab) * 6.6 + 18
    return c.svg()


def fig5_efficiency():
    ref, _ = committed_rate(EXP / "m1-h50-s1337", 50)
    storms = [("k120", "m3-storm-k120", "69 faults/hr"), ("k60", "m3-storm-k60", "85 faults/hr")]
    effs = []
    for _, run, _ in storms:
        r, _ = committed_rate(EXP / run, 50, chaos_window(EXP / run))
        effs.append(r / ref * 100)
    c = Chart(640, 400, mb=58)
    c.title("Step efficiency under failure storms (2 replicas, ~45 min each)")
    c.setx(0, 2); c.sety(0, 100)
    c.yticks([0, 20, 40, 60, 80, 100], lambda v: f"{v:g}%", label="% of fault-free throughput")
    # reference lines
    c.line(c.x0, c.py(100), c.x1, c.py(100), AX, 1, 0.3, "4 3")
    c.text(c.x1 - 4, c.py(100) - 5, "fault-free = 100%", 10, AX, "end", 0.55)
    c.line(c.x0, c.py(82.3), c.x1, c.py(82.3), PRIMARY, 1.2, 0.7, "5 3")
    gw = (c.x1 - c.x0) / 2
    for i, ((lab, _, rate), eff) in enumerate(zip(storms, effs)):
        cx = c.x0 + gw * i + gw / 2
        bw = gw * 0.42
        c.rect(cx - bw / 2, c.py(eff), bw, c.y1 - c.py(eff), PRIMARY)
        c.text(cx, c.py(eff) - 7, f"{eff:.1f}%", 14, PRIMARY, "middle", 1, 600)
        c.text(cx, c.y1 + 18, f"storm {lab}", 11.5, AX, "middle", 0.85, 500)
        c.text(cx, c.y1 + 33, rate, 10, AX, "middle", 0.6)
    # torchft reference label centered in the gap between bars (drawn on top, clear of bars)
    c.text((c.x0 + c.x1) / 2, c.py(82.3) - 6, "torchft: 82.3%", 10.5, PRIMARY, "middle", 1, 600)
    c.text((c.x0 + c.x1) / 2, c.py(82.3) + 12, "(Llama-3 1B, 300 GPUs)", 9, PRIMARY, "middle", 0.8)
    return c.svg()


def _eval_series(run):
    xs = sorted((e["ts"], e["eval_loss"]) for f in (EXP / run).glob("replica*.jsonl")
                for e in load_jsonl(f) if e.get("event") == "eval" and "eval_loss" in e)
    if not xs:
        return [], []
    t0 = xs[0][0]
    return [(t - t0) / 60 for t, _ in xs], [l for _, l in xs]


def fig6_regression():
    nx, ny = _eval_series("m3-storm-k120-nockpt")
    fx, fy = _eval_series("m3-storm-k120")
    # drop the step-0 random-init eval so the descent dominates the y-range
    def trim(xs, ys):
        return zip(*[(x, y) for x, y in zip(xs, ys) if y < 6]) if any(y < 6 for y in ys) else (xs, ys)
    nx, ny = map(list, trim(nx, ny)); fx, fy = map(list, trim(fx, fy))
    c = Chart(760, 400, mb=54)
    c.title("Throughput looked healthy while the model rotted — and the fix")
    xmax = max(max(nx, default=1), max(fx, default=1))
    c.setx(0, xmax); c.sety(1.5, max(max(ny, default=4), max(fy, default=4)) * 1.05)
    c.yticks([2, 3, 4, 5], lambda v: f"{v:g}", label="global eval loss")
    c.xticklabels([0, xmax / 2, xmax], [f"{v:.0f}" for v in (0, xmax / 2, xmax)])
    c.xlabel("storm time (min)")
    c.poly([(c.px(x), c.py(y)) for x, y in zip(nx, ny)], ERROR, 2.4)
    c.poly([(c.px(x), c.py(y)) for x, y in zip(fx, fy)], PRIMARY, 2.4)
    if ny:
        c.text(c.px(nx[-1]), c.py(ny[-1]) - 8, "no checkpoints: regresses", 11, ERROR, "end", 1, 600)
    if fy:
        c.text(c.px(fx[-1]), c.py(fy[-1]) + 16, "commit-coupled checkpoints: holds", 11, PRIMARY, "end", 1, 600)
    return c.svg()


def fig7_wan():
    pts = {1: [], 100: []}
    for rate in (1000, 100, 50, 10):
        for h in (1, 100):
            run = EXP / f"m4-wan-{rate}mbit-h{h}"
            if run.exists():
                pts[h].append((rate, steady_tokens_per_sec(run) / 1000))
    c = Chart(760, 410, mb=56)
    c.title("Throughput vs link speed — DiLoCo holds, per-step sync collapses")
    c.setx(8, 1200, log=True); c.sety(0, 40)
    c.yticks([0, 10, 20, 30, 40], lambda v: f"{v:g}k", label="aggregate throughput (k tok/s)")
    c.xticklabels([10, 50, 100, 1000], ["10", "50", "100", "1000"])
    c.xlabel("link bandwidth (Mbps), 20 ms RTT")
    # starvation zone at 10 Mbps
    c.rect(c.x0, c.y0, c.px(14) - c.x0, c.y1 - c.y0, ERROR, 0.07, 0)
    c.text(c.px(10), c.y0 + 14, "control-plane", 9.5, ERROR, "middle", 0.8)
    c.text(c.px(10), c.y0 + 26, "starvation", 9.5, ERROR, "middle", 0.8)
    for h, col, lab in [(100, PRIMARY, "DiLoCo (H=100)"), (1, ERROR, "sync every step (H=1)")]:
        ps = sorted(pts[h])
        line = [(c.px(r), c.py(t)) for r, t in ps if t > 0]
        c.poly(line, col, 2.4)
        for r, t in ps:
            if t > 0:
                c.dot(c.px(r), c.py(t), col)
            else:
                c.text(c.px(r), c.py(2.4), "DNF", 10, col, "middle", 1, 600)
    # legend
    c.rect(c.x1 - 200, c.y0 + 4, 11, 11, PRIMARY, 1, 2); c.text(c.x1 - 184, c.y0 + 13, "DiLoCo (H=100)", 10.5, AX, "start", 0.85)
    c.rect(c.x1 - 200, c.y0 + 22, 11, 11, ERROR, 1, 2); c.text(c.x1 - 184, c.y0 + 31, "sync every step", 10.5, AX, "start", 0.85)
    return c.svg()


def fig10_liveness():
    n = 32
    starts, commits, kills, stops, parts, quorum, _ = build(EXP / "storm-n32p", n)
    win = chaos_window(EXP / "storm-n32p")
    t0, t1 = win
    c = Chart(820, 380, mb=52)
    c.title("Liveness vs per-sync participation under 125 faults/hr")
    c.setx(0, (t1 - t0) / 60); c.sety(0, n + 1)
    c.yticks([0, 8, 16, 24, 32], lambda v: f"{v:g}", label="replicas")
    c.xticklabels([0, 15, 30], ["0", "15", "30"]); c.xlabel("storm time (min)")
    # cluster ceiling
    c.line(c.x0, c.py(n), c.x1, c.py(n), AX, 1, 0.3, "3 3")
    c.text(c.x0 + 4, c.py(n) - 5, "cluster size N=32", 10, AX, "start", 0.55)
    # liveness area (sampled)
    grid = [t0 + (t1 - t0) * i / 120 for i in range(121)]
    live = [(c.px((t - t0) / 60), c.py(sum(1 for r in range(n)
            if state(r, t, starts, commits, kills, stops, parts) in ("training", "commit", "recover"))))
            for t in grid]
    c.area(live, PRIMARY, 0.18)
    c.poly(live, PRIMARY, 1.0, op=0.45)
    # participation line
    q = [(c.px((t - t0) / 60), c.py(p)) for t, p in quorum if t0 <= t <= t1]
    c.poly(q, PRIMARY, 2.2)
    # kill rug
    for r in range(n):
        for k in kills[r]:
            if t0 <= k <= t1:
                xk = c.px((k - t0) / 60)
                c.line(xk, c.y1, xk, c.y1 - 7, ERROR, 1, 0.6)
    c.text(c.x1 - 4, c.py(30) - 4, "alive (training / healing)", 10.5, PRIMARY, "end", 0.75)
    c.text(c.x1 - 4, c.py(16) - 6, "in each sync's quorum", 10.5, PRIMARY, "end", 1, 600)
    c.text(c.x0 + 4, c.y1 - 12, "red ticks = kills", 9.5, ERROR, "start", 0.8)
    return c.svg()


def fig11_cdf():
    d = fuse(EXP / "storm-n32p")
    tb = sorted(f["t_back_s"] for f in d["faults"] if f["fault"] == "kill_safe" and "t_back_s" in f)
    tr = sorted(f["t_resume_s"] for f in d["faults"] if "t_resume_s" in f)
    c = Chart(680, 380, mb=52)
    c.title(f"Recovery latency across all {len(tb)} kills (CDF)")
    xmax = max(tb[-1], tr[-1]) * 1.05
    c.setx(0, xmax); c.sety(0, 100)
    c.yticks([0, 25, 50, 75, 100], lambda v: f"{v:g}%", label="% of kills ≤ t")
    c.xticklabels([0, xmax / 2, xmax], [f"{v:.0f}" for v in (0, xmax / 2, xmax)])
    c.xlabel("seconds after kill")
    for series, col, lab in [(tr, ERROR, f"T_resume (survivor commits), median {int(st.median(tr))}s"),
                             (tb, PRIMARY, f"T_back (killed worker fully back), median {int(st.median(tb))}s")]:
        pts = [(c.px(0), c.py(0))]
        for i, v in enumerate(series):
            y = (i + 1) / len(series) * 100
            pts.append((c.px(v), c.py(i / len(series) * 100)))
            pts.append((c.px(v), c.py(y)))
        c.poly(pts, col, 2.2)
    ly = c.y0 + 6
    for col, lab in [(ERROR, f"T_resume — median {int(st.median(tr))}s"),
                     (PRIMARY, f"T_back — median {int(st.median(tb))}s")]:
        c.rect(c.x0 + 10, ly, 11, 11, col, 1, 2); c.text(c.x0 + 26, ly + 10, lab, 10.5, AX, "start", 0.85); ly += 18
    return c.svg()


FIGS = {4: fig4_comm, 5: fig5_efficiency, 6: fig6_regression, 7: fig7_wan,
        10: fig10_liveness, 11: fig11_cdf}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fig", type=int, required=True, choices=sorted(FIGS))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    Path(args.out).write_text(FIGS[args.fig]())
    print(f"wrote {args.out} (fig {args.fig})")


if __name__ == "__main__":
    main()
