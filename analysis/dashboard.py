"""Live training dashboard (Rich + plotext) for demos and storm-watching.

  python analysis/dashboard.py --run experiments/m2-gif            # live, 1s refresh
  python analysis/dashboard.py --run experiments/m2-gif --once     # single frame (dev)

Layout: a big shared loss-vs-time chart (every worker as a braille line, fault events
as vertical markers) flanked by per-worker status panels, a cluster summary, and a
chaos event feed. Reads the same JSONL telemetry as every other analysis tool.
"""

import argparse
import json
import time
from pathlib import Path

import plotext as plt
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

WORKER_COLORS = ["cyan", "orange1", "green", "magenta"]
PLT_COLORS = ["cyan", "orange", "green", "magenta"]
FAULT_GLYPH = {
    "kill": ("skull: kill -9", "red"),
    "relaunch": ("relaunch", "green"),
    "partition": ("partition (link down)", "orange1"),
    "heal": ("heal (link up)", "cyan"),
    "stop": ("SIGSTOP", "magenta"),
    "cont": ("SIGCONT", "yellow"),
}


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


def read_run(run_dir: Path) -> tuple[dict[int, list[dict]], list[dict]]:
    replicas = {}
    for f in sorted(run_dir.glob("replica*.jsonl")):
        rid = int(f.stem.replace("replica", ""))
        replicas[rid] = load_jsonl(f)
    chaos = [
        e for e in load_jsonl(run_dir / "chaos.jsonl") if e.get("event") == "fault" and e.get("ok")
    ]
    return replicas, chaos


def worker_status(evs: list[dict], now: float) -> tuple[str, str]:
    steps = [e for e in evs if e["event"] == "step"]
    starts = [e for e in evs if e.get("phase") == "start"]
    if not steps:
        return "STARTING", "yellow"
    age = now - steps[-1]["ts"]
    recent_start = starts and (now - starts[-1]["ts"]) < 150
    if age < 15:
        return "TRAINING", "green3"
    if recent_start:
        return "RECOVERING", "yellow"
    return "DEAD", "red"


def chart_ansi(replicas, chaos, t0, width, height) -> str:
    plt.clf()
    plt.plotsize(max(40, width), max(10, height))
    plt.canvas_color("default")
    plt.axes_color("default")
    plt.ticks_color("grey")
    ymax = 0.0
    for rid, evs in sorted(replicas.items()):
        steps = [e for e in evs if e["event"] == "step"]
        if not steps:
            continue
        # downsample to keep redraws cheap
        stride = max(1, len(steps) // 300)
        steps = steps[::stride]
        xs = [(e["ts"] - t0) / 60 for e in steps]
        ys = [e["loss"] for e in steps]
        ymax = max(ymax, max(ys))
        plt.plot(xs, ys, color=PLT_COLORS[rid % len(PLT_COLORS)], label=f"worker {rid}")
    for c in chaos:
        color = "red" if c["fault"] == "kill" else "gray"
        plt.vline((c["ts"] - t0) / 60, color=color)
    plt.xlabel("minutes")
    plt.title("training loss")
    return plt.build()


def worker_panel(rid: int, evs: list[dict], now: float) -> Panel:
    status, scolor = worker_status(evs, now)
    steps = [e for e in evs if e["event"] == "step"]
    syncs = [e for e in evs if e["event"] == "outer_sync"]
    starts = [e for e in evs if e.get("phase") == "start"]
    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", style="dim")
    t.add_column()
    if steps:
        s = steps[-1]
        t.add_row("loss", f"[bold]{s['loss']:.3f}[/]")
        t.add_row("step", f"{s['step']}  ({s['tokens'] / 1e6:.1f}M tok)")
    if syncs:
        y = syncs[-1]
        t.add_row("sync", f"outer {y['outer_step']}  x{y['num_participants']}")
    if starts:
        t.add_row("pid", f"{starts[-1]['pid']}  (start #{len(starts)})")
    title = Text.assemble((f"worker {rid} ", f"bold {WORKER_COLORS[rid % len(WORKER_COLORS)]}"))
    badge = Text(f" {status} ", style=f"bold black on {scolor}")
    return Panel(t, title=title, subtitle=badge, border_style=WORKER_COLORS[rid % len(WORKER_COLORS)])


def cluster_panel(replicas, chaos, now: float) -> Panel:
    total = 0
    parts = "?"
    latest_ts = 0.0
    for evs in replicas.values():
        syncs = [e for e in evs if e["event"] == "outer_sync"]
        total += sum(1 for e in syncs if e["committed"])
        if syncs and syncs[-1]["ts"] > latest_ts:
            latest_ts = syncs[-1]["ts"]
            parts = syncs[-1]["num_participants"]
    body = [Text.assemble("participants ", (f"{parts}", "bold green3"),
                          "    committed syncs ", (f"{total}", "bold"))]
    for c in chaos[-5:]:
        label, color = FAULT_GLYPH.get(c["fault"], (c["fault"], "white"))
        ago = now - c["ts"]
        body.append(Text.assemble(("  ⚡ ", color), (f"{label} → worker {c['target']}", color),
                                  (f"   {ago:.0f}s ago", "dim")))
    return Panel(Group(*body), title="cluster", border_style="white")


def build_frame(run_dir: Path, console: Console) -> Layout:
    replicas, chaos = read_run(run_dir)
    now = time.time()
    all_steps = [e for evs in replicas.values() for e in evs if e["event"] == "step"]
    t0 = min((e["ts"] for e in all_steps), default=now)

    w, h = console.size
    side_w = 34
    chart = Text.from_ansi(chart_ansi(replicas, chaos, t0, w - side_w - 4, h - 2))

    layout = Layout()
    layout.split_row(Layout(name="chart"), Layout(name="side", size=side_w))
    layout["chart"].update(
        Panel(chart, title=f"[bold]ft-diloco — {run_dir.name}[/]", border_style="bright_black")
    )
    side_items = [worker_panel(rid, evs, now) for rid, evs in sorted(replicas.items())]
    side_items.append(cluster_panel(replicas, chaos, now))
    layout["side"].update(Group(*side_items))
    return layout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    run_dir = Path(args.run)
    console = Console()

    if args.once:
        console.print(build_frame(run_dir, console))
        return
    with Live(console=console, screen=True, auto_refresh=False) as live:
        while True:
            live.update(build_frame(run_dir, console), refresh=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
