"""Re-derive every headline number the blog post will cite, straight from the committed
JSONL via the same loaders the plots use. Run before writing prose so nothing is eyeballed.

  python analysis/verify_numbers.py
"""

import statistics as st
from pathlib import Path

from parse_logs import fuse, load_jsonl
from plot_comm import run_stats
from plot_storm import chaos_window, committed_rate
from plot_wan import steady_tokens_per_sec

EXP = Path("experiments")


def final_eval(run: str) -> float | None:
    evs = [e for f in (EXP / run).glob("replica*.jsonl") for e in load_jsonl(f)
           if e.get("event") == "eval" and "eval_loss" in e]
    if not evs:
        return None
    evs.sort(key=lambda e: e["ts"])
    return evs[-1]["eval_loss"]


def eval_series(run: str) -> list[float]:
    evs = sorted((e["ts"], e["eval_loss"]) for f in (EXP / run).glob("replica*.jsonl")
                 for e in load_jsonl(f) if e.get("event") == "eval" and "eval_loss" in e)
    return [l for _, l in evs]


def digest_matches(run: str) -> str:
    # count post-sync digest points where all present replicas agree
    digs: dict[tuple, dict[int, str]] = {}
    for f in (EXP / run).glob("replica*.jsonl"):
        for e in load_jsonl(f):
            if e.get("event") == "digest":
                digs.setdefault((e["outer_step"], e["kind"]), {})[e["replica_id"]] = e["sha256_16"]
    common = {k: v for k, v in digs.items() if len(v) >= 2}
    matches = sum(1 for v in common.values() if len(set(v.values())) == 1)
    return f"{matches}/{len(common)}"


print("=" * 60, "\nM0 baseline (final eval loss, 3 seeds)")
m0 = [final_eval(f"m0-tiny-s{s}") for s in (1337, 1338, 1339)]
m0 = [x for x in m0 if x is not None]
print(f"  seeds={[round(x,4) for x in m0]}  mean={st.mean(m0):.4f}  std={st.pstdev(m0):.4f}")

print("=" * 60, "\nM0.5 + M2 recovery + digests")
print(f"  m05-rejoin digests (params+momentum): {digest_matches('m05-rejoin')}")
print(f"  m2-kill-rejoin digests: {digest_matches('m2-kill-rejoin')}")
d = fuse(EXP / "m2-kill-rejoin")
for fa in d["faults"]:
    if "t_resume_s" in fa or "t_rejoin_s" in fa:
        print(f"  m2 fault {fa['fault']} t_resume={fa.get('t_resume_s')} t_rejoin={fa.get('t_rejoin_s')}")

print("=" * 60, "\nM1 H-sweep (final eval loss + comm reduction)")
base = st.mean(m0)
for h in (25, 50, 100, 200, 500):
    run = f"m1-h{h}-s1337"
    el = final_eval(run)
    s = run_stats(EXP / run)
    red = s["analytic_ddp"] / max(s["analytic_diloco"], 1)
    meas = f"{s['measured']/1e9:.2f}GB" if s["measured"] else "n/a"
    delta = (el - base) / base * 100 if el else None
    print(f"  H={h:3d}: eval={el:.4f} (+{delta:.1f}%)  comm_reduction={red:.0f}x  "
          f"diloco={s['analytic_diloco']/1e9:.2f}GB measured={meas}")

print("=" * 60, "\nM3 storms (efficiency + the eval regression)")
for storm in ("m3-storm-k120", "m3-storm-k60"):
    win = chaos_window(EXP / storm)
    rate, n = committed_rate(EXP / storm, 50, win)
    print(f"  {storm}: committed_rate={rate:.1f} steps/s, {n} syncs in window")
for run in ("m3-storm-k120-nockpt", "m3-storm-k120"):
    s = eval_series(run)
    if s:
        print(f"  {run}: eval first={s[0]:.2f} max={max(s):.2f} last={s[-1]:.2f} "
              f"({'REGRESSED' if max(s) - s[-1] > 0.3 or s[-1] > s[0] else 'monotonic-ish'})")

print("=" * 60, "\nM4 WAN sweep (steady tok/s)")
for rate in (1000, 100, 50, 10):
    for h in (1, 100):
        run = f"m4-wan-{rate}mbit-h{h}"
        if (EXP / run).exists():
            tps = steady_tokens_per_sec(EXP / run)
            print(f"  {run}: {tps:,.0f} tok/s")
print("  M4 cloud headline (final eval):", round(final_eval("m4-cloud-headline") or -1, 2))

print("=" * 60, "\nM5 N=32 (canonical storm-n32p vs ref storm-n32-refp)")
win = chaos_window(EXP / "storm-n32p")
srate, sn = committed_rate(EXP / "storm-n32p", 20, win)
rrate, _ = committed_rate(EXP / "storm-n32-refp", 20, chaos_window(EXP / "storm-n32-refp"))
parts = [e["num_participants"] for f in (EXP / "storm-n32p").glob("replica*.jsonl")
         for e in load_jsonl(f) if e.get("event") == "outer_sync" and e.get("committed")
         and win[0] <= e["ts"] <= win[1]]
d = fuse(EXP / "storm-n32p")
ks = [f for f in d["faults"] if f["fault"] == "kill_safe" and "t_back_s" in f]
tb = sorted(f["t_back_s"] for f in ks)
tr = sorted(f["t_resume_s"] for f in d["faults"] if "t_resume_s" in f)
raw = [e for e in load_jsonl(EXP / "storm-n32p" / "chaos.jsonl") if e.get("event") == "fault"]
ex = [e for e in raw if e.get("ok") and not (isinstance(e.get("result"), dict) and "skipped" in e["result"])]
es = eval_series("storm-n32p")
ts_, ta_ = sum(d["committed_syncs"].values()), sum(d["total_syncs"].values())
print(f"  efficiency={srate/rrate:.1%} (storm {srate:.1f} / ref {rrate:.1f} steps/s)")
print(f"  quorum: median={int(st.median(parts))} mean={st.mean(parts):.1f} min={min(parts)} max={max(parts)}")
print(f"  faults executed={len(ex)} kills_executed_recovered={len(ks)}")
print(f"  T_back median={st.median(tb):.0f}s p90={tb[int(.9*len(tb))]:.0f}s  T_resume median={st.median(tr):.0f}s")
print(f"  commit rate={ts_}/{ta_}={ts_/ta_:.1%}  eval {es[0]:.1f}->{es[-1]:.1f} (min {min(es):.1f})")
