"""Generate a Poisson storm schedule (YAML) for chaos/controller.py.

  python -m chaos.gen_schedule --run m3-storm-k120 --duration 2700 \
      --mean-kill 120 --mean-straggle 300 --mean-partition 400 --seed 7 \
      --out configs/chaos/m3_storm_k120.yaml

Kills use `kill_safe` (skipped if no other replica is alive — documented rule: the
storm never executes a cluster-wide kill; supervisors handle relaunches, so there is
no `relaunch` event in storm schedules). Stragglers are SIGSTOP/SIGCONT pairs;
partitions are link-down/up pairs. Deterministic given --seed.
"""

import argparse

import numpy as np
import yaml


def gen(duration: float, mean_kill: float, mean_straggle: float, mean_partition: float,
        seed: int, n_replicas: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    events = []

    t = float(rng.exponential(mean_kill))
    while t < duration:
        events.append({"at": round(t, 1), "fault": "kill_safe",
                       "target": int(rng.integers(n_replicas))})
        t += float(rng.exponential(mean_kill))

    t = float(rng.exponential(mean_straggle))
    while t < duration - 30:
        target = int(rng.integers(n_replicas))
        dur = float(rng.uniform(10, 30))
        events.append({"at": round(t, 1), "fault": "stop", "target": target})
        events.append({"at": round(t + dur, 1), "fault": "cont", "target": target})
        t += float(rng.exponential(mean_straggle))

    t = float(rng.exponential(mean_partition))
    while t < duration - 45:
        target = int(rng.integers(n_replicas))
        dur = float(rng.uniform(15, 40))
        events.append({"at": round(t, 1), "fault": "partition", "target": target})
        events.append({"at": round(t + dur, 1), "fault": "heal", "target": target})
        t += float(rng.exponential(mean_partition))

    return sorted(events, key=lambda e: e["at"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--duration", type=float, default=2700)
    p.add_argument("--mean-kill", type=float, default=120)
    p.add_argument("--mean-straggle", type=float, default=300)
    p.add_argument("--mean-partition", type=float, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--replicas", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    events = gen(args.duration, args.mean_kill, args.mean_straggle, args.mean_partition,
                 args.seed, args.replicas)
    doc = {
        "run_id": args.run,
        "generator": {k: getattr(args, k) for k in
                      ("duration", "mean_kill", "mean_straggle", "mean_partition",
                       "seed", "replicas")},
        "events": events,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    kinds = {}
    for e in events:
        kinds[e["fault"]] = kinds.get(e["fault"], 0) + 1
    print(f"{args.out}: {len(events)} events over {args.duration}s — {kinds}")


if __name__ == "__main__":
    main()
