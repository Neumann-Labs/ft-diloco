"""Interface byte-counter sampler (Linux). Samples /sys/class/net/<if>/statistics
at a fixed cadence into JSONL; analysis brackets outer-sync windows against it to
get *measured* comm volume, cross-checked with the analytic estimate.

Run on the host alongside a training run:
    python -m ftdiloco.netmon --ifaces vftd0 vftd1 --out experiments/<run>/netmon.jsonl
"""

import argparse
import json
import time
from pathlib import Path


def read_counters(iface: str) -> dict:
    base = Path("/sys/class/net") / iface / "statistics"
    return {
        "rx_bytes": int((base / "rx_bytes").read_text()),
        "tx_bytes": int((base / "tx_bytes").read_text()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ifaces", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", buffering=1) as f:
        while True:
            rec = {"ts": time.time(), "mono": time.monotonic()}
            for iface in args.ifaces:
                try:
                    rec[iface] = read_counters(iface)
                except FileNotFoundError:
                    rec[iface] = None
            f.write(json.dumps(rec) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
