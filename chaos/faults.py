"""Fault primitives. Every fault is a real OS-level action against a real process or
interface — never a cooperative shutdown. Runs on the training host (worker4) with
passwordless sudo scoped to ip/tc/iptables.

Targets are replica indices; PIDs are resolved fresh from the run's JSONL manifests at
injection time (they change across relaunches).
"""

import json
import os
import signal
import subprocess
from pathlib import Path


def _sudo(*args: str) -> None:
    subprocess.run(["sudo", *args], check=True, capture_output=True, text=True)


def resolve_pid(run_dir: Path, replica_id: int) -> int:
    """Latest start-event PID for a replica (handles relaunches), verified alive."""
    pid = None
    f = run_dir / f"replica{replica_id}.jsonl"
    for line in f.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "lifecycle" and e.get("phase") == "start":
            pid = e["pid"]
    if pid is None:
        raise RuntimeError(f"no start event for replica {replica_id} in {f}")
    os.kill(pid, 0)  # raises if dead
    return pid


def kill(run_dir: Path, replica_id: int) -> dict:
    pid = resolve_pid(run_dir, replica_id)
    os.kill(pid, signal.SIGKILL)
    return {"pid": pid}


def stop(run_dir: Path, replica_id: int) -> dict:
    pid = resolve_pid(run_dir, replica_id)
    os.kill(pid, signal.SIGSTOP)
    return {"pid": pid}


def cont(run_dir: Path, replica_id: int) -> dict:
    pid = resolve_pid(run_dir, replica_id)
    os.kill(pid, signal.SIGCONT)
    return {"pid": pid}


def partition(replica_id: int) -> dict:
    """'Unplug the cable': take the replica's host-side veth down."""
    _sudo("ip", "link", "set", f"vftd{replica_id}", "down")
    return {"iface": f"vftd{replica_id}"}


def heal(replica_id: int) -> dict:
    _sudo("ip", "link", "set", f"vftd{replica_id}", "up")
    return {"iface": f"vftd{replica_id}"}


def throttle(replica_id: int, netem: str) -> dict:
    """Apply a netem profile (e.g. 'rate 10mbit delay 50ms loss 1%') to the replica."""
    args = netem.split()
    _sudo("tc", "qdisc", "replace", "dev", f"vftd{replica_id}", "root", "netem", *args)
    _sudo(
        "ip", "netns", "exec", f"ftd{replica_id}",
        "tc", "qdisc", "replace", "dev", "eth0", "root", "netem", *args,
    )
    return {"iface": f"vftd{replica_id}", "netem": netem}


def unthrottle(replica_id: int) -> dict:
    subprocess.run(
        ["sudo", "tc", "qdisc", "del", "dev", f"vftd{replica_id}", "root"],
        capture_output=True,
    )
    subprocess.run(
        ["sudo", "ip", "netns", "exec", f"ftd{replica_id}",
         "tc", "qdisc", "del", "dev", "eth0", "root"],
        capture_output=True,
    )
    return {"iface": f"vftd{replica_id}"}


def relaunch(launch_cmd: str, replica_id: int) -> dict:
    """Start a replacement worker via the run's launch script (detached tmux inside)."""
    subprocess.run(
        ["bash", "-c", launch_cmd.format(R=replica_id)], check=True,
        capture_output=True, text=True,
    )
    return {"cmd": launch_cmd.format(R=replica_id)}
