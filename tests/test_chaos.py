import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root for chaos/ and analysis/

from chaos.controller import execute  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from parse_logs import fuse  # noqa: E402


def test_controller_dry_run_logs_ordered_events(tmp_path):
    schedule = {
        "run_id": "t",
        "events": [
            {"at": 0.05, "fault": "relaunch", "target": 1},
            {"at": 0.0, "fault": "kill", "target": 1},
        ],
    }
    execute(schedule, tmp_path, dry_run=True)
    recs = [json.loads(line) for line in (tmp_path / "chaos.jsonl").open()]
    kinds = [r["event"] for r in recs]
    assert kinds[0] == "chaos_start" and kinds[-1] == "chaos_end"
    faults = [r for r in recs if r["event"] == "fault"]
    assert [f["fault"] for f in faults] == ["kill", "relaunch"]  # sorted by `at`
    assert all(f["ok"] for f in faults)
    assert faults[0]["mono"] <= faults[1]["mono"]


def _write_jsonl(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_fuse_latency_math(tmp_path):
    t = 1000.0
    # replica 0 (survivor): syncs every 10s, participants drop 2->1 after the kill at t+35
    r0 = [{"event": "lifecycle", "phase": "start", "pid": 11, "ts": t, "replica_id": 0}]
    for i, (ts, parts) in enumerate([(t + 10, 2), (t + 20, 2), (t + 30, 2), (t + 40, 1), (t + 50, 1), (t + 70, 2)]):
        r0.append({"event": "outer_sync", "ts": ts, "replica_id": 0, "outer_step": i,
                   "committed": True, "num_participants": parts})
    # replica 1: killed at t+35, relaunched at t+55, first commit t+70
    r1 = [
        {"event": "lifecycle", "phase": "start", "pid": 12, "ts": t, "replica_id": 1},
        {"event": "outer_sync", "ts": t + 10, "replica_id": 1, "outer_step": 0, "committed": True, "num_participants": 2},
        {"event": "lifecycle", "phase": "start", "pid": 13, "ts": t + 56, "replica_id": 1},
        {"event": "outer_sync", "ts": t + 70, "replica_id": 1, "outer_step": 5, "committed": True, "num_participants": 2},
        {"event": "digest", "ts": t + 71, "replica_id": 1, "outer_step": 5, "kind": "params", "sha256_16": "aa"},
    ]
    r0.append({"event": "digest", "ts": t + 71, "replica_id": 0, "outer_step": 5, "kind": "params", "sha256_16": "aa"})
    chaos = [
        {"event": "fault", "fault": "kill", "target": 1, "at": 35, "ts": t + 35, "mono": 1, "ok": True},
        {"event": "fault", "fault": "relaunch", "target": 1, "at": 55, "ts": t + 55, "mono": 2, "ok": True},
    ]
    _write_jsonl(tmp_path / "replica0.jsonl", r0)
    _write_jsonl(tmp_path / "replica1.jsonl", r1)
    _write_jsonl(tmp_path / "chaos.jsonl", chaos)

    s = fuse(tmp_path)
    kill = next(f for f in s["faults"] if f["fault"] == "kill")
    rel = next(f for f in s["faults"] if f["fault"] == "relaunch")
    assert abs(kill["t_resume_s"] - 5.0) < 1e-6   # t+40 commit
    assert abs(kill["t_detect_s"] - 5.0) < 1e-6   # participants 2->1 at t+40
    assert abs(rel["t_rejoin_s"] - 15.0) < 1e-6   # t+70 commit
    assert s["digest_check"]["common_points"] == 1
    assert s["digest_check"]["matches"] == 1
