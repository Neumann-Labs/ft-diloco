import json

import torch

from ftdiloco.metrics import RunLogger, tensor_digest


def test_jsonl_schema(tmp_path):
    logger = RunLogger(tmp_path, run_id="t1", replica_id=2)
    logger.log_start({"lr": 0.1})
    logger.log("step", step=5, loss=1.0, lr=0.1, tokens=100, dt_ms=2.0)
    logger.close(status="ok")

    lines = [json.loads(line) for line in (tmp_path / "replica2.jsonl").open()]
    assert [r["event"] for r in lines] == ["lifecycle", "step", "lifecycle"]
    for r in lines:
        assert r["run_id"] == "t1" and r["replica_id"] == 2
        assert "ts" in r and "mono" in r
    assert lines[0]["phase"] == "start" and "host" in lines[0] and "pid" in lines[0]
    assert lines[2]["phase"] == "exit" and lines[2]["status"] == "ok"


def test_tensor_digest_sensitivity():
    a = [torch.ones(10), torch.zeros(5)]
    d1 = tensor_digest(a)
    d2 = tensor_digest([torch.ones(10), torch.zeros(5)])
    assert d1 == d2
    b = [torch.ones(10), torch.zeros(5)]
    b[0][0] = 2.0
    d3 = tensor_digest(b)
    assert d3["sha256_16"] != d1["sha256_16"]
    assert abs(d1["l2norm"] - 10**0.5) < 1e-6
