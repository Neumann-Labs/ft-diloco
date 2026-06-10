"""JSONL run telemetry. One file per replica per run; every plot and recovery
measurement in analysis/ is derived from these files (plus chaos/chaos.jsonl).

Event schema (all events carry: event, ts [unix seconds], mono [monotonic seconds],
run_id, replica_id):

  lifecycle   phase=start|exit, host, pid, config, torch_version, torchft_commit?
  step        step, loss, lr, tokens (cumulative this replica), dt_ms
  eval        step, eval_loss, ppl, tokens
  outer_sync  outer_step, step, t_start_mono, t_end_mono, committed, num_participants,
              bytes_analytic            (diloco mode only)
  digest      step, outer_step, kind=params|outer_momentum|original_params,
              l2norm, sha256_16         (diloco mode only; sync boundaries)
"""

import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch


class RunLogger:
    def __init__(self, out_dir: str | Path, run_id: str, replica_id: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.replica_id = replica_id
        self._f = open(self.out_dir / f"replica{replica_id}.jsonl", "a", buffering=1)

    def log(self, event: str, **fields) -> None:
        rec = {
            "event": event,
            "ts": time.time(),
            "mono": time.monotonic(),
            "run_id": self.run_id,
            "replica_id": self.replica_id,
            **fields,
        }
        self._f.write(json.dumps(rec) + "\n")

    def log_start(self, config: dict, **extra) -> None:
        self.log(
            "lifecycle",
            phase="start",
            host=socket.gethostname(),
            pid=os.getpid(),
            config=config,
            torch_version=torch.__version__,
            **extra,
        )

    def close(self, **extra) -> None:
        self.log("lifecycle", phase="exit", **extra)
        self._f.close()


def tensor_digest(tensors: list[torch.Tensor]) -> dict:
    """L2 norm + short content hash over a list of tensors (order-sensitive).

    Used at outer-sync boundaries to prove state equality across replicas and,
    after a rejoin, that P2P recovery restored params/outer momentum exactly.
    """
    h = hashlib.sha256()
    sq = 0.0
    for t in tensors:
        t = t.detach().float().cpu().contiguous()
        h.update(t.numpy().tobytes())
        sq += float((t * t).sum())
    return {"l2norm": sq**0.5, "sha256_16": h.hexdigest()[:16]}
