# torchft friction log & findings (feeds the issue #171 comment)

Anchor: [meta-pytorch/torchft#171](https://github.com/meta-pytorch/torchft/issues/171)
— "Towards Native Fault Tolerance for Semi-Synchronous Training". Open as of 2026-06-10.
Pinned commit for all observations: `4157be16` (+ `torchft-nightly==2026.6.10`).

Every entry: date, what happened, repro, tag (`candidate-issue` / `candidate-pr` /
`candidate-doc` / `evidence-171`). Measured evidence for the eventual #171 comment:
momentum-recovery digests, T_detect/T_resume/T_rejoin, 1-survivor sync semantics,
kill-mid-allreduce behavior on Gloo.

---

## 2026-06-10 — setup notes

- `train_diloco.py` (repo root) pins `CUDA_VISIBLE_DEVICES = REPLICA_GROUP_ID % 4` at
  module import; any same-GPU multi-replica setup must avoid that pattern. The example
  is also MLP/dummy-data only — no small-scale LM example exists. [candidate-doc]
- Known live bugs steered around from day one: #316 (async-quorum SIGSEV → we run
  `use_async_quorum=False`), #323 (PGTransport timeout ineffective → we use
  HTTPTransport). [evidence-171: config guidance]

- `Manager` hard-requires torchrun-style env (`MASTER_ADDR`/`MASTER_PORT`, and the
  TCPStore it implies) even for a single-process replica group — standalone (non-torchrun)
  DiLoCo usage isn't documented anywhere; repro: construct Manager without torchrun →
  `KeyError: 'MASTER_ADDR'` in manager.py store setup. Workaround: export
  MASTER_ADDR=localhost + unique MASTER_PORT per replica group on shared hosts.
  [candidate-doc]
- Root cause of the standalone hang: `Manager.__init__` connects to the replica-group
  TCPStore with `is_master=False` (manager.py:291) — it assumes torchrun is hosting the
  store. Without torchrun the constructor blocks forever (no timeout, no error). A
  standalone launcher must host a `TCPStore(is_master=True)` on group rank 0 first.
  Repro: construct Manager with MASTER_ADDR/PORT set but no store server → hang.
  [candidate-doc; candidate-pr: a store-hosting fallback or a clear timeout error]
