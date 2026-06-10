# ft-diloco

**Fault-tolerant DiLoCo: resilient low-bandwidth LLM training on commodity hardware.**

Train a small LM across cheap, unreliable machines that sync only every H steps
(DiLoCo: inner AdamW, outer Nesterov SGD over pseudo-gradients, via
[torchft](https://github.com/meta-pytorch/torchft)) — and show that killing,
disconnecting, or adding machines mid-run does not break convergence.

> Status: M0 (baseline + harness) in progress. README gets the GIF and plots at M5.

## Layout

- `src/ftdiloco/` — model (nanoGPT-class), data (uint16 memmap shards), train loop,
  torchft integration (`train.py` + `ft.py` are the only torchft touchpoints), JSONL metrics
- `chaos/` — fault-injection controller (kill / partition / throttle / late-join)
- `scripts/` — lighthouse/worker launchers, netns fake-WAN, run recipes
- `analysis/` — log fusion + plots
- `configs/` — model / train / chaos / netem YAML
- `experiments/<run_id>/` — committed JSONL + plots per run
- `docs/` — architecture, runbook, torchft findings (issue #171 evidence)

## Quickstart (dev)

```bash
uv venv && uv pip install -e '.[dev]'
make lint test
# data prep + training run on the GPU host:
python -m ftdiloco.data --dataset tinystories --out data/tinystories
python -m ftdiloco.train --config configs/train/m0_tiny.yaml
python analysis/plot_convergence.py --runs experiments/m0-tiny-* --out plots/m0.png
```

## Hardware

| Node | Role | Spec |
|---|---|---|
| worker4 | GPU trainer | Ryzen 9 5950X, RTX 3060 12GB, Gen4 NVMe |
| worker1 | Lighthouse / CPU worker | 8-core, 16GB |
| link | deliberately commodity | gigabit ethernet + tc/netem WAN simulation |

torchft is installed editable from a pinned fork checkout on the training hosts
(commit recorded in `pyproject.toml`); it is not a pip dependency of this package.
