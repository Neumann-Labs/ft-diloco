"""Dataset preparation and batch sampling.

Prep (run once per dataset, on worker4):
    python -m ftdiloco.data --dataset tinystories --out data/tinystories

Produces train.bin / val.bin: flat uint16 GPT-2 BPE token streams with <|endoftext|>
between documents (nanoGPT convention).

Replica sharding: each DiLoCo replica group samples batches only from its contiguous
slice of the token stream, keyed by (replica_id, num_replicas). On membership change
survivors keep their original slices — some data goes under-sampled, which DiLoCo
tolerates and we document.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

EOT_GPT2 = 50256


def replica_shard(total_len: int, replica_id: int, num_replicas: int) -> tuple[int, int]:
    """Contiguous [lo, hi) token slice for one replica. Covers the stream, no overlap."""
    assert 0 <= replica_id < num_replicas
    per = total_len // num_replicas
    lo = replica_id * per
    hi = total_len if replica_id == num_replicas - 1 else lo + per
    return lo, hi


class TokenBins:
    """Memmap-backed train/val token streams with random-window batch sampling."""

    def __init__(
        self,
        data_dir: str | Path,
        block_size: int,
        replica_id: int = 0,
        num_replicas: int = 1,
        device: str = "cpu",
    ):
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.device = device
        self._train = np.memmap(self.data_dir / "train.bin", dtype=np.uint16, mode="r")
        self._val = np.memmap(self.data_dir / "val.bin", dtype=np.uint16, mode="r")
        self.train_lo, self.train_hi = replica_shard(len(self._train), replica_id, num_replicas)
        # Eval is identical for every replica: full val stream, deterministic windows.

    def get_batch(self, rng: np.random.Generator, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, lo, hi = self._train, self.train_lo, self.train_hi
        ix = rng.integers(lo, hi - self.block_size - 1, size=batch_size)
        x = torch.stack(
            [torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix]
        )
        if "cuda" in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def val_batches(self, batch_size: int, max_batches: int):
        """Deterministic, non-overlapping eval windows from the start of val.bin."""
        data = self._val
        stride = self.block_size * batch_size
        n = min(max_batches, (len(data) - 1) // stride)
        for b in range(n):
            base = b * stride
            xs, ys = [], []
            for j in range(batch_size):
                i = base + j * self.block_size
                xs.append(torch.from_numpy(data[i : i + self.block_size].astype(np.int64)))
                ys.append(torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)))
            x, y = torch.stack(xs), torch.stack(ys)
            if "cuda" in self.device:
                x = x.pin_memory().to(self.device, non_blocking=True)
                y = y.pin_memory().to(self.device, non_blocking=True)
            else:
                x, y = x.to(self.device), y.to(self.device)
            yield x, y


def _prepare_hf(dataset: str, out_dir: Path, num_proc: int) -> None:
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    if dataset == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", num_proc=num_proc)
        split_map = {"train": "train", "val": "validation"}
        text_key = "text"
    elif dataset == "fineweb":
        # 10BT sample; only a few GB of it is needed for headline runs.
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", num_proc=num_proc)
        ds = ds["train"].train_test_split(test_size=0.0005, seed=1337)
        ds = {"train": ds["train"], "test": ds["test"]}
        split_map = {"train": "train", "val": "test"}
        text_key = "text"
    else:
        raise ValueError(f"unknown dataset {dataset}")

    def tokenize(example):
        ids = enc.encode_ordinary(example[text_key])
        ids.append(EOT_GPT2)
        return {"ids": ids, "len": len(ids)}

    out_dir.mkdir(parents=True, exist_ok=True)
    for out_name, split in split_map.items():
        tokenized = ds[split].map(
            tokenize,
            remove_columns=ds[split].column_names,
            desc=f"tokenizing {split}",
            num_proc=num_proc,
        )
        arr_len = int(np.sum(tokenized["len"], dtype=np.uint64))
        path = out_dir / f"{out_name}.bin"
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))
        idx = 0
        n_shards = 1024 if len(tokenized) > 1024 else 1
        for shard_idx in range(n_shards):
            shard = tokenized.shard(num_shards=n_shards, index=shard_idx, contiguous=True)
            shard = shard.with_format("numpy")
            batch = np.concatenate(shard["ids"])
            arr[idx : idx + len(batch)] = batch
            idx += len(batch)
        arr.flush()
        print(f"{path}: {arr_len:,} tokens")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tinystories", "fineweb"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    _prepare_hf(args.dataset, Path(args.out), args.num_proc)


if __name__ == "__main__":
    main()
