import numpy as np
import pytest

from ftdiloco.data import TokenBins, replica_shard


@pytest.fixture
def bins_dir(tmp_path):
    rng = np.random.default_rng(0)
    rng.integers(0, 50304, size=100_000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, 50304, size=20_000, dtype=np.uint16).tofile(tmp_path / "val.bin")
    return tmp_path


def test_replica_shard_partitions():
    total = 100_000
    n = 3
    slices = [replica_shard(total, i, n) for i in range(n)]
    assert slices[0][0] == 0 and slices[-1][1] == total
    for (lo_a, hi_a), (lo_b, _) in zip(slices, slices[1:]):
        assert hi_a == lo_b
        assert hi_a > lo_a


def test_get_batch_shapes_and_shifts(bins_dir):
    bins = TokenBins(bins_dir, block_size=32, device="cpu")
    rng = np.random.default_rng(1)
    import torch

    x, y = bins.get_batch(rng, batch_size=4)
    assert x.shape == (4, 32) and y.shape == (4, 32)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    # y is x shifted by one
    assert (x[0, 1:] == y[0, :-1]).all()


def test_sharded_sampling_stays_in_slice(bins_dir):
    bins = TokenBins(bins_dir, block_size=32, replica_id=1, num_replicas=2, device="cpu")
    rng = np.random.default_rng(2)
    raw = np.memmap(bins_dir / "train.bin", dtype=np.uint16, mode="r")
    lo, hi = bins.train_lo, bins.train_hi
    assert (lo, hi) == (50_000, 100_000)
    for _ in range(10):
        x, _ = bins.get_batch(rng, batch_size=2)
        # every sampled window must reproduce a slice of raw inside [lo, hi)
        for row in x:
            row_np = row.numpy().astype(np.uint16)
            found = False
            for start in range(lo, hi - 33):
                if np.array_equal(raw[start : start + 32], row_np):
                    found = True
                    break
            assert found


def test_val_batches_deterministic(bins_dir):
    bins = TokenBins(bins_dir, block_size=32, device="cpu")
    a = [x.clone() for x, _ in bins.val_batches(batch_size=2, max_batches=3)]
    b = [x.clone() for x, _ in bins.val_batches(batch_size=2, max_batches=3)]
    assert len(a) == 3
    for xa, xb in zip(a, b):
        assert (xa == xb).all()
