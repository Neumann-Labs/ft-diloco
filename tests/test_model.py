import torch

from ftdiloco.model import GPT, GPTConfig

TINY = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=64)


def test_forward_shapes_and_loss():
    m = GPT(TINY)
    x = torch.randint(0, 256, (3, 64))
    y = torch.randint(0, 256, (3, 64))
    logits, loss = m(x, y)
    assert logits.shape == (3, 64, 256)
    assert loss is not None and torch.isfinite(loss)
    logits, loss = m(x)
    assert logits.shape == (3, 1, 256)
    assert loss is None


def test_weight_tying():
    m = GPT(TINY)
    assert m.transformer.wte.weight is m.lm_head.weight


def test_param_count_tiny50m():
    cfg = GPTConfig(block_size=512, vocab_size=50304, n_layer=8, n_head=8, n_embd=512)
    n = GPT(cfg).num_params()
    assert 45e6 < n < 60e6, f"tiny50m param count off: {n:,}"


def test_causality():
    m = GPT(TINY)
    m.eval()
    x = torch.randint(0, 256, (1, 64))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 256
    with torch.no_grad():
        l1, _ = m(x, x)
        l2, _ = m(x2, x2)
    # changing the last token must not affect logits at earlier positions
    assert torch.allclose(l1[0, :-1], l2[0, :-1], atol=1e-5)
    assert not torch.allclose(l1[0, -1], l2[0, -1])
