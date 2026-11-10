"""Sanity checks that catch the classic transformer bugs.

    python -m pytest tests -q
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinygpt import GPT, GPTConfig, CharTokenizer, sample_from_logits  # noqa: E402


def tiny(**kw):
    cfg = dict(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    cfg.update(kw)
    torch.manual_seed(0)
    return GPT(GPTConfig(**cfg)).eval()


def test_shapes_and_initial_loss():
    m = tiny()
    x = torch.randint(0, 20, (3, 16))
    y = torch.randint(0, 20, (3, 16))  # random targets (with tied weights, y=x would be easier)
    logits, loss = m(x, y)
    assert logits.shape == (3, 16, 20)
    # An untrained model should be close to uniform: loss ~ ln(vocab).
    assert abs(loss.item() - torch.log(torch.tensor(20.0)).item()) < 0.3


def test_causal_mask_no_peeking():
    """Changing a future character must not change predictions at earlier positions."""
    m = tiny()
    x = torch.randint(0, 20, (1, 16))
    y = x.clone()
    y[0, 10:] = (y[0, 10:] + 1) % 20
    a, _ = m(x, x)
    b, _ = m(y, y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_kv_cache_matches_full_forward():
    m = tiny()
    x = torch.randint(0, 20, (1, 12))
    full, _ = m(x, x)
    _, caches = m(x[:, :5])
    for t in range(5, 12):
        step_logits, caches = m(x[:, t:t + 1], kv_caches=caches, start_pos=t)
        assert torch.allclose(step_logits[0, -1], full[0, t], atol=1e-5)


def test_greedy_generation_same_with_and_without_cache():
    m = tiny()
    start = torch.tensor([[1, 2, 3]])
    # Long enough to run past block_size and exercise the sliding window.
    a = m.generate(start, 40, temperature=0, use_cache=True)
    b = m.generate(start, 40, temperature=0, use_cache=False)
    assert a.shape == (1, 43)
    assert torch.equal(a[:, :16], b[:, :16])  # identical while the whole text fits


def test_can_overfit_one_batch():
    m = tiny()
    m.train()
    x = torch.randint(0, 20, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = m.configure_optimizer(3e-3, 0.0)
    for _ in range(300):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_top_k_and_top_p_filtering():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
    for _ in range(50):
        assert sample_from_logits(logits, 1.0, top_k=2).item() in (0, 1)
        assert sample_from_logits(logits, 1.0, top_p=0.5).item() == 0
    assert sample_from_logits(logits, 0).item() == 0


def test_weight_tying():
    m = tiny()
    assert m.head.weight is m.tok_emb.weight


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("To be, or not to be")
    s = "be or not"
    assert tok.decode(tok.encode(s)) == s
