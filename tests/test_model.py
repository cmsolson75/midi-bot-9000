import pytest
import torch

from jazzbot.config import ModelConfig
from jazzbot.model import JazzTransformer


def small_model(dropout=0):
    return JazzTransformer(
        ModelConfig(
            vocab_size=32,
            dim=32,
            layers=2,
            heads=4,
            kv_heads=2,
            hidden_dim=64,
            context_length=32,
            dropout=dropout,
        )
    )


def test_causality_and_cache_equivalence():
    torch.manual_seed(123)
    model = small_model().eval()
    ids = torch.randint(0, 32, (2, 12))
    full, _, _ = model(ids)
    changed = ids.clone()
    changed[:, 7:] = torch.randint(0, 32, (2, 5))
    altered, _, _ = model(changed)
    torch.testing.assert_close(full[:, :7], altered[:, :7])
    _, _, cache = model(ids[:, :5], use_cache=True)
    chunk, _, cache = model(ids[:, 5:9], cache=cache, use_cache=True)
    last, _, cache = model(ids[:, 9:10], cache=cache, use_cache=True)
    torch.testing.assert_close(chunk, full[:, 5:9], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(last, full[:, 9:10], atol=2e-6, rtol=2e-5)
    assert cache[0][0].shape == (2, 2, 10, 8)


def test_context_limit_and_padding_loss():
    model = small_model()
    with pytest.raises(ValueError, match="context"):
        model(torch.ones((1, 33), dtype=torch.long))
    ids = torch.randint(0, 32, (1, 16))
    target = ids.clone()
    target[:, 8:] = -100
    _, loss, _ = model(ids, target)
    _, short_loss, _ = model(ids[:, :8], target[:, :8])
    torch.testing.assert_close(loss, short_loss)


def test_model_can_overfit_a_repeating_sequence():
    torch.manual_seed(42)
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    sequence = torch.tensor([[1, 2, 3, 4] * 4 + [1]])
    x, y = sequence[:, :-1], sequence[:, 1:]
    initial = float(model(x, y)[1].detach())
    for _ in range(60):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.1


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_training_and_cache(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} hardware unavailable")
    model = small_model().to(device)
    ids = torch.randint(0, 32, (2, 12), device=device)
    loss = model(ids[:, :-1], ids[:, 1:])[1]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        full = model(ids)[0]
        _, _, cache = model(ids[:, :-1], use_cache=True)
        cached = model(ids[:, -1:], cache=cache, use_cache=True)[0]
    torch.testing.assert_close(cached, full[:, -1:], atol=1e-4, rtol=1e-4)
