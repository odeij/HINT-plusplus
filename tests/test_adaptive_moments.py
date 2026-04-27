"""Tests for HINT++ Phase 2: AdaptiveMomentSafety."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.safety.adaptive_moments import AdaptiveMomentSafety

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "experiments" / "phase2_init" / "results" / "phase2_init.pt"
NUM_CLASSES = 13
FLOOR_IDX = 1
BEAM_IDX = 3


def fresh_module(**overrides) -> AdaptiveMomentSafety:
    kwargs = dict(init_path=INIT_PATH, num_classes=NUM_CLASSES)
    kwargs.update(overrides)
    return AdaptiveMomentSafety(**kwargs)


# ---------- construction / validation ----------

def test_init_path_required():
    with pytest.raises(ValueError, match="init_path is required"):
        AdaptiveMomentSafety()


def test_init_path_nonexistent():
    with pytest.raises(FileNotFoundError):
        AdaptiveMomentSafety(init_path="/tmp/does-not-exist.pt")


def test_loads_eta_k_from_disk():
    m = fresh_module()
    assert m.eta_k.shape == (NUM_CLASSES,)
    assert m.eta_k.dtype == torch.float32
    assert int(torch.argmax(m.eta_k).item()) == FLOOR_IDX
    assert int(torch.argmin(m.eta_k).item()) == BEAM_IDX
    assert m.eta_k[FLOOR_IDX].item() == pytest.approx(0.9928, abs=1e-3)
    assert m.eta_k[BEAM_IDX].item() == pytest.approx(0.7378, abs=1e-3)


def test_initial_state():
    m = fresh_module()
    assert m.t.item() == 0
    assert torch.all(m.m_k == 0)
    assert (m.v_k > 0).all()


def test_beta_asymmetry_strict():
    fresh_module(beta1=0.7, beta2=0.95)
    fresh_module(beta1=0.9, beta2=0.999)  # Adam-defaults ablation: allowed
    for b1, b2 in [(0.95, 0.95), (0.99, 0.95), (0.0, 0.95), (0.7, 1.0)]:
        with pytest.raises(ValueError, match="Asymmetric beta"):
            fresh_module(beta1=b1, beta2=b2)


def test_eps_must_be_positive():
    for bad in [0.0, -1e-8, -1.0]:
        with pytest.raises(ValueError, match="eps"):
            fresh_module(eps=bad)


def test_eta_must_be_positive():
    for bad in [0.0, -1.0]:
        with pytest.raises(ValueError, match="eta"):
            fresh_module(eta=bad)


# ---------- forward semantics ----------

def test_forward_shape_and_dtype():
    m = fresh_module()
    w = m(torch.zeros(NUM_CLASSES))
    assert w.shape == (NUM_CLASSES,)
    assert w.dtype == torch.float32


def test_forward_rejects_wrong_shape():
    m = fresh_module()
    with pytest.raises(ValueError, match="delta must have shape"):
        m(torch.zeros(7))


def test_t_increments():
    m = fresh_module()
    assert m.t.item() == 0
    m(torch.zeros(NUM_CLASSES))
    assert m.t.item() == 1
    m(torch.zeros(NUM_CLASSES))
    assert m.t.item() == 2


def test_cold_start_identity():
    """At t=1 with m_k(0)=0, m_hat must equal delta exactly.

    This is the property that makes m_k(0)=0 the right cold-start choice:
    bias correction puts the first correction into m_hat with weight 1.
    The full safety weight at t=1 is also analytically reproducible.
    """
    m = fresh_module(eta=1.0)
    delta = torch.tensor(
        [0.5, -0.3, 0.1, 0.0, 0.7, -0.2, 0.4,
         0.6, -0.1, 0.8, 0.2, -0.5, 0.3],
        dtype=torch.float32,
    )
    pre_v = m.v_k.clone()

    w = m(delta)

    # m_k after first update: (1-beta1) * delta
    expected_m_k = (1 - m.beta1) * delta
    assert torch.allclose(m.m_k, expected_m_k, atol=1e-7)

    # m_hat at t=1 == delta (the cold-start identity)
    m_hat = m.m_k / (1 - m.beta1 ** 1)
    assert torch.allclose(m_hat, delta, atol=1e-6)

    # Full reconstruction of w at t=1
    expected_v_k = m.beta2 * pre_v + (1 - m.beta2) * delta * delta
    expected_v_hat = expected_v_k / (1 - m.beta2 ** 1)
    expected_w = m.eta * m.eta_k * delta / (torch.sqrt(expected_v_hat) + m.eps)
    assert torch.allclose(w, expected_w, atol=1e-6)


def test_per_class_independence():
    """delta non-zero at one class must only update that class's moments."""
    m = fresh_module()
    pre_m = m.m_k.clone()
    pre_v = m.v_k.clone()

    delta = torch.zeros(NUM_CLASSES)
    delta[BEAM_IDX] = 1.0
    m(delta)

    other = torch.arange(NUM_CLASSES) != BEAM_IDX
    assert torch.allclose(m.m_k[other], m.beta1 * pre_m[other], atol=1e-7)
    assert torch.allclose(m.v_k[other], m.beta2 * pre_v[other], atol=1e-7)
    assert m.m_k[BEAM_IDX].item() == pytest.approx(1 - m.beta1, abs=1e-7)


def test_eta_k_scales_per_class():
    """w uses element-wise eta_k, not a scalar."""
    m_a = fresh_module()
    m_b = fresh_module()
    m_b.eta_k.mul_(2.0)

    delta = torch.full((NUM_CLASSES,), 0.5)
    w_a = m_a(delta)
    w_b = m_b(delta)
    assert torch.allclose(w_b, 2.0 * w_a, atol=1e-6)


def test_global_eta_scales_uniformly():
    """The global scalar eta is a uniform multiplier independent of class."""
    delta = torch.full((NUM_CLASSES,), 0.5)
    w_a = fresh_module(eta=1.0)(delta)
    w_b = fresh_module(eta=3.0)(delta)
    assert torch.allclose(w_b, 3.0 * w_a, atol=1e-6)


def test_conservative_for_high_prior_classes():
    """At t=1 with identical delta across classes, classes with larger
    v_k(0) (sofa, beam) should yield smaller w_k than common classes
    (floor, ceiling). This is the 'conservative until corrections
    accumulate' design property of the v_k(0) prior."""
    m = fresh_module()
    delta = torch.full((NUM_CLASSES,), 0.5)
    w = m(delta)
    # floor (idx 1) is the common-and-confident class with smallest v_k(0)
    # beam (idx 3) has the highest v_k(0) from teacher uncertainty
    # sofa (idx 9) has the highest v_k(0) from rarity
    assert w[FLOOR_IDX].item() > w[BEAM_IDX].item()
    assert w[FLOOR_IDX].item() > w[9].item()  # sofa


# ---------- long-run stability ----------

def test_no_nan_long_run():
    m = fresh_module()
    g = torch.Generator().manual_seed(0)
    for _ in range(10_000):
        delta = torch.randn(NUM_CLASSES, generator=g) * 0.5 + 0.5
        w = m(delta)
        assert torch.isfinite(w).all()
    assert torch.isfinite(m.m_k).all()
    assert torch.isfinite(m.v_k).all()
    assert m.t.item() == 10_000


# ---------- state management ----------

def test_state_dict_round_trip():
    m1 = fresh_module()
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        m1(torch.randn(NUM_CLASSES, generator=g))

    state = m1.state_dict()
    m2 = fresh_module()
    m2.load_state_dict(state)

    assert m2.t.item() == m1.t.item()
    assert torch.equal(m2.m_k, m1.m_k)
    assert torch.equal(m2.v_k, m1.v_k)
    assert torch.equal(m2.eta_k, m1.eta_k)
    assert torch.equal(m2.m_k_init, m1.m_k_init)
    assert torch.equal(m2.v_k_init, m1.v_k_init)


def test_reset_restores_init_state():
    m = fresh_module()
    init_m = m.m_k.clone()
    init_v = m.v_k.clone()

    g = torch.Generator().manual_seed(2)
    for _ in range(20):
        m(torch.randn(NUM_CLASSES, generator=g))

    assert m.t.item() == 20
    assert not torch.allclose(m.m_k, init_m)

    m.reset()
    assert m.t.item() == 0
    assert torch.allclose(m.m_k, init_m)
    assert torch.allclose(m.v_k, init_v)


def test_no_parameters_only_buffers():
    """Phase 2 priors must not be trainable. Phase 6 may add learnable
    eta later -- this test will need updating then."""
    m = fresh_module()
    param_names = [n for n, _ in m.named_parameters()]
    assert param_names == [], f"unexpected parameters: {param_names}"
    buffer_names = {n for n, _ in m.named_buffers()}
    expected = {"eta_k", "m_k", "v_k", "m_k_init", "v_k_init", "t"}
    assert expected.issubset(buffer_names), \
        f"missing buffers: {expected - buffer_names}"
