"""Tests for HINT++ Phase 2: AdaptiveMomentSafety (R1 estimator, memo §3).

R1 test-change ledger (one line of justification per legacy test touched):
- test_cold_start_identity REPLACED by test_t1_worked_check + test_cold_start_zero — the old
  m_hat(1)=delta identity is exactly flaw F3 (bias correction with informed prior ⇒ instant trust).
- test_initial_state UPDATED — R1 zero-inits the EMAs; the v prior lives in a separate v_k_0 buffer.
- test_per_class_independence UPDATED — R1 EMAs are event-indexed (N_k per class): classes without
  an event are untouched, not decayed (decay-by-global-t belonged to the pre-R1 dense update).
- test_conservative_for_high_prior_classes UPDATED — property survives; expected mechanism is now
  the lambda-weighted v prior; delta must be ±1 (F4 outcome semantics).
- test_eta_k_scales_per_class / test_global_eta_scales_uniformly UPDATED — delta in {−1,0,+1} only.
- test_forward_shape_and_dtype UPDATED — all-zeros delta means "no events" and must yield w == 0.
- test_t_increments UPDATED — adds per-class N_k assertions (lambda uses N_k, never global t).
- test_no_nan_long_run UPDATED — drives random ±1 event masks instead of float noise.
- test_state_dict_round_trip / test_reset / test_no_parameters_only_buffers UPDATED — R1 buffer set
  is {eta_k, v_k_0, m_k, v_k, N_k, t}; reset restores zero EMAs and N_k=0 (m_k_init/v_k_init gone).
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch

from src.safety.adaptive_moments import AdaptiveMomentSafety

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = REPO_ROOT / "experiments" / "phase2_init" / "results" / "phase2_init.pt"
CONFIG_PATH = REPO_ROOT / "configs" / "safety.yaml"
NUM_CLASSES = 13
FLOOR_IDX = 1
BEAM_IDX = 3
SOFA_IDX = 9


def fresh_module(**overrides) -> AdaptiveMomentSafety:
    kwargs = dict(init_path=INIT_PATH, num_classes=NUM_CLASSES)
    kwargs.update(overrides)
    return AdaptiveMomentSafety(**kwargs)


def make_synthetic_init(tmp_path: Path, v0: float = 0.6) -> Path:
    """Synthetic phase2_init.pt: eta_k = 1, v_k(0) = v0 — isolates the estimator math."""
    path = tmp_path / "phase2_init.pt"
    torch.save(
        {
            "class_names": [f"c{i}" for i in range(NUM_CLASSES)],
            "eta_k": torch.ones(NUM_CLASSES),
            "m_k_0": torch.zeros(NUM_CLASSES),
            "v_k_0": torch.full((NUM_CLASSES,), v0),
        },
        path,
    )
    return path


def event(idx: int, sign: float = 1.0) -> torch.Tensor:
    delta = torch.zeros(NUM_CLASSES)
    delta[idx] = sign
    return delta


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
    assert torch.all(m.m_k == 0)          # R1: EMAs are zero-init
    assert torch.all(m.v_k == 0)
    assert torch.all(m.N_k == 0)
    assert (m.v_k_0 > 0).all()            # prior lives in its own buffer


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


def test_n0_must_be_positive():
    for bad in [0.0, -5.0]:
        with pytest.raises(ValueError, match="n0"):
            fresh_module(n0=bad)


def test_config_matches_constructor_defaults():
    """configs/safety.yaml is the canonical source; constructor defaults must not drift from it."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["safety"]
    sig = inspect.signature(AdaptiveMomentSafety.__init__)
    for key in ("beta1", "beta2", "eps", "eta", "n0"):
        assert float(cfg[key]) == pytest.approx(sig.parameters[key].default), key


def test_rejects_nonzero_m_k_0(tmp_path):
    """The guard that keeps F3 from re-entering via a stale init artifact must fire."""
    path = tmp_path / "stale_init.pt"
    torch.save(
        {
            "eta_k": torch.ones(NUM_CLASSES),
            "m_k_0": torch.full((NUM_CLASSES,), 0.1),  # pre-R1 prior in the EMA: forbidden
            "v_k_0": torch.full((NUM_CLASSES,), 0.6),
        },
        path,
    )
    with pytest.raises(ValueError, match=r"m_k\(0\) = 0"):
        AdaptiveMomentSafety(init_path=path)


def test_accepts_init_without_m_k_0(tmp_path):
    """m_k_0 is vestigial under R1: an init file without it is valid."""
    path = tmp_path / "lean_init.pt"
    torch.save(
        {
            "eta_k": torch.ones(NUM_CLASSES),
            "v_k_0": torch.full((NUM_CLASSES,), 0.6),
        },
        path,
    )
    m = AdaptiveMomentSafety(init_path=path)
    assert torch.all(m.m_k == 0)


# ---------- forward semantics (delta is an OUTCOME in {-1, 0, +1}; F4) ----------

def test_forward_shape_and_dtype():
    m = fresh_module()
    w = m(torch.zeros(NUM_CLASSES))  # no events anywhere
    assert w.shape == (NUM_CLASSES,)
    assert w.dtype == torch.float32
    assert torch.all(w == 0)         # cold start: N_k = 0 ⇒ w = 0 exactly


def test_forward_rejects_wrong_shape():
    m = fresh_module()
    with pytest.raises(ValueError, match="delta must have shape"):
        m(torch.zeros(7))


def test_forward_rejects_non_outcome_values():
    m = fresh_module()
    for bad in [0.5, 2.0, -3.0]:
        with pytest.raises(ValueError, match="outcome"):
            m(event(BEAM_IDX, bad))


def test_t_increments_and_N_k_counts_per_class():
    m = fresh_module()
    assert m.t.item() == 0
    m(event(BEAM_IDX))
    assert m.t.item() == 1
    assert m.N_k[BEAM_IDX].item() == 1
    assert m.N_k.sum().item() == 1
    m(event(FLOOR_IDX, -1.0))
    assert m.t.item() == 2
    assert m.N_k[BEAM_IDX].item() == 1
    assert m.N_k[FLOOR_IDX].item() == 1
    m(torch.zeros(NUM_CLASSES))      # a forward call with no events
    assert m.t.item() == 3
    assert m.N_k.sum().item() == 2   # N_k counts events, not calls


# ---------- the R1 worked checks (memo §3 — MUST hold) ----------

def test_t1_worked_check(tmp_path):
    """Memo §3: first event, delta=+1, n0=5, v_k(0)=0.6 ⇒ w ≈ 0.204·eta·eta_k (spec bound ±10%).

    Derivation: m = 0.3, m~ = m/(1−0.7) = 1, lambda = 5/6, m^ = (1−lambda)·m~ = 1/6;
    v = 0.05, v~ = v/(1−0.95) = 1, v^ = (5/6)·0.6 + (1/6)·1 = 2/3;
    w = (1/6)/√(2/3) ≈ 0.20412.
    """
    m = AdaptiveMomentSafety(init_path=make_synthetic_init(tmp_path, v0=0.6))
    w = m(event(BEAM_IDX, +1.0))
    expected = (1.0 / 6.0) / math.sqrt(2.0 / 3.0)  # 0.20412…, eta = eta_k = 1
    assert w[BEAM_IDX].item() == pytest.approx(expected, rel=1e-4)   # implementation is exact
    assert w[BEAM_IDX].item() == pytest.approx(0.20, rel=0.10)       # the memo's ±10% bound


def test_cold_start_zero(tmp_path):
    """Memo §3: N_k = 0 ⇒ lambda = 1 ⇒ m^ = 0 ⇒ w = 0 EXACTLY — per class, not just globally."""
    m = AdaptiveMomentSafety(init_path=make_synthetic_init(tmp_path))
    w = m(event(BEAM_IDX))
    untouched = torch.arange(NUM_CLASSES) != BEAM_IDX
    assert torch.all(w[untouched] == 0)          # exact zero, no epsilon leakage
    assert w[BEAM_IDX].item() != 0


def test_lambda_monotone_to_zero(tmp_path):
    """lambda_k = n0/(n0+N_k) must follow the formula and decrease strictly toward 0;
    for a constant +1 stream w must approach eta·eta_k (m^ → 1, v^ → 1)."""
    m = AdaptiveMomentSafety(init_path=make_synthetic_init(tmp_path))
    lambdas = []
    w_last = None
    for i in range(1, 1001):
        w_last = m(event(FLOOR_IDX))
        lam = m.lambda_k[FLOOR_IDX].item()
        assert lam == pytest.approx(5.0 / (5.0 + i), rel=1e-6)
        lambdas.append(lam)
    assert all(a > b for a, b in zip(lambdas, lambdas[1:]))   # strictly decreasing
    assert lambdas[-1] < 0.005
    assert w_last[FLOOR_IDX].item() == pytest.approx(1.0, rel=0.02)  # asymptote eta·eta_k = 1


def test_sign_propagation(tmp_path):
    """w is SIGNED: an all-negative outcome stream drives w below zero with growing magnitude,
    mirroring the all-positive stream exactly (v update uses delta², so trajectories are symmetric)."""
    m_neg = AdaptiveMomentSafety(init_path=make_synthetic_init(tmp_path))
    m_pos = AdaptiveMomentSafety(init_path=make_synthetic_init(tmp_path))
    w_neg_hist = []
    for _ in range(10):
        w_neg = m_neg(event(BEAM_IDX, -1.0))
        w_pos = m_pos(event(BEAM_IDX, +1.0))
        assert w_neg[BEAM_IDX].item() < 0
        assert w_neg[BEAM_IDX].item() == pytest.approx(-w_pos[BEAM_IDX].item(), rel=1e-5)
        w_neg_hist.append(w_neg[BEAM_IDX].item())
    assert all(a > b for a, b in zip(w_neg_hist, w_neg_hist[1:]))  # strictly more negative


def test_no_instant_full_trust(tmp_path):
    """Regression guard on F3: the first event must NOT land with weight ~1 in m^ (the pre-R1
    'cold-start identity'). With n0=5 the first-event trust is damped to (1−5/6) = 1/6."""
    m = AdaptiveMomentSafety(init_path=make_synthetic_init(tmp_path))
    m(event(BEAM_IDX))
    assert m.lambda_k[BEAM_IDX].item() == pytest.approx(5.0 / 6.0, rel=1e-6)
    # m_hat = (1 - lambda) * m_tilde = 1/6, nowhere near the pre-R1 instant value of 1.0
    assert m.m_hat[BEAM_IDX].item() == pytest.approx(1.0 / 6.0, rel=1e-5)


# ---------- per-class structure ----------

def test_per_class_independence():
    """R1 EMAs are event-indexed: an event on one class leaves every other class UNTOUCHED
    (no decay of other classes — that was the pre-R1 dense-update semantics)."""
    m = fresh_module()
    m(event(FLOOR_IDX))  # give floor one event so it has live state to protect
    pre_m = m.m_k.clone()
    pre_v = m.v_k.clone()
    pre_N = m.N_k.clone()

    m(event(BEAM_IDX))

    other = torch.arange(NUM_CLASSES) != BEAM_IDX
    assert torch.equal(m.m_k[other], pre_m[other])
    assert torch.equal(m.v_k[other], pre_v[other])
    assert torch.equal(m.N_k[other], pre_N[other])
    assert m.m_k[BEAM_IDX].item() == pytest.approx(1 - m.beta1, abs=1e-7)


def test_eta_k_scales_per_class():
    """w uses element-wise eta_k, not a scalar."""
    m_a = fresh_module()
    m_b = fresh_module()
    m_b.eta_k.mul_(2.0)
    delta = torch.ones(NUM_CLASSES)   # one +1 event on every class
    w_a = m_a(delta)
    w_b = m_b(delta)
    assert torch.allclose(w_b, 2.0 * w_a, atol=1e-6)


def test_global_eta_scales_uniformly():
    delta = torch.ones(NUM_CLASSES)
    w_a = fresh_module(eta=1.0)(delta)
    w_b = fresh_module(eta=3.0)(delta)
    assert torch.allclose(w_b, 3.0 * w_a, atol=1e-6)


def test_conservative_for_high_prior_classes():
    """With identical first events everywhere, classes with larger v_k(0) (beam: teacher
    uncertainty; sofa: rarity) get smaller w than floor — the lambda-weighted prior keeps
    low-evidence, high-risk classes conservative until corrections accumulate."""
    m = fresh_module()
    w = m(torch.ones(NUM_CLASSES))
    assert w[FLOOR_IDX].item() > w[BEAM_IDX].item()
    assert w[FLOOR_IDX].item() > w[SOFA_IDX].item()


# ---------- long-run stability ----------

def test_no_nan_long_run():
    """10,000 random ±1 event vectors with random sparsity: w, EMAs, and counts stay finite."""
    m = fresh_module()
    g = torch.Generator().manual_seed(0)
    for _ in range(10_000):
        signs = torch.randint(0, 2, (NUM_CLASSES,), generator=g) * 2.0 - 1.0
        mask = torch.rand(NUM_CLASSES, generator=g) < 0.3
        w = m(signs * mask)
        assert torch.isfinite(w).all()
    assert torch.isfinite(m.m_k).all()
    assert torch.isfinite(m.v_k).all()
    assert m.t.item() == 10_000
    assert (m.N_k <= 10_000).all()


def test_single_class_flood_to_1e4(tmp_path):
    """One class driven to N_k = 10^4: the float32 beta^N underflow path must stay finite and
    the constant +1 stream must sit at the asymptote w = eta * eta_k."""
    m = AdaptiveMomentSafety(init_path=make_synthetic_init(tmp_path))
    w = None
    for _ in range(10_000):
        w = m(event(FLOOR_IDX))
    assert m.N_k[FLOOR_IDX].item() == 10_000
    assert torch.isfinite(w).all()
    assert w[FLOOR_IDX].item() == pytest.approx(1.0, rel=1e-3)  # eta = eta_k = 1


# ---------- state management ----------

def test_state_dict_round_trip():
    m1 = fresh_module()
    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        signs = torch.randint(0, 2, (NUM_CLASSES,), generator=g) * 2.0 - 1.0
        mask = torch.rand(NUM_CLASSES, generator=g) < 0.5
        m1(signs * mask)

    state = m1.state_dict()
    m2 = fresh_module()
    m2.load_state_dict(state)

    assert m2.t.item() == m1.t.item()
    assert torch.equal(m2.m_k, m1.m_k)
    assert torch.equal(m2.v_k, m1.v_k)
    assert torch.equal(m2.N_k, m1.N_k)
    assert torch.equal(m2.eta_k, m1.eta_k)
    assert torch.equal(m2.v_k_0, m1.v_k_0)


def test_reset_restores_cold_start():
    m = fresh_module()
    g = torch.Generator().manual_seed(2)
    for _ in range(20):
        signs = torch.randint(0, 2, (NUM_CLASSES,), generator=g) * 2.0 - 1.0
        m(signs)
    assert m.t.item() == 20
    assert not torch.all(m.m_k == 0)

    m.reset()
    assert m.t.item() == 0
    assert torch.all(m.m_k == 0)
    assert torch.all(m.v_k == 0)
    assert torch.all(m.N_k == 0)
    assert torch.all(m(torch.zeros(NUM_CLASSES)) == 0)  # back to exact cold start
    m.reset()


def test_no_parameters_only_buffers():
    """Phase 2 priors must not be trainable. Phase 6 may add learnable eta later."""
    m = fresh_module()
    param_names = [n for n, _ in m.named_parameters()]
    assert param_names == [], f"unexpected parameters: {param_names}"
    buffer_names = {n for n, _ in m.named_buffers()}
    expected = {"eta_k", "v_k_0", "m_k", "v_k", "N_k", "t"}
    assert expected.issubset(buffer_names), \
        f"missing buffers: {expected - buffer_names}"
