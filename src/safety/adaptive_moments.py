"""HINT++ Phase 2: Adaptive Moment Safety Signals (R1 trust estimator).

Per-class trust state from human correction OUTCOMES. Spec: memo §3
(docs/HINTpp_Design_Memo_R1_2026-06-11.md). Supersedes the pre-R1 estimator,
which applied 1/(1-beta^t) bias correction to a prior-initialized v (flaw F3).

Semantics (F4): delta_k(t) in {+1, -1} is the OUTCOME of a correction event on
class k, emitted after the gated update (+1 = local error on the corrected
region decreased; -1 = it did not, or the region was re-corrected within T_rc
events). delta_k = 0 means "no event for class k on this call". The EMAs are
EVENT-indexed: a class without an event is untouched, and all per-class
statistics advance on N_k (that class's cumulative event count), never on the
global call counter t.

Update rules per event (0 < beta1 < beta2 < 1, asserted; beta1=0.7 ~ 3.3-event
window, beta2=0.95 ~ 20-event window — reasoned in events, not wall clock, F5):
    m_k = beta1 * m_k + (1 - beta1) * delta_k        (zero-init)
    v_k = beta2 * v_k + (1 - beta2) * delta_k**2     (zero-init)
    m~_k = m_k / (1 - beta1**N_k)    zero-init bias-corrected internals;
    v~_k = v_k / (1 - beta2**N_k)    bias correction touches ONLY zero-init state
    lambda_k = n0 / (n0 + N_k)       prior pseudo-count mixture, n0 = 5
    m^_k = (1 - lambda_k) * m~_k                       (prior mean 0)
    v^_k = lambda_k * v_k(0) + (1 - lambda_k) * v~_k   (v_k(0) = 0.5 r_k + 0.5 u_k, Sub-step 0B)
    w_k  = eta * eta_k * m^_k / (sqrt(v^_k) + eps)     SIGNED

Worked check (memo §3, asserted in tests): first event, delta=+1, n0=5,
v_k(0)=0.6  =>  m~=1, lambda=5/6, m^=1/6, v^=2/3, w ~= 0.204 * eta * eta_k.
Cold start: N_k = 0  =>  lambda_k = 1  =>  w_k = 0 EXACTLY.

Hyperparameters: configs/safety.yaml is the canonical source (Hydra wiring
lands in Phase 6); constructor defaults mirror it and a test keeps them in sync.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class AdaptiveMomentSafety(nn.Module):
    """Per-class SIGNED safety weights via the R1 lambda-mixture trust estimator."""

    def __init__(
        self,
        init_path: Path | str | None = None,
        num_classes: int = 13,
        beta1: float = 0.7,
        beta2: float = 0.95,
        eps: float = 1e-8,
        eta: float = 1.0,
        n0: float = 5.0,
    ) -> None:
        super().__init__()

        if init_path is None:
            raise ValueError(
                "init_path is required. Pass an explicit path to "
                "phase2_init.pt (output of "
                "experiments/phase2_init/scripts/run_0c_master.py). "
                "Phase 6 Hydra config is the eventual integration point."
            )
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not (0.0 < beta1 < beta2 < 1.0):
            raise ValueError(
                f"Asymmetric beta required: 0 < beta1 < beta2 < 1, "
                f"got beta1={beta1}, beta2={beta2}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        if n0 <= 0:
            raise ValueError(f"n0 must be positive, got {n0}")

        self.num_classes = num_classes
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eta = eta
        self.n0 = n0

        path = Path(init_path)
        if not path.is_file():
            raise FileNotFoundError(f"init_path does not exist: {path}")
        init = torch.load(path, weights_only=False)

        eta_k = init["eta_k"].to(torch.float32)
        v_k_0 = init["v_k_0"].to(torch.float32)

        expected = (num_classes,)
        if eta_k.shape != expected or v_k_0.shape != expected:
            raise ValueError(
                f"phase2_init.pt tensors must have shape {expected}, got "
                f"eta_k={tuple(eta_k.shape)}, v_k_0={tuple(v_k_0.shape)}"
            )
        if (v_k_0 <= 0).any():
            raise ValueError(
                f"v_k(0) must be strictly positive (sqrt is taken in "
                f"forward); min v_k_0 = {float(v_k_0.min()):.3e}"
            )
        if "m_k_0" in init and bool((init["m_k_0"] != 0).any()):
            raise ValueError(
                "R1 requires m_k(0) = 0 (memo §3); phase2_init.pt carries a "
                "nonzero m_k_0, which belongs to the retired pre-R1 estimator."
            )

        # Immutable per-class priors.
        self.register_buffer("eta_k", eta_k)
        self.register_buffer("v_k_0", v_k_0)
        # Zero-init event-indexed EMAs (the lambda mixture injects the prior;
        # the prior never decays inside the EMA — flaw F3).
        self.register_buffer("m_k", torch.zeros(num_classes))
        self.register_buffer("v_k", torch.zeros(num_classes))
        # Per-class cumulative event counts (drives lambda_k and the
        # bias-correction exponents) and a global call counter (diagnostics).
        self.register_buffer("N_k", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("t", torch.tensor(0, dtype=torch.long))

    @property
    def lambda_k(self) -> Tensor:
        """Prior weight lambda_k = n0 / (n0 + N_k); 1 at cold start, -> 0 with evidence."""
        return self.n0 / (self.n0 + self.N_k.to(torch.float32))

    @property
    def m_hat(self) -> Tensor:
        """Damped trust mean m^_k = (1 - lambda_k) * m~_k; exactly 0 where N_k = 0."""
        n = self.N_k.to(torch.float32)
        has_events = self.N_k > 0
        denom = (1.0 - torch.pow(self.beta1, n)).clamp_min(1e-12)
        m_tilde = torch.where(has_events, self.m_k / denom, torch.zeros_like(self.m_k))
        return (1.0 - self.lambda_k) * m_tilde

    @property
    def v_hat(self) -> Tensor:
        """Mixed second moment v^_k = lambda_k * v_k(0) + (1 - lambda_k) * v~_k."""
        n = self.N_k.to(torch.float32)
        has_events = self.N_k > 0
        denom = (1.0 - torch.pow(self.beta2, n)).clamp_min(1e-12)
        v_tilde = torch.where(has_events, self.v_k / denom, torch.zeros_like(self.v_k))
        lam = self.lambda_k
        return lam * self.v_k_0 + (1.0 - lam) * v_tilde

    def forward(self, delta: Tensor) -> Tensor:
        """Consume per-class correction outcomes and return SIGNED safety weights.

        Args:
            delta: shape (num_classes,), values in {-1, 0, +1}. Nonzero entries
                are correction-event outcomes (F4); 0 means no event for that
                class on this call (its state is untouched).

        Returns:
            w_k of shape (num_classes,), dtype float32, SIGNED.
        """
        if delta.shape != (self.num_classes,):
            raise ValueError(
                f"delta must have shape ({self.num_classes},), "
                f"got {tuple(delta.shape)}"
            )
        delta = delta.to(torch.float32)
        valid = (delta == 0) | (delta == 1) | (delta == -1)
        if not bool(valid.all()):
            raise ValueError(
                "delta entries must be correction-event outcomes in "
                f"{{-1, 0, +1}} (memo §3, F4); got {delta.tolist()}"
            )

        self.t += 1
        is_event = delta != 0
        self.N_k.add_(is_event.to(self.N_k.dtype))

        self.m_k.copy_(torch.where(
            is_event, self.beta1 * self.m_k + (1.0 - self.beta1) * delta, self.m_k
        ))
        self.v_k.copy_(torch.where(
            is_event, self.beta2 * self.v_k + (1.0 - self.beta2) * delta * delta, self.v_k
        ))

        w = self.eta * self.eta_k * self.m_hat / (torch.sqrt(self.v_hat) + self.eps)

        assert torch.isfinite(w).all(), (
            "Non-finite safety weight; check eps, v_k(0) positivity, "
            "and the N_k-indexed bias-correction denominators."
        )
        return w

    def reset(self) -> None:
        """Restore the exact cold start: zero EMAs, zero event counts, t = 0.

        After reset, w_k = 0 for every class until new events arrive (Prop 1
        composition: a reset estimator exerts no influence). eta_k and v_k_0
        are immutable priors and are not touched.
        """
        self.m_k.zero_()
        self.v_k.zero_()
        self.N_k.zero_()
        self.t.zero_()
