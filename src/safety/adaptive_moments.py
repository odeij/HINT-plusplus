"""HINT++ Phase 2: Adaptive Moment Safety Signals.

Per-class safety sensitivity via Adam-style moment estimation on the
human correction signal delta_k(t).

Correspondence:
    delta_k(t)  - correction signal               (like Adam's gradient g)
    m_hat_k     - correction consistency           (first moment, bias-corrected)
    v_hat_k     - correction noise                 (second moment, bias-corrected)
    eta_k       - per-class confidence ceiling     (loaded from phase2_init.pt)
    eta         - global learning rate scalar      (default 1.0)
    w_k(t)      - safety weight at time t

Update rules (asymmetric by design: 0 < beta1 < beta2 < 1):
    m_k(t)   = beta1 * m_k(t-1) + (1 - beta1) * delta_k(t)
    v_k(t)   = beta2 * v_k(t-1) + (1 - beta2) * delta_k(t)^2
    m_hat_k  = m_k(t) / (1 - beta1**t)
    v_hat_k  = v_k(t) / (1 - beta2**t)
    w_k(t)   = eta * eta_k * m_hat_k / (sqrt(v_hat_k) + eps)

Initialization (loaded from phase2_init.pt, produced by Sub-step 0C):
    m_k(0)   = 0           cold start; bias correction handles t=1
    v_k(0)   = principled prior from rarity + teacher uncertainty
    eta_k    = per-class teacher confidence ceiling (Sub-step 0A)

The cold-start identity at t=1 with m_k(0)=0:
    m_hat_k = (1-beta1)*delta / (1-beta1) = delta exactly.
The first correction lands in m_hat_k with weight 1.0; downstream phases
can rely on this without a separate "warm-up" code path.

Phase 6 (Hydra integration) supplies init_path via config; for now
callers (tests, scripts, notebooks) pass it explicitly.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class AdaptiveMomentSafety(nn.Module):
    """Per-class safety weights via Adam-style moment estimation."""

    def __init__(
        self,
        init_path: Path | str | None = None,
        num_classes: int = 13,
        beta1: float = 0.7,
        beta2: float = 0.95,
        eps: float = 1e-8,
        eta: float = 1.0,
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

        self.num_classes = num_classes
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eta = eta

        path = Path(init_path)
        if not path.is_file():
            raise FileNotFoundError(f"init_path does not exist: {path}")
        init = torch.load(path, weights_only=False)

        eta_k = init["eta_k"].to(torch.float32)
        m_k_0 = init["m_k_0"].to(torch.float32)
        v_k_0 = init["v_k_0"].to(torch.float32)

        expected = (num_classes,)
        if eta_k.shape != expected or m_k_0.shape != expected \
                or v_k_0.shape != expected:
            raise ValueError(
                f"phase2_init.pt tensors must have shape {expected}, got "
                f"eta_k={tuple(eta_k.shape)}, "
                f"m_k_0={tuple(m_k_0.shape)}, "
                f"v_k_0={tuple(v_k_0.shape)}"
            )
        if (v_k_0 <= 0).any():
            raise ValueError(
                f"v_k(0) must be strictly positive (sqrt is taken in "
                f"forward); min v_k_0 = {float(v_k_0.min()):.3e}"
            )

        # eta_k is the per-class confidence ceiling: immutable prior.
        self.register_buffer("eta_k", eta_k)
        # Running moments. m_k_init / v_k_init kept for reset().
        self.register_buffer("m_k", m_k_0.clone())
        self.register_buffer("v_k", v_k_0.clone())
        self.register_buffer("m_k_init", m_k_0.clone())
        self.register_buffer("v_k_init", v_k_0.clone())
        # Step counter; 0-d long tensor so it persists in state_dict.
        self.register_buffer("t", torch.tensor(0, dtype=torch.long))

    def forward(self, delta: Tensor) -> Tensor:
        """Update moments with a correction signal and return safety weights.

        Args:
            delta: per-class correction signal at the current step,
                shape (num_classes,). Cast to float32 internally.

        Returns:
            Safety weights w_k of shape (num_classes,), dtype float32.
        """
        if delta.shape != (self.num_classes,):
            raise ValueError(
                f"delta must have shape ({self.num_classes},), "
                f"got {tuple(delta.shape)}"
            )
        delta = delta.to(self.m_k.dtype)

        self.t += 1
        t = int(self.t.item())

        self.m_k.mul_(self.beta1).add_(delta, alpha=1.0 - self.beta1)
        self.v_k.mul_(self.beta2).addcmul_(
            delta, delta, value=1.0 - self.beta2
        )

        # beta**t in Python double; underflows to 0.0 for large t, which
        # collapses the bias-correction divisor to 1.0 (exact) -- safe.
        m_hat = self.m_k / (1.0 - self.beta1 ** t)
        v_hat = self.v_k / (1.0 - self.beta2 ** t)

        w = self.eta * self.eta_k * m_hat / (torch.sqrt(v_hat) + self.eps)

        assert torch.isfinite(w).all(), (
            "Non-finite safety weight; check eps, v_k(0) positivity, "
            "and delta magnitudes."
        )
        return w

    def reset(self) -> None:
        """Restore m_k, v_k to their initialization values; reset t to 0.

        For test isolation and Phase 6 multi-domain meta-learning, where
        each new domain begins from the shared prior. eta_k is not reset
        -- it is an immutable per-class ceiling, not a running estimate.
        """
        self.m_k.copy_(self.m_k_init)
        self.v_k.copy_(self.v_k_init)
        self.t.zero_()
