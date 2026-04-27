"""
HINT++ Phase 2 -- Sub-step 0C: master initialization table.

Joins eta_k (from 0A), v_k(0) (from 0B), and m_k(0) = 0 into a single
deployable table. Emits both:

    table3_master_init.csv  -- human-readable, canonical class order
    phase2_init.pt          -- torch dict (float32 tensors + class_names)

m_k(0) = 0 is honest cold start: at deployment t = 0 no human correction
has arrived. Bias correction m_hat_k = m_k(t) / (1 - beta1^t) upweights
the first correction at t = 1 by exactly 1 / (1 - beta1) -- with
beta1 = 0.7 that is 1 / 0.3 = 3.33x. The cold-start zero is not a stand-
in for missing data; the bias-correction term turns it into a clean
identity that lets the first correction set the running estimate.

This file is the input src/safety/adaptive_moments.py loads at module
construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT_DIR = Path(__file__).resolve().parents[1] / "results"

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
NUM_CLASSES = 13


def load_priors() -> tuple[np.ndarray, np.ndarray]:
    eta_dict = json.loads((OUT_DIR / "eta_k.json").read_text())
    v_dict = json.loads((OUT_DIR / "v_k_0.json").read_text())
    eta = np.array([eta_dict[c] for c in CLASS_NAMES], dtype=np.float64)
    v0 = np.array([v_dict[c] for c in CLASS_NAMES], dtype=np.float64)
    return eta, v0


def save_master(eta: np.ndarray, v0: np.ndarray, m0: np.ndarray) -> None:
    pd.DataFrame({
        "class": CLASS_NAMES,
        "eta_k": eta,
        "v_k_0": v0,
        "m_k_0": m0,
    }).to_csv(OUT_DIR / "table3_master_init.csv", index=False)

    torch.save({
        "class_names": list(CLASS_NAMES),
        "eta_k": torch.tensor(eta, dtype=torch.float32),
        "v_k_0": torch.tensor(v0, dtype=torch.float32),
        "m_k_0": torch.tensor(m0, dtype=torch.float32),
    }, OUT_DIR / "phase2_init.pt")


def verify(eta: np.ndarray, v0: np.ndarray, m0: np.ndarray) -> None:
    print("\n=== Sub-step 0C verification ===")
    checks: list[tuple[str, bool, str]] = []

    df = pd.read_csv(OUT_DIR / "table3_master_init.csv")
    c1 = len(df) == NUM_CLASSES and df["class"].tolist() == CLASS_NAMES
    checks.append(("1. CSV has all 13 classes in canonical order",
                   c1, f"rows = {len(df)}"))

    eta_src = json.loads((OUT_DIR / "eta_k.json").read_text())
    eta_csv = dict(zip(df["class"], df["eta_k"]))
    c2 = all(abs(eta_src[c] - float(eta_csv[c])) < 1e-12 for c in CLASS_NAMES)
    checks.append(("2. eta_k matches eta_k.json (per-class)",
                   c2, "exact" if c2 else "MISMATCH"))

    v_src = json.loads((OUT_DIR / "v_k_0.json").read_text())
    v_csv = dict(zip(df["class"], df["v_k_0"]))
    c3 = all(abs(v_src[c] - float(v_csv[c])) < 1e-12 for c in CLASS_NAMES)
    checks.append(("3. v_k_0 matches v_k_0.json (per-class)",
                   c3, "exact" if c3 else "MISMATCH"))

    c4 = bool((m0 == 0).all()) and bool((df["m_k_0"] == 0).all())
    checks.append(("4. m_k(0) = 0 for all classes (cold start)",
                   c4, f"max |m_k(0)| = {float(np.abs(m0).max()):.0e}"))

    pt = torch.load(OUT_DIR / "phase2_init.pt", weights_only=False)
    c5 = (
        pt["class_names"] == CLASS_NAMES
        and pt["eta_k"].shape == (NUM_CLASSES,)
        and pt["eta_k"].dtype == torch.float32
        and pt["v_k_0"].dtype == torch.float32
        and pt["m_k_0"].dtype == torch.float32
        and torch.allclose(pt["eta_k"],
                           torch.tensor(eta, dtype=torch.float32))
        and torch.allclose(pt["v_k_0"],
                           torch.tensor(v0, dtype=torch.float32))
        and torch.allclose(pt["m_k_0"],
                           torch.tensor(m0, dtype=torch.float32))
    )
    checks.append(("5. phase2_init.pt round-trips (names, shape, dtype, values)",
                   c5, "all match" if c5 else "MISMATCH"))

    for name, ok, info in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Sub-step 0C FAILED verification")


def main() -> None:
    eta, v0 = load_priors()
    m0 = np.zeros(NUM_CLASSES, dtype=np.float64)
    save_master(eta, v0, m0)

    df = pd.read_csv(OUT_DIR / "table3_master_init.csv")
    print("Master initialization table:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    verify(eta, v0, m0)
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
