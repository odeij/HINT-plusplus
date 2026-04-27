"""
HINT++ Phase 2 -- Sub-step 0B: per-class frequency prior freq_k and v_k(0).

freq_k is the fraction of voxelized labels of class k across the S3DIS
training split (Areas 1, 2, 3, 4, 6). The teacher operates on voxelized
data, so the frequency prior must match that resolution.

Initial variance combines two independent uncertainty signals on a common
scale:

    r_k    = (1/freq_k)  / max_j (1/freq_j)        # normalized rarity
    u_k    = (1 - eta_k) / max_j (1 - eta_j)       # normalized teacher uncertainty
    v_k(0) = alpha * r_k + (1 - alpha) * u_k       # alpha = 0.5

Both r_k, u_k lie in (0, 1] so v_k(0) lies in (0, 1] and alpha = 0.5 is
LITERAL equal weighting. An earlier draft mixed raw 1/freq_k (range
~4-200) with raw 1 - eta_k (range ~0.01-0.26); the rarity term swamped
the uncertainty term and alpha = 0.5 was decorative. Independence
between r_k and u_k is enforced as a hard gate, not just reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

S3DIS_ROOT = Path(
    "/home/ahmad/Desktop/HINT++/s3dis-compressed/s3dis-compressed"
)
TRAIN_AREAS = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"]
OUT_DIR = Path(__file__).resolve().parents[1] / "results"

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
NUM_CLASSES = 13
ALPHA = 0.5


def count_voxels() -> tuple[np.ndarray, int, int]:
    """Sum voxel counts per class across all training rooms."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    n_rooms = 0
    n_voxels = 0
    for area in TRAIN_AREAS:
        area_dir = S3DIS_ROOT / area
        if not area_dir.is_dir():
            raise RuntimeError(f"Missing area directory: {area_dir}")
        for room in sorted(p for p in area_dir.iterdir() if p.is_dir()):
            seg = np.load(room / "segment.npy").reshape(-1)
            # Defensive: only count valid labels.
            mask = (seg >= 0) & (seg < NUM_CLASSES)
            n_voxels += int(mask.sum())
            counts += np.bincount(seg[mask].astype(np.int64),
                                  minlength=NUM_CLASSES)
            n_rooms += 1
    return counts, n_rooms, n_voxels


def build_v_k_0(
    freq: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r_k, u_k, v_k(0)), all float64.

    Both signals are max-normalized so the rarest class has r_k = 1.0 and
    the most teacher-uncertain class has u_k = 1.0. alpha = 0.5 then
    weights the two signals equally on a common scale.
    """
    inv_freq = 1.0 / freq.astype(np.float64)
    one_minus_eta = 1.0 - eta.astype(np.float64)
    r = inv_freq / inv_freq.max()
    u = one_minus_eta / one_minus_eta.max()
    v0 = ALPHA * r + (1.0 - ALPHA) * u
    return r, u, v0


def save_outputs(
    freq: np.ndarray,
    eta: np.ndarray,
    r: np.ndarray,
    u: np.ndarray,
    v0: np.ndarray,
) -> None:
    (OUT_DIR / "freq_k.json").write_text(
        json.dumps({c: float(v) for c, v in zip(CLASS_NAMES, freq)}, indent=2)
    )
    (OUT_DIR / "v_k_0.json").write_text(
        json.dumps({c: float(v) for c, v in zip(CLASS_NAMES, v0)}, indent=2)
    )
    df = pd.DataFrame({
        "class": CLASS_NAMES,
        "freq_k": freq,
        "inv_freq_k": 1.0 / freq.astype(np.float64),
        "r_k": r,
        "one_minus_eta_k": 1.0 - eta,
        "u_k": u,
        "v_k_0": v0,
    })
    df.to_csv(OUT_DIR / "table2_v_k_0.csv", index=False)

    # Figure 2: bar chart of freq_k sorted descending, colored by tier.
    order = np.argsort(freq)[::-1]
    sorted_freq = freq[order]
    sorted_names = [CLASS_NAMES[i] for i in order]
    colors = []
    for v in sorted_freq:
        if v < 0.02:
            colors.append("crimson")
        elif v < 0.05:
            colors.append("goldenrod")
        else:
            colors.append("steelblue")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sorted_names, sorted_freq, color=colors, edgecolor="black")
    ax.set_ylabel("freq_k (training-split voxel fraction)")
    ax.set_title("S3DIS class frequencies (Areas 1, 2, 3, 4, 6)")
    ax.tick_params(axis="x", rotation=35)
    for i, (n, v) in enumerate(zip(sorted_names, sorted_freq)):
        ax.text(i, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure2_freq.png", dpi=150)
    plt.close(fig)

    # Figure 3: scatter freq_k (log) vs eta_k -- proves independence.
    r_fe, p_fe = pearsonr(freq, eta)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(freq, eta, s=70, c="steelblue", edgecolor="black", zorder=3)
    for i, c in enumerate(CLASS_NAMES):
        ax.annotate(c, (freq[i], eta[i]), textcoords="offset points",
                    xytext=(5, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("freq_k (log scale)")
    ax.set_ylabel(r"$\eta_k$")
    ax.set_title(
        f"freq_k vs $\\eta_k$  -  Pearson r = {r_fe:.3f}  (p = {p_fe:.2e})"
    )
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure3_freq_vs_eta.png", dpi=150)
    plt.close(fig)


def verify(
    freq: np.ndarray,
    eta: np.ndarray,
    r: np.ndarray,
    u: np.ndarray,
    v0: np.ndarray,
) -> None:
    print("\n=== Sub-step 0B verification ===")
    checks: list[tuple[str, bool, str]] = []

    s = float(freq.sum())
    c1 = abs(s - 1.0) < 0.01
    checks.append(("1. sum(freq_k) ~= 1.0 (+/-0.01)",
                   c1, f"sum = {s:.6f}"))

    c2 = bool((v0 > 0).all())
    checks.append(("2. v_k(0) > 0 for all classes",
                   c2, f"min v_k(0) = {v0.min():.6e}"))

    # Informative-spread check: rare-and-uncertain classes get an
    # initial variance at least an order of magnitude above the
    # common-and-confident floor. Replaces an earlier beam-vs-floor
    # check that encoded a wrong assumption about S3DIS frequencies.
    spread = float(v0.max() / v0.min())
    argmax_name = CLASS_NAMES[int(np.argmax(v0))]
    argmin_name = CLASS_NAMES[int(np.argmin(v0))]
    c3 = spread >= 10.0
    checks.append(("3. max(v_k(0)) / min(v_k(0)) >= 10 (informative prior)",
                   c3, f"max = {v0.max():.4f} ({argmax_name}), "
                       f"min = {v0.min():.4f} ({argmin_name}), "
                       f"ratio = {spread:.2f}x"))

    rho_eta_v, p_eta_v = pearsonr(eta, v0)
    c4 = rho_eta_v < 0
    checks.append(("4. Pearson(eta_k, v_k(0)) < 0",
                   c4, f"r = {rho_eta_v:.4f}, p = {p_eta_v:.2e}"))

    # Independence of the two signals -- defends alpha = 0.5 numerically
    # rather than relying on a printed diagnostic the reader can ignore.
    rho_ru, p_ru = pearsonr(r, u)
    c5 = abs(rho_ru) < 0.7
    checks.append(("5. |Pearson(r_k, u_k)| < 0.7 (signals independent, "
                   "alpha = 0.5 honest)",
                   c5, f"r = {rho_ru:+.4f}, p = {p_ru:.2e}"))

    for name, ok, info in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Sub-step 0B FAILED verification")


def report_signal_diagnostics(freq: np.ndarray, eta: np.ndarray) -> None:
    """Extra correlations on raw signals. Hard gate is verification 5."""
    print("\n=== Raw-signal correlations (diagnostic only) ===")
    one_minus_eta = 1.0 - eta
    inv_freq = 1.0 / freq.astype(np.float64)
    r1, p1 = pearsonr(freq, one_minus_eta)
    r2, p2 = pearsonr(inv_freq, one_minus_eta)
    r3, p3 = pearsonr(np.log(freq), one_minus_eta)
    print(f"  rho(freq_k,      1 - eta_k) = {r1:+.4f}  (p = {p1:.2e})")
    print(f"  rho(1/freq_k,    1 - eta_k) = {r2:+.4f}  (p = {p2:.2e})")
    print(f"  rho(log(freq_k), 1 - eta_k) = {r3:+.4f}  (p = {p3:.2e})")


def main() -> None:
    print(f"Counting voxels in {len(TRAIN_AREAS)} training areas...")
    counts, n_rooms, n_voxels = count_voxels()
    print(f"Processed {n_rooms} rooms, {n_voxels:,} labelled voxels")
    freq = counts / counts.sum()

    eta_dict = json.loads((OUT_DIR / "eta_k.json").read_text())
    eta = np.array([eta_dict[c] for c in CLASS_NAMES], dtype=np.float64)

    r, u, v0 = build_v_k_0(freq, eta)
    save_outputs(freq, eta, r, u, v0)

    df = pd.read_csv(OUT_DIR / "table2_v_k_0.csv")
    print("\nPer-class results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    verify(freq, eta, r, u, v0)
    report_signal_diagnostics(freq, eta)
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
