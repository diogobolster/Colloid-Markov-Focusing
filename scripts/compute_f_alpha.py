"""Cutoff-free focusing curve F(alpha), rear alignment A, and angular travel for the center/corner suite.

Run from the Colloid-Markov-Focusing repository root on the machine that holds the production outputs:

    python scripts/compute_f_alpha.py

Reads  outputs/openfoam_full_suite/trajectory_library_refinement_<profile>.npz  (central-core ensemble)
Writes outputs/f_alpha/f_alpha_summary.csv   (one row per condition: N, F30, F(alpha) at fixed alphas, A, medians)
       outputs/f_alpha/f_alpha_curves.csv    (F(alpha) on a 1-degree grid, one column per condition)
       outputs/f_alpha/table_row_snippets.txt (LaTeX fragments for the manuscript)
       outputs/figures/figure_15_f_alpha.png (single panel, F(alpha) versus alpha)

Event selection is identical to scripts/create_focused_mobile_state_metrics.py: no-DLVO uses completed
center-grain near-surface releases, all other conditions use completed center-grain secondary-minimum releases.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUITE = ROOT / "outputs" / "openfoam_full_suite"
OUT = ROOT / "outputs" / "f_alpha"
FIGURES = ROOT / "outputs" / "figures"

PROFILES = [
    ("neutral_resolved", "No DLVO"),
    ("unfavorable_6mM_z70", "6 mM, -70 mV"),
    ("unfavorable_20mM_z70", "20 mM, -70 mV"),
    ("unfavorable_50mM_z70", "50 mM, -70 mV"),
    ("unfavorable_100mM_z70", "100 mM, -70 mV"),
    ("unfavorable_50mM_z50", "50 mM, -50 mV"),
    ("unfavorable_50mM_z30", "50 mM, -30 mV"),
    ("unfavorable_100mM_z30", "100 mM, -30 mV"),
    ("unfavorable_50mM_z70_100xD", "50 mM, -70 mV, 100x D"),
]
PLOT_PROFILES = ["neutral_resolved", "unfavorable_6mM_z70", "unfavorable_20mM_z70", "unfavorable_50mM_z70",
                 "unfavorable_100mM_z70", "unfavorable_50mM_z70_100xD"]
ALPHAS_REPORT = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0)
ALPHA_GRID = np.arange(0.0, 180.0 + 0.5, 1.0)


def abs_angle_deg(theta: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(np.arctan2(np.sin(theta), np.cos(theta))))


def load_events(profile: str) -> dict[str, np.ndarray]:
    path = SUITE / f"trajectory_library_refinement_{profile}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path)
    if profile.startswith("neutral"):
        mask = (d["exited"] & ~d["censored"] & (d["center_interceptions"] > 0)
                & (d["collector_exit"] == 1) & np.isfinite(d["theta_exit"]))
        theta = d["theta_exit"][mask]
        net = cum = None
    else:
        mask = (d["exited"] & ~d["censored"] & (d["center_well_interceptions"] > 0)
                & (d["collector_well_exit"] == 1) & np.isfinite(d["theta_well_exit"]))
        theta = d["theta_well_exit"][mask]
        net = np.degrees(d["well_net_angular_travel"][mask]) if "well_net_angular_travel" in d.files else None
        cum = np.degrees(d["well_angular_travel"][mask]) if "well_angular_travel" in d.files else None
    return {"abs_theta_deg": abs_angle_deg(theta), "net_deg": net, "cum_deg": cum}


def f_alpha(abs_theta_deg: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    if abs_theta_deg.size == 0:
        return np.full(alphas.shape, np.nan)
    sorted_theta = np.sort(abs_theta_deg)
    return np.searchsorted(sorted_theta, alphas, side="right") / sorted_theta.size


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {}
    for profile, label in PROFILES:
        try:
            ev = load_events(profile)
        except FileNotFoundError as exc:
            print(f"skip {profile}: {exc}")
            continue
        th = ev["abs_theta_deg"]
        curve = f_alpha(th, ALPHA_GRID)
        curves[profile] = curve
        row: dict[str, object] = {
            "profile": profile,
            "label": label,
            "N_released": int(th.size),
            "F30": float(np.mean(th <= 30.0)) if th.size else float("nan"),
            "median_abs_theta_deg": float(np.median(th)) if th.size else float("nan"),
            "A_mean_cos": float(np.mean(np.cos(np.radians(th)))) if th.size else float("nan"),
        }
        for a in ALPHAS_REPORT:
            row[f"F({a:g})"] = float(np.mean(th <= a)) if th.size else float("nan")
        for key, arr in (("net", ev["net_deg"]), ("cum", ev["cum_deg"])):
            if arr is None or arr.size == 0:
                row[f"median_dtheta_{key}_deg"] = float("nan")
                row[f"mean_dtheta_{key}_deg"] = float("nan")
            else:
                row[f"median_dtheta_{key}_deg"] = float(np.median(np.abs(arr))) if key == "net" else float(np.median(arr))
                row[f"mean_dtheta_{key}_deg"] = float(np.mean(np.abs(arr))) if key == "net" else float(np.mean(arr))
        rows.append(row)
        print(f"{label:>26s}: N={row['N_released']:6d}  F30={row['F30']:.3f}  A={row['A_mean_cos']:.3f}  "
              f"median|theta|={row['median_abs_theta_deg']:.2f}  median|dnet|={row['median_dtheta_net_deg']:.1f}  "
              f"median cum={row['median_dtheta_cum_deg']:.1f}")

    with (OUT / "f_alpha_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "f_alpha_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha_deg"] + list(curves.keys()))
        for i, a in enumerate(ALPHA_GRID):
            w.writerow([f"{a:g}"] + [f"{curves[p][i]:.5f}" for p in curves])

    snippets = ["% A for Table focused-mobile-state-metrics (mean cos of completed center releases):"]
    for row in rows:
        snippets.append(f"% {row['label']}: A = {row['A_mean_cos']:.2f}, F30 = {row['F30']:.3f}, N = {row['N_released']}")
    snippets.append("% F(alpha) at fixed cutoffs, for the text:")
    for row in rows:
        vals = ", ".join(f"F({a:g})={row[f'F({a:g})']:.3f}" for a in ALPHAS_REPORT)
        snippets.append(f"% {row['label']}: {vals}")
    snippets.append("% angular travel during residence (degrees), medians of |net| and cumulative:")
    for row in rows:
        snippets.append(f"% {row['label']}: |net| = {row['median_dtheta_net_deg']:.1f}, cum = {row['median_dtheta_cum_deg']:.1f}")
    (OUT / "table_row_snippets.txt").write_text("\n".join(snippets) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8})
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)
    styles = {
        "neutral_resolved": ("#4b5563", "-"),
        "unfavorable_6mM_z70": ("#9ca3af", "--"),
        "unfavorable_20mM_z70": ("#60a5fa", "-."),
        "unfavorable_50mM_z70": ("#2563eb", "-"),
        "unfavorable_100mM_z70": ("#7c3aed", "-"),
        "unfavorable_50mM_z70_100xD": ("#dc2626", ":"),
    }
    labels = dict(PROFILES)
    for profile in PLOT_PROFILES:
        if profile not in curves:
            continue
        color, ls = styles[profile]
        ax.plot(ALPHA_GRID, curves[profile], color=color, ls=ls, lw=1.6, label=labels[profile])
    ax.axvline(30.0, color="black", lw=0.8, ls=":")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"cutoff $\alpha$ (degrees from rear stagnation point)")
    ax.set_ylabel(r"$F(\alpha)=\mathrm{Pr}(|\theta_{\mathrm{rel}}| \leq \alpha \mid \mathrm{release})$")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_15_f_alpha.png")
    print("wrote", OUT / "f_alpha_summary.csv", "and", FIGURES / "figure_15_f_alpha.png")


if __name__ == "__main__":
    main()
