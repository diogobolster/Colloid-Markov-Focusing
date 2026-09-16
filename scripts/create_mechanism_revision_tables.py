#!/usr/bin/env python3
"""Create mechanism-first revision tables for the focused manuscript.

The review-driven additions here are deliberately narrow:

* separate conditional focusing from injected-particle focused yield;
* compare matched near-surface event definitions against potential-defined
  secondary-minimum events; and
* summarize the boundary/rejection rule as a reproducibility algorithm.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY = ROOT / "outputs" / "openfoam_full_suite" / "openfoam_full_refinement_summary.csv"
OUT = ROOT / "outputs" / "science_synthesis"
FIG_OUT = ROOT / "outputs" / "figures"
TABLE_OUT = ROOT / "outputs" / "tables"

SELECTED = [
    "neutral_resolved",
    "unfavorable_6mM_z70",
    "unfavorable_50mM_z70",
    "unfavorable_100mM_z70",
    "unfavorable_100mM_z30",
    "unfavorable_50mM_z70_100xD",
]

LABELS = {
    "neutral_resolved": "No DLVO",
    "unfavorable_6mM_z70": "6 mM, -70 mV",
    "unfavorable_50mM_z70": "50 mM, -70 mV",
    "unfavorable_100mM_z70": "100 mM, -70 mV",
    "unfavorable_100mM_z30": "100 mM, -30 mV",
    "unfavorable_50mM_z70_100xD": "50 mM, -70 mV, 100D",
}

COLORS = {
    "neutral_resolved": "#2563eb",
    "unfavorable_6mM_z70": "#7dd3fc",
    "unfavorable_50mM_z70": "#7c3aed",
    "unfavorable_100mM_z70": "#581c87",
    "unfavorable_100mM_z30": "#9333ea",
    "unfavorable_50mM_z70_100xD": "#0f766e",
}


def read_rows() -> dict[str, dict[str, str]]:
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        return {row["profile"]: row for row in csv.DictReader(handle)}


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def i(row: dict[str, str], key: str) -> int:
    try:
        return int(round(float(row[key])))
    except (KeyError, TypeError, ValueError):
        return 0


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value)
    for src, dst in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(src, dst)
    return text


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mechanism_rows(rows_by_profile: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for profile in SELECTED:
        row = rows_by_profile[profile]
        n = i(row, "particles")
        near_event = i(row, "center_intercepted_count")
        near_release = i(row, "center_release_count")
        near_unresolved = i(row, "center_censored_count")
        near_f30 = f(row, "center_release_theta30_fraction")
        well_event = i(row, "center_well_intercepted_count")
        use_well = well_event > 0
        event = well_event if use_well else near_event
        release = i(row, "center_well_release_count") if use_well else near_release
        unresolved = i(row, "center_well_censored_count") if use_well else near_unresolved
        f30 = f(row, "center_well_release_theta30_fraction") if use_well else near_f30
        median_theta = f(row, "center_well_release_median_theta_deg") if use_well else f(row, "center_release_median_theta_deg")
        median_tau = f(row, "center_well_time_median_s") if use_well else f(row, "center_near_time_median_s")
        q_rel = release / event if event else float("nan")
        y_event = q_rel * f30 if math.isfinite(f30) and math.isfinite(q_rel) else float("nan")
        p_event = event / n if n else float("nan")
        p_injected_focused = release * f30 / n if n and math.isfinite(f30) else float("nan")
        near_y_injected = near_release * near_f30 / n if n and math.isfinite(near_f30) else float("nan")
        out.append(
            {
                "profile": profile,
                "case": LABELS[profile],
                "n_injected": n,
                "event_definition": "center well" if use_well else "center near surface",
                "n_event": event,
                "p_event": p_event,
                "q_rel": q_rel,
                "f30": f30,
                "y30_event": y_event,
                "p_injected_focused": p_injected_focused,
                "unresolved_fraction_event": unresolved / event if event else float("nan"),
                "median_theta_deg": median_theta,
                "median_residence_s": median_tau,
                "near_surface_f30": near_f30,
                "near_surface_p_injected_focused": near_y_injected,
                "well_f30": f(row, "center_well_release_theta30_fraction"),
            }
        )
    return out


def write_yield_table(rows: list[dict[str, object]]) -> Path:
    path = TABLE_OUT / "mechanism_injected_yield_table.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Mechanism-first center/corner focusing metrics. \(F_{30}\) is conditional on completed release from the relevant event, \(Y_{30}^{\rm event}=q_{\rm rel}F_{30}\) is event-entry conditioned, and \(P_{\rm inj,30}\) is the unconditional focused-release probability per injected particle in the central-core injection ensemble.}",
        r"\label{tab:mechanism-injected-yield}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"case & \(N_{\rm inj}\) & \(N_{\rm event}\) & \(P_{\rm event}\) & \(q_{\rm rel}\) & \(F_{30}\) & \(Y_{30}^{\rm event}\) & \(P_{\rm inj,30}\) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['case'])} & "
            f"{int(row['n_injected'])} & {int(row['n_event'])} & "
            f"{fmt(float(row['p_event']))} & {fmt(float(row['q_rel']))} & "
            f"{fmt(float(row['f30']))} & {fmt(float(row['y30_event']))} & "
            f"{fmt(float(row['p_injected_focused']))} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_matched_event_table(rows: list[dict[str, object]]) -> Path:
    path = TABLE_OUT / "matched_event_definition_table.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Matched event-definition comparison for the center/corner cell. The near-surface columns use the same geometric shell for all cases; the well columns use the potential-defined secondary-minimum event where it exists.}",
        r"\label{tab:matched-event-definition}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"case & \(F_{30}^{\rm ns}\) & \(P_{\rm inj,30}^{\rm ns}\) & \(F_{30}^{\rm well}\) & median \(\theta_{\rm well}\) & median \(\tau_{\rm well}\) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['case'])} & "
            f"{fmt(float(row['near_surface_f30']))} & "
            f"{fmt(float(row['near_surface_p_injected_focused']))} & "
            f"{fmt(float(row['well_f30']))} & "
            f"{fmt(float(row['median_theta_deg']), 2)} & "
            f"{fmt(float(row['median_residence_s']), 3)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_boundary_algorithm_table() -> Path:
    path = TABLE_OUT / "boundary_algorithm_table.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Boundary and barrier-respecting update used for homogeneous unfavorable surfaces. This is a numerical rule for nonattaching unfavorable collectors, not an empirical attachment model.}",
        r"\label{tab:boundary-algorithm}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.68\linewidth}}",
        r"\hline",
        r"step & rule \\",
        r"\hline",
        r"propose & Advance one adaptive substep with resolved advection, wall-corrected DLVO drift, thermal drift, and anisotropic Brownian displacement. \\",
        r"uphill inward test & If \(h^\ast<h^n\) and \(U(h^\ast)>U(h^n)\), accept with \(p_{\rm acc}=\min[1,\exp(-(U(h^\ast)-U(h^n))/k_BT)]\). Otherwise accept the proposal. \\",
        r"rejection & If the uphill inward move is rejected, keep \(\mathbf X^{n+1}=\mathbf X^n\) for that substep and continue with the next substep. \\",
        r"excluded contact & If an accepted nonattaching unfavorable proposal has \(h^\ast<h_c\), project to \(h_c\) along the local outward normal before reevaluating velocity. \\",
        r"favorable contrast & Favorable homogeneous collectors attach irreversibly at \(h<h_c\); unfavorable homogeneous collectors never attach in this manuscript. \\",
        r"validation & Off-flow Boltzmann equilibrium, timestep/normal-step/raster perturbations, \(h_{\rm ns}\)/\(h_c\) sensitivity, and completion curves are reported; removing the barrier rule is treated as a nonphysical stress test rather than a production model because it permits artificial barrier crossing under the intact-barrier hypothesis. \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_figure(rows: list[dict[str, object]]) -> Path:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    profiles = [str(row["profile"]) for row in rows]
    labels = [str(row["case"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), dpi=240)

    axes[0].bar(x, [float(row["f30"]) for row in rows], color=[COLORS[p] for p in profiles])
    axes[0].axhline(float(rows[0]["f30"]), color="#334155", lw=1.0, ls="--")
    axes[0].set_ylabel(r"conditional $F_{30}$")
    axes[0].set_ylim(0, 1.04)
    axes[0].set_title("Release concentration")

    axes[1].bar(x, [float(row["p_event"]) for row in rows], color=[COLORS[p] for p in profiles])
    axes[1].set_ylabel(r"$P_{\rm event}$")
    axes[1].set_title("Event supply")

    axes[2].bar(x, [float(row["p_injected_focused"]) for row in rows], color=[COLORS[p] for p in profiles])
    axes[2].axhline(float(rows[0]["p_injected_focused"]), color="#334155", lw=1.0, ls="--")
    axes[2].set_ylabel(r"$P_{\rm inj,30}$")
    axes[2].set_title("Focused releases per injected particle")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", color="#cbd5e1", lw=0.6, alpha=0.7)
        ax.tick_params(axis="both", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = FIG_OUT / "mechanism_injected_yield_controls.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    rows = mechanism_rows(read_rows())
    write_csv(OUT / "mechanism_injected_yield_summary.csv", rows)
    write_yield_table(rows)
    write_matched_event_table(rows)
    write_boundary_algorithm_table()
    make_figure(rows)


if __name__ == "__main__":
    main()
