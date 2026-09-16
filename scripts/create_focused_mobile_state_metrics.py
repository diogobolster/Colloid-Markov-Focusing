#!/usr/bin/env python3
"""Create focused-mobile-state metric table and figure."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUITE = ROOT / "outputs" / "openfoam_full_suite"
OUT = ROOT / "outputs" / "focused_mobile_state_metrics"
FIGURES = ROOT / "outputs" / "figures"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

PROFILES = [
    "neutral_resolved",
    "unfavorable_6mM_z70",
    "unfavorable_20mM_z70",
    "unfavorable_50mM_z70",
    "unfavorable_100mM_z70",
    "unfavorable_50mM_z30",
    "unfavorable_100mM_z30",
    "unfavorable_50mM_z70_100xD",
]

LABELS = {
    "neutral_resolved": "No DLVO",
    "unfavorable_6mM_z70": "6 mM, -70 mV",
    "unfavorable_20mM_z70": "20 mM, -70 mV",
    "unfavorable_50mM_z70": "50 mM, -70 mV",
    "unfavorable_100mM_z70": "100 mM, -70 mV",
    "unfavorable_50mM_z30": "50 mM, -30 mV",
    "unfavorable_100mM_z30": "100 mM, -30 mV",
    "unfavorable_50mM_z70_100xD": "50 mM, -70 mV, 100x D",
}

SHORT = {
    "neutral_resolved": "No\nDLVO",
    "unfavorable_6mM_z70": "6\nmM",
    "unfavorable_20mM_z70": "20\nmM",
    "unfavorable_50mM_z70": "50\nmM",
    "unfavorable_100mM_z70": "100\nmM",
    "unfavorable_50mM_z30": "50 mM\n-30 mV",
    "unfavorable_100mM_z30": "100 mM\n-30 mV",
    "unfavorable_50mM_z70_100xD": "50 mM\n100xD",
}

COLORS = {
    "neutral_resolved": "#6b7280",
    "unfavorable_6mM_z70": "#64748b",
    "unfavorable_20mM_z70": "#0f766e",
    "unfavorable_50mM_z70": "#2563eb",
    "unfavorable_100mM_z70": "#1d4ed8",
    "unfavorable_50mM_z30": "#65a30d",
    "unfavorable_100mM_z30": "#7c3aed",
    "unfavorable_50mM_z70_100xD": "#c026d3",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt(value: object, digits: int | None = None) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return tex_escape(value)
    if not np.isfinite(x):
        return "--"
    if digits is None:
        return f"{x:g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        ("label", "case", None),
        ("q_rel", "$q_{\\rm rel}$", 3),
        ("F30", "$F_{30}$", 3),
        ("Y30", "$Y_{30}$", 3),
        ("unresolved_fraction", "unresolved", 3),
        ("G30", "$G_{30}$", 2),
        ("Phi30", "$\\Phi_{30}$", 2),
        ("rear_alignment_A", "$A=\\langle\\cos\\theta\\rangle$", 2),
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Focused-mobile-state metrics for center/corner-cell events. \(F_{30}\) measures angular concentration conditional on completed release, \(q_{\rm rel}\) is the mobile release completion fraction, and \(Y_{30}=q_{\rm rel}F_{30}\) is the entry-conditioned focused mobile yield.}",
        r"\label{tab:focused-mobile-state-metrics}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        " & ".join(header for _, header, _ in columns) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(fmt(row[key], digits) for key, _, digits in columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def abs_angle(theta: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(np.arctan2(np.sin(theta), np.cos(theta))))


def event_angles(profile: str) -> np.ndarray:
    d = np.load(SUITE / f"trajectory_library_refinement_{profile}.npz")
    if profile.startswith("neutral"):
        mask = (
            d["exited"]
            & ~d["censored"]
            & (d["center_interceptions"] > 0)
            & (d["collector_exit"] == 1)
            & np.isfinite(d["theta_exit"])
        )
        return abs_angle(d["theta_exit"][mask])
    mask = (
        d["exited"]
        & ~d["censored"]
        & (d["center_well_interceptions"] > 0)
        & (d["collector_well_exit"] == 1)
        & np.isfinite(d["theta_well_exit"])
    )
    return abs_angle(d["theta_well_exit"][mask])


def metrics() -> list[dict[str, object]]:
    summary = {row["profile"]: row for row in read_csv(SUITE / "openfoam_full_refinement_summary.csv")}
    f30_reference = float(summary["neutral_resolved"]["center_release_theta30_fraction"])
    rows: list[dict[str, object]] = []
    for profile in PROFILES:
        row = summary[profile]
        if profile.startswith("neutral"):
            entries = float(row["center_intercepted_count"])
            releases = float(row["center_release_count"])
            unresolved = float(row["center_censored_count"])
            f30 = float(row["center_release_theta30_fraction"])
        else:
            entries = float(row["center_well_intercepted_count"])
            releases = float(row["center_well_release_count"])
            unresolved = float(row["center_well_censored_count"])
            f30 = float(row["center_well_release_theta30_fraction"])
        q_rel = releases / entries if entries > 0 else float("nan")
        unresolved_fraction = unresolved / entries if entries > 0 else float("nan")
        y30 = q_rel * f30 if np.isfinite(q_rel) and np.isfinite(f30) else float("nan")
        angles = event_angles(profile)
        alignment = float(np.mean(np.cos(np.radians(angles)))) if angles.size else float("nan")
        rows.append(
            {
                "profile": profile,
                "label": LABELS[profile],
                "event_entries": int(entries),
                "released": int(releases),
                "unresolved": int(unresolved),
                "q_rel": q_rel,
                "F30": f30,
                "Y30": y30,
                "unresolved_fraction": unresolved_fraction,
                "G30": f30 / f30_reference if f30_reference > 0 and np.isfinite(f30) else float("nan"),
                "Phi30": (f30 - f30_reference) / (1.0 - f30_reference) if np.isfinite(f30) else float("nan"),
                "rear_alignment_A": alignment,
            }
        )
    return rows


def plot(rows: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.9), dpi=220)
    alpha = np.arange(0, 181, 5)
    curve_profiles = [
        "neutral_resolved",
        "unfavorable_6mM_z70",
        "unfavorable_50mM_z70",
        "unfavorable_100mM_z70",
        "unfavorable_50mM_z70_100xD",
    ]
    ax = axes[0, 0]
    for profile in curve_profiles:
        angles = event_angles(profile)
        values = [float(np.mean(angles <= a)) if angles.size else np.nan for a in alpha]
        ax.plot(alpha, values, lw=2.0, color=COLORS[profile], label=LABELS[profile])
    ax.axvline(30, color="#111827", lw=1.0, ls="--")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"cone half-angle $\alpha$ (deg)")
    ax.set_ylabel(r"$F(\alpha)$")
    ax.set_title("Threshold-sensitivity curve")
    ax.grid(True, color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, loc="lower right")

    x = np.arange(len(rows))
    colors = [COLORS[row["profile"]] for row in rows]
    labels = [SHORT[row["profile"]] for row in rows]
    ax = axes[0, 1]
    width = 0.25
    ax.bar(x - width, [row["F30"] for row in rows], width=width, color="#2563eb", label=r"$F_{30}$")
    ax.bar(x, [row["q_rel"] for row in rows], width=width, color="#0f766e", label=r"$q_{\rm rel}$")
    ax.bar(x + width, [row["Y30"] for row in rows], width=width, color="#7c3aed", label=r"$Y_{30}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Concentration versus mobile yield")
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, ncol=3)

    ax = axes[1, 0]
    ax.bar(x, [row["Phi30"] for row in rows], color=colors, alpha=0.9)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel(r"$\Phi_{30}$")
    ax.set_title("Normalized excess focusing")
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.6)

    ax = axes[1, 1]
    ax.bar(x, [row["rear_alignment_A"] for row in rows], color=colors, alpha=0.9)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.25, 1.0)
    ax.set_ylabel(r"$A=\langle\cos\theta_{\rm rel}\rangle$")
    ax.set_title("Cutoff-free rear alignment")
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.6)

    for label, ax in zip(["a", "b", "c", "d"], axes.flat):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.suptitle("Focused mobile state: concentration, yield, gain, and robustness", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGURES / "focused_mobile_state_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "focused_mobile_state_metrics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = metrics()
    write_csv(OUT / "focused_mobile_state_metrics.csv", rows)
    write_table(OUTPUT_TABLES / "focused_mobile_state_metrics_table.tex", rows)
    plot(rows)
    print(OUT / "focused_mobile_state_metrics.csv")
    print(OUTPUT_TABLES / "focused_mobile_state_metrics_table.tex")
    print(FIGURES / "focused_mobile_state_metrics.png")


if __name__ == "__main__":
    main()
