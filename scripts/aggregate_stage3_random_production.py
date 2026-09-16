#!/usr/bin/env python3
"""Aggregate stage-3 random-packing production outputs."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = ROOT / "outputs" / "stage3_particle_production"
FIGURES = ROOT / "outputs" / "figures"

CANDIDATES = (
    "through_many_small_grains",
    "through_wide_throats",
    "through_random_00",
    "through_random_01",
    "through_low_porosity",
)

PROFILE_ORDER = (
    "neutral_resolved",
    "favorable_50mM_z70",
    "unfavorable_50mM_z70",
    "unfavorable_50mM_z70_100xD",
)

PROFILE_LABELS = {
    "neutral_resolved": "No DLVO",
    "favorable_50mM_z70": "Favorable 50 mM",
    "unfavorable_50mM_z70": "Unfavorable 50 mM",
    "unfavorable_50mM_z70_100xD": "Unfavorable 50 mM, 100D",
}

PROFILE_COLORS = {
    "neutral_resolved": "#64748b",
    "favorable_50mM_z70": "#dc2626",
    "unfavorable_50mM_z70": "#7c3aed",
    "unfavorable_50mM_z70_100xD": "#0f766e",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    production_rows: list[dict[str, object]] = []
    focusing_rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        condition_path = BASE / candidate / "random_condition_screen_summary.csv"
        focusing_path = BASE / candidate / "grain_focusing" / "grain_local_focusing.csv"
        if condition_path.exists():
            for row in read_csv(condition_path):
                row = dict(row)
                row["candidate_id"] = candidate
                production_rows.append(row)
        if focusing_path.exists():
            for row in read_csv(focusing_path):
                out: dict[str, object] = {"candidate_id": candidate}
                out.update(row)
                focusing_rows.append(out)
    return production_rows, focusing_rows


def numeric(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def summarize_focusing(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["profile"])].append(row)
    summary: list[dict[str, object]] = []
    for profile in PROFILE_ORDER:
        profile_rows = grouped.get(profile, [])
        if not profile_rows:
            continue
        f30 = np.array([numeric(row, "fraction_released_within_30deg") for row in profile_rows], dtype=float)
        med = np.array([numeric(row, "median_abs_release_angle_from_rear_deg") for row in profile_rows], dtype=float)
        released = np.array([numeric(row, "released_from_near_zone") for row in profile_rows], dtype=float)
        intercepted = np.array([numeric(row, "intercepted") for row in profile_rows], dtype=float)
        censored = np.array([numeric(row, "censored") for row in profile_rows], dtype=float)
        summary.append(
            {
                "profile": profile,
                "label": PROFILE_LABELS.get(profile, profile),
                "geometries": len(profile_rows),
                "released_total": int(np.nansum(released)),
                "intercepted_total": int(np.nansum(intercepted)),
                "censored_total": int(np.nansum(censored)),
                "F30_mean": float(np.nanmean(f30)),
                "F30_sd": float(np.nanstd(f30, ddof=1)) if f30.size > 1 else float("nan"),
                "F30_min": float(np.nanmin(f30)),
                "F30_max": float(np.nanmax(f30)),
                "median_angle_mean_deg": float(np.nanmean(med)),
                "median_angle_sd_deg": float(np.nanstd(med, ddof=1)) if med.size > 1 else float("nan"),
            }
        )
    return summary


def write_summary_markdown(summary: list[dict[str, object]]) -> Path:
    path = BASE / "stage3_random_focusing_summary.md"
    lines = [
        "# Random-packing focusing summary",
        "",
        "| condition | released events | intercepted particles | F30 mean +- sd | median angle mean +- sd (deg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | {released_total:d} | {intercepted_total:d} | {F30_mean:.3f} +- {F30_sd:.3f} | {median_angle_mean_deg:.1f} +- {median_angle_sd_deg:.1f} |".format(
                **row
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_focusing(rows: list[dict[str, object]], summary: list[dict[str, object]]) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(PROFILE_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=240)
    summary_by_profile = {str(row["profile"]): row for row in summary}
    rng = np.random.default_rng(20260510)
    for i, profile in enumerate(PROFILE_ORDER):
        vals = [numeric(row, "fraction_released_within_30deg") for row in rows if row["profile"] == profile]
        med = [numeric(row, "median_abs_release_angle_from_rear_deg") for row in rows if row["profile"] == profile]
        jitter = rng.normal(0.0, 0.035, size=len(vals))
        axes[0].scatter(np.full(len(vals), i) + jitter, vals, s=38, color=PROFILE_COLORS[profile], edgecolor="white", linewidth=0.5, zorder=3)
        axes[1].scatter(np.full(len(med), i) + jitter, med, s=38, color=PROFILE_COLORS[profile], edgecolor="white", linewidth=0.5, zorder=3)
        if profile in summary_by_profile:
            row = summary_by_profile[profile]
            axes[0].errorbar(i, row["F30_mean"], yerr=row["F30_sd"], fmt="o", color="#111827", capsize=4, ms=4.5, zorder=4)
            axes[1].errorbar(i, row["median_angle_mean_deg"], yerr=row["median_angle_sd_deg"], fmt="o", color="#111827", capsize=4, ms=4.5, zorder=4)
    labels = [PROFILE_LABELS[p] for p in PROFILE_ORDER]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.grid(axis="y", color="#e5e7eb", lw=0.7)
        ax.set_xlim(-0.55, len(PROFILE_ORDER) - 0.45)
    axes[0].set_ylabel(r"$F_{30}$")
    axes[0].set_ylim(0.0, 0.46)
    axes[0].set_title("Release within rear-zone sector")
    axes[1].set_ylabel("median angle from rear (deg)")
    axes[1].set_ylim(0.0, 95.0)
    axes[1].set_title("Release-angle concentration")
    for label, ax in zip("ab", axes):
        ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGURES / "random_stage3_focusing_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(summary: list[dict[str, object]], figure_path: Path, table_path: Path) -> Path:
    lines = [
        "# Stage-3 Random Production Aggregate",
        "",
        "Five production random geometries were run with 10,000 particles per condition and analyzed using grain-local rear stagnation points.",
        "",
        "| condition | geometries | released | intercepted | F30 mean +- sd | median angle mean +- sd (deg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | {geometries} | {released_total} | {intercepted_total} | {F30_mean:.3f} +- {F30_sd:.3f} | {median_angle_mean_deg:.1f} +- {median_angle_sd_deg:.1f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Figure: `{figure_path.relative_to(ROOT)}`",
            f"Table: `{table_path.relative_to(ROOT)}`",
            "",
        ]
    )
    path = BASE / "stage3_random_production_aggregate_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    production_rows, focusing_rows = load_rows()
    summary = summarize_focusing(focusing_rows)
    write_csv(BASE / "stage3_random_condition_summary.csv", production_rows)
    write_csv(BASE / "stage3_random_focusing_by_geometry.csv", focusing_rows)
    write_csv(BASE / "stage3_random_focusing_summary.csv", summary)
    table_path = write_summary_markdown(summary)
    figure_path = plot_focusing(focusing_rows, summary)
    report = write_report(summary, figure_path, table_path)
    print(report)
    print(figure_path)
    print(table_path)


if __name__ == "__main__":
    main()
