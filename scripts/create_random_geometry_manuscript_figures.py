#!/usr/bin/env python3
"""Create manuscript-ready figures for the random-packing focusing extension."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "random_grain_focusing_replicates" / "grain_local_focusing_replicate_summary.csv"
OUT = ROOT / "outputs" / "figures" / "random_geometry_focusing_trend.png"

ORDER = [
    ("neutral_resolved", "No DLVO"),
    ("favorable_50mM_z70", "Favorable\n50 mM"),
    ("unfavorable_50mM_z70", "Unfav.\n50 mM"),
    ("unfavorable_75mM_z70", "Unfav.\n75 mM"),
    ("unfavorable_100mM_z70", "Unfav.\n100 mM"),
    ("unfavorable_50mM_z70_100xD", "Unfav. 50 mM\n100xD"),
]


def load_rows() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["profile"]] = {key: float(value) for key, value in row.items() if key != "profile"}
    return rows


def main() -> None:
    rows = load_rows()
    profiles = [profile for profile, _label in ORDER]
    labels = [label for _profile, label in ORDER]
    x = np.arange(len(profiles), dtype=float)

    median = np.array([rows[p]["median_abs_release_angle_from_rear_deg_mean"] for p in profiles])
    median_sd = np.array([rows[p]["median_abs_release_angle_from_rear_deg_sd"] for p in profiles])
    within = np.array([rows[p]["fraction_released_within_30deg_mean"] for p in profiles])
    within_sd = np.array([rows[p]["fraction_released_within_30deg_sd"] for p in profiles])
    released = np.array([rows[p]["released_from_near_zone_mean"] for p in profiles])
    colors = ["#6b7280", "#991b1b", "#2563eb", "#0f766e", "#7c3aed", "#d97706"]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.8), dpi=240, sharex=True)

    axes[0].bar(x, median, color=colors, width=0.72)
    axes[0].errorbar(x, median, yerr=median_sd, fmt="none", color="#111827", capsize=3, linewidth=1.0)
    axes[0].set_ylabel("Median angle from\nrear point (deg)")
    axes[0].set_ylim(0.0, 78.0)
    axes[0].grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)

    axes[1].bar(x, within, color=colors, width=0.72)
    axes[1].errorbar(x, within, yerr=within_sd, fmt="none", color="#111827", capsize=3, linewidth=1.0)
    axes[1].set_ylabel("Fraction released\nwithin 30 deg")
    axes[1].set_ylim(0.0, 0.72)
    axes[1].grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for xpos, count in zip(x, released):
        axes[1].text(xpos, 0.704, f"n={count:.0f}", ha="center", va="top", fontsize=7.2, color="#374151")

    axes[0].annotate(
        "salt-compressed secondary minimum",
        xy=(3.0, 12.0),
        xytext=(2.1, 28.0),
        arrowprops={"arrowstyle": "->", "color": "#374151", "linewidth": 0.8},
        fontsize=8.0,
        color="#374151",
    )
    axes[1].annotate(
        "high diffusion removes focusing",
        xy=(5.0, within[-1]),
        xytext=(4.12, 0.33),
        arrowprops={"arrowstyle": "->", "color": "#374151", "linewidth": 0.8},
        fontsize=8.0,
        color="#374151",
    )
    fig.suptitle("Grain-local focusing in a random periodic packing", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
