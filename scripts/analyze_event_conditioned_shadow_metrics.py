#!/usr/bin/env python3
"""Event-conditioned exposure-shadow metrics from release and next-event angles."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVENTS = ROOT / "outputs" / "next_interception_kernel" / "release_to_next_interception_events.csv"
OUT = ROOT / "outputs" / "event_conditioned_shadow_metrics"
FIGURES = ROOT / "outputs" / "figures"

PROFILES = [
    "neutral_resolved",
    "unfavorable_50mM_z70",
    "unfavorable_75mM_z70",
    "unfavorable_100mM_z70",
    "unfavorable_50mM_z70_100xD",
]

LABELS = {
    "neutral_resolved": "No DLVO",
    "unfavorable_50mM_z70": "50 mM unfav.",
    "unfavorable_75mM_z70": "75 mM unfav.",
    "unfavorable_100mM_z70": "100 mM unfav.",
    "unfavorable_50mM_z70_100xD": "50 mM unfav., 100D",
}

COLORS = {
    "neutral_resolved": "#6b7280",
    "unfavorable_50mM_z70": "#2563eb",
    "unfavorable_75mM_z70": "#7c3aed",
    "unfavorable_100mM_z70": "#581c87",
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


def distribution(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=edges)
    if np.sum(counts) == 0:
        return np.full(edges.size - 1, np.nan)
    return counts.astype(float) / float(np.sum(counts))


def entropy_effective_bins(prob: np.ndarray) -> float:
    p = prob[np.isfinite(prob) & (prob > 0.0)]
    if p.size == 0:
        return float("nan")
    return float(np.exp(-np.sum(p * np.log(p))))


def weighted_shadow(reference: np.ndarray, condition: np.ndarray, epsilon: float) -> tuple[float, float, int, int]:
    support = np.isfinite(reference) & np.isfinite(condition) & (reference > epsilon)
    if not np.any(support):
        return float("nan"), float("nan"), 0, 0
    shadow = support & (condition <= epsilon)
    weighted = float(np.sum(reference[shadow]) / np.sum(reference[support]))
    count_fraction = float(np.sum(shadow) / np.sum(support))
    return weighted, count_fraction, int(np.sum(shadow)), int(np.sum(support))


def metric_rows(events: list[dict[str, str]], edges: np.ndarray, epsilon: float) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    by_profile = {profile: [row for row in events if row["profile"] == profile] for profile in PROFILES}
    release_dist: dict[str, np.ndarray] = {}
    next_dist: dict[str, np.ndarray] = {}
    for profile, rows in by_profile.items():
        release = np.array([float(row["release_delta_rear_deg"]) for row in rows], dtype=float)
        next_values = np.array(
            [
                float(row["next_delta_rear_deg"])
                for row in rows
                if row["next_interception"] == "True" and row["next_delta_rear_deg"] != "nan"
            ],
            dtype=float,
        )
        release_dist[profile] = distribution(release, edges)
        next_dist[profile] = distribution(next_values, edges)

    reference_release = release_dist["neutral_resolved"]
    reference_next = next_dist["neutral_resolved"]
    rows_out: list[dict[str, object]] = []
    for profile in PROFILES:
        release = np.array([float(row["release_delta_rear_deg"]) for row in by_profile[profile]], dtype=float)
        next_values = np.array(
            [
                float(row["next_delta_rear_deg"])
                for row in by_profile[profile]
                if row["next_interception"] == "True" and row["next_delta_rear_deg"] != "nan"
            ],
            dtype=float,
        )
        release_shadow_w, release_shadow_n, release_shadow_bins, release_support_bins = weighted_shadow(
            reference_release, release_dist[profile], epsilon
        )
        next_shadow_w, next_shadow_n, next_shadow_bins, next_support_bins = weighted_shadow(
            reference_next, next_dist[profile], epsilon
        )
        rows_out.append(
            {
                "profile": profile,
                "label": LABELS[profile],
                "release_count": int(release.size),
                "next_count": int(next_values.size),
                "release_F30": float(np.mean(release <= 30.0)) if release.size else float("nan"),
                "next_F30": float(np.mean(next_values <= 30.0)) if next_values.size else float("nan"),
                "release_effective_angle_bins": entropy_effective_bins(release_dist[profile]),
                "next_effective_angle_bins": entropy_effective_bins(next_dist[profile]),
                "release_broad_mass_gt60": float(np.mean(release > 60.0)) if release.size else float("nan"),
                "next_broad_mass_gt60": float(np.mean(next_values > 60.0)) if next_values.size else float("nan"),
                "release_weighted_shadow": release_shadow_w,
                "release_bin_shadow_fraction": release_shadow_n,
                "release_shadow_bins": release_shadow_bins,
                "release_support_bins": release_support_bins,
                "next_weighted_shadow": next_shadow_w,
                "next_bin_shadow_fraction": next_shadow_n,
                "next_shadow_bins": next_shadow_bins,
                "next_support_bins": next_support_bins,
            }
        )
    return rows_out, {"release": release_dist, "next": next_dist}


def plot(rows: list[dict[str, object]], dists: dict[str, dict[str, np.ndarray]], edges: np.ndarray) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), dpi=220)
    for profile in PROFILES:
        axes[0, 0].plot(centers, dists["release"][profile], marker="o", lw=2, color=COLORS[profile], label=LABELS[profile])
        axes[0, 1].plot(centers, dists["next"][profile], marker="o", lw=2, color=COLORS[profile], label=LABELS[profile])
    for ax, title in zip(axes[0], ["Release-angle exposure", "Next-event angle exposure"]):
        ax.axvspan(0, 30, color="#dbeafe", alpha=0.5, zorder=0)
        ax.set_xlim(0, 180)
        ax.set_xlabel("angle from grain-local rear point (deg)")
        ax.set_ylabel("probability")
        ax.set_title(title)
        ax.grid(True, color="#e5e7eb", lw=0.6)
    axes[0, 1].legend(frameon=False, fontsize=7.5)

    x = np.arange(len(PROFILES))
    width = 0.35
    axes[1, 0].bar(
        x - width / 2,
        [row["release_effective_angle_bins"] for row in rows],
        width=width,
        color="#2563eb",
        label="release",
    )
    axes[1, 0].bar(
        x + width / 2,
        [row["next_effective_angle_bins"] for row in rows],
        width=width,
        color="#7c3aed",
        label="next event",
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([LABELS[p] for p in PROFILES], rotation=20, ha="right")
    axes[1, 0].set_ylabel("entropy-effective angle bins")
    axes[1, 0].set_title("Effective angular support")
    axes[1, 0].grid(True, axis="y", color="#e5e7eb", lw=0.6)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        x - width / 2,
        [row["release_weighted_shadow"] for row in rows],
        width=width,
        color="#2563eb",
        label="release",
    )
    axes[1, 1].bar(
        x + width / 2,
        [row["next_weighted_shadow"] for row in rows],
        width=width,
        color="#7c3aed",
        label="next event",
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([LABELS[p] for p in PROFILES], rotation=20, ha="right")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("weighted angular shadow")
    axes[1, 1].set_title("Reference broad-angle mass suppressed")
    axes[1, 1].grid(True, axis="y", color="#e5e7eb", lw=0.6)
    axes[1, 1].legend(frameon=False)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.suptitle("Event-conditioned exposure shadows in grain-local angle space", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGURES / "event_conditioned_shadow_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "event_conditioned_shadow_metrics.png", bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, 180, 19)
    rows, dists = metric_rows(read_csv(EVENTS), edges, epsilon=0.02)
    write_csv(OUT / "event_conditioned_shadow_metrics.csv", rows)
    fig = plot(rows, dists, edges)
    print(OUT / "event_conditioned_shadow_metrics.csv")
    print(fig)


if __name__ == "__main__":
    main()
