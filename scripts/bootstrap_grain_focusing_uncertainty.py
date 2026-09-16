#!/usr/bin/env python3
"""Bootstrap uncertainty for grain-local focusing metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def entropy_effective_bins(values: np.ndarray, lo: float, hi: float, bins: int) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    counts, _ = np.histogram(finite, bins=bins, range=(lo, hi))
    positive = counts[counts > 0].astype(float)
    if positive.size == 0:
        return float("nan")
    p = positive / np.sum(positive)
    return float(2.0 ** (-np.sum(p * np.log2(p))))


def load_summary(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["profile"]: row for row in csv.DictReader(handle)}


def load_events(path: Path) -> dict[str, np.ndarray]:
    groups: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profile = row["profile"]
            signed_delta = float(row["delta_from_nearest_rear_rad"])
            abs_delta = float(row["abs_delta_from_nearest_rear_deg"])
            groups.setdefault(profile, []).append((signed_delta, abs_delta))
    return {profile: np.asarray(values, dtype=float) for profile, values in groups.items()}


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_profile(
    samples: np.ndarray,
    rng: np.random.Generator,
    draws: int,
    angle_bins: int,
) -> dict[str, float]:
    n = samples.shape[0]
    signed = samples[:, 0]
    abs_deg = samples[:, 1]
    observed_median = float(np.median(abs_deg))
    observed_within30 = float(np.mean(abs_deg <= 30.0))
    observed_effective = entropy_effective_bins(signed, -math.pi, math.pi, angle_bins)
    median_draws = np.empty(draws, dtype=float)
    within30_draws = np.empty(draws, dtype=float)
    effective_draws = np.empty(draws, dtype=float)
    for draw in range(draws):
        take = rng.integers(0, n, size=n)
        signed_take = signed[take]
        abs_take = abs_deg[take]
        median_draws[draw] = np.median(abs_take)
        within30_draws[draw] = np.mean(abs_take <= 30.0)
        effective_draws[draw] = entropy_effective_bins(signed_take, -math.pi, math.pi, angle_bins)
    median_lo, median_hi = percentile_interval(median_draws)
    within_lo, within_hi = percentile_interval(within30_draws)
    effective_lo, effective_hi = percentile_interval(effective_draws)
    return {
        "released_from_near_zone": float(n),
        "median_abs_release_angle_from_rear_deg": observed_median,
        "median_abs_release_angle_from_rear_deg_ci95_low": median_lo,
        "median_abs_release_angle_from_rear_deg_ci95_high": median_hi,
        "fraction_released_within_30deg": observed_within30,
        "fraction_released_within_30deg_ci95_low": within_lo,
        "fraction_released_within_30deg_ci95_high": within_hi,
        "grain_local_effective_angle_bins": observed_effective,
        "grain_local_effective_angle_bins_ci95_low": effective_lo,
        "grain_local_effective_angle_bins_ci95_high": effective_hi,
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float | int | str]], out_dir: Path) -> Path:
    labels = [str(row["profile"]) for row in rows]
    x = np.arange(len(rows))
    median = np.array([float(row["median_abs_release_angle_from_rear_deg"]) for row in rows])
    median_lo = np.array([float(row["median_abs_release_angle_from_rear_deg_ci95_low"]) for row in rows])
    median_hi = np.array([float(row["median_abs_release_angle_from_rear_deg_ci95_high"]) for row in rows])
    within = np.array([float(row["fraction_released_within_30deg"]) for row in rows])
    within_lo = np.array([float(row["fraction_released_within_30deg_ci95_low"]) for row in rows])
    within_hi = np.array([float(row["fraction_released_within_30deg_ci95_high"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), dpi=180, sharex=True)
    axes[0].bar(x, median, color="#2563eb")
    axes[0].errorbar(x, median, yerr=np.vstack((median - median_lo, median_hi - median)), fmt="none", color="#111827", capsize=3)
    axes[0].set_ylabel("median angle from rear (deg)")
    axes[1].bar(x, within, color="#0f766e")
    axes[1].errorbar(x, within, yerr=np.vstack((within - within_lo, within_hi - within)), fmt="none", color="#111827", capsize=3)
    axes[1].set_ylabel("fraction within 30 deg")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    fig.suptitle("Bootstrap uncertainty for grain-local focusing")
    fig.tight_layout()
    path = out_dir / "grain_local_focusing_bootstrap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(path: Path, rows: list[dict[str, float | int | str]], figure_path: Path) -> None:
    lines = [
        "# Grain-local focusing bootstrap uncertainty",
        "",
        "Confidence intervals are nonparametric 95% bootstrap intervals over released near-surface events within each profile.",
        "",
        "| profile | released | median angle from rear (deg) | within 30 deg | effective angle bins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {int(float(row['released_from_near_zone']))} | "
            f"{float(row['median_abs_release_angle_from_rear_deg']):.2f} "
            f"[{float(row['median_abs_release_angle_from_rear_deg_ci95_low']):.2f}, "
            f"{float(row['median_abs_release_angle_from_rear_deg_ci95_high']):.2f}] | "
            f"{float(row['fraction_released_within_30deg']):.3f} "
            f"[{float(row['fraction_released_within_30deg_ci95_low']):.3f}, "
            f"{float(row['fraction_released_within_30deg_ci95_high']):.3f}] | "
            f"{float(row['grain_local_effective_angle_bins']):.2f} "
            f"[{float(row['grain_local_effective_angle_bins_ci95_low']):.2f}, "
            f"{float(row['grain_local_effective_angle_bins_ci95_high']):.2f}] |"
        )
    lines.extend(["", f"Figure: `{figure_path.relative_to(ROOT)}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "grain_focusing_bootstrap")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--angle-bins", type=int, default=72)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = load_summary(args.summary)
    groups = load_events(args.events)
    rows: list[dict[str, float | int | str]] = []
    for profile in sorted(groups):
        row: dict[str, float | int | str] = {"profile": profile}
        if profile in summary:
            row["intercepted"] = int(summary[profile].get("intercepted", 0))
        row.update(bootstrap_profile(groups[profile], rng, args.draws, args.angle_bins))
        rows.append(row)
    csv_path = out_dir / "grain_local_focusing_bootstrap.csv"
    write_csv(csv_path, rows)
    fig = plot_rows(rows, out_dir)
    report = out_dir / "grain_local_focusing_bootstrap_report.md"
    write_report(report, rows, fig)
    print(report)
    print(fig)
    print(csv_path)


if __name__ == "__main__":
    main()
