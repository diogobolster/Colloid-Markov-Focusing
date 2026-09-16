#!/usr/bin/env python3
"""Aggregate grain-local focusing metrics across replicate runs."""

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


METRICS = [
    "released_from_near_zone",
    "median_abs_release_angle_from_rear_deg",
    "fraction_released_within_30deg",
    "grain_local_effective_angle_bins",
]


def run_label(path: Path) -> str:
    name = path.name
    if "seed" in name:
        return name[name.index("seed") :]
    if name.endswith("production_selected"):
        return "seed20260507"
    return name


def load_replicates(paths: list[Path]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        csv_path = path / "grain_local_focusing.csv"
        label = run_label(path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                out: dict[str, float | int | str] = {"run": label, "profile": row["profile"]}
                for metric in METRICS:
                    out[metric] = float(row[metric])
                rows.append(out)
    return rows


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    profiles = sorted({str(row["profile"]) for row in rows})
    out: list[dict[str, float | int | str]] = []
    for profile in profiles:
        group = [row for row in rows if row["profile"] == profile]
        summary: dict[str, float | int | str] = {"profile": profile, "replicates": len(group)}
        for metric in METRICS:
            values = np.array([float(row[metric]) for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        out.append(summary)
    return out


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


def plot_summary(rows: list[dict[str, float | int | str]], out_dir: Path) -> Path:
    labels = [str(row["profile"]) for row in rows]
    x = np.arange(len(rows))
    median_mean = np.array([float(row["median_abs_release_angle_from_rear_deg_mean"]) for row in rows])
    median_sd = np.array([float(row["median_abs_release_angle_from_rear_deg_sd"]) for row in rows])
    within_mean = np.array([float(row["fraction_released_within_30deg_mean"]) for row in rows])
    within_sd = np.array([float(row["fraction_released_within_30deg_sd"]) for row in rows])
    median_sd = np.where(np.isfinite(median_sd), median_sd, 0.0)
    within_sd = np.where(np.isfinite(within_sd), within_sd, 0.0)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), dpi=180, sharex=True)
    axes[0].bar(x, median_mean, color="#2563eb")
    axes[0].errorbar(x, median_mean, yerr=median_sd, fmt="none", color="#111827", capsize=3)
    axes[0].set_ylabel("median angle from rear (deg)")
    axes[1].bar(x, within_mean, color="#0f766e")
    axes[1].errorbar(x, within_mean, yerr=within_sd, fmt="none", color="#111827", capsize=3)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("fraction within 30 deg")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    fig.suptitle("Seed-to-seed grain-local focusing")
    fig.tight_layout()
    path = out_dir / "grain_local_focusing_replicate_summary.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fmt_mean_sd(row: dict[str, float | int | str], metric: str, digits: int) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sd"])
    if math.isfinite(sd):
        return f"{mean:.{digits}f} +/- {sd:.{digits}f}"
    return f"{mean:.{digits}f}"


def write_report(path: Path, rows: list[dict[str, float | int | str]], figure: Path) -> None:
    lines = [
        "# Grain-local focusing replicate summary",
        "",
        "Values are means +/- sample standard deviations across independent particle seeds. The 75 mM bridge has two replicates because it was added after the first production run.",
        "",
        "| profile | replicates | median angle from rear (deg) | within 30 deg | effective angle bins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['replicates']} | "
            f"{fmt_mean_sd(row, 'median_abs_release_angle_from_rear_deg', 2)} | "
            f"{fmt_mean_sd(row, 'fraction_released_within_30deg', 3)} | "
            f"{fmt_mean_sd(row, 'grain_local_effective_angle_bins', 2)} |"
        )
    lines.extend(["", f"Figure: `{figure.relative_to(ROOT)}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("focus_dirs", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "random_grain_focusing_replicates")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_replicates(args.focus_dirs)
    summary = aggregate(rows)
    write_csv(out_dir / "grain_local_focusing_replicate_metrics.csv", rows)
    write_csv(out_dir / "grain_local_focusing_replicate_summary.csv", summary)
    fig = plot_summary(summary, out_dir)
    report = out_dir / "grain_local_focusing_replicate_summary.md"
    write_report(report, summary, fig)
    print(report)
    print(fig)


if __name__ == "__main__":
    main()
