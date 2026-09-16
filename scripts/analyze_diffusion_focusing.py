#!/usr/bin/env python3
"""Summarize focusing metrics for baseline and high-diffusion sensitivity libraries."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import focusing_metrics, load_library


OUT = ROOT / "outputs" / "diffusion_sensitivity"
CONDITIONS = ("neutral", "favorable", "unfavorable")
MULTIPLIERS = (1, 100)

KEYS = (
    "diffusivity_multiplier",
    "condition",
    "particles",
    "intercepted_any_count",
    "intercepted_mobile_count",
    "center_intercepted_any_count",
    "center_intercepted_mobile_count",
    "center_intercepted_censored_count",
    "center_theta_exit_count",
    "center_median_abs_theta_exit_downstream_rad",
    "center_fraction_theta_exit_within_15deg",
    "center_fraction_theta_exit_within_30deg",
    "center_fraction_yout_within_center_band",
    "center_median_abs_yout_center_m",
    "mean_abs_yout_yin_shift_m",
)


def main() -> None:
    rows = []
    for multiplier in MULTIPLIERS:
        for condition in CONDITIONS:
            path = OUT / f"trajectory_library_{condition}_{multiplier}xD.npz"
            lib = load_library(path)
            row = focusing_metrics(lib)
            row["diffusivity_multiplier"] = multiplier
            rows.append(row)

    out_path = OUT / "focusing_metrics_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(KEYS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in KEYS})

    print(out_path)
    print(
        "mult condition all_int mobile_int center_int center_mobile theta_n "
        "median_abs_theta theta_lt_30 y_center_band mean_abs_shift_um"
    )
    for row in rows:
        shift_um = row["mean_abs_yout_yin_shift_m"]
        if shift_um == shift_um:
            shift_um = 1.0e6 * shift_um
        print(
            f"{int(row['diffusivity_multiplier']):3d} "
            f"{row['condition']:11s} "
            f"{int(row['intercepted_any_count']):7d} "
            f"{int(row['intercepted_mobile_count']):10d} "
            f"{int(row['center_intercepted_any_count']):10d} "
            f"{int(row['center_intercepted_mobile_count']):13d} "
            f"{int(row['center_theta_exit_count']):7d} "
            f"{float(row['center_median_abs_theta_exit_downstream_rad']):16.3f} "
            f"{float(row['center_fraction_theta_exit_within_30deg']):11.3f} "
            f"{float(row['center_fraction_yout_within_center_band']):13.3f} "
            f"{float(shift_um):17.2f}"
        )


if __name__ == "__main__":
    main()
