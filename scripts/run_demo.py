#!/usr/bin/env python3
"""Run the single-grain DLVO/interception-history proof-of-concept demo."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm import (
    DLVOParams,
    SingleGrainCell,
    SurfaceChemistry,
    empirical_alpha,
    run_fixed_interception_sequence,
    run_interception_rw,
    sample_trajectories,
    simulate_first_interceptions,
)
from colloid_tsm.raster import write_geometry_png, write_retention_png
from colloid_tsm.svg import PlotStyle, write_geometry_svg, write_retention_svg


OUT = ROOT / "outputs"


def retention_histogram(x: np.ndarray, attached: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(x[attached], bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fraction = counts / max(1, x.size)
    return centers, fraction


def write_training_csv(path: Path, data) -> None:
    complete = data.complete()
    header = [
        "bulk_dx",
        "bulk_dy",
        "bulk_time",
        "near_dx",
        "near_dy",
        "near_time",
        "total_dx",
        "total_dy",
        "total_time",
        "attached",
        "h_min",
        "theta_entry",
    ]
    rows = zip(
        complete.bulk_dx,
        complete.bulk_dy,
        complete.bulk_time,
        complete.near_dx,
        complete.near_dy,
        complete.near_time,
        complete.total_dx,
        complete.total_dy,
        complete.total_time,
        complete.attached.astype(int),
        complete.h_min,
        complete.theta_entry,
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cell = SingleGrainCell(zoi=0.09, max_time=15.0, dt=0.0015)

    favorable_dlvo = DLVOParams(
        barrier_height=0.0,
        attraction_strength=2.0e-3,
        patch_attraction_strength=3.0e-3,
    )
    favorable_chem = SurfaceChemistry(
        coverage=1.0,
        base_attachment_rate=500.0,
        residual_attachment_rate=0.0,
        attachment_decay=0.12,
    )

    unfavorable_dlvo = DLVOParams(
        barrier_height=0.50,
        attraction_strength=3.0e-4,
        patch_attraction_strength=8.0e-4,
    )
    unfavorable_chem = SurfaceChemistry(
        # No discrete heterodomains in the starter favorable/unfavorable
        # comparison. Homogeneous unfavorable conditions are reflecting/
        # repulsive only, with no attachment.
        coverage=0.0,
        base_attachment_rate=0.0,
        residual_attachment_rate=0.0,
        attachment_decay=0.04,
    )

    print("Generating first-interception training libraries...")
    fav_data = simulate_first_interceptions(
        900, cell, favorable_dlvo, favorable_chem, seed=42
    )
    unfav_data = simulate_first_interceptions(
        1100, cell, unfavorable_dlvo, unfavorable_chem, seed=77
    )
    write_training_csv(OUT / "training_favorable.csv", fav_data)
    write_training_csv(OUT / "training_unfavorable.csv", unfav_data)

    print("Running favorable retention and unfavorable reflection walks...")
    fav_rw = run_interception_rw(fav_data, n_particles=25_000, seed=101)
    unfav_reflect = run_fixed_interception_sequence(
        unfav_data, n_particles=25_000, n_interceptions=20, seed=102
    )

    # Homogeneous unfavorable conditions are non-attaching in the starter
    # problem, so the retention figure only shows the favorable retained mass.
    max_x = 50.0
    bins = np.linspace(0.0, max_x, 72)
    series = {
        "favorable": (
            *retention_histogram(fav_rw.x, fav_rw.attached, bins),
            PlotStyle("#2364aa", "favorable: attaching"),
        ),
    }
    write_retention_svg(
        str(OUT / "retention_profiles.svg"),
        series,
        "Retention profile for favorable attachment",
    )
    write_retention_png(
        str(OUT / "retention_profiles.png"),
        series,
        "Retention profile for favorable attachment",
    )

    print("Drawing training-trajectory geometry figure...")
    paths = sample_trajectories(
        10, cell, unfavorable_dlvo, unfavorable_chem, seed=91, stride=7
    )
    write_geometry_svg(
        str(OUT / "single_grain_training_paths.svg"),
        paths,
        cell.length,
        cell.grain_radius,
        cell.zoi,
    )
    write_geometry_png(
        str(OUT / "single_grain_training_paths.png"),
        paths,
        cell.length,
        cell.grain_radius,
        cell.zoi,
    )

    summaries = []
    for name, data, rw, fixed_interceptions in [
        ("favorable", fav_data, fav_rw, 0),
        ("unfavorable_reflecting", unfav_data, unfav_reflect, 20),
    ]:
        complete = data.complete()
        attached = rw.attached
        has_attached = np.any(attached)
        summaries.append(
            {
                "scenario": name,
                "training_particles": data.size,
                "completed_first_interception_fraction": float(np.mean(data.completed)),
                "alpha_first_interception": empirical_alpha(data),
                "mean_bulk_dx": float(np.mean(complete.bulk_dx)),
                "mean_bulk_time": float(np.mean(complete.bulk_time)),
                "mean_near_time": float(np.mean(complete.near_time)),
                "rw_attached_fraction": float(np.mean(attached)),
                "mean_interception_order_attached": float(
                    np.mean(rw.interception_order[attached])
                ) if has_attached else float("nan"),
                "median_attachment_x": float(np.median(rw.x[attached]))
                if has_attached
                else float("nan"),
                "median_attachment_time": float(np.median(rw.t[attached]))
                if has_attached
                else float("nan"),
                "fixed_interceptions": fixed_interceptions,
                "median_x_after_fixed_interceptions": float(np.median(rw.x))
                if fixed_interceptions
                else float("nan"),
                "median_time_after_fixed_interceptions": float(np.median(rw.t))
                if fixed_interceptions
                else float("nan"),
            }
        )

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("Done. Wrote outputs to", OUT)
    for row in summaries:
        if row["fixed_interceptions"]:
            print(
                row["scenario"],
                "alpha1=",
                f"{row['alpha_first_interception']:.3f}",
                "attached fraction=",
                f"{row['rw_attached_fraction']:.3f}",
                "median x after",
                row["fixed_interceptions"],
                "interceptions=",
                f"{row['median_x_after_fixed_interceptions']:.2f}",
            )
        else:
            print(
                row["scenario"],
                "alpha1=",
                f"{row['alpha_first_interception']:.3f}",
                "mean order=",
                f"{row['mean_interception_order_attached']:.2f}",
                "median x=",
                f"{row['median_attachment_x']:.2f}",
            )


if __name__ == "__main__":
    main()
