#!/usr/bin/env python3
"""Run independent center/corner particle seeds for key production cases."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_openfoam_full_condition_suite as full_suite  # noqa: E402
from colloid_tsm.compiled import build_kernel  # noqa: E402
from colloid_tsm.physical import angular_distance, load_library, save_library  # noqa: E402


OUT = ROOT / "outputs" / "center_corner_replicates"
FIGURES = ROOT / "outputs" / "figures"
BASE_FULL = ROOT / "outputs" / "openfoam_full_suite"
DEFAULT_PROFILES = (
    "neutral_resolved",
    "unfavorable_6mM_z70",
    "unfavorable_50mM_z70",
    "unfavorable_100mM_z70",
    "unfavorable_50mM_z70_100xD",
    "unfavorable_100mM_z30",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def initial_y(params, n_particles: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = 0.5 * params.cell_length - full_suite.CORE_HALF_WIDTH_M
    hi = 0.5 * params.cell_length + full_suite.CORE_HALF_WIDTH_M
    return rng.uniform(lo, hi, size=n_particles)


def finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def finite_quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def metric_row(profile: dict[str, object], replicate_id: int, particle_seed: int, tracking_seed: int, params, lib, source: str) -> dict[str, object]:
    mobile = lib.exited & (~lib.attached)
    if str(profile["condition"]) == "neutral":
        event = lib.center_interceptions > 0
        release = mobile & event & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
        unresolved = lib.censored & event
        theta = np.degrees(np.abs(angular_distance(lib.theta_exit[release], 0.0)))
        residence = lib.near_time[event]
        event_name = "center near-surface"
        cumulative_events = int(np.sum(lib.center_interceptions))
        repeated_particles = int(np.sum(lib.center_interceptions > 1))
    else:
        center_well = lib.center_well_interceptions if lib.center_well_interceptions is not None else np.zeros(lib.y_in.size, dtype=int)
        event = center_well > 0
        release = (
            mobile
            & event
            & (lib.collector_well_exit == 1)
            & np.isfinite(lib.theta_well_exit)
        )
        unresolved = lib.censored & event
        theta = np.degrees(np.abs(angular_distance(lib.theta_well_exit[release], 0.0)))
        well_time = lib.well_time if lib.well_time is not None else np.zeros(lib.y_in.size)
        residence = well_time[event]
        event_name = "center secondary-minimum"
        cumulative_events = int(np.sum(center_well))
        repeated_particles = int(np.sum(center_well > 1))
    return {
        "profile": str(profile["profile"]),
        "label": str(profile["label"]),
        "condition": str(profile["condition"]),
        "replicate_id": replicate_id,
        "particle_seed": particle_seed,
        "tracking_seed": tracking_seed,
        "source": source,
        "particles": int(lib.y_in.size),
        "horizon_s": float(params.max_time),
        "event_name": event_name,
        "event_unique_particles": int(np.sum(event)),
        "event_cumulative_entries": cumulative_events,
        "event_repeated_particles": repeated_particles,
        "release_count": int(np.sum(release)),
        "unresolved_count": int(np.sum(unresolved)),
        "unresolved_fraction_of_event": float(np.sum(unresolved) / np.sum(event)) if np.any(event) else float("nan"),
        "F30": float(np.mean(theta <= 30.0)) if theta.size else float("nan"),
        "median_release_angle_deg": finite_median(theta),
        "p25_release_angle_deg": finite_quantile(theta, 0.25),
        "p75_release_angle_deg": finite_quantile(theta, 0.75),
        "median_residence_s": finite_median(residence),
        "p25_residence_s": finite_quantile(residence, 0.25),
        "p75_residence_s": finite_quantile(residence, 0.75),
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1:
        return float(finite[0]), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=1))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    profiles = []
    for row in rows:
        if row["profile"] not in profiles:
            profiles.append(row["profile"])
    for profile in profiles:
        group = [row for row in rows if row["profile"] == profile]
        f30_mean, f30_sd = mean_sd([float(row["F30"]) for row in group])
        angle_mean, angle_sd = mean_sd([float(row["median_release_angle_deg"]) for row in group])
        residence_mean, residence_sd = mean_sd([float(row["median_residence_s"]) for row in group])
        unresolved_mean, unresolved_sd = mean_sd([float(row["unresolved_fraction_of_event"]) for row in group])
        releases_mean, releases_sd = mean_sd([float(row["release_count"]) for row in group])
        event_mean, event_sd = mean_sd([float(row["event_unique_particles"]) for row in group])
        first = group[0]
        out.append(
            {
                "profile": profile,
                "label": first["label"],
                "event_name": first["event_name"],
                "replicates": len(group),
                "particles_per_replicate": int(first["particles"]),
                "horizon_s": first["horizon_s"],
                "event_unique_particles_mean": event_mean,
                "event_unique_particles_sd": event_sd,
                "release_count_mean": releases_mean,
                "release_count_sd": releases_sd,
                "F30_mean": f30_mean,
                "F30_sd": f30_sd,
                "median_release_angle_deg_mean": angle_mean,
                "median_release_angle_deg_sd": angle_sd,
                "median_residence_s_mean": residence_mean,
                "median_residence_s_sd": residence_sd,
                "unresolved_fraction_mean": unresolved_mean,
                "unresolved_fraction_sd": unresolved_sd,
            }
        )
    return out


def plot_summary(summary: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in summary]
    x = np.arange(len(summary))
    colors = ["#2563eb", "#64748b", "#7c3aed", "#581c87", "#0f766e", "#a21caf"][: len(summary)]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=220)
    panels = (
        ("F30_mean", "F30_sd", r"$F_{30}$"),
        ("median_release_angle_deg_mean", "median_release_angle_deg_sd", "median release angle (deg)"),
        ("median_residence_s_mean", "median_residence_s_sd", "median residence (s)"),
        ("unresolved_fraction_mean", "unresolved_fraction_sd", "unresolved fraction"),
    )
    for ax, (mean_key, sd_key, ylabel) in zip(axes.flat, panels):
        means = np.array([float(row[mean_key]) for row in summary], dtype=float)
        sds = np.array([float(row[sd_key]) for row in summary], dtype=float)
        sds = np.where(np.isfinite(sds), sds, 0.0)
        ax.bar(x, means, yerr=sds, color=colors, alpha=0.86, capsize=3)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.24)
        if mean_key == "median_residence_s_mean":
            ax.set_yscale("log")
    fig.suptitle("Center/corner seed-to-seed uncertainty for key cases", fontsize=13)
    fig.tight_layout()
    path = OUT / "center_corner_replicate_summary.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(FIGURES / "center_corner_replicate_summary.png", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    global OUT
    OUT = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    full_suite.OUT = OUT
    build_kernel(force=False)
    profiles = full_suite.selected_profiles(list(args.profiles))
    rows: list[dict[str, object]] = []
    existing_rows = read_csv(OUT / "center_corner_replicate_metrics.csv")
    if args.resume and existing_rows:
        rows.extend(existing_rows)
    existing_keys = {(row["profile"], int(row["replicate_id"])) for row in rows}
    for profile_index, profile in enumerate(profiles):
        name = str(profile["profile"])
        params = full_suite.params_for(profile, full_suite.REFINEMENT_HORIZON_S[name], args.dt_ms)
        for replicate_id in range(args.seeds):
            key = (name, replicate_id)
            if key in existing_keys:
                print(f"reused metrics {name} replicate {replicate_id}", flush=True)
                continue
            particle_seed = full_suite.SEED_REFINEMENT + 100_000 * replicate_id + 17
            tracking_seed = full_suite.SEED_REFINEMENT + 10_000 * replicate_id + 100 * profile_index
            baseline_path = BASE_FULL / f"trajectory_library_refinement_{name}.npz"
            if replicate_id == 0 and args.particles == full_suite.CORE_PARTICLES and baseline_path.exists():
                lib = load_library(baseline_path)
                save_library(OUT / f"trajectory_library_seed{replicate_id}_{name}.npz", lib)
                source = "openfoam_full_suite baseline"
            else:
                y0 = initial_y(params, args.particles, particle_seed)
                lib = full_suite.run_or_load(
                    profile,
                    f"seed{replicate_id}",
                    params,
                    y0,
                    tracking_seed,
                    bool(args.resume),
                    args.dt_ms,
                    int(args.chunk_size),
                    max(int(args.workers), 1),
                )
                source = "new replicate run"
            row = metric_row(profile, replicate_id, particle_seed, tracking_seed, params, lib, source)
            rows.append(row)
            write_csv(OUT / "center_corner_replicate_metrics.csv", rows)
            print(f"finished {name} replicate {replicate_id}: F30={row['F30']:.3f}", flush=True)
    summary = aggregate(rows)
    write_csv(OUT / "center_corner_replicate_summary.csv", summary)
    plot_summary(summary)
    report = [
        "# Center/Corner Seed-to-Seed Replicates",
        "",
        "Independent particle seeds were run for key center/corner cases. The reported unit is the unique completed-release particle support for the relevant center-grain event.",
        "",
        "| case | seeds | event | releases mean | F30 mean +/- sd | median angle mean +/- sd (deg) | median residence mean +/- sd (s) | unresolved mean +/- sd |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            "| {label} | {replicates} | {event_name} | {release_count_mean:.0f} | {F30_mean:.3f} +/- {F30_sd:.3f} | {median_release_angle_deg_mean:.2f} +/- {median_release_angle_deg_sd:.2f} | {median_residence_s_mean:.3g} +/- {median_residence_s_sd:.3g} | {unresolved_fraction_mean:.3f} +/- {unresolved_fraction_sd:.3f} |".format(
                **row
            )
        )
    (OUT / "center_corner_replicate_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT / "center_corner_replicate_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--particles", type=int, default=full_suite.CORE_PARTICLES)
    parser.add_argument("--dt-ms", type=float, default=full_suite.DEFAULT_DT_MS)
    parser.add_argument("--chunk-size", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
