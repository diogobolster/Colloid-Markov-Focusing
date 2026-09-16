#!/usr/bin/env python3
"""Sensitivity of center/corner focusing to h_ns and h_c choices."""

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


OUT = ROOT / "outputs" / "center_corner_hns_hc_sensitivity"
FIGURES = ROOT / "outputs" / "figures"
BASE_FULL = ROOT / "outputs" / "openfoam_full_suite"
DEFAULT_PROFILES = ("unfavorable_50mM_z70", "unfavorable_100mM_z70")
VARIANTS = (
    {"variant": "hns_100nm", "family": "h_ns", "label": r"$h_{ns}=100$ nm", "near_surface": 100.0e-9, "contact_gap": 1.0e-9},
    {"variant": "baseline", "family": "baseline", "label": "baseline", "near_surface": 200.0e-9, "contact_gap": 1.0e-9},
    {"variant": "hns_400nm", "family": "h_ns", "label": r"$h_{ns}=400$ nm", "near_surface": 400.0e-9, "contact_gap": 1.0e-9},
    {"variant": "hc_0p5nm", "family": "h_c", "label": r"$h_c=0.5$ nm", "near_surface": 200.0e-9, "contact_gap": 0.5e-9},
    {"variant": "hc_2nm", "family": "h_c", "label": r"$h_c=2$ nm", "near_surface": 200.0e-9, "contact_gap": 2.0e-9},
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


def summarize(profile: dict[str, object], variant: dict[str, object], params, lib, source: str) -> dict[str, object]:
    mobile = lib.exited & (~lib.attached)
    center_well = lib.center_well_interceptions if lib.center_well_interceptions is not None else np.zeros(lib.y_in.size, dtype=int)
    event = center_well > 0
    release = mobile & event & (lib.collector_well_exit == 1) & np.isfinite(lib.theta_well_exit)
    unresolved = lib.censored & event
    theta = np.degrees(np.abs(angular_distance(lib.theta_well_exit[release], 0.0)))
    well_time = lib.well_time if lib.well_time is not None else np.zeros(lib.y_in.size)
    return {
        "profile": str(profile["profile"]),
        "label": str(profile["label"]),
        "variant": str(variant["variant"]),
        "variant_family": str(variant["family"]),
        "variant_label": str(variant["label"]),
        "source": source,
        "particles": int(lib.y_in.size),
        "horizon_s": float(params.max_time),
        "near_surface_nm": float(params.near_surface * 1.0e9),
        "contact_gap_nm": float(params.contact_gap * 1.0e9),
        "center_well_unique_particles": int(np.sum(event)),
        "center_well_total_entry_events": int(np.sum(center_well)),
        "particles_with_repeated_center_well_events": int(np.sum(center_well > 1)),
        "release_count": int(np.sum(release)),
        "unresolved_count": int(np.sum(unresolved)),
        "unresolved_fraction": float(np.sum(unresolved) / np.sum(event)) if np.any(event) else float("nan"),
        "F30": float(np.mean(theta <= 30.0)) if theta.size else float("nan"),
        "median_release_angle_deg": finite_median(theta),
        "median_residence_s": finite_median(well_time[event]),
        "p25_residence_s": finite_quantile(well_time[event], 0.25),
        "p75_residence_s": finite_quantile(well_time[event], 0.75),
    }


def plot(rows: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    profiles = []
    for row in rows:
        if row["profile"] not in profiles:
            profiles.append(row["profile"])
    fig, axes = plt.subplots(len(profiles), 4, figsize=(14.0, 3.4 * len(profiles)), dpi=220, squeeze=False)
    metrics = (
        ("F30", r"$F_{30}$"),
        ("median_release_angle_deg", "median angle (deg)"),
        ("median_residence_s", "median residence (s)"),
        ("unresolved_fraction", "unresolved fraction"),
    )
    color_map = {"baseline": "#111827", "h_ns": "#2563eb", "h_c": "#7c3aed"}
    for row_id, profile in enumerate(profiles):
        subset = [row for row in rows if row["profile"] == profile]
        labels = [str(row["variant_label"]) for row in subset]
        x = np.arange(len(subset))
        colors = [color_map.get(str(row["variant_family"]), "#64748b") for row in subset]
        for col_id, (key, ylabel) in enumerate(metrics):
            ax = axes[row_id, col_id]
            values = np.array([float(row[key]) for row in subset], dtype=float)
            ax.bar(x, values, color=colors, alpha=0.86)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.24)
            if key == "median_residence_s":
                ax.set_yscale("log")
            if col_id == 0:
                ax.set_title(str(subset[0]["label"]))
            else:
                ax.set_title(ylabel)
    fig.suptitle(r"Center/corner sensitivity to $h_{ns}$ and $h_c$", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "center_corner_hns_hc_sensitivity.png", bbox_inches="tight")
    fig.savefig(FIGURES / "center_corner_hns_hc_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    global OUT
    OUT = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    full_suite.OUT = OUT
    build_kernel(force=False)
    profiles = full_suite.selected_profiles(list(args.profiles))
    rows: list[dict[str, object]] = []
    existing_rows = read_csv(OUT / "center_corner_hns_hc_sensitivity.csv")
    if args.resume and existing_rows:
        rows.extend(existing_rows)
    existing = {(row["profile"], row["variant"]) for row in rows}
    for profile_index, profile in enumerate(profiles):
        name = str(profile["profile"])
        base_params = full_suite.params_for(profile, full_suite.REFINEMENT_HORIZON_S[name], args.dt_ms)
        y0 = initial_y(base_params, args.particles, args.particle_seed)
        for variant_index, variant in enumerate(VARIANTS):
            key = (name, str(variant["variant"]))
            if key in existing:
                print(f"reused sensitivity {name} {variant['variant']}", flush=True)
                continue
            params = replace(
                base_params,
                near_surface=float(variant["near_surface"]),
                contact_gap=float(variant["contact_gap"]),
            )
            baseline_path = BASE_FULL / f"trajectory_library_refinement_{name}.npz"
            if (
                variant["variant"] == "baseline"
                and args.particles == full_suite.CORE_PARTICLES
                and baseline_path.exists()
            ):
                lib = load_library(baseline_path)
                save_library(OUT / f"trajectory_library_sensitivity_baseline_{name}.npz", lib)
                source = "openfoam_full_suite baseline"
            else:
                lib = full_suite.run_or_load(
                    profile,
                    f"sensitivity_{variant['variant']}",
                    params,
                    y0,
                    args.tracking_seed + 1000 * profile_index + variant_index,
                    bool(args.resume),
                    args.dt_ms,
                    int(args.chunk_size),
                    max(int(args.workers), 1),
                )
                source = "new sensitivity run"
            row = summarize(profile, variant, params, lib, source)
            rows.append(row)
            write_csv(OUT / "center_corner_hns_hc_sensitivity.csv", rows)
            print(f"finished {name} {variant['variant']}: F30={row['F30']:.3f}", flush=True)
    plot(rows)
    report = [
        "# Center/Corner h_ns and h_c Sensitivity",
        "",
        "The sensitivity test varies the near-surface/DLVO cutoff and contact gap while leaving flow, chemistry, adaptive stepping, and injection design fixed.",
        "",
        "| case | variant | h_ns (nm) | h_c (nm) | releases | unresolved | F30 | median angle (deg) | median residence (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            "| {label} | {variant_label} | {near_surface_nm:.0f} | {contact_gap_nm:.1f} | {release_count} | {unresolved_count} | {F30:.3f} | {median_release_angle_deg:.2f} | {median_residence_s:.3g} |".format(
                **row
            )
        )
    (OUT / "center_corner_hns_hc_sensitivity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT / "center_corner_hns_hc_sensitivity_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--particles", type=int, default=full_suite.CORE_PARTICLES)
    parser.add_argument("--dt-ms", type=float, default=full_suite.DEFAULT_DT_MS)
    parser.add_argument("--particle-seed", type=int, default=full_suite.SEED_REFINEMENT + 17)
    parser.add_argument("--tracking-seed", type=int, default=full_suite.SEED_REFINEMENT + 2026)
    parser.add_argument("--chunk-size", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
