#!/usr/bin/env python3
"""Run compiled resolved-Langevin parameter sweeps for focusing diagnostics."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled
from colloid_tsm.physical import (
    PhysicalParams,
    focusing_metrics,
    load_flow,
    save_library,
    transition_matrices,
)


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "compiled_parameter_suite"
FIGURES = ROOT / "outputs" / "figures"

VELOCITY_M_PER_DAY = 4.0
MAX_TIME_S = 120.0
MATRIX_BINS = 96
MATRIX_PARTICLES_PER_OPEN_BIN = 350
CORE_PARTICLES = 20_000
CORE_HALF_WIDTH_M = 12.5e-6
SEED_BASE = 880_000
KB = 1.380649e-23


PROFILES = (
    {
        "profile": "neutral_resolved",
        "regime": "control",
        "condition": "neutral",
        "description": "No DLVO control with the same resolved near-wall mobility.",
        "updates": {"ionic_strength_molar": 6.0e-3},
    },
    {
        "profile": "unfavorable_6mM_z70",
        "regime": "realistic",
        "condition": "unfavorable",
        "description": "Reference homogeneous unfavorable chemistry.",
        "updates": {"ionic_strength_molar": 6.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
    {
        "profile": "unfavorable_20mM_z70",
        "regime": "realistic_high_salt",
        "condition": "unfavorable",
        "description": "Compressed double layer, reference like-charge magnitude.",
        "updates": {"ionic_strength_molar": 20.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z70",
        "regime": "realistic_high_salt",
        "condition": "unfavorable",
        "description": "50 mM secondary-minimum case motivated by the DLVO screen.",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
    {
        "profile": "unfavorable_100mM_z70",
        "regime": "upper_salt_screen",
        "condition": "unfavorable",
        "description": "Upper-salt screen for stronger secondary-minimum residence.",
        "updates": {"ionic_strength_molar": 100.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z50",
        "regime": "moderate_zeta",
        "condition": "unfavorable",
        "description": "50 mM with reduced collector magnitude.",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -50.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z30",
        "regime": "weak_zeta",
        "condition": "unfavorable",
        "description": "50 mM weak like-charge magnitude.",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -30.0e-3},
    },
    {
        "profile": "unfavorable_100mM_z30",
        "regime": "weak_zeta_high_salt",
        "condition": "unfavorable",
        "description": "Weak like-charge magnitude at high ionic strength.",
        "updates": {"ionic_strength_molar": 100.0e-3, "zeta_collector_unfavorable": -30.0e-3},
    },
    {
        "profile": "mechanism_50mM_z20_A10x",
        "regime": "mechanism_amplified",
        "condition": "unfavorable",
        "description": "Non-environmental stronger Hamaker and weak repulsion to expose the mechanism.",
        "updates": {
            "ionic_strength_molar": 50.0e-3,
            "zeta_collector_unfavorable": -20.0e-3,
            "hamaker": 1.0e-20,
        },
    },
    {
        "profile": "mechanism_100mM_z20_A10x",
        "regime": "mechanism_amplified",
        "condition": "unfavorable",
        "description": "High-salt amplified secondary-minimum residence.",
        "updates": {
            "ionic_strength_molar": 100.0e-3,
            "zeta_collector_unfavorable": -20.0e-3,
            "hamaker": 1.0e-20,
        },
    },
    {
        "profile": "neutral_100xD",
        "regime": "high_diffusion_control",
        "condition": "neutral",
        "description": "No-DLVO control with 100x Brownian diffusivity.",
        "updates": {"ionic_strength_molar": 6.0e-3, "diffusivity_multiplier": 100.0},
    },
    {
        "profile": "unfavorable_50mM_z70_100xD",
        "regime": "high_diffusion_unfavorable",
        "condition": "unfavorable",
        "description": "50 mM unfavorable case with 100x Brownian diffusivity.",
        "updates": {
            "ionic_strength_molar": 50.0e-3,
            "zeta_collector_unfavorable": -70.0e-3,
            "diffusivity_multiplier": 100.0,
        },
    },
)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def params_for_profile(profile: dict[str, object]) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=MAX_TIME_S,
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
        **dict(profile["updates"]),
    )


def stratified_matrix_y(params: PhysicalParams, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, params.cell_length, MATRIX_BINS + 1)
    ys = []
    eps = max(params.contact_gap, 1.0e-12)
    for lower, upper in zip(edges[:-1], edges[1:]):
        lo = max(lower, params.inlet_y_min + eps)
        hi = min(upper, params.inlet_y_max - eps)
        if hi <= lo:
            continue
        ys.append(rng.uniform(lo, hi, size=MATRIX_PARTICLES_PER_OPEN_BIN))
    return np.concatenate(ys)


def central_core_y(params: PhysicalParams, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = 0.5 * params.cell_length - CORE_HALF_WIDTH_M
    hi = 0.5 * params.cell_length + CORE_HALF_WIDTH_M
    return rng.uniform(lo, hi, size=CORE_PARTICLES)


def dlvo_profile_metrics(params: PhysicalParams, condition: str) -> dict[str, float]:
    if condition == "neutral":
        return {
            "debye_length_nm": params.debye_length * 1.0e9,
            "secondary_minimum_kbt": float("nan"),
            "secondary_minimum_gap_nm": float("nan"),
            "barrier_kbt": float("nan"),
            "barrier_gap_nm": float("nan"),
        }
    h = np.logspace(-9, -6, 80_000)
    eps0 = 8.8541878128e-12
    eps = params.relative_permittivity * eps0
    kappa = 1.0 / params.debye_length
    c_edl = (
        2.0
        * np.pi
        * eps
        * params.particle_radius
        * kappa
        * params.zeta_particle
        * params.zeta_collector_unfavorable
    )
    potential = c_edl / kappa * np.exp(-kappa * h) - params.hamaker * params.particle_radius / (6.0 * h)
    min_id = int(np.argmin(potential))
    max_id = int(np.argmax(potential))
    thermal = KB * params.temperature
    return {
        "debye_length_nm": params.debye_length * 1.0e9,
        "secondary_minimum_kbt": float(potential[min_id] / thermal),
        "secondary_minimum_gap_nm": float(h[min_id] * 1.0e9),
        "barrier_kbt": float(potential[max_id] / thermal),
        "barrier_gap_nm": float(h[max_id] * 1.0e9),
    }


def angular_abs(theta: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(theta), np.cos(theta)))


def finite_quantiles(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(np.quantile(values, q)) for q in (0.05, 0.5, 0.95))


def ensemble_stats(lib) -> dict[str, float | int]:
    mobile = lib.exited & (~lib.attached)
    intercepted_any = lib.interceptions > 0
    mobile_intercepted = mobile & intercepted_any
    censored_intercepted = lib.censored & intercepted_any
    release = mobile_intercepted & np.isfinite(lib.theta_exit)
    center_any = lib.center_interceptions > 0
    center_mobile = mobile & center_any
    center_censored = lib.censored & center_any
    center_release = center_mobile & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
    corner_release = mobile_intercepted & (lib.collector_exit == 0) & np.isfinite(lib.theta_exit)

    theta_all = angular_abs(lib.theta_exit[release])
    theta_center = angular_abs(lib.theta_exit[center_release])
    theta_corner = angular_abs(lib.theta_exit[corner_release])
    intercepted_or_censored = intercepted_any | censored_intercepted
    near_q05, near_q50, near_q95 = finite_quantiles(lib.near_time[intercepted_or_censored])
    h_q05, h_q50, h_q95 = finite_quantiles(lib.h_min[intercepted_or_censored] * 1.0e9)
    center_near_q05, center_near_q50, center_near_q95 = finite_quantiles(lib.near_time[center_any])
    center_h_q05, center_h_q50, center_h_q95 = finite_quantiles(lib.h_min[center_any] * 1.0e9)
    y_center = lib.y_out[center_mobile] - 0.5 * lib.params.cell_length
    y_shift = lib.y_out[mobile_intercepted] - lib.y_in[mobile_intercepted]
    well_interceptions = lib.well_interceptions if lib.well_interceptions is not None else np.zeros(lib.y_in.size, dtype=int)
    well_time = lib.well_time if lib.well_time is not None else np.zeros(lib.y_in.size, dtype=float)
    theta_well_exit = lib.theta_well_exit if lib.theta_well_exit is not None else np.full(lib.y_in.size, np.nan)
    collector_well_exit = lib.collector_well_exit if lib.collector_well_exit is not None else np.full(lib.y_in.size, -1, dtype=int)
    well_angular_travel = (
        lib.well_angular_travel
        if lib.well_angular_travel is not None
        else np.zeros(lib.y_in.size, dtype=float)
    )
    well_net_angular_travel = (
        lib.well_net_angular_travel
        if lib.well_net_angular_travel is not None
        else np.zeros(lib.y_in.size, dtype=float)
    )
    center_well_interceptions = (
        lib.center_well_interceptions
        if lib.center_well_interceptions is not None
        else np.zeros(lib.y_in.size, dtype=int)
    )
    well_any = well_interceptions > 0
    well_mobile = mobile & well_any
    well_censored = lib.censored & well_any
    well_release = well_mobile & np.isfinite(theta_well_exit)
    center_well_any = center_well_interceptions > 0
    center_well_mobile = mobile & center_well_any
    center_well_censored = lib.censored & center_well_any
    center_well_release = center_well_mobile & (collector_well_exit == 1) & np.isfinite(theta_well_exit)
    well_theta = angular_abs(theta_well_exit[well_release])
    center_well_theta = angular_abs(theta_well_exit[center_well_release])
    well_travel = well_angular_travel[well_release]
    center_well_travel = well_angular_travel[center_well_release]
    well_net_travel = np.abs(well_net_angular_travel[well_release])
    center_well_net_travel = np.abs(well_net_angular_travel[center_well_release])
    well_time_q05, well_time_q50, well_time_q95 = finite_quantiles(well_time[well_any])
    center_well_time_q05, center_well_time_q50, center_well_time_q95 = finite_quantiles(well_time[center_well_any])

    def mean_or_nan(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else float("nan")

    def median_or_nan(values: np.ndarray) -> float:
        return float(np.median(values)) if values.size else float("nan")

    return {
        "particles": int(lib.y_in.size),
        "exited_fraction": float(np.mean(lib.exited)),
        "censored_fraction": float(np.mean(lib.censored)),
        "intercepted_any_count": int(np.sum(intercepted_any)),
        "intercepted_mobile_count": int(np.sum(mobile_intercepted)),
        "intercepted_censored_count": int(np.sum(censored_intercepted)),
        "release_count": int(np.sum(release)),
        "center_intercepted_any_count": int(np.sum(center_any)),
        "center_intercepted_mobile_count": int(np.sum(center_mobile)),
        "center_intercepted_censored_count": int(np.sum(center_censored)),
        "center_release_count": int(np.sum(center_release)),
        "corner_release_count": int(np.sum(corner_release)),
        "median_abs_theta_release_rad": median_or_nan(theta_all),
        "fraction_theta_release_within_15deg": mean_or_nan(theta_all <= np.deg2rad(15.0)),
        "fraction_theta_release_within_30deg": mean_or_nan(theta_all <= np.deg2rad(30.0)),
        "center_median_abs_theta_release_rad": median_or_nan(theta_center),
        "center_fraction_theta_release_within_15deg": mean_or_nan(theta_center <= np.deg2rad(15.0)),
        "center_fraction_theta_release_within_30deg": mean_or_nan(theta_center <= np.deg2rad(30.0)),
        "corner_median_abs_theta_release_rad": median_or_nan(theta_corner),
        "corner_fraction_theta_release_within_30deg": mean_or_nan(theta_corner <= np.deg2rad(30.0)),
        "near_time_intercepted_p05_s": near_q05,
        "near_time_intercepted_median_s": near_q50,
        "near_time_intercepted_p95_s": near_q95,
        "hmin_intercepted_p05_nm": h_q05,
        "hmin_intercepted_median_nm": h_q50,
        "hmin_intercepted_p95_nm": h_q95,
        "center_near_time_p05_s": center_near_q05,
        "center_near_time_median_s": center_near_q50,
        "center_near_time_p95_s": center_near_q95,
        "center_hmin_p05_nm": center_h_q05,
        "center_hmin_median_nm": center_h_q50,
        "center_hmin_p95_nm": center_h_q95,
        "center_yout_abs_median_um": median_or_nan(np.abs(y_center) * 1.0e6),
        "center_yout_within_25um_fraction": mean_or_nan(np.abs(y_center) <= 25.0e-6),
        "intercepted_yout_yin_shift_abs_median_um": median_or_nan(np.abs(y_shift) * 1.0e6),
        "contact_fraction": float(np.mean(lib.contact_events > 0)),
        "mean_contact_events": float(np.mean(lib.contact_events)),
        "well_gap_lower_nm": float(lib.well_gap_lower * 1.0e9),
        "well_gap_upper_nm": float(lib.well_gap_upper * 1.0e9),
        "well_gap_minimum_nm": float(lib.well_gap_minimum * 1.0e9),
        "well_potential_minimum_kbt": float(lib.well_potential_minimum_kbt),
        "well_basin_lower_nm": float(lib.well_basin_lower * 1.0e9),
        "well_basin_upper_nm": float(lib.well_basin_upper * 1.0e9),
        "well_intercepted_any_count": int(np.sum(well_any)),
        "well_intercepted_mobile_count": int(np.sum(well_mobile)),
        "well_intercepted_censored_count": int(np.sum(well_censored)),
        "well_release_count": int(np.sum(well_release)),
        "well_median_abs_theta_release_rad": median_or_nan(well_theta),
        "well_fraction_theta_release_within_15deg": mean_or_nan(well_theta <= np.deg2rad(15.0)),
        "well_fraction_theta_release_within_30deg": mean_or_nan(well_theta <= np.deg2rad(30.0)),
        "well_median_angular_travel_rad": median_or_nan(well_travel),
        "well_median_net_angular_travel_rad": median_or_nan(well_net_travel),
        "well_time_p05_s": well_time_q05,
        "well_time_median_s": well_time_q50,
        "well_time_p95_s": well_time_q95,
        "center_well_intercepted_any_count": int(np.sum(center_well_any)),
        "center_well_intercepted_mobile_count": int(np.sum(center_well_mobile)),
        "center_well_intercepted_censored_count": int(np.sum(center_well_censored)),
        "center_well_release_count": int(np.sum(center_well_release)),
        "center_well_median_abs_theta_release_rad": median_or_nan(center_well_theta),
        "center_well_fraction_theta_release_within_15deg": mean_or_nan(center_well_theta <= np.deg2rad(15.0)),
        "center_well_fraction_theta_release_within_30deg": mean_or_nan(center_well_theta <= np.deg2rad(30.0)),
        "center_well_median_angular_travel_rad": median_or_nan(center_well_travel),
        "center_well_median_net_angular_travel_rad": median_or_nan(center_well_net_travel),
        "center_well_time_p05_s": center_well_time_q05,
        "center_well_time_median_s": center_well_time_q50,
        "center_well_time_p95_s": center_well_time_q95,
    }


def matrix_stats(lib) -> dict[str, float | int]:
    matrices = transition_matrices(lib, n_bins=MATRIX_BINS)
    total_by_in = matrices["total_by_in"]
    open_rows = total_by_in > 0
    counts = matrices["counts"]
    mobile_rows = counts.sum(axis=1)
    row_max = np.divide(
        counts.max(axis=1),
        mobile_rows,
        out=np.full(MATRIX_BINS, np.nan),
        where=mobile_rows > 0,
    )
    return {
        "matrix_bins": MATRIX_BINS,
        "matrix_open_rows": int(np.sum(open_rows)),
        "matrix_min_particles_per_open_row": int(np.min(total_by_in[open_rows])) if np.any(open_rows) else 0,
        "matrix_median_particles_per_open_row": float(np.median(total_by_in[open_rows])) if np.any(open_rows) else float("nan"),
        "matrix_min_mobile_per_open_row": int(np.min(mobile_rows[open_rows])) if np.any(open_rows) else 0,
        "matrix_median_mobile_per_open_row": float(np.median(mobile_rows[open_rows])) if np.any(open_rows) else float("nan"),
        "matrix_median_row_max_probability": float(np.nanmedian(row_max)),
        "matrix_max_row_max_probability": float(np.nanmax(row_max)),
    }


def row_for(profile: dict[str, object], ensemble: str, params: PhysicalParams, lib) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "ensemble": ensemble,
        "profile": str(profile["profile"]),
        "regime": str(profile["regime"]),
        "condition": str(profile["condition"]),
        "description": str(profile["description"]),
        "velocity_m_per_day": VELOCITY_M_PER_DAY,
        "max_time_s": MAX_TIME_S,
        "ionic_strength_mM": params.ionic_strength_molar * 1.0e3,
        "debye_length_nm": params.debye_length * 1.0e9,
        "zeta_collector_unfavorable_mV": params.zeta_collector_unfavorable * 1.0e3,
        "hamaker_J": params.hamaker,
        "diffusivity_multiplier": params.diffusivity_multiplier,
        "diffusivity_m2_s": params.diffusivity,
        "peclet": params.particle_peclet,
    }
    row.update(dlvo_profile_metrics(params, str(profile["condition"])))
    row.update(ensemble_stats(lib))
    if ensemble == "matrix_stratified":
        row.update(matrix_stats(lib))
    else:
        row.update(
            {
                "matrix_bins": MATRIX_BINS,
                "matrix_open_rows": 0,
                "matrix_min_particles_per_open_row": 0,
                "matrix_median_particles_per_open_row": float("nan"),
                "matrix_min_mobile_per_open_row": 0,
                "matrix_median_mobile_per_open_row": float("nan"),
                "matrix_median_row_max_probability": float("nan"),
                "matrix_max_row_max_probability": float("nan"),
            }
        )
    return row


def add_neutral_comparisons(rows: list[dict[str, float | int | str]]) -> None:
    by_ensemble = {
        str(row["ensemble"]): row
        for row in rows
        if row["profile"] == "neutral_resolved"
    }
    for row in rows:
        base = by_ensemble[str(row["ensemble"])]
        base_theta = float(base["center_median_abs_theta_release_rad"])
        case_theta = float(row["center_median_abs_theta_release_rad"])
        base_frac = float(base["center_fraction_theta_release_within_30deg"])
        case_frac = float(row["center_fraction_theta_release_within_30deg"])
        base_near = float(base["center_near_time_median_s"])
        case_near = float(row["center_near_time_median_s"])
        row["neutral_to_case_center_theta_ratio"] = (
            base_theta / case_theta if np.isfinite(case_theta) and case_theta > 0.0 else float("nan")
        )
        row["case_to_neutral_theta30_enrichment"] = (
            case_frac / base_frac if np.isfinite(base_frac) and base_frac > 0.0 else float("nan")
        )
        row["case_to_neutral_center_near_time_ratio"] = (
            case_near / base_near if np.isfinite(base_near) and base_near > 0.0 else float("nan")
        )


def fmt(value: float | int | str, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "nan"


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    central = [row for row in rows if row["ensemble"] == "central_core"]
    matrix = [row for row in rows if row["ensemble"] == "matrix_stratified"]
    lines = [
        "# Compiled resolved-Langevin parameter suite",
        "",
        f"All runs use the N=192 resolved center/corner periodic flow at {VELOCITY_M_PER_DAY:.0f} m/day and a {MAX_TIME_S:.0f} s one-cell horizon. The compiled C kernel resolves near-wall Langevin dynamics with adaptive substeps, direct interpolated-LBM flow advection, wall-corrected DLVO drift and Brownian motion, barrier-respecting unfavorable DLVO motion, and no unfavorable attachment. Heterodomains are not included.",
        "",
        "## DLVO Profiles",
        "",
        "| profile | regime | I (mM) | zeta_c (mV) | A_H (J) | D mult | secondary min (kBT) | h_min (nm) | entry shell (nm) | escape basin (nm) | barrier (kBT) | Pe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in central:
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{row['regime']} | "
            f"{fmt(row['ionic_strength_mM'], 2)} | "
            f"{fmt(row['zeta_collector_unfavorable_mV'], 1)} | "
            f"{float(row['hamaker_J']):.2e} | "
            f"{fmt(row['diffusivity_multiplier'], 1)} | "
            f"{fmt(row['secondary_minimum_kbt'])} | "
            f"{fmt(row['secondary_minimum_gap_nm'], 2)} | "
            f"{fmt(row['well_gap_lower_nm'], 2)}-{fmt(row['well_gap_upper_nm'], 2)} | "
            f"{fmt(row['well_basin_lower_nm'], 2)}-{fmt(row['well_basin_upper_nm'], 2)} | "
            f"{fmt(row['barrier_kbt'], 1)} | "
            f"{fmt(row['peclet'], 0)} |"
        )
    lines.extend(
        [
            "",
            "## Central-Core Focusing",
            "",
            "Central-core injection starts within 25 microns of the cell centerline, so center-grain interception and release statistics directly test whether near-wall residence funnels particles toward the downstream stagnation zone.",
            "",
            "| profile | center int | censored int | median near time (s) | median hmin (nm) | median theta (rad) | theta <30 deg | theta enrichment | near-time ratio | contact frac |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in central:
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{int(row['center_intercepted_any_count'])} | "
            f"{int(row['center_intercepted_censored_count'])} | "
            f"{fmt(row['center_near_time_median_s'])} | "
            f"{fmt(row['center_hmin_median_nm'], 2)} | "
            f"{fmt(row['center_median_abs_theta_release_rad'])} | "
            f"{fmt(row['center_fraction_theta_release_within_30deg'])} | "
            f"{fmt(row['case_to_neutral_theta30_enrichment'])} | "
            f"{fmt(row['case_to_neutral_center_near_time_ratio'])} | "
            f"{fmt(row['contact_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Secondary-Minimum Well Focusing",
            "",
            "This table replaces the coarse 200 nm release angle with a hysteretic potential-defined secondary-minimum bound state. A particle enters the bound state in the U(h) <= Umin + 1 kBT shell and remains bound until it leaves the broader U(h) <= -1 kBT basin. Neutral controls have no DLVO well, so well metrics are undefined.",
            "",
            "| profile | entry shell (nm) | escape basin (nm) | center well int | center well censored | center well releases | median bound time (s) | median escape theta (rad) | median net travel (rad) | median cumulative travel (rad) | escape theta <30 deg |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in central:
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{fmt(row['well_gap_lower_nm'], 2)}-{fmt(row['well_gap_upper_nm'], 2)} | "
            f"{fmt(row['well_basin_lower_nm'], 2)}-{fmt(row['well_basin_upper_nm'], 2)} | "
            f"{int(row['center_well_intercepted_any_count'])} | "
            f"{int(row['center_well_intercepted_censored_count'])} | "
            f"{int(row['center_well_release_count'])} | "
            f"{fmt(row['center_well_time_median_s'])} | "
            f"{fmt(row['center_well_median_abs_theta_release_rad'])} | "
            f"{fmt(row['center_well_median_net_angular_travel_rad'])} | "
            f"{fmt(row['center_well_median_angular_travel_rad'])} | "
            f"{fmt(row['center_well_fraction_theta_release_within_30deg'])} |"
        )
    lines.extend(
        [
            "",
            "## Matrix-Ready Stratified Injection",
            "",
            f"Matrix runs seed {MATRIX_PARTICLES_PER_OPEN_BIN} particles in each open inlet row of a {MATRIX_BINS}-bin transverse discretization. These libraries are intended for transition-matrix construction rather than only focusing diagnostics.",
            "",
            "| profile | particles | open rows | min per row | min mobile per row | all int | center int | median theta all (rad) | theta <30 deg all | median row max P |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matrix:
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{int(row['particles'])} | "
            f"{int(row['matrix_open_rows'])} | "
            f"{int(row['matrix_min_particles_per_open_row'])} | "
            f"{int(row['matrix_min_mobile_per_open_row'])} | "
            f"{int(row['intercepted_any_count'])} | "
            f"{int(row['center_intercepted_any_count'])} | "
            f"{fmt(row['median_abs_theta_release_rad'])} | "
            f"{fmt(row['fraction_theta_release_within_30deg'])} | "
            f"{fmt(row['matrix_median_row_max_probability'])} |"
        )
    top = sorted(
        [row for row in central if np.isfinite(float(row["center_fraction_theta_release_within_30deg"]))],
        key=lambda row: float(row["center_fraction_theta_release_within_30deg"]),
        reverse=True,
    )[:4]
    lines.extend(["", "## Quick Read", ""])
    for row in top:
        lines.append(
            f"- {row['profile']}: theta<30 deg = {fmt(row['center_fraction_theta_release_within_30deg'])}, "
            f"median near time = {fmt(row['center_near_time_median_s'])} s, "
            f"median hmin = {fmt(row['center_hmin_median_nm'], 2)} nm."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_bar(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, float | int | str]],
    metric: str,
    rect: tuple[int, int, int, int],
    title: str,
    y_label: str,
    limit: float | None = None,
) -> None:
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    values = np.array([float(row[metric]) for row in rows], dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    ymax = max(float(np.max(values)), 1.0e-12) if limit is None else limit
    draw.rectangle(rect, outline="#111111", width=2)
    draw.text((left + width / 2, top - 30), title, fill="#111111", font=font(22), anchor="mm")
    draw.text((left - 64, top + height / 2), y_label, fill="#111111", font=font(16), anchor="mm")
    colors = {
        "control": "#2563eb",
        "realistic": "#059669",
        "realistic_high_salt": "#0f766e",
        "upper_salt_screen": "#0e7490",
        "moderate_zeta": "#65a30d",
        "weak_zeta": "#84cc16",
        "weak_zeta_high_salt": "#a3e635",
        "mechanism_amplified": "#7c3aed",
        "high_diffusion_control": "#9333ea",
        "high_diffusion_unfavorable": "#c026d3",
    }
    n = len(rows)
    bar_w = width / max(n, 1) * 0.68
    for i, row in enumerate(rows):
        value = float(row[metric])
        if not np.isfinite(value):
            value = 0.0
        x = left + (i + 0.5) * width / n
        y = bottom - min(value, ymax) / ymax * height
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=colors.get(str(row["regime"]), "#555555"))
        label = str(row["profile"]).replace("unfavorable_", "unfav_").replace("mechanism_", "mech_")
        draw.text((x, bottom + 14), label, fill="#111111", font=font(9), anchor="ma")
    for frac in np.linspace(0.0, 1.0, 5):
        y = bottom - frac * height
        draw.line((left - 8, y, left, y), fill="#111111", width=1)
        draw.text((left - 12, y), f"{frac * ymax:.2g}", fill="#111111", font=font(12), anchor="rm")


def write_figure(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    central = [row for row in rows if row["ensemble"] == "central_core"]
    img = Image.new("RGB", (1800, 1450), "white")
    draw = ImageDraw.Draw(img)
    draw.text((900, 48), "Compiled resolved-Langevin focusing parameter suite", fill="#111111", font=font(34), anchor="mm")
    draw.text(
        (900, 86),
        "Central-core injection; homogeneous conditions; center/corner periodic collectors; no unfavorable attachment",
        fill="#333333",
        font=font(18),
        anchor="mm",
    )
    draw_bar(
        draw,
        central,
        "center_well_fraction_theta_release_within_30deg",
        (140, 165, 1700, 420),
        "Fraction of center-well releases within 30 degrees of downstream stagnation",
        "fraction",
        limit=1.0,
    )
    draw_bar(
        draw,
        central,
        "center_well_time_median_s",
        (140, 570, 1700, 825),
        "Median secondary-minimum well residence time for center-grain well interceptions",
        "seconds",
    )
    draw_bar(
        draw,
        central,
        "center_hmin_median_nm",
        (140, 975, 1700, 1230),
        "Median closest approach for center-grain interceptions",
        "nm",
    )
    draw.text((140, 1385), "Colors group controls, realistic screens, weak-zeta screens, mechanism-amplified tests, and 100xD tests.", fill="#333333", font=font(18), anchor="lm")
    img.save(path)


def run_one(profile: dict[str, object], ensemble: str, initial_y: np.ndarray, seed: int):
    params = params_for_profile(profile)
    flow = load_flow(FLOW_PATH, params=params)
    lib = simulate_cell_transitions_compiled(
        flow,
        str(profile["condition"]),
        seed=seed,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="resolved_langevin",
    )
    return params, lib


def save_transition_payload(path: Path, lib) -> None:
    matrices = transition_matrices(lib, n_bins=MATRIX_BINS)
    np.savez_compressed(path, **matrices)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_kernel(force=False)

    rows: list[dict[str, float | int | str]] = []
    for profile_id, profile in enumerate(PROFILES):
        params = params_for_profile(profile)
        matrix_y = stratified_matrix_y(params, SEED_BASE + 100 * profile_id)
        core_y = central_core_y(params, SEED_BASE + 100 * profile_id + 1)
        for ensemble, initial_y, seed in (
            ("matrix_stratified", matrix_y, SEED_BASE + 100 * profile_id + 2),
            ("central_core", core_y, SEED_BASE + 100 * profile_id + 3),
        ):
            print(f"{ensemble}: {profile['profile']} ({initial_y.size} particles)", flush=True)
            run_params, lib = run_one(profile, ensemble, initial_y, seed)
            save_library(OUT / f"trajectory_library_{ensemble}_{profile['profile']}.npz", lib)
            if ensemble == "matrix_stratified":
                save_transition_payload(OUT / f"transition_matrices_{profile['profile']}.npz", lib)
            metrics_row = row_for(profile, ensemble, run_params, lib)
            # Keep the legacy focusing_metrics summary alongside the more
            # explicit all-/center-/corner collector statistics above.
            for key, value in focusing_metrics(lib).items():
                metrics_row[f"legacy_{key}"] = value
            rows.append(metrics_row)
            write_dicts(OUT / "compiled_parameter_suite_summary_partial.csv", rows)

    add_neutral_comparisons(rows)
    write_dicts(OUT / "compiled_parameter_suite_summary.csv", rows)
    write_report(OUT / "compiled_parameter_suite_report.md", rows)
    figure_path = OUT / "compiled_parameter_suite.png"
    write_figure(figure_path, rows)
    write_figure(FIGURES / "fig16_compiled_parameter_suite.png", rows)
    print(f"Done. Outputs written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
