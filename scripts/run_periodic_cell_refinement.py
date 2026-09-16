#!/usr/bin/env python3
"""Focused periodic-cell validation and secondary-minimum focusing diagnostics."""

from __future__ import annotations

import csv
import math
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
    _near_wall_mobility_factors,
    angular_distance,
    dlvo_force_normal,
    interpolate_velocity,
    load_flow,
    load_library,
    save_library,
    secondary_minimum_well,
)


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "periodic_cell_refinement"
FIGURES = ROOT / "outputs" / "figures"

VELOCITY_M_PER_DAY = 4.0
MAX_TIME_S = 240.0
CORE_PARTICLES = 30_000
CORE_HALF_WIDTH_M = 12.5e-6
SEED = 1_281_000
KB = 1.380649e-23
PROFILE_MAX_TIME_S = {
    "neutral_resolved": 240.0,
    "unfavorable_50mM_z70": 600.0,
    "unfavorable_50mM_z50": 600.0,
    "unfavorable_50mM_z30": 600.0,
    "unfavorable_100mM_z70": 2400.0,
}

PROFILES = (
    {
        "profile": "neutral_resolved",
        "label": "no DLVO",
        "condition": "neutral",
        "updates": {"ionic_strength_molar": 6.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z70",
        "label": "50 mM, zeta -70 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z50",
        "label": "50 mM, zeta -50 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -50.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z30",
        "label": "50 mM, zeta -30 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -30.0e-3},
    },
    {
        "profile": "unfavorable_100mM_z70",
        "label": "100 mM, zeta -70 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 100.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
)


def horizon_for(profile: dict[str, object]) -> float:
    return float(PROFILE_MAX_TIME_S.get(str(profile["profile"]), MAX_TIME_S))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def params_for(profile: dict[str, object]) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=horizon_for(profile),
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
        **dict(profile["updates"]),
    )


def central_core_y(params: PhysicalParams) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    lo = 0.5 * params.cell_length - CORE_HALF_WIDTH_M
    hi = 0.5 * params.cell_length + CORE_HALF_WIDTH_M
    return rng.uniform(lo, hi, size=CORE_PARTICLES)


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def finite_quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def fraction(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def binomial_half_width(p: float, n: int) -> float:
    if n <= 0 or not np.isfinite(p):
        return float("nan")
    return float(1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n))


def residence_bote(params: PhysicalParams, condition: str) -> dict[str, float]:
    well = secondary_minimum_well(params, condition)
    if not bool(well["has_well"]):
        return {
            "bote_tau_half_s": float("nan"),
            "bote_tau_full_s": float("nan"),
            "bote_normal_diffusivity_m2_s": float("nan"),
            "bote_outer_inward_speed_um_s": float("nan"),
        }
    h_star = float(well["minimum_gap"])
    u_min = float(well["minimum_potential_kbt"])
    normal_mobility = h_star / (h_star + params.particle_radius)
    diffusivity_normal = params.diffusivity * normal_mobility
    exp_factor = math.exp(-u_min)
    tau_half = h_star * h_star / (2.0 * diffusivity_normal) * exp_factor
    tau_full = h_star * h_star / diffusivity_normal * exp_factor
    force_outer = float(dlvo_force_normal(params, np.array([float(well["upper"])]), condition)[0])
    stokes_mobility = 1.0 / (6.0 * math.pi * params.viscosity * params.particle_radius)
    inward_speed_outer = -stokes_mobility * force_outer * normal_mobility
    return {
        "bote_tau_half_s": tau_half,
        "bote_tau_full_s": tau_full,
        "bote_normal_diffusivity_m2_s": diffusivity_normal,
        "bote_outer_inward_speed_um_s": inward_speed_outer * 1.0e6,
    }


def velocity_samples(flow, params: PhysicalParams, profile_name: str, condition: str) -> list[dict[str, float | str]]:
    well = secondary_minimum_well(params, condition)
    if bool(well["has_well"]):
        gaps = [float(well["minimum_gap"]), float(well["upper"]), float(well["basin_upper"])]
        gap_labels = ["well_minimum", "well_outer_entry", "basin_outer"]
    else:
        gaps = [10.0e-9, 50.0e-9, 200.0e-9]
        gap_labels = ["10nm", "50nm", "200nm"]
    theta = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=True)
    rows: list[dict[str, float | str]] = []
    for collector, cx, cy in (
        ("center", 0.5 * params.cell_length, 0.5 * params.cell_length),
        ("corner", 0.0, 0.0),
    ):
        for gap_label, gap in zip(gap_labels, gaps):
            radius = params.exclusion_radius + gap
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            ux, uy = interpolate_velocity(flow, x, y)
            tx = -np.sin(theta)
            ty = np.cos(theta)
            nx = np.cos(theta)
            ny = np.sin(theta)
            ut = ux * tx + uy * ty
            un = ux * nx + uy * ny
            _, parallel_mobility, _ = _near_wall_mobility_factors(params, np.full(theta.size, gap))
            angular_rate = ut / radius
            rows.append(
                {
                    "profile": profile_name,
                    "collector": collector,
                    "gap_label": gap_label,
                    "gap_nm": gap * 1.0e9,
                    "median_abs_ut_um_s": finite_median(np.abs(ut)) * 1.0e6,
                    "p95_abs_ut_um_s": finite_quantile(np.abs(ut), 0.95) * 1.0e6,
                    "max_abs_ut_um_s": float(np.max(np.abs(ut))) * 1.0e6,
                    "median_abs_un_um_s": finite_median(np.abs(un)) * 1.0e6,
                    "median_abs_angular_rate_deg_s": finite_median(np.abs(angular_rate)) * 180.0 / math.pi,
                    "median_abs_wall_damped_ut_um_s": finite_median(np.abs(ut * parallel_mobility)) * 1.0e6,
                    "theta_min_ut_deg": float(np.degrees(theta[int(np.argmin(np.abs(ut)))])),
                }
            )
    return rows


def summarize_library(profile: dict[str, object], params: PhysicalParams, lib) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    center_any = lib.center_interceptions > 0
    center_release = mobile & center_any & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
    center_censored = lib.censored & center_any
    theta_center = np.abs(angular_distance(lib.theta_exit[center_release], 0.0))
    center_well_any = (
        lib.center_well_interceptions > 0
        if lib.center_well_interceptions is not None
        else np.zeros(lib.y_in.size, dtype=bool)
    )
    center_well_release = (
        mobile
        & center_well_any
        & (lib.collector_well_exit == 1)
        & np.isfinite(lib.theta_well_exit)
    )
    center_well_censored = lib.censored & center_well_any
    theta_well = np.abs(angular_distance(lib.theta_well_exit[center_well_release], 0.0))
    well_time = lib.well_time if lib.well_time is not None else np.zeros(lib.y_in.size)
    well_angular_travel = lib.well_angular_travel if lib.well_angular_travel is not None else np.zeros(lib.y_in.size)
    well_net_angular_travel = (
        lib.well_net_angular_travel
        if lib.well_net_angular_travel is not None
        else np.zeros(lib.y_in.size)
    )
    well = secondary_minimum_well(params, str(profile["condition"]))
    bote = residence_bote(params, str(profile["condition"]))
    f30 = fraction(theta_well <= np.deg2rad(30.0))
    n_release = int(np.sum(center_well_release))
    censored_count = int(np.sum(lib.censored))
    final_status = "complete"
    nonphysical_count = 0
    if censored_count:
        final_status = "non physical trapping/attachment"
        nonphysical_count = censored_count
    return {
        "profile": str(profile["profile"]),
        "label": str(profile["label"]),
        "condition": str(profile["condition"]),
        "particles": int(lib.y_in.size),
        "max_time_s": params.max_time,
        "final_status": final_status,
        "nonphysical_trapping_attachment_count": nonphysical_count,
        "exited_fraction": float(np.mean(lib.exited)),
        "censored_fraction": float(np.mean(lib.censored)),
        "center_intercepted_count": int(np.sum(center_any)),
        "center_release_count": int(np.sum(center_release)),
        "center_censored_count": int(np.sum(center_censored)),
        "center_release_theta30_fraction": fraction(theta_center <= np.deg2rad(30.0)),
        "center_release_median_theta_deg": finite_median(np.degrees(theta_center)),
        "center_release_mean_theta_deg": float(np.mean(np.degrees(theta_center))) if theta_center.size else float("nan"),
        "center_near_time_median_s": finite_median(lib.near_time[center_any]),
        "center_hmin_median_nm": finite_median(lib.h_min[center_any]) * 1.0e9,
        "center_well_intercepted_count": int(np.sum(center_well_any)),
        "center_well_release_count": n_release,
        "center_well_censored_count": int(np.sum(center_well_censored)),
        "center_well_release_theta30_fraction": f30,
        "center_well_release_theta30_halfwidth_95": binomial_half_width(f30, n_release),
        "center_well_release_median_theta_deg": finite_median(np.degrees(theta_well)),
        "center_well_release_mean_theta_deg": float(np.mean(np.degrees(theta_well))) if theta_well.size else float("nan"),
        "center_well_time_p05_s": finite_quantile(well_time[center_well_any], 0.05),
        "center_well_time_median_s": finite_median(well_time[center_well_any]),
        "center_well_time_p95_s": finite_quantile(well_time[center_well_any], 0.95),
        "center_well_net_travel_median_deg": finite_median(np.degrees(np.abs(well_net_angular_travel[center_well_release]))),
        "center_well_cumulative_travel_median_deg": finite_median(np.degrees(well_angular_travel[center_well_release])),
        "well_gap_lower_nm": float(well["lower"]) * 1.0e9,
        "well_gap_upper_nm": float(well["upper"]) * 1.0e9,
        "well_basin_lower_nm": float(well["basin_lower"]) * 1.0e9,
        "well_basin_upper_nm": float(well["basin_upper"]) * 1.0e9,
        "well_minimum_gap_nm": float(well["minimum_gap"]) * 1.0e9,
        "well_minimum_kbt": float(well["minimum_potential_kbt"]),
        **bote,
    }


def fmt(value: float | int | str, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "nan"


def write_report(path: Path, rows: list[dict[str, float | int | str]], velocity_rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# Periodic-cell refinement diagnostic",
        "",
        f"Runs use central-core injection with {CORE_PARTICLES:,} particles, the N=192 center/corner periodic LBM flow at {VELOCITY_M_PER_DAY:.0f} m/day, direct interpolated-LBM flow advection, wall-corrected DLVO/Brownian motion, and adaptive one-cell horizons selected from the long-horizon censoring audit.",
        "",
        "## Focused secondary-minimum cases",
        "",
        "| profile | horizon (s) | status | center int | center releases | center censored | near theta <30 | well int | well releases | well censored | well theta <30 | +/-95% | median well theta (deg) | median well time (s) | median net travel (deg) | median cumulative travel (deg) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{fmt(row['max_time_s'], 0)} | "
            f"{row['final_status']} | "
            f"{int(row['center_intercepted_count'])} | "
            f"{int(row['center_release_count'])} | "
            f"{int(row['center_censored_count'])} | "
            f"{fmt(row['center_release_theta30_fraction'])} | "
            f"{int(row['center_well_intercepted_count'])} | "
            f"{int(row['center_well_release_count'])} | "
            f"{int(row['center_well_censored_count'])} | "
            f"{fmt(row['center_well_release_theta30_fraction'])} | "
            f"{fmt(row['center_well_release_theta30_halfwidth_95'])} | "
            f"{fmt(row['center_well_release_median_theta_deg'], 1)} | "
            f"{fmt(row['center_well_time_median_s'])} | "
            f"{fmt(row['center_well_net_travel_median_deg'], 1)} | "
            f"{fmt(row['center_well_cumulative_travel_median_deg'], 1)} |"
        )
    lines.extend(
        [
            "",
            "## Well geometry and residence anchors",
            "",
            "| profile | Umin (kBT) | h* (nm) | entry shell (nm) | escape basin (nm) | BOTE tau/2 (s) | BOTE tau (s) | outer inward drift (um/s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{fmt(row['well_minimum_kbt'])} | "
            f"{fmt(row['well_minimum_gap_nm'], 2)} | "
            f"{fmt(row['well_gap_lower_nm'], 2)}-{fmt(row['well_gap_upper_nm'], 2)} | "
            f"{fmt(row['well_basin_lower_nm'], 2)}-{fmt(row['well_basin_upper_nm'], 2)} | "
            f"{fmt(row['bote_tau_half_s'])} | "
            f"{fmt(row['bote_tau_full_s'])} | "
            f"{fmt(row['bote_outer_inward_speed_um_s'])} |"
        )
    lines.extend(
        [
            "",
            "## Tangential flow at particle centers",
            "",
            "| profile | collector | gap | gap (nm) | median abs u_t (um/s) | p95 abs u_t (um/s) | median angular rate (deg/s) | wall-damped median abs u_t (um/s) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in velocity_rows:
        if row["gap_label"] not in {"well_minimum", "10nm"}:
            continue
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{row['collector']} | "
            f"{row['gap_label']} | "
            f"{fmt(row['gap_nm'], 2)} | "
            f"{fmt(row['median_abs_ut_um_s'])} | "
            f"{fmt(row['p95_abs_ut_um_s'])} | "
            f"{fmt(row['median_abs_angular_rate_deg_s'])} | "
            f"{fmt(row['median_abs_wall_damped_ut_um_s'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the well-release focusing metric is conditioned on center-grain secondary-minimum entry and release. The 50 mM cases are fully completed after the adaptive horizon extension. Any residual 100 mM censored events are flagged as non physical trapping/attachment because homogeneous unfavorable attachment is disabled.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_bar(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, float | int | str]],
    metric: str,
    rect: tuple[int, int, int, int],
    title: str,
    label: str,
    limit: float | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    draw.text((x0, y0 - 40), title, fill="#111827", font=font(24, bold=True))
    values = []
    for row in rows:
        try:
            values.append(float(row[metric]))
        except (KeyError, TypeError, ValueError):
            values.append(float("nan"))
    finite = [v for v in values if np.isfinite(v)]
    vmax = limit if limit is not None else (max(finite) * 1.15 if finite else 1.0)
    vmax = max(vmax, 1.0e-12)
    bar_h = (y1 - y0) / max(len(rows), 1)
    colors = ["#4b5563", "#2563eb", "#0891b2", "#059669", "#dc2626"]
    for idx, (row, value) in enumerate(zip(rows, values)):
        y = y0 + idx * bar_h + 8
        draw.text((x0, y + 12), str(row["label"]), fill="#111827", font=font(16))
        bx0 = x0 + 260
        bx1 = x1 - 100
        by0 = y
        by1 = y + bar_h - 12
        draw.rectangle((bx0, by0, bx1, by1), outline="#d1d5db", width=1)
        if np.isfinite(value):
            fill_x = bx0 + (bx1 - bx0) * min(max(value / vmax, 0.0), 1.0)
            draw.rectangle((bx0, by0, fill_x, by1), fill=colors[idx % len(colors)])
            draw.text((bx1 + 12, y + 8), fmt(value, 2), fill="#111827", font=font(16))
    draw.text((x1 - 8, y1 + 8), label, fill="#6b7280", font=font(14), anchor="ra")


def write_figure(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    img = Image.new("RGB", (1800, 1320), "#ffffff")
    draw = ImageDraw.Draw(img)
    draw.text((900, 52), "Periodic-cell secondary-minimum focusing refinement", fill="#111827", font=font(34, bold=True), anchor="mm")
    draw.text(
        (900, 90),
        "Central-core injection; N=192 center/corner periodic flow; homogeneous unfavorable cases; no unfavorable attachment",
        fill="#374151",
        font=font(18),
        anchor="mm",
    )
    draw_bar(
        draw,
        rows,
        "center_well_release_theta30_fraction",
        (110, 165, 1700, 430),
        "Fraction of well releases within 30 deg of downstream stagnation",
        "fraction",
        limit=1.0,
    )
    draw_bar(
        draw,
        rows,
        "center_well_time_median_s",
        (110, 560, 1700, 825),
        "Median secondary-minimum residence time",
        "seconds",
    )
    draw_bar(
        draw,
        rows,
        "center_well_censored_count",
        (110, 955, 1700, 1220),
        "Long-residence censored center-well events",
        "count",
    )
    draw.text(
        (110, 1280),
        "Adaptive horizons complete the 50 mM cases; residual 100 mM unresolved particles are flagged for model review.",
        fill="#374151",
        font=font(18),
    )
    img.save(path)


def run_or_load(profile: dict[str, object], params: PhysicalParams, initial_y: np.ndarray, profile_id: int, reuse: bool):
    path = OUT / f"trajectory_library_{profile['profile']}.npz"
    if reuse and path.exists():
        candidate = load_library(path)
        if abs(candidate.params.max_time - params.max_time) < 1.0e-9:
            return candidate
    audit_path = ROOT / "outputs" / "censoring_audit" / (
        f"trajectory_library_central_core_refinement_{profile['profile']}_{int(round(params.max_time))}s.npz"
    )
    if reuse and audit_path.exists():
        lib = load_library(audit_path)
        save_library(path, lib)
        return lib
    flow = load_flow(FLOW_PATH, params=params)
    return simulate_cell_transitions_compiled(
        flow,
        str(profile["condition"]),
        seed=SEED + 10 + profile_id,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_kernel(force=False)
    reuse = "--reuse" in sys.argv
    reference_params = params_for(PROFILES[0])
    initial_y = central_core_y(reference_params)
    rows: list[dict[str, float | int | str]] = []
    velocity_rows: list[dict[str, float | str]] = []
    for idx, profile in enumerate(PROFILES):
        params = params_for(profile)
        flow = load_flow(FLOW_PATH, params=params)
        velocity_rows.extend(velocity_samples(flow, params, str(profile["profile"]), str(profile["condition"])))
        print(f"Running {profile['profile']} ({CORE_PARTICLES} particles)", flush=True)
        lib = run_or_load(profile, params, initial_y, idx, reuse)
        save_library(OUT / f"trajectory_library_{profile['profile']}.npz", lib)
        rows.append(summarize_library(profile, params, lib))
        write_dicts(OUT / "periodic_cell_refinement_summary_partial.csv", rows)
    write_dicts(OUT / "periodic_cell_refinement_summary.csv", rows)
    write_dicts(OUT / "periodic_cell_tangential_velocity.csv", velocity_rows)
    write_report(OUT / "periodic_cell_refinement_report.md", rows, velocity_rows)
    write_figure(OUT / "periodic_cell_refinement.png", rows)
    write_figure(FIGURES / "fig17_periodic_cell_refinement.png", rows)
    print(OUT / "periodic_cell_refinement_report.md", flush=True)


if __name__ == "__main__":
    main()
