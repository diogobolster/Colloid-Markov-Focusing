#!/usr/bin/env python3
"""A/B test near-wall hydrodynamic advection scaling in resolved Langevin tracking."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled
from colloid_tsm.physical import (
    PhysicalParams,
    angular_distance,
    dlvo_force_normal,
    load_flow,
    save_library,
    secondary_minimum_well,
)


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "hydrodynamic_advection_ab"
VELOCITY_M_PER_DAY = 4.0
IONIC_STRENGTH_M = 50.0e-3
MAX_TIME_S = 120.0
PARTICLES = 20_000
CORE_HALF_WIDTH_M = 12.5e-6
SEED = 991_000
KB = 1.380649e-23

CASES = (
    ("hydro_damped", 1.0, "Previous diagnostic mode: hydrodynamic advection scaled by near-wall mobility."),
    ("flow_direct", 0.0, "A/B mode: LBM advection used directly; wall mobility only modifies force drift and Brownian diffusion."),
)


def base_params(hydro_strength: float) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=MAX_TIME_S,
        ionic_strength_molar=IONIC_STRENGTH_M,
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
        resolved_langevin_hydro_strength=hydro_strength,
    )


def central_core_y(params: PhysicalParams) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    lo = 0.5 * params.cell_length - CORE_HALF_WIDTH_M
    hi = 0.5 * params.cell_length + CORE_HALF_WIDTH_M
    return rng.uniform(lo, hi, size=PARTICLES)


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def fraction(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def residence_bote(params: PhysicalParams) -> dict[str, float]:
    well = secondary_minimum_well(params, "unfavorable")
    h_star = float(well["minimum_gap"])
    u_min = float(well["minimum_potential_kbt"])
    normal_mobility = h_star / (h_star + params.particle_radius)
    diffusivity_normal = params.diffusivity * normal_mobility
    length = h_star
    exp_factor = math.exp(-u_min)
    tau_half = length * length / (2.0 * diffusivity_normal) * exp_factor
    tau_full = length * length / diffusivity_normal * exp_factor
    force_outer = float(dlvo_force_normal(params, np.array([float(well["upper"])]), "unfavorable")[0])
    stokes_mobility = 1.0 / (6.0 * math.pi * params.viscosity * params.particle_radius)
    inward_speed_outer = -stokes_mobility * force_outer * normal_mobility
    return {
        "well_lower_nm": float(well["lower"]) * 1.0e9,
        "well_upper_nm": float(well["upper"]) * 1.0e9,
        "well_basin_lower_nm": float(well["basin_lower"]) * 1.0e9,
        "well_basin_upper_nm": float(well["basin_upper"]) * 1.0e9,
        "h_star_nm": h_star * 1.0e9,
        "u_min_kbt": u_min,
        "normal_mobility": normal_mobility,
        "diffusivity_normal_m2_s": diffusivity_normal,
        "length_nm": length * 1.0e9,
        "exp_factor": exp_factor,
        "tau_half_s": tau_half,
        "tau_full_s": tau_full,
        "outer_force_pn": force_outer * 1.0e12,
        "outer_inward_speed_um_s": inward_speed_outer * 1.0e6,
    }


def summarize(label: str, hydro_strength: float, description: str, lib, bote: dict[str, float]) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    center_near_release = (
        mobile
        & (lib.center_interceptions > 0)
        & (lib.collector_exit == 1)
        & np.isfinite(lib.theta_exit)
    )
    center_well_any = lib.center_well_interceptions > 0
    center_well_release = (
        mobile
        & center_well_any
        & (lib.collector_well_exit == 1)
        & np.isfinite(lib.theta_well_exit)
    )
    center_well_censored = lib.censored & center_well_any
    theta_near = np.abs(angular_distance(lib.theta_exit[center_near_release], 0.0))
    theta_well = np.abs(angular_distance(lib.theta_well_exit[center_well_release], 0.0))
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
    well_travel = well_angular_travel[center_well_release]
    well_net_travel = np.abs(well_net_angular_travel[center_well_release])
    return {
        "case": label,
        "description": description,
        "hydro_strength": hydro_strength,
        "particles": int(lib.y_in.size),
        "exited_fraction": float(np.mean(lib.exited)),
        "censored_fraction": float(np.mean(lib.censored)),
        "near_center_releases": int(np.sum(center_near_release)),
        "near_theta30_fraction": fraction(theta_near <= np.deg2rad(30.0)),
        "near_median_abs_theta_deg": finite_median(np.degrees(theta_near)),
        "near_mean_abs_theta_deg": finite_mean(np.degrees(theta_near)),
        "well_center_entries": int(np.sum(center_well_any)),
        "well_center_releases": int(np.sum(center_well_release)),
        "well_center_censored": int(np.sum(center_well_censored)),
        "well_theta30_fraction": fraction(theta_well <= np.deg2rad(30.0)),
        "well_median_abs_theta_deg": finite_median(np.degrees(theta_well)),
        "well_mean_abs_theta_deg": finite_mean(np.degrees(theta_well)),
        "well_median_angular_travel_deg": finite_median(np.degrees(well_travel)),
        "well_p95_angular_travel_deg": float(np.quantile(np.degrees(well_travel), 0.95)) if well_travel.size else float("nan"),
        "well_median_net_angular_travel_deg": finite_median(np.degrees(well_net_travel)),
        "well_time_median_s": finite_median(lib.well_time[center_well_any]),
        "well_time_p95_s": float(np.quantile(lib.well_time[center_well_any], 0.95)) if np.any(center_well_any) else float("nan"),
        "well_lower_nm": bote["well_lower_nm"],
        "well_upper_nm": bote["well_upper_nm"],
        "well_basin_lower_nm": bote["well_basin_lower_nm"],
        "well_basin_upper_nm": bote["well_basin_upper_nm"],
        "bote_tau_half_s": bote["tau_half_s"],
        "bote_tau_full_s": bote["tau_full_s"],
        "bote_normal_diffusivity_m2_s": bote["diffusivity_normal_m2_s"],
        "bote_outer_inward_speed_um_s": bote["outer_inward_speed_um_s"],
    }


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | int | str, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "nan"


def write_report(path: Path, rows: list[dict[str, float | int | str]], bote: dict[str, float]) -> None:
    lines = [
        "# Hydrodynamic advection A/B test",
        "",
        f"Runs use central-core injection with {PARTICLES:,} particles, 50 mM unfavorable DLVO, the N=192 center/corner periodic LBM flow at {VELOCITY_M_PER_DAY:.0f} m/day, and a {MAX_TIME_S:.0f} s horizon.",
        "",
        "## Back-of-the-envelope residence estimate",
        "",
        f"For this chemistry, secondary-minimum bound-state entry is {bote['well_lower_nm']:.2f}-{bote['well_upper_nm']:.2f} nm, escape uses the broader {bote['well_basin_lower_nm']:.2f}-{bote['well_basin_upper_nm']:.2f} nm basin, and the minimum is at {bote['h_star_nm']:.2f} nm with Umin = {bote['u_min_kbt']:.2f} kBT.",
        f"At the minimum, the normal mobility factor is about {bote['normal_mobility']:.4f}, so D_perp = {bote['diffusivity_normal_m2_s']:.2e} m2/s.",
        f"Using tau ~ ell^2/(2 D_perp) exp(Delta U/kBT) with ell ~ h* gives {bote['tau_half_s']:.1f} s; dropping the factor of 2 gives {bote['tau_full_s']:.1f} s. That puts a tens-of-seconds residence time in the right order of magnitude.",
        f"At the outer Umin+1 well edge, the DLVO force corresponds to an inward drift of about {bote['outer_inward_speed_um_s']:.2f} um/s after normal wall mobility scaling.",
        "",
        "## A/B results",
        "",
        "| case | hydro strength | bound entries | bound releases | bound censored | median bound time (s) | escape theta <30 deg | median escape angle (deg) | median cumulative travel (deg) | p95 cumulative travel (deg) | median net travel (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['case']} | "
            f"{fmt(row['hydro_strength'], 1)} | "
            f"{int(row['well_center_entries'])} | "
            f"{int(row['well_center_releases'])} | "
            f"{int(row['well_center_censored'])} | "
            f"{fmt(row['well_time_median_s'])} | "
            f"{fmt(row['well_theta30_fraction'])} | "
            f"{fmt(row['well_median_abs_theta_deg'], 1)} | "
            f"{fmt(row['well_median_angular_travel_deg'], 2)} | "
            f"{fmt(row['well_p95_angular_travel_deg'], 2)} | "
            f"{fmt(row['well_median_net_angular_travel_deg'], 2)} |"
        )
    lines.extend(
        [
            "",
            "The angular-travel columns are path-integrated while the particle is in the secondary-minimum basin; net travel separately reports the wrapped entry-to-exit displacement. The `hydro_damped` case is the previous implementation. The `flow_direct` case uses the same DLVO and Brownian wall corrections but does not multiply the interpolated LBM flow velocity by the wall-mobility factors.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_kernel(force=False)
    reference_params = base_params(1.0)
    y0 = central_core_y(reference_params)
    bote = residence_bote(reference_params)
    rows: list[dict[str, float | int | str]] = []
    for offset, (label, hydro_strength, description) in enumerate(CASES):
        print(f"Running {label}", flush=True)
        params = base_params(hydro_strength)
        flow = load_flow(FLOW_PATH, params=params)
        lib = simulate_cell_transitions_compiled(
            flow,
            "unfavorable",
            seed=SEED + 10 + offset,
            allow_attachment=False,
            initial_y=y0,
            surface_mode="resolved_langevin",
            force_rebuild=False,
        )
        save_library(OUT / f"trajectory_library_{label}.npz", lib)
        rows.append(summarize(label, hydro_strength, description, lib, bote))
        write_dicts(OUT / "hydrodynamic_advection_ab_summary_partial.csv", rows)
    write_dicts(OUT / "hydrodynamic_advection_ab_summary.csv", rows)
    write_report(OUT / "hydrodynamic_advection_ab_report.md", rows, bote)
    print(OUT / "hydrodynamic_advection_ab_report.md")


if __name__ == "__main__":
    main()
