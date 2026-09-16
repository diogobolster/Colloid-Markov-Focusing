#!/usr/bin/env python3
"""Compare old lubrication and resolved near-wall Langevin tracking at 50 mM."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import simulate_cell_transitions_compiled
from colloid_tsm.physical import PhysicalParams, load_flow, save_library


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "resolved_langevin_50mM"

VELOCITY_M_PER_DAY = 4.0
MAX_TIME_S = 120.0
N_UNIFORM = 800
N_CORE = 1_000
N_BINS = 32
CORE_BINS = np.array([15, 16])

PROFILES = (
    ("neutral_lubrication", "no DLVO lubrication control", 6.0, "neutral", "lubrication"),
    ("unfavorable_6mM_lubrication", "6 mM unfavorable old lubrication closure", 6.0, "unfavorable", "lubrication"),
    ("unfavorable_6mM_resolved", "6 mM unfavorable resolved Langevin closure", 6.0, "unfavorable", "resolved_langevin"),
    ("unfavorable_50mM_lubrication", "50 mM unfavorable old lubrication closure", 50.0, "unfavorable", "lubrication"),
    ("unfavorable_50mM_resolved", "50 mM unfavorable resolved Langevin closure", 50.0, "unfavorable", "resolved_langevin"),
)

KB = 1.380649e-23


def angular_distance(theta: np.ndarray, target: float = 0.0) -> np.ndarray:
    return np.arctan2(np.sin(theta - target), np.cos(theta - target))


def dlvo_profile_metrics(params: PhysicalParams) -> dict[str, float]:
    h = np.logspace(-9, -6, 50_000)
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
    }


def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.quantile(values, 0.05)), float(np.median(values)), float(np.quantile(values, 0.95))


def release_stats(lib, mask: np.ndarray) -> dict[str, float | int]:
    theta_mask = mask & np.isfinite(lib.theta_exit)
    theta_abs = np.abs(angular_distance(lib.theta_exit[theta_mask], 0.0))
    h05, h50, h95 = finite_stats(lib.h_min[mask] * 1.0e9)
    t05, t50, t95 = finite_stats(lib.near_time[mask])
    contact = lib.contact_events[mask]
    return {
        "release_count": int(np.sum(theta_mask)),
        "median_abs_theta_rad": float(np.median(theta_abs)) if theta_abs.size else float("nan"),
        "fraction_theta_within_30deg": float(np.mean(theta_abs <= np.deg2rad(30.0))) if theta_abs.size else float("nan"),
        "hmin_p05_nm": h05,
        "hmin_median_nm": h50,
        "hmin_p95_nm": h95,
        "near_time_p05_s": t05,
        "near_time_median_s": t50,
        "near_time_p95_s": t95,
        "contact_fraction": float(np.mean(contact > 0)) if contact.size else float("nan"),
        "mean_contact_events": float(np.mean(contact)) if contact.size else float("nan"),
    }


def summarize(ensemble: str, profile: str, description: str, ionic_m_m: float, condition: str, surface_mode: str, lib, dlvo: dict[str, float]) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    intercepted = mobile & (lib.interceptions > 0)
    all_release = intercepted & np.isfinite(lib.theta_exit)
    center_release = all_release & (lib.collector_exit == 1)
    corner_release = all_release & (lib.collector_exit == 0)
    row: dict[str, float | int | str] = {
        "ensemble": ensemble,
        "profile": profile,
        "description": description,
        "ionic_strength_mM": ionic_m_m,
        "condition": condition,
        "surface_mode": surface_mode,
        "particles": int(lib.y_in.size),
        "exited_fraction": float(np.mean(lib.exited)),
        "censored_fraction": float(np.mean(lib.censored)),
        "intercepted_mobile_count": int(np.sum(intercepted)),
        "intercepted_censored_count": int(np.sum(lib.censored & (lib.interceptions > 0))),
        "center_release_count": int(np.sum(center_release)),
        "corner_release_count": int(np.sum(corner_release)),
        "overall_contact_fraction": float(np.mean(lib.contact_events > 0)),
        "overall_mean_contact_events": float(np.mean(lib.contact_events)),
        **dlvo,
    }
    for prefix, stats in (
        ("all", release_stats(lib, all_release)),
        ("center", release_stats(lib, center_release)),
        ("corner", release_stats(lib, corner_release)),
    ):
        for key, value in stats.items():
            row[f"{prefix}_{key}"] = value
    return row


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
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


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        "# Resolved near-wall Langevin diagnostic at 50 mM",
        "",
        f"Runs use the resolved N=192 center/corner periodic flow at {VELOCITY_M_PER_DAY:.0f} m/day, normal diffusivity, and a {MAX_TIME_S:.0f} s one-cell horizon. Particle tracking uses the compiled C backend. The resolved Langevin mode applies adaptive near-wall substeps inside 75 nm, uses the interpolated LBM velocity directly for flow advection, and applies wall-corrected normal/tangential mobility to DLVO drift, Brownian motion, and thermal drift.",
        "",
        "## Periodic all-collector releases",
        "",
        "| ensemble | profile | mobile interceptions | censored interceptions | releases | median theta | theta <30 deg | near time median (s) | near time p95 (s) | hmin median (nm) | contact frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ensemble']} | "
            f"{row['profile']} | "
            f"{int(row['intercepted_mobile_count'])} | "
            f"{int(row['intercepted_censored_count'])} | "
            f"{int(row['all_release_count'])} | "
            f"{fmt(row['all_median_abs_theta_rad'])} | "
            f"{fmt(row['all_fraction_theta_within_30deg'])} | "
            f"{fmt(row['all_near_time_median_s'])} | "
            f"{fmt(row['all_near_time_p95_s'])} | "
            f"{fmt(row['all_hmin_median_nm'], 2)} | "
            f"{fmt(row['all_contact_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Center and corner-image release split",
            "",
            "| ensemble | profile | center releases | center median theta | center theta <30 deg | corner releases | corner median theta | corner theta <30 deg |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['ensemble']} | "
            f"{row['profile']} | "
            f"{int(row['center_release_count'])} | "
            f"{fmt(row['center_median_abs_theta_rad'])} | "
            f"{fmt(row['center_fraction_theta_within_30deg'])} | "
            f"{int(row['corner_release_count'])} | "
            f"{fmt(row['corner_median_abs_theta_rad'])} | "
            f"{fmt(row['corner_fraction_theta_within_30deg'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(ensemble: str, profile: str, description: str, ionic_m_m: float, condition: str, surface_mode: str, initial_y: np.ndarray | None, seed: int):
    params = replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=MAX_TIME_S,
        ionic_strength_molar=ionic_m_m / 1000.0,
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
    )
    flow = load_flow(FLOW_PATH, params=params)
    n_particles = N_UNIFORM if initial_y is None else int(initial_y.size)
    lib = simulate_cell_transitions_compiled(
        flow,
        condition,
        n_particles=n_particles,
        seed=seed,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode=surface_mode,
    )
    save_library(OUT / f"trajectory_library_{ensemble}_{profile}.npz", lib)
    dlvo = dlvo_profile_metrics(params) if condition == "unfavorable" else {
        "debye_length_nm": params.debye_length * 1.0e9,
        "secondary_minimum_kbt": float("nan"),
        "secondary_minimum_gap_nm": float("nan"),
        "barrier_kbt": float("nan"),
    }
    return summarize(ensemble, profile, description, ionic_m_m, condition, surface_mode, lib, dlvo)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams()
    edges = np.linspace(0.0, params.cell_length, N_BINS + 1)
    core_y_min = edges[int(CORE_BINS[0])]
    core_y_max = edges[int(CORE_BINS[-1]) + 1]
    rng = np.random.default_rng(72000)
    core_y = rng.uniform(core_y_min, core_y_max, size=N_CORE)

    rows: list[dict[str, float | int | str]] = []
    for profile_index, profile in enumerate(PROFILES):
        print(f"Uniform run: {profile[0]}")
        rows.append(run_one("uniform", *profile, None, 71000 + profile_index))
    for profile_index, profile in enumerate(PROFILES):
        print(f"Central-core run: {profile[0]}")
        rows.append(run_one("central_core", *profile, core_y, 73000 + profile_index))
    write_dicts(OUT / "resolved_langevin_50mM_summary.csv", rows)
    write_report(OUT / "resolved_langevin_50mM_report.md", rows)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
