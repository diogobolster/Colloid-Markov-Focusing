#!/usr/bin/env python3
"""Run a 50 mM unfavorable secondary-minimum focusing diagnostic."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import PhysicalParams, load_flow, save_library, simulate_cell_transitions


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "secondary_minimum_50mM"

VELOCITY_M_PER_DAY = 4.0
MAX_TIME_S = 240.0
N_UNIFORM = 12_000
N_CORE = 15_000
N_BINS = 32
CORE_BINS = np.array([15, 16])

PROFILES = (
    ("neutral", "no DLVO", 6.0, "neutral"),
    ("unfavorable_6mM", "unfavorable reference", 6.0, "unfavorable"),
    ("unfavorable_50mM", "unfavorable secondary-minimum test", 50.0, "unfavorable"),
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
        "barrier_gap_nm": float(h[max_id] * 1.0e9),
    }


def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.quantile(values, 0.05)), float(np.median(values)), float(np.quantile(values, 0.95))


def release_stats(lib, mask: np.ndarray) -> dict[str, float | int]:
    theta_mask = mask & np.isfinite(lib.theta_exit)
    theta_abs = np.abs(angular_distance(lib.theta_exit[theta_mask], 0.0))
    hmin_nm = lib.h_min[mask] * 1.0e9
    near_time = lib.near_time[mask]
    h05, h50, h95 = finite_stats(hmin_nm)
    t05, t50, t95 = finite_stats(near_time)
    return {
        "release_count": int(np.sum(theta_mask)),
        "median_abs_theta_rad": float(np.median(theta_abs)) if theta_abs.size else float("nan"),
        "fraction_theta_within_15deg": float(np.mean(theta_abs <= np.deg2rad(15.0))) if theta_abs.size else float("nan"),
        "fraction_theta_within_30deg": float(np.mean(theta_abs <= np.deg2rad(30.0))) if theta_abs.size else float("nan"),
        "hmin_p05_nm": h05,
        "hmin_median_nm": h50,
        "hmin_p95_nm": h95,
        "near_time_p05_s": t05,
        "near_time_median_s": t50,
        "near_time_p95_s": t95,
    }


def summarize_library(ensemble: str, profile: str, description: str, ionic_m_m: float, lib, dlvo: dict[str, float]) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    intercepted_any = lib.interceptions > 0
    intercepted_mobile = mobile & intercepted_any
    intercepted_censored = lib.censored & intercepted_any

    all_release = intercepted_mobile & np.isfinite(lib.theta_exit)
    center_release = all_release & (lib.collector_exit == 1)
    corner_release = all_release & (lib.collector_exit == 0)

    all_stats = release_stats(lib, all_release)
    center_stats = release_stats(lib, center_release)
    corner_stats = release_stats(lib, corner_release)
    intercepted_stats = release_stats(lib, intercepted_mobile)

    row: dict[str, float | int | str] = {
        "ensemble": ensemble,
        "profile": profile,
        "description": description,
        "ionic_strength_mM": ionic_m_m,
        "particles": int(lib.y_in.size),
        "exited_fraction": float(np.mean(lib.exited)),
        "censored_fraction": float(np.mean(lib.censored)),
        "intercepted_any_count": int(np.sum(intercepted_any)),
        "intercepted_mobile_count": int(np.sum(intercepted_mobile)),
        "intercepted_censored_count": int(np.sum(intercepted_censored)),
        "center_intercepted_any_count": int(np.sum(lib.center_interceptions > 0)),
        "center_release_count": int(np.sum(center_release)),
        "corner_intercepted_any_count": int(np.sum(lib.corner_interceptions > 0)),
        "corner_release_count": int(np.sum(corner_release)),
        **dlvo,
    }
    for prefix, stats in (
        ("all", all_stats),
        ("center", center_stats),
        ("corner", corner_stats),
        ("intercepted_mobile", intercepted_stats),
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
        "# 50 mM unfavorable secondary-minimum focusing diagnostic",
        "",
        f"All runs use the resolved N=192 center/corner periodic flow at {VELOCITY_M_PER_DAY:.0f} m/day, the normal Stokes-Einstein diffusivity, and a {MAX_TIME_S:.0f} s one-cell horizon.",
        "Release angles are computed relative to the downstream stagnation zone of each particle's nearest periodic collector, so the all-collector metric includes both the center grain and the corner-image grains.",
        "",
        "## DLVO profile anchors",
        "",
        "| profile | I (mM) | Debye (nm) | secondary min (kBT) | h_min (nm) | barrier (kBT) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    seen_profiles: set[str] = set()
    for row in rows:
        profile = str(row["profile"])
        if profile in seen_profiles or profile == "neutral":
            continue
        seen_profiles.add(profile)
        lines.append(
            "| "
            f"{profile} | "
            f"{float(row['ionic_strength_mM']):.1f} | "
            f"{float(row['debye_length_nm']):.2f} | "
            f"{float(row['secondary_minimum_kbt']):.2f} | "
            f"{float(row['secondary_minimum_gap_nm']):.2f} | "
            f"{float(row['barrier_kbt']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Periodic all-collector focusing",
            "",
            "| ensemble | profile | intercepted mobile | censored intercepted | all releases | all median abs theta (rad) | all theta <30 deg | all median near time (s) | all p95 near time (s) | all median hmin (nm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
            f"{fmt(row['all_hmin_median_nm'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Center versus corner-image releases",
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


def run_case(ensemble: str, profile: str, description: str, ionic_m_m: float, condition: str, initial_y: np.ndarray | None, seed: int):
    params = replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=MAX_TIME_S,
        ionic_strength_molar=ionic_m_m / 1000.0,
    )
    flow = load_flow(FLOW_PATH, params=params)
    n_particles = N_UNIFORM if initial_y is None else int(initial_y.size)
    lib = simulate_cell_transitions(
        flow,
        condition,
        n_particles=n_particles,
        seed=seed,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="lubrication",
    )
    save_library(OUT / f"trajectory_library_{ensemble}_{profile}.npz", lib)
    dlvo = dlvo_profile_metrics(params) if condition == "unfavorable" else {
        "debye_length_nm": params.debye_length * 1.0e9,
        "secondary_minimum_kbt": float("nan"),
        "secondary_minimum_gap_nm": float("nan"),
        "barrier_kbt": float("nan"),
        "barrier_gap_nm": float("nan"),
    }
    return summarize_library(ensemble, profile, description, ionic_m_m, lib, dlvo)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams()
    edges = np.linspace(0.0, params.cell_length, N_BINS + 1)
    core_y_min = edges[int(CORE_BINS[0])]
    core_y_max = edges[int(CORE_BINS[-1]) + 1]
    rng = np.random.default_rng(62000)
    core_y = rng.uniform(core_y_min, core_y_max, size=N_CORE)

    rows: list[dict[str, float | int | str]] = []
    for profile_index, (profile, description, ionic_m_m, condition) in enumerate(PROFILES):
        print(f"Uniform run: {profile}")
        rows.append(run_case("uniform", profile, description, ionic_m_m, condition, None, 61000 + profile_index))
    for profile_index, (profile, description, ionic_m_m, condition) in enumerate(PROFILES):
        print(f"Central-core run: {profile}")
        rows.append(run_case("central_core", profile, description, ionic_m_m, condition, core_y, 63000 + profile_index))

    write_dicts(OUT / "secondary_minimum_50mM_summary.csv", rows)
    write_report(OUT / "secondary_minimum_50mM_report.md", rows)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
