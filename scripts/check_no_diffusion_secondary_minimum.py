#!/usr/bin/env python3
"""Deterministic no-diffusion secondary-minimum retention check."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import simulate_cell_transitions_compiled
from colloid_tsm.physical import PhysicalParams, load_flow


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "no_diffusion_trap_check"
VELOCITY_M_PER_DAY = 4.0
IONIC_STRENGTH_M = 50.0e-3
MAX_TIME_S = 120.0
INITIAL_GAP_M = 10.23e-9
RESIDENCE_TOLERANCE_S = 1.0e-6
ANGLES_RAD = np.array([np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.0, -0.5 * np.pi])


def nearest_surface_state(params: PhysicalParams, x: float, y: float) -> tuple[float, float, int]:
    length = params.cell_length
    corner_cx = length * round(x / length)
    corner_cy = length * round(y / length)
    corner_rx = x - corner_cx
    corner_ry = y - corner_cy
    corner_r2 = corner_rx * corner_rx + corner_ry * corner_ry

    center_cx = length * (round((x - 0.5 * length) / length) + 0.5)
    center_cy = length * (round((y - 0.5 * length) / length) + 0.5)
    center_rx = x - center_cx
    center_ry = y - center_cy
    center_r2 = center_rx * center_rx + center_ry * center_ry

    if center_r2 <= corner_r2:
        rx, ry, r2, collector = center_rx, center_ry, center_r2, 1
    else:
        rx, ry, r2, collector = corner_rx, corner_ry, corner_r2, 0
    r = max(r2**0.5, 1.0e-30)
    h = r - params.exclusion_radius
    theta = np.arctan2(ry, rx)
    return h, theta, collector


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    escaped = [row for row in rows if row["escaped_near_surface"] == 1]
    lines = [
        "# No-diffusion secondary-minimum retention check",
        "",
        f"One deterministic particle is initialized at each listed center-grain angle with `D=0`, 50 mM unfavorable DLVO, an initial gap of {INITIAL_GAP_M * 1.0e9:.2f} nm, and a {MAX_TIME_S:.0f} s horizon. A particle is counted as escaping the near-surface zone if the trajectory records a finite `theta_exit` or accumulates less than the full horizon as near-surface residence time by more than {RESIDENCE_TOLERANCE_S:g} s.",
        "",
        "| start angle (deg) | escaped? | near time (s) | h_min (nm) | h_final (nm) | final angle (deg) | exited cell? | censored? |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{float(row['start_angle_deg']):.1f} | "
            f"{int(row['escaped_near_surface'])} | "
            f"{float(row['near_time_s']):.6f} | "
            f"{float(row['h_min_nm']):.6f} | "
            f"{float(row['h_final_nm']):.6f} | "
            f"{float(row['theta_final_deg']):.3f} | "
            f"{int(row['exited'])} | "
            f"{int(row['censored'])} |"
        )
    lines.extend(
        [
            "",
            f"Result: {len(escaped)} of {len(rows)} initialized particles escaped the near-surface zone.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=MAX_TIME_S,
        ionic_strength_molar=IONIC_STRENGTH_M,
        diffusivity_multiplier=0.0,
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
    )
    flow = load_flow(FLOW_PATH, params=params)
    center = 0.5 * params.cell_length
    radius = params.exclusion_radius + INITIAL_GAP_M
    x0 = center + radius * np.cos(ANGLES_RAD)
    y0 = center + radius * np.sin(ANGLES_RAD)
    lib = simulate_cell_transitions_compiled(
        flow,
        "unfavorable",
        seed=41001,
        allow_attachment=False,
        initial_x=x0,
        initial_y=y0,
        surface_mode="resolved_langevin",
    )

    rows: list[dict[str, float | int | str]] = []
    for i, angle in enumerate(ANGLES_RAD):
        h_final, theta_final, collector_final = nearest_surface_state(params, float(lib.x_final[i]), float(lib.y_final[i]))
        escaped_near_surface = bool(np.isfinite(lib.theta_exit[i])) or lib.near_time[i] < MAX_TIME_S - RESIDENCE_TOLERANCE_S
        rows.append(
            {
                "start_angle_deg": float(np.degrees(angle)),
                "escaped_near_surface": int(escaped_near_surface),
                "near_time_s": float(lib.near_time[i]),
                "h_min_nm": float(lib.h_min[i] * 1.0e9),
                "h_final_nm": float(h_final * 1.0e9),
                "theta_final_deg": float(np.degrees(theta_final)),
                "collector_final": int(collector_final),
                "interceptions": int(lib.interceptions[i]),
                "theta_exit_finite": int(np.isfinite(lib.theta_exit[i])),
                "exited": int(lib.exited[i]),
                "censored": int(lib.censored[i]),
                "x_final_m": float(lib.x_final[i]),
                "y_final_m": float(lib.y_final[i]),
            }
        )
    write_dicts(OUT / "no_diffusion_secondary_minimum_check.csv", rows)
    write_report(OUT / "no_diffusion_secondary_minimum_check.md", rows)
    escaped_count = sum(int(row["escaped_near_surface"]) for row in rows)
    if escaped_count:
        raise SystemExit(f"{escaped_count} no-diffusion particles escaped the near-surface zone")
    print(OUT / "no_diffusion_secondary_minimum_check.md")


if __name__ == "__main__":
    main()
