#!/usr/bin/env python3
"""High-diffusion focusing diagnostic with nonattaching attractive closures."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import (
    PhysicalParams,
    focusing_metrics,
    load_flow,
    save_library,
    simulate_cell_transitions,
)


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "diffusion_sensitivity"

VELOCITY_M_PER_DAY = 4.0
DIFFUSIVITY_MULTIPLIER = 100.0
MAX_TIME_S = 180.0
N_UNIFORM = 12_000
N_CORE = 15_000
N_BINS = 32
CORE_BINS = np.array([15, 16])

CONDITIONS = (
    ("neutral", "no DLVO wall-mobility control", "lubrication"),
    ("favorable_reflecting", "attractive DLVO projection closure, nonattaching", "project"),
    ("favorable_sliding", "attractive DLVO surface-sliding closure, nonattaching", "sliding"),
    ("favorable_lubricated", "attractive DLVO lubrication closure, nonattaching", "lubrication"),
    ("unfavorable", "repulsive DLVO reflecting closure", "lubrication"),
)

REPORT_KEYS = (
    "ensemble",
    "condition",
    "description",
    "particles",
    "intercepted_any_count",
    "intercepted_mobile_count",
    "intercepted_censored_count",
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


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(REPORT_KEYS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REPORT_KEYS})


def fmt(value: float | int | str, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "nan"


def write_report(path: Path, rows: list[dict[str, float | int | str]], params: PhysicalParams) -> None:
    lines = [
        "# High-diffusion focusing sensitivity",
        "",
        f"Runs use the resolved N=192 center/corner flow at {VELOCITY_M_PER_DAY:.0f} m/day with 100x Brownian diffusivity.",
        f"The resulting diffusivity is {params.diffusivity:.4e} m^2/s and Pe = {params.particle_peclet:.1f}.",
        "Attachment is disabled in the attractive cases so release-angle focusing can be tested directly.",
        "",
        "| ensemble | condition | particles | center intercepted | center mobile | center censored | median abs theta_exit (rad) | theta_exit <30 deg | y_out within 25 um | median abs y_out-yc (um) | mean abs y_out-y_in (um) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ensemble']} | "
            f"{row['condition']} | "
            f"{int(row['particles'])} | "
            f"{int(row['center_intercepted_any_count'])} | "
            f"{int(row['center_intercepted_mobile_count'])} | "
            f"{int(row['center_intercepted_censored_count'])} | "
            f"{fmt(row['center_median_abs_theta_exit_downstream_rad'])} | "
            f"{fmt(row['center_fraction_theta_exit_within_30deg'])} | "
            f"{fmt(row['center_fraction_yout_within_center_band'])} | "
            f"{fmt(float(row['center_median_abs_yout_center_m']) * 1.0e6, 2)} | "
            f"{fmt(float(row['mean_abs_yout_yin_shift_m']) * 1.0e6, 2)} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=MAX_TIME_S,
        diffusivity_multiplier=DIFFUSIVITY_MULTIPLIER,
    )
    flow = load_flow(FLOW_PATH, params=params)
    rows: list[dict[str, float | int | str]] = []

    for condition_index, (condition, description, surface_mode) in enumerate(CONDITIONS):
        print(f"Uniform high-D focusing: {condition}")
        lib = simulate_cell_transitions(
            flow,
            condition,
            n_particles=N_UNIFORM,
            seed=31000 + condition_index,
            allow_attachment=False,
            surface_mode=surface_mode,
        )
        save_library(OUT / f"trajectory_library_focus_uniform_{condition}_100xD.npz", lib)
        row = focusing_metrics(lib)
        row["ensemble"] = "uniform"
        row["description"] = description
        rows.append(row)

    edges = np.linspace(0.0, params.cell_length, N_BINS + 1)
    core_y_min = edges[int(CORE_BINS[0])]
    core_y_max = edges[int(CORE_BINS[-1]) + 1]
    rng = np.random.default_rng(42000)
    for condition_index, (condition, description, surface_mode) in enumerate(CONDITIONS):
        print(f"Central-core high-D focusing: {condition}")
        initial_y = rng.uniform(core_y_min, core_y_max, size=N_CORE)
        lib = simulate_cell_transitions(
            flow,
            condition,
            n_particles=N_CORE,
            seed=43000 + condition_index,
            allow_attachment=False,
            initial_y=initial_y,
            surface_mode=surface_mode,
        )
        save_library(OUT / f"trajectory_library_focus_core_{condition}_100xD.npz", lib)
        row = focusing_metrics(lib)
        row["ensemble"] = "central-core"
        row["description"] = description
        rows.append(row)

    write_dicts(OUT / "diffusion_focusing_summary.csv", rows)
    write_report(OUT / "diffusion_focusing_report.md", rows, params)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
