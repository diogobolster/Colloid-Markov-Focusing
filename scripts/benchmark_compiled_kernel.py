#!/usr/bin/env python3
"""Benchmark the compiled particle kernel against the Python resolved solver."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled
from colloid_tsm.physical import PhysicalParams, load_flow, save_library, simulate_cell_transitions


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "compiled_benchmark"
VELOCITY_M_PER_DAY = 4.0
IONIC_STRENGTH_MM = 50.0
MAX_TIME_S = 60.0
N_COMPILED = 3000
N_COMPARE = 80
N_NEAR_WALL = 60
NEAR_WALL_TIME_S = 1.0
N_BINS = 32
CORE_BINS = np.array([15, 16])


def make_params(max_time: float) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=max_time,
        ionic_strength_molar=IONIC_STRENGTH_MM / 1000.0,
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
    )


def core_initial_y(params: PhysicalParams, n_particles: int, seed: int) -> np.ndarray:
    edges = np.linspace(0.0, params.cell_length, N_BINS + 1)
    core_y_min = edges[int(CORE_BINS[0])]
    core_y_max = edges[int(CORE_BINS[-1]) + 1]
    rng = np.random.default_rng(seed)
    return rng.uniform(core_y_min, core_y_max, size=n_particles)


def near_wall_initial(params: PhysicalParams, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(np.pi - 0.45, np.pi + 0.45, n_particles)
    radius = params.exclusion_radius + 10.0e-9
    x0 = 0.5 * params.cell_length + radius * np.cos(theta)
    y0 = 0.5 * params.cell_length + radius * np.sin(theta)
    return x0, y0


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def summarize(label: str, runtime_s: float, lib) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    intercepted = mobile & (lib.interceptions > 0)
    release = intercepted & np.isfinite(lib.theta_exit)
    return {
        "label": label,
        "particles": int(lib.y_in.size),
        "runtime_s": runtime_s,
        "particles_per_s": float(lib.y_in.size / runtime_s) if runtime_s > 0 else float("nan"),
        "exited_fraction": float(np.mean(lib.exited)),
        "censored_fraction": float(np.mean(lib.censored)),
        "any_interception_count": int(np.sum(lib.interceptions > 0)),
        "intercepted_mobile_count": int(np.sum(intercepted)),
        "release_count": int(np.sum(release)),
        "near_time_all_median_s": finite_median(lib.near_time),
        "near_time_median_s": finite_median(lib.near_time[release]),
        "hmin_all_median_nm": finite_median(lib.h_min * 1.0e9),
        "hmin_median_nm": finite_median(lib.h_min[release] * 1.0e9),
        "contact_fraction": float(np.mean(lib.contact_events > 0)),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
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
    transport_speedup = float("nan")
    near_wall_speedup = float("nan")
    compiled_row = next(row for row in rows if row["label"] == "compiled_60s")
    python_row = next(row for row in rows if row["label"] == "python_10s")
    compiled_small = next(row for row in rows if row["label"] == "compiled_10s")
    python_near = next(row for row in rows if row["label"] == "python_near_wall_1s")
    compiled_near = next(row for row in rows if row["label"] == "compiled_near_wall_1s")
    if float(compiled_small["runtime_s"]) > 0.0:
        transport_speedup = float(python_row["runtime_s"]) / float(compiled_small["runtime_s"])
    if float(compiled_near["runtime_s"]) > 0.0:
        near_wall_speedup = float(python_near["runtime_s"]) / float(compiled_near["runtime_s"])
    lines = [
        "# Compiled particle-kernel benchmark",
        "",
        f"C kernel compiled from `colloid_tsm/native/particle_kernel.c` and called through `colloid_tsm/compiled.py`.",
        f"Benchmark condition: {IONIC_STRENGTH_MM:.0f} mM unfavorable, resolved near-wall Langevin, center/corner periodic LBM flow, central-core inlet particles.",
        "",
        "| run | particles | horizon (s) | runtime (s) | particles/s | exited | censored | any interceptions | mobile interceptions | median near time all (s) | median hmin all (nm) | contact frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    horizons = {
        "compiled_60s": MAX_TIME_S,
        "compiled_10s": 10.0,
        "python_10s": 10.0,
        "compiled_near_wall_1s": NEAR_WALL_TIME_S,
        "python_near_wall_1s": NEAR_WALL_TIME_S,
    }
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{int(row['particles'])} | "
            f"{fmt(horizons[str(row['label'])], 1)} | "
            f"{fmt(row['runtime_s'])} | "
            f"{fmt(row['particles_per_s'])} | "
            f"{fmt(row['exited_fraction'])} | "
            f"{fmt(row['censored_fraction'])} | "
            f"{int(row['any_interception_count'])} | "
            f"{int(row['intercepted_mobile_count'])} | "
            f"{fmt(row['near_time_all_median_s'])} | "
            f"{fmt(row['hmin_all_median_nm'], 2)} | "
            f"{fmt(row['contact_fraction'])} |"
        )
    lines.extend(
        [
            "",
            f"Small transport-run wall-clock speedup for the same initial particles and 10 s horizon: {fmt(transport_speedup, 1)}x.",
            f"Near-wall bottleneck speedup for particles initialized at a 10 nm center-grain gap: {fmt(near_wall_speedup, 1)}x.",
            f"The production-sized compiled smoke run completed {int(compiled_row['particles'])} particles over {MAX_TIME_S:.0f} s with no unfavorable attachment allowed.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kernel_path = build_kernel(force=True)
    print(f"Built {kernel_path}")

    params = make_params(MAX_TIME_S)
    flow = load_flow(FLOW_PATH, params=params)
    y_compiled = core_initial_y(params, N_COMPILED, seed=981)

    t0 = time.perf_counter()
    compiled_lib = simulate_cell_transitions_compiled(
        flow,
        "unfavorable",
        seed=982,
        allow_attachment=False,
        initial_y=y_compiled,
        surface_mode="resolved_langevin",
    )
    compiled_runtime = time.perf_counter() - t0
    save_library(OUT / "trajectory_library_compiled_60s.npz", compiled_lib)

    compare_params = make_params(10.0)
    compare_flow = load_flow(FLOW_PATH, params=compare_params)
    y_compare = y_compiled[:N_COMPARE].copy()

    t0 = time.perf_counter()
    compiled_compare = simulate_cell_transitions_compiled(
        compare_flow,
        "unfavorable",
        seed=983,
        allow_attachment=False,
        initial_y=y_compare,
        surface_mode="resolved_langevin",
    )
    compiled_compare_runtime = time.perf_counter() - t0
    save_library(OUT / "trajectory_library_compiled_10s.npz", compiled_compare)

    t0 = time.perf_counter()
    python_compare = simulate_cell_transitions(
        compare_flow,
        "unfavorable",
        seed=983,
        allow_attachment=False,
        initial_y=y_compare,
        surface_mode="resolved_langevin",
    )
    python_compare_runtime = time.perf_counter() - t0
    save_library(OUT / "trajectory_library_python_10s.npz", python_compare)

    near_params = make_params(NEAR_WALL_TIME_S)
    near_flow = load_flow(FLOW_PATH, params=near_params)
    near_x, near_y = near_wall_initial(near_params, N_NEAR_WALL)

    t0 = time.perf_counter()
    compiled_near = simulate_cell_transitions_compiled(
        near_flow,
        "unfavorable",
        seed=984,
        allow_attachment=False,
        initial_x=near_x,
        initial_y=near_y,
        surface_mode="resolved_langevin",
    )
    compiled_near_runtime = time.perf_counter() - t0
    save_library(OUT / "trajectory_library_compiled_near_wall_1s.npz", compiled_near)

    t0 = time.perf_counter()
    python_near = simulate_cell_transitions(
        near_flow,
        "unfavorable",
        seed=984,
        allow_attachment=False,
        initial_x=near_x,
        initial_y=near_y,
        surface_mode="resolved_langevin",
    )
    python_near_runtime = time.perf_counter() - t0
    save_library(OUT / "trajectory_library_python_near_wall_1s.npz", python_near)

    rows = [
        summarize("compiled_60s", compiled_runtime, compiled_lib),
        summarize("compiled_10s", compiled_compare_runtime, compiled_compare),
        summarize("python_10s", python_compare_runtime, python_compare),
        summarize("compiled_near_wall_1s", compiled_near_runtime, compiled_near),
        summarize("python_near_wall_1s", python_near_runtime, python_near),
    ]
    write_csv(OUT / "compiled_kernel_benchmark.csv", rows)
    write_report(OUT / "compiled_kernel_report.md", rows)
    print(f"Done. Outputs written to {OUT}")


if __name__ == "__main__":
    main()
