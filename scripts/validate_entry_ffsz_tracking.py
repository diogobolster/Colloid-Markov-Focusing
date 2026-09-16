#!/usr/bin/env python3
"""Validate paired Grain-X to Grain-Y entry diagnostics in both geometries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import (  # noqa: E402
    RandomFlowGrid,
    simulate_cell_transitions_compiled,
    simulate_random_transitions_compiled,
)
from colloid_tsm.physical import FlowField, PhysicalParams  # noqa: E402


class ValidationFailure(AssertionError):
    """Raised when an entry-pair regression check fails."""


def assert_close(actual: float, expected: float, tolerance: float, message: str) -> None:
    error = abs(actual - expected)
    if error > tolerance:
        raise ValidationFailure(f"{message}: error={error:g}, tolerance={tolerance:g}")


def angular_error(actual: float, expected: float) -> float:
    return abs((actual - expected + math.pi) % (2.0 * math.pi) - math.pi)


def constant_periodic_flow(params: PhysicalParams, speed: float, resolution: int = 64) -> FlowField:
    coords = (np.arange(resolution, dtype=float) + 0.5) * params.cell_length / resolution
    velocity = np.full((resolution, resolution), speed, dtype=float)
    return FlowField(
        x=coords,
        y=coords,
        ux=velocity.copy(),
        uy=velocity.copy(),
        solid=np.zeros_like(velocity, dtype=bool),
        params=params,
        resolution=resolution,
        iterations=0,
        tau=1.0,
        mean_velocity_before_scale=speed,
        scale_factor=1.0,
    )


def periodic_case(dt: float, *, stop: bool) -> dict[str, np.ndarray]:
    params = replace(
        PhysicalParams(),
        diffusivity_multiplier=0.0,
        dt=dt,
        max_time=0.12,
    )
    speed = 1.0e-3
    flow = constant_periodic_flow(params, speed)
    theta = 0.25 * math.pi
    radius = params.exclusion_radius + 10.0e-9
    center = 0.5 * params.cell_length
    _, diagnostics = simulate_cell_transitions_compiled(
        flow,
        "neutral",
        seed=9107,
        allow_attachment=False,
        initial_x=np.array([center + radius * math.cos(theta)]),
        initial_y=np.array([center + radius * math.sin(theta)]),
        stop_at_next_distinct=stop,
        return_entry_diagnostics=True,
    )
    return diagnostics


def random_case(dt: float, *, stop: bool) -> tuple[dict[str, np.ndarray], PhysicalParams, dict]:
    lx = 1.0e-3
    ly = 4.0e-4
    radius = 5.0e-5
    speed = 1.0e-3
    params = replace(
        PhysicalParams(),
        cell_length=lx,
        grain_radius=radius,
        mean_velocity=speed,
        diffusivity_multiplier=0.0,
        dt=dt,
        max_time=0.35,
    )
    geometry = {
        "domain": {"length_x": lx, "length_y": ly},
        "grains": [
            {"x": 2.5e-4, "y": 2.0e-4, "radius": radius},
            {"x": 5.5e-4, "y": 2.0e-4, "radius": radius},
        ],
    }
    flow = RandomFlowGrid(
        lx=lx,
        ly=ly,
        ux=np.full((64, 32), speed, dtype=float),
        uy=np.zeros((64, 32), dtype=float),
    )
    start_radius = radius + params.particle_radius + 0.5 * params.near_surface
    _, diagnostics = simulate_random_transitions_compiled(
        flow,
        geometry,
        params,
        "neutral",
        seed=9109,
        allow_attachment=False,
        initial_x=np.array([geometry["grains"][0]["x"] + start_radius]),
        initial_y=np.array([geometry["grains"][0]["y"]]),
        stop_at_next_distinct=stop,
        return_entry_diagnostics=True,
    )
    return diagnostics, params, geometry


def shell_gap(x: float, y: float, grain: dict, particle_radius: float) -> float:
    return math.hypot(x - float(grain["x"]), y - float(grain["y"])) - (
        float(grain["radius"]) + particle_radius
    )


def validate_periodic() -> dict[str, float | int | str]:
    dts = [2.0e-5, 1.0e-5, 5.0e-6]
    runs = [periodic_case(dt, stop=True) for dt in dts]
    fine = runs[-1]
    if not bool(fine["pair_completed"][0]):
        raise ValidationFailure("periodic diagnostic did not complete a distinct-collector pair")
    if (int(fine["first_entry_collector"][0]), int(fine["next_entry_collector"][0])) != (1, 0):
        raise ValidationFailure("periodic pair was not center-family to corner-family")
    if angular_error(float(fine["first_entry_theta"][0]), 0.25 * math.pi) > 2.0e-6:
        raise ValidationFailure("periodic Grain-X entry angle is incorrect")
    if angular_error(float(fine["next_entry_theta"][0]), 1.25 * math.pi) > 2.0e-6:
        raise ValidationFailure("periodic Grain-Y entry angle is incorrect")

    times = np.array([float(run["next_entry_time"][0]) for run in runs])
    if abs(times[-1] - times[-2]) > 2.0 * dts[-2]:
        raise ValidationFailure("periodic next-entry first-passage time did not converge with timestep")

    continued = periodic_case(dts[-1], stop=False)
    for key in ("first_entry_theta", "first_entry_collector", "next_entry_theta", "next_entry_collector"):
        if not np.allclose(fine[key], continued[key], rtol=0.0, atol=1.0e-12, equal_nan=True):
            raise ValidationFailure(f"periodic write-once diagnostic changed after Grain-Y entry: {key}")
    return {
        "geometry": "periodic_center_corner",
        "status": "pass",
        "first_collector": int(fine["first_entry_collector"][0]),
        "next_collector": int(fine["next_entry_collector"][0]),
        "first_entry_angle_deg": math.degrees(float(fine["first_entry_theta"][0])),
        "next_entry_angle_deg": math.degrees(float(fine["next_entry_theta"][0])),
        "fine_next_entry_time_s": times[-1],
        "medium_to_fine_time_change_s": abs(times[-1] - times[-2]),
    }


def validate_random() -> dict[str, float | int | str]:
    dts = [2.0e-4, 1.0e-4, 5.0e-5]
    results = [random_case(dt, stop=True) for dt in dts]
    fine, params, geometry = results[-1]
    if not bool(fine["pair_completed"][0]):
        raise ValidationFailure("random diagnostic did not complete a distinct-collector pair")
    if (int(fine["first_entry_collector"][0]), int(fine["next_entry_collector"][0])) != (0, 1):
        raise ValidationFailure("random pair did not preserve the expected Grain-X and Grain-Y IDs")
    if angular_error(float(fine["first_entry_theta"][0]), 0.0) > 2.0e-6:
        raise ValidationFailure("random Grain-X entry angle is incorrect")
    if angular_error(float(fine["next_entry_theta"][0]), math.pi) > 2.0e-6:
        raise ValidationFailure("random Grain-Y entry angle is incorrect")

    next_gap = shell_gap(
        float(fine["next_entry_x"][0]),
        float(fine["next_entry_y"][0]),
        geometry["grains"][1],
        params.particle_radius,
    )
    assert_close(next_gap, params.near_surface, 2.0e-12, "random shell-crossing interpolation")
    times = np.array([float(result[0]["next_entry_time"][0]) for result in results])
    if abs(times[-1] - times[-2]) > 2.0 * dts[-2]:
        raise ValidationFailure("random next-entry first-passage time did not converge with timestep")

    continued, _, _ = random_case(dts[-1], stop=False)
    for key in ("first_entry_theta", "first_entry_collector", "next_entry_theta", "next_entry_collector"):
        if not np.allclose(fine[key], continued[key], rtol=0.0, atol=1.0e-12, equal_nan=True):
            raise ValidationFailure(f"random write-once diagnostic changed after Grain-Y entry: {key}")
    return {
        "geometry": "random_two_collector",
        "status": "pass",
        "first_collector": int(fine["first_entry_collector"][0]),
        "next_collector": int(fine["next_entry_collector"][0]),
        "first_entry_angle_deg": math.degrees(float(fine["first_entry_theta"][0])),
        "next_entry_angle_deg": math.degrees(float(fine["next_entry_theta"][0])),
        "next_entry_shell_error_nm": (next_gap - params.near_surface) * 1.0e9,
        "fine_next_entry_time_s": times[-1],
        "medium_to_fine_time_change_s": abs(times[-1] - times[-2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "entry_ffsz_validation" / "entry_ffsz_validation.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    try:
        rows.append(validate_periodic())
        rows.append(validate_random())
    except Exception as exc:
        rows.append({"status": "fail", "error": str(exc)})
        output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"PASS {row['geometry']}: {row}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
