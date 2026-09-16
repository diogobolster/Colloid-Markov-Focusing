#!/usr/bin/env python3
"""Validation checks for random OpenFOAM particle tracking infrastructure."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import RandomFlowGrid, simulate_random_transitions_compiled

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)

DEFAULT_OUT = ROOT / "outputs" / "random_particle_tracking_validation"


class ValidationFailure(AssertionError):
    """Raised when a validation check fails."""


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def flow_interpolator(flow_case: Path, geometry: dict):
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, _ = rpt.flow_io.parse_xml_vtu(vtu)
    return rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8)


def rasterize_flow_grid(interpolator, lx: float, ly: float, nx: int = 96, ny: int = 64) -> RandomFlowGrid:
    xs = (np.arange(nx, dtype=float) + 0.5) * lx / nx
    ys = (np.arange(ny, dtype=float) + 0.5) * ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    ux, uy = interpolator.velocity_at(xx.ravel(), yy.ravel())
    return RandomFlowGrid(
        lx=lx,
        ly=ly,
        ux=np.ascontiguousarray(ux.reshape(nx, ny)),
        uy=np.ascontiguousarray(uy.reshape(nx, ny)),
    )


def test_segment_collision_reflection(geometry: dict) -> dict[str, float | int | str]:
    params = rpt.RandomTrackingParams(diffusivity_multiplier=0.0)
    disks = rpt.PeriodicDiskGeometry(geometry, params.particle_radius)
    gid = int(np.argmax(disks.radii))
    cx, cy = disks.centers[gid]
    radius = disks.radii[gid] + params.particle_radius + 5.0e-6
    x0 = np.array([cx - radius])
    y0 = np.array([cy])
    x1 = np.array([cx + radius])
    y1 = np.array([cy])
    active = np.ones(1, dtype=bool)
    hit, hit_t, hit_grain, _, _ = disks.segment_contacts(x0, y0, x1, y1, active, params)
    assert_true(bool(hit[0]), "straight segment through grain did not report contact")
    assert_true(int(hit_grain[0]) == gid, "segment contact picked the wrong grain")
    xr, yr, reflected = disks.reflect_segments(x0, y0, x1, y1, active, params)
    assert_true(bool(reflected[0]), "segment through grain was not reflected")
    gap, *_ = disks.nearest_surface(xr, yr)
    assert_true(float(gap[0]) >= params.contact_gap - 1.0e-12, "reflected point ended inside exclusion boundary")
    return {
        "test": "segment_collision_reflection",
        "status": "pass",
        "hit_t": float(hit_t[0]),
        "final_gap_nm": float(gap[0] * 1.0e9),
    }


def test_near_surface_no_diffusion(geometry: dict, interpolator) -> dict[str, float | int | str]:
    params = rpt.RandomTrackingParams(
        diffusivity_multiplier=0.0,
        max_time=0.08,
        dt=2.0e-3,
        ionic_strength_molar=50.0e-3,
        adaptive_near_wall=True,
        segment_collision=True,
    )
    disks = rpt.PeriodicDiskGeometry(geometry, params.particle_radius)
    gid = int(np.argmax(disks.radii))
    cx, cy = disks.centers[gid]
    radius = disks.radii[gid] + params.particle_radius + 20.0e-9
    theta = np.linspace(-0.45, 0.45, 24)
    x0 = cx + radius * np.cos(theta)
    y0 = np.mod(cy + radius * np.sin(theta), disks.ly)
    result = rpt.simulate_particles(
        "unfavorable",
        interpolator,
        geometry,
        params,
        n_particles=int(theta.size),
        seed=9191,
        initial_x=x0,
        initial_y=y0,
        record_count=0,
    )
    final_gap, *_ = disks.nearest_surface(result.x_final, result.y_final)
    assert_true(int(np.sum(result.attached)) == 0, "unfavorable no-diffusion near-surface run attached particles")
    assert_true(float(np.min(final_gap)) >= params.contact_gap - 1.0e-12, "near-surface no-diffusion run penetrated collector")
    return {
        "test": "near_surface_no_diffusion",
        "status": "pass",
        "particles": int(theta.size),
        "attached": int(np.sum(result.attached)),
        "min_final_gap_nm": float(np.min(final_gap) * 1.0e9),
        "median_near_time_s": float(np.median(result.near_time)),
    }


def test_reproducible_smoke(geometry: dict, interpolator) -> dict[str, float | int | str]:
    params = rpt.RandomTrackingParams(max_time=0.4, dt=4.0e-3, adaptive_near_wall=True, segment_collision=True)
    rng = np.random.default_rng(771)
    disks = rpt.PeriodicDiskGeometry(geometry, params.particle_radius)
    y0 = rpt.open_inlet_samples(rng, 12, disks.lx, disks.ly, disks.centers, disks.radii, params, 1.0e-6)
    kwargs = dict(
        condition="no_dlvo",
        interpolator=interpolator,
        geometry=geometry,
        params=params,
        n_particles=y0.size,
        seed=818,
        initial_y=y0,
        record_count=0,
    )
    result1 = rpt.simulate_particles(**kwargs)
    result2 = rpt.simulate_particles(**kwargs)
    assert_true(np.allclose(result1.x_final, result2.x_final), "smoke run x_final is not reproducible")
    assert_true(np.allclose(result1.y_final, result2.y_final), "smoke run y_final is not reproducible")
    assert_true(int(np.sum(result1.attached)) == 0, "no-DLVO smoke run attached particles")
    return {
        "test": "reproducible_smoke",
        "status": "pass",
        "particles": int(y0.size),
        "censored": int(np.sum(result1.censored)),
        "attached": int(np.sum(result1.attached)),
    }


def test_compiled_near_surface_no_diffusion(geometry: dict, interpolator) -> dict[str, float | int | str]:
    params = rpt.RandomTrackingParams(
        diffusivity_multiplier=0.0,
        max_time=0.08,
        dt=2.0e-3,
        ionic_strength_molar=50.0e-3,
        adaptive_near_wall=True,
        segment_collision=True,
    )
    disks = rpt.PeriodicDiskGeometry(geometry, params.particle_radius)
    grid = rasterize_flow_grid(interpolator, disks.lx, disks.ly)
    gid = int(np.argmax(disks.radii))
    cx, cy = disks.centers[gid]
    radius = disks.radii[gid] + params.particle_radius + 20.0e-9
    theta = np.linspace(-0.45, 0.45, 24)
    x0 = cx + radius * np.cos(theta)
    y0 = np.mod(cy + radius * np.sin(theta), disks.ly)
    result, diag = simulate_random_transitions_compiled(
        grid,
        geometry,
        params.as_physical_params(),
        "unfavorable",
        n_particles=int(theta.size),
        seed=9191,
        allow_attachment=False,
        initial_x=x0,
        initial_y=y0,
        return_timestep_diagnostics=True,
    )
    final_gap, *_ = disks.nearest_surface(result.x_final, result.y_final)
    assert_true(int(np.sum(result.attached)) == 0, "compiled unfavorable no-diffusion run attached particles")
    assert_true(float(np.min(final_gap)) >= params.contact_gap - 1.0e-12, "compiled no-diffusion run penetrated collector")
    assert_true(int(np.sum(diag["guard_hits"])) == 0, "compiled adaptive random kernel hit guard limit")
    assert_true(np.all(result.grain_entry == gid), "compiled near-wall entries were not tagged to the expected grain")
    assert_true(np.all(result.grain_final == gid), "compiled near-wall final states were not tagged to the expected grain")
    assert_true(np.all(np.isfinite(result.theta_entry)), "compiled near-wall entries did not record entry angles")
    assert_true(np.all((result.theta_entry >= 0.0) & (result.theta_entry < 2.0 * math.pi)), "entry angles are out of range")
    return {
        "test": "compiled_near_surface_no_diffusion",
        "status": "pass",
        "particles": int(theta.size),
        "attached": int(np.sum(result.attached)),
        "min_final_gap_nm": float(np.min(final_gap) * 1.0e9),
        "median_near_time_s": float(np.median(result.near_time)),
        "max_adaptive_substeps": int(np.max(diag["adaptive_substeps"])),
        "tagged_grain": gid,
    }


def write_report(out_dir: Path, rows: list[dict[str, float | int | str]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "random_particle_tracking_validation.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = out_dir / "random_particle_tracking_validation.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Random particle-tracking validation",
        "",
        "| test | status | details |",
        "|---|---|---|",
    ]
    for row in rows:
        details = ", ".join(f"{key}={value}" for key, value in row.items() if key not in {"test", "status"})
        lines.append(f"| {row['test']} | {row['status']} | {details} |")
    path = out_dir / "random_particle_tracking_validation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-case", type=Path, default=rpt.DEFAULT_FLOW_CASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    geometry = rpt.load_geometry()
    rows: list[dict[str, float | int | str]] = []
    try:
        rows.append(test_segment_collision_reflection(geometry))
        interpolator = flow_interpolator(args.flow_case, geometry)
        rows.append(test_near_surface_no_diffusion(geometry, interpolator))
        rows.append(test_reproducible_smoke(geometry, interpolator))
        rows.append(test_compiled_near_surface_no_diffusion(geometry, interpolator))
    except Exception as exc:
        rows.append({"test": "validation", "status": "fail", "error": str(exc)})
        report = write_report(out_dir.resolve(), rows)
        print(report, file=sys.stderr)
        return 1
    report = write_report(out_dir.resolve(), rows)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
