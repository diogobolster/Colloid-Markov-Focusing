#!/usr/bin/env python3
"""Passive-advection sanity checks before Stage 2 random-pore particle pilots."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)

OUT = ROOT / "outputs" / "stage2_particle_pilots"
SUMMARY = OUT / "stage2_flow_transport_sanity.csv"
REPORT = OUT / "stage2_flow_transport_sanity_report.md"
FLOW_SCREEN = ROOT / "outputs" / "stage1_geometry_screen" / "stage1_flow_screen_summary.csv"

STATIC_CANDIDATES = {
    "accepted_current": {
        "geometry": ROOT / "outputs" / "random_porous_geometry" / "random_porous_geometry.json",
        "flow": ROOT / "outputs" / "random_openfoam_flow_wall2_far5",
    },
    "random_00_body": {
        "geometry": ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / "random_00" / "geometry.json",
        "flow": ROOT / "outputs" / "stage1_openfoam_flow" / "random_00_snappy_screen",
    },
    "random_00_mean": {
        "geometry": ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / "random_00" / "geometry.json",
        "flow": ROOT / "outputs" / "stage1_openfoam_flow" / "random_00_snappy_mean",
    },
    "few_large_grains": {
        "geometry": ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / "few_large_grains" / "geometry.json",
        "flow": ROOT / "outputs" / "stage1_openfoam_flow" / "few_large_grains_snappy_screen",
    },
    "tight_throats": {
        "geometry": ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / "tight_throats" / "geometry.json",
        "flow": ROOT / "outputs" / "stage1_openfoam_flow" / "tight_throats_snappy_screen",
    },
    "high_polydispersity": {
        "geometry": ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / "high_polydispersity" / "geometry.json",
        "flow": ROOT / "outputs" / "stage1_openfoam_flow" / "high_polydispersity_snappy_screen",
    },
    "pockety_high_cv": {
        "geometry": ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / "pockety_high_cv" / "geometry.json",
        "flow": ROOT / "outputs" / "stage1_openfoam_flow" / "pockety_high_cv_snappy_screen",
    },
}


def discovered_stage1_candidates() -> dict[str, dict[str, Path]]:
    if not FLOW_SCREEN.exists():
        return {}
    discovered: dict[str, dict[str, Path]] = {}
    with FLOW_SCREEN.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate = str(row.get("candidate_id", ""))
            flow_case = str(row.get("flow_case", ""))
            if not candidate or not flow_case:
                continue
            geometry = ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / candidate / "geometry.json"
            flow = ROOT / flow_case
            if geometry.exists() and flow.exists():
                discovered[candidate] = {"geometry": geometry, "flow": flow}
    return discovered


def candidate_map() -> dict[str, dict[str, Path]]:
    candidates = dict(STATIC_CANDIDATES)
    candidates.update(discovered_stage1_candidates())
    return candidates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shifted_copy(geometry: dict, dx: float) -> dict:
    data = json.loads(json.dumps(geometry))
    lx = float(data["domain"]["length_x"])
    for grain in data["grains"]:
        grain["x"] = float((float(grain["x"]) - dx) % lx)
    return data


def candidate_interpolator(centers_xy: np.ndarray, velocity_xy: np.ndarray, lx: float, ly: float, dx: float) -> rpt.PeriodicIDWFlow:
    centers = centers_xy.copy()
    centers[:, 0] = (centers[:, 0] - dx) % lx
    return rpt.PeriodicIDWFlow(centers, velocity_xy, lx, ly, k=8)


def flux_weighted_start(
    rng: np.random.Generator,
    interpolator: rpt.PeriodicIDWFlow,
    disks: rpt.PeriodicDiskGeometry,
    params: rpt.RandomTrackingParams,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    while sum(chunk.size for chunk in samples) < n:
        y = rng.uniform(0.0, disks.ly, size=max(4096, n * 64))
        x = np.full(y.size, 1.0e-6)
        gap, *_ = disks.nearest_surface(x, y)
        ux, _ = interpolator.velocity_at(x, y)
        weight = np.where((gap > 2.0e-6) & (ux > 0.0), ux, 0.0)
        max_weight = float(np.max(weight))
        if max_weight <= 0.0:
            break
        accept = rng.uniform(0.0, max_weight, size=y.size) <= weight
        if np.any(accept):
            samples.append(y[accept])
    if not samples:
        raise RuntimeError("No positive-flux inlet samples.")
    y0 = np.concatenate(samples)[:n]
    if y0.size < n:
        raise RuntimeError("Insufficient positive-flux inlet samples.")
    return np.full(n, 1.0e-6), y0


def advect(
    interpolator: rpt.PeriodicIDWFlow,
    x: np.ndarray,
    y: np.ndarray,
    lx: float,
    ly: float,
    dt: float,
    max_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exited = np.zeros(x.size, dtype=bool)
    exit_time = np.full(x.size, np.nan)
    steps = int(np.ceil(max_time / dt))
    for step in range(steps):
        active = ~exited
        if not np.any(active):
            break
        ux, uy = interpolator.velocity_at(x[active], y[active])
        x[active] += ux * dt
        y[active] = np.mod(y[active] + uy * dt, ly)
        now = active & (x >= lx)
        if np.any(now):
            exited[now] = True
            exit_time[now] = (step + 1) * dt
    return x, exited, exit_time


def run_case(name: str, geometry_path: Path, flow_case: Path) -> dict[str, object]:
    geometry = rpt.load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=True)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    centers_xy, velocity_xy, areas = rpt.flow_io.parse_xml_vtu(rpt.latest_vtu(flow_case))
    velocity_xy, mean_ux, scale = rpt.rescale_velocity_to_target(velocity_xy, areas, rpt.TARGET_M_PER_DAY)
    params = rpt.RandomTrackingParams(diffusivity_multiplier=0.0)
    rng = np.random.default_rng(20260509)
    best: dict[str, object] | None = None
    for dx in np.linspace(0.0, lx, 24, endpoint=False):
        shifted_geometry = shifted_copy(geometry, float(dx))
        disks = rpt.PeriodicDiskGeometry(shifted_geometry, params.particle_radius)
        interpolator = candidate_interpolator(centers_xy, velocity_xy, lx, ly, float(dx))
        try:
            x0, y0 = flux_weighted_start(rng, interpolator, disks, params, 120)
        except RuntimeError:
            continue
        x_final, exited, exit_time = advect(interpolator, x0, y0, lx, ly, dt=0.02, max_time=300.0)
        row = {
            "candidate_id": name,
            "tracking_origin_shift_um": float(dx * 1.0e6),
            "flow_mean_ux_before_scale_m_s": mean_ux,
            "flow_scale": scale,
            "tracers": int(x0.size),
            "exit_fraction_300s": float(np.mean(exited)),
            "median_exit_time_s": float(np.nanmedian(exit_time)) if np.any(exited) else float("nan"),
            "median_x_300s_um": float(np.median(x_final) * 1.0e6),
            "max_x_300s_um": float(np.max(x_final) * 1.0e6),
        }
        if best is None or (row["exit_fraction_300s"], row["max_x_300s_um"]) > (
            best["exit_fraction_300s"],
            best["max_x_300s_um"],
        ):
            best = row
    if best is None:
        return {"candidate_id": name, "status": "failed", "error": "no positive-flux start plane"}
    best["status"] = "pass" if float(best["exit_fraction_300s"]) >= 0.5 else "fail"
    return best


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Stage 2 Flow Transport Sanity",
        "",
        "Deterministic passive tracers were flux-weighted at the diagnostic cut and advected for 300 s in the rescaled OpenFOAM field.",
        "A Stage 2 particle pilot should use only flows with substantial deterministic through-passage.",
        "",
        "| candidate | status | best shift (um) | exit fraction, 300 s | median x, 300 s (um) | max x, 300 s (um) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {status} | {tracking_origin_shift_um} | {exit_fraction_300s} | {median_x_300s_um} | {max_x_300s_um} |".format(
                candidate_id=row.get("candidate_id", ""),
                status=row.get("status", ""),
                tracking_origin_shift_um=f"{float(row['tracking_origin_shift_um']):.1f}" if "tracking_origin_shift_um" in row else "",
                exit_fraction_300s=f"{float(row['exit_fraction_300s']):.3f}" if "exit_fraction_300s" in row else "",
                median_x_300s_um=f"{float(row['median_x_300s_um']):.1f}" if "median_x_300s_um" in row else "",
                max_x_300s_um=f"{float(row['max_x_300s_um']):.1f}" if "max_x_300s_um" in row else "",
            )
        )
    lines.extend(["", f"CSV: `{SUMMARY.relative_to(ROOT)}`", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = candidate_map()
    requested = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if requested:
        candidates = {name: paths for name, paths in candidates.items() if name in set(requested)}
    rows = []
    for name, paths in candidates.items():
        print(f"sanity: {name}", flush=True)
        try:
            rows.append(run_case(name, paths["geometry"], paths["flow"]))
        except Exception as exc:
            rows.append({"candidate_id": name, "status": "failed", "error": str(exc)})
        write_csv(SUMMARY, rows)
    write_report(rows)
    print(SUMMARY)
    print(REPORT)


if __name__ == "__main__":
    main()
