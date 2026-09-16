#!/usr/bin/env python3
"""Stage 1 geometry ensemble screen for random pore-scale realizations.

This stage intentionally avoids particle tracking. It generates random and
designed periodic disk packings, computes geometry-only descriptors, and writes
each geometry to a self-contained folder so selected candidates can be passed
directly to the OpenFOAM flow solver.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEN_SCRIPT = ROOT / "scripts" / "generate_random_porous_geometry.py"
spec = importlib.util.spec_from_file_location("geometry_generator", GEN_SCRIPT)
gen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gen
spec.loader.exec_module(gen)

OUT = ROOT / "outputs" / "stage1_geometry_screen"
GEOM_DIR = OUT / "geometries"
BASELINE = ROOT / "outputs" / "random_porous_geometry" / "random_porous_geometry.json"


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
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


def periodic_delta(delta: np.ndarray | float, length: float) -> np.ndarray | float:
    return delta - length * np.round(delta / length)


def surface_distance(
    xx: np.ndarray,
    yy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    lx: float,
    ly: float,
) -> np.ndarray:
    distance = np.full_like(xx, np.inf, dtype=float)
    for (cx, cy), radius in zip(centers, radii, strict=True):
        dx = periodic_delta(xx - cx, lx)
        dy = periodic_delta(yy - cy, ly)
        distance = np.minimum(distance, np.hypot(dx, dy) - radius)
    return distance


def clearance_descriptors(centers: np.ndarray, radii: np.ndarray, cfg: gen.GeometryConfig) -> dict[str, float]:
    nx, ny = 480, 320
    x = (np.arange(nx) + 0.5) * cfg.length_x / nx
    y = (np.arange(ny) + 0.5) * cfg.length_y / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    clearance = surface_distance(xx, yy, centers, radii, cfg.length_x, cfg.length_y)
    fluid_clearance = clearance[clearance > 0.0]
    if fluid_clearance.size == 0:
        return {}
    high_clearance = fluid_clearance > np.quantile(fluid_clearance, 0.90)
    p = fluid_clearance / np.sum(fluid_clearance)
    entropy_effective_cells = math.exp(float(-np.sum(p * np.log(p))))
    xwise_open = np.mean(clearance > 0.0, axis=1)
    return {
        "clearance_median_um": float(np.median(fluid_clearance) * 1.0e6),
        "clearance_q10_um": float(np.quantile(fluid_clearance, 0.10) * 1.0e6),
        "clearance_q90_um": float(np.quantile(fluid_clearance, 0.90) * 1.0e6),
        "clearance_q99_um": float(np.quantile(fluid_clearance, 0.99) * 1.0e6),
        "max_clearance_um": float(np.max(fluid_clearance) * 1.0e6),
        "high_clearance_area_fraction": float(np.mean(high_clearance)),
        "clearance_entropy_effective_fraction": float(entropy_effective_cells / fluid_clearance.size),
        "xwise_open_fraction_cv": float(np.std(xwise_open) / max(np.mean(xwise_open), 1.0e-30)),
        "xwise_open_fraction_min": float(np.min(xwise_open)),
        "xwise_open_fraction_max": float(np.max(xwise_open)),
    }


def write_geometry_bundle(candidate_id: str, label: str, cfg: gen.GeometryConfig) -> dict[str, object]:
    candidate_dir = GEOM_DIR / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    centers, radii, sweeps, final_overlap = gen.make_pack(cfg)
    gaps = gen.surface_gaps(centers, radii, cfg)
    grain_mask = gen.solid_mask(centers, radii, cfg)
    connected, connected_fraction = gen.left_right_connected(~grain_mask)
    clearance = cfg.particle_radius_reference + 1.0e-9
    intervals = gen.open_intervals_at_x(centers, radii, cfg, clearance=clearance)
    images = gen.periodic_images(centers, radii, cfg)
    porosity = gen.actual_porosity(radii, cfg)
    pinned = gen.pinned_mask(cfg)
    grains = [
        {
            "id": int(gid),
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "pinned": bool(pinned[gid]),
            "role": "stage1_pinned" if pinned[gid] else "stage1_relaxed_random_pack",
        }
        for gid, ((x, y), radius) in enumerate(zip(centers, radii, strict=True))
    ]
    data = {
        "stage1_candidate": {"id": candidate_id, "label": label},
        "config": asdict(cfg),
        "derived": {
            "equivalent_grain_radius": cfg.equivalent_grain_radius,
            "mean_grain_radius": float(np.mean(radii)),
            "min_grain_radius": float(np.min(radii)),
            "max_grain_radius": float(np.max(radii)),
            "radius_cv": float(np.std(radii) / np.mean(radii)),
            "actual_porosity": porosity,
            "solid_fraction": 1.0 - porosity,
            "minimum_realized_surface_gap": float(np.min(gaps)),
            "fifth_percentile_surface_gap": float(np.quantile(gaps, 0.05)),
            "median_surface_gap": float(np.median(gaps)),
            "left_right_connected": bool(connected),
            "right_boundary_connected_fraction": float(connected_fraction),
            "relaxation_sweeps": int(sweeps),
            "final_overlap": float(final_overlap),
        },
        "domain": {
            "length_x": cfg.length_x,
            "length_y": cfg.length_y,
            "periodic_x": True,
            "periodic_y": True,
        },
        "grains": grains,
        "periodic_images_for_openfoam": images,
        "injection_windows_x0": [
            {"y_min": float(s), "y_max": float(e), "width": float(e - s)} for s, e in intervals
        ],
        "stagnation_detection": {
            "status": "requires_resolved_openfoam_flow",
            "probe_shell": "grain_radius + particle_radius + h_probe",
            "candidate_condition": "zero crossing of tangential velocity on probe shell",
            "downstream_classification": "stable zero with d u_t / d theta < 0",
            "upstream_classification": "unstable zero with d u_t / d theta > 0",
        },
    }
    gen.write_json(candidate_dir / "geometry.json", data)
    gen.write_csv(candidate_dir / "geometry.csv", centers, radii)
    gen.write_openfoam_includes(candidate_dir / "geometry_openfoam.inc", images, cfg)
    gen.write_report(
        candidate_dir / "geometry_report.md",
        cfg,
        centers,
        radii,
        gaps,
        connected,
        connected_fraction,
        intervals,
        images,
        sweeps,
    )
    gen.draw_geometry(candidate_dir / "geometry.png", centers, radii, cfg, intervals)
    throat_q = np.quantile(gaps, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95])
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "label": label,
        "geometry_path": str((candidate_dir / "geometry.json").relative_to(ROOT)),
        "figure_path": str((candidate_dir / "geometry.png").relative_to(ROOT)),
        "seed": cfg.seed,
        "n_grains": cfg.n_grains,
        "target_porosity": cfg.target_porosity,
        "porosity": porosity,
        "radius_cv": float(np.std(radii) / np.mean(radii)),
        "mean_radius_um": float(np.mean(radii) * 1.0e6),
        "min_radius_um": float(np.min(radii) * 1.0e6),
        "max_radius_um": float(np.max(radii) * 1.0e6),
        "minimum_gap_um": float(np.min(gaps) * 1.0e6),
        "gap_q01_um": float(throat_q[0] * 1.0e6),
        "gap_q05_um": float(throat_q[1] * 1.0e6),
        "gap_q25_um": float(throat_q[2] * 1.0e6),
        "gap_median_um": float(throat_q[3] * 1.0e6),
        "gap_q75_um": float(throat_q[4] * 1.0e6),
        "gap_q95_um": float(throat_q[5] * 1.0e6),
        "left_right_connected": bool(connected),
        "right_boundary_connected_fraction": float(connected_fraction),
        "injection_window_count": len(intervals),
        "max_injection_window_um": float(max((e - s for s, e in intervals), default=0.0) * 1.0e6),
        "relaxation_sweeps": int(sweeps),
        "final_overlap_um": float(final_overlap * 1.0e6),
    }
    row.update(clearance_descriptors(centers, radii, cfg))
    return row


def baseline_row() -> dict[str, object]:
    if not BASELINE.exists():
        return {}
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    cfg_data = data["config"]
    cfg = gen.GeometryConfig(**cfg_data)
    centers = np.array([[grain["x"], grain["y"]] for grain in data["grains"]], dtype=float)
    radii = np.array([grain["radius"] for grain in data["grains"]], dtype=float)
    gaps = gen.surface_gaps(centers, radii, cfg)
    intervals = data.get("injection_windows_x0", [])
    row: dict[str, object] = {
        "candidate_id": "baseline_current",
        "label": "accepted geometry",
        "geometry_path": str(BASELINE.relative_to(ROOT)),
        "figure_path": str((ROOT / "outputs" / "random_porous_geometry" / "random_porous_geometry.png").relative_to(ROOT)),
        "seed": cfg.seed,
        "n_grains": len(data["grains"]),
        "target_porosity": cfg.target_porosity,
        "porosity": data["derived"]["actual_porosity"],
        "radius_cv": data["derived"]["radius_cv"],
        "mean_radius_um": data["derived"]["mean_grain_radius"] * 1.0e6,
        "min_radius_um": data["derived"]["min_grain_radius"] * 1.0e6,
        "max_radius_um": data["derived"]["max_grain_radius"] * 1.0e6,
        "minimum_gap_um": data["derived"]["minimum_realized_surface_gap"] * 1.0e6,
        "gap_q01_um": float(np.quantile(gaps, 0.01) * 1.0e6),
        "gap_q05_um": data["derived"]["fifth_percentile_surface_gap"] * 1.0e6,
        "gap_q25_um": float(np.quantile(gaps, 0.25) * 1.0e6),
        "gap_median_um": data["derived"]["median_surface_gap"] * 1.0e6,
        "gap_q75_um": float(np.quantile(gaps, 0.75) * 1.0e6),
        "gap_q95_um": float(np.quantile(gaps, 0.95) * 1.0e6),
        "left_right_connected": data["derived"]["left_right_connected"],
        "right_boundary_connected_fraction": data["derived"]["right_boundary_connected_fraction"],
        "injection_window_count": len(intervals),
        "max_injection_window_um": max((item["width"] for item in intervals), default=0.0) * 1.0e6,
        "relaxation_sweeps": data["derived"]["relaxation_sweeps"],
        "final_overlap_um": data["derived"]["final_overlap"] * 1.0e6,
    }
    row.update(clearance_descriptors(centers, radii, cfg))
    return row


def candidate_configs() -> list[tuple[str, str, gen.GeometryConfig]]:
    # Keep the same pinned void-fill grains and raster used by the accepted
    # random-packing case.  The earlier unconstrained Stage 1 geometries were
    # useful geometry contrasts, but several produced OpenFOAM fields with poor
    # periodic through-passage.  These production-like candidates vary the
    # random free grains and targeted descriptors while preserving the boundary
    # and large-void controls that made the accepted Gmsh flow robust.
    base = gen.GeometryConfig(
        raster_nx=900,
        raster_ny=600,
    )
    configs: list[tuple[str, str, gen.GeometryConfig]] = []
    for i in range(10):
        configs.append(
            (
                f"through_random_{i:02d}",
                "production-like random replicate",
                replace(base, name=f"stage1_through_random_{i:02d}", seed=20260600 + i),
            )
        )
    designed = [
        ("through_high_porosity", "production-like: higher porosity", replace(base, name="stage1_through_high_porosity", seed=20260701, target_porosity=0.52)),
        ("through_low_porosity", "production-like: lower porosity", replace(base, name="stage1_through_low_porosity", seed=20260702, target_porosity=0.40)),
        ("through_low_polydispersity", "production-like: nearly monodisperse", replace(base, name="stage1_through_low_polydispersity", seed=20260703, radius_cv=0.04)),
        ("through_high_polydispersity", "production-like: high polydispersity", replace(base, name="stage1_through_high_polydispersity", seed=20260704, radius_cv=0.30)),
        ("through_few_large_grains", "production-like: fewer larger grains", replace(base, name="stage1_through_few_large_grains", seed=20260705, n_grains=26)),
        ("through_many_small_grains", "production-like: many smaller grains", replace(base, name="stage1_through_many_small_grains", seed=20260706, n_grains=44)),
        ("through_tight_throats", "production-like: tighter throat constraint", replace(base, name="stage1_through_tight_throats", seed=20260707, minimum_surface_gap=3.0e-6, radius_cv=0.22)),
        ("through_wide_throats", "production-like: wider throat constraint", replace(base, name="stage1_through_wide_throats", seed=20260708, minimum_surface_gap=18.0e-6, radius_cv=0.12)),
        ("through_pockety_high_cv", "production-like: high-porosity pockets", replace(base, name="stage1_through_pockety_high_cv", seed=20260709, target_porosity=0.50, radius_cv=0.32, n_grains=30)),
    ]
    configs.extend(designed)
    return configs


def write_report(rows: list[dict[str, object]]) -> None:
    sorted_by_void = sorted(rows, key=lambda row: float(row.get("max_clearance_um", 0.0)), reverse=True)
    sorted_by_gap = sorted(rows, key=lambda row: float(row.get("gap_q05_um", 0.0)))
    sorted_by_channel = sorted(rows, key=lambda row: float(row.get("xwise_open_fraction_cv", 0.0)), reverse=True)
    lines = [
        "# Stage 1 Geometry Screen",
        "",
        "This screen generated candidate periodic disk packings for the random-pore extension.",
        "It is geometry-only: no particle tracking has been run on these candidates yet.",
        "",
        f"- candidates including baseline: {len(rows)}",
        f"- generated candidate directory: `{GEOM_DIR.relative_to(ROOT)}`",
        "",
        "## Largest Pore Voids",
        "",
        "| candidate | max clearance (um) | q90 clearance (um) | porosity | gap q05 (um) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted_by_void[:8]:
        lines.append(
            "| {candidate_id} | {max_clearance_um:.1f} | {clearance_q90_um:.1f} | {porosity:.3f} | {gap_q05_um:.1f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Tightest Throat Candidates",
            "",
            "| candidate | gap q05 (um) | min gap (um) | porosity | radius CV |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted_by_gap[:8]:
        lines.append(
            "| {candidate_id} | {gap_q05_um:.1f} | {minimum_gap_um:.1f} | {porosity:.3f} | {radius_cv:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Most Channelized Geometry Proxies",
            "",
            "| candidate | x-open CV | max clearance (um) | q90 clearance (um) | max inlet window (um) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted_by_channel[:8]:
        lines.append(
            "| {candidate_id} | {xwise_open_fraction_cv:.3f} | {max_clearance_um:.1f} | {clearance_q90_um:.1f} | {max_injection_window_um:.1f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
        "## Next Step",
        "",
        "Solve OpenFOAM flow for a small, deliberately diverse production-like subset before tracking particles:",
        "baseline/current, one random replicate near the ensemble median, one high-clearance/channelized case, one tight-throat case, and one high-polydispersity or pockety case.",
        "Only flows that pass the passive-throughflow gate should be promoted to DLVO particle statistics.",
        "",
    ]
    )
    (OUT / "stage1_geometry_screen_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    base = baseline_row()
    if base:
        rows.append(base)
    for candidate_id, label, cfg in candidate_configs():
        try:
            print(f"generating {candidate_id}", flush=True)
            rows.append(write_geometry_bundle(candidate_id, label, cfg))
        except Exception as exc:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "label": label,
                    "seed": cfg.seed,
                    "n_grains": cfg.n_grains,
                    "target_porosity": cfg.target_porosity,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"failed {candidate_id}: {exc}", flush=True)
    write_rows(OUT / "stage1_geometry_metrics.csv", rows)
    write_report([row for row in rows if row.get("status") != "failed"])
    print(OUT / "stage1_geometry_metrics.csv")
    print(OUT / "stage1_geometry_screen_report.md")


if __name__ == "__main__":
    main()
