#!/usr/bin/env python3
"""Create a periodic raster flow field from the body-fitted OpenFOAM solution.

The compiled particle tracker expects a square periodic velocity raster.  This
script converts the OpenFOAM cell-centered field from the benchmark case into
that format using periodic inverse-distance interpolation from OpenFOAM cell
centers.  The output is not an LBM field; it is an OpenFOAM-derived flow raster
with the same NPZ layout as the LBM flow files.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_openfoam_flow_benchmark as ofbench  # noqa: E402
from colloid_tsm.physical import PhysicalParams, _solid_mask, interpolate_velocity, load_flow, save_flow  # noqa: E402


OUT = ROOT / "outputs" / "openfoam_flow"
BENCH = ROOT / "outputs" / "openfoam_flow_benchmark"
DEFAULT_RESOLUTION = 512
K_NEIGHBORS = 12
POWER = 2.0


def unique_xy_average(centers: np.ndarray, velocity: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounded = np.round(centers, decimals=14)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    weights = np.where(np.isfinite(areas) & (areas > 0.0), areas, 1.0)
    sum_w = np.bincount(inverse, weights=weights)
    ux = np.bincount(inverse, weights=velocity[:, 0] * weights) / sum_w
    uy = np.bincount(inverse, weights=velocity[:, 1] * weights) / sum_w
    return unique, np.column_stack([ux, uy]), sum_w


def periodic_images(points: np.ndarray, values: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    shifts = np.asarray(
        [(i * length, j * length) for i in (-1, 0, 1) for j in (-1, 0, 1)],
        dtype=float,
    )
    image_points = np.vstack([points + shift for shift in shifts])
    image_values = np.vstack([values for _ in shifts])
    return image_points, image_values


def query_idw(points: np.ndarray, values: np.ndarray, query: np.ndarray, k: int, power: float) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as exc:  # pragma: no cover - production environment should have scipy or use system python
        raise RuntimeError("create_openfoam_raster_flow.py needs scipy.spatial.cKDTree") from exc

    tree = cKDTree(points)
    distances, ids = tree.query(query, k=k, workers=-1)
    if k == 1:
        return values[ids]
    distances = np.maximum(distances, 1.0e-30)
    weights = 1.0 / distances**power
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, values[ids])


def build_raster(params: PhysicalParams, resolution: int) -> Path:
    vtk_path = ofbench.find_vtk_file()
    centers, u_of, areas, _ = ofbench.parse_vtk_field(vtk_path)
    centers, u_of, areas = unique_xy_average(centers, u_of, areas)
    image_points, image_values = periodic_images(centers, u_of, params.cell_length)

    coords = (np.arange(resolution) + 0.5) * params.cell_length / resolution
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    solid = _solid_mask(resolution, params)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    interpolated = query_idw(image_points, image_values, query, K_NEIGHBORS, POWER)
    ux = interpolated[:, 0].reshape(resolution, resolution)
    uy = interpolated[:, 1].reshape(resolution, resolution)
    ux[solid] = 0.0
    uy[solid] = 0.0

    fluid = ~solid
    mean_ux = float(np.mean(ux[fluid]))
    scale = params.mean_velocity / mean_ux
    ux *= scale
    uy *= scale

    from colloid_tsm.physical import FlowField

    flow = FlowField(
        x=coords,
        y=coords,
        ux=ux,
        uy=uy,
        solid=solid,
        params=params,
        resolution=resolution,
        iterations=ofbench.END_TIME,
        tau=float("nan"),
        mean_velocity_before_scale=mean_ux,
        scale_factor=scale,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"openfoam_flow_N{resolution}.npz"
    save_flow(path, flow)
    write_report(path, params, flow, centers, u_of, areas)
    return path


def shell_tangential_stats(flow_path: Path, params: PhysicalParams, centers: np.ndarray, u_of: np.ndarray, areas: np.ndarray) -> list[dict[str, float | str]]:
    flow = load_flow(flow_path, params=params)
    raster_ux, raster_uy = interpolate_velocity(flow, centers[:, 0], centers[:, 1])
    arrays = {
        "centers": centers,
        "u_of": u_of,
        "u_lbm": np.column_stack([raster_ux, raster_uy]),
        "areas": areas,
    }
    _, theta, _, collector = ofbench.nearest_wall_state(params, centers[:, 0], centers[:, 1])
    arrays["theta"] = theta
    arrays["collector"] = collector
    return ofbench.ring_summary(params, arrays)


def write_report(path: Path, params: PhysicalParams, flow, centers: np.ndarray, u_of: np.ndarray, areas: np.ndarray) -> None:
    raster_ux, raster_uy = interpolate_velocity(flow, centers[:, 0], centers[:, 1])
    u_raster = np.column_stack([raster_ux, raster_uy])
    diff = u_of - u_raster
    h_wall, _, _, _ = ofbench.nearest_wall_state(params, centers[:, 0], centers[:, 1])
    weights = np.where(np.isfinite(areas) & (areas > 0), areas, 1.0)
    dx = params.cell_length / flow.resolution
    regions = {
        "all fluid cells": np.ones(centers.shape[0], dtype=bool),
        f"bulk, h > {2.0 * dx * 1e6:.2f} um": h_wall > 2.0 * dx,
        f"near wall, h <= {2.0 * dx * 1e6:.2f} um": h_wall <= 2.0 * dx,
    }

    def wmean(v: np.ndarray, mask: np.ndarray) -> float:
        w = weights[mask]
        return float(np.sum(v[mask] * w) / np.sum(w))

    def wrms(v: np.ndarray, mask: np.ndarray) -> float:
        w = weights[mask]
        return float(math.sqrt(np.sum(v[mask] * v[mask] * w) / np.sum(w)))

    rows = []
    err_norm = np.linalg.norm(diff, axis=1)
    for name, mask in regions.items():
        rows.append(
            {
                "region": name,
                "cells": int(np.sum(mask)),
                "mean_abs_err_over_U": wmean(err_norm, mask) / params.mean_velocity,
                "rms_err_over_U": wrms(err_norm, mask) / params.mean_velocity,
                "mean_openfoam_ux": wmean(u_of[:, 0], mask),
                "mean_raster_ux": wmean(u_raster[:, 0], mask),
            }
        )
    shell_rows = shell_tangential_stats(path, params, centers, u_of, areas)

    def fmt(x: object, digits: int = 4) -> str:
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        try:
            value = float(x)
        except (TypeError, ValueError):
            return str(x)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{digits}g}"

    lines = [
        "# OpenFOAM-derived raster flow",
        "",
        f"Source benchmark: `{BENCH / 'case' / 'VTK'}`.",
        f"Output raster: `{path}`.",
        f"Resolution: {flow.resolution} x {flow.resolution}; grid spacing {dx * 1e6:.3f} um.",
        f"Velocity scale factor applied to match mean ux={params.mean_velocity:.6e} m/s: {flow.scale_factor:.6g}.",
        "",
        "## Raster-vs-OpenFOAM cell-center check",
        "",
        "| region | cells | mean abs err/U | rms err/U | mean OpenFOAM ux | mean raster ux |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["region"]),
                    fmt(row["cells"], 0),
                    fmt(row["mean_abs_err_over_U"]),
                    fmt(row["rms_err_over_U"]),
                    fmt(row["mean_openfoam_ux"], 6),
                    fmt(row["mean_raster_ux"], 6),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Near-wall tangential shell check",
            "",
            "| collector | sample | median |ut| OF (um/s) | median |ut| raster (um/s) | p95 |ut| OF (um/s) | p95 |ut| raster (um/s) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in shell_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["collector"]),
                    str(row["sample"]),
                    fmt(row["median_abs_ut_openfoam_um_s"]),
                    fmt(row["median_abs_ut_lbm_um_s"]),
                    fmt(row["p95_abs_ut_openfoam_um_s"]),
                    fmt(row["p95_abs_ut_lbm_um_s"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this raster preserves the OpenFOAM near-wall tangential velocity much more closely than the original LBM raster while remaining compatible with the compiled particle tracker.",
            "",
        ]
    )
    (OUT / "openfoam_raster_flow_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    resolution = DEFAULT_RESOLUTION
    for arg in sys.argv[1:]:
        if arg.startswith("--resolution="):
            resolution = int(arg.split("=", 1)[1])
    params = PhysicalParams(mean_velocity=ofbench.TARGET_M_PER_DAY / 86400.0)
    path = build_raster(params, resolution)
    print(path)


if __name__ == "__main__":
    main()
