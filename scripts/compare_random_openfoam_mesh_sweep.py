#!/usr/bin/env python3
"""Compare random porous OpenFOAM flow fields across mesh refinements."""

from __future__ import annotations

import csv
import importlib.util
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOW_SCRIPT = ROOT / "scripts" / "run_random_openfoam_flow.py"
GEOMETRY_PATH = ROOT / "outputs" / "random_porous_geometry" / "random_porous_geometry.json"
OUT = ROOT / "outputs" / "random_openfoam_mesh_sweep"

spec = importlib.util.spec_from_file_location("random_openfoam_flow", FLOW_SCRIPT)
flow = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(flow)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = [
    ("wall3_far8", ROOT / "outputs" / "random_openfoam_flow", 3.0, 8.0),
    ("wall2_far5", ROOT / "outputs" / "random_openfoam_flow_wall2_far5", 2.0, 5.0),
    ("wall1p25_far3p5", ROOT / "outputs" / "random_openfoam_flow_wall1p25_far3p5", 1.25, 3.5),
]


def latest_vtu(case_dir: Path) -> Path | None:
    vtk_dir = case_dir / "case" / "VTK"
    candidates = sorted(vtk_dir.glob("*/*.vtu"))
    internal = [path for path in candidates if path.name == "internal.vtu"]
    return internal[-1] if internal else (candidates[-1] if candidates else None)


def parse_report_value(report: Path, label: str) -> float:
    if not report.exists():
        return math.nan
    text = report.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"- {re.escape(label)}: ([0-9.eE+-]+)", text)
    return float(match.group(1)) if match else math.nan


def nearest_surface_vectors(centers_xy: np.ndarray, geometry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    grains = geometry["grains"]
    clearance = np.full(centers_xy.shape[0], np.inf)
    dx_best = np.zeros(centers_xy.shape[0])
    dy_best = np.zeros(centers_xy.shape[0])
    radius_best = np.ones(centers_xy.shape[0])
    for grain in grains:
        radius = float(grain["radius"])
        dx = flow.periodic_delta(centers_xy[:, 0] - float(grain["x"]), lx)
        dy = flow.periodic_delta(centers_xy[:, 1] - float(grain["y"]), ly)
        distance = np.sqrt(dx * dx + dy * dy)
        this_clearance = distance - radius
        take = this_clearance < clearance
        clearance[take] = this_clearance[take]
        dx_best[take] = dx[take]
        dy_best[take] = dy[take]
        radius_best[take] = radius
    distance_best = np.sqrt(dx_best * dx_best + dy_best * dy_best)
    distance_best = np.maximum(distance_best, 1.0e-30)
    return clearance, dx_best / distance_best, dy_best / distance_best


def summarize_case(label: str, case_dir: Path, wall_um: float, far_um: float, geometry: dict) -> dict | None:
    vtu = latest_vtu(case_dir)
    if vtu is None:
        return None
    centers, velocity, areas = flow.parse_xml_vtu(vtu)
    speed = np.linalg.norm(velocity, axis=1)
    good = np.isfinite(speed) & np.isfinite(areas) & (areas > 0)
    weights = areas[good]
    clearance, ex, ey = nearest_surface_vectors(centers, geometry)
    tx = -ey
    ty = ex
    ut = velocity[:, 0] * tx + velocity[:, 1] * ty
    ur = velocity[:, 0] * ex + velocity[:, 1] * ey

    def weighted_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is None:
            mask = good
        else:
            mask = mask & good
        if not np.any(mask):
            return math.nan
        return float(np.sum(values[mask] * areas[mask]) / np.sum(areas[mask]))

    def quantile(values: np.ndarray, mask: np.ndarray | None = None, q: float = 0.5) -> float:
        if mask is None:
            mask = good
        else:
            mask = mask & good
        if not np.any(mask):
            return math.nan
        return float(np.quantile(values[mask], q))

    near5 = (clearance > 0) & (clearance < 5e-6)
    near10 = (clearance > 0) & (clearance < 10e-6)
    throat = speed > np.quantile(speed[good], 0.99)

    report = case_dir / "random_openfoam_flow_report.md"
    return {
        "label": label,
        "case_dir": str(case_dir.relative_to(ROOT)),
        "wall_um": wall_um,
        "far_um": far_um,
        "cells": int(speed.size),
        "mean_ux": weighted_mean(velocity[:, 0]),
        "mean_uy": weighted_mean(velocity[:, 1]),
        "mean_speed": weighted_mean(speed),
        "max_speed": float(np.nanmax(speed)),
        "p99_speed": quantile(speed, q=0.99),
        "p999_speed": quantile(speed, q=0.999),
        "near5_count": int(np.sum(near5)),
        "near10_count": int(np.sum(near10)),
        "near10_mean_abs_ut": weighted_mean(np.abs(ut), near10),
        "near10_p90_abs_ut": quantile(np.abs(ut), near10, 0.90),
        "near10_mean_abs_ur": weighted_mean(np.abs(ur), near10),
        "throat_mean_speed": weighted_mean(speed, throat),
        "reported_mean_ux": parse_report_value(report, "Area-weighted mean ux"),
    }


def write_outputs(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "mesh_sweep_summary.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Random OpenFOAM mesh sweep",
        "",
        "| case | cells | mean ux / target | max speed / target | p99 speed / target | near-wall p90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {label} | {cells:,} | {mean_ratio:.4f} | {max_ratio:.2f} | {p99_ratio:.2f} | {near_ratio:.2f} |".format(
                label=row["label"],
                cells=row["cells"],
                mean_ratio=row["mean_ux"] / flow.UBAR,
                max_ratio=row["max_speed"] / flow.UBAR,
                p99_ratio=row["p99_speed"] / flow.UBAR,
                near_ratio=row["near10_p90_abs_ut"] / flow.UBAR,
            )
        )
    md_lines.extend(["", f"CSV: `{csv_path.relative_to(ROOT)}`"])
    (OUT / "mesh_sweep_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), dpi=180)

    axes[0].plot(x, [row["max_speed"] / flow.UBAR for row in rows], marker="o", label="max")
    axes[0].plot(x, [row["p99_speed"] / flow.UBAR for row in rows], marker="o", label="p99")
    axes[0].plot(x, [row["mean_speed"] / flow.UBAR for row in rows], marker="o", label="mean speed")
    axes[0].set_ylabel("speed / target mean ux")
    axes[0].set_title("Speed convergence")
    axes[0].legend(frameon=False)

    axes[1].plot(x, [row["near10_p90_abs_ut"] / flow.UBAR for row in rows], marker="o", label="p90 |ut|")
    axes[1].plot(x, [row["near10_mean_abs_ut"] / flow.UBAR for row in rows], marker="o", label="mean |ut|")
    axes[1].plot(x, [row["near10_mean_abs_ur"] / flow.UBAR for row in rows], marker="o", label="mean |ur|")
    axes[1].set_ylabel("near-wall speed / target mean ux")
    axes[1].set_title("Within 10 um of grains")
    axes[1].legend(frameon=False)

    axes[2].plot(x, [row["cells"] / 1000 for row in rows], marker="o")
    axes[2].set_ylabel("cells (thousands)")
    axes[2].set_title("Cost")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "mesh_sweep_convergence.png")
    plt.close(fig)


def main() -> None:
    geometry = flow.prepare_openfoam_geometry(__import__("json").loads(GEOMETRY_PATH.read_text(encoding="utf-8")))
    rows = []
    for label, case_dir, wall_um, far_um in CASES:
        row = summarize_case(label, case_dir, wall_um, far_um, geometry)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No mesh-sweep VTU files found.")
    write_outputs(rows)
    print(OUT / "mesh_sweep_summary.md")
    print(OUT / "mesh_sweep_convergence.png")


if __name__ == "__main__":
    main()
