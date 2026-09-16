#!/usr/bin/env python3
"""Create a production-data visual of random-packing release focusing."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)


DEFAULT_CANDIDATE = "through_many_small_grains"
BASE = ROOT / "outputs" / "stage3_particle_production" / DEFAULT_CANDIDATE
GEOMETRY_PATH = ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / DEFAULT_CANDIDATE / "geometry.json"
FLOW_CASE = ROOT / "outputs" / "stage1_openfoam_flow" / f"{DEFAULT_CANDIDATE}_gmsh_screen"
FIGURE_PATH = ROOT / "outputs" / "figures" / "random_geometry_mechanism_streamlines.png"

PANELS = (
    ("neutral_resolved", "No DLVO", "#2563eb"),
    ("favorable_50mM_z70", "Favorable 50 mM", "#dc2626"),
    ("unfavorable_50mM_z70", "Unfavorable 50 mM", "#7c3aed"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_interpolator(geometry: dict, flow_case: Path, target_m_per_day: float, rescale: bool) -> rpt.PeriodicIDWFlow:
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, areas = rpt.flow_io.parse_xml_vtu(vtu)
    if rescale:
        velocity_xy, _, _ = rpt.rescale_velocity_to_target(velocity_xy, areas, target_m_per_day)
    return rpt.PeriodicIDWFlow(
        centers_xy,
        velocity_xy,
        lx=float(geometry["domain"]["length_x"]),
        ly=float(geometry["domain"]["length_y"]),
        k=8,
    )


def flow_grid(geometry: dict, interpolator: rpt.PeriodicIDWFlow, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    xs = (np.arange(nx) + 0.5) * lx / nx
    ys = (np.arange(ny) + 0.5) * ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    ux, uy = interpolator.velocity_at(xx.ravel(), yy.ravel())
    ux = ux.reshape(ny, nx)
    uy = uy.reshape(ny, nx)
    inside = np.zeros_like(ux, dtype=bool)
    for grain in geometry["grains"]:
        dx = rpt.periodic_delta(xx - float(grain["x"]), lx)
        dy = rpt.periodic_delta(yy - float(grain["y"]), ly)
        inside |= np.hypot(dx, dy) <= float(grain["radius"])
    return xs, ys, np.ma.masked_where(inside, ux), np.ma.masked_where(inside, uy), inside


def release_points(geometry: dict, release_csv: Path) -> dict[str, np.ndarray]:
    params = rpt.RandomTrackingParams()
    grains = geometry["grains"]
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    out: dict[str, list[tuple[float, float, float]]] = {}
    for row in read_csv(release_csv):
        profile = row["profile"]
        gid = int(row["grain_exit"])
        if gid < 0 or gid >= len(grains):
            continue
        theta = float(row["theta_exit_rad"])
        grain = grains[gid]
        shell = float(grain["radius"]) + params.particle_radius + params.near_surface
        x = (float(grain["x"]) + shell * np.cos(theta)) % lx
        y = (float(grain["y"]) + shell * np.sin(theta)) % ly
        out.setdefault(profile, []).append((x * 1.0e3, y * 1.0e3, float(row["abs_delta_from_nearest_rear_deg"])))
    return {key: np.asarray(values, dtype=float) for key, values in out.items()}


def attachment_points(geometry: dict, payload_path: Path, profile: str, max_points: int) -> np.ndarray:
    data = np.load(payload_path)
    key = f"{profile}_attached"
    if key not in data.files:
        return np.empty((0, 2))
    attached = data[key].astype(bool)
    if not np.any(attached):
        return np.empty((0, 2))
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    points = np.column_stack(
        (
            np.mod(data[f"{profile}_x_final"][attached], lx) * 1.0e3,
            np.mod(data[f"{profile}_y_final"][attached], ly) * 1.0e3,
        )
    )
    if points.shape[0] > max_points:
        rng = np.random.default_rng(20260510)
        points = points[rng.choice(points.shape[0], size=max_points, replace=False)]
    return points


def rear_points(stagnation_csv: Path) -> np.ndarray:
    points = []
    for row in read_csv(stagnation_csv):
        if row["stagnation_type"] == "rear":
            points.append((float(row["x_m"]) * 1.0e3, float(row["y_m"]) * 1.0e3))
    return np.asarray(points, dtype=float)


def summary_metrics(summary_csv: Path) -> dict[str, dict[str, float]]:
    out = {}
    for row in read_csv(summary_csv):
        out[row["profile"]] = {
            "released": float(row["released_from_near_zone"]),
            "intercepted": float(row["intercepted"]),
            "median": float(row["median_abs_release_angle_from_rear_deg"]),
            "f30": float(row["fraction_released_within_30deg"]),
        }
    return out


def draw_grains(ax, geometry: dict) -> None:
    for grain in geometry["grains"]:
        ax.add_patch(
            Circle(
                (float(grain["x"]) * 1.0e3, float(grain["y"]) * 1.0e3),
                float(grain["radius"]) * 1.0e3,
                facecolor="#020617",
                edgecolor="#020617",
                lw=0.45,
                zorder=8,
            )
        )


def plot(args: argparse.Namespace) -> Path:
    geometry = rpt.load_geometry(args.geometry_path, flow_case=args.flow_case, apply_flow_shift=not args.no_flow_origin_shift)
    interpolator = load_interpolator(geometry, args.flow_case, args.target_mean_velocity_m_per_day, not args.no_rescale_flow)
    xs, ys, ux, uy, inside = flow_grid(geometry, interpolator, args.flow_nx, args.flow_ny)
    speed = np.sqrt(np.asarray(ux.filled(np.nan)) ** 2 + np.asarray(uy.filled(np.nan)) ** 2)
    speed[inside] = np.nan
    releases = release_points(geometry, args.release_events)
    attachments = attachment_points(geometry, args.production_payload, "favorable_50mM_z70", args.max_points)
    rear = rear_points(args.stagnation_points)
    metrics = summary_metrics(args.summary_csv)
    lx_mm = float(geometry["domain"]["length_x"]) * 1.0e3
    ly_mm = float(geometry["domain"]["length_y"]) * 1.0e3

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=260, constrained_layout=True)
    norm = Normalize(vmin=0.0, vmax=np.nanpercentile(speed, 97.0) * 1.0e3)
    for ax, (profile, title, color) in zip(axes, PANELS, strict=True):
        im = ax.imshow(
            speed * 1.0e3,
            origin="lower",
            extent=(0.0, lx_mm, 0.0, ly_mm),
            cmap="Greys",
            norm=norm,
            alpha=0.68,
            zorder=0,
        )
        ax.streamplot(
            xs * 1.0e3,
            ys * 1.0e3,
            ux,
            uy,
            density=1.25,
            color=color,
            linewidth=0.58,
            arrowsize=0.62,
            minlength=0.08,
            zorder=2,
        )
        if rear.size:
            ax.scatter(rear[:, 0], rear[:, 1], s=52, facecolors="none", edgecolors="#ef4444", linewidths=1.25, zorder=10)
        pts = releases.get(profile, np.empty((0, 3)))
        if pts.size:
            rng = np.random.default_rng(731 + len(profile))
            if pts.shape[0] > args.max_points:
                pts = pts[rng.choice(pts.shape[0], args.max_points, replace=False)]
            focused = pts[:, 2] <= 30.0
            ax.scatter(pts[~focused, 0], pts[~focused, 1], s=7, color=color, alpha=0.23, linewidth=0, zorder=5)
            ax.scatter(pts[focused, 0], pts[focused, 1], s=15, color="#f97316", alpha=0.74, linewidth=0, zorder=6)
        if profile == "favorable_50mM_z70" and attachments.size:
            ax.scatter(attachments[:, 0], attachments[:, 1], s=17, marker="x", color="#dc2626", linewidth=0.85, alpha=0.85, zorder=11)
        draw_grains(ax, geometry)
        m = metrics[profile]
        ax.text(
            0.03,
            0.96,
            f"{title}\n$F_{{30}}$={m['f30']:.3f}; median={m['median']:.1f} deg",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.8,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.86),
            zorder=12,
        )
        ax.set_xlim(0.0, lx_mm)
        ax.set_ylim(0.0, ly_mm)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#94a3b8")
    cbar = fig.colorbar(im, ax=axes, shrink=0.76, pad=0.014)
    cbar.set_label("speed (mm/s)")
    for label, ax in zip("abc", axes, strict=True):
        ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.suptitle("Resolved-flow streamlines and grain-local release outcomes in a random periodic packing", fontsize=13.2, fontweight="bold")
    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_path, bbox_inches="tight")
    plt.close(fig)
    return args.figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-path", type=Path, default=GEOMETRY_PATH)
    parser.add_argument("--flow-case", type=Path, default=FLOW_CASE)
    parser.add_argument("--production-payload", type=Path, default=BASE / "random_condition_screen_payload.npz")
    parser.add_argument("--release-events", type=Path, default=BASE / "grain_focusing" / "grain_local_release_events.csv")
    parser.add_argument("--stagnation-points", type=Path, default=BASE / "grain_focusing" / "grain_stagnation_points.csv")
    parser.add_argument("--summary-csv", type=Path, default=BASE / "grain_focusing" / "grain_local_focusing.csv")
    parser.add_argument("--figure-path", type=Path, default=FIGURE_PATH)
    parser.add_argument("--flow-nx", type=int, default=150)
    parser.add_argument("--flow-ny", type=int, default=100)
    parser.add_argument("--max-points", type=int, default=700)
    parser.add_argument("--target-mean-velocity-m-per-day", type=float, default=rpt.TARGET_M_PER_DAY)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--no-rescale-flow", action="store_true")
    args = parser.parse_args()
    for key in ("geometry_path", "flow_case", "production_payload", "release_events", "stagnation_points", "summary_csv", "figure_path"):
        path = getattr(args, key)
        if not path.is_absolute():
            setattr(args, key, ROOT / path)
    return args


def main() -> None:
    path = plot(parse_args())
    print(path)


if __name__ == "__main__":
    main()
