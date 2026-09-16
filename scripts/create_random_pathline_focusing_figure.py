#!/usr/bin/env python3
"""Create a pathline-focused visual figure for grain-local focusing."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)


OUT_DIR = ROOT / "outputs" / "figures"
FIGURE_PATH = OUT_DIR / "random_geometry_pathline_focusing.png"
PATH_PAYLOAD = ROOT / "outputs" / "stage3_particle_production" / "through_many_small_grains" / "pathline_visual_payload.npz"
RELEASE_EVENTS = ROOT / "outputs" / "stage3_particle_production" / "through_many_small_grains" / "grain_focusing" / "grain_local_release_events.csv"
STAGNATION_POINTS = ROOT / "outputs" / "stage3_particle_production" / "through_many_small_grains" / "grain_focusing" / "grain_stagnation_points.csv"
PRODUCTION_PAYLOAD = ROOT / "outputs" / "stage3_particle_production" / "through_many_small_grains" / "random_condition_screen_payload.npz"
SUMMARY_CSV = ROOT / "outputs" / "stage3_particle_production" / "through_many_small_grains" / "grain_focusing" / "grain_local_focusing.csv"
GEOMETRY_PATH = ROOT / "outputs" / "stage1_geometry_screen" / "geometries" / "through_many_small_grains" / "geometry.json"
FLOW_CASE = ROOT / "outputs" / "stage1_openfoam_flow" / "through_many_small_grains_gmsh_screen"


PANELS = [
    {
        "profile": "neutral_resolved",
        "condition": "no_dlvo",
        "title": "No DLVO",
        "subtitle": "mobile paths scatter broadly",
        "color": "#2563eb",
        "updates": {},
    },
    {
        "profile": "favorable_50mM_z70",
        "condition": "favorable",
        "title": "Favorable 50 mM",
        "subtitle": "interceptions terminate as attachment",
        "color": "#dc2626",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_favorable": 70.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z70",
        "condition": "unfavorable",
        "title": "Unfavorable 50 mM",
        "subtitle": "mobile release focuses downstream",
        "color": "#7c3aed",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
]


def load_geometry(geometry_path: Path, flow_case: Path, apply_flow_shift: bool) -> dict:
    return rpt.load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=apply_flow_shift)


def load_interpolator(geometry: dict, flow_case: Path, target_m_per_day: float, rescale_flow: bool):
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, areas = rpt.flow_io.parse_xml_vtu(vtu)
    if rescale_flow:
        velocity_xy, _, _ = rpt.rescale_velocity_to_target(velocity_xy, areas, target_m_per_day)
    return rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8)


def selected_initial_positions(production_payload: Path, release_events: Path, profile: str, count: int = 24) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(production_payload)
    events: list[tuple[int, float]] = []
    with release_events.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["profile"] != profile:
                continue
            abs_delta = float(row["abs_delta_from_nearest_rear_deg"])
            if abs_delta <= 15.0:
                events.append((int(row["particle_id"]), abs_delta))
    events.sort(key=lambda item: item[1])
    particle_ids = [pid for pid, _delta in events[:count]]
    y0 = data[f"{profile}_y0"][particle_ids]
    x0 = np.full(y0.size, 1.0e-6, dtype=float)
    return x0, y0


def simulate_pathlines(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray | list[np.ndarray]]]:
    if args.path_payload.exists() and not args.force:
        data = np.load(args.path_payload, allow_pickle=True)
        out: dict[str, dict[str, np.ndarray | list[np.ndarray]]] = {}
        for panel in PANELS:
            profile = panel["profile"]
            out[profile] = {
                "path_x": list(data[f"{profile}_path_x"]),
                "path_y": list(data[f"{profile}_path_y"]),
                "status": data[f"{profile}_status"],
            }
        return out

    geometry = load_geometry(args.geometry_path, args.flow_case, not args.no_flow_origin_shift)
    interpolator = load_interpolator(geometry, args.flow_case, args.target_mean_velocity_m_per_day, not args.no_rescale_flow)
    x0, y0 = selected_initial_positions(args.production_payload, args.release_events, args.seed_profile, args.path_count)
    base = rpt.RandomTrackingParams(
        max_time=args.path_max_time,
        dt=args.path_dt,
        adaptive_near_wall=True,
        segment_collision=True,
        max_substeps=50,
    )
    results: dict[str, dict[str, np.ndarray | list[np.ndarray]]] = {}
    payload: dict[str, np.ndarray] = {}
    args.path_payload.parent.mkdir(parents=True, exist_ok=True)
    for i, panel in enumerate(PANELS):
        params = replace(base, **panel["updates"])
        print(f"recording pathlines for {panel['profile']} ({y0.size} particles)", flush=True)
        result = rpt.simulate_particles(
            str(panel["condition"]),
            interpolator,
            geometry,
            params,
            n_particles=y0.size,
            seed=880000 + 1000 * i,
            initial_x=x0.copy(),
            initial_y=y0.copy(),
            record_count=y0.size,
            record_stride=args.record_stride,
        )
        path_x = np.array(result.path_x, dtype=object)
        path_y = np.array(result.path_y, dtype=object)
        profile = str(panel["profile"])
        results[profile] = {"path_x": list(path_x), "path_y": list(path_y), "status": result.path_status}
        payload[f"{profile}_path_x"] = path_x
        payload[f"{profile}_path_y"] = path_y
        payload[f"{profile}_status"] = result.path_status
        print(f"finished pathlines for {panel['profile']}", flush=True)
    np.savez_compressed(args.path_payload, **payload)
    return results


def split_periodic_path(x: np.ndarray, y: np.ndarray, lx: float, ly: float) -> list[np.ndarray]:
    x = np.minimum(np.asarray(x, dtype=float), lx)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return []
    jumps = np.flatnonzero(np.abs(np.diff(y)) > 0.45 * ly)
    starts = np.r_[0, jumps + 1]
    ends = np.r_[jumps + 1, x.size]
    segments: list[np.ndarray] = []
    for start, end in zip(starts, ends):
        if end - start >= 2:
            segments.append(np.column_stack((x[start:end] * 1.0e3, y[start:end] * 1.0e3)))
    return segments


def load_release_points(geometry: dict, release_events: Path, max_points: int = 420) -> dict[str, np.ndarray]:
    grains = geometry["grains"]
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    rng = np.random.default_rng(2219)
    by_profile: dict[str, list[tuple[float, float, float]]] = {}
    with release_events.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profile = row["profile"]
            if profile not in {panel["profile"] for panel in PANELS}:
                continue
            gid = int(row["grain_exit"])
            theta = float(row["theta_exit_rad"])
            grain = grains[gid]
            shell_r = float(grain["radius"]) + 0.55e-6 + 6.0e-6
            x = (float(grain["x"]) + shell_r * np.cos(theta)) % lx
            y = (float(grain["y"]) + shell_r * np.sin(theta)) % ly
            delta = float(row["abs_delta_from_nearest_rear_deg"])
            by_profile.setdefault(profile, []).append((x * 1.0e3, y * 1.0e3, delta))
    out: dict[str, np.ndarray] = {}
    for profile, values in by_profile.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] > max_points:
            take = rng.choice(arr.shape[0], size=max_points, replace=False)
            arr = arr[take]
        out[profile] = arr
    return out


def load_rear_points(stagnation_points: Path) -> np.ndarray:
    points = []
    with stagnation_points.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stagnation_type"] != "rear":
                continue
            points.append((float(row["x_m"]) * 1.0e3, float(row["y_m"]) * 1.0e3))
    return np.asarray(points, dtype=float)


def load_attachment_points(production_payload: Path, geometry: dict, max_points: int = 420) -> dict[str, np.ndarray]:
    data = np.load(production_payload)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    rng = np.random.default_rng(7717)
    out: dict[str, np.ndarray] = {}
    for panel in PANELS:
        profile = str(panel["profile"])
        attached_key = f"{profile}_attached"
        if attached_key not in data.files:
            continue
        attached = data[attached_key].astype(bool)
        if not np.any(attached):
            continue
        x = np.mod(data[f"{profile}_x_final"][attached], lx) * 1.0e3
        y = np.mod(data[f"{profile}_y_final"][attached], ly) * 1.0e3
        arr = np.column_stack((x, y))
        if arr.shape[0] > max_points:
            take = rng.choice(arr.shape[0], size=max_points, replace=False)
            arr = arr[take]
        out[profile] = arr
    return out


def load_summary_metrics(summary_csv: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "median_abs_release_angle_from_rear_deg_mean" in row:
                median = float(row["median_abs_release_angle_from_rear_deg_mean"])
                f30 = float(row["fraction_released_within_30deg_mean"])
            else:
                median = float(row["median_abs_release_angle_from_rear_deg"])
                f30 = float(row["fraction_released_within_30deg"])
            out[row["profile"]] = (median, f30)
    return out


def draw_grains(ax, geometry: dict) -> None:
    for grain in geometry["grains"]:
        circle = Circle(
            (float(grain["x"]) * 1.0e3, float(grain["y"]) * 1.0e3),
            float(grain["radius"]) * 1.0e3,
            facecolor="#111827",
            edgecolor="#111827",
            linewidth=0.4,
            zorder=3,
        )
        ax.add_patch(circle)


def draw_pathlines(ax, profile: str, paths: dict, lx: float, ly: float, panel_color: str) -> None:
    segments = []
    colors = []
    linewidths = []
    statuses = paths[profile]["status"]
    for x, y, status in zip(paths[profile]["path_x"], paths[profile]["path_y"], statuses):
        path_segments = split_periodic_path(np.asarray(x), np.asarray(y), lx, ly)
        if not path_segments:
            continue
        if profile == "favorable_50mM_z70" and int(status) == 1:
            color = "#dc2626"
            lw = 1.35
        elif profile == "favorable_50mM_z70":
            color = "#64748b"
            lw = 0.75
        elif profile == "unfavorable_50mM_z70":
            color = panel_color
            lw = 1.15
        else:
            color = panel_color
            lw = 0.88
        for segment in path_segments:
            segments.append(segment)
            colors.append(color)
            linewidths.append(lw)
    if segments:
        lc = LineCollection(segments, colors=colors, linewidths=linewidths, alpha=0.62, zorder=2)
        ax.add_collection(lc)


def draw_matched_endpoints(ax, profile: str, paths: dict, lx: float, ly: float) -> None:
    status = paths[profile]["status"]
    end_points = []
    end_status = []
    for x, y, s in zip(paths[profile]["path_x"], paths[profile]["path_y"], status):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size == 0:
            continue
        end_points.append((min(float(x_arr[-1]), lx) * 1.0e3, (float(y_arr[-1]) % ly) * 1.0e3))
        end_status.append(int(s))
    if not end_points:
        return
    arr = np.asarray(end_points, dtype=float)
    end_status = np.asarray(end_status, dtype=int)
    attached = end_status == 1
    mobile = end_status == 0
    censored = end_status == 2
    if np.any(mobile):
        ax.scatter(arr[mobile, 0], arr[mobile, 1], s=18, marker=">", color="#0f172a", alpha=0.65, linewidth=0, zorder=8)
    if np.any(attached):
        ax.scatter(arr[attached, 0], arr[attached, 1], s=30, marker="x", color="#dc2626", alpha=0.95, linewidth=1.4, zorder=9)
    if np.any(censored):
        ax.scatter(arr[censored, 0], arr[censored, 1], s=16, marker="s", color="#6b7280", alpha=0.75, linewidth=0, zorder=8)


def draw_rear_highlight(ax, rear_points: np.ndarray) -> None:
    if not rear_points.size:
        return
    for x, y in rear_points:
        ax.add_patch(
            Circle(
                (x, y),
                0.030,
                facecolor="#fee2e2",
                edgecolor="none",
                alpha=0.42,
                zorder=4.5,
            )
        )
    ax.scatter(
        rear_points[:, 0],
        rear_points[:, 1],
        s=42,
        marker="o",
        facecolor="none",
        edgecolor="#ef4444",
        linewidth=1.25,
        zorder=7,
    )


def draw_panel(
    ax,
    panel: dict,
    paths: dict,
    releases: dict[str, np.ndarray],
    attachments: dict[str, np.ndarray],
    rear_points: np.ndarray,
    metrics: dict,
    geometry: dict,
) -> None:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    profile = str(panel["profile"])
    color = str(panel["color"])

    draw_grains(ax, geometry)
    draw_rear_highlight(ax, rear_points)

    arr = releases.get(profile)
    if arr is not None and arr.size:
        focused = arr[:, 2] <= 30.0
        diffuse_color = "#60a5fa" if profile == "neutral_resolved" else "#94a3b8"
        if profile == "unfavorable_50mM_z70":
            diffuse_color = "#a78bfa"
        ax.scatter(arr[~focused, 0], arr[~focused, 1], s=5, color=diffuse_color, alpha=0.20, linewidth=0, zorder=5)
        ax.scatter(arr[focused, 0], arr[focused, 1], s=10, color="#ef4444", alpha=0.58, linewidth=0, zorder=6)

    attached_points = attachments.get(profile)
    if attached_points is not None and attached_points.size:
        ax.scatter(
            attached_points[:, 0],
            attached_points[:, 1],
            s=15,
            marker="x",
            color="#dc2626",
            alpha=0.75,
            linewidth=0.85,
            zorder=8,
        )

    draw_pathlines(ax, profile, paths, lx, ly, color)
    draw_matched_endpoints(ax, profile, paths, lx, ly)

    median, f30 = metrics[profile]
    ax.text(
        0.025,
        0.955,
        f"{panel['title']}\n{panel['subtitle']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        weight="bold",
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    ax.text(
        0.025,
        0.045,
        f"median rear angle = {median:.1f} deg\nwithin 30 deg = {f30:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.set_xlim(0.0, lx * 1.0e3)
    ax.set_ylim(0.0, ly * 1.0e3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-path", type=Path, default=GEOMETRY_PATH)
    parser.add_argument("--flow-case", type=Path, default=FLOW_CASE)
    parser.add_argument("--production-payload", type=Path, default=PRODUCTION_PAYLOAD)
    parser.add_argument("--release-events", type=Path, default=RELEASE_EVENTS)
    parser.add_argument("--stagnation-points", type=Path, default=STAGNATION_POINTS)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--path-payload", type=Path, default=PATH_PAYLOAD)
    parser.add_argument("--figure-path", type=Path, default=FIGURE_PATH)
    parser.add_argument("--seed-profile", default="unfavorable_50mM_z70")
    parser.add_argument("--path-count", type=int, default=6)
    parser.add_argument("--path-max-time", type=float, default=45.0)
    parser.add_argument("--path-dt", type=float, default=1.5e-2)
    parser.add_argument("--record-stride", type=int, default=8)
    parser.add_argument("--target-mean-velocity-m-per-day", type=float, default=rpt.TARGET_M_PER_DAY)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--no-rescale-flow", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for key in ("geometry_path", "flow_case", "production_payload", "release_events", "stagnation_points", "summary_csv", "path_payload", "figure_path"):
        value = getattr(args, key)
        if not value.is_absolute():
            setattr(args, key, ROOT / value)
    return args


def main() -> None:
    args = parse_args()
    geometry = load_geometry(args.geometry_path, args.flow_case, not args.no_flow_origin_shift)
    paths = simulate_pathlines(args)
    releases = load_release_points(geometry, args.release_events)
    attachments = load_attachment_points(args.production_payload, geometry)
    rear_points = load_rear_points(args.stagnation_points)
    metrics = load_summary_metrics(args.summary_csv)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.8), dpi=260, constrained_layout=True)
    for ax, panel in zip(axes, PANELS):
        draw_panel(ax, panel, paths, releases, attachments, rear_points, metrics, geometry)
    fig.suptitle("Matched pathlines reveal scattering, attachment, and mobile focusing", fontsize=14.0, weight="bold", y=1.012)
    fig.text(
        0.5,
        -0.008,
        "Matched pathlines use the same inlet positions in all panels. Blue/purple dots are mobile releases, red dots are releases within 30 deg of grain-local rear points, red x marks favorable attachment, and red rings mark rear-flow stagnation points.",
        ha="center",
        va="top",
        fontsize=10.0,
        color="#374151",
    )
    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_path, bbox_inches="tight")
    print(args.figure_path)
    print(args.path_payload)


if __name__ == "__main__":
    main()
