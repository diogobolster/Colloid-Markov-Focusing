#!/usr/bin/env python3
"""Analyze grain-local focusing from random-geometry particle releases."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)


def angle_delta(theta: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta - reference), np.cos(theta - reference))


def entropy_effective_bins(values: np.ndarray, lo: float, hi: float, bins: int) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    counts, _ = np.histogram(finite, bins=bins, range=(lo, hi))
    positive = counts[counts > 0].astype(float)
    if positive.size == 0:
        return float("nan")
    p = positive / np.sum(positive)
    return float(2.0 ** (-np.sum(p * np.log2(p))))


def compute_grain_stagnation_points(
    geometry: dict,
    interpolator,
    particle_radius: float,
    sample_gap: float,
    ntheta: int,
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, float | int | str]]]:
    disks = rpt.PeriodicDiskGeometry(geometry, particle_radius)
    angles = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    out: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | int | str]] = []
    for gid, ((cx, cy), radius) in enumerate(zip(disks.centers, disks.radii)):
        shell_r = radius + particle_radius + sample_gap
        x = np.mod(cx + shell_r * np.cos(angles), disks.lx)
        y = np.mod(cy + shell_r * np.sin(angles), disks.ly)
        nearest_gap, *_rest, nearest_gid = disks.nearest_surface(x, y)
        ux, uy = interpolator.velocity_at(x, y)
        nx = np.cos(angles)
        ny = np.sin(angles)
        tx = -ny
        ty = nx
        u_t = ux * tx + uy * ty
        u_n = ux * nx + uy * ny
        valid = (nearest_gid == gid) & np.isfinite(u_t) & np.isfinite(u_n) & (nearest_gap > -1.0e-12)
        candidates: list[tuple[str, float, float, float, str]] = []
        for i in range(ntheta):
            j = (i + 1) % ntheta
            if not (valid[i] and valid[j]):
                continue
            ui = u_t[i]
            uj = u_t[j]
            if ui == 0.0:
                frac = 0.0
            elif ui * uj > 0.0:
                continue
            else:
                frac = abs(ui) / (abs(ui) + abs(uj))
            theta = (angles[i] + frac * ((angles[j] - angles[i]) % (2.0 * np.pi))) % (2.0 * np.pi)
            un = (1.0 - frac) * u_n[i] + frac * u_n[j]
            ut = (1.0 - frac) * u_t[i] + frac * u_t[j]
            stagnation_type = "rear" if un > 0.0 else "forward"
            candidates.append((stagnation_type, theta, un, ut, "tangential_zero"))
        forward = [(theta, un, ut, method) for kind, theta, un, ut, method in candidates if kind == "forward"]
        rear = [(theta, un, ut, method) for kind, theta, un, ut, method in candidates if kind == "rear"]
        if not forward:
            if np.any(valid):
                idx = np.where(valid)[0][np.argmin(u_n[valid])]
                forward = [(float(angles[idx]), float(u_n[idx]), float(u_t[idx]), "fallback_min_normal")]
            else:
                forward = [(0.0, float("nan"), float("nan"), "fallback_no_valid_shell")]
        if not rear:
            if np.any(valid):
                idx = np.where(valid)[0][np.argmax(u_n[valid])]
                rear = [(float(angles[idx]), float(u_n[idx]), float(u_t[idx]), "fallback_max_normal")]
            else:
                rear = [(0.0, float("nan"), float("nan"), "fallback_no_valid_shell")]
        out[gid] = {
            "forward": np.array([theta for theta, *_rest in forward], dtype=float),
            "rear": np.array([theta for theta, *_rest in rear], dtype=float),
        }
        for stagnation_type, points in (("forward", forward), ("rear", rear)):
            for point_index, (theta, un, ut, method) in enumerate(points):
                rows.append(
                    {
                        "grain_id": int(gid),
                        "stagnation_type": stagnation_type,
                        "point_index": int(point_index),
                        "theta_rad": float(theta),
                        "theta_deg": float(theta * 180.0 / np.pi),
                        "x_m": float(np.mod(cx + shell_r * math.cos(theta), disks.lx)),
                        "y_m": float(np.mod(cy + shell_r * math.sin(theta), disks.ly)),
                        "normal_velocity_m_s": float(un),
                        "tangential_velocity_m_s": float(ut),
                        "sample_gap_um": float(sample_gap * 1.0e6),
                        "method": method,
                    }
                )
    return out, rows


def nearest_stagnation_delta(
    theta: np.ndarray,
    grain: np.ndarray,
    stagnation: dict[int, dict[str, np.ndarray]],
    stagnation_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.full(theta.size, np.nan)
    nearest_ref = np.full(theta.size, np.nan)
    for gid in np.unique(grain[grain >= 0]):
        mask = grain == gid
        refs = stagnation.get(int(gid), {}).get(stagnation_type, np.array([0.0]))
        all_delta = np.stack([angle_delta(theta[mask], ref) for ref in refs], axis=0)
        take = np.argmin(np.abs(all_delta), axis=0)
        delta[mask] = all_delta[take, np.arange(np.sum(mask))]
        nearest_ref[mask] = refs[take]
    return delta, nearest_ref


def analyze_payload(
    payload_path: Path,
    stagnation: dict[int, dict[str, np.ndarray]],
    angle_bins: int,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    data = np.load(payload_path)
    prefixes = sorted({key[: -len("_theta_exit")] for key in data.files if key.endswith("_theta_exit")})
    rows: list[dict[str, float | int | str]] = []
    event_rows: list[dict[str, float | int | str]] = []
    grain_rows: list[dict[str, float | int | str]] = []
    for prefix in prefixes:
        theta_exit = data[f"{prefix}_theta_exit"]
        grain_exit = data[f"{prefix}_grain_exit"].astype(int)
        interceptions = data[f"{prefix}_interceptions"].astype(int)
        exited = data[f"{prefix}_exited"].astype(bool)
        attached = data[f"{prefix}_attached"].astype(bool)
        censored = data[f"{prefix}_censored"].astype(bool)
        release = np.isfinite(theta_exit) & (grain_exit >= 0) & exited
        intercepted = interceptions > 0
        release_ids = np.flatnonzero(release)
        delta, rear_ref = nearest_stagnation_delta(theta_exit[release], grain_exit[release], stagnation, "rear")
        abs_delta = np.abs(delta)
        release_grains = grain_exit[release]
        for local_index, particle_id in enumerate(release_ids):
            event_rows.append(
                {
                    "profile": prefix,
                    "particle_id": int(particle_id),
                    "grain_exit": int(grain_exit[particle_id]),
                    "theta_exit_rad": float(theta_exit[particle_id]),
                    "theta_exit_deg": float(theta_exit[particle_id] * 180.0 / np.pi),
                    "nearest_rear_theta_rad": float(rear_ref[local_index]),
                    "nearest_rear_theta_deg": float(rear_ref[local_index] * 180.0 / np.pi),
                    "delta_from_nearest_rear_rad": float(delta[local_index]),
                    "delta_from_nearest_rear_deg": float(delta[local_index] * 180.0 / np.pi),
                    "abs_delta_from_nearest_rear_deg": float(abs_delta[local_index] * 180.0 / np.pi),
                }
            )
        for gid in np.unique(release_grains[release_grains >= 0]):
            grain_mask = release_grains == gid
            grain_delta = delta[grain_mask]
            grain_abs_delta = np.abs(grain_delta)
            grain_rows.append(
                {
                    "profile": prefix,
                    "grain_exit": int(gid),
                    "released_from_near_zone": int(np.sum(grain_mask)),
                    "median_abs_release_angle_from_rear_deg": float(np.median(grain_abs_delta) * 180.0 / np.pi)
                    if grain_abs_delta.size
                    else float("nan"),
                    "fraction_released_within_15deg": float(np.mean(grain_abs_delta <= np.deg2rad(15.0)))
                    if grain_abs_delta.size
                    else float("nan"),
                    "fraction_released_within_30deg": float(np.mean(grain_abs_delta <= np.deg2rad(30.0)))
                    if grain_abs_delta.size
                    else float("nan"),
                    "grain_local_effective_angle_bins": entropy_effective_bins(
                        grain_delta, -np.pi, np.pi, angle_bins
                    ),
                }
            )
        rows.append(
            {
                "profile": prefix,
                "particles": int(theta_exit.size),
                "intercepted": int(np.sum(intercepted)),
                "released_from_near_zone": int(np.sum(release)),
                "attached": int(np.sum(attached)),
                "censored": int(np.sum(censored)),
                "release_fraction_of_intercepted": float(np.sum(release) / np.sum(intercepted)) if np.any(intercepted) else float("nan"),
                "median_abs_release_angle_from_rear_deg": float(np.median(abs_delta) * 180.0 / np.pi)
                if abs_delta.size
                else float("nan"),
                "p25_abs_release_angle_from_rear_deg": float(np.quantile(abs_delta, 0.25) * 180.0 / np.pi)
                if abs_delta.size
                else float("nan"),
                "p75_abs_release_angle_from_rear_deg": float(np.quantile(abs_delta, 0.75) * 180.0 / np.pi)
                if abs_delta.size
                else float("nan"),
                "fraction_released_within_15deg": float(np.mean(abs_delta <= np.deg2rad(15.0))) if abs_delta.size else float("nan"),
                "fraction_released_within_30deg": float(np.mean(abs_delta <= np.deg2rad(30.0))) if abs_delta.size else float("nan"),
                "grain_local_effective_angle_bins": entropy_effective_bins(delta, -np.pi, np.pi, angle_bins),
            }
        )
    return rows, event_rows, grain_rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float | int | str]], out_dir: Path) -> Path:
    labels = [str(row["profile"]) for row in rows]
    x = np.arange(len(rows))
    median = np.array([float(row["median_abs_release_angle_from_rear_deg"]) for row in rows])
    within30 = np.array([float(row["fraction_released_within_30deg"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), dpi=180, sharex=True)
    axes[0].bar(x, median, color="#2563eb")
    axes[0].set_ylabel("median |release angle - rear stagnation| (deg)")
    axes[1].bar(x, within30, color="#0f766e")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("fraction released within 30 deg")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    fig.suptitle("Grain-local focusing diagnostic")
    fig.tight_layout()
    path = out_dir / "grain_local_focusing.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_stagnation_points(
    geometry: dict,
    stagnation_rows: list[dict[str, float | int | str]],
    out_dir: Path,
) -> Path:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    grains = geometry["grains"]
    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=180)
    for gid, grain in enumerate(grains):
        cx = float(grain["x"])
        cy = float(grain["y"])
        radius = float(grain["radius"])
        circle = plt.Circle((cx * 1.0e3, cy * 1.0e3), radius * 1.0e3, color="#111827", alpha=0.9)
        ax.add_patch(circle)
        ax.text(cx * 1.0e3, cy * 1.0e3, str(gid), color="white", ha="center", va="center", fontsize=6)
    for stagnation_type, marker, color, label in (
        ("forward", "^", "#2563eb", "forward"),
        ("rear", "o", "#dc2626", "rear"),
    ):
        xs = [float(row["x_m"]) * 1.0e3 for row in stagnation_rows if row["stagnation_type"] == stagnation_type]
        ys = [float(row["y_m"]) * 1.0e3 for row in stagnation_rows if row["stagnation_type"] == stagnation_type]
        ax.scatter(xs, ys, s=24, marker=marker, color=color, edgecolor="white", linewidth=0.4, label=label, zorder=4)
    ax.set_xlim(0.0, lx * 1.0e3)
    ax.set_ylim(0.0, ly * 1.0e3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Per-grain forward and rear stagnation points")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out_dir / "grain_stagnation_points.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(
    path: Path,
    rows: list[dict[str, float | int | str]],
    focusing_figure_path: Path,
    stagnation_figure_path: Path,
) -> None:
    lines = [
        "# Grain-local focusing diagnostic",
        "",
        "Focusing is measured at the grain scale: release angles are compared with rear-flow stagnation point(s) on the same grain, identified from zeros of the local tangential OpenFOAM velocity around each collector. Forward and rear stagnation points are stored separately for every grain; no global flow-axis alignment is assumed.",
        "",
        "| profile | intercepted | released | censored | median angle from rear (deg) | within 30 deg | effective angle bins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['intercepted']} | {row['released_from_near_zone']} | {row['censored']} | "
            f"{float(row['median_abs_release_angle_from_rear_deg']):.2f} | "
            f"{float(row['fraction_released_within_30deg']):.3f} | "
            f"{float(row['grain_local_effective_angle_bins']):.2f} |"
        )
    lines.extend(
        [
            "",
            f"Focusing figure: `{focusing_figure_path.relative_to(ROOT)}`",
            f"Stagnation-point figure: `{stagnation_figure_path.relative_to(ROOT)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "random_grain_focusing")
    parser.add_argument("--flow-case", type=Path, default=rpt.DEFAULT_FLOW_CASE)
    parser.add_argument("--geometry-path", type=Path, default=rpt.GEOMETRY_PATH)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--sample-gap-um", type=float, default=2.0)
    parser.add_argument("--ntheta", type=int, default=720)
    parser.add_argument("--angle-bins", type=int, default=72)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_case = args.flow_case if args.flow_case.is_absolute() else ROOT / args.flow_case
    geometry_path = args.geometry_path if args.geometry_path.is_absolute() else ROOT / args.geometry_path
    payload_path = args.payload if args.payload.is_absolute() else ROOT / args.payload
    geometry = rpt.load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=not args.no_flow_origin_shift)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, _ = rpt.flow_io.parse_xml_vtu(vtu)
    interpolator = rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8)
    params = rpt.RandomTrackingParams()
    stagnation, stagnation_rows = compute_grain_stagnation_points(
        geometry,
        interpolator,
        params.particle_radius,
        args.sample_gap_um * 1.0e-6,
        args.ntheta,
    )
    rows, event_rows, grain_rows = analyze_payload(payload_path, stagnation, args.angle_bins)
    write_csv(out_dir / "grain_stagnation_points.csv", stagnation_rows)
    write_csv(out_dir / "grain_local_release_events.csv", event_rows)
    write_csv(out_dir / "grain_local_focusing_by_grain.csv", grain_rows)
    write_csv(out_dir / "grain_local_focusing.csv", rows)
    fig = plot_rows(rows, out_dir)
    stagnation_fig = plot_stagnation_points(geometry, stagnation_rows, out_dir)
    write_report(out_dir / "grain_local_focusing_report.md", rows, fig, stagnation_fig)
    print(out_dir / "grain_local_focusing_report.md")
    print(fig)
    print(stagnation_fig)


if __name__ == "__main__":
    main()
