#!/usr/bin/env python3
"""Estimate a release-to-next-interception kernel in the random packing.

The production random-packing tracker records the grain and angle at which a
particle leaves a near-surface event.  This diagnostic reconstructs those
release locations, then advances particles with resolved-flow advection and
bulk Brownian diffusion until the next near-surface encounter.  It is a limited
downstream-consequence analysis: it isolates how the measured release-angle
distribution maps into the next collector event, without adding heterodomain
attachment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = ROOT / "outputs" / "next_interception_kernel"
FIG_OUT = ROOT / "outputs" / "figures" / "next_interception_kernel.png"
RELEASE_DIRS = (
    ROOT / "outputs" / "random_grain_focusing_production_selected",
    ROOT / "outputs" / "random_grain_focusing_production_seed20260508",
    ROOT / "outputs" / "random_grain_focusing_production_seed20260509",
)
DEFAULT_PROFILES = (
    "neutral_resolved",
    "unfavorable_50mM_z70",
    "unfavorable_75mM_z70",
    "unfavorable_100mM_z70",
    "unfavorable_50mM_z70_100xD",
)
PROFILE_LABELS = {
    "neutral_resolved": "No DLVO",
    "unfavorable_50mM_z70": "50 mM unfav.",
    "unfavorable_75mM_z70": "75 mM unfav.",
    "unfavorable_100mM_z70": "100 mM unfav.",
    "unfavorable_50mM_z70_100xD": "50 mM unfav., 100D",
}
PROFILE_COLORS = {
    "neutral_resolved": "#2563eb",
    "unfavorable_50mM_z70": "#7c3aed",
    "unfavorable_75mM_z70": "#9333ea",
    "unfavorable_100mM_z70": "#581c87",
    "unfavorable_50mM_z70_100xD": "#0f766e",
}
PROFILE_UPDATES = {
    "neutral_resolved": {},
    "unfavorable_50mM_z70": {"ionic_strength_molar": 50.0e-3},
    "unfavorable_75mM_z70": {"ionic_strength_molar": 75.0e-3},
    "unfavorable_100mM_z70": {"ionic_strength_molar": 100.0e-3},
    "unfavorable_50mM_z70_100xD": {
        "ionic_strength_molar": 50.0e-3,
        "diffusivity_multiplier": 100.0,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def angular_distance(theta: np.ndarray, target: float) -> np.ndarray:
    return np.arctan2(np.sin(theta - target), np.cos(theta - target))


def load_release_events(profiles: tuple[str, ...], release_dirs: tuple[Path, ...]) -> dict[str, list[dict[str, object]]]:
    events: dict[str, list[dict[str, object]]] = defaultdict(list)
    for directory in release_dirs:
        path = directory / "grain_local_release_events.csv"
        if not path.exists():
            continue
        source = directory.name
        for row in read_csv(path):
            profile = row["profile"]
            if profile not in profiles:
                continue
            events[profile].append(
                {
                    "profile": profile,
                    "source": source,
                    "particle_id": int(row["particle_id"]),
                    "grain_exit": int(row["grain_exit"]),
                    "theta_exit_rad": float(row["theta_exit_rad"]),
                    "abs_delta_from_nearest_rear_deg": float(row["abs_delta_from_nearest_rear_deg"]),
                }
            )
    return events


def load_rear_angles(release_dir: Path) -> dict[int, np.ndarray]:
    path = release_dir / "grain_stagnation_points.csv"
    rear: dict[int, list[float]] = defaultdict(list)
    for row in read_csv(path):
        if row["stagnation_type"] == "rear":
            rear[int(row["grain_id"])].append(float(row["theta_rad"]))
    return {gid: np.asarray(values, dtype=float) for gid, values in rear.items()}


def nearest_rear_angle_deg(theta: np.ndarray, grain: np.ndarray, rear_angles: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full(theta.size, np.nan)
    for gid in np.unique(grain[grain >= 0]):
        mask = grain == gid
        candidates = rear_angles.get(int(gid))
        if candidates is None or candidates.size == 0:
            continue
        deltas = np.abs(angular_distance(theta[mask, None], candidates[None, :]))
        out[mask] = np.degrees(np.min(deltas, axis=1))
    return out


def class_from_angle(angle_deg: np.ndarray) -> np.ndarray:
    out = np.full(angle_deg.size, "transition", dtype=object)
    out[angle_deg <= 30.0] = "focused"
    out[angle_deg > 60.0] = "broad"
    return out


def reconstruct_release_positions(
    geometry: dict,
    rows: list[dict[str, object]],
    params: rpt.RandomTrackingParams,
    release_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([[g["x"], g["y"]] for g in geometry["grains"]], dtype=float)
    radii = np.array([g["radius"] for g in geometry["grains"]], dtype=float)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    grain = np.array([int(row["grain_exit"]) for row in rows], dtype=int)
    theta = np.array([float(row["theta_exit_rad"]) for row in rows], dtype=float)
    shell = radii[grain] + params.particle_radius + params.near_surface + release_offset
    x = np.mod(centers[grain, 0] + shell * np.cos(theta), lx)
    y = np.mod(centers[grain, 1] + shell * np.sin(theta), ly)
    return x, y


def propagate_to_next_interception(
    profile: str,
    rows: list[dict[str, object]],
    interpolator: rpt.PeriodicIDWFlow,
    disks: rpt.PeriodicDiskGeometry,
    geometry: dict,
    rear_angles: dict[int, np.ndarray],
    *,
    max_events: int,
    dt: float,
    max_time: float,
    release_offset: float,
    seed: int,
) -> list[dict[str, object]]:
    if not rows:
        return []
    rng = np.random.default_rng(seed)
    if len(rows) > max_events:
        chosen = rng.choice(len(rows), size=max_events, replace=False)
        rows = [rows[int(i)] for i in chosen]
    params = replace(rpt.RandomTrackingParams(), **PROFILE_UPDATES.get(profile, {}))
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    x_release, y_release = reconstruct_release_positions(geometry, rows, params, release_offset)
    x = x_release.copy()
    y = y_release.copy()
    release_grain = np.array([int(row["grain_exit"]) for row in rows], dtype=int)
    release_angle = np.array([float(row["theta_exit_rad"]) for row in rows], dtype=float)
    release_rear_delta = np.array([float(row["abs_delta_from_nearest_rear_deg"]) for row in rows], dtype=float)
    release_class = class_from_angle(release_rear_delta)

    n = len(rows)
    active = np.ones(n, dtype=bool)
    hit_time = np.full(n, np.nan)
    hit_x = np.full(n, np.nan)
    hit_y = np.full(n, np.nan)
    hit_grain = np.full(n, -1, dtype=int)
    hit_theta = np.full(n, np.nan)
    min_time = 0.02
    n_steps = int(math.ceil(max_time / dt))
    sqrt_2d = math.sqrt(2.0 * params.diffusivity * dt)
    time = 0.0

    for _ in range(n_steps):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        ux, uy = interpolator.velocity_at(x[idx], y[idx])
        x[idx] = x[idx] + ux * dt + sqrt_2d * rng.normal(size=idx.size)
        y[idx] = np.mod(y[idx] + uy * dt + sqrt_2d * rng.normal(size=idx.size), ly)
        time += dt
        gap, _, _, distance, grain = disks.nearest_surface(x[idx], y[idx])
        newly_hit = (gap <= params.near_surface) & (time >= min_time)
        if not np.any(newly_hit):
            continue
        hit_idx = idx[newly_hit]
        hit_time[hit_idx] = time
        hit_x[hit_idx] = x[hit_idx]
        hit_y[hit_idx] = y[hit_idx]
        hit_grain[hit_idx] = grain[newly_hit]
        centers = disks.centers[grain[newly_hit]]
        dx = rpt.periodic_delta(np.mod(x[hit_idx], lx) - centers[:, 0], lx)
        dy = rpt.periodic_delta(np.mod(y[hit_idx], ly) - centers[:, 1], ly)
        hit_theta[hit_idx] = np.mod(np.arctan2(dy, dx), 2.0 * math.pi)
        active[hit_idx] = False

    hit_rear_delta = nearest_rear_angle_deg(hit_theta, hit_grain, rear_angles)
    out: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        out.append(
            {
                "profile": profile,
                "source": row["source"],
                "particle_id": row["particle_id"],
                "release_grain": int(release_grain[i]),
                "release_theta_rad": float(release_angle[i]),
                "release_delta_rear_deg": float(release_rear_delta[i]),
                "release_class": str(release_class[i]),
                "release_x_m": float(x_release[i]),
                "release_y_m": float(y_release[i]),
                "next_interception": bool(np.isfinite(hit_time[i])),
                "next_time_s": float(hit_time[i]) if np.isfinite(hit_time[i]) else float("nan"),
                "next_downstream_displacement_um": float((hit_x[i] - x_release[i]) * 1.0e6) if np.isfinite(hit_time[i]) else float("nan"),
                "next_grain": int(hit_grain[i]),
                "next_same_grain": bool(hit_grain[i] == release_grain[i]) if hit_grain[i] >= 0 else False,
                "next_theta_rad": float(hit_theta[i]) if np.isfinite(hit_theta[i]) else float("nan"),
                "next_delta_rear_deg": float(hit_rear_delta[i]) if np.isfinite(hit_rear_delta[i]) else float("nan"),
                "next_within_30deg_rear": bool(hit_rear_delta[i] <= 30.0) if np.isfinite(hit_rear_delta[i]) else False,
            }
        )
    return out


def summarize_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    profiles = list(DEFAULT_PROFILES)
    classes = ("all", "focused", "transition", "broad")
    for profile in profiles:
        profile_events = [row for row in events if row["profile"] == profile]
        for cls in classes:
            if cls == "all":
                group = profile_events
            else:
                group = [row for row in profile_events if row["release_class"] == cls]
            if not group:
                continue
            hit = [row for row in group if bool(row["next_interception"])]
            next_rear = np.array([float(row["next_delta_rear_deg"]) for row in hit], dtype=float)
            next_time = np.array([float(row["next_time_s"]) for row in hit], dtype=float)
            next_dx = np.array([float(row["next_downstream_displacement_um"]) for row in hit], dtype=float)
            next_grain = np.array([int(row["next_grain"]) for row in hit], dtype=int)
            same = np.array([bool(row["next_same_grain"]) for row in hit], dtype=bool)
            release_rear = np.array([float(row["release_delta_rear_deg"]) for row in group], dtype=float)
            unique, counts = np.unique(next_grain[next_grain >= 0], return_counts=True)
            if counts.size:
                prob = counts / np.sum(counts)
                entropy = -float(np.sum(prob * np.log2(prob)))
                effective_grains = float(2.0**entropy)
            else:
                effective_grains = float("nan")
            rows.append(
                {
                    "profile": profile,
                    "release_class": cls,
                    "release_count": len(group),
                    "release_F30": float(np.mean(release_rear <= 30.0)),
                    "next_interception_count": len(hit),
                    "next_interception_fraction": len(hit) / len(group),
                    "next_same_grain_fraction": float(np.mean(same)) if same.size else float("nan"),
                    "next_F30_rear": float(np.mean(next_rear <= 30.0)) if next_rear.size else float("nan"),
                    "next_median_rear_angle_deg": float(np.nanmedian(next_rear)) if next_rear.size else float("nan"),
                    "next_median_time_s": float(np.nanmedian(next_time)) if next_time.size else float("nan"),
                    "next_median_downstream_displacement_um": float(np.nanmedian(next_dx)) if next_dx.size else float("nan"),
                    "next_effective_grain_count": effective_grains,
                }
            )
    return rows


def plot_summary(summary: list[dict[str, object]], events: list[dict[str, object]]) -> None:
    all_rows = [row for row in summary if row["release_class"] == "all"]
    profiles = [p for p in DEFAULT_PROFILES if any(row["profile"] == p for row in all_rows)]
    x = np.arange(len(profiles))
    labels = [PROFILE_LABELS[p] for p in profiles]
    colors = [PROFILE_COLORS[p] for p in profiles]
    by_profile = {row["profile"]: row for row in all_rows}

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.3), dpi=220)
    ax = axes[0, 0]
    release_f30 = [float(by_profile[p]["release_F30"]) for p in profiles]
    next_f30 = [float(by_profile[p]["next_F30_rear"]) for p in profiles]
    width = 0.38
    ax.bar(x - width / 2, release_f30, width=width, color=colors, alpha=0.48, label="release")
    ax.bar(x + width / 2, next_f30, width=width, color=colors, alpha=0.95, label="next interception")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction within rear 30 deg")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Release focusing carries into the next event", fontsize=11)

    ax = axes[0, 1]
    same_medians = []
    different_medians = []
    for profile in profiles:
        same_times = [
            float(row["next_time_s"])
            for row in events
            if row["profile"] == profile and row["next_interception"] and row["next_same_grain"]
        ]
        diff_times = [
            float(row["next_time_s"])
            for row in events
            if row["profile"] == profile and row["next_interception"] and not row["next_same_grain"]
        ]
        same_medians.append(float(np.median(same_times)) if same_times else np.nan)
        different_medians.append(float(np.median(diff_times)) if diff_times else np.nan)
    ax.bar(x - width / 2, same_medians, width=width, color=colors, alpha=0.88, label="same grain")
    ax.bar(x + width / 2, different_medians, width=width, color="#94a3b8", alpha=0.9, label="different grain")
    ax.set_yscale("log")
    ax.set_ylabel("median time to next event (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Rapid same-grain re-encounter dominates", fontsize=11)

    ax = axes[1, 0]
    same = [float(by_profile[p]["next_same_grain_fraction"]) for p in profiles]
    effective = [float(by_profile[p]["next_effective_grain_count"]) for p in profiles]
    ax.bar(x - width / 2, same, width=width, color=colors, alpha=0.85, label="same grain")
    ax2 = ax.twinx()
    ax2.plot(x + width / 2, effective, marker="o", color="#111827", lw=1.4, label="effective next grains")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("same-grain fraction")
    ax2.set_ylabel("effective next grains")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_title("Next collector support", fontsize=11)

    ax = axes[1, 1]
    group = [
        row
        for row in events
        if row["profile"] == "unfavorable_100mM_z70" and row["next_interception"]
    ]
    if group:
        release = np.array([float(row["release_delta_rear_deg"]) for row in group])
        next_delta = np.array([float(row["next_delta_rear_deg"]) for row in group])
        hb = ax.hexbin(
            release,
            next_delta,
            gridsize=34,
            extent=(0.0, 180.0, 0.0, 180.0),
            cmap="Purples",
            mincnt=1,
            linewidths=0.0,
            alpha=0.95,
        )
        cbar = fig.colorbar(hb, ax=ax, shrink=0.82, pad=0.012)
        cbar.set_label("events/bin", fontsize=8)
    ax.axvline(30.0, color="#dc2626", lw=0.9, ls="--")
    ax.axhline(30.0, color="#dc2626", lw=0.9, ls="--")
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 180.0)
    ax.set_xlabel("release angle from rear (deg)")
    ax.set_ylabel("next-event angle from rear (deg)")
    ax.set_title("100 mM release-to-next-event kernel", fontsize=11)

    for ax in axes.flat:
        ax.grid(alpha=0.22, lw=0.5)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-path", type=Path, default=rpt.GEOMETRY_PATH)
    parser.add_argument("--flow-case", type=Path, default=rpt.DEFAULT_FLOW_CASE)
    parser.add_argument("--release-dir", type=Path, action="append")
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--figure-out", type=Path, default=FIG_OUT)
    parser.add_argument("--max-events", type=int, default=2500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-time", type=float, default=50.0)
    parser.add_argument("--release-offset-nm", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--target-mean-velocity-m-per-day", type=float, default=rpt.TARGET_M_PER_DAY)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--no-rescale-flow", action="store_true")
    return parser.parse_args()


def main() -> None:
    global OUT, FIG_OUT
    args = parse_args()
    OUT = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    FIG_OUT = args.figure_out if args.figure_out.is_absolute() else ROOT / args.figure_out
    OUT.mkdir(parents=True, exist_ok=True)
    geometry_path = args.geometry_path if args.geometry_path.is_absolute() else ROOT / args.geometry_path
    flow_case = args.flow_case if args.flow_case.is_absolute() else ROOT / args.flow_case
    release_dirs = tuple(
        (path if path.is_absolute() else ROOT / path) for path in (args.release_dir or list(RELEASE_DIRS))
    )
    profiles = tuple(str(profile) for profile in args.profiles)
    geometry = rpt.load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=not args.no_flow_origin_shift)
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, areas = rpt.flow_io.parse_xml_vtu(vtu)
    if not args.no_rescale_flow:
        velocity_xy, _, _ = rpt.rescale_velocity_to_target(
            velocity_xy,
            areas,
            args.target_mean_velocity_m_per_day,
        )
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    interpolator = rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx=lx, ly=ly, k=8)
    base_params = rpt.RandomTrackingParams()
    disks = rpt.PeriodicDiskGeometry(geometry, base_params.particle_radius)
    rear_angles = load_rear_angles(release_dirs[0])
    release_events = load_release_events(profiles, release_dirs)

    all_events: list[dict[str, object]] = []
    for i_profile, profile in enumerate(profiles):
        rows = release_events.get(profile, [])
        if not rows:
            continue
        all_events.extend(
            propagate_to_next_interception(
                profile,
                rows,
                interpolator,
                disks,
                geometry,
                rear_angles,
                max_events=args.max_events,
                dt=args.dt,
                max_time=args.max_time,
                release_offset=args.release_offset_nm * 1.0e-9,
                seed=args.seed + 1009 * i_profile,
            )
        )
        print(f"processed {profile}: {min(len(rows), args.max_events)} releases", flush=True)
    summary = summarize_events(all_events)
    write_csv(OUT / "release_to_next_interception_events.csv", all_events)
    write_csv(OUT / "release_to_next_interception_summary.csv", summary)
    plot_summary(summary, all_events)

    report = [
        "# Release-to-Next-Interception Diagnostic",
        "",
        f"Particles were restarted {args.release_offset_nm:.1f} nm outside the near-surface release shell and advanced with the resolved random-packing OpenFOAM flow until the next near-surface event or {args.max_time:.1f} s.",
        "",
        "| profile | releases | next events | release F30 | next F30 | same grain | median next time (s) | median downstream dx (um) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in summary if r["release_class"] == "all"]:
        report.append(
            "| {profile} | {release_count} | {next_interception_count} | {release_F30:.3f} | {next_F30_rear:.3f} | {next_same_grain_fraction:.3f} | {next_median_time_s:.2f} | {next_median_downstream_displacement_um:.1f} |".format(
                **row
            )
        )
    report.extend(["", f"Figure: `{FIG_OUT.relative_to(ROOT)}`"])
    (OUT / "release_to_next_interception_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT / "release_to_next_interception_report.md")


if __name__ == "__main__":
    main()
