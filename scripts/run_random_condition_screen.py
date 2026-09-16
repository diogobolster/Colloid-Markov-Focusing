#!/usr/bin/env python3
"""Run a compact condition screen in the random OpenFOAM porous cell."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colloid_tsm.compiled import RandomFlowGrid, RandomTrajectoryLibrary, simulate_random_transitions_compiled

DEFAULT_OUT = ROOT / "outputs" / "random_condition_screen"


PROFILES = {
    "neutral_resolved": {
        "label": "No DLVO",
        "condition": "no_dlvo",
        "updates": {},
        "role": "baseline",
    },
    "favorable_50mM_z70": {
        "label": "Favorable 50 mM, +70 mV",
        "condition": "favorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_favorable": 70.0e-3},
        "role": "attachment limit",
    },
    "unfavorable_50mM_z70": {
        "label": "Unfavorable 50 mM, -70 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -70.0e-3},
        "role": "realistic residence",
    },
    "unfavorable_100mM_z70": {
        "label": "Unfavorable 100 mM, -70 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 100.0e-3, "zeta_collector_unfavorable": -70.0e-3},
        "role": "compressed double layer",
    },
    "unfavorable_75mM_z70": {
        "label": "Unfavorable 75 mM, -70 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 75.0e-3, "zeta_collector_unfavorable": -70.0e-3},
        "role": "intermediate salt bridge",
    },
    "unfavorable_50mM_z30": {
        "label": "Unfavorable 50 mM, -30 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -30.0e-3},
        "role": "weaker barrier",
    },
    "unfavorable_50mM_z70_100xD": {
        "label": "Unfavorable 50 mM, 100xD",
        "condition": "unfavorable",
        "updates": {
            "ionic_strength_molar": 50.0e-3,
            "zeta_collector_unfavorable": -70.0e-3,
            "diffusivity_multiplier": 100.0,
        },
        "role": "diffusion control",
    },
    "mechanism_100mM_z20_A10x": {
        "label": "Mechanism 100 mM, -20 mV, 10xA",
        "condition": "unfavorable",
        "updates": {
            "ionic_strength_molar": 100.0e-3,
            "zeta_collector_unfavorable": -20.0e-3,
            "hamaker": 3.83e-20,
        },
        "role": "amplified contrast",
    },
}


def selected_profiles(names: list[str] | None) -> list[dict[str, object]]:
    if not names:
        names = [
            "neutral_resolved",
            "favorable_50mM_z70",
            "unfavorable_50mM_z70",
            "unfavorable_100mM_z70",
            "unfavorable_50mM_z70_100xD",
            "mechanism_100mM_z20_A10x",
        ]
    missing = [name for name in names if name not in PROFILES]
    if missing:
        raise KeyError(f"Unknown profile(s): {', '.join(missing)}")
    out = []
    for name in names:
        profile = dict(PROFILES[name])
        profile["profile"] = name
        out.append(profile)
    return out


def make_params(base: rpt.RandomTrackingParams, profile: dict[str, object]) -> rpt.RandomTrackingParams:
    return replace(base, **dict(profile["updates"]))


def rasterize_flow_grid(
    interpolator: rpt.PeriodicIDWFlow,
    lx: float,
    ly: float,
    nx: int,
    ny: int,
    chunk_size: int = 65_536,
) -> RandomFlowGrid:
    xs = (np.arange(nx, dtype=float) + 0.5) * lx / nx
    ys = (np.arange(ny, dtype=float) + 0.5) * ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    flat_ux = np.empty(flat_x.size, dtype=np.float64)
    flat_uy = np.empty(flat_x.size, dtype=np.float64)
    for start in range(0, flat_x.size, chunk_size):
        end = min(start + chunk_size, flat_x.size)
        flat_ux[start:end], flat_uy[start:end] = interpolator.velocity_at(flat_x[start:end], flat_y[start:end])
    return RandomFlowGrid(
        lx=lx,
        ly=ly,
        ux=np.ascontiguousarray(flat_ux.reshape(nx, ny)),
        uy=np.ascontiguousarray(flat_uy.reshape(nx, ny)),
    )


def flux_weighted_inlet_samples(
    rng: np.random.Generator,
    interpolator: rpt.PeriodicIDWFlow,
    disks: rpt.PeriodicDiskGeometry,
    params: rpt.RandomTrackingParams,
    n_particles: int,
    x0: float,
    clearance: float = 2.0e-6,
    candidates_per_particle: int = 64,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    attempts = 0
    while sum(chunk.size for chunk in samples) < n_particles and attempts < 200:
        count = max(candidates_per_particle * n_particles, 4096)
        y = rng.uniform(0.0, disks.ly, size=count)
        x = np.full(count, x0)
        gap, *_ = disks.nearest_surface(x, y)
        ux, _ = interpolator.velocity_at(x, y)
        weight = np.where((gap > clearance) & (ux > 0.0), ux, 0.0)
        max_weight = float(np.max(weight)) if weight.size else 0.0
        if max_weight <= 0.0:
            attempts += 1
            continue
        accept = rng.uniform(0.0, max_weight, size=count) <= weight
        if np.any(accept):
            samples.append(y[accept])
        attempts += 1
    if not samples:
        raise RuntimeError("Could not sample flux-weighted inlet positions.")
    out = np.concatenate(samples)[:n_particles]
    if out.size < n_particles:
        raise RuntimeError("Could not sample enough flux-weighted inlet positions.")
    return out


def save_random_result_chunk(path: Path, result: RandomTrajectoryLibrary) -> None:
    arrays = {
        field.name: getattr(result, field.name)
        for field in fields(RandomTrajectoryLibrary)
        if isinstance(getattr(result, field.name), np.ndarray)
    }
    np.savez_compressed(path, **arrays)


def load_random_result_chunk(path: Path, condition: str, params) -> RandomTrajectoryLibrary:
    data = np.load(path)
    kwargs: dict[str, object] = {}
    for field in fields(RandomTrajectoryLibrary):
        if field.name == "condition":
            kwargs[field.name] = condition
        elif field.name == "params":
            kwargs[field.name] = params
        else:
            kwargs[field.name] = data[field.name]
    return RandomTrajectoryLibrary(**kwargs)


def concatenate_random_results(
    chunks: list[RandomTrajectoryLibrary],
    condition: str,
    params,
) -> RandomTrajectoryLibrary:
    kwargs: dict[str, object] = {}
    for field in fields(RandomTrajectoryLibrary):
        if field.name == "condition":
            kwargs[field.name] = condition
        elif field.name == "params":
            kwargs[field.name] = params
        else:
            kwargs[field.name] = np.concatenate([np.asarray(getattr(chunk, field.name)) for chunk in chunks])
    return RandomTrajectoryLibrary(**kwargs)


def simulate_compiled_profile(
    flow_grid: RandomFlowGrid,
    geometry: dict,
    params,
    profile: dict[str, object],
    profile_seed: int,
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    out_dir: Path,
    chunk_size: int,
    resume: bool,
) -> RandomTrajectoryLibrary:
    condition = str(profile["condition"])
    allow_attachment = condition == "favorable"
    if chunk_size <= 0 or initial_y.size <= chunk_size:
        return simulate_random_transitions_compiled(
            flow_grid,
            geometry,
            params.as_physical_params(),
            condition,
            n_particles=int(initial_y.size),
            seed=profile_seed,
            allow_attachment=allow_attachment,
            initial_x=initial_x,
            initial_y=initial_y,
        )

    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = int(math.ceil(initial_y.size / chunk_size))
    chunks: list[RandomTrajectoryLibrary] = []
    profile_name = str(profile["profile"])
    for chunk_index, start in enumerate(range(0, initial_y.size, chunk_size), start=1):
        end = min(start + chunk_size, initial_y.size)
        chunk_path = chunk_dir / f"{profile_name}_chunk{chunk_index:04d}of{n_chunks:04d}.npz"
        if resume and chunk_path.exists():
            chunk = load_random_result_chunk(chunk_path, condition, params.as_physical_params())
            if chunk.x0.size == end - start:
                print(f"reused {profile_name} chunk {chunk_index}/{n_chunks}", flush=True)
                chunks.append(chunk)
                continue
            print(f"ignoring stale {profile_name} chunk {chunk_index}/{n_chunks}", flush=True)
        chunk = simulate_random_transitions_compiled(
            flow_grid,
            geometry,
            params.as_physical_params(),
            condition,
            n_particles=end - start,
            seed=profile_seed + 1_000_003 * chunk_index,
            allow_attachment=allow_attachment,
            initial_x=initial_x[start:end],
            initial_y=initial_y[start:end],
        )
        save_random_result_chunk(chunk_path, chunk)
        print(f"finished {profile_name} chunk {chunk_index}/{n_chunks}", flush=True)
        chunks.append(chunk)
    return concatenate_random_results(chunks, condition, params.as_physical_params())


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_outcomes(rows: list[dict[str, float | int | str]], out_dir: Path) -> Path:
    labels = [str(row["profile"]) for row in rows]
    x = np.arange(len(rows))
    particles = np.array([float(row["particles"]) for row in rows])
    exited = np.array([float(row["exited"]) for row in rows]) / particles
    attached = np.array([float(row["attached"]) for row in rows]) / particles
    censored = np.array([float(row["censored"]) for row in rows]) / particles
    near = np.array([float(row["median_near_time_intercepted_s"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), dpi=180, sharex=True)
    axes[0].bar(x, exited, label="exited", color="#2563eb")
    axes[0].bar(x, attached, bottom=exited, label="attached", color="#dc2626")
    axes[0].bar(x, censored, bottom=exited + attached, label="censored", color="#6b7280")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("fraction")
    axes[0].legend(ncols=3, frameon=False, loc="upper right")
    axes[1].bar(x, near, color="#0f766e")
    axes[1].set_ylabel("median near time\nif intercepted (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    fig.suptitle("Random-cell initial condition screen")
    fig.tight_layout()
    path = out_dir / "random_condition_screen_outcomes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def transition_matrix(result: rpt.TrackingResult, ly: float, bins: int) -> np.ndarray:
    yin_bin = np.clip((result.y0 / ly * bins).astype(int), 0, bins - 1)
    matrix = np.zeros((bins, bins), dtype=float)
    for i in range(result.x0.size):
        if not result.exited[i] or not np.isfinite(result.y_out[i]):
            continue
        yout_bin = int(np.clip(result.y_out[i] / ly * bins, 0, bins - 1))
        matrix[yin_bin[i], yout_bin] += 1.0
    row_total = np.bincount(yin_bin, minlength=bins).astype(float)
    return np.divide(matrix, row_total[:, None], out=np.zeros_like(matrix), where=row_total[:, None] > 0)


def plot_transition_grid(
    results: list[rpt.TrackingResult],
    profiles: list[dict[str, object]],
    geometry: dict,
    bins: int,
    out_dir: Path,
) -> Path:
    ly = float(geometry["domain"]["length_y"])
    cols = 3
    rows = int(math.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.2, 3.8 * rows), dpi=180, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(rows, cols)
    vmax = 0.35
    image = None
    for ax, result, profile in zip(axes.flat, results, profiles):
        probability = transition_matrix(result, ly, bins)
        image = ax.imshow(
            probability.T,
            origin="lower",
            extent=(0.0, ly * 1.0e3, 0.0, ly * 1.0e3),
            interpolation="nearest",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(str(profile["label"]), fontsize=10)
        ax.set_xlabel("inlet y (mm)")
        ax.set_ylabel("outlet y (mm)")
    for ax in axes.flat[len(results) :]:
        ax.axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.012)
        cbar.set_label("row-normalized probability")
    fig.suptitle(f"Random-cell transition screen ({bins} bins)", y=0.995)
    path = out_dir / "random_condition_screen_transition_grid.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    rows: list[dict[str, float | int | str]],
    outcome_path: Path,
    matrix_path: Path,
    profiles: list[dict[str, object]],
    geometry_path: Path,
    flow_case: Path,
    flow_mean_ux_before_scale: float,
    flow_scale: float,
) -> Path:
    lines = [
        "# Random condition screen",
        "",
        "Compact initial screen for selecting the smaller set of full-compute cases.",
        "",
        f"- Geometry source: `{geometry_path.relative_to(ROOT) if geometry_path.is_absolute() and ROOT in geometry_path.parents else geometry_path}`",
        f"- Flow case: `{flow_case.relative_to(ROOT) if flow_case.is_absolute() and ROOT in flow_case.parents else flow_case}`",
        f"- Area-weighted mean ux before scaling: {flow_mean_ux_before_scale:.4e} m/s",
        f"- Velocity scale factor to {rpt.TARGET_M_PER_DAY:.2f} m/day: {flow_scale:.4g}",
        "",
        "| profile | role | exited | attached | censored | intercepted | median near time if intercepted (s) | intercepted-exit effective bins |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    roles = {str(profile["profile"]): str(profile["role"]) for profile in profiles}
    for row in rows:
        profile_name = str(row["profile"])
        lines.append(
            f"| {profile_name} | {roles[profile_name]} | {row['exited']} | {row['attached']} | {row['censored']} | "
            f"{row['intercepted']} | {float(row['median_near_time_intercepted_s']):.2f} | "
            f"{float(row['intercepted_exit_effective_bins']):.2f} |"
        )
    lines.extend(
        [
            "",
            f"Outcome figure: `{outcome_path.relative_to(ROOT)}`",
            f"Transition figure: `{matrix_path.relative_to(ROOT)}`",
            "",
        ]
    )
    path = out_dir / "random_condition_screen_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-case", type=Path, default=rpt.DEFAULT_FLOW_CASE)
    parser.add_argument("--geometry-path", type=Path, default=rpt.GEOMETRY_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--particles", type=int, default=80)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--matrix-bins", type=int, default=40)
    parser.add_argument("--max-substeps", type=int, default=50)
    parser.add_argument("--backend", choices=["compiled", "python"], default="compiled")
    parser.add_argument("--grid-nx", type=int, default=384)
    parser.add_argument("--grid-ny", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-mean-velocity-m-per-day", type=float, default=rpt.TARGET_M_PER_DAY)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--no-rescale-flow", action="store_true")
    parser.add_argument("--injection-mode", choices=["uniform_open", "flux_weighted"], default="uniform_open")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    flow_case = args.flow_case if args.flow_case.is_absolute() else ROOT / args.flow_case
    geometry_path = args.geometry_path if args.geometry_path.is_absolute() else ROOT / args.geometry_path
    geometry = rpt.load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=not args.no_flow_origin_shift)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, areas = rpt.flow_io.parse_xml_vtu(vtu)
    flow_mean_ux_before_scale = float("nan")
    flow_scale = 1.0
    if not args.no_rescale_flow:
        velocity_xy, flow_mean_ux_before_scale, flow_scale = rpt.rescale_velocity_to_target(
            velocity_xy,
            areas,
            args.target_mean_velocity_m_per_day,
        )
    interpolator = rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8)
    flow_grid = None
    if args.backend == "compiled":
        flow_grid = rasterize_flow_grid(interpolator, lx, ly, args.grid_nx, args.grid_ny)
    profiles = selected_profiles(args.profiles)
    base = rpt.RandomTrackingParams(
        max_time=args.max_time,
        dt=args.dt,
        adaptive_near_wall=True,
        segment_collision=True,
        max_substeps=args.max_substeps,
    )
    disks = rpt.PeriodicDiskGeometry(geometry, base.particle_radius)
    rng = np.random.default_rng(args.seed)
    if args.injection_mode == "flux_weighted":
        initial_y = flux_weighted_inlet_samples(rng, interpolator, disks, base, args.particles, 1.0e-6)
    else:
        initial_y = rpt.open_inlet_samples(rng, args.particles, lx, ly, disks.centers, disks.radii, base, 1.0e-6)
    initial_x = np.full(initial_y.size, 1.0e-6, dtype=np.float64)

    rows: list[dict[str, float | int | str]] = []
    results: list[rpt.TrackingResult] = []
    payload: dict[str, np.ndarray] = {}
    for profile_id, profile in enumerate(profiles):
        params = make_params(base, profile)
        if args.backend == "compiled":
            assert flow_grid is not None
            result = simulate_compiled_profile(
                flow_grid,
                geometry,
                params,
                profile,
                args.seed + 1000 * (profile_id + 1),
                initial_x=initial_x,
                initial_y=initial_y,
                out_dir=out_dir,
                chunk_size=args.chunk_size,
                resume=args.resume,
            )
        else:
            result = rpt.simulate_particles(
                str(profile["condition"]),
                interpolator,
                geometry,
                params,
                n_particles=args.particles,
                seed=args.seed + 1000 * (profile_id + 1),
                initial_y=initial_y,
                record_count=24,
            )
        summary = rpt.summarize(result, params, ly, args.matrix_bins)
        row = {
            "profile": str(profile["profile"]),
            "label": str(profile["label"]),
            "role": str(profile["role"]),
            **summary,
        }
        rows.append(row)
        results.append(result)
        prefix = str(profile["profile"])
        payload[f"{prefix}_y0"] = result.y0
        payload[f"{prefix}_y_out"] = result.y_out
        payload[f"{prefix}_x_final"] = result.x_final
        payload[f"{prefix}_y_final"] = result.y_final
        payload[f"{prefix}_exited"] = result.exited.astype(np.uint8)
        payload[f"{prefix}_attached"] = result.attached.astype(np.uint8)
        payload[f"{prefix}_censored"] = result.censored.astype(np.uint8)
        payload[f"{prefix}_interceptions"] = result.interceptions
        payload[f"{prefix}_near_time"] = result.near_time
        payload[f"{prefix}_h_min"] = result.h_min
        if hasattr(result, "theta_entry"):
            payload[f"{prefix}_theta_entry"] = result.theta_entry
            payload[f"{prefix}_theta_exit"] = result.theta_exit
            payload[f"{prefix}_theta_final"] = result.theta_final
        if hasattr(result, "grain_entry"):
            payload[f"{prefix}_grain_entry"] = result.grain_entry
        if hasattr(result, "grain_exit"):
            payload[f"{prefix}_grain_exit"] = result.grain_exit
        if hasattr(result, "grain_final"):
            payload[f"{prefix}_grain_final"] = result.grain_final
        print(f"finished {profile['profile']}", flush=True)
        write_csv(out_dir / "random_condition_screen_summary_partial.csv", rows)
        np.savez_compressed(out_dir / "random_condition_screen_payload_partial.npz", **payload)

    write_csv(out_dir / "random_condition_screen_summary.csv", rows)
    np.savez_compressed(out_dir / "random_condition_screen_payload.npz", **payload)
    outcome_path = plot_outcomes(rows, out_dir)
    matrix_path = plot_transition_grid(results, profiles, geometry, args.matrix_bins, out_dir)
    report_path = write_report(
        out_dir,
        rows,
        outcome_path,
        matrix_path,
        profiles,
        geometry_path,
        flow_case,
        flow_mean_ux_before_scale,
        flow_scale,
    )
    (out_dir / "random_condition_screen_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    print(report_path)
    print(outcome_path)
    print(matrix_path)


if __name__ == "__main__":
    main()
