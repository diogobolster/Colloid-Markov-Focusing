#!/usr/bin/env python3
"""Run checkpointed Grain-X to Grain-Y entry-FFSZ simulations."""

from __future__ import annotations

import argparse
import hashlib
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

from colloid_tsm.compiled import (  # noqa: E402
    RandomFlowGrid,
    simulate_cell_transitions_compiled,
    simulate_random_transitions_compiled,
)
from colloid_tsm.physical import PhysicalParams, interpolate_velocity, load_flow  # noqa: E402


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rpt = import_script("entry_ffsz_random_tracking", ROOT / "scripts" / "run_random_particle_tracking.py")
rcs = import_script("entry_ffsz_random_screen", ROOT / "scripts" / "run_random_condition_screen.py")

DEFAULT_CONFIG = ROOT / "config" / "entry_ffsz.json"
ENTRY_KEYS = (
    "first_entry_theta",
    "first_entry_time",
    "first_entry_x",
    "first_entry_y",
    "first_entry_collector_cx",
    "first_entry_collector_cy",
    "first_entry_collector",
    "next_entry_theta",
    "next_entry_time",
    "next_entry_x",
    "next_entry_y",
    "next_entry_collector_cx",
    "next_entry_collector_cy",
    "next_entry_collector",
    "departure_theta_to_next",
    "departure_time_to_next",
    "first_collector_near_time",
    "same_collector_reentries_before_next",
    "pair_completed",
)
KERNEL_SHA256 = hashlib.sha256(
    (ROOT / "colloid_tsm" / "native" / "particle_kernel.c").read_bytes()
).hexdigest()


def resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def physical_params(config: dict, profile: dict, *, domain: str, max_time: float) -> PhysicalParams:
    tracking = config["particle_tracking"]
    dt = tracking["periodic_dt_s"] if domain == "periodic" else tracking["random_dt_s"]
    base = PhysicalParams(
        mean_velocity=float(config["target_mean_velocity_m_per_day"]) / 86400.0,
        dt=float(dt),
        max_time=float(max_time),
        near_surface=float(tracking["near_surface_m"]),
        contact_gap=float(tracking["contact_gap_m"]),
        min_gap=float(tracking["contact_gap_m"]),
        resolved_langevin_substep_cutoff=float(tracking["adaptive_cutoff_m"]),
        resolved_langevin_normal_step=float(tracking["normal_step_target_m"]),
        resolved_langevin_min_dt=float(tracking["minimum_substep_s"]),
        resolved_langevin_max_substeps=int(tracking["maximum_substeps"]),
    )
    return replace(base, **dict(profile.get("updates", {})))


def flux_weighted_periodic_samples(
    rng: np.random.Generator,
    flow,
    params: PhysicalParams,
    count: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    accepted = 0
    while accepted < count:
        candidates = max(4096, 16 * (count - accepted))
        y = rng.uniform(params.inlet_y_min, params.inlet_y_max, size=candidates)
        x = np.full(candidates, 1.0e-9)
        ux, _ = interpolate_velocity(flow, x, y)
        weights = np.maximum(ux, 0.0)
        max_weight = float(np.max(weights))
        if max_weight <= 0.0:
            raise RuntimeError("periodic inlet has no positive resolved throughflow")
        keep = rng.uniform(0.0, max_weight, size=candidates) <= weights
        chunks.append(y[keep])
        accepted += int(np.sum(keep))
    return np.ascontiguousarray(np.concatenate(chunks)[:count], dtype=np.float64)


def random_cache_key(entry: dict, config: dict, geometry_path: Path, flow_case: Path) -> str:
    raster = config["random_flow_raster"]
    vtu = rpt.latest_vtu(flow_case)
    payload = {
        "entry": entry,
        "nx": raster["nx"],
        "ny": raster["ny"],
        "neighbors": raster["idw_neighbors"],
        "target": config["target_mean_velocity_m_per_day"],
        "rescale": raster["rescale_to_target_velocity"],
        "shift": raster["apply_flow_origin_shift"],
        "geometry_mtime_ns": geometry_path.stat().st_mtime_ns,
        "vtu": str(vtu),
        "vtu_mtime_ns": vtu.stat().st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_random_resolved_flow(
    entry: dict,
    config: dict,
    data_root: Path,
    cache_dir: Path,
) -> tuple[dict, object, RandomFlowGrid, dict[str, float | str]]:
    geometry_path = resolve_data_path(data_root, str(entry["geometry_path"]))
    flow_case = resolve_data_path(data_root, str(entry["flow_case"]))
    raster_config = config["random_flow_raster"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{entry['name']}_flow_raster.npz"
    key = random_cache_key(entry, config, geometry_path, flow_case)

    geometry = rpt.load_geometry(
        geometry_path,
        flow_case=flow_case,
        apply_flow_shift=bool(raster_config["apply_flow_origin_shift"]),
    )
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(flow_case)
    centers, velocity, areas = rpt.flow_io.parse_xml_vtu(vtu)
    mean_before = float("nan")
    scale = 1.0
    if bool(raster_config["rescale_to_target_velocity"]):
        velocity, mean_before, scale = rpt.rescale_velocity_to_target(
            velocity,
            areas,
            float(config["target_mean_velocity_m_per_day"]),
        )
    interpolator = rpt.PeriodicIDWFlow(
        centers,
        velocity,
        lx,
        ly,
        k=int(raster_config["idw_neighbors"]),
    )

    if cache_path.exists():
        cached = np.load(cache_path)
        cached_key = str(cached["cache_key"].item()) if "cache_key" in cached else ""
        if cached_key == key:
            grid = RandomFlowGrid(
                lx=float(cached["lx"]),
                ly=float(cached["ly"]),
                ux=np.ascontiguousarray(cached["ux"]),
                uy=np.ascontiguousarray(cached["uy"]),
            )
            return geometry, interpolator, grid, {
                "flow_vtu": str(vtu),
                "mean_ux_before_scale_m_s": mean_before,
                "velocity_scale": scale,
                "raster_cache": str(cache_path),
            }

    grid = rcs.rasterize_flow_grid(
        interpolator,
        lx,
        ly,
        int(raster_config["nx"]),
        int(raster_config["ny"]),
    )
    np.savez_compressed(
        cache_path,
        cache_key=np.array(key),
        lx=np.array(lx),
        ly=np.array(ly),
        ux=grid.ux,
        uy=grid.uy,
    )
    return geometry, interpolator, grid, {
        "flow_vtu": str(vtu),
        "mean_ux_before_scale_m_s": mean_before,
        "velocity_scale": scale,
        "raster_cache": str(cache_path),
    }


def run_signature(
    domain: str,
    params: PhysicalParams,
    profile: dict,
    stop_at_next_distinct: bool,
    context: dict | None = None,
) -> str:
    payload = {
        "kernel_sha256": KERNEL_SHA256,
        "domain": domain,
        "params": asdict(params),
        "condition": profile["condition"],
        "allow_attachment": profile["allow_attachment"],
        "stop_at_next_distinct": stop_at_next_distinct,
        "context": context or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def save_chunk(
    path: Path,
    library,
    diagnostics: dict[str, np.ndarray],
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    signature: str,
) -> None:
    payload: dict[str, np.ndarray] = {
        "run_signature": np.array(signature),
        "initial_x": np.asarray(initial_x),
        "initial_y": np.asarray(initial_y),
        "attached": np.asarray(library.attached, dtype=np.uint8),
        "exited": np.asarray(library.exited, dtype=np.uint8),
        "censored": np.asarray(library.censored, dtype=np.uint8),
        "interceptions": np.asarray(library.interceptions, dtype=np.int32),
        "near_time": np.asarray(library.near_time),
        "h_min": np.asarray(library.h_min),
        "travel_time": np.asarray(library.travel_time),
        "x_final": np.asarray(library.x_final),
        "y_final": np.asarray(library.y_final),
    }
    for key in ENTRY_KEYS:
        payload[key] = np.asarray(diagnostics[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def valid_chunk(path: Path, initial_x: np.ndarray, initial_y: np.ndarray, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        count = int(initial_y.size)
        shapes_match = all(key in data and data[key].shape == (count,) for key in ("initial_x", "initial_y", *ENTRY_KEYS))
        return bool(
            shapes_match
            and "run_signature" in data
            and str(data["run_signature"].item()) == signature
            and np.array_equal(data["initial_x"], initial_x)
            and np.array_equal(data["initial_y"], initial_y)
        )
    except Exception:
        return False


def chunk_summary(path: Path) -> dict[str, int]:
    data = np.load(path)
    return {
        "particles": int(data["initial_y"].size),
        "first_entries": int(np.sum(data["first_entry_collector"] >= 0)),
        "paired_entries": int(np.sum(data["pair_completed"].astype(bool))),
        "attached": int(np.sum(data["attached"])),
        "exited": int(np.sum(data["exited"])),
        "censored": int(np.sum(data["censored"])),
    }


def run_periodic(
    config: dict,
    stage_name: str,
    profiles: list[str],
    data_root: Path,
    output: Path,
    resume: bool,
) -> list[dict]:
    stage = config[stage_name]
    tracking = config["particle_tracking"]
    count = int(stage["periodic_particles_per_seed"])
    chunk_size = int(stage["chunk_size"])
    max_time = float(stage["periodic_max_time_s"])
    base_params = physical_params(config, config["profiles"][profiles[0]], domain="periodic", max_time=max_time)
    flow_path = resolve_data_path(data_root, str(config["periodic"]["flow_path"]))
    base_flow = load_flow(flow_path, params=base_params)
    rows: list[dict] = []
    for seed in stage["seeds"]:
        injection_path = output / stage_name / "periodic" / f"seed_{seed}" / "injection.npz"
        if resume and injection_path.exists() and np.load(injection_path)["initial_y"].size == count:
            injection = np.load(injection_path)
            initial_x = injection["initial_x"]
            initial_y = injection["initial_y"]
        else:
            rng = np.random.default_rng(int(seed))
            initial_y = flux_weighted_periodic_samples(rng, base_flow, base_params, count)
            initial_x = np.zeros(count, dtype=np.float64)
            injection_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(injection_path, initial_x=initial_x, initial_y=initial_y)

        for profile_name in profiles:
            profile = config["profiles"][profile_name]
            params = physical_params(config, profile, domain="periodic", max_time=max_time)
            flow = load_flow(flow_path, params=params)
            signature = run_signature(
                "periodic",
                params,
                profile,
                bool(tracking["stop_at_next_distinct_collector"]),
                {
                    "flow_path": str(flow_path),
                    "flow_size": flow_path.stat().st_size,
                    "flow_mtime_ns": flow_path.stat().st_mtime_ns,
                    "resolution": flow.resolution,
                },
            )
            for chunk_index, start in enumerate(range(0, count, chunk_size)):
                end = min(start + chunk_size, count)
                chunk_path = (
                    output
                    / stage_name
                    / "periodic"
                    / f"seed_{seed}"
                    / profile_name
                    / f"chunk_{chunk_index:05d}.npz"
                )
                if resume and valid_chunk(chunk_path, initial_x[start:end], initial_y[start:end], signature):
                    summary = chunk_summary(chunk_path)
                    print(f"reused periodic {profile_name} seed={seed} chunk={chunk_index}: {summary}", flush=True)
                else:
                    library, diagnostics = simulate_cell_transitions_compiled(
                        flow,
                        str(profile["condition"]),
                        seed=int(seed) + 1_000_003 * chunk_index,
                        allow_attachment=bool(profile["allow_attachment"]),
                        initial_x=initial_x[start:end],
                        initial_y=initial_y[start:end],
                        stop_at_next_distinct=bool(tracking["stop_at_next_distinct_collector"]),
                        return_entry_diagnostics=True,
                    )
                    save_chunk(
                        chunk_path,
                        library,
                        diagnostics,
                        initial_x[start:end],
                        initial_y[start:end],
                        signature,
                    )
                    summary = chunk_summary(chunk_path)
                    print(f"finished periodic {profile_name} seed={seed} chunk={chunk_index}: {summary}", flush=True)
                rows.append({
                    "stage": stage_name,
                    "domain": "periodic",
                    "geometry": "center_corner_cell",
                    "profile": profile_name,
                    "seed": int(seed),
                    "chunk": chunk_index,
                    "path": str(chunk_path),
                    **summary,
                })
    return rows


def run_random(
    config: dict,
    stage_name: str,
    profiles: list[str],
    selected_geometries: set[str] | None,
    data_root: Path,
    output: Path,
    resume: bool,
) -> list[dict]:
    stage = config[stage_name]
    tracking = config["particle_tracking"]
    count = int(stage["random_particles_per_seed"])
    chunk_size = int(stage["chunk_size"])
    max_time = float(stage["random_max_time_s"])
    rows: list[dict] = []
    entries = [entry for entry in config["random_geometries"] if selected_geometries is None or entry["name"] in selected_geometries]
    for geometry_index, entry in enumerate(entries):
        name = str(entry["name"])
        geometry, interpolator, flow_grid, flow_meta = load_random_resolved_flow(
            entry,
            config,
            data_root,
            output / "raster_cache",
        )
        base_params = physical_params(config, config["profiles"][profiles[0]], domain="random", max_time=max_time)
        disks = rpt.PeriodicDiskGeometry(geometry, base_params.particle_radius)
        for seed in stage["seeds"]:
            injection_path = output / stage_name / "random" / name / f"seed_{seed}" / "injection.npz"
            if resume and injection_path.exists() and np.load(injection_path)["initial_y"].size == count:
                injection = np.load(injection_path)
                initial_x = injection["initial_x"]
                initial_y = injection["initial_y"]
            else:
                rng = np.random.default_rng(int(seed) + 10_000_019 * (geometry_index + 1))
                initial_x = np.full(count, 1.0e-6, dtype=np.float64)
                if tracking["injection"] == "flux_weighted":
                    random_params = rpt.RandomTrackingParams(
                        particle_radius=base_params.particle_radius,
                        max_time=max_time,
                        dt=base_params.dt,
                    )
                    initial_y = rcs.flux_weighted_inlet_samples(
                        rng,
                        interpolator,
                        disks,
                        random_params,
                        count,
                        float(initial_x[0]),
                    )
                else:
                    initial_y = rpt.open_inlet_samples(
                        rng,
                        count,
                        disks.lx,
                        disks.ly,
                        disks.centers,
                        disks.radii,
                        rpt.RandomTrackingParams(particle_radius=base_params.particle_radius),
                        float(initial_x[0]),
                    )
                injection_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(injection_path, initial_x=initial_x, initial_y=initial_y)

            metadata_path = output / stage_name / "random" / name / "resolved_flow_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(flow_meta, indent=2) + "\n", encoding="utf-8")
            for profile_name in profiles:
                profile = config["profiles"][profile_name]
                params = physical_params(config, profile, domain="random", max_time=max_time)
                signature = run_signature(
                    "random",
                    params,
                    profile,
                    bool(tracking["stop_at_next_distinct_collector"]),
                    {
                        "geometry": name,
                        "grid_nx": flow_grid.nx,
                        "grid_ny": flow_grid.ny,
                        "flow_vtu": flow_meta["flow_vtu"],
                        "velocity_scale": flow_meta["velocity_scale"],
                    },
                )
                for chunk_index, start in enumerate(range(0, count, chunk_size)):
                    end = min(start + chunk_size, count)
                    chunk_path = (
                        output
                        / stage_name
                        / "random"
                        / name
                        / f"seed_{seed}"
                        / profile_name
                        / f"chunk_{chunk_index:05d}.npz"
                    )
                    if resume and valid_chunk(chunk_path, initial_x[start:end], initial_y[start:end], signature):
                        summary = chunk_summary(chunk_path)
                        print(f"reused random/{name} {profile_name} seed={seed} chunk={chunk_index}: {summary}", flush=True)
                    else:
                        library, diagnostics = simulate_random_transitions_compiled(
                            flow_grid,
                            geometry,
                            params,
                            str(profile["condition"]),
                            seed=int(seed) + 1_000_003 * chunk_index + 10_000_019 * (geometry_index + 1),
                            allow_attachment=bool(profile["allow_attachment"]),
                            initial_x=initial_x[start:end],
                            initial_y=initial_y[start:end],
                            stop_at_next_distinct=bool(tracking["stop_at_next_distinct_collector"]),
                            return_entry_diagnostics=True,
                        )
                        save_chunk(
                            chunk_path,
                            library,
                            diagnostics,
                            initial_x[start:end],
                            initial_y[start:end],
                            signature,
                        )
                        summary = chunk_summary(chunk_path)
                        print(f"finished random/{name} {profile_name} seed={seed} chunk={chunk_index}: {summary}", flush=True)
                    rows.append({
                        "stage": stage_name,
                        "domain": "random",
                        "geometry": name,
                        "profile": profile_name,
                        "seed": int(seed),
                        "chunk": chunk_index,
                        "path": str(chunk_path),
                        **summary,
                    })
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    aggregate: dict[tuple[str, str, str, int], dict] = {}
    for row in rows:
        key = (row["domain"], row["geometry"], row["profile"], row["seed"])
        if key not in aggregate:
            aggregate[key] = {
                "domain": row["domain"],
                "geometry": row["geometry"],
                "profile": row["profile"],
                "seed": row["seed"],
                "particles": 0,
                "first_entries": 0,
                "paired_entries": 0,
                "attached": 0,
                "exited": 0,
                "censored": 0,
            }
        for field in ("particles", "first_entries", "paired_entries", "attached", "exited", "censored"):
            aggregate[key][field] += int(row[field])
    payload = {"chunks": rows, "aggregates": list(aggregate.values())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["pilot", "confirmatory", "production"], default="pilot")
    parser.add_argument("--domains", choices=["both", "periodic", "random"], default="both")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--random-geometries", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    profiles = args.profiles or list(config["profile_order"])
    unknown_profiles = set(profiles) - set(config["profiles"])
    if unknown_profiles:
        raise KeyError(f"unknown profiles: {sorted(unknown_profiles)}")
    selected_geometries = set(args.random_geometries) if args.random_geometries else None

    rows: list[dict] = []
    if args.domains in {"both", "periodic"}:
        rows.extend(run_periodic(config, args.stage, profiles, data_root, output, args.resume))
        write_manifest(output / args.stage / "run_manifest.json", rows)
    if args.domains in {"both", "random"}:
        rows.extend(run_random(config, args.stage, profiles, selected_geometries, data_root, output, args.resume))
        write_manifest(output / args.stage / "run_manifest.json", rows)
    manifest = output / args.stage / "run_manifest.json"
    write_manifest(manifest, rows)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
