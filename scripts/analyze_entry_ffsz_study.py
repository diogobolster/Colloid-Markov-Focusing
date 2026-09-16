#!/usr/bin/env python3
"""Analyze paired Grain-X and Grain-Y forward-stagnation-zone entries."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/colloid_entry_ffsz_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from colloid_tsm.physical import PhysicalParams, interpolate_velocity, load_flow  # noqa: E402


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = import_script("entry_ffsz_runner_for_analysis", ROOT / "scripts" / "run_entry_ffsz_study.py")
grain_analysis = import_script(
    "entry_ffsz_grain_analysis",
    ROOT / "scripts" / "analyze_random_grain_focusing.py",
)

DEFAULT_CONFIG = ROOT / "config" / "entry_ffsz.json"
COLORS = {
    "no_dlvo": "#5B6573",
    "favorable_50mM": "#B43C35",
    "unfavorable_50mM": "#007C78",
    "unfavorable_100mM": "#D07A18",
    "unfavorable_50mM_100D0": "#2E6FB0",
}


class RasterInterpolator:
    """Vectorized periodic bilinear interpolation of a cached resolved raster."""

    def __init__(self, grid) -> None:
        self.grid = grid

    def velocity_at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grid = self.grid
        gx = np.mod(x, grid.lx) / grid.lx * grid.nx - 0.5
        gy = np.mod(y, grid.ly) / grid.ly * grid.ny - 0.5
        i0_raw = np.floor(gx).astype(int)
        j0_raw = np.floor(gy).astype(int)
        tx = gx - i0_raw
        ty = gy - j0_raw
        i0 = np.mod(i0_raw, grid.nx)
        j0 = np.mod(j0_raw, grid.ny)
        i1 = np.mod(i0 + 1, grid.nx)
        j1 = np.mod(j0 + 1, grid.ny)
        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty
        ux = w00 * grid.ux[i0, j0] + w10 * grid.ux[i1, j0] + w01 * grid.ux[i0, j1] + w11 * grid.ux[i1, j1]
        uy = w00 * grid.uy[i0, j0] + w10 * grid.uy[i1, j0] + w01 * grid.uy[i0, j1] + w11 * grid.uy[i1, j1]
        return ux, uy


class PeriodicFlowInterpolator:
    def __init__(self, flow) -> None:
        self.flow = flow

    def velocity_at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return interpolate_velocity(self.flow, x, y)


def nearest_delta(theta: np.ndarray, collector: np.ndarray, refs: dict[int, np.ndarray]) -> np.ndarray:
    delta = np.full(theta.size, np.nan)
    for collector_id in np.unique(collector[collector >= 0]):
        mask = collector == collector_id
        candidates = refs.get(int(collector_id), np.array([], dtype=float))
        if candidates.size == 0:
            continue
        all_delta = np.stack(
            [np.arctan2(np.sin(theta[mask] - ref), np.cos(theta[mask] - ref)) for ref in candidates],
            axis=0,
        )
        take = np.argmin(np.abs(all_delta), axis=0)
        delta[mask] = all_delta[take, np.arange(np.sum(mask))]
    return delta


def periodic_stagnation(config: dict) -> tuple[dict[int, np.ndarray], list[dict]]:
    data_root = Path(config["data_root"])
    tracking = config["particle_tracking"]
    params = PhysicalParams(
        mean_velocity=float(config["target_mean_velocity_m_per_day"]) / 86400.0,
        near_surface=float(tracking["near_surface_m"]),
    )
    flow = load_flow(
        runner.resolve_data_path(data_root, str(config["periodic"]["flow_path"])),
        params=params,
    )
    interpolator = PeriodicFlowInterpolator(flow)
    geometry = {
        "domain": {"length_x": params.cell_length, "length_y": params.cell_length},
        "grains": [
            {"x": 0.0, "y": 0.0, "radius": params.grain_radius},
            {"x": 0.5 * params.cell_length, "y": 0.5 * params.cell_length, "radius": params.grain_radius},
        ],
    }
    stagnation, rows = grain_analysis.compute_grain_stagnation_points(
        geometry,
        interpolator,
        params.particle_radius,
        float(config["analysis"]["stagnation_shell_gap_m"]),
        int(config["analysis"]["stagnation_angle_samples"]),
    )
    refs = {collector: values["forward"] for collector, values in stagnation.items()}
    for row in rows:
        row["domain"] = "periodic"
        row["geometry"] = "center_corner_cell"
    return refs, rows


def random_stagnation(entry: dict, config: dict, output: Path) -> tuple[dict[int, np.ndarray], list[dict]]:
    data_root = Path(config["data_root"])
    geometry_path = runner.resolve_data_path(data_root, str(entry["geometry_path"]))
    flow_case = runner.resolve_data_path(data_root, str(entry["flow_case"]))
    geometry = runner.rpt.load_geometry(
        geometry_path,
        flow_case=flow_case,
        apply_flow_shift=bool(config["random_flow_raster"]["apply_flow_origin_shift"]),
    )
    cache_path = output / "raster_cache" / f"{entry['name']}_flow_raster.npz"
    cached = np.load(cache_path)
    grid = runner.RandomFlowGrid(
        lx=float(cached["lx"]),
        ly=float(cached["ly"]),
        ux=np.ascontiguousarray(cached["ux"]),
        uy=np.ascontiguousarray(cached["uy"]),
    )
    stagnation, rows = grain_analysis.compute_grain_stagnation_points(
        geometry,
        RasterInterpolator(grid),
        PhysicalParams().particle_radius,
        float(config["analysis"]["stagnation_shell_gap_m"]),
        int(config["analysis"]["stagnation_angle_samples"]),
    )
    refs = {collector: values["forward"] for collector, values in stagnation.items()}
    for row in rows:
        row["domain"] = "random"
        row["geometry"] = str(entry["name"])
    return refs, rows


def load_group(files: list[Path], refs: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    keys = [
        "pair_completed",
        "first_entry_theta",
        "next_entry_theta",
        "first_entry_collector",
        "next_entry_collector",
        "first_entry_time",
        "next_entry_time",
        "departure_time_to_next",
        "first_collector_near_time",
        "same_collector_reentries_before_next",
        "attached",
        "censored",
        "initial_y",
    ]
    data = {key: np.concatenate([np.load(path)[key] for path in files]) for key in keys}
    pair = data["pair_completed"].astype(bool)
    out = {key: value[pair] for key, value in data.items() if key != "pair_completed"}
    out["delta_x"] = nearest_delta(
        out["first_entry_theta"],
        out["first_entry_collector"].astype(int),
        refs,
    )
    out["delta_y"] = nearest_delta(
        out["next_entry_theta"],
        out["next_entry_collector"].astype(int),
        refs,
    )
    out["particle_index"] = np.flatnonzero(pair)
    out["all_pair_completed"] = pair
    out["all_first_entry"] = data["first_entry_collector"] >= 0
    out["all_first_delta"] = nearest_delta(
        data["first_entry_theta"],
        data["first_entry_collector"].astype(int),
        refs,
    )
    out["all_attached"] = data["attached"].astype(bool)
    out["all_censored"] = data["censored"].astype(bool)
    out["all_initial_y"] = data["initial_y"]
    return out


def metrics(data: dict[str, np.ndarray], window: float) -> dict[str, float | int]:
    dx = np.abs(data["delta_x"])
    dy = np.abs(data["delta_y"])
    finite = np.isfinite(dx) & np.isfinite(dy)
    dx = dx[finite]
    dy = dy[finite]
    total = int(data["all_pair_completed"].size)
    pairs = int(dx.size)
    return {
        "particles": total,
        "first_entries": int(np.sum(data["all_first_entry"])),
        "paired_entries": pairs,
        "paired_fraction": pairs / total if total else float("nan"),
        "attached": int(np.sum(data["all_attached"])),
        "pair_not_completed_by_horizon": int(np.sum(data["all_censored"])),
        "entry_f30_x": float(np.mean(dx <= window)) if pairs else float("nan"),
        "entry_f30_y": float(np.mean(dy <= window)) if pairs else float("nan"),
        "entry_f30_change_y_minus_x": float(np.mean(dy <= window) - np.mean(dx <= window)) if pairs else float("nan"),
        "median_abs_entry_x_deg": float(np.degrees(np.median(dx))) if pairs else float("nan"),
        "median_abs_entry_y_deg": float(np.degrees(np.median(dy))) if pairs else float("nan"),
        "median_angle_change_y_minus_x_deg": float(np.degrees(np.median(dy - dx))) if pairs else float("nan"),
        "probability_y_tighter_than_x": float(np.mean(dy < dx)) if pairs else float("nan"),
        "median_x_to_y_time_s": float(np.median(data["next_entry_time"][finite] - data["first_entry_time"][finite])) if pairs else float("nan"),
        "median_first_collector_near_time_s": float(np.median(data["first_collector_near_time"][finite])) if pairs else float("nan"),
    }


def bootstrap_intervals(
    data: dict[str, np.ndarray],
    window: float,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    dx = np.abs(data["delta_x"])
    dy = np.abs(data["delta_y"])
    finite = np.isfinite(dx) & np.isfinite(dy)
    dx = dx[finite]
    dy = dy[finite]
    if dx.size < 2:
        return {}
    values = np.empty((replicates, 4), dtype=float)
    for replicate in range(replicates):
        take = rng.integers(0, dx.size, size=dx.size)
        bx = dx[take]
        by = dy[take]
        values[replicate] = (
            np.mean(bx <= window),
            np.mean(by <= window),
            np.mean(by <= window) - np.mean(bx <= window),
            np.mean(by < bx),
        )
    names = ("entry_f30_x", "entry_f30_y", "entry_f30_change_y_minus_x", "probability_y_tighter_than_x")
    out: dict[str, float] = {}
    for column, name in enumerate(names):
        out[f"{name}_ci_low"] = float(np.quantile(values[:, column], 0.025))
        out[f"{name}_ci_high"] = float(np.quantile(values[:, column], 0.975))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
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


def matched_chemistry_metrics(
    no_dlvo: dict[str, np.ndarray],
    unfavorable: dict[str, np.ndarray],
    window: float,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    if not np.array_equal(no_dlvo["all_initial_y"], unfavorable["all_initial_y"]):
        raise ValueError("matched chemistry groups do not share the same injected particles")
    no_lookup = {int(pid): index for index, pid in enumerate(no_dlvo["particle_index"])}
    unf_lookup = {int(pid): index for index, pid in enumerate(unfavorable["particle_index"])}
    common = np.array(sorted(set(no_lookup) & set(unf_lookup)), dtype=int)
    if common.size == 0:
        return {"matched_particles": 0}
    no_take = np.array([no_lookup[int(pid)] for pid in common], dtype=int)
    unf_take = np.array([unf_lookup[int(pid)] for pid in common], dtype=int)
    no_x = np.abs(no_dlvo["delta_x"][no_take])
    no_y = np.abs(no_dlvo["delta_y"][no_take])
    unf_x = np.abs(unfavorable["delta_x"][unf_take])
    unf_y = np.abs(unfavorable["delta_y"][unf_take])
    finite = np.isfinite(no_x) & np.isfinite(no_y) & np.isfinite(unf_x) & np.isfinite(unf_y)
    no_x, no_y, unf_x, unf_y = no_x[finite], no_y[finite], unf_x[finite], unf_y[finite]
    no_change = np.mean(no_y <= window) - np.mean(no_x <= window)
    unf_change = np.mean(unf_y <= window) - np.mean(unf_x <= window)
    no_contraction = no_x - no_y
    unf_contraction = unf_x - unf_y
    additional_contraction = unf_contraction - no_contraction
    no_cosine_change = np.cos(no_y) - np.cos(no_x)
    unfavorable_cosine_change = np.cos(unf_y) - np.cos(unf_x)
    out: dict[str, float | int] = {
        "matched_particles": int(no_x.size),
        "no_dlvo_entry_f30_change": float(no_change),
        "unfavorable_entry_f30_change": float(unf_change),
        "unfavorable_amplification_difference_in_differences": float(unf_change - no_change),
        "mean_additional_contraction_deg": float(np.degrees(np.mean(additional_contraction))),
        "median_additional_contraction_deg": float(np.degrees(np.median(additional_contraction))),
        "mean_cosine_focusing_amplification": float(
            np.mean(unfavorable_cosine_change) - np.mean(no_cosine_change)
        ),
        "probability_unfavorable_contracts_more": float(np.mean(unf_contraction > no_contraction)),
        "no_dlvo_entry_f30_y": float(np.mean(no_y <= window)),
        "unfavorable_entry_f30_y": float(np.mean(unf_y <= window)),
    }
    if no_x.size < 2 or replicates <= 0:
        return out
    bootstrap = np.empty((replicates, 5), dtype=float)
    for replicate in range(replicates):
        take = rng.integers(0, no_x.size, size=no_x.size)
        no_delta = np.mean(no_y[take] <= window) - np.mean(no_x[take] <= window)
        unf_delta = np.mean(unf_y[take] <= window) - np.mean(unf_x[take] <= window)
        bootstrap[replicate] = (
            unf_delta - no_delta,
            np.degrees(np.mean(additional_contraction[take])),
            np.degrees(np.median(additional_contraction[take])),
            np.mean(unfavorable_cosine_change[take]) - np.mean(no_cosine_change[take]),
            np.mean(unf_contraction[take] > no_contraction[take]),
        )
    for column, name in enumerate((
        "unfavorable_amplification_difference_in_differences",
        "mean_additional_contraction_deg",
        "median_additional_contraction_deg",
        "mean_cosine_focusing_amplification",
        "probability_unfavorable_contracts_more",
    )):
        out[f"{name}_ci_low"] = float(np.quantile(bootstrap[:, column], 0.025))
        out[f"{name}_ci_high"] = float(np.quantile(bootstrap[:, column], 0.975))
    return out


def support_selection_metrics(
    no_dlvo: dict[str, np.ndarray],
    unfavorable: dict[str, np.ndarray],
    window: float,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    if not np.array_equal(no_dlvo["all_initial_y"], unfavorable["all_initial_y"]):
        raise ValueError("selection groups do not share the same injected particles")
    no_pair = no_dlvo["all_pair_completed"].astype(bool)
    unfavorable_pair = unfavorable["all_pair_completed"].astype(bool)
    both = no_pair & unfavorable_pair
    no_only = no_pair & ~unfavorable_pair
    unfavorable_only = ~no_pair & unfavorable_pair
    neither = ~no_pair & ~unfavorable_pair
    particle_count = int(no_pair.size)
    no_lookup = {int(pid): index for index, pid in enumerate(no_dlvo["particle_index"])}
    unfavorable_lookup = {int(pid): index for index, pid in enumerate(unfavorable["particle_index"])}

    no_x = np.full(particle_count, np.nan)
    no_y = np.full(particle_count, np.nan)
    unfavorable_x = np.full(particle_count, np.nan)
    unfavorable_y = np.full(particle_count, np.nan)
    no_ids = no_dlvo["particle_index"].astype(int)
    unfavorable_ids = unfavorable["particle_index"].astype(int)
    no_x[no_ids] = np.abs(no_dlvo["delta_x"])
    no_y[no_ids] = np.abs(no_dlvo["delta_y"])
    unfavorable_x[unfavorable_ids] = np.abs(unfavorable["delta_x"])
    unfavorable_y[unfavorable_ids] = np.abs(unfavorable["delta_y"])

    def y_f30(ids: np.ndarray, data: dict[str, np.ndarray], lookup: dict[int, int]) -> float:
        if ids.size == 0:
            return float("nan")
        take = np.array([lookup[int(pid)] for pid in ids], dtype=int)
        return float(np.mean(np.abs(data["delta_y"][take]) <= window))

    def decomposition(ids: np.ndarray) -> tuple[float, float, float]:
        no_valid = np.isfinite(no_x[ids]) & np.isfinite(no_y[ids])
        unfavorable_valid = np.isfinite(unfavorable_x[ids]) & np.isfinite(unfavorable_y[ids])
        common_valid = no_valid & unfavorable_valid
        if not np.any(no_valid) or not np.any(unfavorable_valid) or not np.any(common_valid):
            return float("nan"), float("nan"), float("nan")
        no_change = (
            np.mean(no_y[ids][no_valid] <= window)
            - np.mean(no_x[ids][no_valid] <= window)
        )
        unfavorable_change = (
            np.mean(unfavorable_y[ids][unfavorable_valid] <= window)
            - np.mean(unfavorable_x[ids][unfavorable_valid] <= window)
        )
        no_common_change = (
            np.mean(no_y[ids][common_valid] <= window)
            - np.mean(no_x[ids][common_valid] <= window)
        )
        unfavorable_common_change = (
            np.mean(unfavorable_y[ids][common_valid] <= window)
            - np.mean(unfavorable_x[ids][common_valid] <= window)
        )
        observed = float(unfavorable_change - no_change)
        direct = float(unfavorable_common_change - no_common_change)
        return observed, direct, observed - direct

    observed_amplification, direct_amplification, selection_component = decomposition(
        np.arange(particle_count)
    )
    first_entry_union = no_dlvo["all_first_entry"] | unfavorable["all_first_entry"]
    out: dict[str, float | int] = {
        "injected_particles": int(no_pair.size),
        "first_entry_union": int(np.sum(first_entry_union)),
        "both_complete": int(np.sum(both)),
        "no_dlvo_only_complete": int(np.sum(no_only)),
        "unfavorable_only_complete": int(np.sum(unfavorable_only)),
        "neither_complete": int(np.sum(neither)),
        "no_dlvo_completion_fraction": float(np.mean(no_pair)),
        "unfavorable_completion_fraction": float(np.mean(unfavorable_pair)),
        "completion_fraction_change": float(np.mean(unfavorable_pair) - np.mean(no_pair)),
        "both_no_dlvo_entry_f30_y": y_f30(np.flatnonzero(both), no_dlvo, no_lookup),
        "both_unfavorable_entry_f30_y": y_f30(np.flatnonzero(both), unfavorable, unfavorable_lookup),
        "no_dlvo_only_entry_f30_y": y_f30(np.flatnonzero(no_only), no_dlvo, no_lookup),
        "unfavorable_only_entry_f30_y": y_f30(
            np.flatnonzero(unfavorable_only), unfavorable, unfavorable_lookup
        ),
        "observed_condition_specific_amplification": observed_amplification,
        "matched_direct_amplification": direct_amplification,
        "completion_selection_component": selection_component,
    }
    if replicates > 0 and particle_count > 1:
        bootstrap = np.empty((replicates, 3), dtype=float)
        for replicate in range(replicates):
            bootstrap[replicate] = decomposition(
                rng.integers(0, particle_count, size=particle_count)
            )
        for column, name in enumerate((
            "observed_condition_specific_amplification",
            "matched_direct_amplification",
            "completion_selection_component",
        )):
            finite = bootstrap[:, column][np.isfinite(bootstrap[:, column])]
            if finite.size:
                out[f"{name}_ci_low"] = float(np.quantile(finite, 0.025))
                out[f"{name}_ci_high"] = float(np.quantile(finite, 0.975))
    return out


def completion_by_first_entry_angle(
    no_dlvo: dict[str, np.ndarray],
    unfavorable: dict[str, np.ndarray],
    domain: str,
    geometry: str,
    bins_deg: np.ndarray,
) -> list[dict]:
    first_delta = np.abs(no_dlvo["all_first_delta"])
    unfavorable_first_delta = np.abs(unfavorable["all_first_delta"])
    finite_both = np.isfinite(first_delta) & np.isfinite(unfavorable_first_delta)
    difference = np.abs(first_delta[finite_both] - unfavorable_first_delta[finite_both])
    support_matches = np.array_equal(np.isfinite(first_delta), np.isfinite(unfavorable_first_delta))
    if not support_matches or (difference.size and np.max(difference) > math.radians(0.1)):
        raise ValueError("first-entry coordinates differ before chemistry-specific residence")
    angle_deg = np.degrees(first_delta)
    rows: list[dict] = []
    for lo, hi in zip(bins_deg[:-1], bins_deg[1:]):
        mask = np.isfinite(angle_deg) & (angle_deg >= lo) & (angle_deg < hi)
        support = int(np.sum(mask))
        rows.append({
            "domain": domain,
            "geometry": geometry,
            "entry_x_angle_bin_low_deg": float(lo),
            "entry_x_angle_bin_high_deg": float(hi),
            "first_entry_support": support,
            "no_dlvo_completion_probability": float(np.mean(no_dlvo["all_pair_completed"][mask])) if support else float("nan"),
            "unfavorable_completion_probability": float(np.mean(unfavorable["all_pair_completed"][mask])) if support else float("nan"),
        })
    return rows


def discover_groups(stage_dir: Path) -> list[tuple[str, str, int, str, list[Path]]]:
    groups: list[tuple[str, str, int, str, list[Path]]] = []
    periodic = stage_dir / "periodic"
    if periodic.exists():
        for seed_dir in sorted(periodic.glob("seed_*")):
            seed = int(seed_dir.name.split("_", 1)[1])
            for profile_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
                files = sorted(profile_dir.glob("chunk_*.npz"))
                if files:
                    groups.append(("periodic", "center_corner_cell", seed, profile_dir.name, files))
    random_root = stage_dir / "random"
    if random_root.exists():
        for geometry_dir in sorted(path for path in random_root.iterdir() if path.is_dir()):
            for seed_dir in sorted(geometry_dir.glob("seed_*")):
                seed = int(seed_dir.name.split("_", 1)[1])
                for profile_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
                    files = sorted(profile_dir.glob("chunk_*.npz"))
                    if files:
                        groups.append(("random", geometry_dir.name, seed, profile_dir.name, files))
    return groups


def pooled_data(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = items[0].keys()
    return {key: np.concatenate([item[key] for item in items]) for key in keys}


def pooled_for_matching(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Pool groups while keeping particle indices unique across injections."""

    out: dict[str, list[np.ndarray]] = {key: [] for key in items[0]}
    offset = 0
    for item in items:
        for key, value in item.items():
            array = np.asarray(value)
            if key == "particle_index":
                array = array + offset
            out[key].append(array)
        offset += int(item["all_initial_y"].size)
    return {key: np.concatenate(values) for key, values in out.items()}


def plot_summary(rows: list[dict], out_dir: Path) -> Path:
    rows = [row for row in rows if row["profile"] in {"no_dlvo", "unfavorable_50mM"}]
    geometries = ["center_corner_cell"] + sorted({str(row["geometry"]) for row in rows if row["domain"] == "random"})
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), dpi=220, sharex=True)
    x = np.arange(len(geometries), dtype=float)
    width = 0.34
    for offset, profile in zip((-0.5 * width, 0.5 * width), ("no_dlvo", "unfavorable_50mM")):
        selected = {(str(row["geometry"]), str(row["profile"])): row for row in rows}
        change = np.array([float(selected.get((geometry, profile), {}).get("entry_f30_change_y_minus_x", np.nan)) for geometry in geometries])
        tighten = np.array([float(selected.get((geometry, profile), {}).get("probability_y_tighter_than_x", np.nan)) for geometry in geometries])
        axes[0].bar(x + offset, change, width=width, color=COLORS[profile], label=profile.replace("_", " "))
        axes[1].bar(x + offset, tighten, width=width, color=COLORS[profile])
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_ylabel("Entry F30(Y) - Entry F30(X)")
    axes[0].legend(frameon=False, ncols=2)
    axes[1].axhline(0.5, color="black", lw=0.9, ls="--")
    axes[1].set_ylabel("Pr(|entry Y| < |entry X|)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(geometries, rotation=24, ha="right")
    axes[1].set_xlabel("Resolved geometry")
    fig.suptitle("Grain-to-grain entry-FFSZ contraction")
    fig.tight_layout()
    path = out_dir / "entry_ffsz_geometry_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_kernels(groups: dict[tuple[str, str, int, str], dict[str, np.ndarray]], out_dir: Path) -> Path:
    profiles = [profile for profile in ("no_dlvo", "unfavorable_50mM") if any(key[3] == profile for key in groups)]
    fig, axes = plt.subplots(2, len(profiles), figsize=(5.2 * len(profiles), 8.3), dpi=220, sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(2, len(profiles))
    for column, profile in enumerate(profiles):
        periodic_items = [data for key, data in groups.items() if key[0] == "periodic" and key[3] == profile]
        random_items = [data for key, data in groups.items() if key[0] == "random" and key[3] == profile]
        for row, (label, items) in enumerate((("Periodic cell", periodic_items), ("Random ensemble", random_items))):
            ax = axes[row, column]
            if not items:
                ax.axis("off")
                continue
            data = pooled_data(items)
            x = np.degrees(data["delta_x"])
            y = np.degrees(data["delta_y"])
            hist, xedges, yedges = np.histogram2d(x, y, bins=36, range=[[-180, 180], [-180, 180]])
            probability = np.divide(hist, np.sum(hist, axis=1, keepdims=True), out=np.zeros_like(hist), where=np.sum(hist, axis=1, keepdims=True) > 0)
            ax.imshow(probability.T, origin="lower", extent=(-180, 180, -180, 180), cmap="magma", vmin=0.0, vmax=0.35, aspect="equal")
            ax.plot([-180, 180], [-180, 180], color="white", lw=0.8, ls="--", alpha=0.8)
            ax.axhspan(-30, 30, color="white", alpha=0.08)
            ax.set_title(f"{label}: {profile.replace('_', ' ')}\nN={x.size}")
            ax.set_xlabel("Grain X entry relative to FFSZ (deg)")
            ax.set_ylabel("Grain Y entry relative to FFSZ (deg)")
    fig.suptitle("Paired entry-angle transition kernels")
    fig.tight_layout()
    path = out_dir / "entry_ffsz_transition_kernels.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_matched_chemistry(rows: list[dict], out_dir: Path) -> Path:
    rows = [
        row for row in rows
        if row["seed"] in {"pooled_seeds", "pooled_geometries_and_seeds"}
        and row.get("comparison_profile") == "unfavorable_50mM"
    ]
    rows.sort(key=lambda row: (0 if row["domain"] == "periodic" else 1, str(row["geometry"])))
    if not rows:
        return out_dir / "entry_ffsz_matched_chemistry.png"
    labels = [
        "Periodic cell" if row["domain"] == "periodic"
        else "Random ensemble" if row["geometry"] == "pooled_random_ensemble"
        else str(row["geometry"]).replace("through_", "").replace("_", " ")
        for row in rows
    ]
    values = np.array([float(row.get("unfavorable_amplification_difference_in_differences", np.nan)) for row in rows])
    low = np.array([float(row.get("unfavorable_amplification_difference_in_differences_ci_low", np.nan)) for row in rows])
    high = np.array([float(row.get("unfavorable_amplification_difference_in_differences_ci_high", np.nan)) for row in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=220)
    ax.errorbar(
        x,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="o",
        color=COLORS["unfavorable_50mM"],
        ecolor="#2B3038",
        capsize=4,
        markersize=7,
    )
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_ylabel("Unfavorable amplification of Entry F30 change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Matched-particle chemistry effect on Grain X-to-Y entry focusing")
    for index, row in enumerate(rows):
        ax.text(index, high[index] + 0.008, f"N={row['matched_particles']}", ha="center", fontsize=8)
    fig.tight_layout()
    path = out_dir / "entry_ffsz_matched_chemistry.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_selection_decomposition(rows: list[dict], out_dir: Path) -> Path:
    selected = [
        row for row in rows
        if (
            row["geometry"] == "center_corner_cell" and row["seed"] == "pooled_seeds"
        ) or (
            row["geometry"] == "pooled_random_ensemble"
            and row["seed"] == "pooled_geometries_and_seeds"
        )
    ]
    selected.sort(key=lambda row: 0 if row["domain"] == "periodic" else 1)
    labels = ["Periodic cell" if row["domain"] == "periodic" else "Random ensemble" for row in selected]
    observed = np.array([float(row["observed_condition_specific_amplification"]) for row in selected])
    direct = np.array([float(row["matched_direct_amplification"]) for row in selected])
    selection = np.array([float(row["completion_selection_component"]) for row in selected])
    value_names = (
        "observed_condition_specific_amplification",
        "matched_direct_amplification",
        "completion_selection_component",
    )
    values = (observed, direct, selection)
    errors: list[np.ndarray] = []
    for name, value in zip(value_names, values):
        low = np.array([float(row.get(f"{name}_ci_low", np.nan)) for row in selected])
        high = np.array([float(row.get(f"{name}_ci_high", np.nan)) for row in selected])
        errors.append(np.vstack((value - low, high - value)))
    x = np.arange(len(selected))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=220)
    for positions, value, error, color, label in zip(
        (x - width, x, x + width),
        values,
        errors,
        ("#8A98A8", COLORS["unfavorable_50mM"], "#D7A72D"),
        ("Condition-specific apparent effect", "Matched direct effect", "Completion-selection component"),
    ):
        ax.bar(positions, value, width, yerr=error, capsize=3, color=color, label=label)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_ylabel("Unfavorable amplification of Entry F30 change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    handles, labels_legend = ax.get_legend_handles_labels()
    fig.suptitle("Direct entry-angle change versus completion selection", y=0.98)
    fig.legend(
        handles,
        labels_legend,
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    path = out_dir / "entry_ffsz_selection_decomposition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_completion_by_first_entry(rows: list[dict], out_dir: Path) -> Path:
    panels = (
        ("periodic", "center_corner_cell", "Periodic cell"),
        ("random", "pooled_random_ensemble", "Random ensemble"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=220, sharex=True, sharey=True)
    for ax, (domain, geometry, title) in zip(axes, panels):
        selected = [
            row for row in rows
            if row["domain"] == domain and row["geometry"] == geometry
        ]
        selected.sort(key=lambda row: float(row["entry_x_angle_bin_low_deg"]))
        if not selected:
            ax.axis("off")
            continue
        centers = np.array([
            0.5 * (float(row["entry_x_angle_bin_low_deg"]) + float(row["entry_x_angle_bin_high_deg"]))
            for row in selected
        ])
        no_probability = np.array([float(row["no_dlvo_completion_probability"]) for row in selected])
        unfavorable_probability = np.array([float(row["unfavorable_completion_probability"]) for row in selected])
        ax.plot(centers, no_probability, "o-", color=COLORS["no_dlvo"], label="No-DLVO")
        ax.plot(
            centers,
            unfavorable_probability,
            "o-",
            color=COLORS["unfavorable_50mM"],
            label="Unfavorable 50 mM",
        )
        ax.fill_between(
            centers,
            no_probability,
            unfavorable_probability,
            color="#D7A72D",
            alpha=0.18,
        )
        ax.set_title(title)
        ax.set_xlabel("Absolute Grain X entry angle from FFSZ (deg)")
        ax.grid(axis="y", color="#D7DCE2", lw=0.6)
    axes[0].set_ylabel("Probability of completing distinct Grain Y entry")
    axes[0].legend(frameon=False)
    fig.suptitle("Chemistry-dependent completion after the first collector entry")
    fig.tight_layout()
    path = out_dir / "entry_ffsz_completion_by_x_angle.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["pilot", "confirmatory", "production"], default="pilot")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    stage_dir = output / args.stage
    analysis_dir = stage_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    refs_by_geometry: dict[str, dict[int, np.ndarray]] = {}
    stagnation_rows: list[dict] = []
    refs_by_geometry["center_corner_cell"], periodic_rows = periodic_stagnation(config)
    stagnation_rows.extend(periodic_rows)
    for entry in config["random_geometries"]:
        geometry_dir = stage_dir / "random" / str(entry["name"])
        if geometry_dir.exists():
            refs_by_geometry[str(entry["name"])], random_rows = random_stagnation(entry, config, output)
            stagnation_rows.extend(random_rows)
    write_csv(analysis_dir / "forward_stagnation_points.csv", stagnation_rows)

    window = math.radians(float(config["analysis"]["ffsz_window_degrees"]))
    replicates = int(config["analysis"]["bootstrap_replicates"])
    rng = np.random.default_rng(20260819)
    group_data: dict[tuple[str, str, int, str], dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    for domain, geometry, seed, profile, files in discover_groups(stage_dir):
        data = load_group(files, refs_by_geometry[geometry])
        group_data[(domain, geometry, seed, profile)] = data
        row = {
            "stage": args.stage,
            "domain": domain,
            "geometry": geometry,
            "seed": seed,
            "profile": profile,
            **metrics(data, window),
            **bootstrap_intervals(data, window, replicates, rng),
        }
        rows.append(row)
    write_csv(analysis_dir / "entry_ffsz_seed_geometry_summary.csv", rows)

    aggregate_rows: list[dict] = []
    aggregate_groups: dict[tuple[str, str, str], list[dict[str, np.ndarray]]] = {}
    for (domain, geometry, _seed, profile), data in group_data.items():
        aggregate_groups.setdefault((domain, geometry, profile), []).append(data)
    for (domain, geometry, profile), items in aggregate_groups.items():
        data = pooled_data(items)
        aggregate_rows.append({
            "stage": args.stage,
            "domain": domain,
            "geometry": geometry,
            "profile": profile,
            **metrics(data, window),
            **bootstrap_intervals(data, window, replicates, rng),
        })
    for profile in config["profile_order"]:
        items = [data for (domain, _geometry, _seed, item_profile), data in group_data.items() if domain == "random" and item_profile == profile]
        if items:
            data = pooled_data(items)
            aggregate_rows.append({
                "stage": args.stage,
                "domain": "random",
                "geometry": "pooled_random_ensemble",
                "profile": profile,
                **metrics(data, window),
                **bootstrap_intervals(data, window, replicates, rng),
            })
    write_csv(analysis_dir / "entry_ffsz_aggregate_summary.csv", aggregate_rows)

    matched_rows: list[dict] = []
    matched_keys = sorted({(domain, geometry, seed) for domain, geometry, seed, _profile in group_data})
    for domain, geometry, seed in matched_keys:
        no_key = (domain, geometry, seed, "no_dlvo")
        unfavorable_key = (domain, geometry, seed, "unfavorable_50mM")
        if no_key not in group_data or unfavorable_key not in group_data:
            continue
        matched_rows.append({
            "stage": args.stage,
            "domain": domain,
            "geometry": geometry,
            "seed": seed,
            "comparison_profile": "unfavorable_50mM",
            **matched_chemistry_metrics(
                group_data[no_key],
                group_data[unfavorable_key],
                window,
                replicates,
                rng,
            ),
        })
    aggregate_match_keys = sorted({(domain, geometry) for domain, geometry, _seed, _profile in group_data})
    for domain, geometry in aggregate_match_keys:
        seeds = sorted({seed for item_domain, item_geometry, seed, _profile in group_data if item_domain == domain and item_geometry == geometry})
        no_items = [group_data[(domain, geometry, seed, "no_dlvo")] for seed in seeds if (domain, geometry, seed, "no_dlvo") in group_data and (domain, geometry, seed, "unfavorable_50mM") in group_data]
        unfavorable_items = [group_data[(domain, geometry, seed, "unfavorable_50mM")] for seed in seeds if (domain, geometry, seed, "no_dlvo") in group_data and (domain, geometry, seed, "unfavorable_50mM") in group_data]
        if no_items:
            matched_rows.append({
                "stage": args.stage,
                "domain": domain,
                "geometry": geometry,
                "seed": "pooled_seeds",
                "comparison_profile": "unfavorable_50mM",
                **matched_chemistry_metrics(
                    pooled_for_matching(no_items),
                    pooled_for_matching(unfavorable_items),
                    window,
                    replicates,
                    rng,
                ),
            })
    random_units = sorted({(geometry, seed) for domain, geometry, seed, _profile in group_data if domain == "random"})
    random_no = [group_data[("random", geometry, seed, "no_dlvo")] for geometry, seed in random_units if ("random", geometry, seed, "no_dlvo") in group_data and ("random", geometry, seed, "unfavorable_50mM") in group_data]
    random_unfavorable = [group_data[("random", geometry, seed, "unfavorable_50mM")] for geometry, seed in random_units if ("random", geometry, seed, "no_dlvo") in group_data and ("random", geometry, seed, "unfavorable_50mM") in group_data]
    if random_no:
        matched_rows.append({
            "stage": args.stage,
            "domain": "random",
            "geometry": "pooled_random_ensemble",
            "seed": "pooled_geometries_and_seeds",
            "comparison_profile": "unfavorable_50mM",
            **matched_chemistry_metrics(
                pooled_for_matching(random_no),
                pooled_for_matching(random_unfavorable),
                window,
                replicates,
                rng,
            ),
        })

    high_diffusion_profile = "unfavorable_50mM_100D0"
    for domain, geometry, seed in matched_keys:
        no_key = (domain, geometry, seed, "no_dlvo")
        comparator_key = (domain, geometry, seed, high_diffusion_profile)
        if no_key not in group_data or comparator_key not in group_data:
            continue
        matched_rows.append({
            "stage": args.stage,
            "domain": domain,
            "geometry": geometry,
            "seed": seed,
            "comparison_profile": high_diffusion_profile,
            **matched_chemistry_metrics(
                group_data[no_key],
                group_data[comparator_key],
                window,
                replicates,
                rng,
            ),
        })
    for domain, geometry in aggregate_match_keys:
        seeds = sorted({
            seed for item_domain, item_geometry, seed, _profile in group_data
            if item_domain == domain and item_geometry == geometry
        })
        valid_seeds = [
            seed for seed in seeds
            if (domain, geometry, seed, "no_dlvo") in group_data
            and (domain, geometry, seed, high_diffusion_profile) in group_data
        ]
        if not valid_seeds:
            continue
        matched_rows.append({
            "stage": args.stage,
            "domain": domain,
            "geometry": geometry,
            "seed": "pooled_seeds",
            "comparison_profile": high_diffusion_profile,
            **matched_chemistry_metrics(
                pooled_for_matching([
                    group_data[(domain, geometry, seed, "no_dlvo")]
                    for seed in valid_seeds
                ]),
                pooled_for_matching([
                    group_data[(domain, geometry, seed, high_diffusion_profile)]
                    for seed in valid_seeds
                ]),
                window,
                replicates,
                rng,
            ),
        })
    random_high_diffusion_units = [
        (geometry, seed) for geometry, seed in random_units
        if ("random", geometry, seed, "no_dlvo") in group_data
        and ("random", geometry, seed, high_diffusion_profile) in group_data
    ]
    if random_high_diffusion_units:
        matched_rows.append({
            "stage": args.stage,
            "domain": "random",
            "geometry": "pooled_random_ensemble",
            "seed": "pooled_geometries_and_seeds",
            "comparison_profile": high_diffusion_profile,
            **matched_chemistry_metrics(
                pooled_for_matching([
                    group_data[("random", geometry, seed, "no_dlvo")]
                    for geometry, seed in random_high_diffusion_units
                ]),
                pooled_for_matching([
                    group_data[("random", geometry, seed, high_diffusion_profile)]
                    for geometry, seed in random_high_diffusion_units
                ]),
                window,
                replicates,
                rng,
            ),
        })
    write_csv(analysis_dir / "entry_ffsz_matched_chemistry_summary.csv", matched_rows)

    selection_rows: list[dict] = []
    for domain, geometry, seed in matched_keys:
        no_key = (domain, geometry, seed, "no_dlvo")
        unfavorable_key = (domain, geometry, seed, "unfavorable_50mM")
        if no_key not in group_data or unfavorable_key not in group_data:
            continue
        selection_rows.append({
            "stage": args.stage,
            "domain": domain,
            "geometry": geometry,
            "seed": seed,
            **support_selection_metrics(
                group_data[no_key],
                group_data[unfavorable_key],
                window,
                0,
                rng,
            ),
        })

    completion_rows: list[dict] = []
    bins_deg = np.linspace(0.0, 180.0, 13)
    for domain, geometry in aggregate_match_keys:
        seeds = sorted({
            seed for item_domain, item_geometry, seed, _profile in group_data
            if item_domain == domain and item_geometry == geometry
        })
        valid_seeds = [
            seed for seed in seeds
            if (domain, geometry, seed, "no_dlvo") in group_data
            and (domain, geometry, seed, "unfavorable_50mM") in group_data
        ]
        no_items = [group_data[(domain, geometry, seed, "no_dlvo")] for seed in valid_seeds]
        unfavorable_items = [
            group_data[(domain, geometry, seed, "unfavorable_50mM")]
            for seed in valid_seeds
        ]
        if not no_items:
            continue
        pooled_no = pooled_for_matching(no_items)
        pooled_unfavorable = pooled_for_matching(unfavorable_items)
        selection_rows.append({
            "stage": args.stage,
            "domain": domain,
            "geometry": geometry,
            "seed": "pooled_seeds",
            **support_selection_metrics(
                pooled_no,
                pooled_unfavorable,
                window,
                replicates,
                rng,
            ),
        })
        completion_rows.extend(completion_by_first_entry_angle(
            pooled_no,
            pooled_unfavorable,
            domain,
            geometry,
            bins_deg,
        ))
    if random_no:
        pooled_random_no = pooled_for_matching(random_no)
        pooled_random_unfavorable = pooled_for_matching(random_unfavorable)
        selection_rows.append({
            "stage": args.stage,
            "domain": "random",
            "geometry": "pooled_random_ensemble",
            "seed": "pooled_geometries_and_seeds",
            **support_selection_metrics(
                pooled_random_no,
                pooled_random_unfavorable,
                window,
                replicates,
                rng,
            ),
        })
        completion_rows.extend(completion_by_first_entry_angle(
            pooled_random_no,
            pooled_random_unfavorable,
            "random",
            "pooled_random_ensemble",
            bins_deg,
        ))
    selection_csv = analysis_dir / "entry_ffsz_support_selection_summary.csv"
    completion_csv = analysis_dir / "entry_ffsz_completion_by_x_angle.csv"
    write_csv(selection_csv, selection_rows)
    write_csv(completion_csv, completion_rows)

    geometry_figure = plot_summary(
        [row for row in aggregate_rows if row["geometry"] != "pooled_random_ensemble"],
        analysis_dir,
    )
    kernel_figure = plot_kernels(group_data, analysis_dir)
    matched_figure = plot_matched_chemistry(matched_rows, analysis_dir)
    selection_figure = plot_selection_decomposition(selection_rows, analysis_dir)
    completion_figure = plot_completion_by_first_entry(completion_rows, analysis_dir)
    report = {
        "stage": args.stage,
        "groups": len(group_data),
        "summary_csv": str(analysis_dir / "entry_ffsz_aggregate_summary.csv"),
        "stagnation_csv": str(analysis_dir / "forward_stagnation_points.csv"),
        "geometry_figure": str(geometry_figure),
        "kernel_figure": str(kernel_figure),
        "matched_chemistry_csv": str(analysis_dir / "entry_ffsz_matched_chemistry_summary.csv"),
        "matched_chemistry_figure": str(matched_figure),
        "support_selection_csv": str(selection_csv),
        "support_selection_figure": str(selection_figure),
        "completion_by_x_angle_csv": str(completion_csv),
        "completion_by_x_angle_figure": str(completion_figure),
    }
    report_path = analysis_dir / "entry_ffsz_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
