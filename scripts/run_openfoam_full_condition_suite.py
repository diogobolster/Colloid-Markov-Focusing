#!/usr/bin/env python3
"""Run the OpenFOAM-based production suite across the full condition set."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_compiled_parameter_suite as cps  # noqa: E402
import run_focusing_transition_matrices as ftm  # noqa: E402
import run_periodic_cell_refinement as pcr  # noqa: E402
from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled  # noqa: E402
from colloid_tsm.physical import PhysicalParams, TrajectoryLibrary, load_flow, load_library, save_library  # noqa: E402


FLOW_PATH = ROOT / "outputs" / "openfoam_flow" / "openfoam_flow_N512.npz"
OUT = ROOT / "outputs" / "openfoam_full_suite"
FIGURES = ROOT / "outputs" / "figures"
VELOCITY_M_PER_DAY = 4.0
DEFAULT_DT_MS = 1.0
CORE_PARTICLES = 30_000
CORE_HALF_WIDTH_M = 12.5e-6
SEED_REFINEMENT = 2_610_000
SEED_TRANSITION = 2_720_000

PROFILE_LABELS = {
    "neutral_resolved": "No DLVO",
    "unfavorable_6mM_z70": "6 mM, -70 mV",
    "unfavorable_20mM_z70": "20 mM, -70 mV",
    "unfavorable_50mM_z70": "50 mM, -70 mV",
    "unfavorable_100mM_z70": "100 mM, -70 mV",
    "unfavorable_50mM_z50": "50 mM, -50 mV",
    "unfavorable_50mM_z30": "50 mM, -30 mV",
    "unfavorable_100mM_z30": "100 mM, -30 mV",
    "mechanism_50mM_z20_A10x": "50 mM, -20 mV, 10x A",
    "mechanism_100mM_z20_A10x": "100 mM, -20 mV, 10x A",
    "neutral_100xD": "No DLVO, 100x D",
    "unfavorable_50mM_z70_100xD": "50 mM, -70 mV, 100x D",
}

REFINEMENT_HORIZON_S = {
    "neutral_resolved": 240.0,
    "neutral_100xD": 240.0,
    "unfavorable_6mM_z70": 240.0,
    "unfavorable_20mM_z70": 600.0,
    "unfavorable_50mM_z70": 600.0,
    "unfavorable_50mM_z50": 600.0,
    "unfavorable_50mM_z30": 600.0,
    "unfavorable_100mM_z70": 2400.0,
    "unfavorable_100mM_z30": 2400.0,
    "mechanism_50mM_z20_A10x": 2400.0,
    "mechanism_100mM_z20_A10x": 3600.0,
    "unfavorable_50mM_z70_100xD": 240.0,
}

TRANSITION_HORIZON_S = {
    **REFINEMENT_HORIZON_S,
    "unfavorable_50mM_z30": 1200.0,
}

COMPATIBLE_REUSE_DIRS = (
    ROOT / "outputs" / "openfoam_focusing_dt1ms",
    ROOT / "outputs" / "openfoam_focusing",
)


def normalize_profile(profile: dict[str, object]) -> dict[str, object]:
    name = str(profile["profile"])
    return {
        "profile": name,
        "label": PROFILE_LABELS.get(name, name),
        "regime": str(profile["regime"]),
        "condition": str(profile["condition"]),
        "description": str(profile["description"]),
        "updates": dict(profile["updates"]),
    }


PROFILES = tuple(normalize_profile(profile) for profile in cps.PROFILES)


def selected_profiles(names: list[str] | None) -> tuple[dict[str, object], ...]:
    if not names:
        return PROFILES
    by_name = {str(profile["profile"]): profile for profile in PROFILES}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(f"Unknown profile(s): {', '.join(missing)}")
    return tuple(by_name[name] for name in names)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "nan"
    if digits == 0:
        return str(int(round(x)))
    if abs(x) >= 1.0e4 or (abs(x) < 1.0e-2 and x != 0.0):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def params_for(profile: dict[str, object], horizon_s: float, dt_ms: float) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=horizon_s,
        dt=dt_ms * 1.0e-3,
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
        **dict(profile["updates"]),
    )


def central_core_y(params: PhysicalParams) -> np.ndarray:
    rng = np.random.default_rng(SEED_REFINEMENT)
    lo = 0.5 * params.cell_length - CORE_HALF_WIDTH_M
    hi = 0.5 * params.cell_length + CORE_HALF_WIDTH_M
    return rng.uniform(lo, hi, size=CORE_PARTICLES)


def profile_context(profile: dict[str, object], params: PhysicalParams, dt_ms: float) -> dict[str, float | int | str]:
    return {
        "regime": str(profile["regime"]),
        "description": str(profile["description"]),
        "dt_ms": dt_ms,
        "ionic_strength_mM": params.ionic_strength_molar * 1.0e3,
        "zeta_collector_unfavorable_mV": params.zeta_collector_unfavorable * 1.0e3,
        "hamaker_J": params.hamaker,
        "diffusivity_multiplier": params.diffusivity_multiplier,
        "diffusivity_m2_s": params.diffusivity,
        "peclet": params.particle_peclet,
        "debye_length_nm": params.debye_length * 1.0e9,
    }


def library_compatible(lib, params: PhysicalParams, dt_ms: float) -> bool:
    return (
        abs(float(lib.params.max_time) - float(params.max_time)) < 1.0e-9
        and abs(float(lib.params.dt) - dt_ms * 1.0e-3) < 1.0e-12
        and abs(float(lib.params.mean_velocity) - float(params.mean_velocity)) < 1.0e-15
    )


def compatible_source(profile_name: str, ensemble: str, params: PhysicalParams, dt_ms: float):
    file_name = f"trajectory_library_{ensemble}_{profile_name}.npz"
    for base in COMPATIBLE_REUSE_DIRS:
        path = base / file_name
        if not path.exists():
            continue
        lib = load_library(path)
        if library_compatible(lib, params, dt_ms):
            return lib
    return None


def concatenate_libraries(libraries: list[TrajectoryLibrary]) -> TrajectoryLibrary:
    if not libraries:
        raise ValueError("No libraries to concatenate")
    payload: dict[str, object] = {}
    for field in fields(TrajectoryLibrary):
        values = [getattr(library, field.name) for library in libraries]
        first = values[0]
        if isinstance(first, np.ndarray):
            payload[field.name] = np.concatenate(values)
        elif first is None:
            payload[field.name] = None
        else:
            payload[field.name] = first
    return TrajectoryLibrary(**payload)


def chunk_path(profile_name: str, ensemble: str, chunk_id: int, n_chunks: int) -> Path:
    return OUT / "chunks" / f"trajectory_library_{ensemble}_{profile_name}_chunk{chunk_id:03d}of{n_chunks:03d}.npz"


def simulate_chunk_worker(
    flow_path: str,
    profile: dict[str, object],
    params: PhysicalParams,
    initial_y: np.ndarray,
    seed: int,
    out_path: str,
) -> str:
    flow = load_flow(Path(flow_path), params=params)
    lib = simulate_cell_transitions_compiled(
        flow,
        str(profile["condition"]),
        seed=seed,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )
    save_library(Path(out_path), lib)
    return out_path


def run_chunked_or_load(
    profile: dict[str, object],
    ensemble: str,
    params: PhysicalParams,
    initial_y: np.ndarray,
    seed: int,
    reuse: bool,
    dt_ms: float,
    chunk_size: int,
    workers: int,
):
    name = str(profile["profile"])
    full_path = OUT / f"trajectory_library_{ensemble}_{name}.npz"
    if reuse and full_path.exists():
        lib = load_library(full_path)
        if library_compatible(lib, params, dt_ms):
            return lib
    if reuse:
        lib = compatible_source(name, ensemble, params, dt_ms)
        if lib is not None:
            save_library(full_path, lib)
            return lib
    if chunk_size <= 0 or initial_y.size <= chunk_size:
        flow = load_flow(FLOW_PATH, params=params)
        lib = simulate_cell_transitions_compiled(
            flow,
            str(profile["condition"]),
            seed=seed,
            allow_attachment=False,
            initial_y=initial_y,
            surface_mode="resolved_langevin",
            force_rebuild=False,
        )
        save_library(full_path, lib)
        return lib

    chunks = [np.ascontiguousarray(part, dtype=np.float64) for part in np.array_split(initial_y, int(math.ceil(initial_y.size / chunk_size)))]
    n_chunks = len(chunks)
    chunk_paths = [chunk_path(name, ensemble, idx, n_chunks) for idx in range(n_chunks)]
    for path in chunk_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    libraries: list[TrajectoryLibrary | None] = [None] * n_chunks
    for idx, path in enumerate(chunk_paths):
        if reuse and path.exists():
            lib = load_library(path)
            if library_compatible(lib, params, dt_ms) and lib.y_in.size == chunks[idx].size:
                libraries[idx] = lib
                continue
        missing.append(idx)

    if missing:
        print(
            f"  checkpointed {ensemble} {name}: {n_chunks} chunks, {len(missing)} missing, workers={max(workers, 1)}",
            flush=True,
        )
    if missing and workers <= 1:
        for idx in missing:
            simulate_chunk_worker(
                str(FLOW_PATH),
                profile,
                params,
                chunks[idx],
                seed + 1_000_003 * idx,
                str(chunk_paths[idx]),
            )
            libraries[idx] = load_library(chunk_paths[idx])
            print(f"  completed chunk {idx + 1}/{n_chunks} for {ensemble} {name}", flush=True)
    elif missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(
                    simulate_chunk_worker,
                    str(FLOW_PATH),
                    profile,
                    params,
                    chunks[idx],
                    seed + 1_000_003 * idx,
                    str(chunk_paths[idx]),
                ): idx
                for idx in missing
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                path = Path(future.result())
                libraries[idx] = load_library(path)
                print(f"  completed chunk {idx + 1}/{n_chunks} for {ensemble} {name}", flush=True)

    for idx, library in enumerate(libraries):
        if library is None:
            libraries[idx] = load_library(chunk_paths[idx])
    lib = concatenate_libraries([library for library in libraries if library is not None])
    save_library(full_path, lib)
    return lib


def run_or_load(
    profile: dict[str, object],
    ensemble: str,
    params: PhysicalParams,
    initial_y: np.ndarray,
    seed: int,
    reuse: bool,
    dt_ms: float,
    chunk_size: int = 0,
    workers: int = 1,
):
    return run_chunked_or_load(
        profile,
        ensemble,
        params,
        initial_y,
        seed,
        reuse,
        dt_ms,
        chunk_size,
        workers,
    )


def enrich_refinement_row(row: dict[str, float | int | str], profile: dict[str, object], params: PhysicalParams, dt_ms: float) -> dict[str, float | int | str]:
    enriched = dict(row)
    enriched.update(profile_context(profile, params, dt_ms))
    return enriched


def enrich_transition_row(row: dict[str, float | int | str], profile: dict[str, object], params: PhysicalParams, dt_ms: float) -> dict[str, float | int | str]:
    enriched = dict(row)
    enriched["max_time_s"] = params.max_time
    enriched.update(profile_context(profile, params, dt_ms))
    return enriched


def heat_color(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    if not np.isfinite(value):
        return "#f3f4f6"
    alpha = min(max((float(value) - vmin) / max(vmax - vmin, 1.0e-12), 0.0), 1.0)
    lo = np.array([240, 253, 250], dtype=float)
    hi = np.array([15, 118, 110], dtype=float)
    rgb = np.round(lo + alpha * (hi - lo)).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def compact_label(row: dict[str, object]) -> str:
    return str(row["label"]).replace(", ", "\n")


def finite_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, float | int | str]],
    key: str,
    rect: tuple[int, int, int, int],
    title: str,
    limit: float | None = None,
    log_scale: bool = False,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="#d1d5db", width=2)
    draw.text(((x0 + x1) / 2, y0 - 26), title, fill="#111827", font=font(18, True), anchor="mm")
    values = np.array([finite_float(row, key) for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    if log_scale:
        finite = finite[finite > 0.0]
        vmax = math.ceil(math.log10(float(np.max(finite)))) if finite.size else 1.0
        vmin = min(0.0, math.floor(math.log10(float(np.min(finite))))) if finite.size else 0.0
        span = max(vmax - vmin, 1.0)
    else:
        vmax = limit if limit is not None else (float(np.max(finite)) * 1.15 if finite.size else 1.0)
        vmax = max(vmax, 1.0e-12)
        vmin = 0.0
        span = vmax
    for tick in np.linspace(0.0, 1.0, 5):
        y = y1 - tick * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#e5e7eb", width=1)
        label = f"{10 ** (vmin + tick * span):.0g}" if log_scale else f"{tick * vmax:.2g}"
        draw.text((x0 - 8, y), label, fill="#4b5563", font=font(11), anchor="rm")
    colors = {
        "control": "#2563eb",
        "realistic": "#059669",
        "realistic_high_salt": "#0f766e",
        "upper_salt_screen": "#0e7490",
        "moderate_zeta": "#65a30d",
        "weak_zeta": "#84cc16",
        "weak_zeta_high_salt": "#a3e635",
        "mechanism_amplified": "#7c3aed",
        "high_diffusion_control": "#9333ea",
        "high_diffusion_unfavorable": "#c026d3",
    }
    n = len(rows)
    group_w = (x1 - x0) / max(n, 1)
    bar_w = group_w * 0.62
    for i, (row, value) in enumerate(zip(rows, values)):
        if not np.isfinite(value):
            continue
        if log_scale:
            if value <= 0.0:
                height_fraction = 0.0
            else:
                height_fraction = (math.log10(value) - vmin) / span
        else:
            height_fraction = value / vmax
        height_fraction = min(max(height_fraction, 0.0), 1.0)
        cx = x0 + (i + 0.5) * group_w
        y = y1 - height_fraction * (y1 - y0)
        draw.rectangle((cx - bar_w / 2, y, cx + bar_w / 2, y1), fill=colors.get(str(row["regime"]), "#4b5563"))
        draw.text((cx, y - 7), fmt(value, 2), fill="#111827", font=font(10), anchor="mb")


def make_summary_figure(refinement_rows: list[dict[str, float | int | str]], prefix: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2400, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((1200, 45), "OpenFOAM full condition suite", fill="#111827", font=font(34, True), anchor="mm")
    draw.text(
        (1200, 83),
        "Central-core injection; resolved Langevin DLVO tracking; homogeneous unfavorable surfaces do not attach",
        fill="#374151",
        font=font(17),
        anchor="mm",
    )
    panels = (
        (
            "center_well_release_theta30_fraction",
            (155, 155, 2250, 405),
            "F30 for center-grain secondary-minimum releases",
            1.0,
            False,
        ),
        (
            "center_well_release_median_theta_deg",
            (155, 535, 2250, 785),
            "Median well-release angle from downstream stagnation (deg)",
            90.0,
            False,
        ),
        (
            "center_well_time_median_s",
            (155, 915, 2250, 1165),
            "Median secondary-minimum residence time (s, log scale)",
            None,
            True,
        ),
    )
    for key, rect, title, limit, log_scale in panels:
        draw_bar_panel(draw, refinement_rows, key, rect, title, limit=limit, log_scale=log_scale)
    group_w = (2250 - 155) / max(len(refinement_rows), 1)
    for i, row in enumerate(refinement_rows):
        draw.text((155 + (i + 0.5) * group_w, 1218), compact_label(row), fill="#111827", font=font(12), anchor="ma")
    draw.text(
        (155, 1335),
        "Undefined well metrics indicate either no DLVO well or no completed center-well releases; residual censors are reported separately as non physical trapping/attachment.",
        fill="#374151",
        font=font(16),
        anchor="lm",
    )
    path = OUT / "openfoam_full_suite_summary.png"
    image.save(path)
    image.save(FIGURES / f"{prefix}_summary.png")


def make_transition_center_figure(payloads: dict[str, dict[str, np.ndarray]], profiles: tuple[dict[str, object], ...], prefix: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    angle_labels = [
        f"{ftm.ANGLE_EDGES_DEG[i]:.0f}-{ftm.ANGLE_EDGES_DEG[i + 1]:.0f}"
        for i in range(ftm.ANGLE_EDGES_DEG.size - 1)
    ]
    center_state = int(np.where(np.array(ftm.STATE_NAMES) == "center core")[0][0])
    matrix = []
    for profile in profiles:
        name = str(profile["profile"])
        payload = payloads[name]
        if str(profile["condition"]) == "neutral":
            matrix.append(payload["center_near_angle_probability"][center_state])
        else:
            matrix.append(payload["center_well_angle_probability"][center_state])
    data = np.array(matrix, dtype=float)
    cell_w = 150
    cell_h = 64
    left = 520
    top = 150
    image = Image.new("RGB", (1640, 1080), "white")
    draw = ImageDraw.Draw(image)
    draw.text((820, 45), "Center-core focusing transition rows", fill="#111827", font=font(32, True), anchor="mm")
    draw.text(
        (820, 82),
        "Rows are full-suite conditions; columns are release-angle bins from the downstream stagnation point",
        fill="#374151",
        font=font(16),
        anchor="mm",
    )
    for j, label in enumerate(angle_labels):
        draw.text((left + (j + 0.5) * cell_w, top - 22), label, fill="#111827", font=font(13, True), anchor="mm")
    for i, profile in enumerate(profiles):
        y = top + i * cell_h
        draw.text((left - 18, y + cell_h / 2), PROFILE_LABELS[str(profile["profile"])], fill="#111827", font=font(13), anchor="rm")
        for j in range(data.shape[1]):
            x = left + j * cell_w
            value = data[i, j]
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=heat_color(value), outline="#d1d5db")
            if np.isfinite(value):
                fill = "white" if value >= 0.5 else "#111827"
                draw.text((x + cell_w / 2, y + cell_h / 2), f"{value:.2f}", fill=fill, font=font(13), anchor="mm")
            else:
                draw.text((x + cell_w / 2, y + cell_h / 2), "n/a", fill="#6b7280", font=font(12), anchor="mm")
    draw.text((820, 1018), "Neutral rows use center near-surface release; unfavorable rows use center secondary-minimum release.", fill="#374151", font=font(15), anchor="mm")
    path = OUT / "openfoam_full_suite_transition_center_rows.png"
    image.save(path)
    image.save(FIGURES / f"{prefix}_transition_center_rows.png")


def center_transition_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    return [row for row in rows if row["state"] == "center core"]


def write_report(
    refinement_rows: list[dict[str, float | int | str]],
    transition_rows: list[dict[str, float | int | str]],
    prefix: str,
) -> None:
    central_transition = center_transition_rows(transition_rows)
    lines = [
        "# OpenFOAM full condition suite",
        "",
        f"All runs use `{FLOW_PATH.relative_to(ROOT)}`, a 512 x 512 raster of the body-fitted OpenFOAM solution in the center/corner-grain periodic cell. The outer SDE step is {DEFAULT_DT_MS:g} ms unless overridden at run time; near-wall portions are adaptively subcycled by the compiled resolved-Langevin kernel. Homogeneous unfavorable cases are nonattaching.",
        "",
        "## Central-core refinement",
        "",
        "| profile | regime | I (mM) | zeta_c (mV) | D mult | horizon (s) | status | center well entries | releases | unresolved | F30 | +/-95% | median theta (deg) | median residence (s) | cumulative travel (deg) |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in refinement_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["regime"]),
                    fmt(row["ionic_strength_mM"], 2),
                    fmt(row["zeta_collector_unfavorable_mV"], 1),
                    fmt(row["diffusivity_multiplier"], 1),
                    fmt(row["max_time_s"], 0),
                    str(row["final_status"]),
                    fmt(row["center_well_intercepted_count"], 0),
                    fmt(row["center_well_release_count"], 0),
                    fmt(row["center_well_censored_count"], 0),
                    fmt(row["center_well_release_theta30_fraction"]),
                    fmt(row["center_well_release_theta30_halfwidth_95"]),
                    fmt(row["center_well_release_median_theta_deg"]),
                    fmt(row["center_well_time_median_s"]),
                    fmt(row["center_well_cumulative_travel_median_deg"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Center-core transition row",
            "",
            "The full seven-state transition payloads are saved as NPZ files. This table extracts the center-core row because it is the row expected to carry the central-grain focusing mechanism.",
            "",
            "| profile | horizon (s) | mobile exits | unresolved | center near rel | near F30 | near median theta (deg) | center well rel | well F30 | well median theta (deg) | P(yout center | well) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in central_transition:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt(row["max_time_s"], 0),
                    fmt(row["mobile_exits"], 0),
                    fmt(row["censored"], 0),
                    fmt(row["center_near_releases"], 0),
                    fmt(row["center_near_theta30_fraction"]),
                    fmt(row["center_near_median_theta_deg"]),
                    fmt(row["center_well_releases"], 0),
                    fmt(row["center_well_theta30_fraction"]),
                    fmt(row["center_well_median_theta_deg"]),
                    fmt(row["center_well_yout_center_fraction"]),
                ]
            )
            + " |"
        )
    clean = [
        row
        for row in refinement_rows
        if finite_float(row, "center_well_release_theta30_fraction") == finite_float(row, "center_well_release_theta30_fraction")
        and finite_float(row, "center_well_censored_count") == 0.0
    ]
    clean.sort(key=lambda row: finite_float(row, "center_well_release_theta30_fraction"), reverse=True)
    lines.extend(["", "## Quick read", ""])
    if clean:
        for row in clean[:5]:
            lines.append(
                f"- {row['label']}: F30={fmt(row['center_well_release_theta30_fraction'])}, "
                f"median theta={fmt(row['center_well_release_median_theta_deg'])} deg, "
                f"median residence={fmt(row['center_well_time_median_s'])} s, "
                f"releases={fmt(row['center_well_release_count'], 0)}."
            )
    lines.extend(
        [
            "",
            f"Diagnostic figures: `outputs/figures/{prefix}_summary.png` and `outputs/figures/{prefix}_transition_center_rows.png`.",
            "",
        ]
    )
    (OUT / "openfoam_full_suite_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_figures_and_report(profiles: tuple[dict[str, object], ...], prefix: str) -> None:
    refinement_rows = read_csv(OUT / "openfoam_full_refinement_summary.csv")
    transition_rows = read_csv(OUT / "openfoam_full_transition_rows.csv")
    if not refinement_rows or not transition_rows:
        raise FileNotFoundError("Full-suite CSV outputs are missing; run simulations before --figures-only.")
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for profile in profiles:
        path = OUT / f"openfoam_full_transition_payload_{profile['profile']}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            payloads[str(profile["profile"])] = {key: data[key] for key in data.files}
    make_summary_figure(refinement_rows, prefix)
    make_transition_center_figure(payloads, profiles, prefix)
    write_report(refinement_rows, transition_rows, prefix)


def main() -> int:
    global OUT, DEFAULT_DT_MS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="Reuse compatible trajectory libraries.")
    parser.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS, help="Outer SDE time step in milliseconds.")
    parser.add_argument("--out-name", default="openfoam_full_suite", help="Output directory name under outputs/.")
    parser.add_argument("--figure-prefix", default="fig26_openfoam_full_suite", help="Diagnostic figure filename prefix.")
    parser.add_argument("--profiles", nargs="*", default=None, help="Optional profile-name subset.")
    parser.add_argument("--skip-refinement", action="store_true", help="Skip central-core refinement runs.")
    parser.add_argument("--skip-transition", action="store_true", help="Skip seven-state transition runs.")
    parser.add_argument("--figures-only", action="store_true", help="Rebuild report and figures from existing CSV/NPZ outputs.")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Particles per checkpointed chunk for missing libraries; <=0 disables chunking.")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes for checkpointed chunks.")
    args = parser.parse_args()

    DEFAULT_DT_MS = float(args.dt_ms)
    OUT = ROOT / "outputs" / str(args.out_name)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    profiles = selected_profiles(args.profiles)

    if args.figures_only:
        build_figures_and_report(profiles, str(args.figure_prefix))
        print(OUT / "openfoam_full_suite_report.md", flush=True)
        return 0

    if not FLOW_PATH.exists():
        raise FileNotFoundError(f"Create the OpenFOAM raster first: {FLOW_PATH}")
    build_kernel(force=False)

    refinement_rows: list[dict[str, float | int | str]] = []
    transition_rows: list[dict[str, float | int | str]] = []
    payloads: dict[str, dict[str, np.ndarray]] = {}

    if not args.skip_transition:
        ref_params = params_for(profiles[0], TRANSITION_HORIZON_S[str(profiles[0]["profile"])], args.dt_ms)
        transition_y = ftm.sample_y(ref_params, SEED_TRANSITION)
        for idx, profile in enumerate(profiles):
            name = str(profile["profile"])
            params = params_for(profile, TRANSITION_HORIZON_S[name], args.dt_ms)
            print(f"Transition {name}: {transition_y.size} particles, horizon {params.max_time:g} s", flush=True)
            lib = run_or_load(
                profile,
                "transition",
                params,
                transition_y,
                SEED_TRANSITION + 100 * idx,
                bool(args.reuse),
                args.dt_ms,
                int(args.chunk_size),
                max(int(args.workers), 1),
            )
            payload = ftm.build_transition_payload(lib, params, profile)
            payloads[name] = payload
            for row in ftm.summarize_rows(lib, params, profile):
                transition_rows.append(enrich_transition_row(row, profile, params, args.dt_ms))
            np.savez_compressed(OUT / f"openfoam_full_transition_payload_{name}.npz", **payload)
            write_csv(OUT / "openfoam_full_transition_rows_partial.csv", transition_rows)

    if not args.skip_refinement:
        ref_params = params_for(profiles[0], REFINEMENT_HORIZON_S[str(profiles[0]["profile"])], args.dt_ms)
        refinement_y = central_core_y(ref_params)
        for idx, profile in enumerate(profiles):
            name = str(profile["profile"])
            params = params_for(profile, REFINEMENT_HORIZON_S[name], args.dt_ms)
            print(f"Refinement {name}: {refinement_y.size} particles, horizon {params.max_time:g} s", flush=True)
            lib = run_or_load(
                profile,
                "refinement",
                params,
                refinement_y,
                SEED_REFINEMENT + 100 * idx,
                bool(args.reuse),
                args.dt_ms,
                int(args.chunk_size),
                max(int(args.workers), 1),
            )
            refinement_rows.append(enrich_refinement_row(pcr.summarize_library(profile, params, lib), profile, params, args.dt_ms))
            write_csv(OUT / "openfoam_full_refinement_summary_partial.csv", refinement_rows)

    if transition_rows:
        write_csv(OUT / "openfoam_full_transition_rows.csv", transition_rows)
    if refinement_rows:
        write_csv(OUT / "openfoam_full_refinement_summary.csv", refinement_rows)
    if transition_rows and refinement_rows:
        make_summary_figure(refinement_rows, str(args.figure_prefix))
        make_transition_center_figure(payloads, profiles, str(args.figure_prefix))
        write_report(refinement_rows, transition_rows, str(args.figure_prefix))
        print(OUT / "openfoam_full_suite_report.md", flush=True)
    else:
        print(OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
