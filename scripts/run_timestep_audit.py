#!/usr/bin/env python3
"""Audit actual adaptive near-wall time steps for the flagship focusing case."""

from __future__ import annotations

import csv
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_periodic_cell_refinement as pcr  # noqa: E402
from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled  # noqa: E402
from colloid_tsm.physical import load_flow, load_library, save_library  # noqa: E402


FLOW_PATH = ROOT / "outputs" / "openfoam_flow" / "openfoam_flow_N512.npz"
OUT = ROOT / "outputs" / "timestep_audit"
FIGURES = ROOT / "outputs" / "figures"
PROFILE_NAME = "unfavorable_50mM_z50"
DT = 1.0e-3


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def profile() -> dict[str, object]:
    for candidate in pcr.PROFILES:
        if str(candidate["profile"]) == PROFILE_NAME:
            return candidate
    raise KeyError(PROFILE_NAME)


def params_for_audit():
    return replace(pcr.params_for(profile()), dt=DT)


def initial_y() -> np.ndarray:
    return pcr.central_core_y(replace(pcr.params_for(pcr.PROFILES[0]), dt=DT))


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("p05", "p25", "median", "p75", "p95", "max")}
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def summarize_group(name: str, mask: np.ndarray, lib, diag: dict[str, np.ndarray], params) -> dict[str, float | int | str]:
    mask = np.asarray(mask, dtype=bool)
    count = int(np.sum(mask))
    adaptive_count = diag["adaptive_substeps"][mask].astype(float)
    near_outer = diag["near_wall_outer_steps"][mask].astype(float)
    sub_sum = diag["sub_dt_sum"][mask].astype(float)
    sub_sq_sum = diag["sub_dt_sq_sum"][mask].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sub_dt_particle = np.divide(sub_sum, adaptive_count, out=np.full_like(sub_sum, np.nan), where=adaptive_count > 0)
        outer_steps = np.ceil(lib.travel_time[mask] / params.dt)
        near_outer_fraction = np.divide(near_outer, outer_steps, out=np.full_like(near_outer, np.nan), where=outer_steps > 0)
        adaptive_time_fraction = np.divide(sub_sum, lib.travel_time[mask], out=np.full_like(sub_sum, np.nan), where=lib.travel_time[mask] > 0)

    active_sub_mask = mask & (diag["adaptive_substeps"] > 0)
    min_sub = diag["min_sub_dt"][active_sub_mask]
    global_mean_sub = float(np.sum(sub_sum) / np.sum(adaptive_count)) if np.sum(adaptive_count) > 0 else float("nan")
    global_rms_sub = float(np.sqrt(np.sum(sub_sq_sum) / np.sum(adaptive_count))) if np.sum(adaptive_count) > 0 else float("nan")
    q_substeps = quantiles(adaptive_count)
    q_mean_sub = quantiles(mean_sub_dt_particle * 1.0e6)
    q_near_fraction = quantiles(near_outer_fraction)
    q_adaptive_time = quantiles(adaptive_time_fraction)
    return {
        "group": name,
        "particles": count,
        "with_adaptive_substeps": int(np.sum(active_sub_mask)),
        "total_near_wall_outer_steps": int(np.sum(diag["near_wall_outer_steps"][mask])),
        "total_adaptive_substeps": int(np.sum(diag["adaptive_substeps"][mask])),
        "total_guard_hits": int(np.sum(diag["guard_hits"][mask])),
        "total_min_dt_hits": int(np.sum(diag["min_dt_hits"][mask])),
        "global_mean_substep_us": global_mean_sub * 1.0e6,
        "global_rms_substep_us": global_rms_sub * 1.0e6,
        "global_min_substep_us": float(np.nanmin(min_sub) * 1.0e6) if min_sub.size else float("nan"),
        "global_max_substep_us": float(np.nanmax(diag["max_sub_dt"][active_sub_mask]) * 1.0e6) if np.any(active_sub_mask) else float("nan"),
        "adaptive_substeps_median": q_substeps["median"],
        "adaptive_substeps_p95": q_substeps["p95"],
        "adaptive_substeps_max": q_substeps["max"],
        "particle_mean_substep_us_median": q_mean_sub["median"],
        "particle_mean_substep_us_p05": q_mean_sub["p05"],
        "particle_mean_substep_us_p95": q_mean_sub["p95"],
        "near_wall_outer_fraction_median": q_near_fraction["median"],
        "near_wall_outer_fraction_p95": q_near_fraction["p95"],
        "adaptive_time_fraction_median": q_adaptive_time["median"],
        "adaptive_time_fraction_p95": q_adaptive_time["p95"],
        "max_inward_det_normal_step_nm": float(np.nanmax(diag["max_inward_det_normal_step"][mask]) * 1.0e9) if count else float("nan"),
        "max_brownian_normal_std_nm": float(np.nanmax(diag["max_brownian_normal_std"][mask]) * 1.0e9) if count else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "--"
    if digits == 0:
        return str(int(round(x)))
    return f"{x:.{digits}g}"


def make_figure(rows: list[dict[str, float | int | str]], params) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1800, 1040), "white")
    draw = ImageDraw.Draw(image)
    draw.text((900, 45), "Adaptive near-wall timestep audit: OpenFOAM 50 mM, zeta=-50 mV", fill="#111827", font=font(30, True), anchor="mm")
    groups = [row["group"] for row in rows]

    def bar_panel(rect, metric: str, title: str, color: str = "#0f766e") -> None:
        x0, y0, x1, y1 = rect
        draw.rectangle(rect, outline="#d1d5db", width=2)
        draw.text(((x0 + x1) / 2, y0 - 28), title, fill="#111827", font=font(20, True), anchor="mm")
        values = np.array([float(row[metric]) for row in rows], dtype=float)
        ymax = float(np.nanmax(values)) if np.any(np.isfinite(values)) else 1.0
        ymax = max(ymax * 1.15, 1.0)
        for tick in np.linspace(0, ymax, 5):
            y = y1 - tick / ymax * (y1 - y0)
            draw.line((x0, y, x1, y), fill="#e5e7eb", width=1)
            draw.text((x0 - 8, y), fmt(tick), fill="#4b5563", font=font(13), anchor="rm")
        group_w = (x1 - x0) / len(rows)
        bar_w = min(80.0, group_w * 0.42)
        for i, row in enumerate(rows):
            value = float(row[metric])
            cx = x0 + (i + 0.5) * group_w
            by = y1 - value / ymax * (y1 - y0) if np.isfinite(value) else y1
            draw.rectangle((cx - bar_w / 2, by, cx + bar_w / 2, y1), fill=color)
            draw.text((cx, by - 8), fmt(value), fill="#111827", font=font(13), anchor="mb")
            label_lines = str(row["group"]).replace("_", "\n").splitlines()
            for line_id, line in enumerate(label_lines):
                draw.text((cx, y1 + 18 + 15 * line_id), line, fill="#111827", font=font(13), anchor="mt")

    bar_panel((140, 140, 650, 475), "adaptive_substeps_median", "Median adaptive substeps per particle")
    bar_panel((800, 140, 1310, 475), "particle_mean_substep_us_median", "Median particle-mean substep (us)", "#2563eb")
    bar_panel((140, 620, 650, 935), "max_inward_det_normal_step_nm", "Largest inward deterministic normal step (nm)", "#7c3aed")
    bar_panel((800, 620, 1310, 935), "max_brownian_normal_std_nm", "Largest Brownian normal std (nm)", "#c2410c")

    x0, y0, x1, y1 = (1410, 165, 1730, 435)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.text(((x0 + x1) / 2, y0 + 30), "Controls", fill="#111827", font=font(19, True), anchor="mm")
    center = next(row for row in rows if row["group"] == "center_well_release")
    min_dt_fraction = (
        float(center["total_min_dt_hits"]) / float(center["total_adaptive_substeps"])
        if float(center["total_adaptive_substeps"]) > 0.0
        else float("nan")
    )
    lines = [
        f"outer dt: {params.dt * 1e3:.1f} ms",
        f"base cap: {params.dt / params.resolved_langevin_substeps * 1e6:.1f} us",
        f"target normal step: {params.resolved_langevin_normal_step * 1e9:.1f} nm",
        f"min dt hits: {fmt(center['total_min_dt_hits'], 0)}",
        f"guard hits: {fmt(center['total_guard_hits'], 0)}",
        f"center releases: {fmt(center['particles'], 0)}",
    ]
    for i, line in enumerate(lines):
        draw.text((x0 + 20, y0 + 70 + i * 30), line, fill="#111827", font=font(15), anchor="lm")

    draw.text((900, 1000), "Near-wall adaptive subcycling is active only inside the 75 nm gap cutoff; all reported particles are from the same compiled kernel used for production.", fill="#4b5563", font=font(16), anchor="mm")
    image.save(OUT / "timestep_audit.png")
    image.save(FIGURES / "fig25_timestep_audit.png")


def write_report(rows: list[dict[str, float | int | str]], lib, diag: dict[str, np.ndarray], params) -> None:
    center = next(row for row in rows if row["group"] == "center_well_release")
    min_dt_fraction = (
        float(center["total_min_dt_hits"]) / float(center["total_adaptive_substeps"])
        if float(center["total_adaptive_substeps"]) > 0.0
        else float("nan")
    )
    lines = [
        "# Adaptive near-wall timestep audit",
        "",
        "Flagship case: OpenFOAM-derived raster flow, 50 mM unfavorable chemistry, collector zeta -50 mV, central-core injection, dt=1 ms outer step, 600 s horizon.",
        "",
        "## Summary",
        "",
        "| group | particles | with adaptive substeps | total adaptive substeps | median substeps/particle | median mean substep (us) | guard hits | min-dt hits | max det normal step (nm) | max Brownian normal std (nm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    fmt(row["particles"], 0),
                    fmt(row["with_adaptive_substeps"], 0),
                    fmt(row["total_adaptive_substeps"], 0),
                    fmt(row["adaptive_substeps_median"]),
                    fmt(row["particle_mean_substep_us_median"]),
                    fmt(row["total_guard_hits"], 0),
                    fmt(row["total_min_dt_hits"], 0),
                    fmt(row["max_inward_det_normal_step_nm"]),
                    fmt(row["max_brownian_normal_std_nm"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The production `dt=1 ms` is an outer maximum step. Inside the {params.resolved_langevin_substep_cutoff * 1e9:.0f} nm near-wall region, the base substep cap is {params.dt / params.resolved_langevin_substeps * 1e6:.1f} us and the adaptive normal Brownian target is {params.resolved_langevin_normal_step * 1e9:.1f} nm.",
            f"For the center-well released particles, the median adaptive substep count is {fmt(center['adaptive_substeps_median'])} and the median particle-mean substep is {fmt(center['particle_mean_substep_us_median'])} us.",
            f"The audit found {fmt(center['total_guard_hits'], 0)} guard hits and {fmt(center['total_min_dt_hits'], 0)} minimum-dt-floor hits among center-well released particles; the latter are {fmt(100.0 * min_dt_fraction)}% of adaptive substeps.",
            f"The largest predicted inward deterministic normal displacement among center-well released particles was {fmt(center['max_inward_det_normal_step_nm'])} nm, and the largest Brownian normal standard deviation was {fmt(center['max_brownian_normal_std_nm'])} nm.",
            "",
        ]
    )
    (OUT / "timestep_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="Reuse saved audit trajectory and diagnostic arrays.")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not FLOW_PATH.exists():
        raise FileNotFoundError(FLOW_PATH)

    params = params_for_audit()
    library_path = OUT / "trajectory_library_timestep_audit.npz"
    diagnostics_path = OUT / "timestep_diagnostics.npz"
    if args.reuse and library_path.exists() and diagnostics_path.exists():
        lib = load_library(library_path, params=params)
        loaded = np.load(diagnostics_path)
        diag = {name: loaded[name] for name in loaded.files}
    else:
        flow = load_flow(FLOW_PATH, params=params)
        build_kernel(force=True)
        print("Running compiled timestep audit for OpenFOAM 50 mM zeta=-50 mV dt=1 ms", flush=True)
        lib, diag = simulate_cell_transitions_compiled(
            flow,
            str(profile()["condition"]),
            seed=pcr.SEED + 10 + 2,
            allow_attachment=False,
            initial_y=initial_y(),
            surface_mode="resolved_langevin",
            force_rebuild=False,
            return_timestep_diagnostics=True,
        )
        save_library(library_path, lib)
        np.savez_compressed(diagnostics_path, **diag)

    mobile = lib.exited & (~lib.attached)
    center_near_release = mobile & (lib.center_interceptions > 0) & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
    center_well_release = (
        mobile
        & (lib.center_well_interceptions > 0)
        & (lib.collector_well_exit == 1)
        & np.isfinite(lib.theta_well_exit)
    )
    rows = [
        summarize_group("all_particles", np.ones(lib.y_in.size, dtype=bool), lib, diag, params),
        summarize_group("near_wall_any", diag["adaptive_substeps"] > 0, lib, diag, params),
        summarize_group("center_near_release", center_near_release, lib, diag, params),
        summarize_group("center_well_release", center_well_release, lib, diag, params),
    ]
    write_csv(OUT / "timestep_audit_summary.csv", rows)
    make_figure(rows, params)
    write_report(rows, lib, diag, params)
    print(OUT / "timestep_audit_report.md", flush=True)


if __name__ == "__main__":
    main()
