#!/usr/bin/env python3
"""Rerun focused periodic-cell DLVO cases with the OpenFOAM-derived flow."""

from __future__ import annotations

import csv
import sys
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_focusing_transition_matrices as ftm  # noqa: E402
import run_periodic_cell_refinement as pcr  # noqa: E402
from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled  # noqa: E402
from colloid_tsm.physical import angular_distance, load_flow, load_library, save_library  # noqa: E402


FLOW_PATH = ROOT / "outputs" / "openfoam_flow" / "openfoam_flow_N512.npz"
OUT = ROOT / "outputs" / "openfoam_focusing"
FIGURES = ROOT / "outputs" / "figures"
DT_OVERRIDE: float | None = None
RUN_LABEL = "OpenFOAM-derived"
COMPARISON_FIGURE_NAME = "fig21_openfoam_focusing_comparison.png"
MATRICES_FIGURE_NAME = "fig22_openfoam_focusing_transition_matrices.png"

PROFILES = tuple(ftm.PROFILES)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
        return "nan"
    if digits == 0:
        return str(int(round(x)))
    return f"{x:.{digits}g}"


def profile_name(profile: dict[str, object]) -> str:
    return str(profile["profile"])


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def heat_color(value: float) -> str:
    if not np.isfinite(value):
        return "#f3f4f6"
    value = min(max(float(value), 0.0), 1.0)
    lo = np.array([240, 253, 250], dtype=float)
    hi = np.array([15, 118, 110], dtype=float)
    rgb = np.round(lo + value * (hi - lo)).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def draw_heatmap(
    draw: ImageDraw.ImageDraw,
    matrix: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    title: str,
    x_labels: list[str],
    y_labels: list[str] | None = None,
) -> None:
    title_font = font(15, True)
    tick_font = font(11)
    rows, cols = matrix.shape
    draw.text((x0 + 0.5 * cols * cell, y0 - 28), title, fill="#111827", font=title_font, anchor="mm")
    for i in range(rows):
        for j in range(cols):
            left = x0 + j * cell
            top = y0 + i * cell
            draw.rectangle((left, top, left + cell, top + cell), fill=heat_color(matrix[i, j]), outline="#d1d5db")
            if np.isfinite(matrix[i, j]) and matrix[i, j] >= 0.5:
                draw.text((left + 0.5 * cell, top + 0.5 * cell), f"{matrix[i, j]:.2f}", fill="white", font=tick_font, anchor="mm")
    for j, label in enumerate(x_labels):
        draw.text((x0 + (j + 0.5) * cell, y0 + rows * cell + 13), label, fill="#374151", font=tick_font, anchor="mt")
    if y_labels is not None:
        for i, label in enumerate(y_labels):
            compact = label.split("\n")[0]
            draw.text((x0 - 8, y0 + (i + 0.5) * cell), compact, fill="#374151", font=tick_font, anchor="rm")


def apply_run_overrides(params):
    if DT_OVERRIDE is None:
        return params
    return replace(params, dt=DT_OVERRIDE)


def run_or_load_transition(profile: dict[str, object], params, initial_y: np.ndarray, profile_id: int, reuse: bool):
    path = OUT / f"trajectory_library_transition_{profile_name(profile)}.npz"
    if reuse and path.exists():
        return load_library(path)
    flow = load_flow(FLOW_PATH, params=params)
    lib = simulate_cell_transitions_compiled(
        flow,
        str(profile["condition"]),
        seed=ftm.SEED + 100 * profile_id,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )
    save_library(path, lib)
    return lib


def run_or_load_refinement(profile: dict[str, object], params, initial_y: np.ndarray, profile_id: int, reuse: bool):
    path = OUT / f"trajectory_library_refinement_{profile_name(profile)}.npz"
    if reuse and path.exists():
        return load_library(path)
    flow = load_flow(FLOW_PATH, params=params)
    lib = simulate_cell_transitions_compiled(
        flow,
        str(profile["condition"]),
        seed=pcr.SEED + 10 + profile_id,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )
    save_library(path, lib)
    return lib


def make_transition_figure(payloads: dict[str, dict[str, np.ndarray]], params) -> None:
    y_labels = ftm.state_labels(params)
    angle_labels = [
        f"{ftm.ANGLE_EDGES_DEG[i]:.0f}-{ftm.ANGLE_EDGES_DEG[i + 1]:.0f}"
        for i in range(ftm.ANGLE_EDGES_DEG.size - 1)
    ]
    FIGURES.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2200, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((1100, 46), f"Focused transition matrices with {RUN_LABEL} flow", fill="#111827", font=font(32, True), anchor="mm")
    start_x = 260
    top_y = 145
    bottom_y = 630
    panel_gap = 475
    cell_top = 44
    cell_bottom = 44
    for col, profile in enumerate(PROFILES):
        name = profile_name(profile)
        payload = payloads[name]
        x = start_x + col * panel_gap
        draw_heatmap(
            draw,
            payload["mobile_probability_conditional"],
            x,
            top_y,
            cell_top,
            f"{profile['label']}\nP(yout | yin, mobile)",
            [str(i) for i in range(payload["mobile_probability_conditional"].shape[1])],
            y_labels if col == 0 else None,
        )
        if name == "neutral_resolved":
            angle_prob = payload["center_near_angle_probability"]
            event_label = "center near-surface"
        else:
            angle_prob = payload["center_well_angle_probability"]
            event_label = "center secondary min"
        draw_heatmap(
            draw,
            angle_prob,
            x,
            bottom_y,
            cell_bottom,
            f"{event_label}\nP(|theta| | yin)",
            angle_labels,
            y_labels if col == 0 else None,
        )
    draw.text((120, top_y + 155), "inlet state", fill="#111827", font=font(15, True), anchor="mm")
    draw.text((120, bottom_y + 155), "inlet state", fill="#111827", font=font(15, True), anchor="mm")
    draw.text((1100, 1062), "Matrix rows are nonuniform inlet states 0-6; angle bins are degrees from downstream stagnation.", fill="#4b5563", font=font(16), anchor="mm")
    image.save(OUT / "openfoam_focusing_transition_matrices.png")
    image.save(FIGURES / MATRICES_FIGURE_NAME)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def center_transition_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    return [row for row in rows if row["state"] == "center core"]


def make_comparison_figure(transition_rows: list[dict[str, float | int | str]], refinement_rows: list[dict[str, float | int | str]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    lbm_transition = [
        row
        for row in read_csv(ROOT / "outputs" / "focusing_transition_matrices" / "focusing_transition_matrix_rows.csv")
        if row.get("state") == "center core"
    ]
    lbm_refinement = read_csv(ROOT / "outputs" / "periodic_cell_refinement" / "periodic_cell_refinement_summary.csv")

    labels = [str(profile["label"]) for profile in PROFILES[1:]]
    profile_ids = [str(profile["profile"]) for profile in PROFILES[1:]]

    def metric(rows, profile_id, label, key):
        for row in rows:
            if row.get("profile") == profile_id or row.get("label") == label:
                return float(row[key])
        return float("nan")

    image = Image.new("RGB", (1600, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((800, 42), f"Focusing response after replacing LBM with {RUN_LABEL} flow", fill="#111827", font=font(28, True), anchor="mm")

    def draw_panel(rect, title, lbm_rows, of_rows, key, ylabel):
        x0, y0, x1, y1 = rect
        draw.rectangle(rect, outline="#d1d5db", width=2)
        draw.text(((x0 + x1) / 2, y0 - 28), title, fill="#111827", font=font(20, True), anchor="mm")
        for tick in np.linspace(0.0, 1.0, 6):
            y = y1 - tick * (y1 - y0)
            draw.line((x0, y, x1, y), fill="#e5e7eb", width=1)
            draw.text((x0 - 8, y), f"{tick:.1f}", fill="#4b5563", font=font(13), anchor="rm")
        draw.text((x0 - 55, (y0 + y1) / 2), ylabel, fill="#111827", font=font(15, True), anchor="mm")
        group_w = (x1 - x0) / len(labels)
        bar_w = min(54.0, group_w * 0.25)
        lbm_values = np.array([metric(lbm_rows, profile_id, label, key) for profile_id, label in zip(profile_ids, labels)])
        of_values = np.array([metric(of_rows, profile_id, label, key) for profile_id, label in zip(profile_ids, labels)])
        for i, label in enumerate(labels):
            cx = x0 + (i + 0.5) * group_w
            for dx, value, color, name in (
                (-bar_w * 0.65, lbm_values[i], "#6b7280", "LBM"),
                (bar_w * 0.65, of_values[i], "#0f766e", RUN_LABEL),
            ):
                if not np.isfinite(value):
                    continue
                bx0 = cx + dx - bar_w / 2
                bx1 = cx + dx + bar_w / 2
                by = y1 - value * (y1 - y0)
                draw.rectangle((bx0, by, bx1, y1), fill=color)
                draw.text(((bx0 + bx1) / 2, by - 8), f"{value:.3f}", fill="#111827", font=font(12), anchor="mb")
            draw.text((cx, y1 + 22), label.replace("50 mM, ", ""), fill="#111827", font=font(14), anchor="mt")

    panels = (
        (
            (120, 130, 740, 580),
            "Center-core transition matrix",
            lbm_transition,
            center_transition_rows(transition_rows),
            "center_well_theta30_fraction",
            "well-release F30",
        ),
        (
            (900, 130, 1520, 580),
            "Central-core refinement",
            lbm_refinement,
            refinement_rows,
            "center_well_release_theta30_fraction",
            "well-release F30",
        ),
    )
    for panel in panels:
        draw_panel(*panel)
    draw.rectangle((610, 610, 630, 630), fill="#6b7280")
    draw.text((640, 620), "LBM", fill="#111827", font=font(15), anchor="lm")
    draw.rectangle((720, 610, 740, 630), fill="#0f766e")
    draw.text((750, 620), RUN_LABEL, fill="#111827", font=font(15), anchor="lm")
    image.save(OUT / "openfoam_focusing_comparison.png")
    image.save(FIGURES / COMPARISON_FIGURE_NAME)


def write_report(transition_rows: list[dict[str, float | int | str]], refinement_rows: list[dict[str, float | int | str]]) -> None:
    central = center_transition_rows(transition_rows)
    lines = [
        "# OpenFOAM-derived focusing runs",
        "",
        f"Particle tracking uses `{FLOW_PATH}`. This is a 512 x 512 periodic raster interpolated from the body-fitted OpenFOAM solution, not the LBM flow.",
        f"The outer SDE time step is `{fmt(DT_OVERRIDE * 1.0e3 if DT_OVERRIDE is not None else pcr.params_for(PROFILES[0]).dt * 1.0e3)} ms`.",
        "The DLVO, wall-mobility, reflection, and adaptive horizon settings are otherwise the same as the LBM focused cases.",
        "",
        "## Center-core transition-matrix row",
        "",
        "| case | horizon (s) | mobile exits | unresolved | center well releases | well F30 | median well angle (deg) | P(yout center given well) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in central:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt(ftm.profile_horizon_from_name(str(row["profile"])), 0),
                    fmt(row["mobile_exits"], 0),
                    fmt(row["censored"], 0),
                    fmt(row["center_well_releases"], 0),
                    fmt(row["center_well_theta30_fraction"]),
                    fmt(row["center_well_median_theta_deg"]),
                    fmt(row["center_well_yout_center_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Central-core refinement",
            "",
            "| case | horizon (s) | status | center well entries | center well releases | unresolved | well F30 | median well angle (deg) | median well time (s) | cumulative travel (deg) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in refinement_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt(row["max_time_s"], 0),
                    str(row["final_status"]),
                    fmt(row["center_well_intercepted_count"], 0),
                    fmt(row["center_well_release_count"], 0),
                    fmt(row["center_well_censored_count"], 0),
                    fmt(row["center_well_release_theta30_fraction"]),
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
            "Interpretation: the OpenFOAM-derived flow is the production-level flow refinement requested here. It should be read against the LBM reports in `outputs/focusing_transition_matrices` and `outputs/periodic_cell_refinement`.",
            "",
        ]
    )
    (OUT / "openfoam_focusing_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUT, DT_OVERRIDE, RUN_LABEL, COMPARISON_FIGURE_NAME, MATRICES_FIGURE_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="Reuse existing trajectory libraries in the selected output directory.")
    parser.add_argument("--dt-ms", type=float, default=None, help="Override the outer SDE time step in milliseconds.")
    parser.add_argument("--out-name", default=None, help="Output directory name under outputs/.")
    parser.add_argument("--figure-prefix", default=None, help="Prefix for diagnostic figure PNGs written under outputs/figures.")
    args = parser.parse_args()

    if args.dt_ms is not None:
        DT_OVERRIDE = float(args.dt_ms) * 1.0e-3
        safe_dt = str(args.dt_ms).replace(".", "p").replace("-", "m")
        RUN_LABEL = f"OpenFOAM-derived, dt={args.dt_ms:g} ms"
        OUT = ROOT / "outputs" / (args.out_name or f"openfoam_focusing_dt{safe_dt}ms")
        prefix = args.figure_prefix or f"fig_openfoam_focusing_dt{safe_dt}ms"
        COMPARISON_FIGURE_NAME = f"{prefix}_comparison.png"
        MATRICES_FIGURE_NAME = f"{prefix}_transition_matrices.png"
    elif args.out_name is not None:
        OUT = ROOT / "outputs" / args.out_name

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not FLOW_PATH.exists():
        raise FileNotFoundError(f"Create the OpenFOAM raster first: {FLOW_PATH}")
    build_kernel(force=False)
    reuse = bool(args.reuse)

    transition_reference_params = apply_run_overrides(ftm.params_for(PROFILES[0]))
    transition_initial_y = ftm.sample_y(transition_reference_params, ftm.SEED)
    transition_rows: list[dict[str, float | int | str]] = []
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for profile_id, profile in enumerate(PROFILES):
        params = apply_run_overrides(ftm.params_for(profile))
        print(f"Transition {profile_name(profile)} with OpenFOAM flow", flush=True)
        lib = run_or_load_transition(profile, params, transition_initial_y, profile_id, reuse)
        payloads[profile_name(profile)] = ftm.build_transition_payload(lib, params, profile)
        transition_rows.extend(ftm.summarize_rows(lib, params, profile))
        write_csv(OUT / "openfoam_transition_rows_partial.csv", transition_rows)

    refinement_reference_params = apply_run_overrides(pcr.params_for(PROFILES[0]))
    refinement_initial_y = pcr.central_core_y(refinement_reference_params)
    refinement_rows: list[dict[str, float | int | str]] = []
    for profile_id, profile in enumerate(PROFILES):
        params = apply_run_overrides(pcr.params_for(profile))
        print(f"Refinement {profile_name(profile)} with OpenFOAM flow", flush=True)
        lib = run_or_load_refinement(profile, params, refinement_initial_y, profile_id, reuse)
        refinement_rows.append(pcr.summarize_library(profile, params, lib))
        write_csv(OUT / "openfoam_refinement_summary_partial.csv", refinement_rows)

    for name, payload in payloads.items():
        np.savez_compressed(OUT / f"openfoam_transition_payload_{name}.npz", **payload)
    write_csv(OUT / "openfoam_transition_rows.csv", transition_rows)
    write_csv(OUT / "openfoam_refinement_summary.csv", refinement_rows)
    make_transition_figure(payloads, transition_reference_params)
    make_comparison_figure(transition_rows, refinement_rows)
    write_report(transition_rows, refinement_rows)
    print(OUT / "openfoam_focusing_report.md", flush=True)


if __name__ == "__main__":
    main()
