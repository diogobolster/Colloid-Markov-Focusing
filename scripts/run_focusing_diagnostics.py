#!/usr/bin/env python3
"""Test DLVO-induced streamline focusing against a no-DLVO baseline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import (
    PhysicalParams,
    focusing_metrics,
    load_flow,
    rescale_flow,
    save_library,
    simulate_cell_transitions,
    solve_lbm_flow,
    transition_matrices,
)


PHYSICAL_OUT = ROOT / "outputs" / "physical"
OUT = ROOT / "outputs" / "focusing"
VELOCITY_CASES_M_PER_DAY = (2.0, 4.0, 8.0)
CONDITIONS = (
    ("neutral", "advection-diffusion, no DLVO"),
    ("favorable_reflecting", "attractive DLVO, nonattaching"),
    ("favorable_sliding", "attractive DLVO, surface sliding, nonattaching"),
    ("favorable_lubricated", "attractive DLVO, lubrication-limited nonattaching"),
    ("unfavorable", "repulsive DLVO, reflecting"),
)
SURFACE_MODES = {
    "neutral": "lubrication",
    "favorable_reflecting": "project",
    "favorable_sliding": "sliding",
    "favorable_lubricated": "lubrication",
    "unfavorable": "lubrication",
}
CORE_FOCUSING_BINS = np.array([15, 16])


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_dicts(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.10g")


def ensure_base_flow(params: PhysicalParams):
    path = PHYSICAL_OUT / "flow_N192.npz"
    if not path.exists():
        PHYSICAL_OUT.mkdir(parents=True, exist_ok=True)
        flow = solve_lbm_flow(params, resolution=192, iterations=3800)
        np.savez_compressed(
            path,
            x=flow.x,
            y=flow.y,
            ux=flow.ux,
            uy=flow.uy,
            solid=flow.solid,
            resolution=flow.resolution,
            iterations=flow.iterations,
            tau=flow.tau,
        )
    return load_flow(path, params)


def add_comparisons(rows: list[dict[str, float | str]]) -> None:
    by_velocity = {
        float(row["velocity_m_per_day"]): row
        for row in rows
        if row["condition"] == "neutral"
    }
    for row in rows:
        base = by_velocity[float(row["velocity_m_per_day"])]
        for key in (
            "median_abs_yout_center_m",
            "mean_abs_yout_center_m",
            "std_yout_intercepted_m",
            "median_abs_theta_exit_downstream_rad",
            "center_median_abs_yout_center_m",
            "center_median_abs_theta_exit_downstream_rad",
        ):
            numerator = float(base[key])
            denominator = float(row[key])
            row[f"neutral_to_case_ratio_{key}"] = numerator / denominator if np.isfinite(denominator) and denominator > 0 else float("nan")
        base_center_frac = float(base["fraction_yout_within_center_band"])
        case_center_frac = float(row["fraction_yout_within_center_band"])
        row["centerline_enrichment_vs_neutral"] = (
            case_center_frac / base_center_frac
            if np.isfinite(base_center_frac) and base_center_frac > 0
            else float("nan")
        )


def write_report(
    path: Path,
    rows: list[dict[str, float | str]],
    core_rows: list[dict[str, float | str]],
) -> None:
    def fmt(value: float | str, digits: int = 3) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "nan"

    lines = [
        "# DLVO focusing diagnostic",
        "",
        "Focusing is measured by comparing particles that intercepted the near-surface zone against a no-DLVO advection-diffusion baseline in the resolved center/corner cell. The attractive cases use favorable DLVO drift but disable attachment so release position and downstream streamline reassignment can be observed directly.",
        "",
        "The table below is a uniform-injection pilot diagnostic. It reports both all-collector interceptions and center-grain-only interceptions using the collector family recorded for each event. The `favorable_reflecting` case is the old projection closure; the `favorable_sliding` case balances inward normal motion at near contact while retaining tangential hydrodynamic and Brownian motion. The `favorable_lubricated` case applies the near-wall mobility model to DLVO drift and Brownian motion.",
        "",
        "| condition | velocity (m/day) | all intercepted | all mobile | center intercepted | center mobile | center censored | center median abs theta_exit (rad) | center y_out within 25 um | center median abs y_out-y_c (um) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['condition']} | "
            f"{float(row['velocity_m_per_day']):.1f} | "
            f"{int(row['intercepted_any_count'])} | "
            f"{int(row['intercepted_mobile_count'])} | "
            f"{int(row['center_intercepted_any_count'])} | "
            f"{int(row['center_intercepted_mobile_count'])} | "
            f"{int(row['center_intercepted_censored_count'])} | "
            f"{fmt(row['center_median_abs_theta_exit_downstream_rad'])} | "
            f"{fmt(row['center_fraction_yout_within_center_band'])} | "
            f"{fmt(float(row['center_median_abs_yout_center_m']) * 1e6, 2)} |"
        )
    lines.extend(
        [
            "",
            "Here theta_exit is the last near-surface angle before a mobile particle leaves the interaction zone, measured relative to the downstream stagnation point of the nearest collector at theta = 0. Center-grain rows count only particles whose trajectory intercepted the center-grain family.",
            "",
            "## Central-core pilot",
            "",
            "The central-core pilot injects particles only through inlet bins 15 and 16 at 4 m/day. This isolates the center-grain focusing mechanism from the corner collectors.",
            "",
            "| condition | particles | center intercepted | center mobile | center censored | center median abs theta_exit (rad) | center y_out within 25 um | center median abs y_out-y_c (um) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in core_rows:
        lines.append(
            "| "
            f"{row['condition']} | "
            f"{int(row['particles'])} | "
            f"{int(row['center_intercepted_any_count'])} | "
            f"{int(row['center_intercepted_mobile_count'])} | "
            f"{int(row['center_intercepted_censored_count'])} | "
            f"{fmt(row['center_median_abs_theta_exit_downstream_rad'])} | "
            f"{fmt(row['center_fraction_yout_within_center_band'])} | "
            f"{fmt(float(row['center_median_abs_yout_center_m']) * 1e6, 2)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def hist_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values[np.isfinite(values)], bins=bins)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def draw_hist_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    size: tuple[int, int],
    bins: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    xlabel: str,
) -> None:
    ox, oy = origin
    width, height = size
    draw.rectangle((ox, oy, ox + width, oy + height), outline="#111111", width=2)
    draw.text((ox + width / 2, oy - 30), title, fill="#111111", font=font(24), anchor="mm")
    max_y = 1.0e-12
    hist_series = []
    for label, values, color in series:
        h = hist_counts(values, bins)
        hist_series.append((label, h, color))
        max_y = max(max_y, float(np.max(h)))

    for label, counts, color in hist_series:
        pts = []
        centers = 0.5 * (bins[:-1] + bins[1:])
        for xval, yval in zip(centers, counts):
            x = ox + (xval - bins[0]) / (bins[-1] - bins[0]) * width
            y = oy + height - yval / max_y * height
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4, joint="curve")

    for tick in np.linspace(bins[0], bins[-1], 5):
        x = ox + (tick - bins[0]) / (bins[-1] - bins[0]) * width
        draw.line((x, oy + height, x, oy + height + 8), fill="#111111", width=2)
        draw.text((x, oy + height + 28), f"{tick:.1f}", fill="#111111", font=font(16), anchor="mm")
    draw.text((ox + width / 2, oy + height + 58), xlabel, fill="#111111", font=font(20), anchor="mm")

    lx = ox + width - 310
    ly = oy + 20
    for i, (label, _, color) in enumerate(hist_series):
        y = ly + 30 * i
        draw.line((lx, y, lx + 42, y), fill=color, width=5)
        draw.text((lx + 54, y), label, fill="#111111", font=font(17), anchor="lm")


def write_histogram_figure(path: Path, libraries: dict[str, object]) -> None:
    colors = {
        "neutral": "#2563eb",
        "favorable_reflecting": "#dc2626",
        "favorable_sliding": "#d97706",
        "favorable_lubricated": "#7c3aed",
        "unfavorable": "#059669",
    }
    labels = {
        "neutral": "no DLVO",
        "favorable_reflecting": "attractive/project",
        "favorable_sliding": "attractive/sliding",
        "favorable_lubricated": "attractive/lubricated",
        "unfavorable": "repulsive",
    }
    series_y = []
    series_theta = []
    series_theta_final = []
    for condition, lib in libraries.items():
        mobile_intercepted = lib.exited & (~lib.attached) & (lib.center_interceptions > 0)
        center_release = mobile_intercepted & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
        censored_intercepted = lib.censored & (lib.center_interceptions > 0) & (lib.collector_final == 1)
        y_center_um = (lib.y_out[mobile_intercepted] - 0.5 * lib.params.cell_length) * 1e6
        theta = np.abs(np.arctan2(np.sin(lib.theta_exit[center_release]), np.cos(lib.theta_exit[center_release])))
        theta_final = np.abs(np.arctan2(np.sin(lib.theta_final[censored_intercepted]), np.cos(lib.theta_final[censored_intercepted])))
        series_y.append((f"{labels[condition]} (n={int(np.sum(mobile_intercepted))})", y_center_um, colors[condition]))
        series_theta.append((f"{labels[condition]} (n={int(np.sum(center_release))})", theta, colors[condition]))
        series_theta_final.append((f"{labels[condition]} (n={int(np.sum(censored_intercepted))})", theta_final, colors[condition]))

    img = Image.new("RGB", (1280, 1320), "white")
    draw = ImageDraw.Draw(img)
    draw.text((640, 44), "Center-grain near-surface focusing pilot, 4 m/day", fill="#111111", font=font(34), anchor="mm")
    draw_hist_panel(
        draw,
        (110, 120),
        (980, 280),
        np.linspace(-120.0, 120.0, 49),
        series_y,
        "Outlet position of center-intercepted mobile particles",
        "y_out - centerline (um)",
    )
    draw_hist_panel(
        draw,
        (110, 560),
        (980, 250),
        np.linspace(0.0, np.pi, 49),
        series_theta,
        "Release angle from center-grain downstream stagnation",
        "|theta_exit| (rad)",
    )
    draw_hist_panel(
        draw,
        (110, 980),
        (980, 220),
        np.linspace(0.0, np.pi, 49),
        series_theta_final,
        "Final angle of intercepted censored particles",
        "|theta_final| (rad)",
    )
    img.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams()
    base_flow = ensure_base_flow(params)
    particle_count = 15000
    core_particle_count = 20000
    bins = 32
    rows: list[dict[str, float | str]] = []
    core_rows: list[dict[str, float | str]] = []
    libraries_4m = {}

    for velocity_index, velocity_m_per_day in enumerate(VELOCITY_CASES_M_PER_DAY):
        mean_velocity = velocity_m_per_day / 86400.0
        max_time = max(60.0, 6.0 * params.cell_length / mean_velocity)
        flow = rescale_flow(base_flow, mean_velocity, max_time=max_time)
        for condition_index, (condition, description) in enumerate(CONDITIONS):
            seed = 5000 + 100 * velocity_index + condition_index
            print(f"Focusing run: {condition}, {velocity_m_per_day:.0f} m/day")
            surface_mode = SURFACE_MODES[condition]
            lib = simulate_cell_transitions(
                flow,
                condition,
                n_particles=particle_count,
                seed=seed,
                allow_attachment=False,
                surface_mode=surface_mode,
            )
            stem = f"{condition}_{velocity_m_per_day:.0f}m_per_day"
            save_library(OUT / f"trajectory_library_{stem}.npz", lib)
            mats = transition_matrices(lib, n_bins=bins)
            write_matrix(OUT / f"transition_probabilities_{stem}.csv", mats["probabilities"])
            write_matrix(OUT / f"transition_counts_{stem}.csv", mats["counts"])
            row = focusing_metrics(lib)
            row["velocity_m_per_day"] = velocity_m_per_day
            row["description"] = description
            rows.append(row)
            if velocity_m_per_day == 4.0:
                libraries_4m[condition] = lib

    core_velocity_m_per_day = 4.0
    core_flow = rescale_flow(
        base_flow,
        core_velocity_m_per_day / 86400.0,
        max_time=max(60.0, 6.0 * params.cell_length / (core_velocity_m_per_day / 86400.0)),
    )
    core_rng = np.random.default_rng(8800)
    core_edges = np.linspace(0.0, params.cell_length, bins + 1)
    core_y_min = core_edges[int(CORE_FOCUSING_BINS[0])]
    core_y_max = core_edges[int(CORE_FOCUSING_BINS[-1]) + 1]
    for condition_index, (condition, description) in enumerate(CONDITIONS):
        surface_mode = SURFACE_MODES[condition]
        initial_y = core_rng.uniform(core_y_min, core_y_max, size=core_particle_count)
        seed = 9000 + condition_index
        print(f"Central-core focusing run: {condition}, 4 m/day")
        lib = simulate_cell_transitions(
            core_flow,
            condition,
            n_particles=core_particle_count,
            seed=seed,
            allow_attachment=False,
            initial_y=initial_y,
            surface_mode=surface_mode,
        )
        save_library(OUT / f"trajectory_library_core_{condition}_4m_per_day.npz", lib)
        row = focusing_metrics(lib)
        row["velocity_m_per_day"] = core_velocity_m_per_day
        row["description"] = description
        core_rows.append(row)

    add_comparisons(rows)
    write_dicts(OUT / "focusing_summary.csv", rows)
    write_dicts(OUT / "central_core_focusing_summary.csv", core_rows)
    write_report(OUT / "focusing_report.md", rows, core_rows)
    write_histogram_figure(OUT / "focusing_histograms_4m_per_day.png", libraries_4m)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
