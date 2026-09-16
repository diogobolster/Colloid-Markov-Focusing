#!/usr/bin/env python3
"""Audit finite-time censoring in one-cell DLVO trajectory libraries."""

from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import (
    PhysicalParams,
    load_flow,
    rescale_flow,
    simulate_cell_transitions,
    solve_lbm_flow,
)


PHYSICAL_OUT = ROOT / "outputs" / "physical"
OUT = ROOT / "outputs" / "validation"
FIGURES = ROOT / "outputs" / "figures"

VELOCITY_M_PER_DAY = 4.0
N_BINS = 96
N_PER_OPEN_BIN = 160
MAX_TIMES_S = (30.0, 60.0, 90.0, 120.0)

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
TEAL = "#0891b2"


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    condition: str
    surface_mode: str
    allow_attachment: bool
    color: str
    seed: int


CASES = (
    Case("neutral_lubrication", "no DLVO", "neutral", "lubrication", False, BLUE, 51011),
    Case("unfavorable_project", "unfavorable/project", "unfavorable", "project", False, GREEN, 51023),
    Case("unfavorable_lubrication", "unfavorable/lubrication", "unfavorable", "lubrication", False, TEAL, 51037),
    Case("favorable_attach", "favorable attach", "favorable", "project", True, RED, 51049),
)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def stratified_inlet(params: PhysicalParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inlet positions and aperture weights for equal support per open bin."""

    rng = np.random.default_rng(99013)
    bins = np.linspace(0.0, params.cell_length, N_BINS + 1)
    y_values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    bin_ids: list[np.ndarray] = []
    open_widths = []
    open_bins = []
    for i in range(N_BINS):
        low = max(bins[i], params.inlet_y_min)
        high = min(bins[i + 1], params.inlet_y_max)
        width = high - low
        if width <= 0.0:
            continue
        open_bins.append(i)
        open_widths.append(width)
        y_values.append(rng.uniform(low, high, size=N_PER_OPEN_BIN))
        bin_ids.append(np.full(N_PER_OPEN_BIN, i, dtype=int))

    total_width = float(np.sum(open_widths))
    for width in open_widths:
        weights.append(np.full(N_PER_OPEN_BIN, width / (N_PER_OPEN_BIN * total_width)))
    return np.concatenate(y_values), np.concatenate(weights), np.concatenate(bin_ids)


def weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights[mask]) / np.sum(weights))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(ok):
        return float("nan")
    v = values[ok]
    w = weights[ok]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def summarize(case: Case, max_time: float, lib, weights: np.ndarray, bin_ids: np.ndarray) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    completed = lib.exited | lib.attached
    censored = lib.censored
    intercepted = lib.interceptions > 0
    n_open_bins = int(np.unique(bin_ids).size)
    censors_by_bin = np.bincount(bin_ids[censored], minlength=N_BINS)
    return {
        "case": case.key,
        "label": case.label,
        "condition": case.condition,
        "surface_mode": case.surface_mode,
        "allow_attachment": int(case.allow_attachment),
        "velocity_m_per_day": VELOCITY_M_PER_DAY,
        "max_time_s": max_time,
        "particles": int(lib.y_in.size),
        "open_bins": n_open_bins,
        "n_per_open_bin": N_PER_OPEN_BIN,
        "exited_fraction": weighted_fraction(lib.exited, weights),
        "attached_fraction": weighted_fraction(lib.attached, weights),
        "completed_fraction": weighted_fraction(completed, weights),
        "censored_fraction": weighted_fraction(censored, weights),
        "intercepted_fraction": weighted_fraction(intercepted, weights),
        "censored_given_intercepted": (
            float(np.sum(weights[censored & intercepted]) / np.sum(weights[intercepted]))
            if np.any(intercepted)
            else float("nan")
        ),
        "intercepted_given_censored": (
            float(np.sum(weights[censored & intercepted]) / np.sum(weights[censored]))
            if np.any(censored)
            else float("nan")
        ),
        "mobile_median_time_s": weighted_quantile(lib.travel_time[mobile], weights[mobile], 0.50),
        "mobile_q90_time_s": weighted_quantile(lib.travel_time[mobile], weights[mobile], 0.90),
        "mobile_q95_time_s": weighted_quantile(lib.travel_time[mobile], weights[mobile], 0.95),
        "censored_median_x_over_L": weighted_quantile(
            lib.x_final[censored] / lib.params.cell_length,
            weights[censored],
            0.50,
        ),
        "censored_q90_x_over_L": weighted_quantile(
            lib.x_final[censored] / lib.params.cell_length,
            weights[censored],
            0.90,
        ),
        "censored_count": int(np.sum(censored)),
        "censored_bins": int(np.sum(censors_by_bin > 0)),
        "max_censored_in_bin": int(np.max(censors_by_bin)) if censors_by_bin.size else 0,
    }


def axis(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=2)
    for f in (0.25, 0.5, 0.75):
        y = y1 - f * (y1 - y0)
        draw.line((x0, y, x1, y), fill=GRID, width=1)
    draw.text(((x0 + x1) / 2, y0 - 26), title, fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 46), xlabel, fill=INK, font=font(18), anchor="mm")
    draw.text((x0 + 10, y0 + 12), ylabel, fill=MUTED, font=font(13), anchor="la")


def line_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    rows: list[dict[str, float | int | str]],
    metric: str,
    title: str,
    ylabel: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    axis(draw, rect, title, "maximum one-cell tracking time (s)", ylabel)
    x0, y0, x1, y1 = rect
    x_values = np.array(MAX_TIMES_S, dtype=float)
    if y_limits is None:
        values = np.array([float(row[metric]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        ymin = 0.0 if values.size and np.nanmin(values) >= 0.0 else float(np.nanmin(values))
        ymax = float(np.nanmax(values)) if values.size else 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        ymax *= 1.08
    else:
        ymin, ymax = y_limits

    for tick in np.linspace(ymin, ymax, 5):
        y = y1 - (tick - ymin) / max(ymax - ymin, 1.0e-12) * (y1 - y0)
        draw.text((x0 - 12, y), f"{tick:.3g}", fill=MUTED, font=font(14), anchor="rm")
    for tick in x_values:
        x = x0 + (tick - x_values[0]) / (x_values[-1] - x_values[0]) * (x1 - x0)
        draw.line((x, y1, x, y1 + 7), fill=INK, width=1)
        draw.text((x, y1 + 24), f"{tick:.0f}", fill=MUTED, font=font(14), anchor="mm")

    for case in CASES:
        series = [row for row in rows if row["case"] == case.key]
        series.sort(key=lambda row: float(row["max_time_s"]))
        points = []
        for row in series:
            x = x0 + (float(row["max_time_s"]) - x_values[0]) / (x_values[-1] - x_values[0]) * (x1 - x0)
            yv = float(row[metric])
            if not np.isfinite(yv):
                continue
            y = y1 - (yv - ymin) / max(ymax - ymin, 1.0e-12) * (y1 - y0)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=case.color, width=4, joint="curve")
        for p in points:
            draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=case.color)


def write_figure(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    image = Image.new("RGB", (1800, 1450), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 52), "Finite-time censoring audit", fill=INK, font=font(34, True), anchor="la")
    draw.text(
        (90, 92),
        "Stratified 96-bin inlet support, resolved N=192 flow, 4 m/day",
        fill=MUTED,
        font=font(20),
        anchor="la",
    )
    rects = [
        (120, 170, 820, 560),
        (1010, 170, 1710, 560),
        (120, 750, 820, 1140),
        (1010, 750, 1710, 1140),
    ]
    line_panel(draw, rects[0], rows, "censored_fraction", "A. Censored mass", "fraction")
    line_panel(draw, rects[1], rows, "completed_fraction", "B. Completed mass", "exit + attach fraction", (0.86, 1.005))
    line_panel(draw, rects[2], rows, "mobile_q95_time_s", "C. Mobile travel-time tail", "95th percentile (s)")
    line_panel(draw, rects[3], rows, "intercepted_given_censored", "D. Censored particles that intercepted", "fraction of censored", (0.0, 1.0))

    lx, ly = 1070, 1240
    for i, case in enumerate(CASES):
        y = ly + 30 * i
        draw.line((lx, y, lx + 46, y), fill=case.color, width=5)
        draw.text((lx + 58, y), case.label, fill=INK, font=font(18), anchor="lm")
    draw.text(
        (120, 1380),
        "Censoring is a numerical horizon flag, not attachment. Favorable attachment is absorbing; unfavorable cases remain nonattaching.",
        fill=MUTED,
        font=font(18),
        anchor="la",
    )
    image.save(path)


def fmt(value: float | int | str, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "nan"


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        "# One-cell censoring sensitivity",
        "",
        (
            "This audit reruns the 4 m/day resolved N=192 flow with stratified support "
            f"across the {N_BINS}-bin transverse Markov state. Each open inlet bin "
            f"contains {N_PER_OPEN_BIN} particles, and fractions are aperture-weighted "
            "so the clipped edge bins do not receive excess mass."
        ),
        "",
        "| case | max time (s) | exited | attached | completed | censored | censored count | censored/intercepted | intercepted/censored | mobile q95 time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{fmt(row['max_time_s'], 0)} | "
            f"{fmt(row['exited_fraction'])} | "
            f"{fmt(row['attached_fraction'])} | "
            f"{fmt(row['completed_fraction'])} | "
            f"{fmt(row['censored_fraction'])} | "
            f"{int(row['censored_count'])} | "
            f"{fmt(row['censored_given_intercepted'])} | "
            f"{fmt(row['intercepted_given_censored'])} | "
            f"{fmt(row['mobile_q95_time_s'], 2)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Censored particles are particles that have neither exited nor attached before the imposed one-cell time horizon. They are not retained particles. In the homogeneous unfavorable cases, attachment is disabled, so censoring is purely a finite-time numerical artifact unless a later barrier-crossing, roughness, heterodomain, or straining model is explicitly added.",
            "",
            "The preferred production check is the 60--120 s change, because the current one-cell libraries use a 60 s maximum tracking time. If the 120 s censoring fraction is much smaller than the 60 s value while mobile travel-time quantiles remain stable, the long-distance nonattaching Markov-loss term should be reported as a numerical lower bound on eventual mobile survival rather than a physical deposition prediction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams()
    base_flow = ensure_base_flow(params)
    y0, weights, bin_ids = stratified_inlet(params)
    rows: list[dict[str, float | int | str]] = []
    for case in CASES:
        for max_time in MAX_TIMES_S:
            mean_velocity = VELOCITY_M_PER_DAY / 86400.0
            flow = rescale_flow(base_flow, mean_velocity, max_time=max_time)
            print(
                f"Tracking {case.label}, max_time={max_time:.0f} s, "
                f"particles={y0.size}, surface={case.surface_mode}"
            )
            lib = simulate_cell_transitions(
                flow,
                case.condition,
                n_particles=y0.size,
                seed=case.seed,
                allow_attachment=case.allow_attachment,
                initial_y=y0,
                surface_mode=case.surface_mode,
            )
            row = summarize(case, max_time, lib, weights, bin_ids)
            rows.append(row)
            print(
                "  completed",
                f"{row['completed_fraction']:.4f}",
                "censored",
                f"{row['censored_fraction']:.4f}",
                "count",
                row["censored_count"],
            )

    rows.sort(key=lambda row: (str(row["case"]), float(row["max_time_s"])))
    write_dicts(OUT / "censoring_sensitivity_summary.csv", rows)
    write_report(OUT / "censoring_sensitivity_report.md", rows)
    figure_path = OUT / "censoring_sensitivity.png"
    write_figure(figure_path, rows)
    shutil.copyfile(figure_path, FIGURES / "fig13_censoring_sensitivity.png")
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
