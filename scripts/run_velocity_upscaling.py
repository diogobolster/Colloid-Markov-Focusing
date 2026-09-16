#!/usr/bin/env python3
"""Velocity-sensitive many-cell Markov stitching from trajectory libraries."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import TrajectoryLibrary, load_library


PHYSICAL_OUT = ROOT / "outputs" / "physical"
OUT = ROOT / "outputs" / "upscaling"
FIGURES = ROOT / "outputs" / "figures"

N_BINS = 96
N_CELLS = 120
N_PARTICLES = 50_000
TARGET_CELLS = (25, 50, 100, 120)

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"


@dataclass(frozen=True)
class VelocityCase:
    velocity_m_per_day: float
    condition: str
    label: str
    color: str
    library_path: Path


VELOCITY_COLORS = {
    2.0: PURPLE,
    4.0: RED,
    8.0: ORANGE,
}

CONDITION_COLORS = {
    "neutral": BLUE,
    "favorable": RED,
    "unfavorable": GREEN,
}

CASES = tuple(
    VelocityCase(v, condition, label, CONDITION_COLORS[condition], path)
    for v in (2.0, 4.0, 8.0)
    for condition, label, path in (
        (
            "neutral",
            "no DLVO",
            PHYSICAL_OUT / f"trajectory_library_neutral_{int(v)}m_per_day_{300 if v == 2.0 else 180}s.npz",
        ),
        (
            "favorable",
            "favorable attach",
            PHYSICAL_OUT / f"trajectory_library_favorable_{int(v)}m_per_day.npz",
        ),
        (
            "unfavorable",
            "unfavorable reflect",
            PHYSICAL_OUT / f"trajectory_library_unfavorable_{int(v)}m_per_day_{300 if v == 2.0 else 180}s.npz",
        ),
    )
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


def bin_records(library: TrajectoryLibrary) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, int]:
    bins = np.linspace(0.0, library.params.cell_length, N_BINS + 1)
    yin_bin = np.clip(np.digitize(library.y_in, bins) - 1, 0, N_BINS - 1)
    by_bin = [np.flatnonzero(yin_bin == i) for i in range(N_BINS)]
    nonempty = np.flatnonzero([idx.size > 0 for idx in by_bin])
    nearest = np.zeros(N_BINS, dtype=int)
    for i in range(N_BINS):
        nearest[i] = int(nonempty[np.argmin(np.abs(nonempty - i))])
    min_support = int(min(by_bin[int(i)].size for i in nonempty))
    return bins, by_bin, nearest, min_support


def initial_positions(library: TrajectoryLibrary, rng: np.random.Generator, n_particles: int) -> np.ndarray:
    bins, by_bin, _, _ = bin_records(library)
    open_bins = np.array([i for i, ids in enumerate(by_bin) if ids.size > 0], dtype=int)
    counts = np.array([by_bin[i].size for i in open_bins], dtype=float)
    probabilities = counts / np.sum(counts)
    chosen_bins = rng.choice(open_bins, size=n_particles, p=probabilities)
    y = np.empty(n_particles, dtype=float)
    for bin_id in open_bins:
        mask = chosen_bins == bin_id
        if np.any(mask):
            y[mask] = rng.uniform(bins[bin_id], bins[bin_id + 1], size=int(np.sum(mask)))
    return y


def sample_from_bins(
    library: TrajectoryLibrary,
    by_bin: list[np.ndarray],
    nearest: np.ndarray,
    bins: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    ybin = np.clip(np.digitize(y, bins) - 1, 0, N_BINS - 1)
    sampled = np.empty(y.size, dtype=int)
    for bin_id in np.unique(ybin):
        mask = ybin == bin_id
        source_bin = bin_id if by_bin[int(bin_id)].size else nearest[int(bin_id)]
        candidates = by_bin[int(source_bin)]
        sampled[mask] = rng.choice(candidates, size=int(np.sum(mask)), replace=True)
    return sampled


def summarize_times(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.quantile(values, 0.05)),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.95)),
    )


def rng_seed(case: VelocityCase) -> int:
    condition_part = sum(ord(ch) for ch in case.condition)
    velocity_part = int(round(case.velocity_m_per_day * 1000.0))
    return 73000 + velocity_part + condition_part


def run_case(
    case: VelocityCase,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], dict[int, np.ndarray]]:
    if not case.library_path.exists():
        raise FileNotFoundError(case.library_path)

    library = load_library(case.library_path)
    rng = np.random.default_rng(rng_seed(case))
    bins, by_bin, nearest, min_support = bin_records(library)
    y = initial_positions(library, rng, N_PARTICLES)
    time = np.zeros(N_PARTICLES, dtype=float)
    cumulative_interceptions = np.zeros(N_PARTICLES, dtype=int)
    cumulative_near_time = np.zeros(N_PARTICLES, dtype=float)
    alive = np.ones(N_PARTICLES, dtype=bool)
    attached = np.zeros(N_PARTICLES, dtype=bool)
    censored = np.zeros(N_PARTICLES, dtype=bool)
    attached_cell = np.full(N_PARTICLES, -1, dtype=int)
    censored_cell = np.full(N_PARTICLES, -1, dtype=int)
    times_at_target: dict[int, np.ndarray] = {}
    rows: list[dict[str, float | int | str]] = []

    mobile_travel_times = library.travel_time[library.exited & (~library.attached)]
    fallback_travel_time = float(np.nanmedian(mobile_travel_times[np.isfinite(mobile_travel_times)]))

    rows.append(
        {
            "velocity_m_per_day": case.velocity_m_per_day,
            "condition": case.condition,
            "label": case.label,
            "cell": 0,
            "mobile_fraction": 1.0,
            "attached_fraction": 0.0,
            "censored_fraction": 0.0,
            "plume_std_y_um": float(np.std(y) * 1.0e6),
            "median_time_s": 0.0,
            "q05_time_s": 0.0,
            "q95_time_s": 0.0,
            "mean_cumulative_interceptions_mobile": 0.0,
            "mean_cumulative_near_time_mobile_s": 0.0,
            "one_cell_attached_fraction": float(np.mean(library.attached)),
            "one_cell_censored_fraction": float(np.mean(library.censored)),
            "min_96_bin_support": min_support,
        }
    )

    for cell in range(1, N_CELLS + 1):
        active_ids = np.flatnonzero(alive)
        if active_ids.size:
            picks = sample_from_bins(library, by_bin, nearest, bins, y[active_ids], rng)
            picked_attached = library.attached[picks]
            picked_exited = library.exited[picks] & (~library.attached[picks])
            picked_censored = ~(picked_attached | picked_exited)

            if np.any(picked_attached):
                ids = active_ids[picked_attached]
                selected = picks[picked_attached]
                attached[ids] = True
                alive[ids] = False
                attached_cell[ids] = cell
                cumulative_interceptions[ids] += library.interceptions[selected]
                cumulative_near_time[ids] += library.near_time[selected]

            if np.any(picked_censored):
                ids = active_ids[picked_censored]
                censored[ids] = True
                alive[ids] = False
                censored_cell[ids] = cell

            if np.any(picked_exited):
                ids = active_ids[picked_exited]
                selected = picks[picked_exited]
                y[ids] = library.y_out[selected]
                dt = library.travel_time[selected]
                dt = np.where(np.isfinite(dt), dt, fallback_travel_time)
                time[ids] += dt
                cumulative_interceptions[ids] += library.interceptions[selected]
                cumulative_near_time[ids] += library.near_time[selected]

        if cell in TARGET_CELLS:
            times_at_target[cell] = time[alive].copy()

        mobile_ids = np.flatnonzero(alive)
        q05, q50, q95 = summarize_times(time[mobile_ids])
        rows.append(
            {
                "velocity_m_per_day": case.velocity_m_per_day,
                "condition": case.condition,
                "label": case.label,
                "cell": cell,
                "mobile_fraction": float(np.mean(alive)),
                "attached_fraction": float(np.mean(attached)),
                "censored_fraction": float(np.mean(censored)),
                "plume_std_y_um": float(np.std(y[mobile_ids]) * 1.0e6) if mobile_ids.size else float("nan"),
                "median_time_s": q50,
                "q05_time_s": q05,
                "q95_time_s": q95,
                "mean_cumulative_interceptions_mobile": float(np.mean(cumulative_interceptions[mobile_ids]))
                if mobile_ids.size
                else float("nan"),
                "mean_cumulative_near_time_mobile_s": float(np.mean(cumulative_near_time[mobile_ids]))
                if mobile_ids.size
                else float("nan"),
                "one_cell_attached_fraction": float(np.mean(library.attached)),
                "one_cell_censored_fraction": float(np.mean(library.censored)),
                "min_96_bin_support": min_support,
            }
        )

    retention_rows: list[dict[str, float | int | str]] = []
    for cell in range(1, N_CELLS + 1):
        retention_rows.append(
            {
                "velocity_m_per_day": case.velocity_m_per_day,
                "condition": case.condition,
                "cell": cell,
                "new_attached_fraction": float(np.mean(attached_cell == cell)),
                "new_censored_fraction": float(np.mean(censored_cell == cell)),
            }
        )
    return rows, retention_rows, times_at_target


def panel_axes(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str, xlabel: str, ylabel: str) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=2)
    draw.text(((x0 + x1) / 2, y0 - 30), title, fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 44), xlabel, fill=INK, font=font(18), anchor="mm")
    draw.text((x0 + 13, y0 + 16), ylabel, fill=MUTED, font=font(13), anchor="la")


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    series: list[tuple[str, str, np.ndarray, np.ndarray]],
    title: str,
    xlabel: str,
    ylabel: str,
    y_limits: tuple[float, float] | None = None,
    legend_loc: str = "upper_right",
) -> None:
    x0, y0, x1, y1 = rect
    panel_axes(draw, rect, title, xlabel, ylabel)
    all_x = np.concatenate([x for _, _, x, _ in series])
    all_y = np.concatenate([y[np.isfinite(y)] for _, _, _, y in series])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    if y_limits is None:
        ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
        pad = 0.08 * max(ymax - ymin, 1.0e-12)
        ymin -= pad
        ymax += pad
    else:
        ymin, ymax = y_limits

    def sx(v: float) -> float:
        return x0 + (v - xmin) / max(xmax - xmin, 1.0e-12) * (x1 - x0)

    def sy(v: float) -> float:
        return y1 - (v - ymin) / max(ymax - ymin, 1.0e-12) * (y1 - y0)

    x_tick_format = "{:.0f}" if abs(xmax - xmin) >= 5.0 else "{:.1f}"
    for tick in np.linspace(xmin, xmax, 5):
        px = sx(float(tick))
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 24), x_tick_format.format(float(tick)), fill=INK, font=font(13), anchor="mm")
    for tick in np.linspace(ymin, ymax, 5):
        py = sy(float(tick))
        draw.line((x0 - 7, py, x0, py), fill=INK, width=2)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((x0 - 11, py), f"{tick:.2g}", fill=INK, font=font(13), anchor="rm")

    for label, color, x, y in series:
        finite = np.isfinite(y)
        pts = [(sx(float(a)), sy(float(b))) for a, b in zip(x[finite], y[finite])]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4, joint="curve")
        for px, py in pts[:: max(1, len(pts) // 8)]:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)

    if legend_loc == "lower_left":
        lx, ly = x0 + 24, y1 - 92
    else:
        lx, ly = x1 - 235, y0 + 25
    for i, (label, color, _, _) in enumerate(series):
        yy = ly + 28 * i
        draw.line((lx, yy, lx + 38, yy), fill=color, width=5)
        draw.text((lx + 50, yy), label, fill=INK, font=font(14), anchor="lm")


def draw_grouped_metric_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    rows: list[dict[str, float | int | str]],
    metric: str,
    scale: float,
    title: str,
    ylabel: str,
) -> None:
    x0, y0, x1, y1 = rect
    panel_axes(draw, rect, title, "velocity (m/day)", ylabel)
    velocities = np.array([2.0, 4.0, 8.0])
    conditions = ("neutral", "favorable", "unfavorable")
    labels = {"neutral": "no DLVO", "favorable": "favorable", "unfavorable": "unfavorable"}
    terminal = {
        (float(r["velocity_m_per_day"]), str(r["condition"])): float(r[metric]) * scale
        for r in rows
        if int(r["cell"]) == N_CELLS
    }
    values = np.array([terminal[(v, c)] for c in conditions for v in velocities], dtype=float)
    ymin = 0.0
    ymax = max(float(np.nanmax(values)) * 1.18, 1.0)

    def sx_group(i: int) -> float:
        return x0 + (i + 0.5) / len(velocities) * (x1 - x0)

    def sy(v: float) -> float:
        return y1 - (v - ymin) / max(ymax - ymin, 1.0e-12) * (y1 - y0)

    def tick_label(v: float) -> str:
        if ymax >= 100.0:
            return f"{v:.0f}"
        if ymax >= 10.0:
            return f"{v:.1f}".rstrip("0").rstrip(".")
        return f"{v:.2g}"

    for tick in np.linspace(ymin, ymax, 5):
        py = sy(float(tick))
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((x0 - 11, py), tick_label(float(tick)), fill=INK, font=font(13), anchor="rm")
    for i, velocity in enumerate(velocities):
        px = sx_group(i)
        draw.text((px, y1 + 24), f"{velocity:.0f}", fill=INK, font=font(13), anchor="mm")
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)

    group_width = (x1 - x0) / len(velocities)
    bar_width = min(46.0, group_width / 5.0)
    offsets = (-bar_width * 1.25, 0.0, bar_width * 1.25)
    for j, condition in enumerate(conditions):
        color = CONDITION_COLORS[condition]
        for i, velocity in enumerate(velocities):
            val = terminal[(velocity, condition)]
            cx = sx_group(i) + offsets[j]
            draw.rectangle((cx - bar_width / 2, sy(val), cx + bar_width / 2, y1), fill=color)
    lx, ly = x1 - 245, y0 + 24
    for j, condition in enumerate(conditions):
        yy = ly + 28 * j
        draw.rectangle((lx, yy - 8, lx + 30, yy + 8), fill=CONDITION_COLORS[condition])
        draw.text((lx + 43, yy), labels[condition], fill=INK, font=font(14), anchor="lm")


def write_figure(path: Path, summary_rows: list[dict[str, float | int | str]], retention_rows: list[dict[str, float | int | str]]) -> None:
    img = Image.new("RGB", (1800, 1380), "white")
    draw = ImageDraw.Draw(img)
    draw.text((900, 46), "Velocity sensitivity of 120-cell trajectory-library stitching", fill=INK, font=font(36, True), anchor="mm")

    favorable_by_velocity = {
        velocity: [r for r in summary_rows if str(r["condition"]) == "favorable" and float(r["velocity_m_per_day"]) == velocity]
        for velocity in (2.0, 4.0, 8.0)
    }
    survival_series = []
    attachment_series = []
    for velocity, rows in favorable_by_velocity.items():
        cells = np.array([int(r["cell"]) for r in rows])
        mobile = np.array([float(r["mobile_fraction"]) for r in rows])
        attached = np.array([float(r["attached_fraction"]) for r in rows])
        label = f"{velocity:.0f} m/day"
        survival_series.append((label, VELOCITY_COLORS[velocity], cells, mobile))
        attachment_series.append((label, VELOCITY_COLORS[velocity], cells, attached))

    draw_line_panel(
        draw,
        (130, 145, 820, 505),
        survival_series,
        "favorable mobile survival",
        "cell number",
        "mobile fraction",
        y_limits=(0.0, 1.0),
    )
    draw_line_panel(
        draw,
        (1040, 145, 1700, 505),
        attachment_series,
        "favorable cumulative attachment",
        "cell number",
        "attached fraction",
        y_limits=(0.0, 1.0),
    )
    draw_grouped_metric_panel(
        draw,
        (130, 710, 820, 1070),
        summary_rows,
        metric="plume_std_y_um",
        scale=1.0,
        title="mobile plume width at cell 120",
        ylabel="std(y) (um)",
    )
    draw_grouped_metric_panel(
        draw,
        (1040, 710, 1700, 1070),
        summary_rows,
        metric="mean_cumulative_interceptions_mobile",
        scale=1.0,
        title="mobile interception history at cell 120",
        ylabel="mean interceptions/mobile",
    )

    terminal_fav = {
        float(r["velocity_m_per_day"]): r
        for r in summary_rows
        if str(r["condition"]) == "favorable" and int(r["cell"]) == N_CELLS
    }
    summary = " ; ".join(
        [
            f"{v:.0f} m/day: S={float(terminal_fav[v]['mobile_fraction']):.3f}, R={float(terminal_fav[v]['attached_fraction']):.3f}"
            for v in (2.0, 4.0, 8.0)
        ]
    )
    draw.text((900, 1260), summary, fill=INK, font=font(21), anchor="mm")
    draw.text(
        (900, 1302),
        "No-DLVO and homogeneous unfavorable libraries are nonattaching and censor-free; velocity primarily changes travel time, transverse spreading, and favorable depletion.",
        fill=MUTED,
        font=font(19),
        anchor="mm",
    )
    img.save(path)


def write_report(
    path: Path,
    summary_rows: list[dict[str, float | int | str]],
    retention_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Velocity-sensitive Markov upscaling",
        "",
        f"Each library is stitched through {N_CELLS} cells with {N_PARTICLES:,} particles and {N_BINS} transverse inlet bins.",
        "The no-DLVO and homogeneous unfavorable cases use the long-horizon nonattaching libraries.",
        "",
        "| velocity (m/day) | condition | cell-120 mobile | cell-120 attached | cell-120 censored | plume std (um) | median time (h) | mean interceptions/mobile | one-cell attached | one-cell censored | min 96-bin support |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for velocity in (2.0, 4.0, 8.0):
        for condition in ("neutral", "favorable", "unfavorable"):
            row = next(
                r
                for r in summary_rows
                if float(r["velocity_m_per_day"]) == velocity
                and str(r["condition"]) == condition
                and int(r["cell"]) == N_CELLS
            )
            lines.append(
                "| "
                f"{velocity:.0f} | "
                f"{condition} | "
                f"{float(row['mobile_fraction']):.4f} | "
                f"{float(row['attached_fraction']):.4f} | "
                f"{float(row['censored_fraction']):.4f} | "
                f"{float(row['plume_std_y_um']):.2f} | "
                f"{float(row['median_time_s']) / 3600.0:.3f} | "
                f"{float(row['mean_cumulative_interceptions_mobile']):.3f} | "
                f"{float(row['one_cell_attached_fraction']):.4f} | "
                f"{float(row['one_cell_censored_fraction']):.4f} | "
                f"{int(row['min_96_bin_support'])} |"
            )

    lines.extend(["", "## Favorable retained-mass peaks", "", "| velocity (m/day) | peak cell | peak new attached fraction |", "|---:|---:|---:|"])
    for velocity in (2.0, 4.0, 8.0):
        rows = [r for r in retention_rows if float(r["velocity_m_per_day"]) == velocity and str(r["condition"]) == "favorable"]
        values = np.array([float(r["new_attached_fraction"]) for r in rows])
        cells = np.array([int(r["cell"]) for r in rows])
        peak_id = int(np.argmax(values))
        lines.append(f"| {velocity:.0f} | {int(cells[peak_id])} | {float(values[peak_id]):.4f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict[str, float | int | str]] = []
    all_retention: list[dict[str, float | int | str]] = []
    for case in CASES:
        print(f"Upscaling {case.condition} at {case.velocity_m_per_day:.0f} m/day")
        summary, retention, _ = run_case(case)
        all_summary.extend(summary)
        all_retention.extend(retention)

    write_dicts(OUT / "velocity_upscaling_summary.csv", all_summary)
    write_dicts(OUT / "velocity_upscaling_retention.csv", all_retention)
    write_report(OUT / "velocity_upscaling_report.md", all_summary, all_retention)
    write_figure(OUT / "velocity_upscaling.png", all_summary, all_retention)
    write_figure(FIGURES / "fig14_velocity_upscaling.png", all_summary, all_retention)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
