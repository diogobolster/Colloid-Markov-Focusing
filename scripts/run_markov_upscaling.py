#!/usr/bin/env python3
"""Stitch one-cell trajectory libraries into multi-cell Markov predictions."""

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
FOCUSING_OUT = ROOT / "outputs" / "focusing"
OUT = ROOT / "outputs" / "upscaling"
FIGURES = ROOT / "outputs" / "figures"

N_BINS = 96
N_CELLS = 120
N_PARTICLES = 50_000
TARGET_CELLS = (25, 50, 100)

INK = "#111827"
MUTED = "#6b7280"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
GOLD = "#d97706"


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    color: str
    library_path: Path


CASES = (
    Case(
        key="neutral",
        label="no DLVO",
        color=BLUE,
        library_path=PHYSICAL_OUT / "trajectory_library_neutral_4m_per_day_180s.npz",
    ),
    Case(
        key="favorable",
        label="favorable attach",
        color=RED,
        library_path=PHYSICAL_OUT / "trajectory_library_favorable_4m_per_day.npz",
    ),
    Case(
        key="unfavorable",
        label="unfavorable reflect",
        color=GREEN,
        library_path=PHYSICAL_OUT / "trajectory_library_unfavorable_4m_per_day_180s.npz",
    ),
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


def bin_records(library: TrajectoryLibrary) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    bins = np.linspace(0.0, library.params.cell_length, N_BINS + 1)
    yin_bin = np.clip(np.digitize(library.y_in, bins) - 1, 0, N_BINS - 1)
    by_bin = [np.flatnonzero(yin_bin == i) for i in range(N_BINS)]
    nonempty = np.flatnonzero([idx.size > 0 for idx in by_bin])
    nearest = np.zeros(N_BINS, dtype=int)
    for i in range(N_BINS):
        nearest[i] = int(nonempty[np.argmin(np.abs(nonempty - i))])
    return bins, by_bin, nearest


def initial_positions(library: TrajectoryLibrary, rng: np.random.Generator, n_particles: int) -> np.ndarray:
    bins, by_bin, _ = bin_records(library)
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


def run_case(case: Case) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]], dict[int, np.ndarray]]:
    library = load_library(case.library_path)
    rng = np.random.default_rng(41000 + sum(ord(ch) for ch in case.key))
    bins, by_bin, nearest = bin_records(library)
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

    rows.append(
        {
            "case": case.key,
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
        }
    )

    fallback_travel_time = float(np.nanmedian(library.travel_time[library.exited]))
    for cell in range(1, N_CELLS + 1):
        active_ids = np.flatnonzero(alive)
        if active_ids.size:
            picks = sample_from_bins(library, by_bin, nearest, bins, y[active_ids], rng)
            picked_attached = library.attached[picks]
            picked_exited = library.exited[picks] & (~library.attached[picks])
            picked_censored = ~(picked_attached | picked_exited)

            if np.any(picked_attached):
                ids = active_ids[picked_attached]
                attached[ids] = True
                alive[ids] = False
                attached_cell[ids] = cell
                cumulative_interceptions[ids] += library.interceptions[picks[picked_attached]]
                cumulative_near_time[ids] += library.near_time[picks[picked_attached]]

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
                "case": case.key,
                "cell": cell,
                "mobile_fraction": float(np.mean(alive)),
                "attached_fraction": float(np.mean(attached)),
                "censored_fraction": float(np.mean(censored)),
                "plume_std_y_um": float(np.std(y[mobile_ids]) * 1.0e6) if mobile_ids.size else float("nan"),
                "median_time_s": q50,
                "q05_time_s": q05,
                "q95_time_s": q95,
                "mean_cumulative_interceptions_mobile": float(np.mean(cumulative_interceptions[mobile_ids])) if mobile_ids.size else float("nan"),
                "mean_cumulative_near_time_mobile_s": float(np.mean(cumulative_near_time[mobile_ids])) if mobile_ids.size else float("nan"),
            }
        )

    retention_rows: list[dict[str, float | int | str]] = []
    for cell in range(1, N_CELLS + 1):
        retention_rows.append(
            {
                "case": case.key,
                "cell": cell,
                "new_attached_fraction": float(np.mean(attached_cell == cell)),
                "new_censored_fraction": float(np.mean(censored_cell == cell)),
            }
        )
    return rows, retention_rows, times_at_target


def panel_axes(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str, xlabel: str, ylabel: str) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=2)
    draw.text(((x0 + x1) / 2, y0 - 28), title, fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 42), xlabel, fill=INK, font=font(18), anchor="mm")
    draw.text((x0 + 12, y0 + 14), ylabel, fill=MUTED, font=font(13), anchor="la")


def draw_line_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    series: list[tuple[str, str, np.ndarray, np.ndarray]],
    title: str,
    xlabel: str,
    ylabel: str,
    y_limits: tuple[float, float] | None = None,
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

    x_tick_format = "{:.0f}" if abs(xmax - xmin) >= 5.0 else "{:.2f}"
    for tick in np.linspace(xmin, xmax, 5):
        px = sx(float(tick))
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 24), x_tick_format.format(float(tick)), fill=INK, font=font(13), anchor="mm")
    for tick in np.linspace(ymin, ymax, 5):
        py = sy(float(tick))
        draw.line((x0 - 7, py, x0, py), fill=INK, width=2)
        draw.line((x0, py, x1, py), fill="#e5e7eb", width=1)
        draw.text((x0 - 11, py), f"{tick:.2g}", fill=INK, font=font(13), anchor="rm")

    for label, color, x, y in series:
        finite = np.isfinite(y)
        pts = [(sx(float(a)), sy(float(b))) for a, b in zip(x[finite], y[finite])]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4, joint="curve")
    lx, ly = x1 - 230, y0 + 24
    for i, (label, color, _, _) in enumerate(series):
        yy = ly + 28 * i
        draw.line((lx, yy, lx + 38, yy), fill=color, width=5)
        draw.text((lx + 50, yy), label, fill=INK, font=font(14), anchor="lm")


def draw_retention_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    retention_rows: list[dict[str, float | int | str]],
) -> None:
    x0, y0, x1, y1 = rect
    panel_axes(draw, rect, "favorable retained mass by cell", "cell number", "new attached fraction")
    rows = [r for r in retention_rows if r["case"] == "favorable"]
    cells = np.array([int(r["cell"]) for r in rows])
    values = np.array([float(r["new_attached_fraction"]) for r in rows])
    ymin, ymax = 0.0, max(float(np.max(values)) * 1.15, 1.0e-4)

    def sx(v: float) -> float:
        return x0 + (v - 1.0) / (N_CELLS - 1.0) * (x1 - x0)

    def sy(v: float) -> float:
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    for tick in np.linspace(0.0, ymax, 5):
        py = sy(float(tick))
        draw.line((x0, py, x1, py), fill="#e5e7eb", width=1)
        draw.text((x0 - 11, py), f"{tick:.3f}", fill=INK, font=font(13), anchor="rm")
    for tick in np.linspace(1, N_CELLS, 5):
        px = sx(float(tick))
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 24), f"{tick:.0f}", fill=INK, font=font(13), anchor="mm")

    pts = [(sx(float(c)), sy(float(v))) for c, v in zip(cells, values)]
    draw.line(pts, fill=RED, width=4, joint="curve")
    peak_cell = int(cells[int(np.argmax(values))])
    peak_value = float(np.max(values))
    draw.text((x1 - 220, y0 + 35), f"mode cell = {peak_cell}\npeak = {peak_value:.3f}", fill=INK, font=font(15), anchor="lm")


def draw_cdf_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    times_at_target: dict[str, dict[int, np.ndarray]],
    target_cell: int = 50,
) -> None:
    series = []
    for case in CASES:
        times = times_at_target[case.key][target_cell] / 60.0
        times = np.sort(times[np.isfinite(times)])
        if times.size == 0:
            continue
        cdf = np.linspace(1.0 / times.size, 1.0, times.size)
        series.append((case.label, case.color, times, cdf))
    draw_line_panel(
        draw,
        rect,
        series,
        f"breakthrough CDF at cell {target_cell}",
        "time (min)",
        "mobile CDF",
        y_limits=(0.0, 1.0),
    )


def write_figure(
    path: Path,
    summary_rows: list[dict[str, float | int | str]],
    retention_rows: list[dict[str, float | int | str]],
    times_at_target: dict[str, dict[int, np.ndarray]],
) -> None:
    by_case = {case.key: [r for r in summary_rows if r["case"] == case.key] for case in CASES}
    img = Image.new("RGB", (1800, 1320), "white")
    draw = ImageDraw.Draw(img)
    draw.text((900, 44), "Trajectory-library Markov upscaling from one resolved cell", fill=INK, font=font(38, True), anchor="mm")

    survival_series = []
    width_series = []
    interception_series = []
    for case in CASES:
        rows = by_case[case.key]
        cells = np.array([int(r["cell"]) for r in rows])
        survival = np.array([float(r["mobile_fraction"]) for r in rows])
        width = np.array([float(r["plume_std_y_um"]) for r in rows])
        interceptions = np.array([float(r["mean_cumulative_interceptions_mobile"]) for r in rows])
        survival_series.append((case.label, case.color, cells, survival))
        width_series.append((case.label, case.color, cells, width))
        interception_series.append((case.label, case.color, cells, interceptions))

    draw_line_panel(
        draw,
        (130, 140, 820, 500),
        survival_series,
        "mobile survival",
        "cell number",
        "mobile fraction",
        y_limits=(0.0, 1.0),
    )
    draw_retention_panel(draw, (1040, 140, 1700, 500), retention_rows)
    draw_line_panel(
        draw,
        (130, 690, 820, 1050),
        width_series,
        "transverse plume width",
        "cell number",
        "std(y) (um)",
    )
    draw_cdf_panel(draw, (1040, 690, 1700, 1050), times_at_target, target_cell=50)
    draw.text(
        (900, 1240),
        "Each step samples a one-cell trajectory conditioned on the current inlet bin; attachment is absorbing and unfavorable reflection remains mobile.",
        fill=MUTED,
        font=font(22),
        anchor="mm",
    )
    img.save(path)


def write_report(
    path: Path,
    summary_rows: list[dict[str, float | int | str]],
    retention_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Markov upscaling from one-cell trajectory libraries",
        "",
        f"Each case stitches {N_PARTICLES:,} particles through {N_CELLS} cells at 4 m/day by sampling one-cell trajectories conditioned on the current inlet bin. The stitching state uses {N_BINS} transverse inlet bins.",
        "",
        "| case | cell | mobile | attached | censored | plume std (um) | median time (h) | mean interceptions/mobile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wanted = {0, 10, 25, 50, 100, 120}
    for row in summary_rows:
        if int(row["cell"]) not in wanted:
            continue
        lines.append(
            "| "
            f"{row['case']} | "
            f"{int(row['cell'])} | "
            f"{float(row['mobile_fraction']):.4f} | "
            f"{float(row['attached_fraction']):.4f} | "
            f"{float(row['censored_fraction']):.4f} | "
            f"{float(row['plume_std_y_um']):.2f} | "
            f"{float(row['median_time_s']) / 3600.0:.3f} | "
            f"{float(row['mean_cumulative_interceptions_mobile']):.3f} |"
        )
    fav_retention = [r for r in retention_rows if r["case"] == "favorable"]
    if fav_retention:
        values = np.array([float(r["new_attached_fraction"]) for r in fav_retention])
        cells = np.array([int(r["cell"]) for r in fav_retention])
        peak = int(cells[int(np.argmax(values))])
        lines.extend(
            [
                "",
                f"The favorable retained-mass profile peaks at cell {peak}, with {float(np.max(values)):.4f} of the initial ensemble attaching in that cell.",
                "The homogeneous no-DLVO and unfavorable cases use the long-horizon nonattaching libraries and remain fully mobile in this production stitch.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, float | int | str]] = []
    all_retention: list[dict[str, float | int | str]] = []
    times_by_case: dict[str, dict[int, np.ndarray]] = {}
    for case in CASES:
        print(f"Upscaling case: {case.key}")
        summary, retention, times = run_case(case)
        all_summary.extend(summary)
        all_retention.extend(retention)
        times_by_case[case.key] = times
    write_dicts(OUT / "markov_upscaling_summary.csv", all_summary)
    write_dicts(OUT / "markov_upscaling_retention.csv", all_retention)
    write_report(OUT / "markov_upscaling_report.md", all_summary, all_retention)
    figure_path = OUT / "markov_upscaling.png"
    write_figure(figure_path, all_summary, all_retention, times_by_case)
    figure_path = FIGURES / "fig09_markov_upscaling.png"
    write_figure(figure_path, all_summary, all_retention, times_by_case)
    legacy_path = FIGURES / "fig09_rough_markov_upscaling.png"
    write_figure(legacy_path, all_summary, all_retention, times_by_case)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
