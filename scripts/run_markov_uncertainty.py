#!/usr/bin/env python3
"""Replicate Markov stitching to quantify upscaled prediction uncertainty."""

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
N_PARTICLES = 25_000
REPLICATES = 48
REPORT_CELLS = (25, 50, 100, 120)

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    color: str
    library_path: Path


CASES = (
    Case("neutral", "no DLVO", BLUE, PHYSICAL_OUT / "trajectory_library_neutral_4m_per_day_180s.npz"),
    Case("favorable", "favorable attach", RED, PHYSICAL_OUT / "trajectory_library_favorable_4m_per_day.npz"),
    Case("unfavorable", "unfavorable reflect", GREEN, PHYSICAL_OUT / "trajectory_library_unfavorable_4m_per_day_180s.npz"),
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


def initial_positions(
    library: TrajectoryLibrary,
    bins: np.ndarray,
    by_bin: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    open_bins = np.array([i for i, ids in enumerate(by_bin) if ids.size > 0], dtype=int)
    counts = np.array([by_bin[i].size for i in open_bins], dtype=float)
    probabilities = counts / np.sum(counts)
    chosen_bins = rng.choice(open_bins, size=N_PARTICLES, p=probabilities)
    y = np.empty(N_PARTICLES, dtype=float)
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
        source_bin = int(bin_id) if by_bin[int(bin_id)].size else int(nearest[int(bin_id)])
        candidates = by_bin[source_bin]
        sampled[mask] = rng.choice(candidates, size=int(np.sum(mask)), replace=True)
    return sampled


def run_replicate(
    case: Case,
    library: TrajectoryLibrary,
    replicate: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(93000 + 1000 * replicate + sum(ord(ch) for ch in case.key))
    bins, by_bin, nearest = bin_records(library)
    y = initial_positions(library, bins, by_bin, rng)
    time = np.zeros(N_PARTICLES, dtype=float)
    alive = np.ones(N_PARTICLES, dtype=bool)
    attached = np.zeros(N_PARTICLES, dtype=bool)
    censored = np.zeros(N_PARTICLES, dtype=bool)
    attached_cell = np.full(N_PARTICLES, -1, dtype=int)
    censored_cell = np.full(N_PARTICLES, -1, dtype=int)
    fallback_travel_time = float(np.nanmedian(library.travel_time[library.exited]))
    rows: list[dict[str, float | int | str]] = []

    for cell in range(0, N_CELLS + 1):
        if cell > 0:
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
                    time[ids] += np.where(np.isfinite(dt), dt, fallback_travel_time)

        mobile_ids = np.flatnonzero(alive)
        mobile_time = time[mobile_ids]
        rows.append(
            {
                "case": case.key,
                "replicate": replicate,
                "cell": cell,
                "mobile_fraction": float(np.mean(alive)),
                "attached_fraction": float(np.mean(attached)),
                "censored_fraction": float(np.mean(censored)),
                "new_attached_fraction": float(np.mean(attached_cell == cell)) if cell > 0 else 0.0,
                "new_censored_fraction": float(np.mean(censored_cell == cell)) if cell > 0 else 0.0,
                "plume_std_y_um": float(np.std(y[mobile_ids]) * 1.0e6) if mobile_ids.size else float("nan"),
                "median_time_s": float(np.median(mobile_time)) if mobile_ids.size else float("nan"),
            }
        )
    return rows


def summarize_replicates(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    metrics = (
        "mobile_fraction",
        "attached_fraction",
        "censored_fraction",
        "new_attached_fraction",
        "new_censored_fraction",
        "plume_std_y_um",
        "median_time_s",
    )
    out: list[dict[str, float | int | str]] = []
    for case in CASES:
        for cell in range(N_CELLS + 1):
            subset = [row for row in rows if row["case"] == case.key and int(row["cell"]) == cell]
            item: dict[str, float | int | str] = {"case": case.key, "cell": cell, "replicates": len(subset)}
            for metric in metrics:
                values = np.array([float(row[metric]) for row in subset], dtype=float)
                values = values[np.isfinite(values)]
                item[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
                item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                item[f"{metric}_q05"] = float(np.quantile(values, 0.05)) if values.size else float("nan")
                item[f"{metric}_q95"] = float(np.quantile(values, 0.95)) if values.size else float("nan")
            out.append(item)
    return out


def line_points(
    rect: tuple[int, int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = rect
    return [
        (
            x0 + (float(a) - xlim[0]) / max(xlim[1] - xlim[0], 1.0e-12) * (x1 - x0),
            y1 - (float(b) - ylim[0]) / max(ylim[1] - ylim[0], 1.0e-12) * (y1 - y0),
        )
        for a, b in zip(x, y)
        if np.isfinite(b)
    ]


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    rows: list[dict[str, float | int | str]],
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=2)
    draw.text(((x0 + x1) / 2, y0 - 26), title, fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 44), "cell number", fill=INK, font=font(17), anchor="mm")
    draw.text((x0 + 12, y0 + 15), ylabel, fill=MUTED, font=font(13), anchor="la")
    xlim = (0.0, float(N_CELLS))
    if ylim is None:
        values = [float(row[f"{metric}_q05"]) for row in rows] + [float(row[f"{metric}_q95"]) for row in rows]
        values = [v for v in values if np.isfinite(v)]
        ymin, ymax = min(values), max(values)
        pad = 0.08 * max(ymax - ymin, 1.0e-12)
        starts_at_zero = metric.endswith("_fraction") or metric == "median_time_s"
        lower = 0.0 if starts_at_zero and ymin >= 0.0 else ymin - pad
        ylim = (lower, ymax + pad)
    for tick in np.linspace(xlim[0], xlim[1], 5):
        px = x0 + (tick - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 25), f"{tick:.0f}", fill=INK, font=font(12), anchor="mm")
    for tick in np.linspace(ylim[0], ylim[1], 5):
        py = y1 - (tick - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((x0 - 10, py), f"{tick:.3g}", fill=INK, font=font(12), anchor="rm")
    for case in CASES:
        subset = sorted([row for row in rows if row["case"] == case.key], key=lambda row: int(row["cell"]))
        x = np.array([int(row["cell"]) for row in subset], dtype=float)
        q05 = np.array([float(row[f"{metric}_q05"]) for row in subset], dtype=float)
        q95 = np.array([float(row[f"{metric}_q95"]) for row in subset], dtype=float)
        mean = np.array([float(row[f"{metric}_mean"]) for row in subset], dtype=float)
        band = line_points(rect, x, q05, xlim, ylim) + list(reversed(line_points(rect, x, q95, xlim, ylim)))
        if len(band) > 2:
            draw.polygon(band, fill=case.color + "33")
        pts = line_points(rect, x, mean, xlim, ylim)
        if len(pts) > 1:
            draw.line(pts, fill=case.color, width=5, joint="curve")


def write_figure(path: Path, summary_rows: list[dict[str, float | int | str]]) -> None:
    img = Image.new("RGB", (1800, 1320), "white")
    draw = ImageDraw.Draw(img)
    draw.text((900, 44), "Replicate uncertainty in 120-cell Markov upscaling", fill=INK, font=font(34, True), anchor="mm")
    draw_panel(draw, (130, 140, 820, 500), summary_rows, "mobile_fraction", "mobile survival", "mobile fraction", ylim=(0.0, 1.0))
    draw_panel(draw, (1040, 140, 1700, 500), summary_rows, "attached_fraction", "cumulative attachment", "attached fraction", ylim=(0.0, 0.5))
    draw_panel(draw, (130, 690, 820, 1050), summary_rows, "plume_std_y_um", "mobile plume width", "std(y) (um)")
    draw_panel(draw, (1040, 690, 1700, 1050), summary_rows, "median_time_s", "median mobile time", "time (s)")
    lx, ly = 635, 1140
    for i, case in enumerate(CASES):
        yy = ly + 34 * i
        draw.line((lx, yy, lx + 48, yy), fill=case.color, width=5)
        draw.text((lx + 62, yy), case.label, fill=INK, font=font(16), anchor="lm")
    draw.text((1040, 1280), f"Mean lines and 5-95% replicate bands; {REPLICATES} replicates, {N_PARTICLES:,} particles each", fill=MUTED, font=font(18), anchor="mm")
    img.save(path)


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        "# Markov upscaling replicate uncertainty",
        "",
        f"The uncertainty calculation runs {REPLICATES} Markov-stitching replicates with {N_PARTICLES:,} particles each, using {N_BINS} transverse conditioning bins over {N_CELLS} cells.",
        "",
        "| case | cell | mobile mean | mobile 5-95% | attached mean | attached 5-95% | censored mean | plume std mean (um) | median time mean (h) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wanted = set(REPORT_CELLS)
    for row in rows:
        if int(row["cell"]) not in wanted:
            continue
        lines.append(
            "| "
            f"{row['case']} | "
            f"{int(row['cell'])} | "
            f"{float(row['mobile_fraction_mean']):.4f} | "
            f"{float(row['mobile_fraction_q05']):.4f}-{float(row['mobile_fraction_q95']):.4f} | "
            f"{float(row['attached_fraction_mean']):.4f} | "
            f"{float(row['attached_fraction_q05']):.4f}-{float(row['attached_fraction_q95']):.4f} | "
            f"{float(row['censored_fraction_mean']):.4f} | "
            f"{float(row['plume_std_y_um_mean']):.2f} | "
            f"{float(row['median_time_s_mean']) / 3600.0:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    libraries = {case.key: load_library(case.library_path) for case in CASES}
    rows: list[dict[str, float | int | str]] = []
    for replicate in range(REPLICATES):
        print(f"Uncertainty replicate {replicate + 1}/{REPLICATES}")
        for case in CASES:
            rows.extend(run_replicate(case, libraries[case.key], replicate))
    summary_rows = summarize_replicates(rows)
    write_dicts(OUT / "markov_uncertainty_replicates.csv", rows)
    write_dicts(OUT / "markov_uncertainty_summary.csv", summary_rows)
    write_report(OUT / "markov_uncertainty_report.md", summary_rows)
    write_figure(OUT / "markov_uncertainty.png", summary_rows)
    write_figure(FIGURES / "fig12_markov_uncertainty.png", summary_rows)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
