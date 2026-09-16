#!/usr/bin/env python3
"""Sweep Markov conditioning-bin resolution against direct multi-cell tracking."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import PhysicalParams, TrajectoryLibrary, load_library


PHYSICAL_OUT = ROOT / "outputs" / "physical"
FOCUSING_OUT = ROOT / "outputs" / "focusing"
VALIDATION_OUT = ROOT / "outputs" / "validation"
FIGURES = ROOT / "outputs" / "figures"

BIN_COUNTS = (16, 24, 32, 48, 64, 96, 128)
REPLICATES = 16
N_CELLS = 25
N_PARTICLES = 3000

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


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def initial_positions(params: PhysicalParams) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(50001)
    return {
        case.key: rng.uniform(params.inlet_y_min, params.inlet_y_max, size=N_PARTICLES)
        for case in CASES
    }


def bin_records(library: TrajectoryLibrary, n_bins: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    bins = np.linspace(0.0, library.params.cell_length, n_bins + 1)
    yin_bin = np.clip(np.digitize(library.y_in, bins) - 1, 0, n_bins - 1)
    by_bin = [np.flatnonzero(yin_bin == i) for i in range(n_bins)]
    nonempty = np.flatnonzero([idx.size > 0 for idx in by_bin])
    nearest = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        nearest[i] = int(nonempty[np.argmin(np.abs(nonempty - i))])
    return bins, by_bin, nearest


def sample_from_bins(
    library: TrajectoryLibrary,
    n_bins: int,
    by_bin: list[np.ndarray],
    nearest: np.ndarray,
    bins: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    ybin = np.clip(np.digitize(y, bins) - 1, 0, n_bins - 1)
    sampled = np.empty(y.size, dtype=int)
    for bin_id in np.unique(ybin):
        mask = ybin == bin_id
        source_bin = int(bin_id) if by_bin[int(bin_id)].size else int(nearest[int(bin_id)])
        sampled[mask] = rng.choice(by_bin[source_bin], size=int(np.sum(mask)), replace=True)
    return sampled


def summarize(
    case: Case,
    cell: int,
    y: np.ndarray,
    time: np.ndarray,
    alive: np.ndarray,
    attached: np.ndarray,
    censored: np.ndarray,
) -> dict[str, float | int | str]:
    mobile_ids = np.flatnonzero(alive)
    mobile_time = time[mobile_ids]
    return {
        "case": case.key,
        "method": "markov",
        "cell": cell,
        "mobile_fraction": float(np.mean(alive)),
        "attached_fraction": float(np.mean(attached)),
        "censored_fraction": float(np.mean(censored)),
        "plume_std_y_um": float(np.std(y[mobile_ids]) * 1.0e6) if mobile_ids.size else float("nan"),
        "median_time_s": float(np.median(mobile_time)) if mobile_time.size else float("nan"),
    }


def run_markov(
    case: Case,
    library: TrajectoryLibrary,
    n_bins: int,
    replicate: int,
    initial_y: np.ndarray,
) -> list[dict[str, float | int | str]]:
    bins, by_bin, nearest = bin_records(library, n_bins)
    rng = np.random.default_rng(81000 + 1000 * replicate + 17 * n_bins + sum(ord(ch) for ch in case.key))
    y = initial_y.copy()
    time = np.zeros(N_PARTICLES, dtype=float)
    alive = np.ones(N_PARTICLES, dtype=bool)
    attached = np.zeros(N_PARTICLES, dtype=bool)
    censored = np.zeros(N_PARTICLES, dtype=bool)
    rows = [summarize(case, 0, y, time, alive, attached, censored)]
    fallback_travel_time = float(np.nanmedian(library.travel_time[library.exited]))

    for cell in range(1, N_CELLS + 1):
        active_ids = np.flatnonzero(alive)
        if active_ids.size:
            picks = sample_from_bins(library, n_bins, by_bin, nearest, bins, y[active_ids], rng)
            picked_attached = library.attached[picks]
            picked_exited = library.exited[picks] & (~library.attached[picks])
            picked_censored = ~(picked_attached | picked_exited)
            if np.any(picked_attached):
                ids = active_ids[picked_attached]
                attached[ids] = True
                alive[ids] = False
            if np.any(picked_censored):
                ids = active_ids[picked_censored]
                censored[ids] = True
                alive[ids] = False
            if np.any(picked_exited):
                ids = active_ids[picked_exited]
                selected = picks[picked_exited]
                y[ids] = library.y_out[selected]
                dt = library.travel_time[selected]
                time[ids] += np.where(np.isfinite(dt), dt, fallback_travel_time)
        rows.append(summarize(case, cell, y, time, alive, attached, censored))
    return rows


def series(metric: str, rows: dict[int, dict[str, float | int | str]], cells: list[int]) -> np.ndarray:
    return np.array([float(rows[cell][metric]) for cell in cells], dtype=float)


def compare_case(
    direct_rows: list[dict[str, str]],
    markov_rows: list[dict[str, float | int | str]],
    case: Case,
    n_bins: int,
    replicate: int,
) -> dict[str, float | int | str]:
    direct = {int(row["cell"]): row for row in direct_rows if row["case"] == case.key and row["method"] == "direct"}
    markov = {int(row["cell"]): row for row in markov_rows if row["case"] == case.key}
    cells = sorted(set(direct) & set(markov))
    d_mobile = series("mobile_fraction", direct, cells)
    m_mobile = series("mobile_fraction", markov, cells)
    d_attached = series("attached_fraction", direct, cells)
    m_attached = series("attached_fraction", markov, cells)
    d_censored = series("censored_fraction", direct, cells)
    m_censored = series("censored_fraction", markov, cells)
    d_width = series("plume_std_y_um", direct, cells)
    m_width = series("plume_std_y_um", markov, cells)
    d_time = series("median_time_s", direct, cells)
    m_time = series("median_time_s", markov, cells)
    final_cell = max(cells)
    return {
        "n_bins": n_bins,
        "replicate": replicate,
        "case": case.key,
        "max_abs_mobile_fraction_error": float(np.nanmax(np.abs(d_mobile - m_mobile))),
        "max_abs_attached_fraction_error": float(np.nanmax(np.abs(d_attached - m_attached))),
        "max_abs_censored_fraction_error": float(np.nanmax(np.abs(d_censored - m_censored))),
        "plume_std_rmse_um": float(np.sqrt(np.nanmean((d_width - m_width) ** 2))),
        "median_time_rmse_s": float(np.sqrt(np.nanmean((d_time - m_time) ** 2))),
        "final_direct_mobile": float(direct[final_cell]["mobile_fraction"]),
        "final_markov_mobile": float(markov[final_cell]["mobile_fraction"]),
        "final_direct_attached": float(direct[final_cell]["attached_fraction"]),
        "final_markov_attached": float(markov[final_cell]["attached_fraction"]),
        "final_direct_censored": float(direct[final_cell]["censored_fraction"]),
        "final_markov_censored": float(markov[final_cell]["censored_fraction"]),
    }


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    metrics = (
        "max_abs_mobile_fraction_error",
        "max_abs_attached_fraction_error",
        "max_abs_censored_fraction_error",
        "plume_std_rmse_um",
        "median_time_rmse_s",
        "final_markov_mobile",
        "final_markov_attached",
        "final_markov_censored",
    )
    out: list[dict[str, float | int | str]] = []
    for n_bins in BIN_COUNTS:
        for case in CASES:
            subset = [row for row in rows if int(row["n_bins"]) == n_bins and row["case"] == case.key]
            item: dict[str, float | int | str] = {"n_bins": n_bins, "case": case.key, "replicates": len(subset)}
            for metric in metrics:
                values = np.array([float(row[metric]) for row in subset], dtype=float)
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                item[f"{metric}_min"] = float(np.min(values))
                item[f"{metric}_max"] = float(np.max(values))
            out.append(item)
    return out


def axis_map(
    rect: tuple[int, int, int, int],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple:
    x0, y0, x1, y1 = rect

    def sx(x: float) -> float:
        return x0 + (x - xlim[0]) / max(xlim[1] - xlim[0], 1.0e-12) * (x1 - x0)

    def sy(y: float) -> float:
        return y1 - (y - ylim[0]) / max(ylim[1] - ylim[0], 1.0e-12) * (y1 - y0)

    return sx, sy


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    agg_rows: list[dict[str, float | int | str]],
    metric_base: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=2)
    draw.text(((x0 + x1) / 2, y0 - 26), title, fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 46), "conditioning bins", fill=INK, font=font(17), anchor="mm")
    draw.text((x0 + 12, y0 + 15), ylabel, fill=MUTED, font=font(13), anchor="la")

    x_values = np.array(BIN_COUNTS, dtype=float)
    if ylim is None:
        vals = [float(row[f"{metric_base}_mean"]) for row in agg_rows]
        ymax = max(vals) * 1.18 if vals else 1.0
        ylim = (0.0, max(ymax, 1.0e-6))
    sx, sy = axis_map(rect, (float(min(BIN_COUNTS)), float(max(BIN_COUNTS))), ylim)

    for tick in BIN_COUNTS:
        px = sx(float(tick))
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 25), str(tick), fill=INK, font=font(12), anchor="mm")
    for tick in np.linspace(ylim[0], ylim[1], 5):
        py = sy(float(tick))
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((x0 - 10, py), f"{tick:.3g}", fill=INK, font=font(12), anchor="rm")

    for case in CASES:
        rows = [row for row in agg_rows if row["case"] == case.key]
        rows = sorted(rows, key=lambda row: int(row["n_bins"]))
        mean = np.array([float(row[f"{metric_base}_mean"]) for row in rows])
        low = np.array([float(row[f"{metric_base}_min"]) for row in rows])
        high = np.array([float(row[f"{metric_base}_max"]) for row in rows])
        x = np.array([int(row["n_bins"]) for row in rows], dtype=float)
        pts = [(sx(float(a)), sy(float(b))) for a, b in zip(x, mean)]
        if len(pts) > 1:
            draw.line(pts, fill=case.color, width=5, joint="curve")
        for a, lo, hi in zip(x, low, high):
            px = sx(float(a))
            draw.line((px, sy(float(lo)), px, sy(float(hi))), fill=case.color, width=2)
            draw.ellipse((px - 4, sy(float(mean[np.where(x == a)[0][0]])) - 4, px + 4, sy(float(mean[np.where(x == a)[0][0]])) + 4), fill=case.color)


def draw_attached_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    agg_rows: list[dict[str, float | int | str]],
    direct_rows: list[dict[str, str]],
) -> None:
    case = next(c for c in CASES if c.key == "favorable")
    fav_rows = sorted([row for row in agg_rows if row["case"] == "favorable"], key=lambda row: int(row["n_bins"]))
    direct_final = next(
        float(row["attached_fraction"])
        for row in direct_rows
        if row["case"] == "favorable" and row["method"] == "direct" and int(row["cell"]) == N_CELLS
    )
    max_value = max([float(row["final_markov_attached_mean"]) for row in fav_rows] + [direct_final])
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=2)
    draw.text(((x0 + x1) / 2, y0 - 26), "favorable attachment at cell 25", fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 46), "conditioning bins", fill=INK, font=font(17), anchor="mm")
    draw.text((x0 + 12, y0 + 15), "attached fraction", fill=MUTED, font=font(13), anchor="la")
    sx, sy = axis_map(rect, (float(min(BIN_COUNTS)), float(max(BIN_COUNTS))), (0.0, max_value * 1.2))
    for tick in BIN_COUNTS:
        px = sx(float(tick))
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 25), str(tick), fill=INK, font=font(12), anchor="mm")
    for tick in np.linspace(0.0, max_value * 1.2, 5):
        py = sy(float(tick))
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((x0 - 10, py), f"{tick:.3f}", fill=INK, font=font(12), anchor="rm")
    py_direct = sy(direct_final)
    draw.line((x0, py_direct, x1, py_direct), fill=INK, width=2)
    draw.text((x1 - 10, py_direct - 12), f"direct = {direct_final:.3f}", fill=INK, font=font(14), anchor="ra")
    x = np.array([int(row["n_bins"]) for row in fav_rows], dtype=float)
    mean = np.array([float(row["final_markov_attached_mean"]) for row in fav_rows])
    pts = [(sx(float(a)), sy(float(b))) for a, b in zip(x, mean)]
    draw.line(pts, fill=case.color, width=5, joint="curve")
    for a, b in zip(x, mean):
        px, py = sx(float(a)), sy(float(b))
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=case.color)


def write_figure(
    path: Path,
    agg_rows: list[dict[str, float | int | str]],
    direct_rows: list[dict[str, str]],
) -> None:
    img = Image.new("RGB", (1800, 1320), "white")
    draw = ImageDraw.Draw(img)
    draw.text((900, 44), "Markov state-resolution convergence against direct tracking", fill=INK, font=font(34, True), anchor="mm")
    draw_panel(draw, (130, 140, 820, 500), agg_rows, "max_abs_mobile_fraction_error", "mobile-survival error", "max abs error")
    draw_attached_panel(draw, (1040, 140, 1700, 500), agg_rows, direct_rows)
    draw_panel(draw, (130, 690, 820, 1050), agg_rows, "plume_std_rmse_um", "plume-width error", "RMSE (um)")
    draw_panel(draw, (1040, 690, 1700, 1050), agg_rows, "median_time_rmse_s", "median-time error", "RMSE (s)")
    lx, ly = 620, 1135
    for i, case in enumerate(CASES):
        yy = ly + 34 * i
        draw.line((lx, yy, lx + 48, yy), fill=case.color, width=5)
        draw.text((lx + 62, yy), case.label, fill=INK, font=font(16), anchor="lm")
    draw.text((1040, 1280), f"{REPLICATES} Markov replicates per bin count; direct run fixed at {N_PARTICLES:,} particles over {N_CELLS} cells", fill=MUTED, font=font(18), anchor="mm")
    img.save(path)


def write_report(
    path: Path,
    agg_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Markov state-resolution sweep",
        "",
        f"The sweep compares Markov stitching at {', '.join(str(b) for b in BIN_COUNTS)} transverse conditioning bins against the saved direct 25-cell validation run.",
        f"Each bin count uses {REPLICATES} Markov replicates with {N_PARTICLES:,} particles.",
        "",
        "| bins | case | mobile error mean | mobile error range | final Markov mobile mean | plume RMSE mean (um) | time RMSE mean (s) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            "| "
            f"{int(row['n_bins'])} | "
            f"{row['case']} | "
            f"{float(row['max_abs_mobile_fraction_error_mean']):.4f} | "
            f"{float(row['max_abs_mobile_fraction_error_min']):.4f}-{float(row['max_abs_mobile_fraction_error_max']):.4f} | "
            f"{float(row['final_markov_mobile_mean']):.4f} | "
            f"{float(row['plume_std_rmse_um_mean']):.3f} | "
            f"{float(row['median_time_rmse_s_mean']):.3f} |"
        )
    fav96 = next(row for row in agg_rows if int(row["n_bins"]) == 96 and row["case"] == "favorable")
    fav32 = next(row for row in agg_rows if int(row["n_bins"]) == 32 and row["case"] == "favorable")
    lines.extend(
        [
            "",
            f"For favorable attachment, the mean maximum mobile-survival error decreases from {float(fav32['max_abs_mobile_fraction_error_mean']):.4f} at 32 bins to {float(fav96['max_abs_mobile_fraction_error_mean']):.4f} at 96 bins.",
            "This supports using 96 bins for the many-cell stitched calculations while keeping the 32-bin matrices as compact diagnostic figures.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    direct_path = VALIDATION_OUT / "multicell_validation_summary.csv"
    if not direct_path.exists():
        raise FileNotFoundError(f"Run scripts/run_multicell_validation.py first: {direct_path}")
    direct_rows = read_dicts(direct_path)
    params = PhysicalParams()
    y0 = initial_positions(params)
    libraries = {case.key: load_library(case.library_path) for case in CASES}

    rows: list[dict[str, float | int | str]] = []
    for n_bins in BIN_COUNTS:
        print(f"State-resolution sweep: {n_bins} bins")
        for replicate in range(REPLICATES):
            for case in CASES:
                markov_rows = run_markov(case, libraries[case.key], n_bins, replicate, y0[case.key])
                rows.append(compare_case(direct_rows, markov_rows, case, n_bins, replicate))
    agg_rows = aggregate(rows)
    write_dicts(VALIDATION_OUT / "state_resolution_sweep.csv", rows)
    write_dicts(VALIDATION_OUT / "state_resolution_sweep_summary.csv", agg_rows)
    write_report(VALIDATION_OUT / "state_resolution_sweep_report.md", agg_rows)
    write_figure(VALIDATION_OUT / "state_resolution_sweep.png", agg_rows, direct_rows)
    write_figure(FIGURES / "fig11_state_resolution_sweep.png", agg_rows, direct_rows)
    print("Done. Outputs written to", VALIDATION_OUT)


if __name__ == "__main__":
    main()
