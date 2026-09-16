#!/usr/bin/env python3
"""Compare trajectory-library Markov stitching with direct repeated-cell tracking."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import (
    FlowField,
    PhysicalParams,
    TrajectoryLibrary,
    load_flow,
    load_library,
    rescale_flow,
    simulate_cell_transitions,
)


PHYSICAL_OUT = ROOT / "outputs" / "physical"
FOCUSING_OUT = ROOT / "outputs" / "focusing"
OUT = ROOT / "outputs" / "validation"
FIGURES = ROOT / "outputs" / "figures"

N_BINS = 96
N_CELLS = 25
N_PARTICLES = 3000
VELOCITY_M_PER_DAY = 4.0

INK = "#111827"
MUTED = "#6b7280"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    color: str
    condition: str
    surface_mode: str
    allow_attachment: bool
    direct_max_time_s: float
    library_path: Path


CASES = (
    Case(
        key="neutral",
        label="no DLVO",
        color=BLUE,
        condition="neutral",
        surface_mode="lubrication",
        allow_attachment=False,
        direct_max_time_s=180.0,
        library_path=PHYSICAL_OUT / "trajectory_library_neutral_4m_per_day_180s.npz",
    ),
    Case(
        key="favorable",
        label="favorable attach",
        color=RED,
        condition="favorable",
        surface_mode="project",
        allow_attachment=True,
        direct_max_time_s=60.0,
        library_path=PHYSICAL_OUT / "trajectory_library_favorable_4m_per_day.npz",
    ),
    Case(
        key="unfavorable",
        label="unfavorable reflect",
        color=GREEN,
        condition="unfavorable",
        surface_mode="project",
        allow_attachment=False,
        direct_max_time_s=180.0,
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
        sampled[mask] = rng.choice(by_bin[int(source_bin)], size=int(np.sum(mask)), replace=True)
    return sampled


def summarize(
    case: Case,
    method: str,
    cell: int,
    y: np.ndarray,
    time: np.ndarray,
    alive: np.ndarray,
    attached: np.ndarray,
    censored: np.ndarray,
    cumulative_interceptions: np.ndarray,
    cumulative_near_time: np.ndarray,
) -> dict[str, float | int | str]:
    mobile_ids = np.flatnonzero(alive)
    mobile_time = time[mobile_ids]
    return {
        "case": case.key,
        "method": method,
        "cell": cell,
        "mobile_fraction": float(np.mean(alive)),
        "attached_fraction": float(np.mean(attached)),
        "censored_fraction": float(np.mean(censored)),
        "plume_std_y_um": float(np.std(y[mobile_ids]) * 1.0e6) if mobile_ids.size else float("nan"),
        "median_time_s": float(np.median(mobile_time)) if mobile_time.size else float("nan"),
        "mean_cumulative_interceptions_mobile": float(np.mean(cumulative_interceptions[mobile_ids])) if mobile_ids.size else float("nan"),
        "mean_cumulative_near_time_mobile_s": float(np.mean(cumulative_near_time[mobile_ids])) if mobile_ids.size else float("nan"),
    }


def initial_positions(params: PhysicalParams, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(params.inlet_y_min, params.inlet_y_max, size=N_PARTICLES)


def run_direct(case: Case, flow: FlowField, initial_y: np.ndarray) -> list[dict[str, float | int | str]]:
    y = initial_y.copy()
    time = np.zeros(N_PARTICLES, dtype=float)
    alive = np.ones(N_PARTICLES, dtype=bool)
    attached = np.zeros(N_PARTICLES, dtype=bool)
    censored = np.zeros(N_PARTICLES, dtype=bool)
    cumulative_interceptions = np.zeros(N_PARTICLES, dtype=int)
    cumulative_near_time = np.zeros(N_PARTICLES, dtype=float)
    rows = [summarize(case, "direct", 0, y, time, alive, attached, censored, cumulative_interceptions, cumulative_near_time)]

    for cell in range(1, N_CELLS + 1):
        active_ids = np.flatnonzero(alive)
        if active_ids.size:
            lib = simulate_cell_transitions(
                flow,
                case.condition,
                n_particles=active_ids.size,
                seed=51000 + 1000 * cell + sum(ord(ch) for ch in case.key),
                allow_attachment=case.allow_attachment,
                initial_y=y[active_ids],
                surface_mode=case.surface_mode,
            )
            hit_attached = lib.attached
            hit_censored = lib.censored
            hit_exited = lib.exited & (~lib.attached)
            if np.any(hit_attached):
                ids = active_ids[hit_attached]
                attached[ids] = True
                alive[ids] = False
                cumulative_interceptions[ids] += lib.interceptions[hit_attached]
                cumulative_near_time[ids] += lib.near_time[hit_attached]
            if np.any(hit_censored):
                ids = active_ids[hit_censored]
                censored[ids] = True
                alive[ids] = False
            if np.any(hit_exited):
                ids = active_ids[hit_exited]
                y[ids] = lib.y_out[hit_exited]
                time[ids] += lib.travel_time[hit_exited]
                cumulative_interceptions[ids] += lib.interceptions[hit_exited]
                cumulative_near_time[ids] += lib.near_time[hit_exited]
        rows.append(summarize(case, "direct", cell, y, time, alive, attached, censored, cumulative_interceptions, cumulative_near_time))
    return rows


def run_markov(case: Case, initial_y: np.ndarray) -> list[dict[str, float | int | str]]:
    library = load_library(case.library_path)
    rng = np.random.default_rng(61000 + sum(ord(ch) for ch in case.key))
    bins, by_bin, nearest = bin_records(library)
    y = initial_y.copy()
    time = np.zeros(N_PARTICLES, dtype=float)
    alive = np.ones(N_PARTICLES, dtype=bool)
    attached = np.zeros(N_PARTICLES, dtype=bool)
    censored = np.zeros(N_PARTICLES, dtype=bool)
    cumulative_interceptions = np.zeros(N_PARTICLES, dtype=int)
    cumulative_near_time = np.zeros(N_PARTICLES, dtype=float)
    rows = [summarize(case, "markov", 0, y, time, alive, attached, censored, cumulative_interceptions, cumulative_near_time)]
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
                cumulative_interceptions[ids] += library.interceptions[picks[picked_attached]]
                cumulative_near_time[ids] += library.near_time[picks[picked_attached]]
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
                cumulative_interceptions[ids] += library.interceptions[selected]
                cumulative_near_time[ids] += library.near_time[selected]
        rows.append(summarize(case, "markov", cell, y, time, alive, attached, censored, cumulative_interceptions, cumulative_near_time))
    return rows


def line_points(
    rect: tuple[int, int, int, int],
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = rect
    xmin, xmax = xlim
    ymin, ymax = ylim
    return [
        (
            x0 + (float(a) - xmin) / max(xmax - xmin, 1.0e-12) * (x1 - x0),
            y1 - (float(b) - ymin) / max(ymax - ymin, 1.0e-12) * (y1 - y0),
        )
        for a, b in zip(x, y)
        if np.isfinite(b)
    ]


def dashed_line(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color: str, width: int = 3, dash: float = 12.0) -> None:
    for p0, p1 in zip(pts[:-1], pts[1:]):
        x0, y0 = p0
        x1, y1 = p1
        dist = float(np.hypot(x1 - x0, y1 - y0))
        if dist == 0:
            continue
        n = max(int(dist / dash), 1)
        for k in range(n):
            if k % 2:
                continue
            a = k / n
            b = min((k + 1) / n, 1.0)
            draw.line((x0 + a * (x1 - x0), y0 + a * (y1 - y0), x0 + b * (x1 - x0), y0 + b * (y1 - y0)), fill=color, width=width)


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
    draw.text(((x0 + x1) / 2, y0 - 25), title, fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 40), "cell number", fill=INK, font=font(18), anchor="mm")
    draw.text((x0 + 12, y0 + 14), ylabel, fill=MUTED, font=font(13), anchor="la")
    xlim = (0.0, float(N_CELLS))
    if ylim is None:
        vals = [float(row[metric]) for row in rows if np.isfinite(float(row[metric]))]
        ymin, ymax = min(vals), max(vals)
        pad = 0.08 * max(ymax - ymin, 1.0e-12)
        ylim = (ymin - pad, ymax + pad)
    for tick in np.linspace(xlim[0], xlim[1], 6):
        px = x0 + (tick - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 24), f"{tick:.0f}", fill=INK, font=font(13), anchor="mm")
    for tick in np.linspace(ylim[0], ylim[1], 5):
        py = y1 - (tick - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
        draw.line((x0, py, x1, py), fill="#e5e7eb", width=1)
        draw.text((x0 - 10, py), f"{tick:.2g}", fill=INK, font=font(13), anchor="rm")
    for case in CASES:
        for method in ("direct", "markov"):
            subset = [row for row in rows if row["case"] == case.key and row["method"] == method]
            x = np.array([int(row["cell"]) for row in subset], dtype=float)
            y = np.array([float(row[metric]) for row in subset], dtype=float)
            pts = line_points(rect, x, y, xlim, ylim)
            if len(pts) > 1:
                if method == "direct":
                    draw.line(pts, fill=case.color, width=5, joint="curve")
                else:
                    dashed_line(draw, pts, case.color, width=3)


def write_figure(path: Path, rows: list[dict[str, float | int | str]], comparisons: list[dict[str, float | int | str]]) -> None:
    img = Image.new("RGB", (1800, 1280), "white")
    draw = ImageDraw.Draw(img)
    draw.text((900, 44), "Direct repeated-cell tracking vs trajectory-library Markov stitching", fill=INK, font=font(34, True), anchor="mm")
    draw_panel(draw, (130, 140, 820, 500), rows, "mobile_fraction", "mobile survival", "mobile fraction", ylim=(0.0, 1.0))
    draw_panel(draw, (1040, 140, 1700, 500), rows, "attached_fraction", "absorbing attachment", "attached fraction", ylim=(0.0, 0.5))
    draw_panel(draw, (130, 690, 820, 1050), rows, "plume_std_y_um", "mobile plume width", "std(y) (um)")
    draw_panel(draw, (1040, 690, 1700, 1050), rows, "median_time_s", "median mobile time", "time (s)")

    lx, ly = 430, 1135
    for i, case in enumerate(CASES):
        yy = ly + 28 * i
        draw.line((lx, yy, lx + 48, yy), fill=case.color, width=5)
        draw.text((lx + 62, yy), f"{case.label} direct", fill=INK, font=font(15), anchor="lm")
        dashed_line(draw, [(lx + 240, yy), (lx + 288, yy)], case.color, width=3)
        draw.text((lx + 302, yy), f"{case.label} Markov", fill=INK, font=font(15), anchor="lm")
    max_surv = max(float(row["max_abs_mobile_fraction_error"]) for row in comparisons)
    draw.text((1180, 1180), f"max survival discrepancy over {N_CELLS} cells: {max_surv:.3f}", fill=MUTED, font=font(20), anchor="mm")
    img.save(path)


def write_report(
    path: Path,
    rows: list[dict[str, float | int | str]],
    comparisons: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Direct multi-cell validation",
        "",
        f"Direct repeated-cell tracking and trajectory-library Markov stitching were compared for {N_PARTICLES:,} particles over {N_CELLS} cells at 4 m/day. The Markov stitch uses {N_BINS} transverse inlet bins.",
        "",
        "| case | max mobile error | max attached error | max censored error | plume RMSE (um) | time RMSE (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| "
            f"{row['case']} | "
            f"{float(row['max_abs_mobile_fraction_error']):.4f} | "
            f"{float(row['max_abs_attached_fraction_error']):.4f} | "
            f"{float(row['max_abs_censored_fraction_error']):.4f} | "
            f"{float(row['plume_std_rmse_um']):.3f} | "
            f"{float(row['median_time_rmse_s']):.3f} |"
        )
    lines.extend(["", "| case | method | cell | mobile | attached | censored | plume std (um) | median time (s) |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        if int(row["cell"]) not in {0, 5, 10, 25}:
            continue
        lines.append(
            "| "
            f"{row['case']} | "
            f"{row['method']} | "
            f"{int(row['cell'])} | "
            f"{float(row['mobile_fraction']):.4f} | "
            f"{float(row['attached_fraction']):.4f} | "
            f"{float(row['censored_fraction']):.4f} | "
            f"{float(row['plume_std_y_um']):.2f} | "
            f"{float(row['median_time_s']):.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    comparisons = []
    for case in CASES:
        direct = [row for row in rows if row["case"] == case.key and row["method"] == "direct"]
        markov = [row for row in rows if row["case"] == case.key and row["method"] == "markov"]
        direct_by_cell = {int(row["cell"]): row for row in direct}
        markov_by_cell = {int(row["cell"]): row for row in markov}
        cells = sorted(set(direct_by_cell) & set(markov_by_cell))
        def series(metric: str, method_rows: dict[int, dict[str, float | int | str]]) -> np.ndarray:
            return np.array([float(method_rows[cell][metric]) for cell in cells], dtype=float)
        d_mobile = series("mobile_fraction", direct_by_cell)
        m_mobile = series("mobile_fraction", markov_by_cell)
        d_att = series("attached_fraction", direct_by_cell)
        m_att = series("attached_fraction", markov_by_cell)
        d_cen = series("censored_fraction", direct_by_cell)
        m_cen = series("censored_fraction", markov_by_cell)
        d_width = series("plume_std_y_um", direct_by_cell)
        m_width = series("plume_std_y_um", markov_by_cell)
        d_time = series("median_time_s", direct_by_cell)
        m_time = series("median_time_s", markov_by_cell)
        comparisons.append(
            {
                "case": case.key,
                "max_abs_mobile_fraction_error": float(np.nanmax(np.abs(d_mobile - m_mobile))),
                "max_abs_attached_fraction_error": float(np.nanmax(np.abs(d_att - m_att))),
                "max_abs_censored_fraction_error": float(np.nanmax(np.abs(d_cen - m_cen))),
                "plume_std_rmse_um": float(np.sqrt(np.nanmean((d_width - m_width) ** 2))),
                "median_time_rmse_s": float(np.sqrt(np.nanmean((d_time - m_time) ** 2))),
            }
        )
    return comparisons


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams()
    base_flow = load_flow(PHYSICAL_OUT / "flow_N192.npz", params)
    rng = np.random.default_rng(50001)
    rows: list[dict[str, float | int | str]] = []
    for case in CASES:
        print(f"Validation case: {case.key} direct")
        flow = rescale_flow(
            base_flow,
            VELOCITY_M_PER_DAY / 86400.0,
            max_time=case.direct_max_time_s,
        )
        y0 = initial_positions(params, rng)
        rows.extend(run_direct(case, flow, y0))
        print(f"Validation case: {case.key} Markov")
        rows.extend(run_markov(case, y0))
    comparisons = compare_rows(rows)
    write_dicts(OUT / "multicell_validation_summary.csv", rows)
    write_dicts(OUT / "multicell_validation_comparison.csv", comparisons)
    write_report(OUT / "multicell_validation_report.md", rows, comparisons)
    write_figure(OUT / "multicell_validation.png", rows, comparisons)
    write_figure(FIGURES / "fig10_multicell_validation.png", rows, comparisons)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
