#!/usr/bin/env python3
"""Compare baseline tracking with a 100x Brownian diffusivity sensitivity case."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, replace
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
    save_library,
    simulate_cell_transitions,
    transition_matrices,
)


OUT = ROOT / "outputs" / "diffusion_sensitivity"
FIGURES = ROOT / "outputs" / "figures"
FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"

MEAN_VELOCITY_M_PER_DAY = 4.0
N_ONE_CELL = 12_000
N_MARKOV = 50_000
N_CELLS = 120
N_BINS = 96
MAX_TIME_S = 180.0
DIFFUSIVITY_MULTIPLIERS = (1.0, 100.0)
CONDITIONS = ("neutral", "favorable", "unfavorable")

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"

CONDITION_COLORS = {"neutral": BLUE, "favorable": RED, "unfavorable": GREEN}
MULTIPLIER_COLORS = {1.0: PURPLE, 100.0: ORANGE}
CONDITION_LABELS = {"neutral": "no DLVO", "favorable": "favorable", "unfavorable": "unfavorable"}


@dataclass(frozen=True)
class SensitivityCase:
    diffusivity_multiplier: float
    condition: str
    params: PhysicalParams
    flow: FlowField


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


def summarize_library(multiplier: float, condition: str, lib: TrajectoryLibrary) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    intercepted = lib.interceptions > 0
    y_out_mobile = lib.y_out[mobile]
    travel_mobile = lib.travel_time[mobile]
    return {
        "diffusivity_multiplier": multiplier,
        "condition": condition,
        "diffusivity_m2_s": lib.params.diffusivity,
        "peclet": lib.params.particle_peclet,
        "particles": int(lib.y_in.size),
        "exited_fraction": float(np.mean(lib.exited)),
        "attached_fraction": float(np.mean(lib.attached)),
        "censored_fraction": float(np.mean(lib.censored)),
        "intercepted_fraction": float(np.mean(intercepted)),
        "attached_given_intercepted": float(np.mean(lib.attached[intercepted])) if np.any(intercepted) else float("nan"),
        "mean_interceptions": float(np.mean(lib.interceptions)),
        "mean_interceptions_mobile": float(np.mean(lib.interceptions[mobile])) if np.any(mobile) else float("nan"),
        "mean_near_time_s": float(np.mean(lib.near_time)),
        "median_travel_time_mobile_s": float(np.nanmedian(travel_mobile)) if np.any(mobile) else float("nan"),
        "q95_travel_time_mobile_s": float(np.nanquantile(travel_mobile, 0.95)) if np.any(mobile) else float("nan"),
        "mobile_yout_std_um": float(np.nanstd(y_out_mobile) * 1.0e6) if np.any(mobile) else float("nan"),
        "mobile_yout_centered_std_um": float(np.nanstd(y_out_mobile - 0.5 * lib.params.cell_length) * 1.0e6)
        if np.any(mobile)
        else float("nan"),
    }


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


def initial_positions(library: TrajectoryLibrary, rng: np.random.Generator) -> np.ndarray:
    bins, by_bin, _, _ = bin_records(library)
    open_bins = np.array([i for i, ids in enumerate(by_bin) if ids.size > 0], dtype=int)
    counts = np.array([by_bin[i].size for i in open_bins], dtype=float)
    probabilities = counts / np.sum(counts)
    chosen_bins = rng.choice(open_bins, size=N_MARKOV, p=probabilities)
    y = np.empty(N_MARKOV, dtype=float)
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
        sampled[mask] = rng.choice(by_bin[int(source_bin)], size=int(np.sum(mask)), replace=True)
    return sampled


def run_markov(multiplier: float, condition: str, library: TrajectoryLibrary) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(88000 + int(multiplier * 10) + sum(ord(ch) for ch in condition))
    bins, by_bin, nearest, min_support = bin_records(library)
    y = initial_positions(library, rng)
    time = np.zeros(N_MARKOV, dtype=float)
    cumulative_interceptions = np.zeros(N_MARKOV, dtype=int)
    alive = np.ones(N_MARKOV, dtype=bool)
    attached = np.zeros(N_MARKOV, dtype=bool)
    censored = np.zeros(N_MARKOV, dtype=bool)
    rows: list[dict[str, float | int | str]] = []
    mobile_times = library.travel_time[library.exited & (~library.attached)]
    fallback_time = float(np.nanmedian(mobile_times[np.isfinite(mobile_times)]))

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
                    selected = picks[picked_attached]
                    attached[ids] = True
                    alive[ids] = False
                    cumulative_interceptions[ids] += library.interceptions[selected]

                if np.any(picked_censored):
                    ids = active_ids[picked_censored]
                    censored[ids] = True
                    alive[ids] = False

                if np.any(picked_exited):
                    ids = active_ids[picked_exited]
                    selected = picks[picked_exited]
                    y[ids] = library.y_out[selected]
                    dt = library.travel_time[selected]
                    time[ids] += np.where(np.isfinite(dt), dt, fallback_time)
                    cumulative_interceptions[ids] += library.interceptions[selected]

        mobile_ids = np.flatnonzero(alive)
        rows.append(
            {
                "diffusivity_multiplier": multiplier,
                "condition": condition,
                "cell": cell,
                "mobile_fraction": float(np.mean(alive)),
                "attached_fraction": float(np.mean(attached)),
                "censored_fraction": float(np.mean(censored)),
                "plume_std_y_um": float(np.std(y[mobile_ids]) * 1.0e6) if mobile_ids.size else float("nan"),
                "median_time_s": float(np.nanmedian(time[mobile_ids])) if mobile_ids.size else float("nan"),
                "mean_cumulative_interceptions_mobile": float(np.mean(cumulative_interceptions[mobile_ids]))
                if mobile_ids.size
                else float("nan"),
                "min_96_bin_support": min_support,
            }
        )
    return rows


def panel_axes(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str, xlabel: str, ylabel: str) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline=INK, width=2)
    draw.text(((x0 + x1) / 2, y0 - 28), title, fill=INK, font=font(22, True), anchor="mm")
    draw.text(((x0 + x1) / 2, y1 + 42), xlabel, fill=INK, font=font(18), anchor="mm")
    draw.text((x0 + 13, y0 + 15), ylabel, fill=MUTED, font=font(13), anchor="la")


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

    for tick in np.linspace(xmin, xmax, 5):
        px = sx(float(tick))
        draw.line((px, y1, px, y1 + 7), fill=INK, width=2)
        draw.text((px, y1 + 24), f"{tick:.0f}", fill=INK, font=font(13), anchor="mm")
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
    lx, ly = x1 - 255, y0 + 25
    for i, (label, color, _, _) in enumerate(series):
        yy = ly + 28 * i
        draw.line((lx, yy, lx + 40, yy), fill=color, width=5)
        draw.text((lx + 52, yy), label, fill=INK, font=font(14), anchor="lm")


def draw_grouped_bar_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    rows: list[dict[str, float | int | str]],
    metric: str,
    title: str,
    ylabel: str,
    cell_filter: int | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    panel_axes(draw, rect, title, "condition", ylabel)
    data_rows = rows if cell_filter is None else [r for r in rows if int(r["cell"]) == cell_filter]
    values = {
        (str(r["condition"]), float(r["diffusivity_multiplier"])): float(r[metric])
        for r in data_rows
    }
    finite_values = [v for v in values.values() if np.isfinite(v)]
    ymax = max(max(finite_values) * 1.18, 1.0e-9) if finite_values else 1.0

    def sx_group(i: int) -> float:
        return x0 + (i + 0.5) / len(CONDITIONS) * (x1 - x0)

    def sy(v: float) -> float:
        return y1 - v / ymax * (y1 - y0)

    for tick in np.linspace(0.0, ymax, 5):
        py = sy(float(tick))
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((x0 - 11, py), f"{tick:.2g}", fill=INK, font=font(13), anchor="rm")
    group_width = (x1 - x0) / len(CONDITIONS)
    bar_width = min(64.0, group_width / 4.5)
    offsets = (-bar_width * 0.65, bar_width * 0.65)
    for i, condition in enumerate(CONDITIONS):
        cx = sx_group(i)
        draw.text((cx, y1 + 24), CONDITION_LABELS[condition], fill=INK, font=font(13), anchor="mm")
        for j, multiplier in enumerate(DIFFUSIVITY_MULTIPLIERS):
            val = values[(condition, multiplier)]
            bx = cx + offsets[j]
            if not np.isfinite(val):
                draw.text((bx, y1 - 12), "n/a", fill=MUTED, font=font(13), anchor="mm")
                continue
            draw.rectangle((bx - bar_width / 2, sy(val), bx + bar_width / 2, y1), fill=MULTIPLIER_COLORS[multiplier])
    lx, ly = x1 - 235, y0 + 24
    for j, multiplier in enumerate(DIFFUSIVITY_MULTIPLIERS):
        yy = ly + 28 * j
        draw.rectangle((lx, yy - 8, lx + 30, yy + 8), fill=MULTIPLIER_COLORS[multiplier])
        label = "baseline D" if multiplier == 1.0 else "100x D"
        draw.text((lx + 42, yy), label, fill=INK, font=font(14), anchor="lm")


def write_figure(
    path: Path,
    one_cell_rows: list[dict[str, float | int | str]],
    markov_rows: list[dict[str, float | int | str]],
) -> None:
    img = Image.new("RGB", (1800, 1380), "white")
    draw = ImageDraw.Draw(img)
    draw.text((900, 48), "Sensitivity to 100x Brownian diffusivity at 4 m/day", fill=INK, font=font(36, True), anchor="mm")

    draw_grouped_bar_panel(
        draw,
        (130, 145, 820, 505),
        one_cell_rows,
        "intercepted_fraction",
        "one-cell interception",
        "intercepted fraction",
    )

    survival_series = []
    for multiplier in DIFFUSIVITY_MULTIPLIERS:
        rows = [
            r
            for r in markov_rows
            if str(r["condition"]) == "favorable" and float(r["diffusivity_multiplier"]) == multiplier
        ]
        cells = np.array([int(r["cell"]) for r in rows])
        survival = np.array([float(r["mobile_fraction"]) for r in rows])
        label = "baseline D" if multiplier == 1.0 else "100x D"
        survival_series.append((label, MULTIPLIER_COLORS[multiplier], cells, survival))
    draw_line_panel(
        draw,
        (1040, 145, 1700, 505),
        survival_series,
        "favorable survival through 120 cells",
        "cell number",
        "mobile fraction",
        y_limits=(0.0, 1.0),
    )

    draw_grouped_bar_panel(
        draw,
        (130, 710, 820, 1070),
        markov_rows,
        "plume_std_y_um",
        "cell-120 mobile plume width",
        "std(y) (um)",
        cell_filter=N_CELLS,
    )
    draw_grouped_bar_panel(
        draw,
        (1040, 710, 1700, 1070),
        markov_rows,
        "mean_cumulative_interceptions_mobile",
        "cell-120 mobile interception history",
        "mean interceptions/mobile",
        cell_filter=N_CELLS,
    )

    terminal = {
        (float(r["diffusivity_multiplier"]), str(r["condition"])): r
        for r in markov_rows
        if int(r["cell"]) == N_CELLS
    }
    baseline_fav = terminal[(1.0, "favorable")]
    high_fav = terminal[(100.0, "favorable")]
    draw.text(
        (900, 1260),
        "Favorable cell-120 attachment: "
        f"baseline {float(baseline_fav['attached_fraction']):.3f}; "
        f"100x D {float(high_fav['attached_fraction']):.3f}",
        fill=INK,
        font=font(22),
        anchor="mm",
    )
    draw.text(
        (900, 1302),
        "The 100x-D case has Pe near 117 instead of 11660, increasing cross-stream mixing and near-surface encounters.",
        fill=MUTED,
        font=font(19),
        anchor="mm",
    )
    img.save(path)


def write_report(
    path: Path,
    one_cell_rows: list[dict[str, float | int | str]],
    markov_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Diffusion sensitivity: 100x Brownian diffusivity",
        "",
        f"All runs use the resolved N=192 center/corner flow at {MEAN_VELOCITY_M_PER_DAY:.0f} m/day.",
        f"Each one-cell library uses {N_ONE_CELL:,} particles and a {MAX_TIME_S:.0f} s horizon. Each Markov stitch uses {N_MARKOV:,} particles, {N_CELLS} cells, and {N_BINS} transverse bins.",
        "",
        "## Parameter change",
        "",
        "| multiplier | D (m^2/s) | Pe |",
        "|---:|---:|---:|",
    ]
    seen: set[float] = set()
    for row in one_cell_rows:
        multiplier = float(row["diffusivity_multiplier"])
        if multiplier in seen:
            continue
        seen.add(multiplier)
        lines.append(f"| {multiplier:.0f} | {float(row['diffusivity_m2_s']):.4e} | {float(row['peclet']):.1f} |")

    lines.extend(
        [
            "",
            "## One-cell libraries",
            "",
            "| multiplier | condition | exited | attached | censored | intercepted | attach/intercept | median time (s) | y-out std (um) | mean interceptions |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for multiplier in DIFFUSIVITY_MULTIPLIERS:
        for condition in CONDITIONS:
            row = next(
                r
                for r in one_cell_rows
                if float(r["diffusivity_multiplier"]) == multiplier and str(r["condition"]) == condition
            )
            lines.append(
                "| "
                f"{multiplier:.0f} | {condition} | "
                f"{float(row['exited_fraction']):.4f} | "
                f"{float(row['attached_fraction']):.4f} | "
                f"{float(row['censored_fraction']):.4f} | "
                f"{float(row['intercepted_fraction']):.4f} | "
                f"{float(row['attached_given_intercepted']):.4f} | "
                f"{float(row['median_travel_time_mobile_s']):.2f} | "
                f"{float(row['mobile_yout_std_um']):.2f} | "
                f"{float(row['mean_interceptions']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## 120-cell Markov stitch",
            "",
            "| multiplier | condition | mobile | attached | censored | plume std (um) | median time (h) | mean interceptions/mobile |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for multiplier in DIFFUSIVITY_MULTIPLIERS:
        for condition in CONDITIONS:
            row = next(
                r
                for r in markov_rows
                if float(r["diffusivity_multiplier"]) == multiplier
                and str(r["condition"]) == condition
                and int(r["cell"]) == N_CELLS
            )
            lines.append(
                "| "
                f"{multiplier:.0f} | {condition} | "
                f"{float(row['mobile_fraction']):.4f} | "
                f"{float(row['attached_fraction']):.4f} | "
                f"{float(row['censored_fraction']):.4f} | "
                f"{float(row['plume_std_y_um']):.2f} | "
                f"{float(row['median_time_s']) / 3600.0:.3f} | "
                f"{float(row['mean_cumulative_interceptions_mobile']):.3f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    base_params = replace(
        PhysicalParams(),
        mean_velocity=MEAN_VELOCITY_M_PER_DAY / 86400.0,
        max_time=MAX_TIME_S,
        diffusivity_multiplier=1.0,
    )
    one_cell_rows: list[dict[str, float | int | str]] = []
    markov_rows: list[dict[str, float | int | str]] = []

    for multiplier in DIFFUSIVITY_MULTIPLIERS:
        params = replace(base_params, diffusivity_multiplier=multiplier)
        flow = load_flow(FLOW_PATH, params=params)
        print(
            f"Diffusivity multiplier {multiplier:.0f}: "
            f"D={params.diffusivity:.4e} m^2/s, Pe={params.particle_peclet:.1f}"
        )
        for condition in CONDITIONS:
            seed = 19000 + int(multiplier * 17) + sum(ord(ch) for ch in condition)
            print(f"  Tracking {condition} with {N_ONE_CELL:,} particles")
            lib = simulate_cell_transitions(
                flow,
                condition,
                n_particles=N_ONE_CELL,
                seed=seed,
                allow_attachment=(condition == "favorable"),
            )
            save_library(OUT / f"trajectory_library_{condition}_{int(multiplier)}xD.npz", lib)
            matrices = transition_matrices(lib, n_bins=32)
            np.savetxt(OUT / f"transition_probabilities_{condition}_{int(multiplier)}xD.csv", matrices["probabilities"], delimiter=",")
            row = summarize_library(multiplier, condition, lib)
            one_cell_rows.append(row)
            print(
                "    exited",
                f"{float(row['exited_fraction']):.3f}",
                "attached",
                f"{float(row['attached_fraction']):.3f}",
                "intercepted",
                f"{float(row['intercepted_fraction']):.3f}",
            )
            markov_rows.extend(run_markov(multiplier, condition, lib))

    write_dicts(OUT / "one_cell_summary.csv", one_cell_rows)
    write_dicts(OUT / "markov_summary.csv", markov_rows)
    write_report(OUT / "diffusion_sensitivity_report.md", one_cell_rows, markov_rows)
    write_figure(OUT / "diffusion_sensitivity.png", one_cell_rows, markov_rows)
    write_figure(FIGURES / "fig15_diffusion_sensitivity.png", one_cell_rows, markov_rows)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
