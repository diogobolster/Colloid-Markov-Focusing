#!/usr/bin/env python3
"""Build long-horizon nonattaching one-cell libraries for production upscaling."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import (
    PhysicalParams,
    load_flow,
    load_library,
    rescale_flow,
    save_library,
    simulate_cell_transitions,
)


PHYSICAL_OUT = ROOT / "outputs" / "physical"

N_PARTICLES = 30_000
N_BINS = 96
VELOCITY_CASES_M_PER_DAY = (2.0, 4.0, 8.0)
MAX_TIME_BY_VELOCITY_S = {
    2.0: 300.0,
    4.0: 180.0,
    8.0: 180.0,
}


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    condition: str
    surface_mode: str
    allow_attachment: bool
    seed_offset: int


CASES = (
    Case(
        key="neutral",
        label="no DLVO",
        condition="neutral",
        surface_mode="lubrication",
        allow_attachment=False,
        seed_offset=11,
    ),
    Case(
        key="unfavorable",
        label="unfavorable reflect",
        condition="unfavorable",
        surface_mode="project",
        allow_attachment=False,
        seed_offset=23,
    ),
)


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_support(y_in: np.ndarray, params: PhysicalParams) -> tuple[int, float, int]:
    bins = np.linspace(0.0, params.cell_length, N_BINS + 1)
    ybin = np.clip(np.digitize(y_in, bins) - 1, 0, N_BINS - 1)
    counts = np.bincount(ybin, minlength=N_BINS)
    nonzero = counts[counts > 0]
    return int(np.min(nonzero)), float(np.median(nonzero)), int(np.sum(counts == 0))


def max_time_for_velocity(velocity_m_per_day: float) -> float:
    return MAX_TIME_BY_VELOCITY_S[velocity_m_per_day]


def library_path(case: Case, velocity_m_per_day: float) -> Path:
    max_time_s = max_time_for_velocity(velocity_m_per_day)
    return PHYSICAL_OUT / f"trajectory_library_{case.key}_{velocity_m_per_day:.0f}m_per_day_{max_time_s:.0f}s.npz"


def summarize(case: Case, velocity_m_per_day: float, max_time_s: float, lib, output_path: Path) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    intercepted = lib.interceptions > 0
    h_finite = lib.h_min[np.isfinite(lib.h_min)]
    min_support, median_support, zero_rows = row_support(lib.y_in, lib.params)
    return {
        "case": case.key,
        "label": case.label,
        "condition": case.condition,
        "surface_mode": case.surface_mode,
        "velocity_m_per_day": velocity_m_per_day,
        "max_time_s": max_time_s,
        "particles": int(lib.y_in.size),
        "exited_fraction": float(np.mean(lib.exited)),
        "attached_fraction": float(np.mean(lib.attached)),
        "censored_fraction": float(np.mean(lib.censored)),
        "intercepted_fraction": float(np.mean(intercepted)),
        "mean_interceptions": float(np.mean(lib.interceptions)),
        "mean_near_time_s": float(np.mean(lib.near_time)),
        "mobile_median_time_s": float(np.nanmedian(lib.travel_time[mobile])) if np.any(mobile) else float("nan"),
        "mobile_q95_time_s": float(np.nanquantile(lib.travel_time[mobile], 0.95)) if np.any(mobile) else float("nan"),
        "median_h_min_m": float(np.median(h_finite)) if h_finite.size else float("nan"),
        "min_96bin_row_support": min_support,
        "median_96bin_row_support": median_support,
        "zero_96bin_rows": zero_rows,
        "output_path": str(output_path.relative_to(ROOT)),
    }


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        "# Long-horizon nonattaching trajectory libraries",
        "",
        (
            f"These libraries use the resolved N=192 flow at {', '.join(f'{v:g}' for v in VELOCITY_CASES_M_PER_DAY)} m/day, "
            f"{N_PARTICLES:,} particles per chemistry/velocity case, and velocity-specific one-cell tracking horizons "
            f"({', '.join(f'{v:g} m/day: {MAX_TIME_BY_VELOCITY_S[v]:g} s' for v in VELOCITY_CASES_M_PER_DAY)}). "
            "They are intended for production nonattaching no-DLVO and homogeneous unfavorable cases."
        ),
        "",
        "| case | velocity (m/day) | max time (s) | particles | exited | censored | intercepted | median mobile time (s) | q95 mobile time (s) | min 96-bin support | median 96-bin support | output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['label']} | "
            f"{float(row['velocity_m_per_day']):.0f} | "
            f"{float(row['max_time_s']):.0f} | "
            f"{int(row['particles'])} | "
            f"{float(row['exited_fraction']):.4f} | "
            f"{float(row['censored_fraction']):.4f} | "
            f"{float(row['intercepted_fraction']):.4f} | "
            f"{float(row['mobile_median_time_s']):.2f} | "
            f"{float(row['mobile_q95_time_s']):.2f} | "
            f"{int(row['min_96bin_row_support'])} | "
            f"{float(row['median_96bin_row_support']):.1f} | "
            f"`{row['output_path']}` |"
        )
    lines.extend(
        [
            "",
            "A nonzero censored fraction here would still be a numerical horizon flag, not attachment. With the present velocity-specific horizons, the production nonattaching libraries are expected to complete all particles.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PHYSICAL_OUT.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams()
    base_flow = load_flow(PHYSICAL_OUT / "flow_N192.npz", params)
    rows: list[dict[str, float | int | str]] = []
    for velocity_index, velocity_m_per_day in enumerate(VELOCITY_CASES_M_PER_DAY):
        max_time_s = max_time_for_velocity(velocity_m_per_day)
        flow = rescale_flow(base_flow, velocity_m_per_day / 86400.0, max_time=max_time_s)
        for case in CASES:
            output_path = library_path(case, velocity_m_per_day)
            if output_path.exists():
                print(f"Loading existing {output_path.relative_to(ROOT)}", flush=True)
                lib = load_library(output_path)
            else:
                print(
                    f"Tracking {case.label}: velocity={velocity_m_per_day:.0f} m/day, "
                    f"{N_PARTICLES:,} particles, max_time={max_time_s:g} s, surface={case.surface_mode}",
                    flush=True,
                )
                lib = simulate_cell_transitions(
                    flow,
                    case.condition,
                    n_particles=N_PARTICLES,
                    seed=72000 + 100 * velocity_index + case.seed_offset,
                    allow_attachment=case.allow_attachment,
                    surface_mode=case.surface_mode,
                )
                save_library(output_path, lib)
            row = summarize(case, velocity_m_per_day, max_time_s, lib, output_path)
            rows.append(row)
            print(
                "  exited",
                f"{row['exited_fraction']:.4f}",
                "censored",
                f"{row['censored_fraction']:.4f}",
                "min 96-bin support",
                row["min_96bin_row_support"],
                flush=True,
            )
    write_dicts(PHYSICAL_OUT / "long_horizon_library_summary.csv", rows)
    write_report(PHYSICAL_OUT / "long_horizon_library_report.md", rows)
    print("Done. Outputs written to", PHYSICAL_OUT, flush=True)


if __name__ == "__main__":
    main()
