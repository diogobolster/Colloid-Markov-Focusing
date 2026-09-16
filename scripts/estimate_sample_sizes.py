#!/usr/bin/env python3
"""Estimate particle counts needed for transition and focusing statistics."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import load_library


OUT = ROOT / "outputs" / "sample_size"
FOCUSING_OUT = ROOT / "outputs" / "focusing"
PHYSICAL_OUT = ROOT / "outputs" / "physical"
VELOCITIES = (2, 4, 8)
FOCUSING_CONDITIONS = (
    "neutral",
    "favorable_reflecting",
    "favorable_sliding",
    "favorable_lubricated",
    "unfavorable",
)
PHYSICAL_CONDITIONS = ("favorable", "unfavorable")
N_BINS = 32
CORE_FOCUSING_BINS = np.array([15, 16])


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ceil_or_nan(value: float) -> int | str:
    if not np.isfinite(value) or value <= 0:
        return "nan"
    return int(math.ceil(value))


def required_particles(target_events: float, event_probability: float) -> int | str:
    if not np.isfinite(event_probability) or event_probability <= 0:
        return "nan"
    return ceil_or_nan(target_events / event_probability)


def focusing_rows() -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    summary_rows: list[dict[str, float | int | str]] = []
    estimate_rows: list[dict[str, float | int | str]] = []
    for velocity in VELOCITIES:
        for condition in FOCUSING_CONDITIONS:
            lib = load_library(FOCUSING_OUT / f"trajectory_library_{condition}_{velocity}m_per_day.npz")
            n = lib.y_in.size
            bins = np.linspace(0.0, lib.params.cell_length, N_BINS + 1)
            inlet_bin = np.clip(np.digitize(lib.y_in, bins) - 1, 0, N_BINS - 1)
            intercepted = lib.interceptions > 0
            mobile_intercepted = intercepted & lib.exited & (~lib.attached)
            active_bins = np.flatnonzero(np.bincount(inlet_bin[intercepted], minlength=N_BINS) > 0)
            int_counts = np.bincount(inlet_bin[intercepted], minlength=N_BINS)[active_bins]
            mob_counts = np.bincount(inlet_bin[mobile_intercepted], minlength=N_BINS)[active_bins]
            total_counts = np.bincount(inlet_bin, minlength=N_BINS)[active_bins]
            core_bins = CORE_FOCUSING_BINS
            core_int_counts = np.bincount(inlet_bin[intercepted], minlength=N_BINS)[core_bins]
            core_mob_counts = np.bincount(inlet_bin[mobile_intercepted], minlength=N_BINS)[core_bins]
            core_total_counts = np.bincount(inlet_bin, minlength=N_BINS)[core_bins]
            p_intercept = float(np.mean(intercepted))
            p_mobile_intercept = float(np.mean(mobile_intercepted))
            per_active_intercept_rates = int_counts / np.maximum(total_counts, 1)
            per_active_mobile_rates = mob_counts / np.maximum(total_counts, 1)
            per_core_intercept_rates = core_int_counts / np.maximum(core_total_counts, 1)
            per_core_mobile_rates = core_mob_counts / np.maximum(core_total_counts, 1)
            min_core_intercept_rate = float(np.min(per_core_intercept_rates))
            min_core_mobile_rate = float(np.min(per_core_mobile_rates))

            summary_rows.append(
                {
                    "velocity_m_per_day": velocity,
                    "condition": condition,
                    "pilot_particles": n,
                    "active_inlet_bins": " ".join(str(int(v)) for v in active_bins),
                    "active_bin_count": int(active_bins.size),
                    "core_inlet_bins": " ".join(str(int(v)) for v in core_bins),
                    "core_bin_count": int(core_bins.size),
                    "intercepted_total": int(np.sum(intercepted)),
                    "intercepted_mobile": int(np.sum(mobile_intercepted)),
                    "p_intercept_uniform": p_intercept,
                    "p_intercept_mobile_uniform": p_mobile_intercept,
                    "min_p_intercept_per_core_bin": min_core_intercept_rate,
                    "min_p_mobile_per_core_bin": min_core_mobile_rate,
                }
            )

            for target in (1000, 5000):
                estimate_rows.append(
                    {
                        "velocity_m_per_day": velocity,
                        "condition": condition,
                        "target": f"{target} intercepted events total",
                        "uniform_injection_particles": required_particles(target, p_intercept),
                        "central_band_particles": required_particles(target, np.mean(per_core_intercept_rates)),
                        "basis": "all intercepted particles; assumes central-band injection is split over central inlet bins 15 and 16",
                    }
                )
                estimate_rows.append(
                    {
                        "velocity_m_per_day": velocity,
                        "condition": condition,
                        "target": f"{target} mobile intercepted exits total",
                        "uniform_injection_particles": required_particles(target, p_mobile_intercept),
                        "central_band_particles": required_particles(target, np.mean(per_core_mobile_rates)),
                        "basis": "current closure; mobile intercepted exits only",
                    }
                )

            for target_per_bin in (500, 1000, 5000):
                estimate_rows.append(
                    {
                        "velocity_m_per_day": velocity,
                        "condition": condition,
                        "target": f"{target_per_bin} intercepted events per core inlet bin",
                        "uniform_injection_particles": required_particles(target_per_bin * int(core_bins.size), p_intercept),
                        "central_band_particles": required_particles(target_per_bin * int(core_bins.size), min_core_intercept_rate),
                        "basis": "post-refinement release-capable estimate; counts all intercepted particles in central bins 15 and 16",
                    }
                )
                estimate_rows.append(
                    {
                        "velocity_m_per_day": velocity,
                        "condition": condition,
                        "target": f"{target_per_bin} mobile exits per core inlet bin",
                        "uniform_injection_particles": required_particles(target_per_bin * int(core_bins.size), p_mobile_intercept),
                        "central_band_particles": required_particles(target_per_bin * int(core_bins.size), min_core_mobile_rate),
                        "basis": "current closure; mobile exits after interception",
                    }
                )

    return summary_rows, estimate_rows


def transition_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for velocity in VELOCITIES:
        for condition in PHYSICAL_CONDITIONS:
            lib = load_library(PHYSICAL_OUT / f"trajectory_library_{condition}_{velocity}m_per_day.npz")
            mobile = lib.exited & (~lib.attached)
            p_mobile = float(np.mean(mobile))
            bins = np.linspace(0.0, lib.params.cell_length, N_BINS + 1)
            inlet_bin = np.clip(np.digitize(lib.y_in, bins) - 1, 0, N_BINS - 1)
            active_inlet_bins = int(np.count_nonzero(np.bincount(inlet_bin, minlength=N_BINS)))
            for target_per_row in (500, 1000, 2000):
                rows.append(
                    {
                        "velocity_m_per_day": velocity,
                        "condition": condition,
                        "target_mobile_exits_per_inlet_row": target_per_row,
                        "active_inlet_bins": active_inlet_bins,
                        "uniform_injection_particles": required_particles(target_per_row * active_inlet_bins, p_mobile),
                        "pilot_mobile_fraction": p_mobile,
                    }
                )
    return rows


def write_report(
    path: Path,
    focusing_summary: list[dict[str, float | int | str]],
    focusing_estimates: list[dict[str, float | int | str]],
    transition_estimates: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Particle-count estimates",
        "",
        "The main conclusion is that ordinary full-cell transition matrices and focusing-conditioned transition matrices have very different sample-size requirements.",
        "",
        "## Full-cell transition matrices",
        "",
        "The corrected center/corner periodic cell has 32 outlet bins but only 16 open inlet bins because the corner collectors block the upper and lower inlet segments. With 10,000 particles, the pilot runs already give roughly 580--630 mobile exits per open inlet row. A cleaner production target is:",
        "",
        "- 10,000 particles per velocity/condition for about 500 mobile exits per open inlet row.",
        "- 20,000 particles per velocity/condition for about 1,000 mobile exits per open inlet row.",
        "- 40,000 particles per velocity/condition for about 2,000 mobile exits per open inlet row.",
        "",
        "These estimates use the observed mobile exit fractions in the physical runs.",
        "",
        "## Focusing-conditioned statistics",
        "",
        "Focusing is harder than the full-cell transition matrix because only particles injected near a collector enter the near-surface zone. In the corrected cell, active interception bins include both the central collector and the corner collectors. The core bins for the central downstream-stagnation focusing test remain bins 15 and 16; these central-core bins should be kept separate from corner-collector events when building the publication focusing kernel.",
        "",
        "| velocity | condition | active bins | core bins | p(intercept), uniform | p(mobile intercept), uniform | min p(intercept) in core bin | min p(mobile intercept) in core bin |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in focusing_summary:
        lines.append(
            "| "
            f"{row['velocity_m_per_day']} | "
            f"{row['condition']} | "
            f"{row['active_inlet_bins']} | "
            f"{row['core_inlet_bins']} | "
            f"{float(row['p_intercept_uniform']):.4g} | "
            f"{float(row['p_intercept_mobile_uniform']):.4g} | "
            f"{float(row['min_p_intercept_per_core_bin']):.4g} | "
            f"{float(row['min_p_mobile_per_core_bin']):.4g} |"
        )

    lines.extend(
        [
            "",
            "The attractive-reflecting projection closure remains a warning case: it produces many interceptions but relatively few mobile releases, and at least one central core bin can have zero mobile releases. The attractive-sliding diagnostic is the release-capable numerical limit. The attractive-lubricated case is the more physical nonattaching limit and should be interpreted together with the parameter sweep.",
            "",
            "## Recommended runs",
            "",
            "- Use 20,000 uniform particles per velocity/condition for the ordinary mobile transition matrix in the corrected center/corner cell.",
            "- For focusing, add a stratified central-core ensemble over inlet bins 15 and 16. Use at least 50,000 particles per velocity/condition in this central band after the near-surface mobility/reflection model is fixed.",
            "- For publication-grade focusing kernels, use 100,000 to 350,000 central-band particles per velocity/condition. That should give thousands of intercepted central-grain events if the refined model releases intercepted particles rather than pinning them.",
            "- Do not use the attractive-reflecting projection closure for final focusing statistics. Use the lubrication-aware closure and the parameter sweep to choose regimes where intercepted particles release rather than pin.",
            "",
            "## Selected numerical estimates",
            "",
            "| velocity | condition | target | uniform particles | central-band particles | basis |",
            "|---:|---|---|---:|---:|---|",
        ]
    )

    interesting = []
    for row in focusing_estimates:
        target = str(row["target"])
        condition = str(row["condition"])
        if (
            target in {
                "1000 intercepted events per core inlet bin",
                "5000 intercepted events per core inlet bin",
                "1000 mobile exits per core inlet bin",
            }
            and condition in {"neutral", "favorable_reflecting", "favorable_sliding", "favorable_lubricated", "unfavorable"}
        ):
            interesting.append(row)

    for row in interesting:
        lines.append(
            "| "
            f"{row['velocity_m_per_day']} | "
            f"{row['condition']} | "
            f"{row['target']} | "
            f"{row['uniform_injection_particles']} | "
            f"{row['central_band_particles']} | "
            f"{row['basis']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    focusing_summary, focusing_estimates = focusing_rows()
    transition_estimates = transition_rows()
    write_dicts(OUT / "focusing_pilot_rates.csv", focusing_summary)
    write_dicts(OUT / "focusing_sample_size_estimates.csv", focusing_estimates)
    write_dicts(OUT / "transition_matrix_sample_size_estimates.csv", transition_estimates)
    write_report(OUT / "sample_size_report.md", focusing_summary, focusing_estimates, transition_estimates)
    print("Wrote sample-size estimates to", OUT)


if __name__ == "__main__":
    main()
