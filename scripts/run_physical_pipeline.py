#!/usr/bin/env python3
"""Run the physical-unit flow/tracking/transition-matrix pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import (
    PhysicalParams,
    flow_quality_metrics,
    rescale_flow,
    save_flow,
    save_library,
    simulate_cell_transitions,
    solve_lbm_flow,
    transition_matrices,
)


OUT = ROOT / "outputs" / "physical"
VELOCITY_CASES_M_PER_DAY = (2.0, 4.0, 8.0)
FLOW_CASES = ((96, 1800), (128, 2400), (160, 3000), (192, 3800))


def write_dicts(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.10g")


def summarize_library(
    name: str,
    velocity_m_per_day: float,
    lib,
    matrices: dict[str, np.ndarray],
) -> dict[str, float | str]:
    mobile = lib.exited & (~lib.attached)
    completed = lib.exited | lib.attached
    intercepted = lib.interceptions > 0
    row_support = matrices["counts"].sum(axis=1)
    nonzero_rows = row_support[row_support > 0]
    h_min_mobile = lib.h_min[np.isfinite(lib.h_min)]
    return {
        "condition": name,
        "velocity_m_per_day": velocity_m_per_day,
        "max_time_s": lib.params.max_time,
        "particles": int(lib.y_in.size),
        "exited_fraction": float(np.mean(lib.exited)),
        "attached_fraction": float(np.mean(lib.attached)),
        "intercepted_fraction": float(np.mean(intercepted)),
        "attached_given_intercepted": float(np.mean(lib.attached[intercepted])) if np.any(intercepted) else float("nan"),
        "censored_fraction": float(np.mean(lib.censored)),
        "completed_fraction": float(np.mean(completed)),
        "mean_travel_time_mobile_s": float(np.nanmean(lib.travel_time[mobile])) if np.any(mobile) else float("nan"),
        "median_travel_time_mobile_s": float(np.nanmedian(lib.travel_time[mobile])) if np.any(mobile) else float("nan"),
        "mean_interceptions": float(np.mean(lib.interceptions)),
        "mean_near_time_s": float(np.mean(lib.near_time)),
        "median_h_min_m": float(np.median(h_min_mobile)) if h_min_mobile.size else float("nan"),
        "p05_h_min_m": float(np.quantile(h_min_mobile, 0.05)) if h_min_mobile.size else float("nan"),
        "min_transition_row_support": int(np.min(nonzero_rows)) if nonzero_rows.size else 0,
        "median_transition_row_support": float(np.median(nonzero_rows)) if nonzero_rows.size else 0.0,
        "zero_transition_rows": int(np.sum(row_support == 0)),
    }


def write_quality_report(
    path: Path,
    params: PhysicalParams,
    flow_rows: list[dict[str, float | str]],
    summaries: list[dict[str, float | str]],
) -> None:
    lines = [
        "# Physical pipeline quality report",
        "",
        "## Parameter anchors",
        "",
        "- Geometry: one full center collector plus four periodic quarter collectors at the corners",
        f"- Cell length: {params.cell_length:.3g} m",
        f"- Collector diameter: {2.0 * params.grain_radius:.3g} m",
        f"- Colloid diameter: {2.0 * params.particle_radius:.3g} m",
        f"- Ionic strength: {params.ionic_strength_molar * 1e3:.3g} mM",
        f"- Hamaker constant: {params.hamaker:.3g} J",
        f"- Particle zeta potential: {params.zeta_particle * 1e3:.3g} mV",
        f"- Favorable collector zeta potential: {params.zeta_collector_favorable * 1e3:.3g} mV",
        f"- Unfavorable collector zeta potential: {params.zeta_collector_unfavorable * 1e3:.3g} mV",
        f"- Debye length: {params.debye_length:.3g} m",
        f"- Diffusivity: {params.diffusivity:.3g} m^2/s",
        "",
        "## Flow iteration",
        "",
        "| resolution | iterations | porosity | max speed (m/s) | rms divergence (1/s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in flow_rows:
        lines.append(
            "| "
            f"{int(float(row['resolution']))} | "
            f"{int(float(row['iterations']))} | "
            f"{float(row['porosity']):.4f} | "
            f"{float(row['max_speed']):.4e} | "
            f"{float(row['rms_divergence']):.4e} |"
        )
    lines.extend(
        [
            "",
            "## Particle tracking and transition support",
            "",
            "| condition | velocity (m/day) | particles | intercepted | exited | attached | attached/intercepted | censored | min row support | median row support | zero rows |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| "
            f"{row['condition']} | "
            f"{float(row['velocity_m_per_day']):.1f} | "
            f"{int(row['particles'])} | "
            f"{float(row['intercepted_fraction']):.3f} | "
            f"{float(row['exited_fraction']):.3f} | "
            f"{float(row['attached_fraction']):.3f} | "
            f"{float(row['attached_given_intercepted']):.3f} | "
            f"{float(row['censored_fraction']):.3f} | "
            f"{int(row['min_transition_row_support'])} | "
            f"{float(row['median_transition_row_support']):.1f} | "
            f"{int(row['zero_transition_rows'])} |"
        )
    lines.extend(
        [
            "",
            "The unfavorable cases are homogeneous repulsive collectors. Attachment is disabled by construction, so any particle reaching an exclusion radius is reflected/projected back to the minimum gap and remains mobile. The 16 zero rows are the inlet bins blocked by the corner collectors.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = PhysicalParams()
    print("Physical parameter anchors:")
    print("  L =", params.cell_length, "m")
    print("  grain diameter =", 2 * params.grain_radius, "m")
    print("  particle diameter =", 2 * params.particle_radius, "m")
    print("  velocity =", params.mean_velocity, "m/s")
    print("  D =", params.diffusivity, "m^2/s")
    print("  Debye length =", params.debye_length, "m")
    print("  Re =", params.reynolds_number)
    print("  Pe_p =", params.particle_peclet)

    flow_rows: list[dict[str, float | str]] = []
    flow = None
    for resolution, iterations in FLOW_CASES:
        print(f"Solving LBM flow: N={resolution}, iterations={iterations}")
        trial = solve_lbm_flow(params, resolution=resolution, iterations=iterations)
        metrics = flow_quality_metrics(trial)
        metrics["case"] = f"N{resolution}"
        flow_rows.append(metrics)
        save_flow(OUT / f"flow_N{resolution}.npz", trial)
        flow = trial
        print(
            "  mean ux",
            f"{metrics['mean_ux']:.4e}",
            "max speed",
            f"{metrics['max_speed']:.4e}",
            "porosity",
            f"{metrics['porosity']:.3f}",
        )
    write_dicts(OUT / "flow_quality.csv", flow_rows)
    assert flow is not None

    particle_count = 10000
    bins = 32
    summaries: list[dict[str, float | str]] = []
    for velocity_index, velocity_m_per_day in enumerate(VELOCITY_CASES_M_PER_DAY):
        mean_velocity = velocity_m_per_day / 86400.0
        advective_time = params.cell_length / mean_velocity
        max_time = max(60.0, 6.0 * advective_time)
        case_flow = rescale_flow(flow, mean_velocity, max_time=max_time)
        save_flow(OUT / f"flow_N{flow.resolution}_{velocity_m_per_day:.0f}m_per_day.npz", case_flow)
        for condition_index, condition in enumerate(("favorable", "unfavorable")):
            seed = 1000 + 100 * velocity_index + condition_index
            stem = f"{condition}_{velocity_m_per_day:.0f}m_per_day"
            print(
                f"Tracking {particle_count} particles: {condition}, "
                f"velocity={velocity_m_per_day:.0f} m/day, max_time={max_time:.1f} s"
            )
            lib = simulate_cell_transitions(case_flow, condition, n_particles=particle_count, seed=seed)
            save_library(OUT / f"trajectory_library_{stem}.npz", lib)
            mats = transition_matrices(lib, n_bins=bins)
            write_matrix(OUT / f"transition_counts_{stem}.csv", mats["counts"])
            write_matrix(OUT / f"transition_probabilities_{stem}.csv", mats["probabilities"])
            write_matrix(OUT / f"transition_mean_travel_time_{stem}.csv", mats["mean_travel_time"])
            write_matrix(OUT / f"transition_mean_dy_{stem}.csv", mats["mean_dy"])
            write_matrix(OUT / f"transition_mean_interceptions_{stem}.csv", mats["mean_interceptions"])
            write_matrix(OUT / f"transition_mean_near_time_{stem}.csv", mats["mean_near_time"])
            bin_rows = []
            for i in range(bins):
                bin_rows.append(
                    {
                        "bin": i,
                        "y_left_m": mats["bins"][i],
                        "y_right_m": mats["bins"][i + 1],
                        "total": int(mats["total_by_in"][i]),
                        "exited": int(mats["exited_by_in"][i]),
                        "attached": int(mats["attached_by_in"][i]),
                        "censored": int(mats["censored_by_in"][i]),
                        "exit_probability": float(mats["exit_prob_by_in"][i]),
                        "attachment_probability": float(mats["attach_prob_by_in"][i]),
                    }
                )
            write_dicts(OUT / f"bin_outcomes_{stem}.csv", bin_rows)
            if velocity_m_per_day == 4.0:
                save_library(OUT / f"trajectory_library_{condition}.npz", lib)
                write_matrix(OUT / f"transition_counts_{condition}.csv", mats["counts"])
                write_matrix(OUT / f"transition_probabilities_{condition}.csv", mats["probabilities"])
                write_matrix(OUT / f"transition_mean_travel_time_{condition}.csv", mats["mean_travel_time"])
                write_matrix(OUT / f"transition_mean_dy_{condition}.csv", mats["mean_dy"])
                write_matrix(OUT / f"transition_mean_interceptions_{condition}.csv", mats["mean_interceptions"])
                write_matrix(OUT / f"transition_mean_near_time_{condition}.csv", mats["mean_near_time"])
                write_dicts(OUT / f"bin_outcomes_{condition}.csv", bin_rows)
            summary = summarize_library(condition, velocity_m_per_day, lib, mats)
            summaries.append(summary)
            print(
                "  exited",
                f"{summary['exited_fraction']:.3f}",
                "attached",
                f"{summary['attached_fraction']:.3f}",
                "censored",
                f"{summary['censored_fraction']:.3f}",
                "row support",
                summary["min_transition_row_support"],
                "/",
                f"{summary['median_transition_row_support']:.1f}",
                "zero rows",
                summary["zero_transition_rows"],
            )
    write_dicts(OUT / "trajectory_summary.csv", summaries)
    write_quality_report(OUT / "quality_report.md", params, flow_rows, summaries)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
