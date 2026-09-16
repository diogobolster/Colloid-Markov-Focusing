#!/usr/bin/env python3
"""Focused periodic-cell transition matrices for DLVO focusing.

The matrix design is intentionally not a uniform fine grid. It separates the
center-grazing inlet support from outer inlet states so the focusing mechanism
cannot be averaged away by particles that never interact with the center grain.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled  # noqa: E402
from colloid_tsm.physical import (  # noqa: E402
    PhysicalParams,
    angular_distance,
    load_flow,
    load_library,
    save_library,
)


FLOW_PATH = ROOT / "outputs" / "physical" / "flow_N192.npz"
OUT = ROOT / "outputs" / "focusing_transition_matrices"
FIGURES = ROOT / "outputs" / "figures"

VELOCITY_M_PER_DAY = 4.0
MAX_TIME_S = 1200.0
SEED = 1_740_000
PROFILE_MAX_TIME_S = {
    "neutral_resolved": 240.0,
    "unfavorable_50mM_z70": 600.0,
    "unfavorable_50mM_z50": 600.0,
    "unfavorable_50mM_z30": 1200.0,
}

PROFILES = (
    {
        "profile": "neutral_resolved",
        "label": "No DLVO",
        "condition": "neutral",
        "updates": {"ionic_strength_molar": 6.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z70",
        "label": "50 mM, -70 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -70.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z50",
        "label": "50 mM, -50 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -50.0e-3},
    },
    {
        "profile": "unfavorable_50mM_z30",
        "label": "50 mM, -30 mV",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -30.0e-3},
    },
)

STATE_NAMES = (
    "outer lower",
    "shoulder lower",
    "inner lower",
    "center core",
    "inner upper",
    "shoulder upper",
    "outer upper",
)
STATE_PARTICLES = np.array([2500, 3500, 5000, 24000, 5000, 3500, 2500], dtype=int)
ANGLE_EDGES_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 180.0])


def horizon_for(profile: dict[str, object]) -> float:
    return float(PROFILE_MAX_TIME_S.get(str(profile["profile"]), MAX_TIME_S))


def params_for(profile: dict[str, object]) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        mean_velocity=VELOCITY_M_PER_DAY / 86400.0,
        max_time=horizon_for(profile),
        resolved_langevin_substeps=25,
        resolved_langevin_substep_cutoff=75.0e-9,
        resolved_langevin_normal_step=3.0e-9,
        resolved_langevin_min_dt=1.0e-6,
        **dict(profile["updates"]),
    )


def state_edges(params: PhysicalParams) -> np.ndarray:
    center = 0.5 * params.cell_length
    fixed = np.array([150.0, 175.0, 187.5, 212.5, 225.0, 250.0]) * 1.0e-6
    return np.concatenate(([params.inlet_y_min], fixed, [params.inlet_y_max]))


def state_labels(params: PhysicalParams) -> list[str]:
    edges_um = state_edges(params) * 1.0e6
    return [f"{name}\n{edges_um[i]:.0f}-{edges_um[i + 1]:.0f}" for i, name in enumerate(STATE_NAMES)]


def sample_y(params: PhysicalParams, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = state_edges(params)
    chunks = []
    for n, lo, hi in zip(STATE_PARTICLES, edges[:-1], edges[1:]):
        chunks.append(rng.uniform(lo, hi, size=int(n)))
    return np.concatenate(chunks)


def bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges) - 1, 0, edges.size - 2)


def row_normalize(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.full_like(counts, np.nan, dtype=float), where=totals > 0)


def angle_bin_indices(theta: np.ndarray) -> np.ndarray:
    abs_theta_deg = np.degrees(np.abs(angular_distance(theta, 0.0)))
    edges = ANGLE_EDGES_DEG
    return np.clip(np.digitize(abs_theta_deg, edges) - 1, 0, edges.size - 2)


def matrix_from_events(row_ids: np.ndarray, col_ids: np.ndarray, mask: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    counts = np.zeros((n_rows, n_cols), dtype=int)
    for i, j in zip(row_ids[mask], col_ids[mask]):
        counts[int(i), int(j)] += 1
    return counts


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def finite_fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if mask.size else float("nan")


def profile_horizon_from_name(profile_name: str) -> float:
    return float(PROFILE_MAX_TIME_S.get(profile_name, MAX_TIME_S))


def build_transition_payload(lib, params: PhysicalParams, profile: dict[str, object]) -> dict[str, np.ndarray]:
    edges = state_edges(params)
    n_states = edges.size - 1
    y_in_state = bin_indices(lib.y_in, edges)
    y_out_state = bin_indices(lib.y_out, edges)
    row_total = np.bincount(y_in_state, minlength=n_states)
    mobile = lib.exited & (~lib.attached)
    center_near = lib.center_interceptions > 0
    center_near_release = mobile & center_near & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
    near_angle_state = np.zeros(lib.y_in.size, dtype=int)
    near_angle_state[center_near_release] = angle_bin_indices(lib.theta_exit[center_near_release])

    if lib.center_well_interceptions is None:
        center_well = np.zeros(lib.y_in.size, dtype=bool)
    else:
        center_well = lib.center_well_interceptions > 0
    center_well_release = (
        mobile
        & center_well
        & (lib.collector_well_exit == 1)
        & np.isfinite(lib.theta_well_exit)
    )
    well_angle_state = np.zeros(lib.y_in.size, dtype=int)
    well_angle_state[center_well_release] = angle_bin_indices(lib.theta_well_exit[center_well_release])

    mobile_counts = matrix_from_events(y_in_state, y_out_state, mobile, n_states, n_states)
    near_angle_counts = matrix_from_events(
        y_in_state,
        near_angle_state,
        center_near_release,
        n_states,
        ANGLE_EDGES_DEG.size - 1,
    )
    well_angle_counts = matrix_from_events(
        y_in_state,
        well_angle_state,
        center_well_release,
        n_states,
        ANGLE_EDGES_DEG.size - 1,
    )
    well_yout_counts = matrix_from_events(y_in_state, y_out_state, center_well_release, n_states, n_states)

    return {
        "state_edges_m": edges,
        "row_total": row_total,
        "mobile_counts": mobile_counts,
        "mobile_probability_conditional": row_normalize(mobile_counts),
        "center_near_angle_counts": near_angle_counts,
        "center_near_angle_probability": row_normalize(near_angle_counts),
        "center_well_angle_counts": well_angle_counts,
        "center_well_angle_probability": row_normalize(well_angle_counts),
        "center_well_yout_counts": well_yout_counts,
        "center_well_yout_probability": row_normalize(well_yout_counts),
        "angle_edges_deg": ANGLE_EDGES_DEG,
        "y_in_state": y_in_state,
        "y_out_state": y_out_state,
        "mobile": mobile.astype(np.uint8),
        "center_near_release": center_near_release.astype(np.uint8),
        "center_well_release": center_well_release.astype(np.uint8),
    }


def summarize_rows(lib, params: PhysicalParams, profile: dict[str, object]) -> list[dict[str, float | int | str]]:
    edges = state_edges(params)
    y_in_state = bin_indices(lib.y_in, edges)
    mobile = lib.exited & (~lib.attached)
    y_out_center = np.abs(lib.y_out - 0.5 * params.cell_length) <= 12.5e-6
    center_near = lib.center_interceptions > 0
    center_near_release = mobile & center_near & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
    theta_near = np.abs(angular_distance(lib.theta_exit, 0.0))
    if lib.center_well_interceptions is None:
        center_well = np.zeros(lib.y_in.size, dtype=bool)
    else:
        center_well = lib.center_well_interceptions > 0
    center_well_release = (
        mobile
        & center_well
        & (lib.collector_well_exit == 1)
        & np.isfinite(lib.theta_well_exit)
    )
    theta_well = np.abs(angular_distance(lib.theta_well_exit, 0.0))
    rows: list[dict[str, float | int | str]] = []
    for state_id, state_name in enumerate(STATE_NAMES):
        row_mask = y_in_state == state_id
        mobile_row = row_mask & mobile
        near_row = row_mask & center_near
        near_release_row = row_mask & center_near_release
        well_row = row_mask & center_well
        well_release_row = row_mask & center_well_release
        rows.append(
            {
                "profile": str(profile["profile"]),
                "label": str(profile["label"]),
                "state": state_name,
                "state_id": state_id,
                "y_min_um": float(edges[state_id] * 1.0e6),
                "y_max_um": float(edges[state_id + 1] * 1.0e6),
                "particles": int(np.sum(row_mask)),
                "mobile_exits": int(np.sum(mobile_row)),
                "censored": int(np.sum(row_mask & lib.censored)),
                "mobile_exit_fraction": finite_fraction(mobile[row_mask]),
                "yout_center_fraction_all_mobile": finite_fraction(y_out_center[mobile_row]),
                "median_abs_yout_center_um": finite_median(np.abs(lib.y_out[mobile_row] - 0.5 * params.cell_length) * 1.0e6),
                "center_near_entries": int(np.sum(near_row)),
                "center_near_releases": int(np.sum(near_release_row)),
                "center_near_censored": int(np.sum(row_mask & center_near & lib.censored)),
                "center_near_theta30_fraction": finite_fraction(theta_near[near_release_row] <= np.deg2rad(30.0)),
                "center_near_median_theta_deg": finite_median(np.degrees(theta_near[near_release_row])),
                "center_well_entries": int(np.sum(well_row)),
                "center_well_releases": int(np.sum(well_release_row)),
                "center_well_censored": int(np.sum(row_mask & center_well & lib.censored)),
                "center_well_theta30_fraction": finite_fraction(theta_well[well_release_row] <= np.deg2rad(30.0)),
                "center_well_median_theta_deg": finite_median(np.degrees(theta_well[well_release_row])),
                "center_well_yout_center_fraction": finite_fraction(y_out_center[well_release_row]),
                "center_well_median_abs_yout_center_um": finite_median(
                    np.abs(lib.y_out[well_release_row] - 0.5 * params.cell_length) * 1.0e6
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_or_load(profile: dict[str, object], params: PhysicalParams, initial_y: np.ndarray, profile_id: int, reuse: bool):
    path = OUT / f"trajectory_library_{profile['profile']}.npz"
    if reuse and path.exists():
        candidate = load_library(path)
        if abs(candidate.params.max_time - params.max_time) < 1.0e-9:
            return candidate
    audit_path = ROOT / "outputs" / "censoring_audit" / (
        f"trajectory_library_transition_matrix_{profile['profile']}_{int(round(params.max_time))}s.npz"
    )
    if reuse and audit_path.exists():
        lib = load_library(audit_path)
        save_library(path, lib)
        return lib
    flow = load_flow(FLOW_PATH, params=params)
    lib = simulate_cell_transitions_compiled(
        flow,
        str(profile["condition"]),
        seed=SEED + 100 * profile_id,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )
    save_library(path, lib)
    return lib


def save_payloads(payloads: dict[str, dict[str, np.ndarray]]) -> None:
    for profile, payload in payloads.items():
        np.savez_compressed(OUT / f"transition_payload_{profile}.npz", **payload)


def plot_heatmap(ax, data: np.ndarray, title: str, x_labels: list[str], y_labels: list[str], vmin=0.0, vmax=1.0) -> None:
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value) and value >= 0.08:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white")
    return im


def make_figure(payloads: dict[str, dict[str, np.ndarray]], params: PhysicalParams) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    y_labels = state_labels(params)
    angle_labels = [f"{ANGLE_EDGES_DEG[i]:.0f}-{ANGLE_EDGES_DEG[i + 1]:.0f}" for i in range(ANGLE_EDGES_DEG.size - 1)]
    fig, axes = plt.subplots(2, len(PROFILES), figsize=(15, 7.5), constrained_layout=True)
    for col, profile in enumerate(PROFILES):
        name = str(profile["profile"])
        payload = payloads[name]
        im0 = plot_heatmap(
            axes[0, col],
            payload["mobile_probability_conditional"],
            f"{profile['label']}\nP(y_out | y_in, mobile)",
            [label.replace("\n", " ") for label in y_labels],
            y_labels if col == 0 else ["" for _ in y_labels],
        )
        if name == "neutral_resolved":
            angle_prob = payload["center_near_angle_probability"]
            event_label = "center near-surface"
        else:
            angle_prob = payload["center_well_angle_probability"]
            event_label = "center secondary min"
        im1 = plot_heatmap(
            axes[1, col],
            angle_prob,
            f"{event_label}\nP(|theta_release| | y_in)",
            angle_labels,
            y_labels if col == 0 else ["" for _ in y_labels],
        )
    axes[0, 0].set_ylabel("inlet state")
    axes[1, 0].set_ylabel("inlet state")
    fig.colorbar(im0, ax=axes[0, :], shrink=0.82, label="transition probability")
    fig.colorbar(im1, ax=axes[1, :], shrink=0.82, label="event-conditioned probability")
    fig.suptitle("Focused transition matrices with center-core and outer inlet states", fontsize=14)
    fig.savefig(OUT / "focusing_transition_matrices.png", dpi=220)
    fig.savefig(FIGURES / "fig19_focusing_transition_matrices.png", dpi=220)
    plt.close(fig)


def write_report(rows: list[dict[str, float | int | str]]) -> None:
    central = [r for r in rows if r["state"] == "center core"]

    def fmt(value: object, digits: int = 3) -> str:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        try:
            x = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(x):
            return "nan"
        if digits == 0:
            return str(int(round(x)))
        return f"{x:.{digits}g}"

    lines = [
        "# Focused transition matrices",
        "",
        f"Runs use the N=192 center/corner periodic flow, {VELOCITY_M_PER_DAY:.0f} m/day mean velocity, homogeneous surfaces, no unfavorable attachment, and adaptive one-cell horizons selected from the long-horizon censoring audit.",
        "The inlet state is intentionally nonuniform: outer and shoulder bins preserve background mobile transitions, while the center-core bin carries high support for center-grain focusing.",
        "",
        "## Inlet states",
        "",
        "| state | y range (um) | particles per case |",
        "|---|---:|---:|",
    ]
    params = params_for(PROFILES[0])
    edges = state_edges(params) * 1.0e6
    for i, state in enumerate(STATE_NAMES):
        lines.append(f"| {state} | {edges[i]:.2f}-{edges[i+1]:.2f} | {int(STATE_PARTICLES[i])} |")
    lines.extend(
        [
            "",
            "## Center-core row",
            "",
            "This is the row expected to show focusing. For the no-DLVO case the angle matrix uses center near-surface releases. For unfavorable cases it uses center secondary-minimum releases.",
            "",
            "| case | horizon (s) | mobile exits | censored | P(yout center given mobile) | center near rel | near F30 | near median theta | center well rel | well F30 | well median theta | P(yout center given well) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in central:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt(profile_horizon_from_name(str(row["profile"])), 0),
                    fmt(row["mobile_exits"], 0),
                    fmt(row["censored"], 0),
                    fmt(row["yout_center_fraction_all_mobile"]),
                    fmt(row["center_near_releases"], 0),
                    fmt(row["center_near_theta30_fraction"]),
                    fmt(row["center_near_median_theta_deg"]),
                    fmt(row["center_well_releases"], 0),
                    fmt(row["center_well_theta30_fraction"]),
                    fmt(row["center_well_median_theta_deg"]),
                    fmt(row["center_well_yout_center_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## All inlet rows",
            "",
            "| case | state | mobile exits | center near entries | center near releases | near F30 | center well entries | center well releases | well F30 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["state"]),
                    fmt(row["mobile_exits"], 0),
                    fmt(row["center_near_entries"], 0),
                    fmt(row["center_near_releases"], 0),
                    fmt(row["center_near_theta30_fraction"]),
                    fmt(row["center_well_entries"], 0),
                    fmt(row["center_well_releases"], 0),
                    fmt(row["center_well_theta30_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The full mobile y_out matrices retain the ordinary one-cell transition information. The lower angle matrices are the focusing diagnostic: only the center-grazing row carries substantial center-grain event support, so outer rows provide a clean negative control.",
            "A clear focusing signal is present when the center-core unfavorable secondary-minimum row shifts probability into the 0-30 degree release-angle columns relative to the no-DLVO center near-surface row.",
            "",
        ]
    )
    (OUT / "focusing_transition_matrices_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_kernel(force=False)
    reuse = "--reuse" in sys.argv
    reference_params = params_for(PROFILES[0])
    initial_y = sample_y(reference_params, SEED)
    payloads: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | int | str]] = []
    for profile_id, profile in enumerate(PROFILES):
        params = params_for(profile)
        print(f"Running {profile['profile']} with {initial_y.size} particles", flush=True)
        lib = run_or_load(profile, params, initial_y, profile_id, reuse)
        payload = build_transition_payload(lib, params, profile)
        payloads[str(profile["profile"])] = payload
        rows.extend(summarize_rows(lib, params, profile))
        write_csv(OUT / "focusing_transition_matrix_rows_partial.csv", rows)
    save_payloads(payloads)
    write_csv(OUT / "focusing_transition_matrix_rows.csv", rows)
    make_figure(payloads, reference_params)
    write_report(rows)
    print(OUT / "focusing_transition_matrices_report.md", flush=True)


if __name__ == "__main__":
    main()
