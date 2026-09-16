#!/usr/bin/env python3
"""Adaptive long-horizon audit for unresolved periodic-cell trajectories.

The goal is to keep finite-time censoring out of the reported physics.  The
script reruns the focused transition-matrix and central-core refinement
ensembles at increasing one-cell horizons until every particle has either
exited or attached.  If unresolved particles remain at the final horizon and
their histories are dominated by the near-surface/secondary-minimum zone, they
are flagged as "non physical trapping/attachment" for manual inspection rather
than treated as homogeneous unfavorable attachment.
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
sys.path.insert(0, str(ROOT / "scripts"))

import run_focusing_transition_matrices as ftm  # noqa: E402
import run_periodic_cell_refinement as pcr  # noqa: E402
from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled  # noqa: E402
from colloid_tsm.physical import (  # noqa: E402
    PhysicalParams,
    _surface_state_with_collector,
    angular_distance,
    load_flow,
    load_library,
    save_library,
)


OUT = ROOT / "outputs" / "censoring_audit"
FIGURES = ROOT / "outputs" / "figures"
MATRIX_BASE = ROOT / "outputs" / "focusing_transition_matrices"
REFINEMENT_BASE = ROOT / "outputs" / "periodic_cell_refinement"

MATRIX_HORIZONS_S = (240.0, 600.0, 1200.0, 2400.0)
REFINEMENT_HORIZONS_S = (240.0, 600.0, 1200.0, 2400.0)
FINAL_NEAR_RESIDENCE_FRACTION = 0.80


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def finite_quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def profile_key(profile: dict[str, object]) -> str:
    return str(profile["profile"])


def with_horizon(params: PhysicalParams, horizon: float) -> PhysicalParams:
    return replace(params, max_time=float(horizon))


def final_surface_state(params: PhysicalParams, lib) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, theta, _, _, _, collector = _surface_state_with_collector(params, lib.x_final, lib.y_final)
    return h, theta, collector


def classify_censoring(
    *,
    lib,
    params: PhysicalParams,
    horizon: float,
    final_horizon: float,
) -> dict[str, float | int | str]:
    censored = lib.censored
    h_final, theta_final, collector_final = final_surface_state(params, lib)
    near_final = censored & (h_final <= params.near_surface)
    well_interceptions = (
        lib.well_interceptions if lib.well_interceptions is not None else np.zeros(lib.y_in.size, dtype=int)
    )
    center_well_interceptions = (
        lib.center_well_interceptions
        if lib.center_well_interceptions is not None
        else np.zeros(lib.y_in.size, dtype=int)
    )
    corner_well_interceptions = (
        lib.corner_well_interceptions
        if lib.corner_well_interceptions is not None
        else np.zeros(lib.y_in.size, dtype=int)
    )
    well_time = lib.well_time if lib.well_time is not None else np.zeros(lib.y_in.size)
    near_residence_fraction = np.divide(
        lib.near_time,
        horizon,
        out=np.zeros_like(lib.near_time, dtype=float),
        where=horizon > 0.0,
    )
    well_residence_fraction = np.divide(
        well_time,
        horizon,
        out=np.zeros_like(well_time, dtype=float),
        where=horizon > 0.0,
    )
    persistent_near = censored & (
        near_final
        | (near_residence_fraction >= FINAL_NEAR_RESIDENCE_FRACTION)
        | (well_residence_fraction >= FINAL_NEAR_RESIDENCE_FRACTION)
    )
    center_persistent = persistent_near & (collector_final == 1)
    corner_persistent = persistent_near & (collector_final == 0)
    unresolved = int(np.sum(censored))
    persistent_count = int(np.sum(persistent_near))
    if unresolved == 0:
        status = "complete"
    elif horizon < final_horizon:
        status = "extend"
    elif persistent_count > 0:
        status = "non physical trapping/attachment"
    else:
        status = "needs longer horizon"

    return {
        "status": status,
        "censored_count": unresolved,
        "near_surface_final_count": int(np.sum(near_final)),
        "persistent_near_surface_count": persistent_count,
        "nonphysical_trapping_attachment_count": persistent_count if status == "non physical trapping/attachment" else 0,
        "needs_longer_horizon_count": unresolved - persistent_count if status == "needs longer horizon" else 0,
        "censored_center_final_count": int(np.sum(censored & (collector_final == 1))),
        "censored_corner_final_count": int(np.sum(censored & (collector_final == 0))),
        "persistent_center_final_count": int(np.sum(center_persistent)),
        "persistent_corner_final_count": int(np.sum(corner_persistent)),
        "censored_center_well_count": int(np.sum(censored & (center_well_interceptions > 0))),
        "censored_corner_well_count": int(np.sum(censored & (corner_well_interceptions > 0))),
        "censored_well_count": int(np.sum(censored & (well_interceptions > 0))),
        "censored_median_h_final_nm": finite_median(h_final[censored]) * 1.0e9,
        "censored_p95_h_final_nm": finite_quantile(h_final[censored], 0.95) * 1.0e9,
        "censored_median_theta_final_deg": finite_median(
            np.degrees(np.abs(angular_distance(theta_final[censored], 0.0)))
        ),
        "censored_median_near_time_s": finite_median(lib.near_time[censored]),
        "censored_median_well_time_s": finite_median(well_time[censored]),
        "censored_median_near_residence_fraction": finite_median(near_residence_fraction[censored]),
        "censored_median_well_residence_fraction": finite_median(well_residence_fraction[censored]),
    }


def summarize_common(
    *,
    ensemble: str,
    profile: dict[str, object],
    params: PhysicalParams,
    horizon: float,
    final_horizon: float,
    lib,
    output_path: Path,
) -> dict[str, float | int | str]:
    mobile = lib.exited & (~lib.attached)
    center_well = (
        lib.center_well_interceptions > 0
        if lib.center_well_interceptions is not None
        else np.zeros(lib.y_in.size, dtype=bool)
    )
    center_well_release = (
        mobile
        & center_well
        & (lib.collector_well_exit == 1)
        & np.isfinite(lib.theta_well_exit)
    )
    theta_well = np.abs(angular_distance(lib.theta_well_exit[center_well_release], 0.0))
    center_near = lib.center_interceptions > 0
    center_near_release = mobile & center_near & (lib.collector_exit == 1) & np.isfinite(lib.theta_exit)
    theta_near = np.abs(angular_distance(lib.theta_exit[center_near_release], 0.0))
    yout_center = np.abs(lib.y_out - 0.5 * params.cell_length) <= 12.5e-6
    classification = classify_censoring(
        lib=lib,
        params=params,
        horizon=horizon,
        final_horizon=final_horizon,
    )
    row: dict[str, float | int | str] = {
        "ensemble": ensemble,
        "profile": profile_key(profile),
        "label": str(profile["label"]),
        "condition": str(profile["condition"]),
        "horizon_s": float(horizon),
        "particles": int(lib.y_in.size),
        "exited_count": int(np.sum(lib.exited)),
        "attached_count": int(np.sum(lib.attached)),
        "mobile_count": int(np.sum(mobile)),
        "mobile_yout_center_fraction": float(np.mean(yout_center[mobile])) if np.any(mobile) else float("nan"),
        "center_near_entries": int(np.sum(center_near)),
        "center_near_releases": int(np.sum(center_near_release)),
        "center_near_release_f30": float(np.mean(theta_near <= np.deg2rad(30.0))) if theta_near.size else float("nan"),
        "center_near_release_median_theta_deg": finite_median(np.degrees(theta_near)),
        "center_well_entries": int(np.sum(center_well)),
        "center_well_releases": int(np.sum(center_well_release)),
        "center_well_release_f30": float(np.mean(theta_well <= np.deg2rad(30.0))) if theta_well.size else float("nan"),
        "center_well_release_median_theta_deg": finite_median(np.degrees(theta_well)),
        "center_well_yout_center_fraction": float(np.mean(yout_center[center_well_release]))
        if np.any(center_well_release)
        else float("nan"),
        "center_well_time_median_s": finite_median(
            (lib.well_time if lib.well_time is not None else np.zeros(lib.y_in.size))[center_well]
        ),
        "travel_time_mobile_median_s": finite_median(lib.travel_time[mobile]),
        "travel_time_mobile_p95_s": finite_quantile(lib.travel_time[mobile], 0.95),
        "library_path": str(output_path),
    }
    row.update(classification)
    return row


def source_library_path(ensemble: str, profile_name: str, horizon: float) -> Path | None:
    if abs(horizon - 240.0) > 1.0e-9:
        return None
    if ensemble == "transition_matrix":
        path = MATRIX_BASE / f"trajectory_library_{profile_name}.npz"
    elif ensemble == "central_core_refinement":
        path = REFINEMENT_BASE / f"trajectory_library_{profile_name}.npz"
    else:
        return None
    return path if path.exists() else None


def audit_library_path(ensemble: str, profile_name: str, horizon: float) -> Path:
    seconds = int(round(horizon))
    return OUT / f"trajectory_library_{ensemble}_{profile_name}_{seconds}s.npz"


def run_or_load(
    *,
    ensemble: str,
    profile: dict[str, object],
    params: PhysicalParams,
    initial_y: np.ndarray,
    seed: int,
    horizon: float,
    reuse: bool,
):
    profile_name = profile_key(profile)
    audit_path = audit_library_path(ensemble, profile_name, horizon)
    if reuse and audit_path.exists():
        return load_library(audit_path, params=params), audit_path
    source_path = source_library_path(ensemble, profile_name, horizon)
    if source_path is not None:
        lib = load_library(source_path, params=params)
        save_library(audit_path, lib)
        return lib, audit_path
    flow = load_flow(ftm.FLOW_PATH, params=params)
    lib = simulate_cell_transitions_compiled(
        flow,
        str(profile["condition"]),
        seed=seed,
        allow_attachment=False,
        initial_y=initial_y,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )
    save_library(audit_path, lib)
    return lib, audit_path


def run_ensemble(
    *,
    ensemble: str,
    profiles: tuple[dict[str, object], ...],
    base_params_for,
    initial_y: np.ndarray,
    horizons: tuple[float, ...],
    seed_base: int,
    seed_stride: int,
    reuse: bool,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    final_horizon = horizons[-1]
    for profile_id, profile in enumerate(profiles):
        profile_name = profile_key(profile)
        completed = False
        for horizon in horizons:
            params = with_horizon(base_params_for(profile), horizon)
            print(f"{ensemble}: {profile_name}, horizon {horizon:.0f} s", flush=True)
            lib, path = run_or_load(
                ensemble=ensemble,
                profile=profile,
                params=params,
                initial_y=initial_y,
                seed=seed_base + seed_stride * profile_id,
                horizon=horizon,
                reuse=reuse,
            )
            row = summarize_common(
                ensemble=ensemble,
                profile=profile,
                params=params,
                horizon=horizon,
                final_horizon=final_horizon,
                lib=lib,
                output_path=path,
            )
            rows.append(row)
            write_csv(OUT / "long_horizon_censoring_audit_partial.csv", rows)
            if int(row["censored_count"]) == 0:
                completed = True
                break
        if not completed:
            print(f"{ensemble}: {profile_name} still unresolved at {final_horizon:.0f} s", flush=True)
    return rows


def make_figure(rows: list[dict[str, float | int | str]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, ensemble, title in (
        (axes[0], "transition_matrix", "focused transition ensemble"),
        (axes[1], "central_core_refinement", "central-core refinement ensemble"),
    ):
        subset = [row for row in rows if row["ensemble"] == ensemble]
        labels = []
        for row in subset:
            label = str(row["label"])
            if label not in labels:
                labels.append(label)
        for label in labels:
            series = [row for row in subset if row["label"] == label]
            x = np.array([float(row["horizon_s"]) for row in series])
            y = np.array([float(row["censored_count"]) for row in series])
            ax.plot(x, y, marker="o", linewidth=1.8, label=label)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("one-cell tracking horizon (s)")
        ax.set_ylabel("unresolved particles")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Adaptive horizon audit for finite-time censoring", fontsize=14)
    fig.savefig(OUT / "long_horizon_censoring_audit.png", dpi=220)
    fig.savefig(FIGURES / "fig20_long_horizon_censoring.png", dpi=220)
    plt.close(fig)


def write_report(rows: list[dict[str, float | int | str]]) -> None:
    final_rows: list[dict[str, float | int | str]] = []
    for ensemble in ("transition_matrix", "central_core_refinement"):
        subset = [row for row in rows if row["ensemble"] == ensemble]
        labels = []
        for row in subset:
            key = str(row["profile"])
            if key not in labels:
                labels.append(key)
        for key in labels:
            series = [row for row in subset if row["profile"] == key]
            final_rows.append(series[-1])

    lines = [
        "# Long-horizon censoring audit",
        "",
        "Finite-time censoring is treated as a simulation-design problem, not as a physical attachment outcome. Each ensemble is rerun at increasing one-cell horizons until all particles exit or attach. If unresolved particles remain at the final tested horizon and are still dominated by near-surface or secondary-minimum residence, they are flagged as `non physical trapping/attachment` for model review.",
        "",
        "## Final status by reported ensemble",
        "",
        "| ensemble | case | final horizon (s) | particles | exited | censored | status | non physical trapping/attachment | center-well entries | center-well releases | center-well F30 | median well theta (deg) | median censored h (nm) | median censored well time (s) |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ensemble"]),
                    str(row["label"]),
                    fmt(row["horizon_s"], 0),
                    fmt(row["particles"], 0),
                    fmt(row["exited_count"], 0),
                    fmt(row["censored_count"], 0),
                    str(row["status"]),
                    fmt(row["nonphysical_trapping_attachment_count"], 0),
                    fmt(row["center_well_entries"], 0),
                    fmt(row["center_well_releases"], 0),
                    fmt(row["center_well_release_f30"]),
                    fmt(row["center_well_release_median_theta_deg"]),
                    fmt(row["censored_median_h_final_nm"]),
                    fmt(row["censored_median_well_time_s"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Horizon sequence",
            "",
            "| ensemble | case | horizon (s) | censored | near-final | persistent near-surface | center final | corner final | status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ensemble"]),
                    str(row["label"]),
                    fmt(row["horizon_s"], 0),
                    fmt(row["censored_count"], 0),
                    fmt(row["near_surface_final_count"], 0),
                    fmt(row["persistent_near_surface_count"], 0),
                    fmt(row["censored_center_final_count"], 0),
                    fmt(row["censored_corner_final_count"], 0),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: a completed row means the transition statistics no longer contain finite-time mobile censoring. A `non physical trapping/attachment` row means the numerical SDE/near-wall closure has produced an effectively irreversible near-surface state under a homogeneous unfavorable chemistry where no attachment mechanism was enabled.",
            "",
        ]
    )
    (OUT / "long_horizon_censoring_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_kernel(force=False)
    reuse = "--reuse" in sys.argv

    matrix_reference_params = ftm.params_for(ftm.PROFILES[0])
    matrix_initial_y = ftm.sample_y(matrix_reference_params, ftm.SEED)
    refinement_reference_params = pcr.params_for(pcr.PROFILES[0])
    refinement_initial_y = pcr.central_core_y(refinement_reference_params)

    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        run_ensemble(
            ensemble="transition_matrix",
            profiles=ftm.PROFILES,
            base_params_for=ftm.params_for,
            initial_y=matrix_initial_y,
            horizons=MATRIX_HORIZONS_S,
            seed_base=ftm.SEED,
            seed_stride=100,
            reuse=reuse,
        )
    )
    rows.extend(
        run_ensemble(
            ensemble="central_core_refinement",
            profiles=pcr.PROFILES,
            base_params_for=pcr.params_for,
            initial_y=refinement_initial_y,
            horizons=REFINEMENT_HORIZONS_S,
            seed_base=pcr.SEED + 10,
            seed_stride=1,
            reuse=reuse,
        )
    )
    write_csv(OUT / "long_horizon_censoring_audit.csv", rows)
    make_figure(rows)
    write_report(rows)
    print(OUT / "long_horizon_censoring_report.md", flush=True)


if __name__ == "__main__":
    main()
