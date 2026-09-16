#!/usr/bin/env python3
"""Create survival/completion curves for finite-horizon center-well residence."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colloid_tsm.physical import load_library  # noqa: E402


OUT = ROOT / "outputs" / "center_well_survival"
FIGURES = ROOT / "outputs" / "figures"
FULL_SUITE = ROOT / "outputs" / "openfoam_full_suite"
PROFILES = (
    ("unfavorable_50mM_z70", "50 mM, -70 mV"),
    ("unfavorable_100mM_z70", "100 mM, -70 mV"),
    ("unfavorable_100mM_z30", "100 mM, -30 mV"),
    ("mechanism_50mM_z20_A10x", "50 mM, -20 mV, 10x A"),
    ("mechanism_100mM_z20_A10x", "100 mM, -20 mV, 10x A"),
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile_arrays(profile: str):
    lib = load_library(FULL_SUITE / f"trajectory_library_refinement_{profile}.npz")
    center_well = lib.center_well_interceptions if lib.center_well_interceptions is not None else np.zeros(lib.y_in.size, dtype=int)
    event = center_well > 0
    mobile = lib.exited & (~lib.attached)
    released = mobile & event & (lib.collector_well_exit == 1) & np.isfinite(lib.theta_well_exit)
    unresolved = lib.censored & event
    well_time = lib.well_time if lib.well_time is not None else np.zeros(lib.y_in.size)
    durations = well_time[event]
    released_event = released[event]
    unresolved_event = unresolved[event]
    return lib, durations, released_event, unresolved_event


def curve_rows(profile: str, label: str, n_grid: int = 240) -> tuple[list[dict[str, object]], dict[str, object]]:
    lib, duration, released, unresolved = profile_arrays(profile)
    if duration.size == 0:
        return [], {}
    positive = duration[duration > 0.0]
    t_min = max(float(np.min(positive)) if positive.size else 1.0e-3, 1.0e-3)
    t_max = float(np.max(duration))
    grid = np.unique(np.concatenate(([0.0], np.geomspace(t_min, max(t_max, t_min), n_grid))))
    rows: list[dict[str, object]] = []
    n = duration.size
    for t in grid:
        rows.append(
            {
                "profile": profile,
                "label": label,
                "time_s": float(t),
                "survival_fraction": float(np.mean(duration > t)),
                "completed_release_fraction": float(np.mean(released & (duration <= t))),
                "unresolved_fraction_at_or_beyond_t": float(np.mean(unresolved & (duration >= t))),
            }
        )
    summary = {
        "profile": profile,
        "label": label,
        "particles": int(lib.y_in.size),
        "horizon_s": float(lib.params.max_time),
        "center_well_entries": n,
        "completed_releases": int(np.sum(released)),
        "unresolved_at_horizon": int(np.sum(unresolved)),
        "unresolved_fraction": float(np.mean(unresolved)),
        "median_observed_residence_s": float(np.median(duration)),
        "p90_observed_residence_s": float(np.quantile(duration, 0.90)),
        "p99_observed_residence_s": float(np.quantile(duration, 0.99)),
        "max_observed_residence_s": float(np.max(duration)),
    }
    return rows, summary


def plot(curves: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "unfavorable_50mM_z70": "#7c3aed",
        "unfavorable_100mM_z70": "#581c87",
        "unfavorable_100mM_z30": "#a21caf",
        "mechanism_50mM_z20_A10x": "#dc2626",
        "mechanism_100mM_z20_A10x": "#7f1d1d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), dpi=220)
    for profile, label in PROFILES:
        rows = [row for row in curves if row["profile"] == profile]
        if not rows:
            continue
        t = np.array([float(row["time_s"]) for row in rows])
        survival = np.array([float(row["survival_fraction"]) for row in rows])
        complete = np.array([float(row["completed_release_fraction"]) for row in rows])
        axes[0].plot(t, survival, lw=2.0, color=colors[profile], label=label)
        axes[1].plot(t, complete, lw=2.0, color=colors[profile], label=label)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("observed center-well residence time (s)")
    axes[0].set_ylabel(r"$S(t)=P(\tau_{\rm well}>t)$")
    axes[1].set_ylabel("fraction released by time t")
    axes[0].set_title("Observed residence survival")
    axes[1].set_title("Release completion curve")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Finite-horizon center-well residence and release completion", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "center_well_survival_curves.png", bbox_inches="tight")
    fig.savefig(FIGURES / "center_well_survival_curves.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_curves: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for profile, label in PROFILES:
        path = FULL_SUITE / f"trajectory_library_refinement_{profile}.npz"
        if not path.exists():
            continue
        rows, summary = curve_rows(profile, label)
        all_curves.extend(rows)
        if summary:
            summaries.append(summary)
    write_csv(OUT / "center_well_survival_curves.csv", all_curves)
    write_csv(OUT / "center_well_survival_summary.csv", summaries)
    plot(all_curves, summaries)
    report = [
        "# Center-Well Survival and Release Completion",
        "",
        "| case | entries | releases | unresolved | unresolved fraction | median residence (s) | p99 residence (s) | horizon (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            "| {label} | {center_well_entries} | {completed_releases} | {unresolved_at_horizon} | {unresolved_fraction:.3f} | {median_observed_residence_s:.3g} | {p99_observed_residence_s:.3g} | {horizon_s:.0f} |".format(
                **row
            )
        )
    (OUT / "center_well_survival_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT / "center_well_survival_report.md")


if __name__ == "__main__":
    main()
