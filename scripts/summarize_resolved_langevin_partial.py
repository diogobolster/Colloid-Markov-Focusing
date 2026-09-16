#!/usr/bin/env python3
"""Summarize completed resolved-Langevin diagnostic libraries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import load_library


OUT = ROOT / "outputs" / "resolved_langevin_50mM"
FILES = (
    ("uniform neutral lubrication", OUT / "trajectory_library_uniform_neutral_lubrication.npz"),
    ("uniform 6mM lubrication", OUT / "trajectory_library_uniform_unfavorable_6mM_lubrication.npz"),
    ("uniform 6mM resolved", OUT / "trajectory_library_uniform_unfavorable_6mM_resolved.npz"),
    ("uniform 50mM lubrication", OUT / "trajectory_library_uniform_unfavorable_50mM_lubrication.npz"),
    ("uniform 50mM resolved", OUT / "trajectory_library_uniform_unfavorable_50mM_resolved.npz"),
)


def angular_distance(theta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta), np.cos(theta))


def summarize(label: str, path: Path) -> None:
    lib = load_library(path)
    mobile = lib.exited & (~lib.attached)
    release = mobile & (lib.interceptions > 0) & np.isfinite(lib.theta_exit)
    theta = np.abs(angular_distance(lib.theta_exit[release]))
    near = lib.near_time[release]
    hmin = lib.h_min[release] * 1.0e9
    contacts = lib.contact_events[release]
    print(
        f"{label:28s} "
        f"releases={int(np.sum(release)):4d} "
        f"censored={float(np.mean(lib.censored)):.4f} "
        f"theta_med={float(np.median(theta)) if theta.size else float('nan'):.3f} "
        f"theta_lt30={float(np.mean(theta <= np.deg2rad(30))) if theta.size else float('nan'):.3f} "
        f"near_med={float(np.median(near)) if near.size else float('nan'):.3f}s "
        f"near_p95={float(np.quantile(near, 0.95)) if near.size else float('nan'):.3f}s "
        f"hmin_med={float(np.median(hmin)) if hmin.size else float('nan'):.2f}nm "
        f"contact_frac={float(np.mean(contacts > 0)) if contacts.size else float('nan'):.3f} "
        f"mean_contacts={float(np.mean(contacts)) if contacts.size else float('nan'):.3f}"
    )


def main() -> None:
    for label, path in FILES:
        if path.exists():
            summarize(label, path)


if __name__ == "__main__":
    main()
