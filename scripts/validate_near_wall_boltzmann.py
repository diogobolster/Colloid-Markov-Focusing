#!/usr/bin/env python3
"""Validate the near-wall Langevin update against Boltzmann equilibrium.

This is a deliberately local validation problem: flow is off, the particle is
confined to the unfavorable secondary-minimum basin by reflecting boundaries,
and the one-dimensional wall-normal SDE is advanced with the same mobility,
DLVO drift, and thermal drift formulas used by the production tracker.  The
target stationary density is the conditional Boltzmann distribution
proportional to exp(-U(h)/kBT) on the basin interval.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colloid_tsm.physical import (  # noqa: E402
    PhysicalParams,
    _near_wall_mobility_factors,
    dlvo_force_normal,
    dlvo_potential_energy,
    secondary_minimum_well,
)


OUT = ROOT / "outputs" / "near_wall_boltzmann_validation"
FIG_OUT = ROOT / "outputs" / "figures" / "near_wall_boltzmann_validation.png"
KB = 1.380649e-23


def reflect_interval(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Mirror values into [lower, upper] using periodic reflection."""

    width = upper - lower
    if width <= 0.0:
        raise ValueError("reflection interval has non-positive width")
    z = np.mod(values - lower, 2.0 * width)
    return lower + np.where(z <= width, z, 2.0 * width - z)


def target_grid(
    params: PhysicalParams,
    condition: str,
    lower: float,
    upper: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.linspace(lower, upper, n_grid)
    u = dlvo_potential_energy(params, h, condition) / (KB * params.temperature)
    shifted = np.exp(-(u - np.min(u)))
    density = shifted / np.trapezoid(shifted, h)
    cdf = np.zeros_like(h)
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(h)
    cdf[1:] = np.cumsum(increments)
    cdf /= cdf[-1]
    return h, density, cdf


def sample_from_cdf(
    rng: np.random.Generator,
    h_grid: np.ndarray,
    cdf: np.ndarray,
    n: int,
) -> np.ndarray:
    return np.interp(rng.random(n), cdf, h_grid)


def ks_distance(sample: np.ndarray, h_grid: np.ndarray, cdf: np.ndarray) -> float:
    x = np.sort(sample)
    target = np.interp(x, h_grid, cdf)
    empirical_upper = np.arange(1, x.size + 1, dtype=float) / x.size
    empirical_lower = np.arange(0, x.size, dtype=float) / x.size
    return float(np.max(np.maximum(np.abs(empirical_upper - target), np.abs(empirical_lower - target))))


def advance_wall_sde(
    params: PhysicalParams,
    condition: str,
    h0: np.ndarray,
    lower: float,
    upper: float,
    dt: float,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = h0.copy()
    stokes_mobility = 1.0 / (6.0 * math.pi * params.viscosity * params.particle_radius)
    for _ in range(n_steps):
        normal_mobility, _, dnormal_dh = _near_wall_mobility_factors(params, h)
        force = dlvo_force_normal(params, h, condition)
        drift = stokes_mobility * normal_mobility * force
        if params.wall_mobility_thermal_drift:
            drift = drift + params.diffusivity * dnormal_dh
        sigma = np.sqrt(2.0 * params.diffusivity * normal_mobility * dt)
        h = h + drift * dt + sigma * rng.normal(size=h.size)
        h = reflect_interval(h, lower, upper)
    return h


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

    condition = "unfavorable"
    params = replace(
        PhysicalParams(),
        mean_velocity=0.0,
        ionic_strength_molar=50.0e-3,
        zeta_collector_unfavorable=-70.0e-3,
        dt=1.0e-5,
        max_time=0.04,
    )
    well = secondary_minimum_well(params, condition)
    if not bool(well["has_well"]):
        raise RuntimeError("selected parameter set has no secondary minimum")
    lower = float(well["basin_lower"])
    upper = float(well["basin_upper"])
    h_minimum = float(well["minimum_gap"])
    u_minimum = float(well["minimum_potential_kbt"])

    h_grid, target_density, target_cdf = target_grid(params, condition, lower, upper, 5000)
    rng = np.random.default_rng(20260507)
    n_particles = 30_000
    n_steps = 4000
    dt = 1.0e-5
    stationary_initial = sample_from_cdf(rng, h_grid, target_cdf, n_particles)
    uniform_initial = rng.uniform(lower, upper, n_particles)
    stationary_final = advance_wall_sde(
        params,
        condition,
        stationary_initial,
        lower,
        upper,
        dt,
        n_steps,
        seed=20260508,
    )
    uniform_final = advance_wall_sde(
        params,
        condition,
        uniform_initial,
        lower,
        upper,
        dt,
        n_steps,
        seed=20260509,
    )

    rows = [
        {
            "case": "stationary_initialized",
            "particles": n_particles,
            "dt_s": dt,
            "steps": n_steps,
            "simulated_time_s": dt * n_steps,
            "basin_lower_nm": lower * 1.0e9,
            "basin_upper_nm": upper * 1.0e9,
            "minimum_gap_nm": h_minimum * 1.0e9,
            "minimum_kBT": u_minimum,
            "ks_initial_vs_boltzmann": ks_distance(stationary_initial, h_grid, target_cdf),
            "ks_final_vs_boltzmann": ks_distance(stationary_final, h_grid, target_cdf),
            "mean_initial_nm": float(np.mean(stationary_initial) * 1.0e9),
            "mean_final_nm": float(np.mean(stationary_final) * 1.0e9),
            "mean_boltzmann_nm": float(np.trapezoid(h_grid * target_density, h_grid) * 1.0e9),
            "median_initial_nm": float(np.median(stationary_initial) * 1.0e9),
            "median_final_nm": float(np.median(stationary_final) * 1.0e9),
            "median_boltzmann_nm": float(np.interp(0.5, target_cdf, h_grid) * 1.0e9),
        },
        {
            "case": "uniform_relaxation",
            "particles": n_particles,
            "dt_s": dt,
            "steps": n_steps,
            "simulated_time_s": dt * n_steps,
            "basin_lower_nm": lower * 1.0e9,
            "basin_upper_nm": upper * 1.0e9,
            "minimum_gap_nm": h_minimum * 1.0e9,
            "minimum_kBT": u_minimum,
            "ks_initial_vs_boltzmann": ks_distance(uniform_initial, h_grid, target_cdf),
            "ks_final_vs_boltzmann": ks_distance(uniform_final, h_grid, target_cdf),
            "mean_initial_nm": float(np.mean(uniform_initial) * 1.0e9),
            "mean_final_nm": float(np.mean(uniform_final) * 1.0e9),
            "mean_boltzmann_nm": float(np.trapezoid(h_grid * target_density, h_grid) * 1.0e9),
            "median_initial_nm": float(np.median(uniform_initial) * 1.0e9),
            "median_final_nm": float(np.median(uniform_final) * 1.0e9),
            "median_boltzmann_nm": float(np.interp(0.5, target_cdf, h_grid) * 1.0e9),
        },
    ]
    write_csv(OUT / "near_wall_boltzmann_validation.csv", rows)

    bins = np.linspace(lower, upper, 75)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), dpi=220, sharey=True)
    for ax, initial, final, title in (
        (axes[0], stationary_initial, stationary_final, "Stationary initialization"),
        (axes[1], uniform_initial, uniform_final, "Uniform initialization"),
    ):
        ax.hist(
            initial * 1.0e9,
            bins=bins * 1.0e9,
            density=True,
            histtype="step",
            lw=1.5,
            color="#94a3b8",
            label="initial",
        )
        ax.hist(
            final * 1.0e9,
            bins=bins * 1.0e9,
            density=True,
            histtype="stepfilled",
            alpha=0.34,
            color="#2563eb",
            label="after SDE",
        )
        ax.plot(h_grid * 1.0e9, target_density / 1.0e9, color="#111827", lw=1.8, label="Boltzmann")
        ax.axvline(h_minimum * 1.0e9, color="#dc2626", lw=1.0, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("gap h (nm)")
        ax.grid(alpha=0.22, lw=0.5)
    axes[0].set_ylabel("probability density (nm$^{-1}$)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Off-flow near-wall Langevin validation in the 50 mM unfavorable basin", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)

    report = [
        "# Near-Wall Boltzmann Validation",
        "",
        "The off-flow validation confines the one-dimensional gap coordinate to the 50 mM unfavorable secondary-minimum basin with reflecting boundaries.",
        "The target density is the conditional Boltzmann distribution proportional to exp(-U/kBT).",
        "",
    ]
    for row in rows:
        report.append(
            "- {case}: KS initial={ks_initial_vs_boltzmann:.4f}, KS final={ks_final_vs_boltzmann:.4f}, "
            "mean final={mean_final_nm:.2f} nm, target mean={mean_boltzmann_nm:.2f} nm.".format(**row)
        )
    report.extend(["", f"Figure: `{FIG_OUT.relative_to(ROOT)}`"])
    (OUT / "near_wall_boltzmann_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT / "near_wall_boltzmann_validation_report.md")


if __name__ == "__main__":
    main()
