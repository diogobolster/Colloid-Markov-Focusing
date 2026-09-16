#!/usr/bin/env python3
"""Create reproducibility, parameter, outcome, and validation tables.

The manuscript now has enough moving pieces that manually transcribing
production settings is a liability. This script reads the generated simulation
artifacts and writes both CSV audit tables and compact LaTeX tables for the
focused manuscript.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

from colloid_tsm.physical import PhysicalParams, angular_distance, load_library  # noqa: E402

import run_compiled_parameter_suite as cps  # noqa: E402
import run_openfoam_full_condition_suite as full_suite  # noqa: E402


OUT = ROOT / "outputs" / "submission_readiness"
PAPER_TABLES = ROOT / "outputs" / "tables"
FULL_SUMMARY = ROOT / "outputs" / "openfoam_full_suite" / "openfoam_full_refinement_summary.csv"
FULL_TRANSITIONS = ROOT / "outputs" / "openfoam_full_suite" / "openfoam_full_transition_rows.csv"
RANDOM_REPLICATES = (
    ROOT
    / "outputs"
    / "random_grain_focusing_replicates"
    / "grain_local_focusing_replicate_summary.csv"
)
RANDOM_SCREEN_DIRS = (
    ROOT / "outputs" / "random_condition_grainlocal_production_selected",
    ROOT / "outputs" / "random_condition_grainlocal_production_seed20260508",
    ROOT / "outputs" / "random_condition_grainlocal_production_seed20260509",
)
GEOMETRY_JSON = ROOT / "outputs" / "random_porous_geometry" / "random_porous_geometry.json"
STAGE3_RANDOM_SUMMARY = ROOT / "outputs" / "stage3_particle_production" / "stage3_random_focusing_summary.csv"
STAGE3_RANDOM_VALIDATION = ROOT / "outputs" / "stage3_particle_production" / "random_stage3_validation_report.md"
STAGE3_SHADOW_METRICS = (
    ROOT
    / "outputs"
    / "stage3_transport_shadow"
    / "through_many_small_grains_n100k"
    / "transport_shadow_metrics.csv"
)
STAGE3_SHADOW_SUMMARY = (
    ROOT
    / "outputs"
    / "stage3_transport_shadow"
    / "through_many_small_grains_n100k"
    / "transport_shadow_condition_summary.csv"
)
CENTER_REPLICATES = ROOT / "outputs" / "center_corner_replicates" / "center_corner_replicate_summary.csv"
HNS_HC_SENSITIVITY = ROOT / "outputs" / "center_corner_hns_hc_sensitivity" / "center_corner_hns_hc_sensitivity.csv"
CENTER_WELL_SURVIVAL = ROOT / "outputs" / "center_well_survival" / "center_well_survival_summary.csv"

KB = 1.380649e-23


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(round(float(row[key])))
    except (KeyError, TypeError, ValueError):
        return default


def tex_escape(value: object) -> str:
    text = str(value)
    for src, dst in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(src, dst)
    return text


def fmt(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        text = str(value)
        if "$" in text or "\\" in text:
            return text
        return tex_escape(text)
    if not math.isfinite(x):
        return "--"
    if digits == 0:
        return f"{x:.0f}"
    if abs(x) >= 1.0e5 or (0.0 < abs(x) < 1.0e-3):
        return f"{x:.{digits}e}"
    if abs(x) >= 100.0:
        return f"{x:.0f}"
    if abs(x) >= 10.0:
        return f"{x:.1f}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def mean_sd(values: list[float], digits: int = 2) -> str:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return "--"
    if len(finite) == 1:
        return f"{finite[0]:.{digits}f}"
    return f"{mean(finite):.{digits}f} +/- {stdev(finite):.{digits}f}"


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    columns: list[tuple[str, str, int | None]],
    rows: list[dict[str, object]],
    *,
    tiny: bool = True,
    resize: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
    ]
    if tiny:
        lines.append(r"\scriptsize")
    if resize:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(header for _, header, _ in columns) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        entries = []
        for key, _header, digits in columns:
            value = row.get(key, "")
            entries.append(fmt(value, digits) if digits is not None else tex_escape(value))
        lines.append(" & ".join(entries) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_latex(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Current numerical validation status. Complete checks are production-ready for the present homogeneous-surface mechanism; limited-complete checks document the scope of finite perturbation suites rather than exhaustive numerical closure.}",
        r"\label{tab:validation-status}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}>{\raggedright\arraybackslash}p{0.55\linewidth}}",
        r"\hline",
        r"test & status & finding \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['test'])} & {tex_escape(row['status'])} & {tex_escape(row['finding'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def params_from_row(row: dict[str, str]) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        mean_velocity=full_suite.VELOCITY_M_PER_DAY / 86400.0,
        ionic_strength_molar=f(row, "ionic_strength_mM") * 1.0e-3,
        zeta_collector_unfavorable=f(row, "zeta_collector_unfavorable_mV") * 1.0e-3,
        hamaker=f(row, "hamaker_J"),
        diffusivity_multiplier=f(row, "diffusivity_multiplier", 1.0),
    )


def stable_seed(text: str) -> int:
    value = 0
    for byte in text.encode("utf-8"):
        value = (value * 131 + byte) % (2**32 - 1)
    return value or 1


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_release_uncertainty(
    lib,
    profile: str,
    *,
    n_bootstrap: int = 2000,
) -> dict[str, float | int | str]:
    """Particle-level bootstrap for release-angle metrics.

    The sampling unit is a particle with a completed release from the relevant
    center-collector event, not an individual secondary-minimum crossing.
    """

    if profile.startswith("neutral"):
        theta = lib.theta_exit
        collector = lib.collector_exit
        support = lib.exited & (~lib.attached) & (collector == 1) & np.isfinite(theta)
        event_type = "center near-surface release"
        event_type_short = "near surface"
    else:
        if lib.theta_well_exit is None or lib.collector_well_exit is None:
            support = np.zeros(lib.y_in.size, dtype=bool)
            theta = np.full(lib.y_in.size, np.nan)
        else:
            theta = lib.theta_well_exit
            support = (
                lib.exited
                & (~lib.attached)
                & (lib.collector_well_exit == 1)
                & np.isfinite(theta)
            )
        event_type = "center secondary-minimum release"
        event_type_short = "secondary min."

    theta_abs_deg = np.degrees(np.abs(angular_distance(theta[support], 0.0)))
    n_support = int(theta_abs_deg.size)
    out: dict[str, float | int | str] = {
        "release_event_type": event_type,
        "release_event_type_short": event_type_short,
        "release_support_particles": n_support,
        "F30_particle_bootstrap_low95": float("nan"),
        "F30_particle_bootstrap_high95": float("nan"),
        "median_angle_particle_bootstrap_low95_deg": float("nan"),
        "median_angle_particle_bootstrap_high95_deg": float("nan"),
        "bootstrap_replicates": 0,
    }
    if n_support == 0:
        return out

    rng = np.random.default_rng(stable_seed(profile))
    f30_values = np.empty(n_bootstrap)
    median_values = np.empty(n_bootstrap)
    for i_boot in range(n_bootstrap):
        sample = theta_abs_deg[rng.integers(0, n_support, size=n_support)]
        f30_values[i_boot] = np.mean(sample <= 30.0)
        median_values[i_boot] = np.median(sample)
    f30_lo, f30_hi = percentile_interval(f30_values)
    med_lo, med_hi = percentile_interval(median_values)
    out.update(
        {
            "F30_particle_bootstrap_low95": f30_lo,
            "F30_particle_bootstrap_high95": f30_hi,
            "median_angle_particle_bootstrap_low95_deg": med_lo,
            "median_angle_particle_bootstrap_high95_deg": med_hi,
            "bootstrap_replicates": n_bootstrap,
        }
    )
    return out


def dlvo_barrier_metrics(params: PhysicalParams, condition: str) -> dict[str, float | str]:
    if condition == "neutral":
        return {
            "barrier_kBT": float("nan"),
            "barrier_gap_nm": float("nan"),
            "barrier_uncrossable": "n/a",
        }
    h = np.logspace(-9, -6, 100_000)
    eps0 = 8.8541878128e-12
    eps = params.relative_permittivity * eps0
    kappa = 1.0 / params.debye_length
    edl_prefactor = 2.0 * math.pi * eps * params.particle_radius * params.zeta_particle * params.zeta_collector_unfavorable
    potential = edl_prefactor * np.exp(-kappa * h) - params.hamaker * params.particle_radius / (6.0 * h)
    max_id = int(np.argmax(potential))
    barrier = float(potential[max_id] / (KB * params.temperature))
    return {
        "barrier_kBT": barrier,
        "barrier_gap_nm": float(h[max_id] * 1.0e9) if barrier > 0.0 else float("nan"),
        "barrier_uncrossable": "yes" if barrier >= 20.0 else "no",
    }


def dlvo_parameter_rows(full_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in full_rows:
        params = params_from_row(row)
        barrier = dlvo_barrier_metrics(params, row["condition"])
        rows.append(
            {
                "profile": row["profile"],
                "label": row["label"],
                "condition": row["condition"],
                "ionic_strength_mM": f(row, "ionic_strength_mM"),
                "debye_length_nm": f(row, "debye_length_nm"),
                "zeta_particle_mV": params.zeta_particle * 1.0e3,
                "zeta_collector_mV": f(row, "zeta_collector_unfavorable_mV"),
                "hamaker_J": f(row, "hamaker_J"),
                "diffusivity_multiplier": f(row, "diffusivity_multiplier"),
                "secondary_minimum_kBT": f(row, "well_minimum_kbt"),
                "secondary_minimum_gap_nm": f(row, "well_minimum_gap_nm"),
                **barrier,
            }
        )
    return rows


def center_corner_rows(full_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in full_rows:
        profile = row["profile"]
        lib_path = (
            ROOT
            / "outputs"
            / "openfoam_full_suite"
            / f"trajectory_library_refinement_{profile}.npz"
        )
        center_well_total_events = float("nan")
        max_center_well_events_per_particle = float("nan")
        particles_with_repeated_center_well_events = float("nan")
        if lib_path.exists():
            lib = load_library(lib_path)
            center_well = (
                lib.center_well_interceptions
                if lib.center_well_interceptions is not None
                else np.zeros(lib.y_in.size, dtype=int)
            )
            center_well_total_events = int(np.sum(center_well))
            max_center_well_events_per_particle = int(np.max(center_well)) if center_well.size else 0
            particles_with_repeated_center_well_events = int(np.sum(center_well > 1))
            bootstrap = bootstrap_release_uncertainty(lib, profile)
        else:
            bootstrap = {
                "release_event_type": "unknown",
                "release_event_type_short": "unknown",
                "release_support_particles": 0,
                "F30_particle_bootstrap_low95": float("nan"),
                "F30_particle_bootstrap_high95": float("nan"),
                "median_angle_particle_bootstrap_low95_deg": float("nan"),
                "median_angle_particle_bootstrap_high95_deg": float("nan"),
                "bootstrap_replicates": 0,
            }
        unique_entries = i(row, "center_well_intercepted_count")
        if profile.startswith("neutral"):
            reported_unique_entries = i(row, "center_intercepted_count")
            reported_releases = i(row, "center_release_count")
            reported_censored = i(row, "center_censored_count")
            reported_median_angle = f(row, "center_release_median_theta_deg")
            reported_f30 = f(row, "center_release_theta30_fraction")
            reported_residence = f(row, "center_near_time_median_s")
            reported_residence_p05 = float("nan")
            reported_residence_p95 = float("nan")
        else:
            reported_unique_entries = unique_entries
            reported_releases = i(row, "center_well_release_count")
            reported_censored = i(row, "center_well_censored_count")
            reported_median_angle = f(row, "center_well_release_median_theta_deg")
            reported_f30 = f(row, "center_well_release_theta30_fraction")
            reported_residence = f(row, "center_well_time_median_s")
            reported_residence_p05 = f(row, "center_well_time_p05_s")
            reported_residence_p95 = f(row, "center_well_time_p95_s")
        rows.append(
            {
                "profile": profile,
                "label": row["label"],
                "particles": i(row, "particles"),
                "independent_particle_seeds": 1,
                "injection": "uniform central-core y",
                "horizon_s": f(row, "max_time_s"),
                "release_event_type": bootstrap["release_event_type"],
                "release_event_type_short": bootstrap["release_event_type_short"],
                "release_support_particles": bootstrap["release_support_particles"],
                "center_well_unique_particles": unique_entries,
                "center_well_total_entry_events": center_well_total_events,
                "particles_with_repeated_center_well_events": particles_with_repeated_center_well_events,
                "max_center_well_events_per_particle": max_center_well_events_per_particle,
                "reported_event_unique_particles": reported_unique_entries,
                "reported_event_releases_unique": reported_releases,
                "reported_event_censored_unique": reported_censored,
                "median_residence_s": reported_residence,
                "residence_p05_s": reported_residence_p05,
                "residence_p95_s": reported_residence_p95,
                "median_release_angle_deg": reported_median_angle,
                "median_angle_particle_bootstrap_low95_deg": bootstrap["median_angle_particle_bootstrap_low95_deg"],
                "median_angle_particle_bootstrap_high95_deg": bootstrap["median_angle_particle_bootstrap_high95_deg"],
                "F30": reported_f30,
                "F30_binomial_halfwidth_95": f(row, "center_well_release_theta30_halfwidth_95"),
                "F30_particle_bootstrap_low95": bootstrap["F30_particle_bootstrap_low95"],
                "F30_particle_bootstrap_high95": bootstrap["F30_particle_bootstrap_high95"],
                "bootstrap_replicates": bootstrap["bootstrap_replicates"],
                "final_status": "complete" if reported_censored == 0 and f(row, "censored_fraction") == 0.0 else "finite-horizon unresolved residence",
                "count_basis": "unique release-support particles plus cumulative entry-events",
            }
        )
    return rows


def random_screen_rows() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for directory in RANDOM_SCREEN_DIRS:
        path = directory / "random_condition_screen_summary.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            out.setdefault(row["profile"], []).append(row)
    return out


def random_packing_rows() -> list[dict[str, object]]:
    replicate_rows = read_csv(RANDOM_REPLICATES)
    screen = random_screen_rows()
    rows: list[dict[str, object]] = []
    for row in replicate_rows:
        profile = row["profile"]
        screen_group = screen.get(profile, [])
        particles = [f(item, "particles") for item in screen_group]
        exited = [f(item, "exited") for item in screen_group]
        attached = [f(item, "attached") for item in screen_group]
        censored = [f(item, "censored") for item in screen_group]
        rows.append(
            {
                "profile": profile,
                "independent_particle_seeds": i(row, "replicates"),
                "particles_per_seed": int(round(mean(particles))) if particles else 5000,
                "exited_per_seed": mean_sd(exited, 0),
                "attached_per_seed": mean_sd(attached, 0),
                "censored_per_seed": mean_sd(censored, 0),
                "near_surface_releases": mean_sd([f(row, "released_from_near_zone_mean")], 0),
                "near_surface_releases_sd": f(row, "released_from_near_zone_sd"),
                "median_rear_angle_deg_mean": f(row, "median_abs_release_angle_from_rear_deg_mean"),
                "median_rear_angle_deg_sd": f(row, "median_abs_release_angle_from_rear_deg_sd"),
                "F30_rear_mean": f(row, "fraction_released_within_30deg_mean"),
                "F30_rear_sd": f(row, "fraction_released_within_30deg_sd"),
                "effective_angle_bins_mean": f(row, "grain_local_effective_angle_bins_mean"),
                "effective_angle_bins_sd": f(row, "grain_local_effective_angle_bins_sd"),
                "count_basis": "near-surface release events, grain-local rear-point conditioned",
            }
        )
    return rows


def random_stage3_rows() -> list[dict[str, object]]:
    if not STAGE3_RANDOM_SUMMARY.exists():
        return []
    return [
        {
            "profile": row["profile"],
            "label": row["label"],
            "geometries": i(row, "geometries"),
            "released_total": i(row, "released_total"),
            "intercepted_total": i(row, "intercepted_total"),
            "censored_total": i(row, "censored_total"),
            "F30_mean": f(row, "F30_mean"),
            "F30_sd": f(row, "F30_sd"),
            "F30_min": f(row, "F30_min"),
            "F30_max": f(row, "F30_max"),
            "median_angle_mean_deg": f(row, "median_angle_mean_deg"),
            "median_angle_sd_deg": f(row, "median_angle_sd_deg"),
            "count_basis": "five independent screened periodic geometries; 10,000 injected particles per condition per geometry",
        }
        for row in read_csv(STAGE3_RANDOM_SUMMARY)
    ]


def center_replicate_rows() -> list[dict[str, object]]:
    if not CENTER_REPLICATES.exists():
        return []
    rows: list[dict[str, object]] = []
    for row in read_csv(CENTER_REPLICATES):
        rows.append(
            {
                "label": row["label"],
                "event_name": row["event_name"],
                "replicates": i(row, "replicates"),
                "particles_per_replicate": i(row, "particles_per_replicate"),
                "F30_mean": f(row, "F30_mean"),
                "F30_sd": f(row, "F30_sd"),
                "median_angle_mean": f(row, "median_release_angle_deg_mean"),
                "median_angle_sd": f(row, "median_release_angle_deg_sd"),
                "median_residence_mean": f(row, "median_residence_s_mean"),
                "median_residence_sd": f(row, "median_residence_s_sd"),
                "unresolved_fraction_mean": f(row, "unresolved_fraction_mean"),
                "unresolved_fraction_sd": f(row, "unresolved_fraction_sd"),
            }
        )
    return rows


def hns_hc_rows() -> list[dict[str, object]]:
    if not HNS_HC_SENSITIVITY.exists():
        return []
    rows: list[dict[str, object]] = []
    for row in read_csv(HNS_HC_SENSITIVITY):
        rows.append(
            {
                "label": row["label"],
                "variant_label": row["variant_label"].replace("$", ""),
                "near_surface_nm": f(row, "near_surface_nm"),
                "contact_gap_nm": f(row, "contact_gap_nm"),
                "release_count": i(row, "release_count"),
                "unresolved_count": i(row, "unresolved_count"),
                "F30": f(row, "F30"),
                "median_angle": f(row, "median_release_angle_deg"),
                "median_residence": f(row, "median_residence_s"),
            }
        )
    return rows


def reproducibility_rows() -> list[dict[str, object]]:
    geometry = json.loads(GEOMETRY_JSON.read_text(encoding="utf-8"))
    full_rows = read_csv(FULL_SUMMARY)
    first = full_rows[0]
    params = params_from_row(first)
    random_config = geometry["config"]
    random_derived = geometry["derived"]
    random_domain = geometry["domain"]
    return [
        {
            "category": "Center/corner geometry",
            "item": "domain and collectors",
            "value": "L=400 um; R=100 um; one center collector plus four quarter-corner periodic images",
        },
        {
            "category": "Center/corner geometry",
            "item": "porosity",
            "value": "0.607 analytic/OpenFOAM mesh porosity 0.607484",
        },
        {
            "category": "Center/corner injection",
            "item": "focusing ensemble",
            "value": "30,000 particles; uniform y in [L/2-12.5 um, L/2+12.5 um]; x=0",
        },
        {
            "category": "Center/corner injection",
            "item": "matrix ensemble",
            "value": "stratified uniform y over open inlet subsets for 100x100 transition matrices",
        },
        {
            "category": "Flow solver",
            "item": "OpenFOAM setup",
            "value": "laminar simpleFoam, meanVelocityForce, cyclicAMI x/y, no-slip grains, symmetry/empty thin slab",
        },
        {
            "category": "Flow mesh",
            "item": "center/corner mesh",
            "value": "snappyHexMesh on 96x96x4 base mesh, surface refinement level 2; raster convergence N=256--768",
        },
        {
            "category": "Flow raster",
            "item": "production interpolation",
            "value": "OpenFOAM field mapped to N=512 periodic raster; bilinear periodic interpolation at particle centers",
        },
        {
            "category": "Flow raster",
            "item": "solid/excluded regions",
            "value": "particle centers are excluded from h<hc; proposed crossings are projected to hc before the next velocity evaluation",
        },
        {
            "category": "Physical parameters",
            "item": "particle and fluid",
            "value": f"a={params.particle_radius * 1e6:.2f} um; mu={params.viscosity:.1e} Pa s; T={params.temperature:.2f} K; D0={params.diffusivity:.3e} m2/s",
        },
        {
            "category": "Particle tracking",
            "item": "timestep",
            "value": "outer dt=1 ms; adaptive near-wall subcycling for h<=75 nm; base 25 substeps; min substep=1 us",
        },
        {
            "category": "Particle tracking",
            "item": "adaptive criteria",
            "value": "substep is min(base, 0.25*normal_step/ inward speed, 0.25*normal_step^2/(2 D_perp)); normal_step=3 nm",
        },
        {
            "category": "Mobility",
            "item": "normal",
            "value": "Mperp/M0=clip(h/(h+a), 1e-4, 1) inside 2 um wall-mobility cutoff",
        },
        {
            "category": "Mobility",
            "item": "tangential",
            "value": "Mpar/M0=clip(min(Brenner far-field series, 1/(1+0.9588 ln(a/h))), 0.05, 1)",
        },
        {
            "category": "Thermal drift",
            "item": "divergence of D",
            "value": "implemented as D0*d(Mperp/M0)/dh in the outward normal direction",
        },
        {
            "category": "Boundary algorithm",
            "item": "unfavorable surfaces",
            "value": "no attachment; exclusion crossing projected to hc=1 nm; uphill inward DLVO moves rejected by Metropolis probability exp(-Delta U/kBT)",
        },
        {
            "category": "Event definition",
            "item": "near surface and secondary minimum",
            "value": "near event h<=200 nm; well entry U<=Umin+kBT; well residence persists until U>-kBT",
        },
        {
            "category": "Random packing geometry",
            "item": "domain and porosity",
            "value": f"{random_domain['length_x'] * 1e3:.3f} x {random_domain['length_y'] * 1e3:.3f} mm; porosity={random_derived['actual_porosity']:.3f}; {random_config['n_grains']} grains",
        },
        {
            "category": "Random packing geometry",
            "item": "radii and packing",
            "value": f"seed={random_config['seed']}; radii {random_derived['min_grain_radius'] * 1e6:.1f}--{random_derived['max_grain_radius'] * 1e6:.1f} um; minimum gap {random_derived['minimum_realized_surface_gap'] * 1e6:.1f} um",
        },
        {
            "category": "Random packing flow",
            "item": "OpenFOAM mesh",
            "value": "gmsh 180x120x1 base, 2D empty-patch formulation, near-wall/far mesh sizes 2/5 um",
        },
        {
            "category": "Random packing particles",
            "item": "production ensemble",
            "value": "5000 particles per condition per seed; checkpointed 500-particle chunks; grain-local rear stagnation points from OpenFOAM shell",
        },
    ]


def validation_rows() -> list[dict[str, object]]:
    convergence = read_csv(ROOT / "outputs" / "openfoam_convergence" / "openfoam_convergence_summary.csv")
    grid_deltas = [abs(f(row, "delta_f30_vs_baseline")) for row in convergence if row["suite"] == "grid"]
    time_deltas = [abs(f(row, "delta_f30_vs_baseline")) for row in convergence if row["suite"] == "time_step"]
    normal_step_deltas = [abs(f(row, "delta_f30_vs_baseline")) for row in convergence if row["suite"] == "normal_step"]
    boltz_path = ROOT / "outputs" / "near_wall_boltzmann_validation" / "near_wall_boltzmann_validation.csv"
    if boltz_path.exists():
        boltz = read_csv(boltz_path)
        stationary = next((row for row in boltz if row["case"] == "stationary_initialized"), boltz[0])
        boltz_status = "complete"
        boltz_evidence = "outputs/near_wall_boltzmann_validation/near_wall_boltzmann_validation_report.md"
        boltz_finding = (
            "off-flow 50 mM unfavorable basin preserves conditional Boltzmann density; "
            f"KS final={f(stationary, 'ks_final_vs_boltzmann'):.3f}, "
            f"mean gap={f(stationary, 'mean_final_nm'):.2f} nm vs target {f(stationary, 'mean_boltzmann_nm'):.2f} nm"
        )
    else:
        boltz_status = "missing"
        boltz_evidence = "not yet generated"
        boltz_finding = "needed to validate variable-mobility Langevin statistics independently of flow"

    next_path = ROOT / "outputs" / "next_interception_kernel" / "release_to_next_interception_summary.csv"
    if next_path.exists():
        next_rows = read_csv(next_path)
        neutral = next(row for row in next_rows if row["profile"] == "neutral_resolved" and row["release_class"] == "all")
        unf100 = next(row for row in next_rows if row["profile"] == "unfavorable_100mM_z70" and row["release_class"] == "all")
        next_status = "limited-complete"
        next_evidence = "outputs/next_interception_kernel/release_to_next_interception_report.md"
        next_finding = (
            "release-to-next-near-surface kernel generated in random packing; "
            f"release F30/next-event F30 increase from {f(neutral, 'release_F30'):.3f}/{f(neutral, 'next_F30_rear'):.3f} "
            f"without DLVO to {f(unf100, 'release_F30'):.3f}/{f(unf100, 'next_F30_rear'):.3f} at 100 mM unfavorable"
        )
    else:
        next_status = "missing"
        next_evidence = "not yet generated"
        next_finding = "need next-interception or hypothetical attractive-patch encounter analysis to demonstrate retention consequence"
    if CENTER_WELL_SURVIVAL.exists():
        survival = read_csv(CENTER_WELL_SURVIVAL)
        unresolved = [row for row in survival if i(row, "unresolved_at_horizon") > 0]
        censor_status = "complete"
        censor_evidence = "outputs/center_well_survival/center_well_survival_report.md"
        censor_finding = (
            "survival/completion curves added; clean 50 mM and 100 mM z70 complete, "
            f"{len(unresolved)} stronger cases retain finite-horizon unresolved residence"
        )
    else:
        censor_status = "partial-complete"
        censor_evidence = "outputs/validation/censoring_sensitivity_report.md and openfoam_full_refinement_summary.csv"
        censor_finding = "clean 50 mM cases have zero center-well unresolved particles; stronger cases require survival-curve reporting"

    if CENTER_REPLICATES.exists():
        replicate_rows = read_csv(CENTER_REPLICATES)
        min_reps = min((i(row, "replicates") for row in replicate_rows), default=0)
        f50 = next((row for row in replicate_rows if row["profile"] == "unfavorable_50mM_z70"), None)
        replicate_status = "complete" if min_reps >= 3 else "partial-complete"
        replicate_evidence = "outputs/center_corner_replicates/center_corner_replicate_report.md"
        replicate_finding = (
            f"{len(replicate_rows)} key cases run with at least {min_reps} seeds; "
            + (
                f"50 mM z70 F30={f(f50, 'F30_mean'):.3f}+/-{f(f50, 'F30_sd'):.3f}"
                if f50 is not None
                else "seed-level uncertainty table generated"
            )
        )
    else:
        replicate_status = "missing"
        replicate_evidence = "not yet generated"
        replicate_finding = "current center/corner production suite uses one particle seed per condition; replicate seeds should be run before submission"

    if HNS_HC_SENSITIVITY.exists():
        sensitivity_rows = read_csv(HNS_HC_SENSITIVITY)
        profiles = sorted(set(row["profile"] for row in sensitivity_rows))
        sensitivity_status = "limited-complete"
        sensitivity_evidence = "outputs/center_corner_hns_hc_sensitivity/center_corner_hns_hc_sensitivity_report.md"
        sensitivity_finding = f"h_ns=100,200,400 nm and h_c=0.5,1,2 nm variants tested for {len(profiles)} key cases"
    else:
        sensitivity_status = "missing"
        sensitivity_evidence = "not yet generated"
        sensitivity_finding = "near-surface cutoff 200 nm and contact gap 1 nm should be swept explicitly"

    if STAGE3_RANDOM_SUMMARY.exists() and STAGE3_RANDOM_VALIDATION.exists():
        stage3_rows = read_csv(STAGE3_RANDOM_SUMMARY)
        neutral = next((row for row in stage3_rows if row["profile"] == "neutral_resolved"), None)
        unf50 = next((row for row in stage3_rows if row["profile"] == "unfavorable_50mM_z70"), None)
        stage3_status = "complete"
        stage3_evidence = "outputs/stage3_particle_production/stage3_random_production_aggregate_report.md"
        stage3_finding = (
            "five screened random geometries run with 10,000 particles per condition; "
            f"F30 increases from {f(neutral, 'F30_mean'):.3f}+/-{f(neutral, 'F30_sd'):.3f} without DLVO "
            f"to {f(unf50, 'F30_mean'):.3f}+/-{f(unf50, 'F30_sd'):.3f} at 50 mM unfavorable"
        )
    else:
        stage3_status = "missing"
        stage3_evidence = "not yet generated"
        stage3_finding = "five-geometry random-packing production ensemble has not been aggregated"

    if STAGE3_SHADOW_METRICS.exists() and STAGE3_SHADOW_SUMMARY.exists():
        shadow_rows = read_csv(STAGE3_SHADOW_METRICS)
        f50_shadow = next(
            (
                row
                for row in shadow_rows
                if row["profile"] == "unfavorable_50mM_z70" and abs(f(row, "epsilon") - 0.005) < 1.0e-12
            ),
            None,
        )
        d100_shadow = next(
            (
                row
                for row in shadow_rows
                if row["profile"] == "unfavorable_50mM_z70_100xD" and abs(f(row, "epsilon") - 0.005) < 1.0e-12
            ),
            None,
        )
        shadow_status = "complete"
        shadow_evidence = "outputs/stage3_transport_shadow/through_many_small_grains_n100k/transport_shadow_diagnostic_report.md"
        shadow_finding = (
            "100,000-particle finite-time accessibility diagnostic complete; "
            f"pore S0.005={f(f50_shadow, 'pore_shadow_fraction'):.4f} for 50 mM unfavorable and "
            f"{f(d100_shadow, 'pore_shadow_fraction'):.4f} for the 100D control"
        )
    else:
        shadow_status = "missing"
        shadow_evidence = "not yet generated"
        shadow_finding = "100,000-particle pore/surface shadow diagnostic has not been completed"

    return [
        {
            "test": "production regression checks",
            "status": "complete",
            "evidence": "outputs/regression/production_validation_report.md",
            "finding": "50 mM focusing metrics, transition rows, timestep audit, convergence, and full-suite row counts pass configured tolerances",
        },
        {
            "test": "adaptive timestep audit",
            "status": "complete",
            "evidence": "outputs/timestep_audit/timestep_audit_report.md",
            "finding": "center-well releases use median 6.11e5 adaptive substeps; guard hits zero; maximum Brownian normal std 1.5 nm",
        },
        {
            "test": "raster resolution convergence",
            "status": "partial-complete",
            "evidence": "outputs/openfoam_convergence/openfoam_convergence_summary.csv",
            "finding": f"N=256--768 tested; max |Delta F30| vs N=512 baseline = {max(grid_deltas):.3f}",
        },
        {
            "test": "outer timestep convergence",
            "status": "partial-complete",
            "evidence": "outputs/openfoam_convergence/openfoam_convergence_summary.csv",
            "finding": f"dt=1,2,4 ms tested; max |Delta F30| vs 2 ms baseline = {max(time_deltas):.3f}; production uses 1 ms",
        },
        {
            "test": "normal step convergence",
            "status": "partial-complete",
            "evidence": "outputs/openfoam_convergence/openfoam_convergence_summary.csv",
            "finding": f"normal step 1.5,3,6 nm tested; max |Delta F30| vs 3 nm baseline = {max(normal_step_deltas):.3f}",
        },
        {
            "test": "finite-horizon censoring",
            "status": censor_status,
            "evidence": censor_evidence,
            "finding": censor_finding,
        },
        {
            "test": "random-packing seed-to-seed uncertainty",
            "status": "complete for random packing",
            "evidence": "outputs/random_grain_focusing_replicates/grain_local_focusing_replicate_summary.csv",
            "finding": "three seeds for main random-packing cases; two seeds for 75 mM bridge",
        },
        {
            "test": "random-packing five-geometry production ensemble",
            "status": stage3_status,
            "evidence": stage3_evidence,
            "finding": stage3_finding,
        },
        {
            "test": "100k finite-time transport-shadow diagnostic",
            "status": shadow_status,
            "evidence": shadow_evidence,
            "finding": shadow_finding,
        },
        {
            "test": "center/corner seed-to-seed uncertainty",
            "status": replicate_status,
            "evidence": replicate_evidence,
            "finding": replicate_finding,
        },
        {
            "test": "off-flow Boltzmann wall equilibrium",
            "status": boltz_status,
            "evidence": boltz_evidence,
            "finding": boltz_finding,
        },
        {
            "test": "h_ns and h_c sensitivity",
            "status": sensitivity_status,
            "evidence": sensitivity_evidence,
            "finding": sensitivity_finding,
        },
        {
            "test": "downstream consequence of focusing",
            "status": next_status,
            "evidence": next_evidence,
            "finding": next_finding,
        },
    ]


def write_reproducibility_report(
    dlvo_rows: list[dict[str, object]],
    center_rows: list[dict[str, object]],
    random_rows: list[dict[str, object]],
    stage3_rows: list[dict[str, object]],
    validation: list[dict[str, object]],
) -> None:
    missing = [row for row in validation if str(row["status"]).startswith("missing")]
    completed = [row for row in validation if not str(row["status"]).startswith("missing")]
    high_d = next(row for row in center_rows if row["profile"] == "unfavorable_50mM_z70_100xD")
    lines = [
        "# Submission Readiness and Reproducibility Package",
        "",
        "This report is generated from the current simulation artifacts by `scripts/create_submission_readiness_tables.py`.",
        "",
        "## Generated tables",
        "",
        "- `dlvo_parameter_table.csv`: chemistry, Debye length, secondary minima, and repulsive barriers.",
        "- `center_corner_outcome_table.csv`: per-condition center/corner particle outcomes, horizons, unique counts, and cumulative well-entry events.",
        "- `random_packing_outcome_table.csv`: seed-to-seed grain-local focusing outcomes in the random packing.",
        "- `random_stage3_focusing_summary.csv`: five-geometry random-packing production focusing ensemble.",
        "- `center_corner_replicate_table.csv`: seed-to-seed uncertainty for the key center/corner cases.",
        "- `hns_hc_sensitivity_table.csv`: sensitivity of release-angle focusing to the near-surface cutoff and contact gap.",
        "- `reproducibility_parameters.csv`: geometry, OpenFOAM, raster, SDE, mobility, and event-definition settings.",
        "- `validation_status_table.csv`: completed, limited-complete, and remaining validation/archival checks.",
        "",
        "## Count-basis clarification",
        "",
        (
            "The center/corner focusing summaries count unique particles that enter the reported release-support event at least once "
            "(near-surface release for no-DLVO controls, secondary-minimum release for unfavorable cases). "
            "The compiled kernel also stores cumulative center-well entry events per particle; these are reported separately for unfavorable cases. "
            f"For the high-diffusion unfavorable case, {high_d['center_well_unique_particles']} unique particles entered the center well and "
            f"{high_d['center_well_total_entry_events']} cumulative center-well entries were recorded."
        ),
        "",
        "## Validation status",
        "",
        f"Completed or partially completed checks: {len(completed)}.",
        f"Missing checks before submission: {len(missing)}.",
        "",
    ]
    for row in missing:
        lines.append(f"- {row['test']}: {row['finding']}")
    lines.append("")
    if stage3_rows:
        unf50 = next((row for row in stage3_rows if row["profile"] == "unfavorable_50mM_z70"), None)
        neutral = next((row for row in stage3_rows if row["profile"] == "neutral_resolved"), None)
        lines.extend(
            [
                "## Random stage-3 ensemble",
                "",
                (
                    "The five-geometry random-packing production ensemble uses 10,000 injected particles per condition per geometry. "
                    f"The no-DLVO reference gives F30={f(neutral, 'F30_mean'):.3f}+/-{f(neutral, 'F30_sd'):.3f}, "
                    f"while the 50 mM unfavorable case gives F30={f(unf50, 'F30_mean'):.3f}+/-{f(unf50, 'F30_sd'):.3f}."
                ),
                "",
            ]
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submission_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    full_rows = read_csv(FULL_SUMMARY)

    dlvo_rows = dlvo_parameter_rows(full_rows)
    center_rows = center_corner_rows(full_rows)
    random_rows = random_packing_rows()
    stage3_rows = random_stage3_rows()
    replicate_rows = center_replicate_rows()
    sensitivity_rows = hns_hc_rows()
    repro_rows = reproducibility_rows()
    validation = validation_rows()

    write_csv(OUT / "dlvo_parameter_table.csv", dlvo_rows)
    write_csv(OUT / "center_corner_outcome_table.csv", center_rows)
    write_csv(OUT / "random_packing_outcome_table.csv", random_rows)
    if stage3_rows:
        write_csv(OUT / "random_stage3_focusing_summary.csv", stage3_rows)
    if replicate_rows:
        write_csv(OUT / "center_corner_replicate_table.csv", replicate_rows)
    if sensitivity_rows:
        write_csv(OUT / "hns_hc_sensitivity_table.csv", sensitivity_rows)
    write_csv(OUT / "reproducibility_parameters.csv", repro_rows)
    write_csv(OUT / "validation_status_table.csv", validation)

    write_latex_table(
        PAPER_TABLES / "dlvo_parameter_table.tex",
        "DLVO parameters and computed barrier structure for the center/corner production suite. Barrier heights greater than $20\\,k_BT$ are marked effectively uncrossable for the present Brownian tracking horizons.",
        "tab:dlvo-parameters",
        [
            ("label", "case", None),
            ("ionic_strength_mM", "I (mM)", 1),
            ("debye_length_nm", "$\\kappa^{-1}$ (nm)", 2),
            ("zeta_particle_mV", "$\\zeta_p$ (mV)", 1),
            ("zeta_collector_mV", "$\\zeta_c$ (mV)", 1),
            ("hamaker_J", "$A_H$ (J)", 2),
            ("secondary_minimum_kBT", "$U_{min}$ ($k_BT$)", 2),
            ("secondary_minimum_gap_nm", "$h_{min}$ (nm)", 2),
            ("barrier_kBT", "barrier ($k_BT$)", 1),
            ("barrier_gap_nm", "$h_b$ (nm)", 2),
            ("barrier_uncrossable", "uncrossable", None),
        ],
        dlvo_rows,
    )
    write_latex_table(
        PAPER_TABLES / "center_corner_outcome_table.tex",
        "Center/corner-cell outcome audit. The reported event is center-grain near-surface release for no-DLVO controls and center-grain secondary-minimum release for unfavorable cases. Uncertainty intervals are particle-level bootstrap intervals over unique completed-release particles; cumulative well-entry events are reported separately to expose repeated visits by the same particle.",
        "tab:center-corner-outcomes",
        [
            ("label", "case", None),
            ("release_event_type_short", "event", None),
            ("particles", "$N_p$", 0),
            ("independent_particle_seeds", "seeds", 0),
            ("horizon_s", "$T_{max}$ (s)", 0),
            ("reported_event_unique_particles", "event unique", 0),
            ("center_well_total_entry_events", "well events", 0),
            ("reported_event_releases_unique", "releases", 0),
            ("reported_event_censored_unique", "unresolved", 0),
            ("median_residence_s", "$\\tilde{\\tau}$ (s)", 2),
            ("median_release_angle_deg", "$\\tilde{|\\theta|}$ (deg)", 2),
            ("F30", "$F_{30}$", 3),
            ("F30_particle_bootstrap_low95", "$F_{30}$ 2.5\\%", 3),
            ("F30_particle_bootstrap_high95", "$F_{30}$ 97.5\\%", 3),
            ("final_status", "status", None),
        ],
        center_rows,
    )
    write_latex_table(
        PAPER_TABLES / "random_packing_outcome_table.tex",
        "Random-packing grain-local focusing outcome audit. Values are means across independent particle seeds except where noted.",
        "tab:random-packing-outcomes",
        [
            ("profile", "case", None),
            ("independent_particle_seeds", "seeds", 0),
            ("particles_per_seed", "$N_p$/seed", 0),
            ("exited_per_seed", "exited/seed", None),
            ("attached_per_seed", "attached/seed", None),
            ("censored_per_seed", "unresolved/seed", None),
            ("near_surface_releases", "near releases", None),
            ("median_rear_angle_deg_mean", "$\\tilde{|\\theta_r|}$ (deg)", 2),
            ("median_rear_angle_deg_sd", "sd", 2),
            ("F30_rear_mean", "$F^{rand}_{30}$", 3),
            ("F30_rear_sd", "sd", 3),
        ],
        random_rows,
    )
    if stage3_rows:
        write_latex_table(
            PAPER_TABLES / "random_stage3_focusing_summary.tex",
            "Production random-packing grain-local release focusing across five screened periodic geometries. Values are mean and sample standard deviation across geometries.",
            "tab:random-stage3-focusing",
            [
                ("label", "condition", None),
                ("geometries", "geometries", 0),
                ("released_total", "released events", 0),
                ("intercepted_total", "intercepted particles", 0),
                ("F30_mean", "$F_{30}$ mean", 3),
                ("F30_sd", "sd", 3),
                ("median_angle_mean_deg", "median angle mean (deg)", 1),
                ("median_angle_sd_deg", "sd", 1),
            ],
            stage3_rows,
            tiny=True,
            resize=True,
        )
    if replicate_rows:
        write_latex_table(
            PAPER_TABLES / "center_corner_replicate_table.tex",
            "Independent particle-seed uncertainty for key center/corner-cell cases. Values are means and standard deviations across replicate particle seeds.",
            "tab:center-corner-replicates",
            [
                ("label", "case", None),
                ("replicates", "seeds", 0),
                ("particles_per_replicate", "$N_p$/seed", 0),
                ("F30_mean", "$F_{30}$ mean", 3),
                ("F30_sd", "sd", 3),
                ("median_angle_mean", "$\\tilde{|\\theta|}$ mean (deg)", 2),
                ("median_angle_sd", "sd", 2),
                ("median_residence_mean", "$\\tilde{\\tau}$ mean (s)", 2),
                ("median_residence_sd", "sd", 2),
                ("unresolved_fraction_mean", "unresolved mean", 3),
                ("unresolved_fraction_sd", "sd", 3),
            ],
            replicate_rows,
        )
    if sensitivity_rows:
        write_latex_table(
            PAPER_TABLES / "hns_hc_sensitivity_table.tex",
            "Sensitivity of center/corner-cell focusing metrics to the near-surface event cutoff $h_{\\rm ns}$ and contact gap $h_c$.",
            "tab:hns-hc-sensitivity",
            [
                ("label", "case", None),
                ("variant_label", "variant", None),
                ("near_surface_nm", "$h_{\\rm ns}$ (nm)", 0),
                ("contact_gap_nm", "$h_c$ (nm)", 1),
                ("release_count", "releases", 0),
                ("unresolved_count", "unresolved", 0),
                ("F30", "$F_{30}$", 3),
                ("median_angle", "$\\tilde{|\\theta|}$ (deg)", 2),
                ("median_residence", "$\\tilde{\\tau}$ (s)", 2),
            ],
            sensitivity_rows,
        )
    write_validation_latex(PAPER_TABLES / "validation_status_table.tex", validation)

    write_reproducibility_report(dlvo_rows, center_rows, random_rows, stage3_rows, validation)
    print(OUT / "submission_readiness_report.md")


if __name__ == "__main__":
    main()
