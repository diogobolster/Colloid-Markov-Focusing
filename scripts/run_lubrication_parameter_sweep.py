#!/usr/bin/env python3
"""Screen near-wall mobility and DLVO parameter regimes for focusing."""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import (
    PhysicalParams,
    focusing_metrics,
    load_flow,
    rescale_flow,
    save_library,
    simulate_cell_transitions,
)


PHYSICAL_OUT = ROOT / "outputs" / "physical"
OUT = ROOT / "outputs" / "lubrication_sweep"
PARTICLE_COUNT = 8000
VELOCITY_M_PER_DAY = 4.0
CORE_BINS = np.array([15, 16])


PROFILES = (
    {
        "profile": "neutral_lubricated",
        "regime": "realistic control",
        "condition": "neutral",
        "surface_mode": "lubrication",
        "description": "No DLVO, wall-corrected Brownian motion.",
        "updates": {},
    },
    {
        "profile": "favorable_realistic",
        "regime": "realistic favorable",
        "condition": "favorable_lubricated",
        "surface_mode": "lubrication",
        "description": "Oppositely charged collector using the current reference chemistry.",
        "updates": {},
    },
    {
        "profile": "favorable_softened",
        "regime": "softened favorable",
        "condition": "favorable_lubricated",
        "surface_mode": "lubrication",
        "description": "Weak favorable attraction and low Hamaker constant to test releasable attraction.",
        "updates": {"zeta_collector_favorable": 20.0e-3, "hamaker": 0.5e-21},
    },
    {
        "profile": "unfavorable_realistic",
        "regime": "realistic unfavorable",
        "condition": "unfavorable",
        "surface_mode": "lubrication",
        "description": "Like-charged reference unfavorable surface with no attachment.",
        "updates": {},
    },
    {
        "profile": "unfavorable_weak_barrier",
        "regime": "secondary-minimum candidate",
        "condition": "unfavorable",
        "surface_mode": "lubrication",
        "description": "Moderate like-charge repulsion with the reference ionic strength.",
        "updates": {"zeta_collector_unfavorable": -20.0e-3},
    },
    {
        "profile": "unfavorable_compressed_double_layer",
        "regime": "secondary-minimum candidate",
        "condition": "unfavorable",
        "surface_mode": "lubrication",
        "description": "Compressed double layer to strengthen near-surface residence without attachment.",
        "updates": {"zeta_collector_unfavorable": -35.0e-3, "ionic_strength_molar": 50.0e-3},
    },
    {
        "profile": "mechanism_amplified_secondary_minimum",
        "regime": "mechanism-amplified",
        "condition": "unfavorable",
        "surface_mode": "lubrication",
        "description": "Non-environmental high Hamaker/longer interaction layer to exaggerate secondary-minimum sampling.",
        "updates": {
            "zeta_collector_unfavorable": -20.0e-3,
            "ionic_strength_molar": 20.0e-3,
            "hamaker": 1.0e-20,
            "near_surface": 500.0e-9,
        },
    },
    {
        "profile": "mechanism_amplified_long_range_favorable",
        "regime": "mechanism-amplified",
        "condition": "favorable_lubricated",
        "surface_mode": "lubrication",
        "description": "Non-environmental long-range favorable attraction; useful mainly as a pinning stress test.",
        "updates": {
            "zeta_collector_favorable": 120.0e-3,
            "ionic_strength_molar": 0.3e-3,
            "hamaker": 1.0e-20,
            "near_surface": 500.0e-9,
            "dlvo_velocity_cap": 5.0e-4,
        },
    },
)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_dicts(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile_params(base: PhysicalParams, updates: dict[str, float]) -> PhysicalParams:
    return replace(base, **updates)


def row_from_metrics(
    profile: dict[str, object],
    params: PhysicalParams,
    metrics: dict[str, float | str],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "profile": str(profile["profile"]),
        "regime": str(profile["regime"]),
        "condition": str(profile["condition"]),
        "surface_mode": str(profile["surface_mode"]),
        "description": str(profile["description"]),
        "particles": int(metrics["particles"]),
        "velocity_m_per_day": VELOCITY_M_PER_DAY,
        "ionic_strength_mM": params.ionic_strength_molar * 1.0e3,
        "debye_length_nm": params.debye_length * 1.0e9,
        "zeta_collector_favorable_mV": params.zeta_collector_favorable * 1.0e3,
        "zeta_collector_unfavorable_mV": params.zeta_collector_unfavorable * 1.0e3,
        "hamaker_J": params.hamaker,
        "near_surface_nm": params.near_surface * 1.0e9,
        "dlvo_velocity_cap_m_per_s": params.dlvo_velocity_cap,
    }
    for key in (
        "center_intercepted_any_count",
        "center_intercepted_mobile_count",
        "center_intercepted_censored_count",
        "center_intercepted_exit_fraction",
        "center_median_abs_theta_exit_downstream_rad",
        "center_fraction_theta_exit_within_15deg",
        "center_fraction_theta_exit_within_30deg",
        "center_median_abs_yout_center_m",
        "center_fraction_yout_within_center_band",
    ):
        row[key] = metrics[key]
    row["center_median_abs_yout_center_um"] = float(metrics["center_median_abs_yout_center_m"]) * 1.0e6
    return row


def draw_bar_group(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, float | int | str]],
    metric: str,
    rect: tuple[int, int, int, int],
    title: str,
    ylim: float | None = None,
) -> None:
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    values = np.array([float(row[metric]) for row in rows], dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    ymax = float(np.max(values)) if ylim is None else ylim
    ymax = max(ymax, 1.0e-12)
    draw.rectangle(rect, outline="#111111", width=2)
    draw.text((left + width / 2, top - 28), title, fill="#111111", font=font(24), anchor="mm")
    bar_w = width / max(len(rows), 1) * 0.65
    colors = {
        "realistic control": "#2563eb",
        "realistic favorable": "#dc2626",
        "softened favorable": "#d97706",
        "realistic unfavorable": "#059669",
        "secondary-minimum candidate": "#0f766e",
        "mechanism-amplified": "#7c3aed",
    }
    for i, row in enumerate(rows):
        x = left + (i + 0.5) * width / len(rows)
        value = float(row[metric]) if np.isfinite(float(row[metric])) else 0.0
        y = bottom - value / ymax * height
        color = colors.get(str(row["regime"]), "#444444")
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, bottom), fill=color)
        label = str(row["profile"]).replace("_", "\n")
        draw.text((x, bottom + 18), label, fill="#111111", font=font(12), anchor="ma")
    for frac in np.linspace(0.0, 1.0, 5):
        y = bottom - frac * height
        draw.line((left - 8, y, left, y), fill="#111111", width=2)
        draw.text((left - 14, y), f"{frac * ymax:.2g}", fill="#111111", font=font(13), anchor="rm")


def write_figure(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    img = Image.new("RGB", (1680, 1220), "white")
    draw = ImageDraw.Draw(img)
    draw.text((840, 44), "Lubrication-aware parameter screen, central-core injection", fill="#111111", font=font(34), anchor="mm")
    draw_bar_group(
        draw,
        rows,
        "center_intercepted_exit_fraction",
        (120, 130, 1560, 410),
        "Fraction of center interceptions that release",
        ylim=1.0,
    )
    draw_bar_group(
        draw,
        rows,
        "center_median_abs_theta_exit_downstream_rad",
        (120, 540, 1560, 820),
        "Median release angle from downstream stagnation (rad)",
        ylim=np.pi,
    )
    draw_bar_group(
        draw,
        rows,
        "center_fraction_theta_exit_within_30deg",
        (120, 950, 1560, 1150),
        "Fraction of releases within 30 degrees of downstream stagnation",
        ylim=1.0,
    )
    img.save(path)


def write_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        "# Lubrication-aware parameter screen",
        "",
        "All runs use central-core injection through inlet bins 15 and 16 at 4 m/day with the near-wall mobility correction enabled. Favorable profiles are nonattaching diagnostics; homogeneous favorable production runs would attach at contact. Unfavorable profiles remain nonattaching/reflecting.",
        "",
        "| profile | regime | I (mM) | zeta_c fav (mV) | zeta_c unfav (mV) | Hamaker (J) | center intercepted | center mobile | center censored | exit fraction | median abs theta_exit (rad) | frac within 30 deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['profile']} | "
            f"{row['regime']} | "
            f"{float(row['ionic_strength_mM']):.3g} | "
            f"{float(row['zeta_collector_favorable_mV']):.3g} | "
            f"{float(row['zeta_collector_unfavorable_mV']):.3g} | "
            f"{float(row['hamaker_J']):.3g} | "
            f"{int(row['center_intercepted_any_count'])} | "
            f"{int(row['center_intercepted_mobile_count'])} | "
            f"{int(row['center_intercepted_censored_count'])} | "
            f"{float(row['center_intercepted_exit_fraction']):.3f} | "
            f"{float(row['center_median_abs_theta_exit_downstream_rad']):.3f} | "
            f"{float(row['center_fraction_theta_exit_within_30deg']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the realistic favorable nonattaching diagnostic is expected to pin because attraction plus suppressed normal mobility keeps particles near contact. The more promising focusing window is a like-charged unfavorable chemistry with a secondary-minimum-like residence zone: it keeps attachment disabled while still perturbing near-grain paths. The mechanism-amplified profiles are not environmental claims; they are stress tests for whether the streamline-funneling mechanism can be made visible before returning to realistic chemistry.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = PhysicalParams()
    bins = np.linspace(0.0, base.cell_length, 33)
    y_min = bins[int(CORE_BINS[0])]
    y_max = bins[int(CORE_BINS[-1]) + 1]
    initial_y = np.random.default_rng(11001).uniform(y_min, y_max, size=PARTICLE_COUNT)
    rows: list[dict[str, float | int | str]] = []

    for i, profile in enumerate(PROFILES):
        params = profile_params(base, profile["updates"])  # type: ignore[arg-type]
        flow = load_flow(PHYSICAL_OUT / "flow_N192.npz", params)
        flow = rescale_flow(
            flow,
            VELOCITY_M_PER_DAY / 86400.0,
            max_time=max(60.0, 6.0 * params.cell_length / (VELOCITY_M_PER_DAY / 86400.0)),
        )
        print(f"Lubrication sweep: {profile['profile']}")
        lib = simulate_cell_transitions(
            flow,
            str(profile["condition"]),
            n_particles=PARTICLE_COUNT,
            seed=12001 + i,
            allow_attachment=False,
            initial_y=initial_y,
            surface_mode=str(profile["surface_mode"]),
        )
        save_library(OUT / f"trajectory_library_{profile['profile']}.npz", lib)
        rows.append(row_from_metrics(profile, params, focusing_metrics(lib)))

    write_dicts(OUT / "lubrication_parameter_sweep.csv", rows)
    write_report(OUT / "lubrication_parameter_report.md", rows)
    write_figure(OUT / "lubrication_parameter_sweep.png", rows)
    print("Done. Outputs written to", OUT)


if __name__ == "__main__":
    main()
