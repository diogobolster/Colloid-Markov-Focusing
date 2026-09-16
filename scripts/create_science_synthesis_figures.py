#!/usr/bin/env python3
"""Create science-synthesis figures for the focused colloid manuscript.

These figures do not rerun particle tracking. They consolidate the production
CSV/NPZ outputs into higher-level diagnostics that address interpretation:
finite-time shadowing, release-to-next-interception memory, geometry dependence,
event-state bookkeeping, and the focusing regime window.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle


OUT = ROOT / "outputs" / "science_synthesis"
PAPER_FIGURES = ROOT / "outputs" / "figures"
PAPER_TABLES = ROOT / "outputs" / "tables"
SHADOW_OUT = ROOT / "outputs" / "stage3_transport_shadow" / "through_many_small_grains_n100k"
STAGE3_OUT = ROOT / "outputs" / "stage3_particle_production"
NEXT_OUT = ROOT / "outputs" / "next_interception_kernel"
GEOM_ROOT = ROOT / "outputs" / "stage1_geometry_screen" / "geometries"
FULL_SUITE = ROOT / "outputs" / "openfoam_full_suite" / "openfoam_full_refinement_summary.csv"

PROFILE_LABELS = {
    "neutral_resolved": "No DLVO",
    "favorable_50mM_z70": "Favorable 50 mM",
    "unfavorable_50mM_z70": "Unfavorable 50 mM",
    "unfavorable_75mM_z70": "Unfavorable 75 mM",
    "unfavorable_100mM_z70": "Unfavorable 100 mM",
    "unfavorable_50mM_z70_100xD": "Unfavorable 50 mM, 100D",
}

PROFILE_COLORS = {
    "neutral_resolved": "#2563eb",
    "favorable_50mM_z70": "#dc2626",
    "unfavorable_50mM_z70": "#7c3aed",
    "unfavorable_75mM_z70": "#9333ea",
    "unfavorable_100mM_z70": "#581c87",
    "unfavorable_50mM_z70_100xD": "#0f766e",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def load_geometry(candidate: str) -> dict:
    return json.loads((GEOM_ROOT / candidate / "geometry.json").read_text(encoding="utf-8"))


def add_grains(ax: plt.Axes, geometry: dict, *, alpha: float = 1.0) -> None:
    for grain in geometry["grains"]:
        ax.add_patch(
            Circle(
                (float(grain["x"]) * 1.0e3, float(grain["y"]) * 1.0e3),
                float(grain["radius"]) * 1.0e3,
                fc="#020617",
                ec="none",
                alpha=alpha,
                zorder=4,
            )
        )


def cell_xy(shape: tuple[int, int], geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = shape
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    x = (np.arange(nx) + 0.5) * lx * 1.0e3 / nx
    y = (np.arange(ny) + 0.5) * ly * 1.0e3 / ny
    return np.meshgrid(x, y, indexing="ij")


def occupancy(profile: str) -> np.ndarray:
    matches = sorted(SHADOW_OUT.glob(f"{profile}_occupancy_*.npz"))
    if not matches:
        raise FileNotFoundError(f"missing occupancy for {profile}")
    return np.load(matches[0])["pore_accessibility"]


def top_depletion_mask(reference: np.ndarray, condition: np.ndarray, support: np.ndarray, count: int) -> np.ndarray:
    depletion = np.maximum(reference - condition, 0.0)
    mask = np.zeros_like(support, dtype=bool)
    positive = depletion[support]
    positive = positive[positive > 0.0]
    if count <= 0 or positive.size == 0:
        return mask
    threshold = np.partition(positive, -min(count, positive.size))[-min(count, positive.size)]
    return support & (depletion >= threshold) & (depletion > 0.0)


def shadow_science_figure() -> Path:
    geometry = load_geometry("through_many_small_grains")
    reference = occupancy("neutral_resolved")
    unfav = occupancy("unfavorable_50mM_z70")
    highd = occupancy("unfavorable_50mM_z70_100xD")
    eps = 0.005
    pseudocount = 0.5 / 100000.0
    support = reference > eps
    binary_shadow = support & (unfav <= eps)
    top_dep = top_depletion_mask(reference, unfav, support, int(binary_shadow.sum()))
    xx, yy = cell_xy(reference.shape, geometry)
    lx = float(geometry["domain"]["length_x"]) * 1.0e3
    ly = float(geometry["domain"]["length_y"]) * 1.0e3

    ratio = np.log2((unfav + pseudocount) / (reference + pseudocount))
    ratio[~support] = np.nan
    depletion = np.where(support, np.maximum(reference - unfav, 0.0), np.nan)
    max_dep = float(np.nanquantile(depletion, 0.995))

    metrics = [row for row in read_csv(SHADOW_OUT / "pore_shadow_metrics" / "pore_shadow_metric_summary.csv")]
    bins = read_csv(SHADOW_OUT / "shadow_flow_characteristics" / "shadow_mechanism_bins.csv")
    speed_bins = [row for row in bins if row["bin_type"] == "speed_quintile"]
    speed_bins.sort(key=lambda row: i(row, "bin_index"))

    fig = plt.figure(figsize=(12.4, 8.4), dpi=240)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 0.98], hspace=0.28, wspace=0.22)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f8fafc")
    im = axes[0].imshow(
        ratio.T,
        origin="lower",
        extent=(0, lx, 0, ly),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
        aspect="equal",
        zorder=1,
    )
    axes[0].scatter(xx[binary_shadow], yy[binary_shadow], marker="s", s=34, facecolors="none", edgecolors="#06b6d4", linewidths=1.1, label="binary shadows", zorder=5)
    axes[0].scatter(xx[top_dep], yy[top_dep], marker="o", s=34, facecolors="none", edgecolors="#f97316", linewidths=1.1, label="largest depletions", zorder=5)
    add_grains(axes[0], geometry)
    axes[0].add_patch(Rectangle((0, 0), lx, ly, fill=False, ec="#0f172a", lw=0.8))
    axes[0].set_title(r"Local accessibility ratio, $\log_2(H_\chi/H_0)$")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].legend(frameon=True, fontsize=7, loc="lower right")
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.78, pad=0.012)
    cbar.ax.tick_params(labelsize=7)

    cmap2 = plt.get_cmap("viridis").copy()
    cmap2.set_bad("#f8fafc")
    im2 = axes[1].imshow(
        depletion.T,
        origin="lower",
        extent=(0, lx, 0, ly),
        cmap=cmap2,
        vmin=0.0,
        vmax=max_dep,
        aspect="equal",
        zorder=1,
    )
    axes[1].scatter(xx[binary_shadow], yy[binary_shadow], marker="s", s=34, facecolors="none", edgecolors="#06b6d4", linewidths=1.1, zorder=5)
    axes[1].scatter(xx[top_dep], yy[top_dep], marker="o", s=34, facecolors="none", edgecolors="#f97316", linewidths=1.1, zorder=5)
    add_grains(axes[1], geometry)
    axes[1].add_patch(Rectangle((0, 0), lx, ly, fill=False, ec="#0f172a", lw=0.8))
    axes[1].set_title(r"Continuous depletion, $\max(H_0-H_\chi,0)$")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.78, pad=0.012)
    cbar2.ax.tick_params(labelsize=7)

    x = np.arange(len(speed_bins))
    depleted = np.array([f(row, "depleted_mass_fraction") for row in speed_bins])
    enriched = np.array([f(row, "enriched_mass_fraction") for row in speed_bins])
    net = np.array([f(row, "net_change_fraction") for row in speed_bins])
    shadows = np.array([i(row, "shadow_count") for row in speed_bins])
    axes[2].bar(x, -depleted, color="#2563eb", label="depleted mass")
    axes[2].bar(x, enriched, color="#f97316", label="enriched mass")
    axes[2].plot(x, net, color="#111827", marker="o", lw=1.6, label="net")
    axes[2].axhline(0.0, color="#475569", lw=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(j + 1) for j in x])
    axes[2].set_xlabel("speed quintile, low to high")
    axes[2].set_ylabel("fraction of reference accessibility mass")
    axes[2].set_title("Continuous depletion is carried by fast pathways")
    axes[2].grid(True, axis="y", color="#e5e7eb", lw=0.7)
    twin = axes[2].twinx()
    twin.plot(x, shadows, color="#06b6d4", marker="s", lw=1.4, label="binary-shadow cells")
    twin.set_ylabel("binary-shadow cells", color="#0891b2")
    twin.tick_params(axis="y", labelcolor="#0891b2")

    profile_order = ["unfavorable_50mM_z70", "unfavorable_50mM_z70_100xD"]
    row_by_profile = {row["profile"]: row for row in metrics}
    labels = ["50 mM unfav.", "50 mM unfav., 100D"]
    x = np.arange(len(profile_order))
    width = 0.22
    axes[3].bar(x - width, [f(row_by_profile[p], "binary_shadow_fraction") for p in profile_order], width=width, color="#94a3b8", label="binary S")
    axes[3].bar(x, [f(row_by_profile[p], "net_shadow_loss_fraction") for p in profile_order], width=width, color="#2563eb", label="net loss")
    axes[3].bar(x + width, [f(row_by_profile[p], "total_variation_distribution") for p in profile_order], width=width, color="#7c3aed", label="TV distance")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=12, ha="right")
    axes[3].set_ylabel("metric value")
    axes[3].set_title("Binary shadows are small; redistribution is larger")
    axes[3].legend(frameon=False, fontsize=8)
    axes[3].grid(True, axis="y", color="#e5e7eb", lw=0.7)

    for label, ax in zip("abcd", axes, strict=True):
        ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.suptitle("Finite-time transport shadows are exposure redistribution, not broad pore exclusion", fontsize=14, fontweight="bold")
    path = PAPER_FIGURES / "science_shadow_synthesis.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "science_shadow_synthesis.png", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for profile in profile_order:
        row = row_by_profile[profile]
        rows.append(
            {
                "profile": profile,
                "label": PROFILE_LABELS[profile],
                "binary_shadow_fraction": f(row, "binary_shadow_fraction"),
                "depleted_mass_fraction": f(row, "depleted_accessibility_mass_fraction"),
                "enriched_mass_fraction": f(row, "enriched_accessibility_mass_fraction"),
                "net_shadow_loss_fraction": f(row, "net_shadow_loss_fraction"),
                "total_variation_distribution": f(row, "total_variation_distribution"),
            }
        )
    write_csv(OUT / "science_shadow_summary.csv", rows)
    return path


def next_interception_science_figure() -> Path:
    summary = read_csv(NEXT_OUT / "release_to_next_interception_summary.csv")
    events = read_csv(NEXT_OUT / "release_to_next_interception_events.csv")
    profiles = [
        "neutral_resolved",
        "unfavorable_50mM_z70",
        "unfavorable_75mM_z70",
        "unfavorable_100mM_z70",
        "unfavorable_50mM_z70_100xD",
    ]
    all_rows = {row["profile"]: row for row in summary if row["release_class"] == "all"}

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), dpi=240)
    x = np.arange(len(profiles))
    labels = [PROFILE_LABELS[p].replace("Unfavorable ", "") for p in profiles]
    colors = [PROFILE_COLORS[p] for p in profiles]
    width = 0.38
    axes[0, 0].bar(x - width / 2, [f(all_rows[p], "release_F30") for p in profiles], width, color=colors, alpha=0.45, label="release")
    axes[0, 0].bar(x + width / 2, [f(all_rows[p], "next_F30_rear") for p in profiles], width, color=colors, alpha=0.95, label="next event")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("fraction within rear 30 deg")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0, 0].set_title("Focused release remains focused at the next event")
    axes[0, 0].legend(frameon=False, fontsize=8)

    classes = ["focused", "transition", "broad"]
    selected = ["neutral_resolved", "unfavorable_50mM_z70", "unfavorable_100mM_z70", "unfavorable_50mM_z70_100xD"]
    by_key = {(row["profile"], row["release_class"]): row for row in summary}
    xx = np.arange(len(selected))
    offsets = np.linspace(-0.24, 0.24, len(classes))
    class_colors = {"focused": "#7c3aed", "transition": "#f59e0b", "broad": "#64748b"}
    for cls, offset in zip(classes, offsets, strict=True):
        axes[0, 1].bar(
            xx + offset,
            [f(by_key[(p, cls)], "next_F30_rear") for p in selected],
            width=0.22,
            color=class_colors[cls],
            label=cls,
        )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("next-event rear-zone fraction")
    axes[0, 1].set_xticks(xx)
    axes[0, 1].set_xticklabels([PROFILE_LABELS[p].replace("Unfavorable ", "") for p in selected], rotation=18, ha="right")
    axes[0, 1].set_title("Release-angle class predicts the next encounter")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].bar(x, [f(all_rows[p], "next_same_grain_fraction") for p in profiles], color=colors, alpha=0.88)
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("same-grain next-event fraction")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=18, ha="right")
    axes[1, 0].set_title("Wake-region re-encounter is common")
    ax2 = axes[1, 0].twinx()
    ax2.plot(x, [f(all_rows[p], "next_effective_grain_count") for p in profiles], color="#111827", marker="o", lw=1.5)
    ax2.set_ylabel("effective next grains")

    group = [row for row in events if row["profile"] == "unfavorable_100mM_z70" and truth(row["next_interception"])]
    release = np.array([f(row, "release_delta_rear_deg") for row in group])
    next_delta = np.array([f(row, "next_delta_rear_deg") for row in group])
    hb = axes[1, 1].hexbin(
        release,
        next_delta,
        gridsize=36,
        extent=(0.0, 180.0, 0.0, 180.0),
        cmap="Purples",
        norm=LogNorm(vmin=1),
        mincnt=1,
        linewidths=0.0,
    )
    axes[1, 1].axvline(30.0, color="#dc2626", lw=0.9, ls="--")
    axes[1, 1].axhline(30.0, color="#dc2626", lw=0.9, ls="--")
    axes[1, 1].set_xlim(0.0, 180.0)
    axes[1, 1].set_ylim(0.0, 180.0)
    axes[1, 1].set_xlabel("release angle from rear (deg)")
    axes[1, 1].set_ylabel("next-event angle from rear (deg)")
    axes[1, 1].set_title("100 mM event-state kernel")
    cbar = fig.colorbar(hb, ax=axes[1, 1], shrink=0.80, pad=0.012)
    cbar.set_label("events/bin (log)")

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.11, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.7, alpha=0.85)
    fig.tight_layout()
    path = PAPER_FIGURES / "science_next_interception_kernel.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "science_next_interception_kernel.png", bbox_inches="tight")
    plt.close(fig)
    return path


def geometry_features(candidate: str) -> dict[str, object]:
    geometry = load_geometry(candidate)
    derived = geometry["derived"]
    config = geometry["config"]
    inlet_width = sum(float(window["width"]) for window in geometry.get("injection_windows_x0", []))
    ly = float(geometry["domain"]["length_y"])
    return {
        "candidate_id": candidate,
        "grains": int(config["n_grains"]),
        "porosity": float(derived["actual_porosity"]),
        "radius_cv": float(derived["radius_cv"]),
        "minimum_gap_um": float(derived["minimum_realized_surface_gap"]) * 1.0e6,
        "fifth_percentile_gap_um": float(derived["fifth_percentile_surface_gap"]) * 1.0e6,
        "median_gap_um": float(derived["median_surface_gap"]) * 1.0e6,
        "inlet_open_fraction": inlet_width / ly if ly > 0.0 else float("nan"),
    }


def random_geometry_science_figure() -> Path:
    focus_rows = read_csv(STAGE3_OUT / "stage3_random_focusing_by_geometry.csv")
    condition_rows = read_csv(STAGE3_OUT / "stage3_random_condition_summary.csv")
    candidates = ["through_many_small_grains", "through_wide_throats", "through_random_00", "through_random_01", "through_low_porosity"]
    profiles = ["neutral_resolved", "favorable_50mM_z70", "unfavorable_50mM_z70", "unfavorable_50mM_z70_100xD"]
    focus = {(row["candidate_id"], row["profile"]): row for row in focus_rows}
    cond = {(row["candidate_id"], row["profile"]): row for row in condition_rows}
    features = {candidate: geometry_features(candidate) for candidate in candidates}
    short = {
        "through_many_small_grains": "many small",
        "through_wide_throats": "wide throats",
        "through_random_00": "random 00",
        "through_random_01": "random 01",
        "through_low_porosity": "low porosity",
    }

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        row = dict(features[candidate])
        row["label"] = short[candidate]
        for profile in profiles:
            row[f"F30_{profile}"] = f(focus[(candidate, profile)], "fraction_released_within_30deg")
            row[f"release_fraction_{profile}"] = f(focus[(candidate, profile)], "release_fraction_of_intercepted")
            row[f"intercepted_fraction_{profile}"] = f(cond[(candidate, profile)], "intercepted_fraction")
        row["delta_F30_unfavorable_minus_neutral"] = row["F30_unfavorable_50mM_z70"] - row["F30_neutral_resolved"]
        row["delta_F30_unfavorable_minus_100D"] = row["F30_unfavorable_50mM_z70"] - row["F30_unfavorable_50mM_z70_100xD"]
        rows.append(row)
    write_csv(OUT / "random_geometry_science_summary.csv", rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), dpi=240)
    x = np.arange(len(candidates))
    offsets = np.linspace(-0.27, 0.27, len(profiles))
    for profile, offset in zip(profiles, offsets, strict=True):
        axes[0, 0].scatter(
            x + offset,
            [f(focus[(candidate, profile)], "fraction_released_within_30deg") for candidate in candidates],
            s=70,
            color=PROFILE_COLORS[profile],
            label=PROFILE_LABELS[profile],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([short[c] for c in candidates], rotation=18, ha="right")
    axes[0, 0].set_ylabel(r"grain-local $F_{30}^{\rm rand}$")
    axes[0, 0].set_ylim(0.0, 0.45)
    axes[0, 0].set_title("Focusing contrast across screened geometries")
    axes[0, 0].legend(frameon=False, fontsize=7, ncols=2)

    delta = np.array([row["delta_F30_unfavorable_minus_neutral"] for row in rows], dtype=float)
    min_gap = np.array([row["minimum_gap_um"] for row in rows], dtype=float)
    porosity = np.array([row["porosity"] for row in rows], dtype=float)
    sc = axes[0, 1].scatter(min_gap, delta, s=90 + 600 * (0.52 - porosity), c=porosity, cmap="viridis_r", edgecolor="#111827", linewidth=0.7)
    for row in rows:
        axes[0, 1].annotate(row["label"], (row["minimum_gap_um"], row["delta_F30_unfavorable_minus_neutral"]), xytext=(5, 4), textcoords="offset points", fontsize=7)
    axes[0, 1].set_xlabel("minimum surface gap (um)")
    axes[0, 1].set_ylabel(r"$\Delta F_{30}$, unfavorable - no DLVO")
    axes[0, 1].set_title("Designed geometry spread changes magnitude, not sign")
    cbar = fig.colorbar(sc, ax=axes[0, 1], shrink=0.82, pad=0.012)
    cbar.set_label("porosity")

    for profile in ["neutral_resolved", "unfavorable_50mM_z70", "unfavorable_50mM_z70_100xD"]:
        axes[1, 0].scatter(
            [f(focus[(candidate, profile)], "release_fraction_of_intercepted") for candidate in candidates],
            [f(focus[(candidate, profile)], "fraction_released_within_30deg") for candidate in candidates],
            s=75,
            color=PROFILE_COLORS[profile],
            label=PROFILE_LABELS[profile],
            edgecolor="white",
            linewidth=0.6,
        )
    axes[1, 0].set_xlabel("mobile releases per intercepted particle")
    axes[1, 0].set_ylabel(r"$F_{30}^{\rm rand}$")
    axes[1, 0].set_title("Focusing is not just more release support")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].bar(x, [row["delta_F30_unfavorable_minus_neutral"] for row in rows], color="#7c3aed", label="unfav. - no DLVO")
    axes[1, 1].bar(x, [row["delta_F30_unfavorable_minus_100D"] for row in rows], bottom=[row["delta_F30_unfavorable_minus_neutral"] for row in rows], color="#0f766e", alpha=0.55, label="unfav. - 100D")
    axes[1, 1].axhline(0.0, color="#475569", lw=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([short[c] for c in candidates], rotation=18, ha="right")
    axes[1, 1].set_ylabel(r"focusing contrast")
    axes[1, 1].set_title("Normal-diffusivity unfavorable cases focus in every geometry")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.11, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.grid(True, color="#e5e7eb", lw=0.7, alpha=0.85)
    fig.tight_layout()
    path = PAPER_FIGURES / "science_random_geometry_controls.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "science_random_geometry_controls.png", bbox_inches="tight")
    plt.close(fig)
    write_random_geometry_table(rows)
    return path


def classify_regime(row: dict[str, str]) -> str:
    profile = row["profile"]
    q_release = release_fraction(row)
    f30 = focusing_value(row)
    unresolved = unresolved_fraction(row)
    if "100xD" in profile:
        return "diffusive decorrelation"
    if profile.startswith("neutral"):
        return "no well / broad"
    if unresolved >= 0.10 or q_release < 0.5:
        return "finite-horizon residence"
    if f30 >= 0.50:
        return "focused mobile"
    return "weak residence / broad"


def short_case_label(label: str) -> str:
    replacements = (
        (" mM, -70 mV", " z70"),
        (" mM, -50 mV", " z50"),
        (" mM, -30 mV", " z30"),
        (" mM, -20 mV, 10x A", " z20 10A"),
        ("100x D", "100D"),
        ("No DLVO", "No DLVO"),
    )
    out = label
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def focusing_value(row: dict[str, str]) -> float:
    well_f30 = f(row, "center_well_release_theta30_fraction")
    return well_f30 if math.isfinite(well_f30) else f(row, "center_release_theta30_fraction")


def residence_time(row: dict[str, str]) -> float:
    well_tau = f(row, "center_well_time_median_s")
    return well_tau if math.isfinite(well_tau) and well_tau > 0.0 else f(row, "center_near_time_median_s")


def release_fraction(row: dict[str, str]) -> float:
    well_in = i(row, "center_well_intercepted_count")
    if well_in > 0:
        return i(row, "center_well_release_count") / well_in
    center_in = i(row, "center_intercepted_count")
    return i(row, "center_release_count") / center_in if center_in > 0 else float("nan")


def unresolved_fraction(row: dict[str, str]) -> float:
    well_in = i(row, "center_well_intercepted_count")
    if well_in > 0:
        return i(row, "center_well_censored_count") / well_in
    center_in = i(row, "center_intercepted_count")
    return i(row, "center_censored_count") / center_in if center_in > 0 else float("nan")


def regime_science_figure() -> Path:
    raw = read_csv(FULL_SUITE)
    rows: list[dict[str, object]] = []
    for row in raw:
        f30 = focusing_value(row)
        tau = residence_time(row)
        qrel = release_fraction(row)
        unresolved = unresolved_fraction(row)
        rows.append(
            {
                "profile": row["profile"],
                "label": row["label"],
                "median_residence_s": tau,
                "F30": f30,
                "release_fraction": qrel,
                "unresolved_fraction": unresolved,
                "focused_mobile_yield": qrel * f30 if math.isfinite(qrel) and math.isfinite(f30) else float("nan"),
                "well_depth_kbt": f(row, "well_minimum_kbt"),
                "diffusivity_multiplier": f(row, "diffusivity_multiplier"),
                "regime_synthesis": classify_regime(row),
            }
        )
    write_csv(OUT / "science_regime_summary.csv", rows)
    write_regime_table(rows)

    colors = {
        "no well / broad": "#2563eb",
        "weak residence / broad": "#94a3b8",
        "focused mobile": "#7c3aed",
        "finite-horizon residence": "#dc2626",
        "diffusive decorrelation": "#0f766e",
    }
    markers = {"diffusive decorrelation": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.5), dpi=240)
    annotate_profiles = {
        "neutral_resolved",
        "unfavorable_6mM_z70",
        "unfavorable_50mM_z70",
        "unfavorable_100mM_z70",
        "unfavorable_100mM_z30",
        "unfavorable_50mM_z70_100xD",
    }

    for row in rows:
        tau = float(row["median_residence_s"])
        f30 = float(row["F30"])
        if not (math.isfinite(tau) and math.isfinite(f30) and tau > 0):
            continue
        regime = str(row["regime_synthesis"])
        axes[0, 0].scatter(tau, f30, s=70 + 170 * float(row["release_fraction"]), color=colors[regime], marker=markers.get(regime, "o"), edgecolor="white", linewidth=0.7)
        if row["profile"] in annotate_profiles:
            axes[0, 0].annotate(short_case_label(str(row["label"])), (tau, f30), xytext=(4, 4), textcoords="offset points", fontsize=6.7)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("median near-wall or well residence (s)")
    axes[0, 0].set_ylabel(r"$F_{30}$")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_title("Focusing occupies an intermediate residence window")

    for row in rows:
        tau = float(row["median_residence_s"])
        yield30 = float(row["focused_mobile_yield"])
        if not (math.isfinite(tau) and math.isfinite(yield30) and tau > 0):
            continue
        regime = str(row["regime_synthesis"])
        axes[0, 1].scatter(tau, yield30, s=90, color=colors[regime], marker=markers.get(regime, "o"), edgecolor="white", linewidth=0.7)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("median residence (s)")
    axes[0, 1].set_ylabel(r"focused mobile yield, $Y_{30}=q_{\rm rel}F_{30}$")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("Yield separates clean focusing from unresolved residence")

    salt_rows = [row for row in rows if math.isfinite(float(row["well_depth_kbt"])) and float(row["diffusivity_multiplier"]) == 1.0]
    salt_rows.sort(key=lambda row: abs(float(row["well_depth_kbt"])))
    axes[1, 0].plot(
        [abs(float(row["well_depth_kbt"])) for row in salt_rows],
        [float(row["median_residence_s"]) for row in salt_rows],
        color="#7c3aed",
        marker="o",
        lw=1.6,
    )
    for row in salt_rows:
        if row["profile"] in annotate_profiles:
            axes[1, 0].annotate(short_case_label(str(row["label"])), (abs(float(row["well_depth_kbt"])), float(row["median_residence_s"])), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel(r"secondary-minimum depth, $|U_{\min}|/k_BT$")
    axes[1, 0].set_ylabel("median residence (s)")
    axes[1, 0].set_title("DLVO well depth controls residence")

    display = [row for row in rows if math.isfinite(float(row["F30"]))]
    display.sort(key=lambda row: float(row["median_residence_s"]))
    y = np.arange(len(display))
    focused = np.array([float(row["focused_mobile_yield"]) for row in display])
    qrel = np.array([float(row["release_fraction"]) for row in display])
    unresolved = np.array([float(row["unresolved_fraction"]) for row in display])
    axes[1, 1].barh(y, focused, color="#7c3aed", label="focused mobile")
    axes[1, 1].barh(y, np.maximum(qrel - focused, 0.0), left=focused, color="#94a3b8", label="other mobile release")
    axes[1, 1].barh(y, unresolved, left=qrel, color="#dc2626", label="finite-horizon unresolved")
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels([short_case_label(str(row["label"])) for row in display], fontsize=7)
    axes[1, 1].set_xlim(0.0, 1.05)
    axes[1, 1].set_xlabel("fraction of center-well or near-surface entrants")
    axes[1, 1].set_title("Regime accounting")
    axes[1, 1].legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.54, -0.29), ncols=3)

    handles = []
    for name, color in colors.items():
        handles.append(plt.Line2D([0], [0], marker=markers.get(name, "o"), color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=name))
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.11, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.grid(True, color="#e5e7eb", lw=0.7, alpha=0.85)
    fig.tight_layout()
    path = PAPER_FIGURES / "science_regime_map.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "science_regime_map.png", bbox_inches="tight")
    plt.close(fig)
    return path


def write_random_geometry_table(rows: list[dict[str, object]]) -> Path:
    path = PAPER_TABLES / "random_geometry_science_summary.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Geometry-control summary for the five-geometry random-packing ensemble. \(\Delta F_{30}\) is the difference between unfavorable 50 mM and no-DLVO grain-local rear-zone release fractions.}",
        r"\label{tab:random-geometry-science}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"geometry & porosity & grains & min gap (um) & \(F_{30}^{0}\) & \(F_{30}^{50}\) & \(\Delta F_{30}\) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            "{label} & {porosity:.2f} & {grains:.0f} & {minimum_gap_um:.1f} & {F30_neutral_resolved:.3f} & {F30_unfavorable_50mM_z70:.3f} & {delta_F30_unfavorable_minus_neutral:.3f} \\\\".format(
                **row
            )
        )
    lines.extend([r"\hline", r"\end{tabular}", r"}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_regime_table(rows: list[dict[str, object]]) -> Path:
    selected = [
        row
        for row in rows
        if row["profile"]
        in {
            "neutral_resolved",
            "unfavorable_6mM_z70",
            "unfavorable_50mM_z70",
            "unfavorable_100mM_z70",
            "unfavorable_100mM_z30",
            "unfavorable_50mM_z70_100xD",
        }
    ]
    path = PAPER_TABLES / "science_regime_summary.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Synthesis of the focusing regime map. \(Y_{30}=q_{\rm rel}F_{30}\) is the completed focused mobile yield among center-well or near-surface entrants.}",
        r"\label{tab:science-regime-summary}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrl}",
        r"\hline",
        r"case & median \(\tau\) (s) & \(F_{30}\) & \(q_{\rm rel}\) & \(Y_{30}\) & regime \\",
        r"\hline",
    ]
    for row in selected:
        lines.append(
            "{label} & {median_residence_s:.3g} & {F30:.3f} & {release_fraction:.3f} & {focused_mobile_yield:.3f} & {regime_synthesis} \\\\".format(
                **row
            )
        )
    lines.extend([r"\hline", r"\end{tabular}", r"}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_event_state_table() -> Path:
    rows = [
        ("mobile displacement", r"\(y_{\rm in},y_{\rm out},T\)", "needed for ordinary plume motion; insufficient for focusing by itself"),
        ("collector event", r"\(c,g,E,o\)", "retains collector family or grain identity, event type, and outcome class"),
        ("release memory", r"\(\theta_{\rm rel},\tau_{\rm res},\Delta\theta\)", "stores the residence-driven angular state that exposes focusing"),
        ("next encounter", r"\(g_{\rm next},\theta_{\rm next},T_{\rm next}\)", "tests whether release memory is carried into the next near-surface event"),
        ("finite-time exposure", r"\(H_\chi(\mathbf x),H_\chi(g,\theta)\)", "records redistribution and shadows over the simulated horizon"),
    ]
    path = PAPER_TABLES / "event_state_kernel_table.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Event-state variables retained by the transition-library view. The ordinary transverse matrix is the mobile-displacement marginal of this larger kernel.}",
        r"\label{tab:event-state-kernel}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}>{\raggedright\arraybackslash}p{0.46\linewidth}}",
        r"\hline",
        r"state block & variables & role in this paper \\",
        r"\hline",
    ]
    for name, variables, role in rows:
        lines.append(f"{tex_escape(name)} & {variables} & {tex_escape(role)} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    outputs = [
        shadow_science_figure(),
        next_interception_science_figure(),
        random_geometry_science_figure(),
        regime_science_figure(),
        write_event_state_table(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
