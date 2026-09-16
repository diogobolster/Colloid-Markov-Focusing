#!/usr/bin/env python3
"""Alternative pore-shadow metrics from aggregate occupancy fields."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle


OUT = ROOT / "outputs" / "pore_shadow_metrics"
FIGURES = ROOT / "outputs" / "figures"
TRANSPORT_OUT = ROOT / "outputs" / "transport_shadow_diagnostic"

PROFILES = {
    "unfavorable_50mM_z70": {"label": "50 mM unfavorable", "color": "#2563eb"},
    "unfavorable_100mM_z70": {"label": "100 mM unfavorable", "color": "#581c87"},
    "unfavorable_50mM_z70_100xD": {"label": "50 mM unfavorable, 100D", "color": "#0f766e"},
}


def default_tag(args: argparse.Namespace) -> str:
    return (
        f"n{args.particles}_t{args.max_time:g}_dt{args.dt:g}_"
        f"p{args.pore_nx}x{args.pore_ny}_s{args.surface_bins}_stride{args.sample_stride}_seed{args.seed}"
    ).replace(".", "p")


def load_accessibility(profile: str, tag: str) -> np.ndarray:
    path = TRANSPORT_OUT / f"{profile}_occupancy_{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)["pore_accessibility"]


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


def periodic_delta(values: np.ndarray, length: float) -> np.ndarray:
    return (values + 0.5 * length) % length - 0.5 * length


def cell_geometry(geometry: dict, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    xs = (np.arange(nx) + 0.5) * lx / nx
    ys = (np.arange(ny) + 0.5) * ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    gap = np.full((nx, ny), np.inf)
    inside = np.zeros((nx, ny), dtype=bool)
    for grain in geometry["grains"]:
        dx = periodic_delta(xx - float(grain["x"]), lx)
        dy = periodic_delta(yy - float(grain["y"]), ly)
        dist = np.hypot(dx, dy)
        radius = float(grain["radius"])
        gap = np.minimum(gap, dist - radius)
        inside |= dist <= radius
    return xx, yy, gap, inside


def normalized(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    total = float(np.sum(values[mask]))
    if total > 0.0:
        out[mask] = values[mask] / total
    return out


def shannon_entropy_effective(values: np.ndarray, mask: np.ndarray) -> float:
    p = normalized(values, mask)
    positive = p[p > 0.0]
    if positive.size == 0:
        return float("nan")
    return float(math.exp(-np.sum(positive * np.log(positive))))


def distribution_metrics(reference: np.ndarray, condition: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    p = normalized(reference, mask)
    q = normalized(condition, mask)
    positive = mask & ((p > 0.0) | (q > 0.0))
    tv = 0.5 * float(np.sum(np.abs(p[positive] - q[positive])))
    hellinger = math.sqrt(0.5 * float(np.sum((np.sqrt(p[positive]) - np.sqrt(q[positive])) ** 2)))
    m = 0.5 * (p + q)
    pm = positive & (p > 0.0)
    qm = positive & (q > 0.0)
    js = 0.5 * float(np.sum(p[pm] * np.log2(p[pm] / m[pm]))) + 0.5 * float(np.sum(q[qm] * np.log2(q[qm] / m[qm])))
    return tv, hellinger, js


def component_metrics(mask: np.ndarray) -> tuple[int, int, float]:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return 0, 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max())
    largest_fraction = float(largest / np.sum(mask))
    return int(count), largest, largest_fraction


def metrics_for_profile(
    profile: str,
    reference: np.ndarray,
    condition: np.ndarray,
    gap: np.ndarray,
    fluid: np.ndarray,
    eps: float,
    near_gap_m: float,
    pseudocount: float,
) -> dict[str, object]:
    support = fluid & (reference > eps)
    union = fluid & ((reference > 0.0) | (condition > 0.0))
    ratio = np.divide(condition + pseudocount, reference + pseudocount, out=np.ones_like(reference), where=reference + pseudocount > 0.0)
    raw_ratio = np.divide(condition, reference, out=np.full_like(reference, np.nan), where=reference > 0.0)
    depletion = np.maximum(reference - condition, 0.0)
    enrichment = np.maximum(condition - reference, 0.0)
    shadow = support & (condition <= eps)
    half_depleted = support & (raw_ratio < 0.5)
    near = support & (gap <= near_gap_m)
    core = support & (gap > near_gap_m)
    support_count = int(np.sum(support))
    shadow_count = int(np.sum(shadow))
    support_ref_mass = float(np.sum(reference[support]))

    def sum_fraction(values: np.ndarray, mask: np.ndarray, denom: float) -> float:
        if denom <= 0.0:
            return float("nan")
        return float(np.sum(values[mask]) / denom)

    depleted_fraction = sum_fraction(depletion, support, support_ref_mass)
    enriched_fraction = sum_fraction(enrichment, support, support_ref_mass)
    net_change = float((np.sum(condition[support]) - support_ref_mass) / support_ref_mass) if support_ref_mass > 0.0 else float("nan")
    redistribution_total = depleted_fraction + enriched_fraction
    tv, hellinger, js = distribution_metrics(reference, condition, union)
    clusters, largest_cluster, largest_cluster_fraction = component_metrics(shadow)

    def median_ratio(mask: np.ndarray) -> float:
        vals = raw_ratio[mask & np.isfinite(raw_ratio)]
        if vals.size == 0:
            return float("nan")
        return float(np.median(vals))

    def quantile_ratio(q: float) -> float:
        vals = raw_ratio[support & np.isfinite(raw_ratio)]
        if vals.size == 0:
            return float("nan")
        return float(np.quantile(vals, q))

    return {
        "profile": profile,
        "label": PROFILES[profile]["label"],
        "epsilon": eps,
        "reference_supported_cells": support_count,
        "binary_shadow_cells": shadow_count,
        "binary_shadow_fraction": float(shadow_count / support_count) if support_count else float("nan"),
        "reference_accessibility_mass": support_ref_mass,
        "depleted_accessibility_mass_fraction": depleted_fraction,
        "enriched_accessibility_mass_fraction": enriched_fraction,
        "net_accessibility_change_fraction": net_change,
        "net_shadow_loss_fraction": max(-net_change, 0.0) if np.isfinite(net_change) else float("nan"),
        "shadow_dominance_fraction": float(depleted_fraction / redistribution_total) if redistribution_total > 0.0 else float("nan"),
        "half_ratio_cells": int(np.sum(half_depleted)),
        "half_ratio_cell_fraction": float(np.sum(half_depleted) / support_count) if support_count else float("nan"),
        "half_ratio_reference_mass_fraction": sum_fraction(reference, half_depleted, support_ref_mass),
        "near_gap_um": near_gap_m * 1.0e6,
        "near_supported_cells": int(np.sum(near)),
        "near_depleted_mass_fraction": sum_fraction(depletion, near, float(np.sum(reference[near]))),
        "core_supported_cells": int(np.sum(core)),
        "core_depleted_mass_fraction": sum_fraction(depletion, core, float(np.sum(reference[core]))),
        "median_ratio_supported": median_ratio(support),
        "q10_ratio_supported": quantile_ratio(0.10),
        "q25_ratio_supported": quantile_ratio(0.25),
        "total_variation_distribution": tv,
        "hellinger_distribution": hellinger,
        "jensen_shannon_bits": js,
        "effective_cells_reference": shannon_entropy_effective(reference, union),
        "effective_cells_condition": shannon_entropy_effective(condition, union),
        "shadow_component_count": clusters,
        "largest_shadow_component_cells": largest_cluster,
        "largest_shadow_component_fraction": largest_cluster_fraction,
    }


def plot_maps(
    geometry: dict,
    reference: np.ndarray,
    conditions: dict[str, np.ndarray],
    rows: list[dict[str, object]],
    gap: np.ndarray,
    inside: np.ndarray,
    eps: float,
    pseudocount: float,
) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    profiles = list(conditions)
    fig, axes = plt.subplots(2, len(profiles), figsize=(4.2 * len(profiles), 7.2), dpi=240, squeeze=False)
    norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f8fafc")
    loss_cmap = plt.get_cmap("viridis").copy()
    loss_cmap.set_bad("#f8fafc")
    support = reference > eps
    max_loss = 0.0
    losses: dict[str, np.ndarray] = {}
    for profile, condition in conditions.items():
        loss = np.where(support, np.maximum(reference - condition, 0.0), np.nan)
        loss[inside] = np.nan
        losses[profile] = loss
        finite = loss[np.isfinite(loss)]
        if finite.size:
            max_loss = max(max_loss, float(np.quantile(finite, 0.995)))
    max_loss = max(max_loss, 1.0e-4)
    row_by_profile = {str(row["profile"]): row for row in rows}
    for col, profile in enumerate(profiles):
        condition = conditions[profile]
        ratio = np.log2((condition + pseudocount) / (reference + pseudocount))
        ratio[~support] = np.nan
        ratio[inside] = np.nan
        shadow = support & (condition <= eps)
        ix, iy = np.where(shadow)
        ax = axes[0, col]
        im = ax.imshow(
            ratio.T,
            origin="lower",
            extent=(0, lx * 1.0e3, 0, ly * 1.0e3),
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        ax.scatter(
            (ix + 0.5) * lx * 1.0e3 / reference.shape[0],
            (iy + 0.5) * ly * 1.0e3 / reference.shape[1],
            marker="s",
            s=20,
            facecolors="none",
            edgecolors="#22d3ee",
            linewidths=0.85,
        )
        ax.set_title(PROFILES[profile]["label"])
        row = row_by_profile[profile]
        ax.text(
            0.03,
            0.96,
            f"S={row['binary_shadow_fraction']:.3f}\nD={row['depleted_accessibility_mass_fraction']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.88),
        )

        ax2 = axes[1, col]
        im2 = ax2.imshow(
            losses[profile].T,
            origin="lower",
            extent=(0, lx * 1.0e3, 0, ly * 1.0e3),
            cmap=loss_cmap,
            vmin=0.0,
            vmax=max_loss,
            aspect="equal",
        )
        ax2.scatter(
            (ix + 0.5) * lx * 1.0e3 / reference.shape[0],
            (iy + 0.5) * ly * 1.0e3 / reference.shape[1],
            marker="s",
            s=20,
            facecolors="none",
            edgecolors="#22d3ee",
            linewidths=0.85,
        )
        ax2.text(
            0.03,
            0.96,
            f"TV={row['total_variation_distribution']:.3f}\nH={row['hellinger_distribution']:.3f}",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.88),
        )
        for axis in (ax, ax2):
            axis.add_patch(Rectangle((0, 0), lx * 1.0e3, ly * 1.0e3, fill=False, ec="#0f172a", lw=0.8))
            for grain in geometry["grains"]:
                axis.add_patch(Circle((float(grain["x"]) * 1.0e3, float(grain["y"]) * 1.0e3), float(grain["radius"]) * 1.0e3, fc="#020617", ec="none"))
            axis.set_xlabel("x (mm)")
            if col == 0:
                axis.set_ylabel("y (mm)")
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.88, hspace=0.26, wspace=0.16)
    cax = fig.add_axes([0.90, 0.56, 0.012, 0.27])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\log_2[(H_\chi+\delta)/(H_0+\delta)]$")
    cax2 = fig.add_axes([0.90, 0.16, 0.012, 0.27])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label(r"depleted accessibility, $\max(H_0-H_\chi,0)$")
    axes[0, 0].text(-0.16, 1.08, "a", transform=axes[0, 0].transAxes, fontsize=13, fontweight="bold")
    axes[1, 0].text(-0.16, 1.08, "b", transform=axes[1, 0].transAxes, fontsize=13, fontweight="bold")
    fig.suptitle("Continuous pore-shadow structure relative to no-DLVO transport", fontsize=14, fontweight="bold")
    path = FIGURES / "pore_shadow_continuous_maps.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "pore_shadow_continuous_maps.png", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_suite(
    rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = [str(row["profile"]) for row in rows]
    labels = [PROFILES[p]["label"] for p in profiles]
    colors = [PROFILES[p]["color"] for p in profiles]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), dpi=240)
    x = np.arange(len(profiles))
    width = 0.26
    axes[0, 0].bar(x - width, [float(r["binary_shadow_fraction"]) for r in rows], width, label="binary cell fraction", color="#94a3b8")
    axes[0, 0].bar(x, [float(r["net_shadow_loss_fraction"]) for r in rows], width, label="net shadow loss", color="#2563eb")
    axes[0, 0].bar(x + width, [float(r["total_variation_distribution"]) for r in rows], width, label="distribution TV", color="#7c3aed")
    axes[0, 0].set_ylabel("metric value")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 0].set_title(r"Shadow versus redistribution at $\epsilon=0.005$")

    axes[0, 1].bar(x - width / 2, [float(r["depleted_accessibility_mass_fraction"]) for r in rows], width, label="depleted", color="#2563eb")
    axes[0, 1].bar(x + width / 2, [float(r["enriched_accessibility_mass_fraction"]) for r in rows], width, label="enriched", color="#f97316")
    axes[0, 1].set_ylabel("accessibility mass fraction")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=18, ha="right")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].set_title("Losses balanced by gains?")

    epsilons = sorted({float(row["epsilon"]) for row in threshold_rows})
    for profile, color in zip(profiles, colors):
        profile_rows = [row for row in threshold_rows if row["profile"] == profile]
        profile_rows.sort(key=lambda row: float(row["epsilon"]))
        axes[1, 0].plot(epsilons, [float(row["binary_shadow_fraction"]) for row in profile_rows], marker="o", lw=2, color=color, label=PROFILES[profile]["label"])
        axes[1, 1].plot(epsilons, [float(row["net_shadow_loss_fraction"]) for row in profile_rows], marker="o", lw=2, color=color, label=PROFILES[profile]["label"])
    for ax in axes[1, :]:
        ax.set_xscale("log")
        ax.set_xlabel(r"reference support threshold $\epsilon$")
        ax.grid(True, color="#e5e7eb", lw=0.7)
    axes[1, 0].set_ylabel("binary shadow fraction")
    axes[1, 0].set_title("Thresholded shadow cells")
    axes[1, 1].set_ylabel("net shadow loss fraction")
    axes[1, 1].set_title("Threshold-robust net loss")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGURES / "pore_shadow_metric_suite.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "pore_shadow_metric_suite.png", bbox_inches="tight")
    plt.close(fig)
    return path


def write_table(rows: list[dict[str, object]]) -> Path:
    table_path = OUT / "pore_shadow_metric_table.tex"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Alternative pore-shadow metrics from the 100,000-particle random-packing diagnostic. The binary shadow fraction is the fraction of no-DLVO-supported cells with \(H_0>0.005\) and \(H_\chi\le0.005\). Depleted and enriched mass fractions are \(\sum\max(H_0-H_\chi,0)/\sum H_0\) and \(\sum\max(H_\chi-H_0,0)/\sum H_0\) over the same support. Net shadow loss is the positive part of the net accessibility deficit. TV is the total-variation distance between normalized pore-accessibility distributions over the union of visited cells.}",
        r"\label{tab:pore-shadow-alt-metrics}",
        r"\scriptsize",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"case & binary \(S_{0.005}\) & depleted & enriched & net loss & TV & shadow dominance & largest cluster \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            "{label} & {binary_shadow_fraction:.4f} & {depleted_accessibility_mass_fraction:.4f} & {enriched_accessibility_mass_fraction:.4f} & {net_shadow_loss_fraction:.4f} & {total_variation_distribution:.4f} & {shadow_dominance_fraction:.3f} & {largest_shadow_component_cells} \\\\".format(
                **row
            )
        )
    lines.extend([r"\hline", r"\end{tabular}", r"}", r"\end{table}", ""])
    table_path.write_text("\n".join(lines), encoding="utf-8")
    return table_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-path", type=Path, default=rpt.GEOMETRY_PATH)
    parser.add_argument("--flow-case", type=Path, default=None)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--transport-out", type=Path, default=TRANSPORT_OUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument("--particles", type=int, default=100000)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--pore-nx", type=int, default=96)
    parser.add_argument("--pore-ny", type=int, default=64)
    parser.add_argument("--surface-bins", type=int, default=36)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--tag", default="")
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES))
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--near-gap-um", type=float, default=25.0)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.02])
    return parser.parse_args()


def main() -> None:
    global OUT, FIGURES, TRANSPORT_OUT
    args = parse_args()
    OUT = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    FIGURES = args.figures_dir if args.figures_dir.is_absolute() else ROOT / args.figures_dir
    TRANSPORT_OUT = args.transport_out if args.transport_out.is_absolute() else ROOT / args.transport_out
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or default_tag(args)
    geometry_path = args.geometry_path if args.geometry_path.is_absolute() else ROOT / args.geometry_path
    flow_case = None if args.flow_case is None else (args.flow_case if args.flow_case.is_absolute() else ROOT / args.flow_case)
    geometry = rpt.load_geometry(
        geometry_path,
        flow_case=flow_case,
        apply_flow_shift=(flow_case is not None and not args.no_flow_origin_shift),
    )
    reference = load_accessibility("neutral_resolved", tag)
    xx, yy, gap, inside = cell_geometry(geometry, reference.shape[0], reference.shape[1])
    # Use the same raster-cell support convention as run_transport_shadow_diagnostic.py.
    # Some coarse cells overlap the curved solid boundary, so masking by the cell-center
    # solid indicator would make these summary metrics inconsistent with the pore support definition
    # shadow fraction. The geometry is still used for plotting and near-grain partitioning.
    fluid = np.ones_like(reference, dtype=bool)
    unknown = [profile for profile in args.profiles if profile not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}")
    conditions = {profile: load_accessibility(profile, tag) for profile in args.profiles}
    pseudocount = 0.5 / float(args.particles)
    near_gap_m = args.near_gap_um * 1.0e-6

    rows = [
        metrics_for_profile(profile, reference, condition, gap, fluid, args.epsilon, near_gap_m, pseudocount)
        for profile, condition in conditions.items()
    ]
    threshold_rows: list[dict[str, object]] = []
    for eps in args.epsilons:
        for profile, condition in conditions.items():
            threshold_rows.append(metrics_for_profile(profile, reference, condition, gap, fluid, eps, near_gap_m, pseudocount))

    metrics_path = OUT / "pore_shadow_metric_summary.csv"
    threshold_path = OUT / "pore_shadow_threshold_sweep.csv"
    write_csv(metrics_path, rows)
    write_csv(threshold_path, threshold_rows)
    table_path = write_table(rows)
    map_path = plot_maps(geometry, reference, conditions, rows, gap, inside, args.epsilon, pseudocount)
    suite_path = plot_metric_suite(rows, threshold_rows)
    print(metrics_path)
    print(threshold_path)
    print(table_path)
    print(map_path)
    print(suite_path)


if __name__ == "__main__":
    main()
