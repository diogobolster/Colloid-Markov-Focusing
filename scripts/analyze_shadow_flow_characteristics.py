#!/usr/bin/env python3
"""Flow-field characteristics of pore-shadow cells."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

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
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import cKDTree


OUT = ROOT / "outputs" / "shadow_flow_characteristics"
FIGURES = ROOT / "outputs" / "figures"
TRANSPORT_OUT = ROOT / "outputs" / "transport_shadow_diagnostic"


def run_tag(args: argparse.Namespace) -> str:
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


def nearest_geometry(geometry: dict, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    xs = (np.arange(nx) + 0.5) * lx / nx
    ys = (np.arange(ny) + 0.5) * ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    centers = np.array([[grain["x"], grain["y"]] for grain in geometry["grains"]], dtype=float)
    radii = np.array([grain["radius"] for grain in geometry["grains"]], dtype=float)
    gap = np.full((nx, ny), np.inf)
    grain_id = np.full((nx, ny), -1, dtype=int)
    angle = np.full((nx, ny), np.nan)
    inside = np.zeros((nx, ny), dtype=bool)
    for gid, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        dx = periodic_delta(xx - center[0], lx)
        dy = periodic_delta(yy - center[1], ly)
        dist = np.hypot(dx, dy)
        candidate_gap = dist - radius
        update = candidate_gap < gap
        gap[update] = candidate_gap[update]
        grain_id[update] = gid
        angle[update] = np.mod(np.arctan2(dy[update], dx[update]), 2.0 * np.pi)
        inside |= dist <= radius
    return xx, yy, gap, grain_id, angle, inside


def flow_interpolator(
    flow_case: Path,
    geometry: dict,
    target_m_per_day: float,
    rescale_flow: bool,
) -> rpt.PeriodicIDWFlow:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, areas = rpt.flow_io.parse_xml_vtu(vtu)
    if rescale_flow:
        velocity_xy, _, _ = rpt.rescale_velocity_to_target(velocity_xy, areas, target_m_per_day)
    return rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8)


def rasterize_velocity(interpolator: rpt.PeriodicIDWFlow, lx: float, ly: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(nx, dtype=float) + 0.5) * lx / nx
    ys = (np.arange(ny, dtype=float) + 0.5) * ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    ux, uy = interpolator.velocity_at(xx.ravel(), yy.ravel())
    return ux.reshape(nx, ny), uy.reshape(nx, ny)


def block_mean(field: np.ndarray, coarse_nx: int, coarse_ny: int) -> np.ndarray:
    nx, ny = field.shape
    if nx % coarse_nx != 0 or ny % coarse_ny != 0:
        raise ValueError("fine grid dimensions must be integer multiples of coarse grid dimensions")
    bx = nx // coarse_nx
    by = ny // coarse_ny
    return field.reshape(coarse_nx, bx, coarse_ny, by).mean(axis=(1, 3))


def flow_descriptors(geometry: dict, flow_case: Path, args: argparse.Namespace) -> dict[str, np.ndarray]:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    interpolator = flow_interpolator(
        flow_case,
        geometry,
        args.target_mean_velocity_m_per_day,
        not args.no_rescale_flow,
    )
    ux, uy = rasterize_velocity(interpolator, lx, ly, args.flow_nx, args.flow_ny)
    dx = lx / args.flow_nx
    dy = ly / args.flow_ny
    du_dx = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dx)
    du_dy = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dy)
    dv_dx = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dx)
    dv_dy = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dy)
    speed = np.hypot(ux, uy)
    vorticity = dv_dx - du_dy
    sxx = du_dx
    syy = dv_dy
    sxy = 0.5 * (du_dy + dv_dx)
    strain = np.sqrt(2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy))
    divergence = du_dx + dv_dy
    rotation_rate = np.abs(vorticity)
    okubo_weiss = strain * strain - vorticity * vorticity
    grad_speed_x = (np.roll(speed, -1, axis=0) - np.roll(speed, 1, axis=0)) / (2.0 * dx)
    grad_speed_y = (np.roll(speed, -1, axis=1) - np.roll(speed, 1, axis=1)) / (2.0 * dy)
    speed_gradient = np.hypot(grad_speed_x, grad_speed_y)
    curvature_proxy = np.hypot(ux * du_dx + uy * du_dy, ux * dv_dx + uy * dv_dy)
    raw = {
        "speed": speed,
        "vorticity_abs": rotation_rate,
        "vorticity_signed": vorticity,
        "strain_rate": strain,
        "divergence_abs": np.abs(divergence),
        "okubo_weiss": okubo_weiss,
        "speed_gradient": speed_gradient,
        "curvature_proxy": curvature_proxy,
    }
    return {name: block_mean(value, args.pore_nx, args.pore_ny) for name, value in raw.items()}


def percentile_of_distribution(values: np.ndarray, x: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(x):
        return float("nan")
    return float(np.mean(values <= x))


def summarize_features(
    descriptors: dict[str, np.ndarray],
    gap: np.ndarray,
    shadow: np.ndarray,
    support: np.ndarray,
    depletion: np.ndarray,
    reference: np.ndarray,
) -> list[dict[str, object]]:
    descriptor_fields = dict(descriptors)
    descriptor_fields["gap_to_grain_um"] = gap * 1.0e6
    rows: list[dict[str, object]] = []
    non_shadow = support & ~shadow
    support_mass = float(np.sum(reference[support]))
    shadow_mass = float(np.sum(reference[shadow]))
    for name, field in descriptor_fields.items():
        all_values = field[support]
        shadow_values = field[shadow]
        nons_values = field[non_shadow]
        weight = depletion[shadow]
        weighted_shadow = float(np.average(shadow_values, weights=weight)) if shadow_values.size and np.sum(weight) > 0.0 else float("nan")
        support_median = float(np.median(all_values)) if all_values.size else float("nan")
        shadow_median = float(np.median(shadow_values)) if shadow_values.size else float("nan")
        rows.append(
            {
                "feature": name,
                "support_median": support_median,
                "shadow_median": shadow_median,
                "nonshadow_median": float(np.median(nons_values)) if nons_values.size else float("nan"),
                "shadow_mean": float(np.mean(shadow_values)) if shadow_values.size else float("nan"),
                "shadow_depletion_weighted_mean": weighted_shadow,
                "shadow_median_percentile_within_support": percentile_of_distribution(all_values, shadow_median),
                "shadow_to_support_median_ratio": shadow_median / support_median if support_median != 0.0 else float("nan"),
                "support_q10": float(np.quantile(all_values, 0.10)) if all_values.size else float("nan"),
                "support_q90": float(np.quantile(all_values, 0.90)) if all_values.size else float("nan"),
                "shadow_cell_count": int(np.sum(shadow)),
                "support_cell_count": int(np.sum(support)),
                "shadow_reference_mass_fraction": shadow_mass / support_mass if support_mass > 0.0 else float("nan"),
            }
        )
    return rows


def add_region(rows: list[dict[str, object]], region: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        tagged = dict(row)
        tagged["region"] = region
        out.append(tagged)
    return out


def top_depletion_mask(depletion: np.ndarray, support: np.ndarray, count: int) -> np.ndarray:
    mask = np.zeros_like(support, dtype=bool)
    if count <= 0:
        return mask
    values = depletion[support]
    positive = values[values > 0.0]
    if positive.size == 0:
        return mask
    count = min(count, positive.size)
    threshold = np.partition(positive, -count)[-count]
    mask = support & (depletion >= threshold) & (depletion > 0.0)
    # Ties are possible on the occupancy grid. Keep them; the report records the
    # actual count.
    return mask


def nearest_stagnation_distance(
    speed: np.ndarray,
    gap: np.ndarray,
    support: np.ndarray,
    threshold_quantile: float = 0.03,
) -> np.ndarray:
    candidates = support & (gap < 60.0e-6)
    speeds = speed[candidates]
    if speeds.size == 0:
        return np.full(speed.shape, np.nan)
    threshold = float(np.quantile(speeds, threshold_quantile))
    low = candidates & (speed <= threshold)
    coords = np.column_stack(np.where(low))
    query = np.column_stack(np.where(support))
    distance = np.full(speed.shape, np.nan)
    if coords.size == 0 or query.size == 0:
        return distance
    tree = cKDTree(coords)
    dist_cells, _ = tree.query(query)
    distance[query[:, 0], query[:, 1]] = dist_cells
    return distance


def plot_feature_maps(
    geometry: dict,
    descriptors: dict[str, np.ndarray],
    reference: np.ndarray,
    condition: np.ndarray,
    shadow: np.ndarray,
    support: np.ndarray,
    gap: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    fields = [
        ("speed", "speed (m/s)", "viridis"),
        ("strain_rate", r"strain rate (s$^{-1}$)", "magma"),
        ("vorticity_abs", r"$|\omega|$ (s$^{-1}$)", "magma"),
        ("okubo_weiss", r"strain$^2-\omega^2$ (s$^{-2}$)", "coolwarm"),
        ("gap_to_grain_um", "gap to nearest grain (um)", "cividis"),
        ("depletion", r"$\max(H_0-H_\chi,0)$", "viridis"),
    ]
    extra = {
        "gap_to_grain_um": gap * 1.0e6,
        "depletion": np.maximum(reference - condition, 0.0),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), dpi=240)
    ix, iy = np.where(shadow)
    for ax, (name, title, cmap_name) in zip(axes.flat, fields, strict=True):
        field = extra[name] if name in extra else descriptors[name]
        masked = np.where(support, field, np.nan)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#f8fafc")
        finite = masked[np.isfinite(masked)]
        if finite.size and name != "okubo_weiss":
            vmin, vmax = np.quantile(finite, [0.02, 0.98])
        elif finite.size:
            lim = float(np.quantile(np.abs(finite), 0.98))
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(masked.T, origin="lower", extent=(0, lx * 1.0e3, 0, ly * 1.0e3), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.scatter(
            (ix + 0.5) * lx * 1.0e3 / reference.shape[0],
            (iy + 0.5) * ly * 1.0e3 / reference.shape[1],
            marker="s",
            s=16,
            facecolors="none",
            edgecolors="#22d3ee",
            linewidths=0.75,
        )
        for grain in geometry["grains"]:
            ax.add_patch(Circle((float(grain["x"]) * 1.0e3, float(grain["y"]) * 1.0e3), float(grain["radius"]) * 1.0e3, fc="#020617", ec="none"))
        ax.add_patch(Rectangle((0, 0), lx * 1.0e3, ly * 1.0e3, fill=False, ec="#0f172a", lw=0.7))
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.015)
        cbar.ax.tick_params(labelsize=7)
    for label, ax in zip("abcdef", axes.flat, strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.suptitle("Flow descriptors at 100 mM pore-shadow cells", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = FIGURES / "shadow_flow_characteristic_maps.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "shadow_flow_characteristic_maps.png", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_summary(rows: list[dict[str, object]]) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    names = [
        "speed",
        "strain_rate",
        "vorticity_abs",
        "okubo_weiss",
        "speed_gradient",
        "curvature_proxy",
        "gap_to_grain_um",
        "distance_to_low_speed_near_grain_cells",
    ]
    row_by_name = {str(row["feature"]): row for row in rows}
    labels = [
        "speed",
        "strain",
        "|vorticity|",
        "OW",
        "|grad speed|",
        "curvature",
        "gap",
        "low-speed dist.",
    ]
    percentiles = [float(row_by_name[name]["shadow_median_percentile_within_support"]) for name in names]
    ratios = [float(row_by_name[name]["shadow_to_support_median_ratio"]) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=240)
    x = np.arange(len(names))
    axes[0].bar(x, percentiles, color="#2563eb")
    axes[0].axhline(0.5, color="#64748b", lw=1.0, ls="--")
    axes[0].set_ylabel("shadow median percentile in support")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Are shadow cells unusual?")
    axes[1].bar(x, ratios, color="#7c3aed")
    axes[1].axhline(1.0, color="#64748b", lw=1.0, ls="--")
    axes[1].set_ylabel("shadow/support median ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_title("Magnitude relative to support median")
    for label, ax in zip("ab", axes, strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.7)
    fig.tight_layout()
    path = FIGURES / "shadow_flow_feature_summary.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "shadow_flow_feature_summary.png", bbox_inches="tight")
    plt.close(fig)
    return path


def quantile_bin_rows(
    bin_type: str,
    field: np.ndarray,
    support: np.ndarray,
    reference: np.ndarray,
    condition: np.ndarray,
    depletion: np.ndarray,
    enrichment: np.ndarray,
    shadow: np.ndarray,
    bins: int = 5,
) -> list[dict[str, object]]:
    values = field[support]
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1.0e-99
    edges[-1] += 1.0e-99
    total_ref = float(np.sum(reference[support]))
    rows: list[dict[str, object]] = []
    for idx in range(bins):
        mask = support & (field >= edges[idx]) & (field < edges[idx + 1])
        ref_mass = float(np.sum(reference[mask]))
        cond_mass = float(np.sum(condition[mask]))
        rows.append(
            {
                "bin_type": bin_type,
                "bin_index": idx + 1,
                "lower": float(edges[idx]),
                "upper": float(edges[idx + 1]),
                "cell_count": int(np.sum(mask)),
                "shadow_count": int(np.sum(mask & shadow)),
                "reference_mass_fraction": ref_mass / total_ref if total_ref > 0.0 else float("nan"),
                "condition_mass_fraction": cond_mass / total_ref if total_ref > 0.0 else float("nan"),
                "net_change_fraction": (cond_mass - ref_mass) / total_ref if total_ref > 0.0 else float("nan"),
                "depleted_mass_fraction": float(np.sum(depletion[mask]) / total_ref) if total_ref > 0.0 else float("nan"),
                "enriched_mass_fraction": float(np.sum(enrichment[mask]) / total_ref) if total_ref > 0.0 else float("nan"),
                "median_field": float(np.median(field[mask])) if np.any(mask) else float("nan"),
                "median_reference_accessibility": float(np.median(reference[mask])) if np.any(mask) else float("nan"),
                "median_condition_accessibility": float(np.median(condition[mask])) if np.any(mask) else float("nan"),
            }
        )
    return rows


def plot_mechanism_decomposition(
    descriptors: dict[str, np.ndarray],
    reference: np.ndarray,
    condition: np.ndarray,
    support: np.ndarray,
    shadow: np.ndarray,
    top_depletion: np.ndarray,
    bin_rows: list[dict[str, object]],
    epsilon: float,
) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    depletion = np.maximum(reference - condition, 0.0)
    speed = descriptors["speed"]
    support_speed = speed[support]
    speed_percentile = np.zeros_like(speed, dtype=float)
    if support_speed.size:
        sorted_speed = np.sort(support_speed)
        speed_percentile[support] = np.searchsorted(sorted_speed, speed[support], side="right") / sorted_speed.size

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), dpi=240)
    ax = axes[0, 0]
    sc = ax.scatter(
        reference[support],
        condition[support],
        c=speed_percentile[support],
        s=12,
        cmap="viridis",
        alpha=0.65,
        linewidths=0,
    )
    ax.scatter(reference[shadow], condition[shadow], s=42, marker="s", facecolors="none", edgecolors="#06b6d4", linewidths=1.2, label="threshold shadows")
    ax.scatter(reference[top_depletion], condition[top_depletion], s=38, marker="o", facecolors="none", edgecolors="#f97316", linewidths=1.1, label="largest depletions")
    limit_low = max(1.0e-4, min(float(np.min(reference[support])), float(np.min(condition[support & (condition > 0.0)]))))
    limit_high = max(float(np.max(reference[support])), float(np.max(condition[support])))
    ax.plot([limit_low, limit_high], [limit_low, limit_high], color="#334155", lw=1.0, ls="--")
    ax.axvline(epsilon, color="#64748b", lw=0.9, ls=":")
    ax.axhline(epsilon, color="#64748b", lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"no-DLVO accessibility $H_0$")
    ax.set_ylabel(r"100 mM accessibility $H_\chi$")
    ax.set_title("Threshold shadows vs continuous depletion")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    cbar = fig.colorbar(sc, ax=ax, pad=0.012)
    cbar.set_label("speed percentile")

    def rows_for(kind: str) -> list[dict[str, object]]:
        return [row for row in bin_rows if row["bin_type"] == kind]

    def bar_panel(ax: plt.Axes, rows: list[dict[str, object]], title: str, xlabel: str) -> None:
        x = np.arange(len(rows))
        net = np.array([float(row["net_change_fraction"]) for row in rows])
        deplete = np.array([float(row["depleted_mass_fraction"]) for row in rows])
        shadow_counts = np.array([int(row["shadow_count"]) for row in rows])
        ax.bar(x, -deplete, color="#2563eb", alpha=0.85, label="depleted mass")
        ax.bar(x, np.array([float(row["enriched_mass_fraction"]) for row in rows]), color="#f97316", alpha=0.75, label="enriched mass")
        ax.plot(x, net, color="#111827", marker="o", lw=1.5, label="net change")
        ax.axhline(0.0, color="#475569", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(len(rows))])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fraction of total reference mass")
        ax.set_title(title)
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.7)
        twin = ax.twinx()
        twin.plot(x, shadow_counts, color="#06b6d4", marker="s", lw=1.3, label="threshold-shadow cells")
        twin.set_ylabel("threshold-shadow cells", color="#0891b2")
        twin.tick_params(axis="y", labelcolor="#0891b2")

    bar_panel(axes[0, 1], rows_for("speed_quintile"), "Mass change by speed quintile", "speed quintile")
    bar_panel(axes[1, 0], rows_for("reference_accessibility_quintile"), r"Mass change by $H_0$ quintile", r"$H_0$ quintile")

    ax = axes[1, 1]
    x = np.arange(5)
    speed_rows = rows_for("speed_quintile")
    h0_rows = rows_for("reference_accessibility_quintile")
    ax.plot(x, np.cumsum([float(row["depleted_mass_fraction"]) for row in speed_rows]), marker="o", lw=2.0, label="depletion by speed")
    ax.plot(x, np.cumsum([float(row["depleted_mass_fraction"]) for row in h0_rows]), marker="s", lw=2.0, label=r"depletion by $H_0$")
    ax.plot(x, np.cumsum([float(row["reference_mass_fraction"]) for row in speed_rows]), marker="o", lw=1.2, ls="--", color="#94a3b8", label="reference mass by speed")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(5)])
    ax.set_xlabel("quintile accumulated low to high")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Where depletion accumulates")
    ax.grid(True, color="#e5e7eb", lw=0.7)
    ax.legend(frameon=False, fontsize=8)

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGURES / "shadow_mechanism_decomposition.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "shadow_mechanism_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(rows: list[dict[str, object]], map_path: Path, summary_path: Path, args: argparse.Namespace) -> Path:
    path = OUT / "shadow_flow_characteristics_report.md"
    lines = [
        "# Shadow Flow Characteristics",
        "",
        f"Profile: `{args.profile}` relative to no-DLVO reference.",
        f"Shadow threshold: H0 > {args.epsilon:g}, Hchi <= {args.epsilon:g}.",
        "",
        "| feature | support median | shadow median | percentile | ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {support_median:.6g} | {shadow_median:.6g} | {shadow_median_percentile_within_support:.3f} | {shadow_to_support_median_ratio:.3f} |".format(
                **row
            )
        )
    lines.extend(["", f"Map figure: `{map_path.relative_to(ROOT)}`", f"Summary figure: `{summary_path.relative_to(ROOT)}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="unfavorable_100mM_z70")
    parser.add_argument("--particles", type=int, default=100000)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--pore-nx", type=int, default=96)
    parser.add_argument("--pore-ny", type=int, default=64)
    parser.add_argument("--flow-nx", type=int, default=384)
    parser.add_argument("--flow-ny", type=int, default=256)
    parser.add_argument("--surface-bins", type=int, default=36)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--tag", default="")
    parser.add_argument("--flow-case", type=Path, default=rpt.DEFAULT_FLOW_CASE)
    parser.add_argument("--geometry-path", type=Path, default=rpt.GEOMETRY_PATH)
    parser.add_argument("--transport-out", type=Path, default=TRANSPORT_OUT)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument("--target-mean-velocity-m-per-day", type=float, default=rpt.TARGET_M_PER_DAY)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--no-rescale-flow", action="store_true")
    return parser.parse_args()


def main() -> None:
    global OUT, FIGURES, TRANSPORT_OUT
    args = parse_args()
    OUT = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    FIGURES = args.figures_dir if args.figures_dir.is_absolute() else ROOT / args.figures_dir
    TRANSPORT_OUT = args.transport_out if args.transport_out.is_absolute() else ROOT / args.transport_out
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_tag(args)
    flow_case = args.flow_case if args.flow_case.is_absolute() else ROOT / args.flow_case
    geometry_path = args.geometry_path if args.geometry_path.is_absolute() else ROOT / args.geometry_path
    geometry = rpt.load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=not args.no_flow_origin_shift)
    reference = load_accessibility("neutral_resolved", tag)
    condition = load_accessibility(args.profile, tag)
    descriptors = flow_descriptors(geometry, flow_case, args)
    xx, yy, gap, grain_id, angle, inside = nearest_geometry(geometry, args.pore_nx, args.pore_ny)
    support = reference > args.epsilon
    shadow = support & (condition <= args.epsilon)
    depletion = np.maximum(reference - condition, 0.0)
    descriptors["distance_to_low_speed_near_grain_cells"] = nearest_stagnation_distance(
        descriptors["speed"],
        gap,
        support,
    )
    rows = summarize_features(descriptors, gap, shadow, support, depletion, reference)
    top_depletion = top_depletion_mask(depletion, support, int(np.sum(shadow)))
    top_rows = summarize_features(descriptors, gap, top_depletion, support, depletion, reference)
    region_rows = add_region(rows, "binary_threshold_shadow") + add_region(top_rows, "top_depletion_same_count")
    bin_rows = (
        quantile_bin_rows("speed_quintile", descriptors["speed"], support, reference, condition, depletion, np.maximum(condition - reference, 0.0), shadow)
        + quantile_bin_rows("reference_accessibility_quintile", reference, support, reference, condition, depletion, np.maximum(condition - reference, 0.0), shadow)
    )
    write_csv(OUT / "shadow_flow_feature_summary.csv", rows)
    write_csv(OUT / "shadow_flow_feature_summary_by_region.csv", region_rows)
    write_csv(OUT / "shadow_mechanism_bins.csv", bin_rows)
    map_path = plot_feature_maps(geometry, descriptors, reference, condition, shadow, support, gap, args)
    summary_path = plot_feature_summary(rows)
    decomposition_path = plot_mechanism_decomposition(descriptors, reference, condition, support, shadow, top_depletion, bin_rows, args.epsilon)
    report = write_report(rows, map_path, summary_path, args)
    print(OUT / "shadow_flow_feature_summary.csv")
    print(OUT / "shadow_flow_feature_summary_by_region.csv")
    print(OUT / "shadow_mechanism_bins.csv")
    print(report)
    print(map_path)
    print(summary_path)
    print(decomposition_path)


if __name__ == "__main__":
    main()
