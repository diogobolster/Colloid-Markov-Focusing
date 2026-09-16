#!/usr/bin/env python3
"""Targeted transport-shadow simulations in the random OpenFOAM packing."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

from colloid_tsm.compiled import RandomFlowGrid, build_kernel, simulate_random_occupancy_compiled


OUT = ROOT / "outputs" / "transport_shadow_diagnostic"
FIGURES = ROOT / "outputs" / "figures"

PROFILES = {
    "neutral_resolved": {
        "label": "No DLVO",
        "condition": "no_dlvo",
        "updates": {},
        "color": "#6b7280",
    },
    "unfavorable_50mM_z70": {
        "label": "50 mM unfavorable",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 50.0e-3, "zeta_collector_unfavorable": -70.0e-3},
        "color": "#2563eb",
    },
    "unfavorable_100mM_z70": {
        "label": "100 mM unfavorable",
        "condition": "unfavorable",
        "updates": {"ionic_strength_molar": 100.0e-3, "zeta_collector_unfavorable": -70.0e-3},
        "color": "#581c87",
    },
    "unfavorable_50mM_z70_100xD": {
        "label": "50 mM unfavorable, 100D",
        "condition": "unfavorable",
        "updates": {
            "ionic_strength_molar": 50.0e-3,
            "zeta_collector_unfavorable": -70.0e-3,
            "diffusivity_multiplier": 100.0,
        },
        "color": "#0f766e",
    },
}


def run_tag(args: argparse.Namespace) -> str:
    return (
        f"n{args.particles}_t{args.max_time:g}_dt{args.dt:g}_"
        f"p{args.pore_nx}x{args.pore_ny}_s{args.surface_bins}_stride{args.sample_stride}_seed{args.seed}"
    ).replace(".", "p")


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


def flow_interpolator(
    flow_case: Path,
    geometry: dict,
    target_m_per_day: float,
    rescale_flow: bool,
) -> tuple[rpt.PeriodicIDWFlow, float, float]:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(flow_case)
    centers_xy, velocity_xy, areas = rpt.flow_io.parse_xml_vtu(vtu)
    mean_before = float("nan")
    scale = 1.0
    if rescale_flow:
        velocity_xy, mean_before, scale = rpt.rescale_velocity_to_target(velocity_xy, areas, target_m_per_day)
    return rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8), mean_before, scale


def rasterize_flow_grid(interpolator: rpt.PeriodicIDWFlow, lx: float, ly: float, nx: int, ny: int) -> RandomFlowGrid:
    xs = (np.arange(nx, dtype=float) + 0.5) * lx / nx
    ys = (np.arange(ny, dtype=float) + 0.5) * ly / ny
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    ux, uy = interpolator.velocity_at(xx.ravel(), yy.ravel())
    return RandomFlowGrid(
        lx=lx,
        ly=ly,
        ux=np.ascontiguousarray(ux.reshape(nx, ny)),
        uy=np.ascontiguousarray(uy.reshape(nx, ny)),
    )


def profile_params(profile: str, args: argparse.Namespace) -> rpt.RandomTrackingParams:
    base = rpt.RandomTrackingParams(
        dt=args.dt,
        max_time=args.max_time,
        adaptive_near_wall=True,
        adaptive_cutoff=args.adaptive_cutoff_nm * 1.0e-9,
        normal_step_target=args.normal_step_target_nm * 1.0e-9,
        max_substeps=args.max_substeps,
        segment_collision=True,
    )
    return replace(base, **PROFILES[profile]["updates"])


def effective_area(accessibility: np.ndarray, cell_area: float) -> tuple[float, float]:
    values = accessibility[np.isfinite(accessibility) & (accessibility > 0.0)]
    if values.size == 0:
        return float("nan"), float("nan")
    p = values / np.sum(values)
    entropy_area = float(cell_area * math.exp(-np.sum(p * np.log(p))))
    ipr_area = float(cell_area / np.sum(p * p))
    return entropy_area, ipr_area


def shadow_fraction(reference: np.ndarray, condition: np.ndarray, epsilon: float) -> tuple[float, int, int]:
    support = reference > epsilon
    if not np.any(support):
        return float("nan"), 0, 0
    shadow = support & (condition <= epsilon)
    return float(np.sum(shadow) / np.sum(support)), int(np.sum(shadow)), int(np.sum(support))


def mark_occupancy(
    pore_seen: np.ndarray,
    surface_seen: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    active_or_done: np.ndarray,
    disks: rpt.PeriodicDiskGeometry,
    params: rpt.RandomTrackingParams,
    pore_nx: int,
    pore_ny: int,
    surface_bins: int,
) -> None:
    idx = np.flatnonzero(active_or_done)
    if idx.size == 0:
        return
    lx = disks.lx
    ly = disks.ly
    xm = np.mod(x[idx], lx)
    ym = np.mod(y[idx], ly)
    ix = np.clip((xm / lx * pore_nx).astype(int), 0, pore_nx - 1)
    iy = np.clip((ym / ly * pore_ny).astype(int), 0, pore_ny - 1)
    pore_seen[idx, ix, iy] = True

    gap, _, _, _, grain = disks.nearest_surface(xm, ym)
    near = gap <= params.near_surface
    if not np.any(near):
        return
    near_idx = idx[near]
    grain_near = grain[near]
    centers = disks.centers[grain_near]
    dx = rpt.periodic_delta(xm[near] - centers[:, 0], lx)
    dy = rpt.periodic_delta(ym[near] - centers[:, 1], ly)
    theta = np.mod(np.arctan2(dy, dx), 2.0 * math.pi)
    sector = np.clip((theta / (2.0 * math.pi) * surface_bins).astype(int), 0, surface_bins - 1)
    surface_seen[near_idx, grain_near, sector] = True


def simulate_profile(
    profile: str,
    flow: RandomFlowGrid,
    geometry: dict,
    initial_y: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    params = profile_params(profile, args)
    condition = str(PROFILES[profile]["condition"])
    disks = rpt.PeriodicDiskGeometry(geometry, params.particle_radius)
    n_particles = int(initial_y.size)
    result = simulate_random_occupancy_compiled(
        flow,
        geometry,
        params.as_physical_params(),
        condition,
        n_particles=n_particles,
        seed=seed,
        allow_attachment=condition == "favorable",
        initial_x=np.full(n_particles, args.inlet_x_um * 1.0e-6, dtype=float),
        initial_y=initial_y,
        pore_nx=args.pore_nx,
        pore_ny=args.pore_ny,
        surface_bins=args.surface_bins,
        sample_stride=args.sample_stride,
    )
    pore_access = result["pore_counts"] / float(n_particles)
    surface_access = result["surface_counts"] / float(n_particles)
    exited = result["exited"]
    attached = result["attached"]
    censored = result["censored"]
    interceptions = result["interceptions"]
    near_time = result["near_time"]
    h_min = result["h_min"]
    cell_area = disks.lx * disks.ly / (args.pore_nx * args.pore_ny)
    eff_entropy, eff_ipr = effective_area(pore_access, cell_area)
    surface_positive = surface_access[surface_access > 0.0]
    surface_p = surface_positive / np.sum(surface_positive) if surface_positive.size else np.array([])
    surface_eff = float(math.exp(-np.sum(surface_p * np.log(surface_p)))) if surface_p.size else float("nan")
    row = {
        "profile": profile,
        "label": PROFILES[profile]["label"],
        "condition": condition,
        "particles": n_particles,
        "seed": seed,
        "max_time_s": params.max_time,
        "dt_s": params.dt,
        "sample_stride": args.sample_stride,
        "pore_grid_nx": args.pore_nx,
        "pore_grid_ny": args.pore_ny,
        "surface_bins": args.surface_bins,
        "exited": int(np.sum(exited)),
        "attached": int(np.sum(attached)),
        "censored": int(np.sum(censored)),
        "intercepted": int(np.sum(interceptions > 0)),
        "intercepted_fraction": float(np.mean(interceptions > 0)),
        "mean_interceptions": float(np.mean(interceptions)),
        "median_near_time_intercepted_s": float(np.median(near_time[interceptions > 0])) if np.any(interceptions > 0) else 0.0,
        "median_h_min_nm": float(np.median(h_min[np.isfinite(h_min)]) * 1.0e9),
        "pore_cells_visited": int(np.sum(pore_access > 0.0)),
        "pore_cells_visited_fraction": float(np.mean(pore_access > 0.0)),
        "pore_effective_area_entropy_m2": eff_entropy,
        "pore_effective_area_ipr_m2": eff_ipr,
        "pore_effective_fraction_entropy": float(eff_entropy / (disks.lx * disks.ly)),
        "pore_effective_fraction_ipr": float(eff_ipr / (disks.lx * disks.ly)),
        "surface_sectors_visited": int(np.sum(surface_access > 0.0)),
        "surface_sectors_visited_fraction": float(np.mean(surface_access > 0.0)),
        "surface_effective_sectors": surface_eff,
    }
    return row, pore_access, surface_access


def summarize_aggregate(
    profile: str,
    params: rpt.RandomTrackingParams,
    disks: rpt.PeriodicDiskGeometry,
    n_particles: int,
    seed: int,
    pore_counts: np.ndarray,
    surface_counts: np.ndarray,
    exited: np.ndarray,
    attached: np.ndarray,
    censored: np.ndarray,
    interceptions: np.ndarray,
    near_time: np.ndarray,
    h_min: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    pore_access = pore_counts / float(n_particles)
    surface_access = surface_counts / float(n_particles)
    cell_area = disks.lx * disks.ly / (args.pore_nx * args.pore_ny)
    eff_entropy, eff_ipr = effective_area(pore_access, cell_area)
    surface_positive = surface_access[surface_access > 0.0]
    surface_p = surface_positive / np.sum(surface_positive) if surface_positive.size else np.array([])
    surface_eff = float(math.exp(-np.sum(surface_p * np.log(surface_p)))) if surface_p.size else float("nan")
    row = {
        "profile": profile,
        "label": PROFILES[profile]["label"],
        "condition": str(PROFILES[profile]["condition"]),
        "particles": n_particles,
        "seed": seed,
        "max_time_s": params.max_time,
        "dt_s": params.dt,
        "sample_stride": args.sample_stride,
        "pore_grid_nx": args.pore_nx,
        "pore_grid_ny": args.pore_ny,
        "surface_bins": args.surface_bins,
        "exited": int(np.sum(exited)),
        "attached": int(np.sum(attached)),
        "censored": int(np.sum(censored)),
        "intercepted": int(np.sum(interceptions > 0)),
        "intercepted_fraction": float(np.mean(interceptions > 0)),
        "mean_interceptions": float(np.mean(interceptions)),
        "median_near_time_intercepted_s": float(np.median(near_time[interceptions > 0])) if np.any(interceptions > 0) else 0.0,
        "median_h_min_nm": float(np.median(h_min[np.isfinite(h_min)]) * 1.0e9),
        "pore_cells_visited": int(np.sum(pore_access > 0.0)),
        "pore_cells_visited_fraction": float(np.mean(pore_access > 0.0)),
        "pore_effective_area_entropy_m2": eff_entropy,
        "pore_effective_area_ipr_m2": eff_ipr,
        "pore_effective_fraction_entropy": float(eff_entropy / (disks.lx * disks.ly)),
        "pore_effective_fraction_ipr": float(eff_ipr / (disks.lx * disks.ly)),
        "surface_sectors_visited": int(np.sum(surface_access > 0.0)),
        "surface_sectors_visited_fraction": float(np.mean(surface_access > 0.0)),
        "surface_effective_sectors": surface_eff,
    }
    return row, pore_access, surface_access


def simulate_profile_chunked(
    profile: str,
    flow: RandomFlowGrid,
    geometry: dict,
    initial_y: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    if args.chunk_size <= 0 or args.chunk_size >= initial_y.size:
        return simulate_profile(profile, flow, geometry, initial_y, seed, args)

    params = profile_params(profile, args)
    condition = str(PROFILES[profile]["condition"])
    disks = rpt.PeriodicDiskGeometry(geometry, params.particle_radius)
    n_particles = int(initial_y.size)
    n_chunks = int(math.ceil(n_particles / args.chunk_size))
    chunk_dir = OUT / "chunks" / run_tag(args)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pore_counts = np.zeros((args.pore_nx, args.pore_ny), dtype=float)
    surface_counts = np.zeros((len(disks.radii), args.surface_bins), dtype=float)
    exited_parts: list[np.ndarray] = []
    attached_parts: list[np.ndarray] = []
    censored_parts: list[np.ndarray] = []
    interceptions_parts: list[np.ndarray] = []
    near_time_parts: list[np.ndarray] = []
    h_min_parts: list[np.ndarray] = []

    for chunk_id in range(n_chunks):
        start = chunk_id * args.chunk_size
        stop = min(n_particles, start + args.chunk_size)
        path = chunk_dir / f"{profile}_chunk{chunk_id:04d}of{n_chunks:04d}.npz"
        if args.resume and path.exists():
            data = np.load(path)
            result = {
                "pore_counts": data["pore_counts"],
                "surface_counts": data["surface_counts"],
                "attached": data["attached"].astype(bool),
                "exited": data["exited"].astype(bool),
                "censored": data["censored"].astype(bool),
                "interceptions": data["interceptions"].astype(int),
                "near_time": data["near_time"],
                "h_min": data["h_min"],
            }
            print(f"  reused {profile} chunk {chunk_id + 1}/{n_chunks}", flush=True)
        else:
            result = simulate_random_occupancy_compiled(
                flow,
                geometry,
                params.as_physical_params(),
                condition,
                n_particles=stop - start,
                seed=seed + 1_000_003 * chunk_id,
                allow_attachment=condition == "favorable",
                initial_x=np.full(stop - start, args.inlet_x_um * 1.0e-6, dtype=float),
                initial_y=initial_y[start:stop],
                pore_nx=args.pore_nx,
                pore_ny=args.pore_ny,
                surface_bins=args.surface_bins,
                sample_stride=args.sample_stride,
            )
            np.savez_compressed(
                path,
                pore_counts=result["pore_counts"],
                surface_counts=result["surface_counts"],
                attached=result["attached"].astype(np.uint8),
                exited=result["exited"].astype(np.uint8),
                censored=result["censored"].astype(np.uint8),
                interceptions=result["interceptions"],
                near_time=result["near_time"],
                h_min=result["h_min"],
            )
            print(f"  completed {profile} chunk {chunk_id + 1}/{n_chunks}", flush=True)
        pore_counts += result["pore_counts"]
        surface_counts += result["surface_counts"]
        exited_parts.append(result["exited"])
        attached_parts.append(result["attached"])
        censored_parts.append(result["censored"])
        interceptions_parts.append(result["interceptions"])
        near_time_parts.append(result["near_time"])
        h_min_parts.append(result["h_min"])

    return summarize_aggregate(
        profile,
        params,
        disks,
        n_particles,
        seed,
        pore_counts,
        surface_counts,
        np.concatenate(exited_parts),
        np.concatenate(attached_parts),
        np.concatenate(censored_parts),
        np.concatenate(interceptions_parts),
        np.concatenate(near_time_parts),
        np.concatenate(h_min_parts),
        args,
    )


def plot_maps(
    geometry: dict,
    summaries: list[dict[str, object]],
    pore: dict[str, np.ndarray],
    surface: dict[str, np.ndarray],
    eps: float,
) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    grains = geometry["grains"]
    reference = pore["neutral_resolved"]
    compare = [profile for profile in ("unfavorable_50mM_z70", "unfavorable_100mM_z70", "unfavorable_50mM_z70_100xD") if profile in pore]
    if not compare:
        raise ValueError("At least one non-reference profile is required for transport-shadow maps.")
    ncols = len(compare)
    fig, axes = plt.subplots(3, ncols, figsize=(4.8 * ncols + 0.8, 10.2), dpi=220, squeeze=False)
    ratio_cmap = plt.get_cmap("magma").copy()
    ratio_cmap.set_bad("#f1f5f9")
    for col, profile in enumerate(compare):
        ratio = np.divide(pore[profile], reference, out=np.full_like(reference, np.nan), where=reference > eps)
        shadow_mask = (reference > eps) & (pore[profile] <= eps)
        shadow_ix, shadow_iy = np.where(shadow_mask)
        shadow_fraction_value = float(shadow_ix.size / np.sum(reference > eps)) if np.any(reference > eps) else float("nan")
        ax = axes[0, col]
        im = ax.imshow(
            np.clip(ratio.T, 0.0, 2.0),
            origin="lower",
            extent=(0, lx * 1e3, 0, ly * 1e3),
            cmap=ratio_cmap,
            vmin=0,
            vmax=2,
            aspect="equal",
        )
        ax.add_patch(Rectangle((0, 0), lx * 1e3, ly * 1e3, fill=False, ec="#111827", lw=0.8))
        for grain in grains:
            ax.add_patch(Circle((grain["x"] * 1e3, grain["y"] * 1e3), grain["radius"] * 1e3, fc="#020617", ec="none"))
        if shadow_ix.size:
            ax.scatter(
                (shadow_ix + 0.5) * lx * 1e3 / reference.shape[0],
                (shadow_iy + 0.5) * ly * 1e3 / reference.shape[1],
                marker="s",
                s=22,
                facecolors="none",
                edgecolors="#22d3ee",
                linewidths=1.0,
            )
        ax.text(
            0.03,
            0.96,
            rf"$S_{{{eps:g}}}={shadow_fraction_value:.3f}$" + f"\n{shadow_ix.size} cells",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.88),
        )
        ax.set_title(f"{PROFILES[profile]['label']}\n$pore H_\\chi/H_0$")
        ax.set_xlabel("x (mm)")
        if col == 0:
            ax.set_ylabel("y (mm)")
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.07, top=0.88, hspace=0.54, wspace=0.24)
    cax = fig.add_axes([0.89, 0.705, 0.018, 0.18])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("occupancy ratio")

    epsilons = [0.002, 0.005, 0.01, 0.02]
    x = np.arange(len(epsilons))
    for col, profile in enumerate(compare):
        ax = axes[1, col]
        pore_shadow = [shadow_fraction(reference, pore[profile], e)[0] for e in epsilons]
        surf_shadow = [shadow_fraction(surface["neutral_resolved"], surface[profile], e)[0] for e in epsilons]
        ax.plot(x, pore_shadow, marker="o", lw=2.0, label="pore cells", color="#2563eb")
        ax.plot(x, surf_shadow, marker="s", lw=2.0, label="surface sectors", color="#7c3aed")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:g}" for e in epsilons])
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"threshold $\epsilon$")
        if col == 0:
            ax.set_ylabel(r"$S_\epsilon$")
            ax.legend(frameon=False)
        ax.set_title("shadow fraction")
        ax.grid(True, color="#e5e7eb", lw=0.6)

    sector_angle = np.linspace(0, 360, surface["neutral_resolved"].shape[1] + 1)
    grain_axis = np.arange(surface["neutral_resolved"].shape[0] + 1)
    for col, profile in enumerate(compare):
        ax = axes[2, col]
        surf_ratio = np.divide(
            surface[profile],
            surface["neutral_resolved"],
            out=np.full_like(surface["neutral_resolved"], np.nan),
            where=surface["neutral_resolved"] > eps,
        )
        surface_shadow_mask = (surface["neutral_resolved"] > eps) & (surface[profile] <= eps)
        surface_shadow_grain, surface_shadow_sector = np.where(surface_shadow_mask)
        im2 = ax.imshow(
            np.clip(surf_ratio, 0.0, 2.0),
            origin="lower",
            aspect="auto",
            extent=(sector_angle[0], sector_angle[-1], grain_axis[0], grain_axis[-1]),
            cmap=ratio_cmap,
            vmin=0,
            vmax=2,
        )
        if surface_shadow_grain.size:
            ax.scatter(
                (surface_shadow_sector + 0.5) * 360.0 / surface["neutral_resolved"].shape[1],
                surface_shadow_grain + 0.5,
                marker="v",
                s=18,
                facecolors="#22d3ee",
                edgecolors="#0f172a",
                linewidths=0.35,
            )
        ax.text(
            0.03,
            0.96,
            f"{surface_shadow_grain.size} shadow sectors",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.88),
        )
        ax.set_xlabel("grain-local sector angle (deg)")
        if col == 0:
            ax.set_ylabel("grain id")
        ax.set_title(r"surface-sector $H_\chi/H_0$")
    cax2 = fig.add_axes([0.89, 0.105, 0.018, 0.18])
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label("exposure ratio")
    for label, ax in zip("abcdefghi", axes.flat):
        ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    fig.suptitle("Finite-time transport-shadow diagnostic in the random packing", fontsize=14, fontweight="bold")
    path = FIGURES / "transport_shadow_diagnostic.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(OUT / "transport_shadow_diagnostic.png", bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    summaries: list[dict[str, object]],
    shadow_rows: list[dict[str, object]],
    figure: Path,
    args: argparse.Namespace,
    geometry_path: Path,
    flow_case: Path,
    flow_mean_ux_before_scale: float,
    flow_scale: float,
) -> Path:
    lines = [
        "# Transport Shadow Diagnostic",
        "",
        "Targeted random-packing simulations with per-particle pore-cell and surface-sector occupancy.",
        "",
        f"- Geometry: `{geometry_path.relative_to(ROOT) if geometry_path.is_absolute() and ROOT in geometry_path.parents else geometry_path}`",
        f"- Flow case: `{flow_case.relative_to(ROOT) if flow_case.is_absolute() and ROOT in flow_case.parents else flow_case}`",
        f"- Flow rescale factor: {flow_scale:.6g}",
        f"- Area-weighted mean ux before rescale: {flow_mean_ux_before_scale:.6g} m/s",
        f"- Particles per condition: {args.particles}",
        f"- Horizon: {args.max_time:g} s",
        f"- Time step: {args.dt:g} s",
        f"- Pore grid: {args.pore_nx} x {args.pore_ny}",
        f"- Surface sectors per grain: {args.surface_bins}",
        f"- Occupancy sampled every {args.sample_stride} outer steps",
        "",
        "## Condition summary",
        "",
        "| profile | exited | censored | intercepted | effective pore fraction | visited surface sectors | median near time if intercepted (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {profile} | {exited} | {censored} | {intercepted} | {pore_effective_fraction_entropy:.3f} | {surface_sectors_visited} | {median_near_time_intercepted_s:.3g} |".format(
                **row
            )
        )
    lines.extend(["", "## Shadow fractions", "", "| profile | epsilon | pore shadow | surface shadow | pore shadow cells/support | surface shadow sectors/support |", "|---|---:|---:|---:|---:|---:|"])
    for row in shadow_rows:
        lines.append(
            "| {profile} | {epsilon:.3g} | {pore_shadow_fraction:.3f} | {surface_shadow_fraction:.3f} | {pore_shadow_count}/{pore_support_count} | {surface_shadow_count}/{surface_support_count} |".format(
                **row
            )
        )
    lines.extend(["", f"Figure: `{figure.relative_to(ROOT)}`", ""])
    path = OUT / "transport_shadow_diagnostic_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-case", type=Path, default=rpt.DEFAULT_FLOW_CASE)
    parser.add_argument("--geometry-path", type=Path, default=rpt.GEOMETRY_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES))
    parser.add_argument("--particles", type=int, default=1600)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--pore-nx", type=int, default=96)
    parser.add_argument("--pore-ny", type=int, default=64)
    parser.add_argument("--flow-nx", type=int, default=384)
    parser.add_argument("--flow-ny", type=int, default=256)
    parser.add_argument("--surface-bins", type=int, default=36)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--inlet-x-um", type=float, default=1.0)
    parser.add_argument("--adaptive-cutoff-nm", type=float, default=75.0)
    parser.add_argument("--normal-step-target-nm", type=float, default=3.0)
    parser.add_argument("--max-substeps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--progress", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-mean-velocity-m-per-day", type=float, default=rpt.TARGET_M_PER_DAY)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--no-rescale-flow", action="store_true")
    return parser.parse_args()


def main() -> None:
    global OUT, FIGURES
    args = parse_args()
    OUT = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    FIGURES = args.figures_dir if args.figures_dir.is_absolute() else ROOT / args.figures_dir
    OUT.mkdir(parents=True, exist_ok=True)
    flow_case = args.flow_case if args.flow_case.is_absolute() else ROOT / args.flow_case
    geometry_path = args.geometry_path if args.geometry_path.is_absolute() else ROOT / args.geometry_path
    geometry = rpt.load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=not args.no_flow_origin_shift)
    interpolator, flow_mean_ux_before_scale, flow_scale = flow_interpolator(
        flow_case,
        geometry,
        args.target_mean_velocity_m_per_day,
        not args.no_rescale_flow,
    )
    build_kernel(force=True)
    disks = rpt.PeriodicDiskGeometry(geometry, rpt.RandomTrackingParams().particle_radius)
    flow = rasterize_flow_grid(interpolator, disks.lx, disks.ly, args.flow_nx, args.flow_ny)
    rng = np.random.default_rng(args.seed)
    tag = run_tag(args)
    initial_y_path = OUT / f"initial_y_{tag}.npy"
    if args.resume and initial_y_path.exists():
        initial_y = np.load(initial_y_path)
        if initial_y.size != args.particles:
            raise ValueError(f"Stored initial_y has {initial_y.size} particles, expected {args.particles}")
    else:
        initial_y = rpt.open_inlet_samples(
            rng,
            args.particles,
            disks.lx,
            disks.ly,
            disks.centers,
            disks.radii,
            rpt.RandomTrackingParams(),
            args.inlet_x_um * 1.0e-6,
        )
        np.save(initial_y_path, initial_y)
    summaries: list[dict[str, object]] = []
    pore: dict[str, np.ndarray] = {}
    surface: dict[str, np.ndarray] = {}
    for profile_id, profile in enumerate(args.profiles):
        profile_path = OUT / f"{profile}_occupancy_{tag}.npz"
        if args.resume and profile_path.exists():
            data = np.load(profile_path)
            row = json.loads(str(data["summary_json"].item()))
            pore_access = data["pore_accessibility"]
            surface_access = data["surface_accessibility"]
            print(f"reused {profile}", flush=True)
        else:
            print(f"running {profile}", flush=True)
            row, pore_access, surface_access = simulate_profile_chunked(
                profile,
                flow,
                geometry,
                initial_y,
                args.seed + 1000 * (profile_id + 1),
                args,
            )
            np.savez_compressed(
                profile_path,
                pore_accessibility=pore_access,
                surface_accessibility=surface_access,
                summary_json=json.dumps(row),
            )
            print(f"finished {profile}", flush=True)
        summaries.append(row)
        pore[profile] = pore_access
        surface[profile] = surface_access
        write_csv(OUT / "transport_shadow_condition_summary.csv", summaries)
    reference_pore = pore["neutral_resolved"]
    reference_surface = surface["neutral_resolved"]
    epsilons = [0.002, 0.005, 0.01, 0.02]
    shadow_rows: list[dict[str, object]] = []
    for profile in args.profiles:
        if profile == "neutral_resolved":
            continue
        for eps in epsilons:
            pore_fraction, pore_shadow, pore_support = shadow_fraction(reference_pore, pore[profile], eps)
            surface_fraction, surface_shadow, surface_support = shadow_fraction(reference_surface, surface[profile], eps)
            shadow_rows.append(
                {
                    "profile": profile,
                    "label": PROFILES[profile]["label"],
                    "epsilon": eps,
                    "pore_shadow_fraction": pore_fraction,
                    "pore_shadow_count": pore_shadow,
                    "pore_support_count": pore_support,
                    "surface_shadow_fraction": surface_fraction,
                    "surface_shadow_count": surface_shadow,
                    "surface_support_count": surface_support,
                }
            )
    write_csv(OUT / "transport_shadow_metrics.csv", shadow_rows)
    figure = plot_maps(geometry, summaries, pore, surface, args.epsilon)
    report = write_report(
        summaries,
        shadow_rows,
        figure,
        args,
        geometry_path,
        flow_case,
        flow_mean_ux_before_scale,
        flow_scale,
    )
    print(report)
    print(figure)


if __name__ == "__main__":
    main()
