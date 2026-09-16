#!/usr/bin/env python3
"""Pilot particle tracking in the random OpenFOAM porous cell."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FLOW_SCRIPT = ROOT / "scripts" / "run_random_openfoam_flow.py"
DEFAULT_FLOW_CASE = ROOT / "outputs" / "random_openfoam_flow_wall2_far5"
GEOMETRY_PATH = ROOT / "outputs" / "random_porous_geometry" / "random_porous_geometry.json"
OUT = ROOT / "outputs" / "random_particle_tracking"
TARGET_M_PER_DAY = 4.0

spec = importlib.util.spec_from_file_location("random_openfoam_flow", FLOW_SCRIPT)
flow_io = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(flow_io)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

from colloid_tsm.physical import PhysicalParams, _near_wall_mobility_factors


@dataclass(frozen=True)
class RandomTrackingParams:
    particle_radius: float = 0.55e-6
    temperature: float = 298.15
    viscosity: float = 1.0e-3
    relative_permittivity: float = 78.5
    ionic_strength_molar: float = 50.0e-3
    hamaker: float = 3.83e-21
    zeta_particle: float = -50.1e-3
    zeta_collector_favorable: float = 70.0e-3
    zeta_collector_unfavorable: float = -70.0e-3
    near_surface: float = 200.0e-9
    contact_gap: float = 1.0e-9
    min_gap: float = 1.0e-9
    dt: float = 2.0e-3
    max_time: float = 80.0
    dlvo_velocity_cap: float = 2.0e-4
    wall_mobility_cutoff: float = 2.0e-6
    wall_mobility_normal_floor: float = 1.0e-4
    wall_mobility_parallel_floor: float = 5.0e-2
    diffusivity_multiplier: float = 1.0
    adaptive_near_wall: bool = True
    adaptive_cutoff: float = 75.0e-9
    normal_step_target: float = 3.0e-9
    min_substep: float = 1.0e-6
    max_substeps: int = 50
    segment_collision: bool = True

    @property
    def diffusivity(self) -> float:
        k_b = 1.380649e-23
        return self.diffusivity_multiplier * k_b * self.temperature / (
            6.0 * math.pi * self.viscosity * self.particle_radius
        )

    @property
    def debye_length(self) -> float:
        eps0 = 8.8541878128e-12
        e = 1.602176634e-19
        n_a = 6.02214076e23
        k_b = 1.380649e-23
        ionic_strength_m3 = 1000.0 * self.ionic_strength_molar
        kappa2 = 2.0 * e * e * n_a * ionic_strength_m3 / (
            self.relative_permittivity * eps0 * k_b * self.temperature
        )
        return 1.0 / math.sqrt(kappa2)

    def as_physical_params(self) -> PhysicalParams:
        return PhysicalParams(
            particle_radius=self.particle_radius,
            temperature=self.temperature,
            viscosity=self.viscosity,
            relative_permittivity=self.relative_permittivity,
            ionic_strength_molar=self.ionic_strength_molar,
            hamaker=self.hamaker,
            zeta_particle=self.zeta_particle,
            zeta_collector_favorable=self.zeta_collector_favorable,
            zeta_collector_unfavorable=self.zeta_collector_unfavorable,
            near_surface=self.near_surface,
            contact_gap=self.contact_gap,
            min_gap=self.min_gap,
            dt=self.dt,
            max_time=self.max_time,
            dlvo_velocity_cap=self.dlvo_velocity_cap,
            wall_mobility_cutoff=self.wall_mobility_cutoff,
            wall_mobility_normal_floor=self.wall_mobility_normal_floor,
            wall_mobility_parallel_floor=self.wall_mobility_parallel_floor,
            diffusivity_multiplier=self.diffusivity_multiplier,
            resolved_langevin_substeps=max(1, self.max_substeps),
            resolved_langevin_substep_cutoff=self.adaptive_cutoff,
            resolved_langevin_normal_step=self.normal_step_target,
            resolved_langevin_min_dt=self.min_substep,
            resolved_langevin_max_substeps=max(1, self.max_substeps),
        )


@dataclass
class TrackingResult:
    condition: str
    x0: np.ndarray
    y0: np.ndarray
    x_final: np.ndarray
    y_final: np.ndarray
    y_out: np.ndarray
    travel_time: np.ndarray
    exited: np.ndarray
    attached: np.ndarray
    censored: np.ndarray
    interceptions: np.ndarray
    near_time: np.ndarray
    h_min: np.ndarray
    grain_entry: np.ndarray
    grain_final: np.ndarray
    path_x: list[np.ndarray]
    path_y: list[np.ndarray]
    path_status: np.ndarray


class PeriodicIDWFlow:
    """Periodic inverse-distance interpolation of OpenFOAM cell-centered velocity."""

    def __init__(self, centers_xy: np.ndarray, velocity_xy: np.ndarray, lx: float, ly: float, k: int = 8) -> None:
        self.lx = lx
        self.ly = ly
        self.k = k
        offsets = np.array(
            [(ix * lx, iy * ly) for ix in (-1, 0, 1) for iy in (-1, 0, 1)],
            dtype=float,
        )
        tiled_points = []
        tiled_velocity = []
        for offset in offsets:
            tiled_points.append(centers_xy + offset)
            tiled_velocity.append(velocity_xy)
        self.points = np.vstack(tiled_points)
        self.velocity = np.vstack(tiled_velocity)
        self.tree = cKDTree(self.points)

    def velocity_at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.column_stack((np.mod(x, self.lx), np.mod(y, self.ly)))
        dist, idx = self.tree.query(query, k=self.k)
        if self.k == 1:
            out = self.velocity[idx]
            return out[:, 0], out[:, 1]
        dist = np.maximum(dist, 1.0e-12)
        weights = 1.0 / (dist * dist)
        weights /= np.sum(weights, axis=1, keepdims=True)
        out = np.sum(self.velocity[idx] * weights[:, :, None], axis=1)
        return out[:, 0], out[:, 1]


class PeriodicDiskGeometry:
    """Periodic disk packing with nearest-surface and segment-contact queries."""

    def __init__(self, geometry: dict, particle_radius: float, candidate_count: int = 12) -> None:
        self.lx = float(geometry["domain"]["length_x"])
        self.ly = float(geometry["domain"]["length_y"])
        self.centers = np.array([[g["x"], g["y"]] for g in geometry["grains"]], dtype=float)
        self.radii = np.array([g["radius"] for g in geometry["grains"]], dtype=float)
        self.particle_radius = particle_radius
        self.candidate_count = min(candidate_count, 9 * len(self.radii))

        offsets = np.array(
            [(ix * self.lx, iy * self.ly) for ix in (-1, 0, 1) for iy in (-1, 0, 1)],
            dtype=float,
        )
        points = []
        grain_ids = []
        for offset in offsets:
            points.append(self.centers + offset)
            grain_ids.append(np.arange(self.centers.shape[0], dtype=int))
        self.tiled_centers = np.vstack(points)
        self.tiled_grain_ids = np.concatenate(grain_ids)
        self.tree = cKDTree(self.tiled_centers)

    def nearest_surface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query = np.column_stack((np.mod(x, self.lx), np.mod(y, self.ly)))
        query_k = self.candidate_count if k is None else min(k, self.candidate_count)
        dist, idx = self.tree.query(query, k=query_k)
        if query_k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        candidate_gid = self.tiled_grain_ids[idx]
        candidate_centers = self.tiled_centers[idx]
        dx = query[:, None, 0] - candidate_centers[:, :, 0]
        dy = query[:, None, 1] - candidate_centers[:, :, 1]
        distance = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-30)
        gap = distance - (self.radii[candidate_gid] + self.particle_radius)
        best = np.argmin(gap, axis=1)
        rows = np.arange(query.shape[0])
        best_distance = distance[rows, best]
        best_dx = dx[rows, best]
        best_dy = dy[rows, best]
        best_gid = candidate_gid[rows, best]
        return (
            gap[rows, best],
            best_dx / best_distance,
            best_dy / best_distance,
            best_distance,
            best_gid,
        )

    def project_outside(
        self,
        x: np.ndarray,
        y: np.ndarray,
        active_mask: np.ndarray,
        params: RandomTrackingParams,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if not np.any(active_mask):
            return x, y, 0
        idx = np.flatnonzero(active_mask)
        gap, _, _, _, grain_id = self.nearest_surface(x[idx], y[idx])
        hit_local = gap < params.contact_gap
        if not np.any(hit_local):
            return x, y, 0
        hit = idx[hit_local]
        gid = grain_id[hit_local]
        target = self.radii[gid] + params.particle_radius + params.contact_gap
        cx = self.centers[gid, 0]
        cy = self.centers[gid, 1]
        old_mod_x = np.mod(x[hit], self.lx)
        old_mod_y = np.mod(y[hit], self.ly)
        dx = periodic_delta(old_mod_x - cx, self.lx)
        dy = periodic_delta(old_mod_y - cy, self.ly)
        distance = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-30)
        projected_mod_x = np.mod(cx + dx / distance * target, self.lx)
        projected_mod_y = np.mod(cy + dy / distance * target, self.ly)
        x[hit] += periodic_delta(projected_mod_x - old_mod_x, self.lx)
        y[hit] = projected_mod_y
        return x, y, int(hit.size)

    def segment_contacts(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
        active_mask: np.ndarray,
        params: RandomTrackingParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return contact state for active segments crossing an exclusion circle."""

        hit = np.zeros(x0.size, dtype=bool)
        hit_t = np.full(x0.size, np.inf)
        hit_grain = np.full(x0.size, -1, dtype=int)
        hit_nx = np.ones(x0.size)
        hit_ny = np.zeros(x0.size)
        if not np.any(active_mask):
            return hit, hit_t, hit_grain, hit_nx, hit_ny

        idx = np.flatnonzero(active_mask)
        p0x = np.mod(x0[idx], self.lx)
        p0y = np.mod(y0[idx], self.ly)
        dpx = periodic_delta(np.mod(x1[idx], self.lx) - p0x, self.lx)
        dpy = periodic_delta(np.mod(y1[idx], self.ly) - p0y, self.ly)
        a = dpx * dpx + dpy * dpy
        moving = a > 1.0e-32
        local_best_t = np.full(idx.size, np.inf)
        local_best_gid = np.full(idx.size, -1, dtype=int)
        local_best_nx = np.ones(idx.size)
        local_best_ny = np.zeros(idx.size)

        for gid, ((cx, cy), radius) in enumerate(zip(self.centers, self.radii)):
            cx_img = p0x - periodic_delta(p0x - cx, self.lx)
            cy_img = p0y - periodic_delta(p0y - cy, self.ly)
            fx = p0x - cx_img
            fy = p0y - cy_img
            boundary_radius = radius + params.particle_radius + params.contact_gap
            c = fx * fx + fy * fy - boundary_radius * boundary_radius
            b = fx * dpx + fy * dpy
            disc = b * b - a * c
            crossing = moving & (disc >= 0.0)
            t = np.full(idx.size, np.inf)
            t[crossing] = (-b[crossing] - np.sqrt(disc[crossing])) / a[crossing]
            valid = ((t > 1.0e-12) & (t <= 1.0)) | ((c <= 0.0) & (b < 0.0))
            t = np.where((c <= 0.0) & (b < 0.0), 0.0, t)
            take = valid & (t < local_best_t)
            if not np.any(take):
                continue
            contact_x = p0x[take] + t[take] * dpx[take]
            contact_y = p0y[take] + t[take] * dpy[take]
            normal_x = contact_x - cx_img[take]
            normal_y = contact_y - cy_img[take]
            normal_length = np.maximum(np.sqrt(normal_x * normal_x + normal_y * normal_y), 1.0e-30)
            local_best_t[take] = t[take]
            local_best_gid[take] = gid
            local_best_nx[take] = normal_x / normal_length
            local_best_ny[take] = normal_y / normal_length

        local_hit = local_best_gid >= 0
        if np.any(local_hit):
            hit_idx = idx[local_hit]
            hit[hit_idx] = True
            hit_t[hit_idx] = local_best_t[local_hit]
            hit_grain[hit_idx] = local_best_gid[local_hit]
            hit_nx[hit_idx] = local_best_nx[local_hit]
            hit_ny[hit_idx] = local_best_ny[local_hit]
        return hit, hit_t, hit_grain, hit_nx, hit_ny

    def reflect_segments(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
        active_mask: np.ndarray,
        params: RandomTrackingParams,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hit, hit_t, _, hit_nx, hit_ny = self.segment_contacts(x0, y0, x1, y1, active_mask, params)
        if not np.any(hit):
            return x1, y1, hit
        x_ref = x1.copy()
        y_ref = y1.copy()
        hidx = np.flatnonzero(hit)
        p0x = np.mod(x0[hidx], self.lx)
        p0y = np.mod(y0[hidx], self.ly)
        dpx = periodic_delta(np.mod(x1[hidx], self.lx) - p0x, self.lx)
        dpy = periodic_delta(np.mod(y1[hidx], self.ly) - p0y, self.ly)
        t = hit_t[hidx]
        contact_x = p0x + t * dpx
        contact_y = p0y + t * dpy
        remaining_x = (1.0 - t) * dpx
        remaining_y = (1.0 - t) * dpy
        inward = remaining_x * hit_nx[hidx] + remaining_y * hit_ny[hidx]
        reflect = inward < 0.0
        remaining_x[reflect] -= 2.0 * inward[reflect] * hit_nx[hidx][reflect]
        remaining_y[reflect] -= 2.0 * inward[reflect] * hit_ny[hidx][reflect]
        new_mod_x = np.mod(contact_x + remaining_x + 1.0e-12 * hit_nx[hidx], self.lx)
        new_mod_y = np.mod(contact_y + remaining_y + 1.0e-12 * hit_ny[hidx], self.ly)
        x_ref[hidx] = x0[hidx] + periodic_delta(new_mod_x - p0x, self.lx)
        y_ref[hidx] = new_mod_y
        x_ref, y_ref, _ = self.project_outside(x_ref, y_ref, hit, params)
        return x_ref, y_ref, hit


def latest_vtu(flow_case: Path) -> Path:
    candidates = sorted((flow_case / "case" / "VTK").glob("*/*.vtu"))
    internal = [path for path in candidates if path.name == "internal.vtu"]
    if internal:
        return internal[-1]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No VTU found under {flow_case / 'case' / 'VTK'}")


def apply_periodic_origin_shift(geometry: dict, x_cut: float, y_cut: float) -> dict:
    shifted = json.loads(json.dumps(geometry))
    lx = float(shifted["domain"]["length_x"])
    ly = float(shifted["domain"]["length_y"])
    for grain in shifted["grains"]:
        grain["x"] = float((float(grain["x"]) - x_cut) % lx)
        grain["y"] = float((float(grain["y"]) - y_cut) % ly)
    shifted["openfoam_origin_shift"] = {
        "enabled": True,
        "x_cut": float(x_cut),
        "y_cut": float(y_cut),
        "source": "random_openfoam_flow_report.md",
    }
    return shifted


def report_origin_shift(flow_case: Path) -> tuple[float, float] | None:
    report = flow_case / "random_openfoam_flow_report.md"
    if not report.exists():
        return None
    text = report.read_text(encoding="utf-8", errors="replace")
    import re

    match = re.search(r"OpenFOAM periodic-window shift: x_cut=([0-9.eE+-]+) um, y_cut=([0-9.eE+-]+) um", text)
    if match is None:
        return None
    return float(match.group(1)) * 1.0e-6, float(match.group(2)) * 1.0e-6


def load_geometry(geometry_path: Path = GEOMETRY_PATH, flow_case: Path | None = None, apply_flow_shift: bool = False) -> dict:
    path = geometry_path if geometry_path.is_absolute() else ROOT / geometry_path
    geometry = json.loads(path.read_text(encoding="utf-8"))
    if flow_case is not None and apply_flow_shift:
        case_path = flow_case if flow_case.is_absolute() else ROOT / flow_case
        shift = report_origin_shift(case_path)
        if shift is not None:
            geometry = apply_periodic_origin_shift(geometry, *shift)
    return geometry


def rescale_velocity_to_target(
    velocity_xy: np.ndarray,
    areas: np.ndarray,
    target_m_per_day: float,
) -> tuple[np.ndarray, float, float]:
    target = target_m_per_day / 86400.0
    good = np.isfinite(areas) & (areas > 0.0) & np.all(np.isfinite(velocity_xy), axis=1)
    if not np.any(good):
        raise ValueError("No valid OpenFOAM cells for velocity rescaling.")
    mean_ux = float(np.sum(velocity_xy[good, 0] * areas[good]) / np.sum(areas[good]))
    if abs(mean_ux) < 1.0e-30:
        raise ValueError("Cannot rescale flow with zero area-weighted mean ux.")
    scale = target / mean_ux
    return velocity_xy * scale, mean_ux, scale


def periodic_delta(delta: np.ndarray, length: float) -> np.ndarray:
    return delta - length * np.round(delta / length)


def nearest_surface(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    lx: float,
    ly: float,
    particle_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest collector surface state for arbitrary periodic disks."""

    best_gap = np.full(x.size, np.inf)
    best_nx = np.ones(x.size)
    best_ny = np.zeros(x.size)
    best_distance = np.ones(x.size)
    best_grain = np.full(x.size, -1, dtype=int)
    for grain_id, ((cx, cy), radius) in enumerate(zip(centers, radii)):
        dx = periodic_delta(x - cx, lx)
        dy = periodic_delta(y - cy, ly)
        distance = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-30)
        gap = distance - (radius + particle_radius)
        take = gap < best_gap
        best_gap[take] = gap[take]
        best_nx[take] = dx[take] / distance[take]
        best_ny[take] = dy[take] / distance[take]
        best_distance[take] = distance[take]
        best_grain[take] = grain_id
    return best_gap, best_nx, best_ny, best_distance, best_grain


def project_outside_grains(
    x: np.ndarray,
    y: np.ndarray,
    active_mask: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    lx: float,
    ly: float,
    params: RandomTrackingParams,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not np.any(active_mask):
        return x, y, 0
    idx = np.flatnonzero(active_mask)
    gap, nx, ny, _, grain_id = nearest_surface(
        np.mod(x[idx], lx),
        np.mod(y[idx], ly),
        centers,
        radii,
        lx,
        ly,
        params.particle_radius,
    )
    hit_local = gap < params.contact_gap
    if not np.any(hit_local):
        return x, y, 0
    hit = idx[hit_local]
    gid = grain_id[hit_local]
    target = radii[gid] + params.particle_radius + params.contact_gap
    cx = centers[gid, 0]
    cy = centers[gid, 1]
    old_mod_x = np.mod(x[hit], lx)
    old_mod_y = np.mod(y[hit], ly)
    dx = periodic_delta(old_mod_x - cx, lx)
    dy = periodic_delta(old_mod_y - cy, ly)
    distance = np.maximum(np.sqrt(dx * dx + dy * dy), 1.0e-30)
    projected_mod_x = np.mod(cx + dx / distance * target, lx)
    projected_mod_y = np.mod(cy + dy / distance * target, ly)
    x[hit] += periodic_delta(projected_mod_x - old_mod_x, lx)
    y[hit] = projected_mod_y
    return x, y, int(hit.size)


def dlvo_force_from_gap(params: RandomTrackingParams, h: np.ndarray, condition: str) -> np.ndarray:
    if condition in {"no_dlvo", "neutral", "advection_diffusion"}:
        return np.zeros_like(h)
    in_layer = h <= params.near_surface
    h_eff = np.maximum(h, params.min_gap)
    eps0 = 8.8541878128e-12
    eps = params.relative_permittivity * eps0
    kappa = 1.0 / params.debye_length
    zeta_c = params.zeta_collector_favorable if condition.startswith("favorable") else params.zeta_collector_unfavorable
    f_edl = (
        2.0
        * math.pi
        * eps
        * params.particle_radius
        * kappa
        * params.zeta_particle
        * zeta_c
        * np.exp(-kappa * h_eff)
    )
    f_vdw = -params.hamaker * params.particle_radius / (6.0 * h_eff * h_eff)
    return np.where(in_layer, f_edl + f_vdw, 0.0)


def estimate_adaptive_substeps(
    condition: str,
    interpolator: PeriodicIDWFlow,
    disks: PeriodicDiskGeometry,
    params: RandomTrackingParams,
    phys_params: PhysicalParams,
    x: np.ndarray,
    y: np.ndarray,
    active: np.ndarray,
) -> int:
    if not params.adaptive_near_wall or not np.any(active):
        return 1
    idx = np.flatnonzero(active)
    gap, nx, ny, _, _ = disks.nearest_surface(x[idx], y[idx])
    near = gap <= params.adaptive_cutoff
    if not np.any(near):
        return 1

    ux, uy = interpolator.velocity_at(x[idx][near], y[idx][near])
    normal_mobility, _, _ = _near_wall_mobility_factors(phys_params, gap[near])
    hydro_n = ux * nx[near] + uy * ny[near]
    stokes_mobility = 1.0 / (6.0 * math.pi * params.viscosity * params.particle_radius)
    f_dlvo = dlvo_force_from_gap(params, gap[near], condition)
    v_dlvo_n = np.clip(stokes_mobility * f_dlvo, -params.dlvo_velocity_cap, params.dlvo_velocity_cap)
    normal_velocity = normal_mobility * (hydro_n + v_dlvo_n)
    brownian_std = np.sqrt(2.0 * params.diffusivity * normal_mobility * params.dt)
    n_brownian = np.max((brownian_std / params.normal_step_target) ** 2)
    n_deterministic = np.max(np.abs(normal_velocity) * params.dt / params.normal_step_target)
    requested = int(math.ceil(max(1.0, float(n_brownian), float(n_deterministic))))
    max_by_min_dt = max(1, int(math.floor(params.dt / params.min_substep)))
    return max(1, min(params.max_substeps, max_by_min_dt, requested))


def open_inlet_samples(
    rng: np.random.Generator,
    n: int,
    lx: float,
    ly: float,
    centers: np.ndarray,
    radii: np.ndarray,
    params: RandomTrackingParams,
    x0: float,
    clearance: float = 2.0e-6,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    attempts = 0
    while sum(chunk.size for chunk in samples) < n and attempts < 200:
        candidates = rng.uniform(0.0, ly, size=max(4 * n, 256))
        x_arr = np.full(candidates.size, x0)
        gap, *_ = nearest_surface(x_arr, candidates, centers, radii, lx, ly, params.particle_radius)
        good = gap > clearance
        samples.append(candidates[good])
        attempts += 1
    y = np.concatenate(samples)[:n]
    if y.size < n:
        raise RuntimeError("Could not sample enough open inlet positions.")
    return y


def simulate_particles(
    condition: str,
    interpolator: PeriodicIDWFlow,
    geometry: dict,
    params: RandomTrackingParams,
    n_particles: int,
    seed: int,
    initial_x: np.ndarray | None = None,
    initial_y: np.ndarray | None = None,
    record_count: int = 36,
    record_stride: int = 40,
) -> TrackingResult:
    rng = np.random.default_rng(seed)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    disks = PeriodicDiskGeometry(geometry, params.particle_radius)
    centers = disks.centers
    radii = disks.radii
    x_start = 1.0e-6
    if initial_y is None:
        y0 = open_inlet_samples(rng, n_particles, lx, ly, centers, radii, params, x_start)
    else:
        y0 = np.asarray(initial_y, dtype=float)
        n_particles = int(y0.size)
    if initial_x is None:
        x = np.full(n_particles, x_start)
    else:
        x = np.asarray(initial_x, dtype=float)
        if x.size != n_particles:
            raise ValueError("initial_x and initial_y must have the same length.")
    y = y0.copy()
    x0 = x.copy()

    active = np.ones(n_particles, dtype=bool)
    exited = np.zeros(n_particles, dtype=bool)
    attached = np.zeros(n_particles, dtype=bool)
    censored = np.zeros(n_particles, dtype=bool)
    y_out = np.full(n_particles, np.nan)
    travel_time = np.full(n_particles, np.nan)
    interceptions = np.zeros(n_particles, dtype=int)
    in_near = np.zeros(n_particles, dtype=bool)
    near_time = np.zeros(n_particles)
    h_min = np.full(n_particles, np.inf)
    grain_entry = np.full(n_particles, -1, dtype=int)
    grain_final = np.full(n_particles, -1, dtype=int)

    record_ids = np.linspace(0, n_particles - 1, min(record_count, n_particles), dtype=int)
    path_x = [list([x[i]]) for i in record_ids]
    path_y = [list([y[i]]) for i in record_ids]
    record_lookup = {int(pid): k for k, pid in enumerate(record_ids)}

    phys_params = params.as_physical_params()
    stokes_mobility = 1.0 / (6.0 * math.pi * params.viscosity * params.particle_radius)
    n_steps = int(math.ceil(params.max_time / params.dt))
    attach_gap = 2.0e-9
    time = 0.0

    for step in range(n_steps):
        if not np.any(active) or time >= params.max_time:
            break
        n_sub = estimate_adaptive_substeps(condition, interpolator, disks, params, phys_params, x, y, active)
        dt_sub = min(params.dt / n_sub, params.max_time - time)
        for _ in range(n_sub):
            if not np.any(active) or time >= params.max_time:
                break
            if time + dt_sub > params.max_time:
                dt_sub = params.max_time - time
            idx = np.flatnonzero(active)
            xa = x[idx]
            ya = np.mod(y[idx], ly)
            gap, nx, ny, _, grain_id = disks.nearest_surface(xa, ya)
            h_min[idx] = np.minimum(h_min[idx], gap)
            near_now = gap <= params.near_surface
            entering = near_now & (~in_near[idx])
            if np.any(entering):
                global_entering = idx[entering]
                interceptions[global_entering] += 1
                first = grain_entry[global_entering] < 0
                grain_entry[global_entering[first]] = grain_id[entering][first]
            in_near[idx] = near_now
            near_time[idx[near_now]] += dt_sub
            grain_final[idx] = grain_id

            ux, uy = interpolator.velocity_at(xa, ya)
            normal_mobility, parallel_mobility, _ = _near_wall_mobility_factors(phys_params, gap)
            tx = -ny
            ty = nx
            hydro_n = ux * nx + uy * ny
            hydro_t = ux * tx + uy * ty
            ux_eff = normal_mobility * hydro_n * nx + parallel_mobility * hydro_t * tx
            uy_eff = normal_mobility * hydro_n * ny + parallel_mobility * hydro_t * ty

            f_dlvo = dlvo_force_from_gap(params, gap, condition)
            v_dlvo_n = np.clip(stokes_mobility * f_dlvo, -params.dlvo_velocity_cap, params.dlvo_velocity_cap)
            ux_eff += normal_mobility * v_dlvo_n * nx
            uy_eff += normal_mobility * v_dlvo_n * ny

            sigma_n = np.sqrt(2.0 * params.diffusivity * normal_mobility * dt_sub)
            sigma_t = np.sqrt(2.0 * params.diffusivity * parallel_mobility * dt_sub)
            brown_n = rng.normal(0.0, sigma_n)
            brown_t = rng.normal(0.0, sigma_t)
            dx = ux_eff * dt_sub + brown_n * nx + brown_t * tx
            dy = uy_eff * dt_sub + brown_n * ny + brown_t * ty

            x_old = x.copy()
            y_old = y.copy()
            x_proposed = x.copy()
            y_proposed = y.copy()
            x_proposed[idx] = xa + dx
            y_proposed[idx] = np.mod(ya + dy, ly)

            segment_hit = np.zeros(n_particles, dtype=bool)
            if params.segment_collision:
                segment_hit, *_ = disks.segment_contacts(x_old, y_old, x_proposed, y_proposed, active, params)

            x[:] = x_proposed
            y[:] = y_proposed
            gap_new, *_ = disks.nearest_surface(x[idx], y[idx])
            if condition.startswith("favorable"):
                attach_local = segment_hit[idx] | (gap_new <= attach_gap)
                if np.any(attach_local):
                    attach_idx = idx[attach_local]
                    attached[attach_idx] = True
                    active[attach_idx] = False
                    travel_time[attach_idx] = time + dt_sub

            reflect_mask = active.copy()
            if params.segment_collision:
                x_ref, y_ref, _ = disks.reflect_segments(x_old, y_old, x, y, reflect_mask, params)
                x[:] = x_ref
                y[:] = y_ref
            x, y, _ = disks.project_outside(x, y, reflect_mask, params)

            time += dt_sub
            exit_now = active & (x >= lx)
            if np.any(exit_now):
                exited[exit_now] = True
                active[exit_now] = False
                y_out[exit_now] = np.mod(y[exit_now], ly)
                travel_time[exit_now] = time
            upstream_lost = active & (x < -0.25 * lx)
            if np.any(upstream_lost):
                censored[upstream_lost] = True
                active[upstream_lost] = False
                travel_time[upstream_lost] = time

        if step % record_stride == 0:
            for pid in record_ids:
                k = record_lookup[int(pid)]
                path_x[k].append(float(x[pid]))
                path_y[k].append(float(np.mod(y[pid], ly)))

    if np.any(active):
        censored[active] = True
        travel_time[active] = params.max_time
        active[:] = False
    path_status = np.full(record_ids.size, 0, dtype=int)
    for k, pid in enumerate(record_ids):
        if attached[pid]:
            path_status[k] = 1
        elif censored[pid]:
            path_status[k] = 2
    return TrackingResult(
        condition=condition,
        x0=x0,
        y0=y0,
        x_final=np.mod(x, lx),
        y_final=np.mod(y, ly),
        y_out=y_out,
        travel_time=travel_time,
        exited=exited,
        attached=attached,
        censored=censored,
        interceptions=interceptions,
        near_time=near_time,
        h_min=h_min,
        grain_entry=grain_entry,
        grain_final=grain_final,
        path_x=[np.asarray(p) for p in path_x],
        path_y=[np.asarray(p) for p in path_y],
        path_status=path_status,
    )


def _histogram_entropy(values: np.ndarray, domain_length: float, bins: int) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    counts, _ = np.histogram(finite, bins=bins, range=(0.0, domain_length))
    positive = counts[counts > 0].astype(float)
    if positive.size == 0:
        return float("nan"), float("nan")
    probability = positive / np.sum(positive)
    entropy_bits = -float(np.sum(probability * np.log2(probability)))
    return entropy_bits, float(2.0**entropy_bits)


def summarize(
    result: TrackingResult,
    params: RandomTrackingParams,
    ly: float,
    matrix_bins: int,
) -> dict[str, float | int | str]:
    intercepted = result.interceptions > 0
    mobile = result.exited & (~result.attached)
    h_finite = result.h_min[np.isfinite(result.h_min)]
    exit_entropy, exit_effective_bins = _histogram_entropy(result.y_out[result.exited], ly, matrix_bins)
    intercepted_exit = result.exited & intercepted
    intercepted_exit_entropy, intercepted_exit_effective_bins = _histogram_entropy(
        result.y_out[intercepted_exit],
        ly,
        matrix_bins,
    )
    return {
        "condition": result.condition,
        "particles": int(result.x0.size),
        "exited": int(np.sum(result.exited)),
        "attached": int(np.sum(result.attached)),
        "censored": int(np.sum(result.censored)),
        "intercepted": int(np.sum(intercepted)),
        "intercepted_fraction": float(np.mean(intercepted)),
        "attached_fraction": float(np.mean(result.attached)),
        "median_travel_time_exited_s": float(np.nanmedian(result.travel_time[mobile])) if np.any(mobile) else float("nan"),
        "mean_near_time_s": float(np.mean(result.near_time)),
        "median_near_time_intercepted_s": float(np.median(result.near_time[intercepted])) if np.any(intercepted) else 0.0,
        "median_h_min_nm": float(np.median(h_finite) * 1e9) if h_finite.size else float("nan"),
        "p05_h_min_nm": float(np.quantile(h_finite, 0.05) * 1e9) if h_finite.size else float("nan"),
        "exit_entropy_bits": exit_entropy,
        "exit_effective_bins": exit_effective_bins,
        "intercepted_exit_entropy_bits": intercepted_exit_entropy,
        "intercepted_exit_effective_bins": intercepted_exit_effective_bins,
        "diffusivity_m2_s": params.diffusivity,
        "debye_length_nm": params.debye_length * 1e9,
    }


def write_outputs(results: list[TrackingResult], summaries: list[dict[str, float | int | str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "random_particle_tracking_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    npz_path = OUT / "random_particle_tracking_paths.npz"
    payload: dict[str, np.ndarray] = {}
    for result in results:
        payload[f"{result.condition}_x0"] = result.x0
        payload[f"{result.condition}_y0"] = result.y0
        payload[f"{result.condition}_x_final"] = result.x_final
        payload[f"{result.condition}_y_final"] = result.y_final
        payload[f"{result.condition}_y_out"] = result.y_out
        payload[f"{result.condition}_travel_time"] = result.travel_time
        payload[f"{result.condition}_near_time"] = result.near_time
        payload[f"{result.condition}_h_min"] = result.h_min
        payload[f"{result.condition}_exited"] = result.exited.astype(np.uint8)
        payload[f"{result.condition}_attached"] = result.attached.astype(np.uint8)
        payload[f"{result.condition}_censored"] = result.censored.astype(np.uint8)
        payload[f"{result.condition}_interceptions"] = result.interceptions
        payload[f"{result.condition}_grain_entry"] = result.grain_entry
        payload[f"{result.condition}_grain_final"] = result.grain_final
    np.savez_compressed(npz_path, **payload)


def plot_pathlines(results: list[TrackingResult], geometry: dict, params: RandomTrackingParams) -> Path:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    centers = np.array([[g["x"], g["y"]] for g in geometry["grains"]], dtype=float)
    radii = np.array([g["radius"] for g in geometry["grains"]], dtype=float)
    pinned = np.array([bool(g.get("pinned", False)) for g in geometry["grains"]])

    titles = {
        "no_dlvo": "No DLVO",
        "favorable": "Favorable",
        "unfavorable": "Unfavorable",
    }
    fig, axes = plt.subplots(1, len(results), figsize=(15.5, 4.9), dpi=180, sharex=True, sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        ax.add_patch(Rectangle((0.0, 0.0), lx, ly, fill=False, lw=1.1, ec="#111827"))
        for gid, ((cx, cy), radius) in enumerate(zip(centers, radii)):
            edge = "#f97316" if pinned[gid] else "#020617"
            lw = 1.2 if pinned[gid] else 0.45
            ax.add_patch(Circle((cx, cy), radius + params.particle_radius, fc="#111827", ec=edge, lw=lw, zorder=4))
        segments = []
        colors = []
        for px, py, status in zip(result.path_x, result.path_y, result.path_status):
            if px.size < 2:
                continue
            wrapped_x = np.mod(px, lx)
            jumps = (np.abs(np.diff(wrapped_x)) > 0.5 * lx) | (np.abs(np.diff(py)) > 0.5 * ly)
            start = 0
            for jump_id in np.flatnonzero(jumps):
                if jump_id + 1 - start >= 2:
                    segments.append(np.column_stack((wrapped_x[start : jump_id + 1], py[start : jump_id + 1])))
                    colors.append(status)
                start = jump_id + 1
            if wrapped_x.size - start >= 2:
                segments.append(np.column_stack((wrapped_x[start:], py[start:])))
                colors.append(status)
        color_map = {0: "#2563eb", 1: "#dc2626", 2: "#6b7280"}
        lc = LineCollection(segments, colors=[color_map[int(c)] for c in colors], linewidths=0.85, alpha=0.7, zorder=2)
        ax.add_collection(lc)
        ax.scatter(np.mod(result.x0, lx), result.y0, s=5, c="#10b981", alpha=0.5, zorder=3)
        if np.any(result.attached):
            ax.scatter(result.x_final[result.attached], result.y_final[result.attached], s=9, c="#dc2626", alpha=0.9, zorder=6)
        if np.any(result.exited):
            ax.scatter(np.full(np.sum(result.exited), lx), result.y_out[result.exited], s=7, c="#2563eb", alpha=0.6, zorder=6)
        ax.set_title(titles.get(result.condition, result.condition))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0.0, lx)
        ax.set_ylim(0.0, ly)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle("Pilot pathlines in random periodic OpenFOAM cell", y=0.98)
    fig.tight_layout()
    path = OUT / "random_particle_tracking_pathlines.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_transition_matrices(results: list[TrackingResult], geometry: dict, bins: int) -> Path:
    ly = float(geometry["domain"]["length_y"])
    titles = {
        "no_dlvo": "No DLVO",
        "favorable": "Favorable",
        "unfavorable": "Unfavorable",
    }
    fig, axes = plt.subplots(1, len(results), figsize=(15.2, 4.6), dpi=180, sharex=True, sharey=True)
    if len(results) == 1:
        axes = [axes]
    vmax = 0.35
    image = None
    for ax, result in zip(axes, results):
        yin_bin = np.clip((result.y0 / ly * bins).astype(int), 0, bins - 1)
        matrix = np.zeros((bins, bins), dtype=float)
        for i in range(result.x0.size):
            if not result.exited[i] or not np.isfinite(result.y_out[i]):
                continue
            yout_bin = int(np.clip(result.y_out[i] / ly * bins, 0, bins - 1))
            matrix[yin_bin[i], yout_bin] += 1.0
        row_total = np.bincount(yin_bin, minlength=bins).astype(float)
        probability = np.divide(matrix, row_total[:, None], out=np.zeros_like(matrix), where=row_total[:, None] > 0)
        image = ax.imshow(
            probability.T,
            origin="lower",
            extent=(0.0, ly * 1e3, 0.0, ly * 1e3),
            interpolation="nearest",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(
            f"{titles.get(result.condition, result.condition)}\n"
            f"exit {np.sum(result.exited)}, attach {np.sum(result.attached)}, censor {np.sum(result.censored)}"
        )
        ax.set_xlabel("inlet y (mm)")
    axes[0].set_ylabel("outlet y (mm)")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.015)
        cbar.set_label("row-normalized probability")
    fig.suptitle(f"Pilot y-in to y-out transition matrices ({bins} bins)", y=0.99)
    path = OUT / "random_particle_tracking_transition_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    summaries: list[dict[str, float | int | str]],
    pathline_path: Path,
    matrix_path: Path,
    params: RandomTrackingParams,
    geometry_path: Path,
    flow_case: Path,
    flow_mean_ux_before_scale: float,
    flow_scale: float,
) -> Path:
    lines = [
        "# Random porous particle-tracking pilot",
        "",
        "This pilot uses the refined random OpenFOAM flow field with periodic top/bottom geometry, left-boundary injection, right-boundary exit, Brownian diffusion with wall mobility correction, explicit hard-wall projection, and DLVO normal drift for favorable/unfavorable cases.",
        "",
        f"- Particle radius: {params.particle_radius * 1e6:.3f} um",
        f"- Diffusivity: {params.diffusivity:.4e} m2/s",
        f"- Ionic strength for DLVO cases: {params.ionic_strength_molar * 1e3:.1f} mM",
        f"- Debye length: {params.debye_length * 1e9:.2f} nm",
        f"- Time step: {params.dt:.4g} s",
        f"- Maximum time: {params.max_time:.1f} s",
        f"- Geometry source: `{geometry_path.relative_to(ROOT) if geometry_path.is_absolute() and ROOT in geometry_path.parents else geometry_path}`",
        f"- Flow case: `{flow_case.relative_to(ROOT) if flow_case.is_absolute() and ROOT in flow_case.parents else flow_case}`",
        f"- Area-weighted mean ux before scaling: {flow_mean_ux_before_scale:.4e} m/s",
        f"- Velocity scale factor to {TARGET_M_PER_DAY:.2f} m/day: {flow_scale:.4g}",
        f"- Adaptive near-wall stepping: {params.adaptive_near_wall} (cutoff {params.adaptive_cutoff * 1e9:.1f} nm, target normal step {params.normal_step_target * 1e9:.1f} nm, max substeps {params.max_substeps})",
        f"- Segment collision/reflection: {params.segment_collision}",
        "",
        "| condition | particles | exited | attached | censored | intercepted | median h_min (nm) | p05 h_min (nm) | median near time if intercepted (s) | median exit time (s) | exit effective bins | intercepted-exit effective bins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {condition} | {particles} | {exited} | {attached} | {censored} | {intercepted} | {median_h_min_nm:.2f} | {p05_h_min_nm:.2f} | {median_near_time_intercepted_s:.2f} | {median_travel_time_exited_s:.2f} | {exit_effective_bins:.2f} | {intercepted_exit_effective_bins:.2f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Pathline figure: `{pathline_path.relative_to(ROOT)}`",
            f"Transition-matrix figure: `{matrix_path.relative_to(ROOT)}`",
            "",
            "Next checks should focus on near-wall time-step adaptivity and a stricter line-segment collision/reflection test before large ensembles.",
        ]
    )
    path = OUT / "random_particle_tracking_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-case", type=Path, default=DEFAULT_FLOW_CASE)
    parser.add_argument("--geometry-path", type=Path, default=GEOMETRY_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--particles", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--max-time", type=float, default=80.0)
    parser.add_argument("--matrix-bins", type=int, default=40)
    parser.add_argument("--conditions", nargs="+", default=["no_dlvo", "favorable", "unfavorable"])
    parser.add_argument("--ionic-strength-mM", type=float, default=50.0)
    parser.add_argument("--zeta-unfavorable-mV", type=float, default=-70.0)
    parser.add_argument("--zeta-favorable-mV", type=float, default=70.0)
    parser.add_argument("--hamaker-multiplier", type=float, default=1.0)
    parser.add_argument("--diffusivity-multiplier", type=float, default=1.0)
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--adaptive-cutoff-nm", type=float, default=75.0)
    parser.add_argument("--normal-step-target-nm", type=float, default=3.0)
    parser.add_argument("--max-substeps", type=int, default=50)
    parser.add_argument("--no-segment-collision", action="store_true")
    parser.add_argument("--target-mean-velocity-m-per-day", type=float, default=TARGET_M_PER_DAY)
    parser.add_argument("--no-flow-origin-shift", action="store_true")
    parser.add_argument("--no-rescale-flow", action="store_true")
    return parser.parse_args()


def main() -> None:
    global OUT
    args = parse_args()
    OUT = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    OUT = OUT.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    flow_case = args.flow_case if args.flow_case.is_absolute() else ROOT / args.flow_case
    geometry_path = args.geometry_path if args.geometry_path.is_absolute() else ROOT / args.geometry_path
    geometry = load_geometry(geometry_path, flow_case=flow_case, apply_flow_shift=not args.no_flow_origin_shift)
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = latest_vtu(flow_case)
    centers_xy, velocity_xy, areas = flow_io.parse_xml_vtu(vtu)
    flow_mean_ux_before_scale = float("nan")
    flow_scale = 1.0
    if not args.no_rescale_flow:
        velocity_xy, flow_mean_ux_before_scale, flow_scale = rescale_velocity_to_target(
            velocity_xy,
            areas,
            args.target_mean_velocity_m_per_day,
        )
    interpolator = PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8)
    params = RandomTrackingParams(
        dt=args.dt,
        max_time=args.max_time,
        ionic_strength_molar=args.ionic_strength_mM * 1.0e-3,
        zeta_collector_unfavorable=args.zeta_unfavorable_mV * 1.0e-3,
        zeta_collector_favorable=args.zeta_favorable_mV * 1.0e-3,
        hamaker=RandomTrackingParams.hamaker * args.hamaker_multiplier,
        diffusivity_multiplier=args.diffusivity_multiplier,
        adaptive_near_wall=not args.no_adaptive,
        adaptive_cutoff=args.adaptive_cutoff_nm * 1.0e-9,
        normal_step_target=args.normal_step_target_nm * 1.0e-9,
        max_substeps=args.max_substeps,
        segment_collision=not args.no_segment_collision,
    )

    rng = np.random.default_rng(args.seed)
    centers = np.array([[g["x"], g["y"]] for g in geometry["grains"]], dtype=float)
    radii = np.array([g["radius"] for g in geometry["grains"]], dtype=float)
    initial_y = open_inlet_samples(rng, args.particles, lx, ly, centers, radii, params, 1.0e-6)

    results = []
    summaries = []
    for condition_id, condition in enumerate(args.conditions):
        result = simulate_particles(
            condition,
            interpolator,
            geometry,
            params,
            n_particles=args.particles,
            seed=args.seed + 1000 * (condition_id + 1),
            initial_y=initial_y,
        )
        results.append(result)
        summaries.append(summarize(result, params, ly, args.matrix_bins))
    write_outputs(results, summaries)
    pathline_path = plot_pathlines(results, geometry, params)
    matrix_path = plot_transition_matrices(results, geometry, args.matrix_bins)
    report_path = write_report(
        summaries,
        pathline_path,
        matrix_path,
        params,
        geometry_path,
        flow_case,
        flow_mean_ux_before_scale,
        flow_scale,
    )
    print(report_path)
    print(pathline_path)
    print(matrix_path)


if __name__ == "__main__":
    main()
