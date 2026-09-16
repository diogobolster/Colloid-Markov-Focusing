"""Compiled particle-tracking backend for the physical periodic-cell model."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .physical import FlowField, PhysicalParams, TrajectoryLibrary, secondary_minimum_well


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "colloid_tsm" / "native" / "particle_kernel.c"
LIB_PATH = ROOT / "colloid_tsm" / "native" / (
    "libparticle_kernel.dylib" if sys.platform == "darwin" else "libparticle_kernel.so"
)


class _KernelParams(ctypes.Structure):
    _fields_ = [
        ("cell_length", ctypes.c_double),
        ("grain_radius", ctypes.c_double),
        ("particle_radius", ctypes.c_double),
        ("mean_velocity", ctypes.c_double),
        ("temperature", ctypes.c_double),
        ("viscosity", ctypes.c_double),
        ("relative_permittivity", ctypes.c_double),
        ("ionic_strength_molar", ctypes.c_double),
        ("hamaker", ctypes.c_double),
        ("zeta_particle", ctypes.c_double),
        ("zeta_collector_favorable", ctypes.c_double),
        ("zeta_collector_unfavorable", ctypes.c_double),
        ("near_surface", ctypes.c_double),
        ("contact_gap", ctypes.c_double),
        ("min_gap", ctypes.c_double),
        ("dt", ctypes.c_double),
        ("max_time", ctypes.c_double),
        ("dlvo_velocity_cap", ctypes.c_double),
        ("surface_sliding_gap", ctypes.c_double),
        ("wall_mobility_cutoff", ctypes.c_double),
        ("wall_mobility_normal_floor", ctypes.c_double),
        ("wall_mobility_parallel_floor", ctypes.c_double),
        ("wall_mobility_hydro_strength", ctypes.c_double),
        ("diffusivity_multiplier", ctypes.c_double),
        ("resolved_langevin_substep_cutoff", ctypes.c_double),
        ("resolved_langevin_normal_step", ctypes.c_double),
        ("resolved_langevin_min_dt", ctypes.c_double),
        ("diffusivity", ctypes.c_double),
        ("debye_length", ctypes.c_double),
        ("exclusion_radius", ctypes.c_double),
        ("inlet_y_min", ctypes.c_double),
        ("inlet_y_max", ctypes.c_double),
        ("well_lower_gap", ctypes.c_double),
        ("well_upper_gap", ctypes.c_double),
        ("well_basin_lower_gap", ctypes.c_double),
        ("well_basin_upper_gap", ctypes.c_double),
        ("well_minimum_gap", ctypes.c_double),
        ("well_minimum_potential_kbt", ctypes.c_double),
        ("resolved_langevin_hydro_strength", ctypes.c_double),
        ("wall_mobility_thermal_drift", ctypes.c_int),
        ("resolved_langevin_substeps", ctypes.c_int),
        ("resolved_langevin_max_substeps", ctypes.c_int),
        ("has_secondary_well", ctypes.c_int),
    ]


_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_UINT8_PTR = ctypes.POINTER(ctypes.c_uint8)
_INT32_PTR = ctypes.POINTER(ctypes.c_int32)
_LOADED_LIBRARY: ctypes.CDLL | None = None


@dataclass
class RandomFlowGrid:
    """Regular-grid representation of a resolved random-geometry flow field."""

    lx: float
    ly: float
    ux: np.ndarray
    uy: np.ndarray

    @property
    def nx(self) -> int:
        return int(self.ux.shape[0])

    @property
    def ny(self) -> int:
        return int(self.ux.shape[1])


@dataclass
class RandomTrajectoryLibrary:
    """Compiled tracking result for an arbitrary periodic disk packing."""

    x0: np.ndarray
    y0: np.ndarray
    y_out: np.ndarray
    travel_time: np.ndarray
    dy: np.ndarray
    attached: np.ndarray
    exited: np.ndarray
    censored: np.ndarray
    interceptions: np.ndarray
    near_time: np.ndarray
    h_min: np.ndarray
    theta_entry: np.ndarray
    theta_exit: np.ndarray
    theta_final: np.ndarray
    grain_entry: np.ndarray
    grain_exit: np.ndarray
    grain_final: np.ndarray
    contact_events: np.ndarray
    x_final: np.ndarray
    y_final: np.ndarray
    condition: str
    params: PhysicalParams


def build_kernel(force: bool = False) -> Path:
    """Compile the C particle kernel if needed and return the shared-library path."""

    if not SRC_PATH.exists():
        raise FileNotFoundError(SRC_PATH)
    needs_build = force or not LIB_PATH.exists() or SRC_PATH.stat().st_mtime > LIB_PATH.stat().st_mtime
    if not needs_build:
        return LIB_PATH
    cmd = ["cc", "-O3", "-std=c99", "-fPIC"]
    if sys.platform == "darwin":
        cmd.append("-dynamiclib")
    else:
        cmd.append("-shared")
    cmd.extend([str(SRC_PATH), "-o", str(LIB_PATH), "-lm"])
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return LIB_PATH


def load_kernel(force_rebuild: bool = False) -> ctypes.CDLL:
    """Load the compiled kernel and declare its ABI."""

    global _LOADED_LIBRARY
    if force_rebuild or _LOADED_LIBRARY is None:
        lib_path = build_kernel(force=force_rebuild)
        lib = ctypes.CDLL(str(lib_path))
        lib.simulate_particle_kernel.argtypes = [
            ctypes.POINTER(_KernelParams),
            ctypes.c_int,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _UINT8_PTR,
            _UINT8_PTR,
            _UINT8_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
        ]
        lib.simulate_particle_kernel.restype = ctypes.c_int
        lib.simulate_random_particle_kernel.argtypes = [
            ctypes.POINTER(_KernelParams),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            ctypes.c_int,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _UINT8_PTR,
            _UINT8_PTR,
            _UINT8_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
        ]
        lib.simulate_random_particle_kernel.restype = ctypes.c_int
        lib.simulate_random_occupancy_kernel.argtypes = [
            ctypes.POINTER(_KernelParams),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            ctypes.c_int,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _UINT8_PTR,
            _UINT8_PTR,
            _UINT8_PTR,
            _INT32_PTR,
            _DOUBLE_PTR,
            _DOUBLE_PTR,
            _INT32_PTR,
        ]
        lib.simulate_random_occupancy_kernel.restype = ctypes.c_int
        _LOADED_LIBRARY = lib
    return _LOADED_LIBRARY


def _as_kernel_params(params: PhysicalParams, condition: str) -> _KernelParams:
    well = secondary_minimum_well(params, condition)
    return _KernelParams(
        cell_length=params.cell_length,
        grain_radius=params.grain_radius,
        particle_radius=params.particle_radius,
        mean_velocity=params.mean_velocity,
        temperature=params.temperature,
        viscosity=params.viscosity,
        relative_permittivity=params.relative_permittivity,
        ionic_strength_molar=params.ionic_strength_molar,
        hamaker=params.hamaker,
        zeta_particle=params.zeta_particle,
        zeta_collector_favorable=params.zeta_collector_favorable,
        zeta_collector_unfavorable=params.zeta_collector_unfavorable,
        near_surface=params.near_surface,
        contact_gap=params.contact_gap,
        min_gap=params.min_gap,
        dt=params.dt,
        max_time=params.max_time,
        dlvo_velocity_cap=params.dlvo_velocity_cap,
        surface_sliding_gap=params.surface_sliding_gap,
        wall_mobility_cutoff=params.wall_mobility_cutoff,
        wall_mobility_normal_floor=params.wall_mobility_normal_floor,
        wall_mobility_parallel_floor=params.wall_mobility_parallel_floor,
        wall_mobility_hydro_strength=params.wall_mobility_hydro_strength,
        diffusivity_multiplier=params.diffusivity_multiplier,
        resolved_langevin_substep_cutoff=params.resolved_langevin_substep_cutoff,
        resolved_langevin_normal_step=params.resolved_langevin_normal_step,
        resolved_langevin_min_dt=params.resolved_langevin_min_dt,
        diffusivity=params.diffusivity,
        debye_length=params.debye_length,
        exclusion_radius=params.exclusion_radius,
        inlet_y_min=params.inlet_y_min,
        inlet_y_max=params.inlet_y_max,
        well_lower_gap=float(well["lower"]),
        well_upper_gap=float(well["upper"]),
        well_basin_lower_gap=float(well["basin_lower"]),
        well_basin_upper_gap=float(well["basin_upper"]),
        well_minimum_gap=float(well["minimum_gap"]),
        well_minimum_potential_kbt=float(well["minimum_potential_kbt"]),
        resolved_langevin_hydro_strength=params.resolved_langevin_hydro_strength,
        wall_mobility_thermal_drift=int(params.wall_mobility_thermal_drift),
        resolved_langevin_substeps=params.resolved_langevin_substeps,
        resolved_langevin_max_substeps=params.resolved_langevin_max_substeps,
        has_secondary_well=int(bool(well["has_well"])),
    )


def _condition_code(condition: str) -> int:
    if condition in {"neutral", "no_dlvo", "advection_diffusion"}:
        return 0
    if condition.startswith("favorable"):
        return 1
    if condition.startswith("unfavorable"):
        return 2
    raise ValueError(f"unsupported condition for compiled kernel: {condition!r}")


def _surface_mode_code(surface_mode: str) -> int:
    if surface_mode == "sliding":
        return 1
    if surface_mode == "lubrication":
        return 2
    if surface_mode == "resolved_langevin":
        return 3
    raise ValueError("compiled kernel currently supports 'sliding', 'lubrication', and 'resolved_langevin'")


def _ptr(array: np.ndarray, c_type: type[ctypes._SimpleCData]) -> ctypes.POINTER:  # type: ignore[name-defined]
    return array.ctypes.data_as(ctypes.POINTER(c_type))


def simulate_cell_transitions_compiled(
    flow: FlowField,
    condition: str,
    n_particles: int = 5000,
    seed: int = 0,
    allow_attachment: bool | None = None,
    initial_x: np.ndarray | None = None,
    initial_y: np.ndarray | None = None,
    surface_mode: str = "resolved_langevin",
    force_rebuild: bool = False,
    return_timestep_diagnostics: bool = False,
    stop_at_next_distinct: bool = False,
    return_entry_diagnostics: bool = False,
) -> TrajectoryLibrary | tuple[TrajectoryLibrary, dict[str, np.ndarray]]:
    """Track particles with the compiled C backend.

    The returned object matches :func:`colloid_tsm.physical.simulate_cell_transitions`.
    The compiled path is intended for production runs of the lubrication and
    resolved near-wall Langevin closures.
    """

    params = flow.params
    if allow_attachment is None:
        allow_attachment = condition == "favorable"

    rng = np.random.default_rng(seed)
    if initial_y is not None:
        y0 = np.ascontiguousarray(initial_y, dtype=np.float64)
        n_particles = int(y0.size)
    else:
        if params.inlet_y_min >= params.inlet_y_max:
            raise ValueError("collector geometry closes the inlet aperture")
        y0 = np.ascontiguousarray(
            rng.uniform(params.inlet_y_min, params.inlet_y_max, size=n_particles),
            dtype=np.float64,
        )
    if initial_x is not None:
        x0 = np.ascontiguousarray(initial_x, dtype=np.float64)
        if x0.size != n_particles:
            raise ValueError("initial_x must have the same length as initial_y")
    else:
        x0 = np.zeros(n_particles, dtype=np.float64)

    ux = np.ascontiguousarray(flow.ux, dtype=np.float64)
    uy = np.ascontiguousarray(flow.uy, dtype=np.float64)
    if ux.shape != (flow.resolution, flow.resolution) or uy.shape != ux.shape:
        raise ValueError("flow velocity arrays must be square and match flow.resolution")

    y_in = np.empty(n_particles, dtype=np.float64)
    y_out = np.empty(n_particles, dtype=np.float64)
    travel_time = np.empty(n_particles, dtype=np.float64)
    dy = np.empty(n_particles, dtype=np.float64)
    attached_u8 = np.empty(n_particles, dtype=np.uint8)
    exited_u8 = np.empty(n_particles, dtype=np.uint8)
    censored_u8 = np.empty(n_particles, dtype=np.uint8)
    interceptions = np.empty(n_particles, dtype=np.int32)
    near_time = np.empty(n_particles, dtype=np.float64)
    h_min = np.empty(n_particles, dtype=np.float64)
    theta_entry = np.empty(n_particles, dtype=np.float64)
    theta_exit = np.empty(n_particles, dtype=np.float64)
    theta_final = np.empty(n_particles, dtype=np.float64)
    collector_entry = np.empty(n_particles, dtype=np.int32)
    collector_exit = np.empty(n_particles, dtype=np.int32)
    collector_final = np.empty(n_particles, dtype=np.int32)
    center_interceptions = np.empty(n_particles, dtype=np.int32)
    corner_interceptions = np.empty(n_particles, dtype=np.int32)
    contact_events = np.empty(n_particles, dtype=np.int32)
    x_final = np.empty(n_particles, dtype=np.float64)
    y_final = np.empty(n_particles, dtype=np.float64)
    well_interceptions = np.empty(n_particles, dtype=np.int32)
    well_time = np.empty(n_particles, dtype=np.float64)
    theta_well_entry = np.empty(n_particles, dtype=np.float64)
    theta_well_exit = np.empty(n_particles, dtype=np.float64)
    theta_well_final = np.empty(n_particles, dtype=np.float64)
    collector_well_entry = np.empty(n_particles, dtype=np.int32)
    collector_well_exit = np.empty(n_particles, dtype=np.int32)
    collector_well_final = np.empty(n_particles, dtype=np.int32)
    center_well_interceptions = np.empty(n_particles, dtype=np.int32)
    corner_well_interceptions = np.empty(n_particles, dtype=np.int32)
    well_angular_travel = np.empty(n_particles, dtype=np.float64)
    well_net_angular_travel = np.empty(n_particles, dtype=np.float64)
    near_wall_outer_steps = np.empty(n_particles, dtype=np.int32)
    adaptive_substeps = np.empty(n_particles, dtype=np.int32)
    guard_hits = np.empty(n_particles, dtype=np.int32)
    min_dt_hits = np.empty(n_particles, dtype=np.int32)
    sub_dt_sum = np.empty(n_particles, dtype=np.float64)
    sub_dt_sq_sum = np.empty(n_particles, dtype=np.float64)
    min_sub_dt = np.empty(n_particles, dtype=np.float64)
    max_sub_dt = np.empty(n_particles, dtype=np.float64)
    max_inward_det_normal_step = np.empty(n_particles, dtype=np.float64)
    max_brownian_normal_std = np.empty(n_particles, dtype=np.float64)
    first_entry_theta = np.empty(n_particles, dtype=np.float64)
    first_entry_time = np.empty(n_particles, dtype=np.float64)
    first_entry_x = np.empty(n_particles, dtype=np.float64)
    first_entry_y = np.empty(n_particles, dtype=np.float64)
    first_entry_collector_cx = np.empty(n_particles, dtype=np.float64)
    first_entry_collector_cy = np.empty(n_particles, dtype=np.float64)
    first_entry_collector = np.empty(n_particles, dtype=np.int32)
    next_entry_theta = np.empty(n_particles, dtype=np.float64)
    next_entry_time = np.empty(n_particles, dtype=np.float64)
    next_entry_x = np.empty(n_particles, dtype=np.float64)
    next_entry_y = np.empty(n_particles, dtype=np.float64)
    next_entry_collector_cx = np.empty(n_particles, dtype=np.float64)
    next_entry_collector_cy = np.empty(n_particles, dtype=np.float64)
    next_entry_collector = np.empty(n_particles, dtype=np.int32)
    departure_theta_to_next = np.empty(n_particles, dtype=np.float64)
    departure_time_to_next = np.empty(n_particles, dtype=np.float64)
    first_collector_near_time = np.empty(n_particles, dtype=np.float64)
    same_collector_reentries_before_next = np.empty(n_particles, dtype=np.int32)

    lib = load_kernel(force_rebuild=force_rebuild)
    c_params = _as_kernel_params(params, condition)
    status = lib.simulate_particle_kernel(
        ctypes.byref(c_params),
        ctypes.c_int(flow.resolution),
        _ptr(ux, ctypes.c_double),
        _ptr(uy, ctypes.c_double),
        ctypes.c_int(_condition_code(condition)),
        ctypes.c_int(_surface_mode_code(surface_mode)),
        ctypes.c_int(int(allow_attachment)),
        ctypes.c_int(int(stop_at_next_distinct)),
        ctypes.c_int(n_particles),
        ctypes.c_uint64(seed),
        _ptr(x0, ctypes.c_double),
        _ptr(y0, ctypes.c_double),
        _ptr(y_in, ctypes.c_double),
        _ptr(y_out, ctypes.c_double),
        _ptr(travel_time, ctypes.c_double),
        _ptr(dy, ctypes.c_double),
        _ptr(attached_u8, ctypes.c_uint8),
        _ptr(exited_u8, ctypes.c_uint8),
        _ptr(censored_u8, ctypes.c_uint8),
        _ptr(interceptions, ctypes.c_int32),
        _ptr(near_time, ctypes.c_double),
        _ptr(h_min, ctypes.c_double),
        _ptr(theta_entry, ctypes.c_double),
        _ptr(theta_exit, ctypes.c_double),
        _ptr(theta_final, ctypes.c_double),
        _ptr(collector_entry, ctypes.c_int32),
        _ptr(collector_exit, ctypes.c_int32),
        _ptr(collector_final, ctypes.c_int32),
        _ptr(center_interceptions, ctypes.c_int32),
        _ptr(corner_interceptions, ctypes.c_int32),
        _ptr(contact_events, ctypes.c_int32),
        _ptr(x_final, ctypes.c_double),
        _ptr(y_final, ctypes.c_double),
        _ptr(well_interceptions, ctypes.c_int32),
        _ptr(well_time, ctypes.c_double),
        _ptr(theta_well_entry, ctypes.c_double),
        _ptr(theta_well_exit, ctypes.c_double),
        _ptr(theta_well_final, ctypes.c_double),
        _ptr(collector_well_entry, ctypes.c_int32),
        _ptr(collector_well_exit, ctypes.c_int32),
        _ptr(collector_well_final, ctypes.c_int32),
        _ptr(center_well_interceptions, ctypes.c_int32),
        _ptr(corner_well_interceptions, ctypes.c_int32),
        _ptr(well_angular_travel, ctypes.c_double),
        _ptr(well_net_angular_travel, ctypes.c_double),
        _ptr(near_wall_outer_steps, ctypes.c_int32),
        _ptr(adaptive_substeps, ctypes.c_int32),
        _ptr(guard_hits, ctypes.c_int32),
        _ptr(min_dt_hits, ctypes.c_int32),
        _ptr(sub_dt_sum, ctypes.c_double),
        _ptr(sub_dt_sq_sum, ctypes.c_double),
        _ptr(min_sub_dt, ctypes.c_double),
        _ptr(max_sub_dt, ctypes.c_double),
        _ptr(max_inward_det_normal_step, ctypes.c_double),
        _ptr(max_brownian_normal_std, ctypes.c_double),
        _ptr(first_entry_theta, ctypes.c_double),
        _ptr(first_entry_time, ctypes.c_double),
        _ptr(first_entry_x, ctypes.c_double),
        _ptr(first_entry_y, ctypes.c_double),
        _ptr(first_entry_collector_cx, ctypes.c_double),
        _ptr(first_entry_collector_cy, ctypes.c_double),
        _ptr(first_entry_collector, ctypes.c_int32),
        _ptr(next_entry_theta, ctypes.c_double),
        _ptr(next_entry_time, ctypes.c_double),
        _ptr(next_entry_x, ctypes.c_double),
        _ptr(next_entry_y, ctypes.c_double),
        _ptr(next_entry_collector_cx, ctypes.c_double),
        _ptr(next_entry_collector_cy, ctypes.c_double),
        _ptr(next_entry_collector, ctypes.c_int32),
        _ptr(departure_theta_to_next, ctypes.c_double),
        _ptr(departure_time_to_next, ctypes.c_double),
        _ptr(first_collector_near_time, ctypes.c_double),
        _ptr(same_collector_reentries_before_next, ctypes.c_int32),
    )
    if status != 0:
        raise RuntimeError(f"compiled particle kernel failed with status {status}")

    library = TrajectoryLibrary(
        y_in=y_in,
        y_out=y_out,
        travel_time=travel_time,
        dy=dy,
        attached=attached_u8.astype(bool),
        exited=exited_u8.astype(bool),
        censored=censored_u8.astype(bool),
        interceptions=interceptions.astype(int),
        near_time=near_time,
        h_min=h_min,
        theta_entry=theta_entry,
        theta_exit=theta_exit,
        theta_final=theta_final,
        collector_entry=collector_entry.astype(int),
        collector_exit=collector_exit.astype(int),
        collector_final=collector_final.astype(int),
        center_interceptions=center_interceptions.astype(int),
        corner_interceptions=corner_interceptions.astype(int),
        contact_events=contact_events.astype(int),
        x_final=x_final,
        y_final=y_final,
        condition=condition,
        params=params,
        resolution=flow.resolution,
        surface_mode=surface_mode,
        well_interceptions=well_interceptions.astype(int),
        well_time=well_time,
        theta_well_entry=theta_well_entry,
        theta_well_exit=theta_well_exit,
        theta_well_final=theta_well_final,
        collector_well_entry=collector_well_entry.astype(int),
        collector_well_exit=collector_well_exit.astype(int),
        collector_well_final=collector_well_final.astype(int),
        center_well_interceptions=center_well_interceptions.astype(int),
        corner_well_interceptions=corner_well_interceptions.astype(int),
        well_angular_travel=well_angular_travel,
        well_net_angular_travel=well_net_angular_travel,
        well_gap_lower=c_params.well_lower_gap,
        well_gap_upper=c_params.well_upper_gap,
        well_gap_minimum=c_params.well_minimum_gap,
        well_potential_minimum_kbt=c_params.well_minimum_potential_kbt,
        well_basin_lower=c_params.well_basin_lower_gap,
        well_basin_upper=c_params.well_basin_upper_gap,
    )
    if not return_timestep_diagnostics and not return_entry_diagnostics:
        return library
    diagnostics: dict[str, np.ndarray] = {}
    if return_timestep_diagnostics:
        diagnostics.update({
            "near_wall_outer_steps": near_wall_outer_steps.astype(int),
            "adaptive_substeps": adaptive_substeps.astype(int),
            "guard_hits": guard_hits.astype(int),
            "min_dt_hits": min_dt_hits.astype(int),
            "sub_dt_sum": sub_dt_sum,
            "sub_dt_sq_sum": sub_dt_sq_sum,
            "min_sub_dt": min_sub_dt,
            "max_sub_dt": max_sub_dt,
            "max_inward_det_normal_step": max_inward_det_normal_step,
            "max_brownian_normal_std": max_brownian_normal_std,
        })
    if return_entry_diagnostics:
        diagnostics.update({
            "first_entry_theta": first_entry_theta,
            "first_entry_time": first_entry_time,
            "first_entry_x": first_entry_x,
            "first_entry_y": first_entry_y,
            "first_entry_collector_cx": first_entry_collector_cx,
            "first_entry_collector_cy": first_entry_collector_cy,
            "first_entry_collector": first_entry_collector.astype(int),
            "next_entry_theta": next_entry_theta,
            "next_entry_time": next_entry_time,
            "next_entry_x": next_entry_x,
            "next_entry_y": next_entry_y,
            "next_entry_collector_cx": next_entry_collector_cx,
            "next_entry_collector_cy": next_entry_collector_cy,
            "next_entry_collector": next_entry_collector.astype(int),
            "departure_theta_to_next": departure_theta_to_next,
            "departure_time_to_next": departure_time_to_next,
            "first_collector_near_time": first_collector_near_time,
            "same_collector_reentries_before_next": same_collector_reentries_before_next.astype(int),
            "pair_completed": next_entry_collector >= 0,
        })
    return library, diagnostics


def simulate_random_transitions_compiled(
    flow: RandomFlowGrid,
    geometry: dict,
    params: PhysicalParams,
    condition: str,
    n_particles: int = 5000,
    seed: int = 0,
    allow_attachment: bool | None = None,
    initial_x: np.ndarray | None = None,
    initial_y: np.ndarray | None = None,
    force_rebuild: bool = False,
    return_timestep_diagnostics: bool = False,
    stop_at_next_distinct: bool = False,
    return_entry_diagnostics: bool = False,
) -> RandomTrajectoryLibrary | tuple[RandomTrajectoryLibrary, dict[str, np.ndarray]]:
    """Track particles in an arbitrary periodic disk packing with the C backend."""

    if allow_attachment is None:
        allow_attachment = condition == "favorable"
    ux = np.ascontiguousarray(flow.ux, dtype=np.float64)
    uy = np.ascontiguousarray(flow.uy, dtype=np.float64)
    if ux.ndim != 2 or uy.shape != ux.shape:
        raise ValueError("random flow grid ux/uy must be two-dimensional arrays with matching shape")
    if flow.lx <= 0.0 or flow.ly <= 0.0:
        raise ValueError("random flow grid dimensions must be positive")

    grains = geometry.get("grains", [])
    if not grains:
        raise ValueError("random geometry must contain at least one grain")
    grain_x = np.ascontiguousarray([float(g["x"]) for g in grains], dtype=np.float64)
    grain_y = np.ascontiguousarray([float(g["y"]) for g in grains], dtype=np.float64)
    grain_r = np.ascontiguousarray([float(g["radius"]) for g in grains], dtype=np.float64)
    n_grains = int(grain_x.size)

    rng = np.random.default_rng(seed)
    if initial_y is not None:
        y0 = np.ascontiguousarray(initial_y, dtype=np.float64)
        n_particles = int(y0.size)
    else:
        y0 = np.ascontiguousarray(rng.uniform(0.0, flow.ly, size=n_particles), dtype=np.float64)
    if initial_x is not None:
        x0 = np.ascontiguousarray(initial_x, dtype=np.float64)
        if x0.size != n_particles:
            raise ValueError("initial_x must have the same length as initial_y")
    else:
        x0 = np.full(n_particles, 1.0e-6, dtype=np.float64)

    y_in = np.empty(n_particles, dtype=np.float64)
    y_out = np.empty(n_particles, dtype=np.float64)
    travel_time = np.empty(n_particles, dtype=np.float64)
    dy = np.empty(n_particles, dtype=np.float64)
    attached_u8 = np.empty(n_particles, dtype=np.uint8)
    exited_u8 = np.empty(n_particles, dtype=np.uint8)
    censored_u8 = np.empty(n_particles, dtype=np.uint8)
    interceptions = np.empty(n_particles, dtype=np.int32)
    near_time = np.empty(n_particles, dtype=np.float64)
    h_min = np.empty(n_particles, dtype=np.float64)
    theta_entry = np.empty(n_particles, dtype=np.float64)
    theta_exit = np.empty(n_particles, dtype=np.float64)
    theta_final = np.empty(n_particles, dtype=np.float64)
    grain_entry = np.empty(n_particles, dtype=np.int32)
    grain_exit = np.empty(n_particles, dtype=np.int32)
    grain_final = np.empty(n_particles, dtype=np.int32)
    contact_events = np.empty(n_particles, dtype=np.int32)
    x_final = np.empty(n_particles, dtype=np.float64)
    y_final = np.empty(n_particles, dtype=np.float64)
    near_wall_outer_steps = np.empty(n_particles, dtype=np.int32)
    adaptive_substeps = np.empty(n_particles, dtype=np.int32)
    guard_hits = np.empty(n_particles, dtype=np.int32)
    min_dt_hits = np.empty(n_particles, dtype=np.int32)
    max_inward_det_normal_step = np.empty(n_particles, dtype=np.float64)
    max_brownian_normal_std = np.empty(n_particles, dtype=np.float64)
    first_entry_theta = np.empty(n_particles, dtype=np.float64)
    first_entry_time = np.empty(n_particles, dtype=np.float64)
    first_entry_x = np.empty(n_particles, dtype=np.float64)
    first_entry_y = np.empty(n_particles, dtype=np.float64)
    first_entry_collector_cx = np.empty(n_particles, dtype=np.float64)
    first_entry_collector_cy = np.empty(n_particles, dtype=np.float64)
    first_entry_collector = np.empty(n_particles, dtype=np.int32)
    next_entry_theta = np.empty(n_particles, dtype=np.float64)
    next_entry_time = np.empty(n_particles, dtype=np.float64)
    next_entry_x = np.empty(n_particles, dtype=np.float64)
    next_entry_y = np.empty(n_particles, dtype=np.float64)
    next_entry_collector_cx = np.empty(n_particles, dtype=np.float64)
    next_entry_collector_cy = np.empty(n_particles, dtype=np.float64)
    next_entry_collector = np.empty(n_particles, dtype=np.int32)
    departure_theta_to_next = np.empty(n_particles, dtype=np.float64)
    departure_time_to_next = np.empty(n_particles, dtype=np.float64)
    first_collector_near_time = np.empty(n_particles, dtype=np.float64)
    same_collector_reentries_before_next = np.empty(n_particles, dtype=np.int32)

    lib = load_kernel(force_rebuild=force_rebuild)
    c_params = _as_kernel_params(params, condition)
    status = lib.simulate_random_particle_kernel(
        ctypes.byref(c_params),
        ctypes.c_double(flow.lx),
        ctypes.c_double(flow.ly),
        ctypes.c_int(flow.nx),
        ctypes.c_int(flow.ny),
        _ptr(ux, ctypes.c_double),
        _ptr(uy, ctypes.c_double),
        ctypes.c_int(n_grains),
        _ptr(grain_x, ctypes.c_double),
        _ptr(grain_y, ctypes.c_double),
        _ptr(grain_r, ctypes.c_double),
        ctypes.c_int(_condition_code(condition)),
        ctypes.c_int(int(allow_attachment)),
        ctypes.c_int(int(stop_at_next_distinct)),
        ctypes.c_int(n_particles),
        ctypes.c_uint64(seed),
        _ptr(x0, ctypes.c_double),
        _ptr(y0, ctypes.c_double),
        _ptr(y_in, ctypes.c_double),
        _ptr(y_out, ctypes.c_double),
        _ptr(travel_time, ctypes.c_double),
        _ptr(dy, ctypes.c_double),
        _ptr(attached_u8, ctypes.c_uint8),
        _ptr(exited_u8, ctypes.c_uint8),
        _ptr(censored_u8, ctypes.c_uint8),
        _ptr(interceptions, ctypes.c_int32),
        _ptr(near_time, ctypes.c_double),
        _ptr(h_min, ctypes.c_double),
        _ptr(theta_entry, ctypes.c_double),
        _ptr(theta_exit, ctypes.c_double),
        _ptr(theta_final, ctypes.c_double),
        _ptr(grain_entry, ctypes.c_int32),
        _ptr(grain_exit, ctypes.c_int32),
        _ptr(grain_final, ctypes.c_int32),
        _ptr(contact_events, ctypes.c_int32),
        _ptr(x_final, ctypes.c_double),
        _ptr(y_final, ctypes.c_double),
        _ptr(near_wall_outer_steps, ctypes.c_int32),
        _ptr(adaptive_substeps, ctypes.c_int32),
        _ptr(guard_hits, ctypes.c_int32),
        _ptr(min_dt_hits, ctypes.c_int32),
        _ptr(max_inward_det_normal_step, ctypes.c_double),
        _ptr(max_brownian_normal_std, ctypes.c_double),
        _ptr(first_entry_theta, ctypes.c_double),
        _ptr(first_entry_time, ctypes.c_double),
        _ptr(first_entry_x, ctypes.c_double),
        _ptr(first_entry_y, ctypes.c_double),
        _ptr(first_entry_collector_cx, ctypes.c_double),
        _ptr(first_entry_collector_cy, ctypes.c_double),
        _ptr(first_entry_collector, ctypes.c_int32),
        _ptr(next_entry_theta, ctypes.c_double),
        _ptr(next_entry_time, ctypes.c_double),
        _ptr(next_entry_x, ctypes.c_double),
        _ptr(next_entry_y, ctypes.c_double),
        _ptr(next_entry_collector_cx, ctypes.c_double),
        _ptr(next_entry_collector_cy, ctypes.c_double),
        _ptr(next_entry_collector, ctypes.c_int32),
        _ptr(departure_theta_to_next, ctypes.c_double),
        _ptr(departure_time_to_next, ctypes.c_double),
        _ptr(first_collector_near_time, ctypes.c_double),
        _ptr(same_collector_reentries_before_next, ctypes.c_int32),
    )
    if status != 0:
        raise RuntimeError(f"compiled random particle kernel failed with status {status}")

    library = RandomTrajectoryLibrary(
        x0=x0,
        y0=y_in,
        y_out=y_out,
        travel_time=travel_time,
        dy=dy,
        attached=attached_u8.astype(bool),
        exited=exited_u8.astype(bool),
        censored=censored_u8.astype(bool),
        interceptions=interceptions.astype(int),
        near_time=near_time,
        h_min=h_min,
        theta_entry=theta_entry,
        theta_exit=theta_exit,
        theta_final=theta_final,
        grain_entry=grain_entry.astype(int),
        grain_exit=grain_exit.astype(int),
        grain_final=grain_final.astype(int),
        contact_events=contact_events.astype(int),
        x_final=x_final,
        y_final=y_final,
        condition=condition,
        params=params,
    )
    if not return_timestep_diagnostics and not return_entry_diagnostics:
        return library
    diagnostics: dict[str, np.ndarray] = {}
    if return_timestep_diagnostics:
        diagnostics.update({
            "near_wall_outer_steps": near_wall_outer_steps.astype(int),
            "adaptive_substeps": adaptive_substeps.astype(int),
            "guard_hits": guard_hits.astype(int),
            "min_dt_hits": min_dt_hits.astype(int),
            "max_inward_det_normal_step": max_inward_det_normal_step,
            "max_brownian_normal_std": max_brownian_normal_std,
        })
    if return_entry_diagnostics:
        diagnostics.update({
            "first_entry_theta": first_entry_theta,
            "first_entry_time": first_entry_time,
            "first_entry_x": first_entry_x,
            "first_entry_y": first_entry_y,
            "first_entry_collector_cx": first_entry_collector_cx,
            "first_entry_collector_cy": first_entry_collector_cy,
            "first_entry_collector": first_entry_collector.astype(int),
            "next_entry_theta": next_entry_theta,
            "next_entry_time": next_entry_time,
            "next_entry_x": next_entry_x,
            "next_entry_y": next_entry_y,
            "next_entry_collector_cx": next_entry_collector_cx,
            "next_entry_collector_cy": next_entry_collector_cy,
            "next_entry_collector": next_entry_collector.astype(int),
            "departure_theta_to_next": departure_theta_to_next,
            "departure_time_to_next": departure_time_to_next,
            "first_collector_near_time": first_collector_near_time,
            "same_collector_reentries_before_next": same_collector_reentries_before_next.astype(int),
            "pair_completed": next_entry_collector >= 0,
        })
    return library, diagnostics


def simulate_random_occupancy_compiled(
    flow: RandomFlowGrid,
    geometry: dict,
    params: PhysicalParams,
    condition: str,
    *,
    n_particles: int = 5000,
    seed: int = 0,
    allow_attachment: bool | None = None,
    initial_x: np.ndarray | None = None,
    initial_y: np.ndarray | None = None,
    pore_nx: int = 96,
    pore_ny: int = 64,
    surface_bins: int = 36,
    sample_stride: int = 10,
    force_rebuild: bool = False,
) -> dict[str, np.ndarray]:
    """Track random-packing particles and return unique-visit occupancy counts."""

    if allow_attachment is None:
        allow_attachment = condition == "favorable"
    ux = np.ascontiguousarray(flow.ux, dtype=np.float64)
    uy = np.ascontiguousarray(flow.uy, dtype=np.float64)
    if ux.ndim != 2 or uy.shape != ux.shape:
        raise ValueError("random flow grid ux/uy must be two-dimensional arrays with matching shape")
    grains = geometry.get("grains", [])
    if not grains:
        raise ValueError("random geometry must contain at least one grain")
    grain_x = np.ascontiguousarray([float(g["x"]) for g in grains], dtype=np.float64)
    grain_y = np.ascontiguousarray([float(g["y"]) for g in grains], dtype=np.float64)
    grain_r = np.ascontiguousarray([float(g["radius"]) for g in grains], dtype=np.float64)
    n_grains = int(grain_x.size)

    rng = np.random.default_rng(seed)
    if initial_y is not None:
        y0 = np.ascontiguousarray(initial_y, dtype=np.float64)
        n_particles = int(y0.size)
    else:
        y0 = np.ascontiguousarray(rng.uniform(0.0, flow.ly, size=n_particles), dtype=np.float64)
    if initial_x is not None:
        x0 = np.ascontiguousarray(initial_x, dtype=np.float64)
        if x0.size != n_particles:
            raise ValueError("initial_x must have the same length as initial_y")
    else:
        x0 = np.full(n_particles, 1.0e-6, dtype=np.float64)

    pore_counts = np.zeros((int(pore_nx), int(pore_ny)), dtype=np.float64)
    surface_counts = np.zeros((n_grains, int(surface_bins)), dtype=np.float64)
    attached_u8 = np.empty(n_particles, dtype=np.uint8)
    exited_u8 = np.empty(n_particles, dtype=np.uint8)
    censored_u8 = np.empty(n_particles, dtype=np.uint8)
    interceptions = np.empty(n_particles, dtype=np.int32)
    near_time = np.empty(n_particles, dtype=np.float64)
    h_min = np.empty(n_particles, dtype=np.float64)
    contact_events = np.empty(n_particles, dtype=np.int32)

    lib = load_kernel(force_rebuild=force_rebuild)
    c_params = _as_kernel_params(params, condition)
    status = lib.simulate_random_occupancy_kernel(
        ctypes.byref(c_params),
        ctypes.c_double(flow.lx),
        ctypes.c_double(flow.ly),
        ctypes.c_int(flow.nx),
        ctypes.c_int(flow.ny),
        _ptr(ux, ctypes.c_double),
        _ptr(uy, ctypes.c_double),
        ctypes.c_int(n_grains),
        _ptr(grain_x, ctypes.c_double),
        _ptr(grain_y, ctypes.c_double),
        _ptr(grain_r, ctypes.c_double),
        ctypes.c_int(_condition_code(condition)),
        ctypes.c_int(int(allow_attachment)),
        ctypes.c_int(n_particles),
        ctypes.c_uint64(seed),
        _ptr(x0, ctypes.c_double),
        _ptr(y0, ctypes.c_double),
        ctypes.c_int(int(pore_nx)),
        ctypes.c_int(int(pore_ny)),
        ctypes.c_int(int(surface_bins)),
        ctypes.c_int(max(int(sample_stride), 1)),
        _ptr(pore_counts, ctypes.c_double),
        _ptr(surface_counts, ctypes.c_double),
        _ptr(attached_u8, ctypes.c_uint8),
        _ptr(exited_u8, ctypes.c_uint8),
        _ptr(censored_u8, ctypes.c_uint8),
        _ptr(interceptions, ctypes.c_int32),
        _ptr(near_time, ctypes.c_double),
        _ptr(h_min, ctypes.c_double),
        _ptr(contact_events, ctypes.c_int32),
    )
    if status != 0:
        raise RuntimeError(f"compiled random occupancy kernel failed with status {status}")
    return {
        "pore_counts": pore_counts,
        "surface_counts": surface_counts,
        "attached": attached_u8.astype(bool),
        "exited": exited_u8.astype(bool),
        "censored": censored_u8.astype(bool),
        "interceptions": interceptions.astype(int),
        "near_time": near_time,
        "h_min": h_min,
        "contact_events": contact_events.astype(int),
        "x0": x0,
        "y0": y0,
    }
