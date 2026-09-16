"""Physical-unit periodic-cell flow, tracking, and transition-matrix tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PhysicalParams:
    """Physical parameters guided by the Volponi/Al-Zghoul studies."""

    cell_length: float = 4.0e-4
    grain_radius: float = 1.0e-4
    particle_radius: float = 0.55e-6
    mean_velocity: float = 4.0 / 86400.0
    temperature: float = 298.15
    viscosity: float = 1.0e-3
    relative_permittivity: float = 78.5
    ionic_strength_molar: float = 6.0e-3
    hamaker: float = 3.83e-21
    zeta_particle: float = -50.1e-3
    zeta_collector_favorable: float = 70.0e-3
    zeta_collector_unfavorable: float = -70.0e-3
    near_surface: float = 200.0e-9
    contact_gap: float = 1.0e-9
    min_gap: float = 1.0e-9
    dt: float = 2.0e-3
    max_time: float = 30.0
    dlvo_velocity_cap: float = 2.0e-4
    surface_sliding_gap: float = 5.0e-9
    wall_mobility_cutoff: float = 2.0e-6
    wall_mobility_normal_floor: float = 1.0e-4
    wall_mobility_parallel_floor: float = 5.0e-2
    wall_mobility_hydro_strength: float = 0.0
    wall_mobility_thermal_drift: bool = True
    diffusivity_multiplier: float = 1.0
    resolved_langevin_substeps: int = 25
    resolved_langevin_substep_cutoff: float = 75.0e-9
    resolved_langevin_normal_step: float = 3.0e-9
    resolved_langevin_min_dt: float = 1.0e-6
    resolved_langevin_max_substeps: int = 5000
    resolved_langevin_hydro_strength: float = 0.0

    @property
    def exclusion_radius(self) -> float:
        return self.grain_radius + self.particle_radius

    @property
    def inlet_y_min(self) -> float:
        return self.exclusion_radius + self.contact_gap

    @property
    def inlet_y_max(self) -> float:
        return self.cell_length - self.exclusion_radius - self.contact_gap

    @property
    def diffusivity(self) -> float:
        k_b = 1.380649e-23
        stokes_einstein = k_b * self.temperature / (6.0 * np.pi * self.viscosity * self.particle_radius)
        return self.diffusivity_multiplier * stokes_einstein

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
        return 1.0 / np.sqrt(kappa2)

    @property
    def reynolds_number(self) -> float:
        return self.mean_velocity * (2.0 * self.grain_radius) / 1.0e-6

    @property
    def particle_peclet(self) -> float:
        return self.mean_velocity * self.grain_radius / self.diffusivity


@dataclass
class FlowField:
    x: Array
    y: Array
    ux: Array
    uy: Array
    solid: Array
    params: PhysicalParams
    resolution: int
    iterations: int
    tau: float
    mean_velocity_before_scale: float
    scale_factor: float

    @property
    def dx(self) -> float:
        return self.params.cell_length / self.resolution


@dataclass
class TrajectoryLibrary:
    y_in: Array
    y_out: Array
    travel_time: Array
    dy: Array
    attached: Array
    exited: Array
    censored: Array
    interceptions: Array
    near_time: Array
    h_min: Array
    theta_entry: Array
    theta_exit: Array
    theta_final: Array
    collector_entry: Array
    collector_exit: Array
    collector_final: Array
    center_interceptions: Array
    corner_interceptions: Array
    contact_events: Array
    x_final: Array
    y_final: Array
    condition: str
    params: PhysicalParams
    resolution: int
    surface_mode: str = "unspecified"
    well_interceptions: Array | None = None
    well_time: Array | None = None
    theta_well_entry: Array | None = None
    theta_well_exit: Array | None = None
    theta_well_final: Array | None = None
    collector_well_entry: Array | None = None
    collector_well_exit: Array | None = None
    collector_well_final: Array | None = None
    center_well_interceptions: Array | None = None
    corner_well_interceptions: Array | None = None
    well_angular_travel: Array | None = None
    well_net_angular_travel: Array | None = None
    well_gap_lower: float = float("nan")
    well_gap_upper: float = float("nan")
    well_gap_minimum: float = float("nan")
    well_potential_minimum_kbt: float = float("nan")
    well_basin_lower: float = float("nan")
    well_basin_upper: float = float("nan")


def _nearest_grain_displacement(
    params: PhysicalParams,
    x: Array,
    y: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return displacement to the nearest grain in the periodic center/corner pattern.

    The unit cell contains one full collector at the cell center and four
    quarter collectors at the corners. In the periodically repeated geometry
    this is equivalent to two square lattices of collector centers: one at
    integer multiples of L and one shifted by (L/2, L/2).
    """

    length = params.cell_length
    corner_cx = length * np.round(x / length)
    corner_cy = length * np.round(y / length)
    corner_rx = x - corner_cx
    corner_ry = y - corner_cy
    corner_r2 = corner_rx * corner_rx + corner_ry * corner_ry

    center_cx = length * (np.round((x - 0.5 * length) / length) + 0.5)
    center_cy = length * (np.round((y - 0.5 * length) / length) + 0.5)
    center_rx = x - center_cx
    center_ry = y - center_cy
    center_r2 = center_rx * center_rx + center_ry * center_ry

    use_center = center_r2 <= corner_r2
    rx = np.where(use_center, center_rx, corner_rx)
    ry = np.where(use_center, center_ry, corner_ry)
    cx = np.where(use_center, center_cx, corner_cx)
    cy = np.where(use_center, center_cy, corner_cy)
    r = np.maximum(np.sqrt(np.minimum(center_r2, corner_r2)), 1.0e-30)
    return rx, ry, r, cx, cy


def _nearest_grain_state(
    params: PhysicalParams,
    x: Array,
    y: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return displacement and collector family for the nearest periodic grain.

    The collector id is 1 for the shifted center-grain lattice and 0 for the
    corner-grain lattice.
    """

    length = params.cell_length
    corner_cx = length * np.round(x / length)
    corner_cy = length * np.round(y / length)
    corner_rx = x - corner_cx
    corner_ry = y - corner_cy
    corner_r2 = corner_rx * corner_rx + corner_ry * corner_ry

    center_cx = length * (np.round((x - 0.5 * length) / length) + 0.5)
    center_cy = length * (np.round((y - 0.5 * length) / length) + 0.5)
    center_rx = x - center_cx
    center_ry = y - center_cy
    center_r2 = center_rx * center_rx + center_ry * center_ry

    use_center = center_r2 <= corner_r2
    rx = np.where(use_center, center_rx, corner_rx)
    ry = np.where(use_center, center_ry, corner_ry)
    cx = np.where(use_center, center_cx, corner_cx)
    cy = np.where(use_center, center_cy, corner_cy)
    r = np.maximum(np.sqrt(np.minimum(center_r2, corner_r2)), 1.0e-30)
    collector = np.where(use_center, 1, 0).astype(int)
    return rx, ry, r, cx, cy, collector


def _solid_mask(n: int, params: PhysicalParams) -> Array:
    coords = (np.arange(n) + 0.5) * params.cell_length / n
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    _, _, rr, _, _ = _nearest_grain_displacement(params, xx, yy)
    return rr <= params.grain_radius


def solve_lbm_flow(
    params: PhysicalParams,
    resolution: int = 160,
    iterations: int = 3500,
    tau: float = 0.8,
    force_lattice: float = 1.0e-6,
) -> FlowField:
    """Solve periodic, body-force-driven D2Q9 LBM flow around the collector pattern."""

    n = resolution
    solid = _solid_mask(n, params)
    fluid = ~solid
    c = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
        dtype=int,
    )
    w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    f = np.zeros((9, n, n), dtype=float)
    rho0 = np.ones((n, n), dtype=float)
    for i in range(9):
        f[i] = w[i] * rho0

    force = np.array([force_lattice, 0.0])
    cs2 = 1.0 / 3.0
    omega = 1.0 / tau

    for _ in range(iterations):
        rho = np.sum(f, axis=0)
        mom_x = np.tensordot(c[:, 0], f, axes=(0, 0)) + 0.5 * force[0]
        mom_y = np.tensordot(c[:, 1], f, axes=(0, 0)) + 0.5 * force[1]
        ux = np.where(fluid, mom_x / rho, 0.0)
        uy = np.where(fluid, mom_y / rho, 0.0)
        u2 = ux * ux + uy * uy

        feq = np.empty_like(f)
        forcing = np.empty_like(f)
        for i, (cx, cy) in enumerate(c):
            cu = cx * ux + cy * uy
            feq[i] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
            c_minus_u_dot_f = (cx - ux) * force[0] + (cy - uy) * force[1]
            cu_cf = cu * (cx * force[0] + cy * force[1])
            forcing[i] = w[i] * (1.0 - 0.5 * omega) * (
                c_minus_u_dot_f / cs2 + cu_cf / (cs2 * cs2)
            )
        f_post = f - omega * (f - feq) + forcing
        for i in range(9):
            f_post[i, solid] = f[opp[i], solid]

        streamed = np.empty_like(f)
        for i, (cx, cy) in enumerate(c):
            streamed[i] = np.roll(np.roll(f_post[i], cx, axis=0), cy, axis=1)
        f = streamed

    rho = np.sum(f, axis=0)
    mom_x = np.tensordot(c[:, 0], f, axes=(0, 0)) + 0.5 * force[0]
    mom_y = np.tensordot(c[:, 1], f, axes=(0, 0)) + 0.5 * force[1]
    ux_l = np.where(fluid, mom_x / rho, 0.0)
    uy_l = np.where(fluid, mom_y / rho, 0.0)
    ux_l[solid] = 0.0
    uy_l[solid] = 0.0
    mean_l = float(np.mean(ux_l[fluid]))
    scale = params.mean_velocity / mean_l
    coords = (np.arange(n) + 0.5) * params.cell_length / n
    return FlowField(
        x=coords,
        y=coords,
        ux=ux_l * scale,
        uy=uy_l * scale,
        solid=solid,
        params=params,
        resolution=resolution,
        iterations=iterations,
        tau=tau,
        mean_velocity_before_scale=mean_l,
        scale_factor=scale,
    )


def flow_quality_metrics(flow: FlowField) -> dict[str, float]:
    fluid = ~flow.solid
    dx = flow.dx
    div = (
        np.roll(flow.ux, -1, axis=0)
        - np.roll(flow.ux, 1, axis=0)
        + np.roll(flow.uy, -1, axis=1)
        - np.roll(flow.uy, 1, axis=1)
    ) / (2.0 * dx)
    return {
        "resolution": float(flow.resolution),
        "iterations": float(flow.iterations),
        "mean_ux": float(np.mean(flow.ux[fluid])),
        "max_speed": float(np.max(np.sqrt(flow.ux[fluid] ** 2 + flow.uy[fluid] ** 2))),
        "porosity": float(np.mean(fluid)),
        "rms_divergence": float(np.sqrt(np.mean(div[fluid] ** 2))),
        "mean_lattice_ux_before_scale": flow.mean_velocity_before_scale,
        "scale_factor": flow.scale_factor,
    }


def rescale_flow(
    flow: FlowField,
    mean_velocity: float,
    *,
    max_time: float | None = None,
    dt: float | None = None,
) -> FlowField:
    """Return a copy of a low-Re flow field scaled to a new mean velocity."""

    factor = mean_velocity / flow.params.mean_velocity
    params = replace(
        flow.params,
        mean_velocity=mean_velocity,
        max_time=flow.params.max_time if max_time is None else max_time,
        dt=flow.params.dt if dt is None else dt,
    )
    return FlowField(
        x=flow.x.copy(),
        y=flow.y.copy(),
        ux=flow.ux * factor,
        uy=flow.uy * factor,
        solid=flow.solid.copy(),
        params=params,
        resolution=flow.resolution,
        iterations=flow.iterations,
        tau=flow.tau,
        mean_velocity_before_scale=flow.mean_velocity_before_scale,
        scale_factor=flow.scale_factor * factor,
    )


def interpolate_velocity(flow: FlowField, x: Array, y: Array) -> tuple[Array, Array]:
    p = flow.params
    n = flow.resolution
    gx = (np.mod(x, p.cell_length) / p.cell_length) * n - 0.5
    gy = (np.mod(y, p.cell_length) / p.cell_length) * n - 0.5
    i0 = np.floor(gx).astype(int)
    j0 = np.floor(gy).astype(int)
    tx = gx - i0
    ty = gy - j0
    i0 %= n
    j0 %= n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n

    def interp(a: Array) -> Array:
        return (
            (1 - tx) * (1 - ty) * a[i0, j0]
            + tx * (1 - ty) * a[i1, j0]
            + (1 - tx) * ty * a[i0, j1]
            + tx * ty * a[i1, j1]
        )

    return interp(flow.ux), interp(flow.uy)


def _surface_state_with_collector(
    params: PhysicalParams,
    x: Array,
    y: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    rx, ry, r, _, _, collector = _nearest_grain_state(params, x, y)
    h = r - params.exclusion_radius
    theta = np.mod(np.arctan2(ry, rx), 2.0 * np.pi)
    return h, theta, rx / r, ry / r, r, collector


def _surface_state(params: PhysicalParams, x: Array, y: Array) -> tuple[Array, Array, Array, Array, Array]:
    h, theta, nx, ny, r, _ = _surface_state_with_collector(params, x, y)
    return h, theta, nx, ny, r


def _project_to_exclusion_surface(
    params: PhysicalParams,
    x: Array,
    y: Array,
    project: Array,
) -> tuple[Array, Array]:
    if not np.any(project):
        return x, y
    rx, ry, r, cx, cy = _nearest_grain_displacement(params, x, y)
    nx = rx / r
    ny = ry / r
    target = params.exclusion_radius + params.contact_gap
    x = x.copy()
    y = y.copy()
    x[project] = cx[project] + nx[project] * target
    y[project] = cy[project] + ny[project] * target
    return x, y


def _near_wall_mobility_factors(params: PhysicalParams, h: Array) -> tuple[Array, Array, Array]:
    """Return normal/parallel mobility factors and d(normal factor)/dh.

    The corrections are intentionally simple but physically oriented. The
    normal factor follows the small-gap lubrication asymptote M_perp/M0 ~ h/a
    at small gaps. The parallel factor combines a far-field wall correction
    with a logarithmic near-contact resistance. Both factors smoothly recover
    bulk Stokes-Einstein behavior outside ``wall_mobility_cutoff``.
    """

    h_eff = np.maximum(h, params.min_gap)
    active = h <= params.wall_mobility_cutoff
    a = params.particle_radius

    normal_raw = h_eff / (h_eff + a)
    normal = np.where(
        active,
        np.clip(normal_raw, params.wall_mobility_normal_floor, 1.0),
        1.0,
    )
    dnormal_dh_raw = a / (h_eff + a) ** 2
    dnormal_dh = np.where(
        active & (normal_raw > params.wall_mobility_normal_floor),
        dnormal_dh_raw,
        0.0,
    )

    alpha = np.clip(a / (a + h_eff), 0.0, 0.999999)
    parallel_far = (
        1.0
        - 9.0 / 16.0 * alpha
        + 1.0 / 8.0 * alpha**3
        - 45.0 / 256.0 * alpha**4
        - 1.0 / 16.0 * alpha**5
    )
    log_ratio = np.maximum(np.log(a / h_eff), 0.0)
    parallel_near = 1.0 / (1.0 + 0.9588 * log_ratio)
    parallel_raw = np.minimum(parallel_far, parallel_near)
    parallel = np.where(
        active,
        np.clip(parallel_raw, params.wall_mobility_parallel_floor, 1.0),
        1.0,
    )
    return normal, parallel, dnormal_dh


def dlvo_force_normal(params: PhysicalParams, h: Array, condition: str) -> Array:
    """Return the DLVO force component along the outward surface normal."""

    if condition in {"neutral", "no_dlvo", "advection_diffusion"}:
        return np.zeros_like(h, dtype=float)
    in_layer = h <= params.near_surface
    h_eff = np.maximum(h, params.min_gap)
    eps0 = 8.8541878128e-12
    eps = params.relative_permittivity * eps0
    kappa = 1.0 / params.debye_length
    zeta_c = params.zeta_collector_favorable if condition.startswith("favorable") else params.zeta_collector_unfavorable
    # Sphere-plane, constant-potential, low-potential scaling. The sign follows
    # zeta_p*zeta_c: same sign is repulsive, opposite sign attractive.
    f_edl = (
        2.0
        * np.pi
        * eps
        * params.particle_radius
        * kappa
        * params.zeta_particle
        * zeta_c
        * np.exp(-kappa * h_eff)
    )
    f_vdw = -params.hamaker * params.particle_radius / (6.0 * h_eff * h_eff)
    return np.where(in_layer, f_edl + f_vdw, 0.0)


def dlvo_potential_energy(params: PhysicalParams, h: Array, condition: str) -> Array:
    """Return the local sphere-plane DLVO potential energy."""

    if condition in {"neutral", "no_dlvo", "advection_diffusion"}:
        return np.zeros_like(h, dtype=float)
    h_eff = np.maximum(h, params.min_gap)
    eps0 = 8.8541878128e-12
    eps = params.relative_permittivity * eps0
    kappa = 1.0 / params.debye_length
    zeta_c = params.zeta_collector_favorable if condition.startswith("favorable") else params.zeta_collector_unfavorable
    c_edl = (
        2.0
        * np.pi
        * eps
        * params.particle_radius
        * kappa
        * params.zeta_particle
        * zeta_c
    )
    u_edl = c_edl / kappa * np.exp(-kappa * h_eff)
    u_vdw = -params.hamaker * params.particle_radius / (6.0 * h_eff)
    return u_edl + u_vdw


def secondary_minimum_well(
    params: PhysicalParams,
    condition: str,
    tolerance_kbt: float = 1.0,
    escape_potential_kbt: float = -1.0,
    h_min: float | None = None,
    h_max: float | None = None,
    n: int = 80_000,
) -> dict[str, float | bool]:
    """Return a potential-defined secondary-minimum well shell.

    The entry shell is the contiguous gap interval around the local unfavorable
    DLVO minimum where ``U(h) <= U_min + tolerance_kBT``. The broader basin
    shell is the contiguous interval around the same minimum where
    ``U(h) <= escape_potential_kbt``. A particle enters the bound state in the
    entry shell and remains bound until it leaves the basin.
    """

    def empty(minimum_gap: float = float("nan"), minimum_potential_kbt: float = float("nan")) -> dict[str, float | bool]:
        return {
            "has_well": False,
            "lower": float("nan"),
            "upper": float("nan"),
            "minimum_gap": minimum_gap,
            "minimum_potential_kbt": minimum_potential_kbt,
            "basin_lower": float("nan"),
            "basin_upper": float("nan"),
        }

    if condition in {"neutral", "no_dlvo", "advection_diffusion"}:
        return empty()
    if not condition.startswith("unfavorable"):
        return empty()
    lower_bound = params.min_gap if h_min is None else h_min
    upper_bound = params.near_surface if h_max is None else h_max
    if lower_bound <= 0.0 or upper_bound <= lower_bound:
        return empty()
    h = np.logspace(np.log10(lower_bound), np.log10(upper_bound), n)
    kbt = 1.380649e-23 * params.temperature
    potential_kbt = dlvo_potential_energy(params, h, condition) / kbt
    min_id = int(np.argmin(potential_kbt))
    u_min = float(potential_kbt[min_id])
    if not np.isfinite(u_min) or u_min >= 0.0:
        return empty(float(h[min_id]), u_min)
    threshold = u_min + tolerance_kbt
    in_well = potential_kbt <= threshold

    left = min_id
    while left > 0 and in_well[left - 1]:
        left -= 1
    right = min_id
    while right < n - 1 and in_well[right + 1]:
        right += 1
    in_basin = potential_kbt <= escape_potential_kbt
    basin_left = min_id
    while basin_left > 0 and in_basin[basin_left - 1]:
        basin_left -= 1
    basin_right = min_id
    while basin_right < n - 1 and in_basin[basin_right + 1]:
        basin_right += 1
    return {
        "has_well": bool(right >= left),
        "lower": float(h[left]),
        "upper": float(h[right]),
        "minimum_gap": float(h[min_id]),
        "minimum_potential_kbt": u_min,
        "basin_lower": float(h[basin_left]),
        "basin_upper": float(h[basin_right]),
    }


def dlvo_velocity(params: PhysicalParams, x: Array, y: Array, condition: str) -> tuple[Array, Array]:
    if condition in {"neutral", "no_dlvo", "advection_diffusion"}:
        return np.zeros_like(x, dtype=float), np.zeros_like(y, dtype=float)
    h, _, nx, ny, _ = _surface_state(params, x, y)
    force = dlvo_force_normal(params, h, condition)
    mobility = 1.0 / (6.0 * np.pi * params.viscosity * params.particle_radius)
    speed = np.clip(mobility * force, -params.dlvo_velocity_cap, params.dlvo_velocity_cap)
    return speed * nx, speed * ny


def simulate_cell_transitions(
    flow: FlowField,
    condition: str,
    n_particles: int = 5000,
    seed: int = 0,
    allow_attachment: bool | None = None,
    initial_x: Array | None = None,
    initial_y: Array | None = None,
    surface_mode: str = "project",
) -> TrajectoryLibrary:
    """Track particles from inlet to outlet, attachment, or censoring."""

    params = flow.params
    if surface_mode not in {"project", "sliding", "lubrication", "resolved_langevin"}:
        raise ValueError("surface_mode must be 'project', 'sliding', 'lubrication', or 'resolved_langevin'")
    rng = np.random.default_rng(seed)
    if initial_y is not None:
        y0 = np.asarray(initial_y, dtype=float)
        n_particles = int(y0.size)
    else:
        if params.inlet_y_min >= params.inlet_y_max:
            raise ValueError("collector geometry closes the inlet aperture")
        y0 = rng.uniform(params.inlet_y_min, params.inlet_y_max, size=n_particles)
    if initial_x is not None:
        x0 = np.asarray(initial_x, dtype=float)
        if x0.size != n_particles:
            raise ValueError("initial_x must have the same length as initial_y")
    else:
        x0 = np.zeros(n_particles)
    if allow_attachment is None:
        allow_attachment = condition == "favorable"
    x = x0.copy()
    y = y0.copy()
    t = np.zeros(n_particles)
    active = np.ones(n_particles, dtype=bool)
    attached = np.zeros(n_particles, dtype=bool)
    exited = np.zeros(n_particles, dtype=bool)
    censored = np.zeros(n_particles, dtype=bool)
    interceptions = np.zeros(n_particles, dtype=int)
    in_event = np.zeros(n_particles, dtype=bool)
    well = secondary_minimum_well(params, condition)
    has_well = bool(well["has_well"])
    well_lower = float(well["lower"])
    well_upper = float(well["upper"])
    well_basin_lower = float(well["basin_lower"])
    well_basin_upper = float(well["basin_upper"])
    well_interceptions = np.zeros(n_particles, dtype=int)
    in_well_event = np.zeros(n_particles, dtype=bool)
    well_angular_travel = np.zeros(n_particles)
    well_net_angular_travel = np.zeros(n_particles)
    near_time = np.zeros(n_particles)
    well_time = np.zeros(n_particles)
    h_min = np.full(n_particles, np.inf)
    theta_entry = np.full(n_particles, np.nan)
    theta_exit = np.full(n_particles, np.nan)
    theta_last_near = np.full(n_particles, np.nan)
    theta_well_entry = np.full(n_particles, np.nan)
    theta_well_exit = np.full(n_particles, np.nan)
    theta_well_last = np.full(n_particles, np.nan)
    collector_entry = np.full(n_particles, -1, dtype=int)
    collector_exit = np.full(n_particles, -1, dtype=int)
    collector_last_near = np.full(n_particles, -1, dtype=int)
    collector_well_entry = np.full(n_particles, -1, dtype=int)
    collector_well_exit = np.full(n_particles, -1, dtype=int)
    collector_well_last = np.full(n_particles, -1, dtype=int)
    center_interceptions = np.zeros(n_particles, dtype=int)
    corner_interceptions = np.zeros(n_particles, dtype=int)
    center_well_interceptions = np.zeros(n_particles, dtype=int)
    corner_well_interceptions = np.zeros(n_particles, dtype=int)
    contact_events = np.zeros(n_particles, dtype=int)
    y_out = np.full(n_particles, np.nan)
    travel_time = np.full(n_particles, np.nan)
    dy = np.full(n_particles, np.nan)

    def advance_once(idx: Array, step_dt: float) -> None:
        if idx.size == 0:
            return
        xa = x[idx]
        ya = y[idx]
        h0, _, nx0, ny0, _, _ = _surface_state_with_collector(params, xa, ya)
        ux, uy = interpolate_velocity(flow, xa, ya)
        if surface_mode == "resolved_langevin":
            force_normal = dlvo_force_normal(params, h0, condition)
            stokes_mobility = 1.0 / (6.0 * np.pi * params.viscosity * params.particle_radius)
            vx = stokes_mobility * force_normal * nx0
            vy = stokes_mobility * force_normal * ny0
        else:
            vx, vy = dlvo_velocity(params, xa, ya, condition)
        use_wall_langevin = surface_mode in {"lubrication", "resolved_langevin"}
        if use_wall_langevin:
            normal_mobility, parallel_mobility, dnormal_dh = _near_wall_mobility_factors(params, h0)
            tx0 = -ny0
            ty0 = nx0

            hydro_normal = ux * nx0 + uy * ny0
            hydro_parallel = ux * tx0 + uy * ty0
            hydro_strength = (
                params.resolved_langevin_hydro_strength
                if surface_mode == "resolved_langevin"
                else params.wall_mobility_hydro_strength
            )
            hydro_normal_scale = 1.0 - hydro_strength * (1.0 - normal_mobility)
            hydro_parallel_scale = 1.0 - hydro_strength * (1.0 - parallel_mobility)
            ux = hydro_normal * hydro_normal_scale * nx0 + hydro_parallel * hydro_parallel_scale * tx0
            uy = hydro_normal * hydro_normal_scale * ny0 + hydro_parallel * hydro_parallel_scale * ty0

            dlvo_normal = vx * nx0 + vy * ny0
            dlvo_parallel = vx * tx0 + vy * ty0
            vx = dlvo_normal * normal_mobility * nx0 + dlvo_parallel * parallel_mobility * tx0
            vy = dlvo_normal * normal_mobility * ny0 + dlvo_parallel * parallel_mobility * ty0
            if params.wall_mobility_thermal_drift:
                thermal_drift = params.diffusivity * dnormal_dh
                vx = vx + thermal_drift * nx0
                vy = vy + thermal_drift * ny0
        vtot_x = ux + vx
        vtot_y = uy + vy
        if surface_mode == "sliding":
            sliding = (
                (~allow_attachment)
                & condition.startswith("favorable")
                & (h0 <= params.contact_gap + params.surface_sliding_gap)
            )
            normal_velocity = vtot_x * nx0 + vtot_y * ny0
            inward = sliding & (normal_velocity < 0.0)
            vtot_x = vtot_x - np.where(inward, normal_velocity * nx0, 0.0)
            vtot_y = vtot_y - np.where(inward, normal_velocity * ny0, 0.0)
        noise = rng.normal(size=(idx.size, 2))
        if use_wall_langevin:
            brownian_normal = np.sqrt(2.0 * params.diffusivity * normal_mobility * step_dt) * noise[:, 0]
            brownian_parallel = np.sqrt(2.0 * params.diffusivity * parallel_mobility * step_dt) * noise[:, 1]
            tx0 = -ny0
            ty0 = nx0
            xa = (
                xa
                + vtot_x * step_dt
                + brownian_normal * nx0
                + brownian_parallel * tx0
            )
            ya = (
                ya
                + vtot_y * step_dt
                + brownian_normal * ny0
                + brownian_parallel * ty0
            )
        else:
            sqrt_2d_step = np.sqrt(2.0 * params.diffusivity * step_dt)
            xa = xa + vtot_x * step_dt + sqrt_2d_step * noise[:, 0]
            ya = ya + vtot_y * step_dt + sqrt_2d_step * noise[:, 1]

        h, theta, _, _, _, _ = _surface_state_with_collector(params, xa, ya)
        if surface_mode == "resolved_langevin" and condition.startswith("unfavorable"):
            old_u = dlvo_potential_energy(params, h0, condition)
            new_u = dlvo_potential_energy(params, h, condition)
            uphill_inward = (h < h0) & (new_u > old_u)
            if np.any(uphill_inward):
                kbt = 1.380649e-23 * params.temperature
                accept_probability = np.exp(-np.minimum((new_u - old_u) / kbt, 700.0))
                reject = uphill_inward & (rng.random(idx.size) > accept_probability)
                if np.any(reject):
                    xa[reject] = x[idx][reject]
                    ya[reject] = y[idx][reject]
                    h, theta, _, _, _, _ = _surface_state_with_collector(params, xa, ya)
        inside = h < params.contact_gap
        if np.any(inside):
            contact_events[idx[inside]] += 1
        if allow_attachment:
            hit = inside
            if np.any(hit):
                hit_idx = idx[hit]
                attached[hit_idx] = True
                active[hit_idx] = False
        else:
            hit = np.zeros(idx.size, dtype=bool)

        # Reflect/project non-attached particles that crossed an exclusion zone.
        project = inside & (~hit)
        xa, ya = _project_to_exclusion_surface(params, xa, ya, project)

        x[idx] = xa
        y[idx] = ya
        t[idx] += step_dt
        h, theta, _, _, _, collector = _surface_state_with_collector(params, x[idx], y[idx])
        h_min[idx] = np.minimum(h_min[idx], h)
        now_near = h <= params.near_surface
        entering = now_near & (~in_event[idx])
        leaving = (~now_near) & in_event[idx]
        if np.any(leaving):
            leave_idx = idx[leaving]
            theta_exit[leave_idx] = theta_last_near[leave_idx]
            collector_exit[leave_idx] = collector_last_near[leave_idx]
        if np.any(entering):
            enter_idx = idx[entering]
            interceptions[enter_idx] += 1
            theta_entry[enter_idx] = theta[entering]
            collector_entry[enter_idx] = collector[entering]
            center_entering = entering & (collector == 1)
            corner_entering = entering & (collector == 0)
            center_interceptions[idx[center_entering]] += 1
            corner_interceptions[idx[corner_entering]] += 1
        theta_last_near[idx[now_near]] = theta[now_near]
        collector_last_near[idx[now_near]] = collector[now_near]
        in_event[idx] = now_near
        near_time[idx[now_near]] += step_dt
        if has_well:
            in_entry_shell = (h >= well_lower) & (h <= well_upper)
            in_basin = (h >= well_basin_lower) & (h <= well_basin_upper)
            now_well = np.where(in_well_event[idx], in_basin, in_entry_shell)
            was_well = in_well_event[idx]
            well_entering = in_entry_shell & (~in_well_event[idx])
            well_leaving = (~in_basin) & in_well_event[idx]
            continuing_or_leaving = was_well & np.isfinite(theta_well_last[idx])
            if np.any(continuing_or_leaving):
                dtheta = angular_distance(theta[continuing_or_leaving], theta_well_last[idx[continuing_or_leaving]])
                travel_idx = idx[continuing_or_leaving]
                well_angular_travel[travel_idx] += np.abs(dtheta)
                well_net_angular_travel[travel_idx] += dtheta
            if np.any(well_leaving):
                leave_idx = idx[well_leaving]
                theta_well_exit[leave_idx] = theta[well_leaving]
                collector_well_exit[leave_idx] = collector[well_leaving]
            if np.any(well_entering):
                enter_idx = idx[well_entering]
                well_interceptions[enter_idx] += 1
                theta_well_entry[enter_idx] = theta[well_entering]
                collector_well_entry[enter_idx] = collector[well_entering]
                center_entering = well_entering & (collector == 1)
                corner_entering = well_entering & (collector == 0)
                center_well_interceptions[idx[center_entering]] += 1
                corner_well_interceptions[idx[corner_entering]] += 1
            theta_well_last[idx[now_well]] = theta[now_well]
            collector_well_last[idx[now_well]] = collector[now_well]
            in_well_event[idx] = now_well
            well_time[idx[now_well]] += step_dt

        still = active[idx]
        exit_now = (x[idx] >= params.cell_length) & still
        if np.any(exit_now):
            exit_idx = idx[exit_now]
            exited[exit_idx] = True
            active[exit_idx] = False
            y_out[exit_idx] = np.mod(y[exit_idx], params.cell_length)
            travel_time[exit_idx] = t[exit_idx]
            dy[exit_idx] = y[exit_idx] - y0[exit_idx]

    def resolved_near_wall_dt(idx: Array, remaining_dt: float) -> float:
        if idx.size == 0:
            return remaining_dt
        xa = x[idx]
        ya = y[idx]
        h0, _, nx0, ny0, _, _ = _surface_state_with_collector(params, xa, ya)
        ux, uy = interpolate_velocity(flow, xa, ya)
        normal_mobility, _, dnormal_dh = _near_wall_mobility_factors(params, h0)
        hydro_normal = ux * nx0 + uy * ny0
        force_normal = dlvo_force_normal(params, h0, condition)
        stokes_mobility = 1.0 / (6.0 * np.pi * params.viscosity * params.particle_radius)
        hydro_strength = params.resolved_langevin_hydro_strength
        hydro_normal_scale = 1.0 - hydro_strength * (1.0 - normal_mobility)
        normal_speed = hydro_normal * hydro_normal_scale + stokes_mobility * force_normal * normal_mobility
        if params.wall_mobility_thermal_drift:
            normal_speed = normal_speed + params.diffusivity * dnormal_dh

        target = max(params.resolved_langevin_normal_step, params.min_gap * 0.05)
        inward_speed = np.maximum(-normal_speed, 0.0)
        max_speed = float(np.max(inward_speed)) if inward_speed.size else 0.0
        dt_det = remaining_dt if max_speed <= 1.0e-30 else 0.25 * target / max_speed
        max_normal_mobility = float(np.max(normal_mobility)) if normal_mobility.size else 1.0
        dt_diff = remaining_dt
        if params.diffusivity > 0.0 and max_normal_mobility > 0.0:
            dt_diff = 0.25 * target * target / (2.0 * params.diffusivity * max_normal_mobility)
        dt = min(remaining_dt, params.dt / max(int(params.resolved_langevin_substeps), 1), dt_det, dt_diff)
        return max(float(dt), params.resolved_langevin_min_dt)

    max_steps = int(np.ceil(params.max_time / params.dt))
    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        if surface_mode == "resolved_langevin":
            h0, _, _, _, _, _ = _surface_state_with_collector(params, x[idx], y[idx])
            near_cohort = idx[h0 <= params.resolved_langevin_substep_cutoff]
            far_cohort = idx[h0 > params.resolved_langevin_substep_cutoff]
            advance_once(far_cohort, params.dt)
            remaining_dt = params.dt
            for _sub in range(max(int(params.resolved_langevin_max_substeps), 1)):
                sub_idx = near_cohort[active[near_cohort]]
                if sub_idx.size == 0:
                    break
                if remaining_dt <= 1.0e-15:
                    break
                sub_dt = min(resolved_near_wall_dt(sub_idx, remaining_dt), remaining_dt)
                advance_once(sub_idx, sub_dt)
                remaining_dt -= sub_dt
        else:
            advance_once(idx, params.dt)

    censored[active] = True
    _, theta_final, _, _, _, collector_final = _surface_state_with_collector(params, x, y)
    theta_well_final = theta_final.copy()
    collector_well_final = collector_final.copy()
    return TrajectoryLibrary(
        y_in=y0,
        y_out=y_out,
        travel_time=travel_time,
        dy=dy,
        attached=attached,
        exited=exited,
        censored=censored,
        interceptions=interceptions,
        near_time=near_time,
        h_min=h_min,
        theta_entry=theta_entry,
        theta_exit=theta_exit,
        theta_final=theta_final,
        collector_entry=collector_entry,
        collector_exit=collector_exit,
        collector_final=collector_final,
        center_interceptions=center_interceptions,
        corner_interceptions=corner_interceptions,
        contact_events=contact_events,
        x_final=x,
        y_final=y,
        condition=condition,
        params=params,
        resolution=flow.resolution,
        surface_mode=surface_mode,
        well_interceptions=well_interceptions,
        well_time=well_time,
        theta_well_entry=theta_well_entry,
        theta_well_exit=theta_well_exit,
        theta_well_final=theta_well_final,
        collector_well_entry=collector_well_entry,
        collector_well_exit=collector_well_exit,
        collector_well_final=collector_well_final,
        center_well_interceptions=center_well_interceptions,
        corner_well_interceptions=corner_well_interceptions,
        well_angular_travel=well_angular_travel,
        well_net_angular_travel=well_net_angular_travel,
        well_gap_lower=well_lower,
        well_gap_upper=well_upper,
        well_gap_minimum=float(well["minimum_gap"]),
        well_potential_minimum_kbt=float(well["minimum_potential_kbt"]),
        well_basin_lower=well_basin_lower,
        well_basin_upper=well_basin_upper,
    )


def transition_matrices(library: TrajectoryLibrary, n_bins: int = 32) -> dict[str, Array]:
    bins = np.linspace(0.0, library.params.cell_length, n_bins + 1)
    yin_bin = np.clip(np.digitize(library.y_in, bins) - 1, 0, n_bins - 1)
    counts = np.zeros((n_bins, n_bins), dtype=int)
    time_sums = np.zeros((n_bins, n_bins), dtype=float)
    dy_sums = np.zeros((n_bins, n_bins), dtype=float)
    interception_sums = np.zeros((n_bins, n_bins), dtype=float)
    near_time_sums = np.zeros((n_bins, n_bins), dtype=float)
    mobile = library.exited & (~library.attached)
    mobile_ids = np.flatnonzero(mobile)
    yout_bin = np.clip(np.digitize(library.y_out[mobile], bins) - 1, 0, n_bins - 1)
    for event_id, (i, j) in enumerate(zip(yin_bin[mobile], yout_bin)):
        counts[i, j] += 1
        particle_id = mobile_ids[event_id]
        time_sums[i, j] += library.travel_time[particle_id]
        dy_sums[i, j] += library.dy[particle_id]
        interception_sums[i, j] += library.interceptions[particle_id]
        near_time_sums[i, j] += library.near_time[particle_id]
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)
    mean_time = np.divide(time_sums, counts, out=np.full_like(time_sums, np.nan), where=counts > 0)
    mean_dy = np.divide(dy_sums, counts, out=np.full_like(dy_sums, np.nan), where=counts > 0)
    mean_interceptions = np.divide(
        interception_sums,
        counts,
        out=np.full_like(interception_sums, np.nan),
        where=counts > 0,
    )
    mean_near_time = np.divide(
        near_time_sums,
        counts,
        out=np.full_like(near_time_sums, np.nan),
        where=counts > 0,
    )

    total_by_in = np.bincount(yin_bin, minlength=n_bins)
    attached_by_in = np.bincount(yin_bin[library.attached], minlength=n_bins)
    exited_by_in = np.bincount(yin_bin[library.exited], minlength=n_bins)
    censored_by_in = np.bincount(yin_bin[library.censored], minlength=n_bins)
    attach_prob = np.divide(
        attached_by_in,
        total_by_in,
        out=np.zeros(n_bins, dtype=float),
        where=total_by_in > 0,
    )
    exit_prob = np.divide(
        exited_by_in,
        total_by_in,
        out=np.zeros(n_bins, dtype=float),
        where=total_by_in > 0,
    )
    return {
        "bins": bins,
        "counts": counts,
        "probabilities": probs,
        "mean_travel_time": mean_time,
        "mean_dy": mean_dy,
        "mean_interceptions": mean_interceptions,
        "mean_near_time": mean_near_time,
        "total_by_in": total_by_in,
        "attached_by_in": attached_by_in,
        "exited_by_in": exited_by_in,
        "censored_by_in": censored_by_in,
        "attach_prob_by_in": attach_prob,
        "exit_prob_by_in": exit_prob,
    }


def angular_distance(theta: Array, target: float = 0.0) -> Array:
    return np.arctan2(np.sin(theta - target), np.cos(theta - target))


def focusing_metrics(
    library: TrajectoryLibrary,
    centerline_band: float = 25.0e-6,
) -> dict[str, float | str]:
    """Summarize near-surface release and outlet-streamline focusing."""

    params = library.params
    mobile = library.exited & (~library.attached)
    intercepted = mobile & (library.interceptions > 0)
    intercepted_any = library.interceptions > 0
    intercepted_censored = library.censored & intercepted_any
    theta_ok = intercepted & np.isfinite(library.theta_exit)
    theta_censored = intercepted_censored & np.isfinite(library.theta_final)
    center_intercepted_any = library.center_interceptions > 0
    center_intercepted_mobile = mobile & center_intercepted_any
    center_intercepted_censored = library.censored & center_intercepted_any
    center_theta_ok = center_intercepted_mobile & (library.collector_exit == 1) & np.isfinite(library.theta_exit)
    center_theta_censored = (
        center_intercepted_censored
        & (library.collector_final == 1)
        & np.isfinite(library.theta_final)
    )

    def mean_or_nan(values: Array) -> float:
        return float(np.mean(values)) if values.size else float("nan")

    def median_or_nan(values: Array) -> float:
        return float(np.median(values)) if values.size else float("nan")

    def std_or_nan(values: Array) -> float:
        return float(np.std(values)) if values.size else float("nan")

    yout = library.y_out[intercepted]
    yin = library.y_in[intercepted]
    abs_center = np.abs(yout - 0.5 * params.cell_length)
    abs_shift = np.abs(yout - yin)
    theta_abs = np.abs(angular_distance(library.theta_exit[theta_ok], 0.0))
    theta_final_abs = np.abs(angular_distance(library.theta_final[theta_censored], 0.0))
    center_yout = library.y_out[center_intercepted_mobile]
    center_abs_center = np.abs(center_yout - 0.5 * params.cell_length)
    center_theta_abs = np.abs(angular_distance(library.theta_exit[center_theta_ok], 0.0))
    center_theta_final_abs = np.abs(angular_distance(library.theta_final[center_theta_censored], 0.0))
    well_interceptions = (
        library.well_interceptions
        if library.well_interceptions is not None
        else np.zeros(library.y_in.size, dtype=int)
    )
    well_time = (
        library.well_time
        if library.well_time is not None
        else np.zeros(library.y_in.size, dtype=float)
    )
    theta_well_exit = (
        library.theta_well_exit
        if library.theta_well_exit is not None
        else np.full(library.y_in.size, np.nan)
    )
    collector_well_exit = (
        library.collector_well_exit
        if library.collector_well_exit is not None
        else np.full(library.y_in.size, -1, dtype=int)
    )
    center_well_interceptions = (
        library.center_well_interceptions
        if library.center_well_interceptions is not None
        else np.zeros(library.y_in.size, dtype=int)
    )
    well_angular_travel = (
        library.well_angular_travel
        if library.well_angular_travel is not None
        else np.zeros(library.y_in.size, dtype=float)
    )
    well_net_angular_travel = (
        library.well_net_angular_travel
        if library.well_net_angular_travel is not None
        else np.zeros(library.y_in.size, dtype=float)
    )
    well_any = well_interceptions > 0
    well_mobile = mobile & well_any
    well_release = well_mobile & np.isfinite(theta_well_exit)
    center_well_any = center_well_interceptions > 0
    center_well_release = (
        mobile
        & center_well_any
        & (collector_well_exit == 1)
        & np.isfinite(theta_well_exit)
    )
    well_theta_abs = np.abs(angular_distance(theta_well_exit[well_release], 0.0))
    center_well_theta_abs = np.abs(angular_distance(theta_well_exit[center_well_release], 0.0))
    well_travel = well_angular_travel[well_release]
    center_well_travel = well_angular_travel[center_well_release]
    well_net_travel_abs = np.abs(well_net_angular_travel[well_release])
    center_well_net_travel_abs = np.abs(well_net_angular_travel[center_well_release])

    return {
        "condition": library.condition,
        "particles": int(library.y_in.size),
        "exited_fraction": float(np.mean(library.exited)),
        "attached_fraction": float(np.mean(library.attached)),
        "censored_fraction": float(np.mean(library.censored)),
        "intercepted_mobile_fraction": float(np.mean(intercepted)),
        "intercepted_mobile_count": int(np.sum(intercepted)),
        "intercepted_any_count": int(np.sum(intercepted_any)),
        "intercepted_censored_count": int(np.sum(intercepted_censored)),
        "intercepted_exit_fraction": float(np.sum(intercepted) / np.sum(intercepted_any)) if np.any(intercepted_any) else float("nan"),
        "center_intercepted_any_count": int(np.sum(center_intercepted_any)),
        "center_intercepted_mobile_count": int(np.sum(center_intercepted_mobile)),
        "center_intercepted_censored_count": int(np.sum(center_intercepted_censored)),
        "center_intercepted_exit_fraction": float(np.sum(center_intercepted_mobile) / np.sum(center_intercepted_any)) if np.any(center_intercepted_any) else float("nan"),
        "corner_intercepted_any_count": int(np.sum(library.corner_interceptions > 0)),
        "theta_exit_count": int(np.sum(theta_ok)),
        "mean_abs_theta_exit_downstream_rad": mean_or_nan(theta_abs),
        "median_abs_theta_exit_downstream_rad": median_or_nan(theta_abs),
        "fraction_theta_exit_within_15deg": float(np.mean(theta_abs <= np.deg2rad(15.0))) if theta_abs.size else float("nan"),
        "fraction_theta_exit_within_30deg": float(np.mean(theta_abs <= np.deg2rad(30.0))) if theta_abs.size else float("nan"),
        "theta_final_censored_count": int(np.sum(theta_censored)),
        "median_abs_theta_final_censored_downstream_rad": median_or_nan(theta_final_abs),
        "fraction_theta_final_censored_within_15deg": float(np.mean(theta_final_abs <= np.deg2rad(15.0))) if theta_final_abs.size else float("nan"),
        "fraction_theta_final_censored_within_30deg": float(np.mean(theta_final_abs <= np.deg2rad(30.0))) if theta_final_abs.size else float("nan"),
        "center_theta_exit_count": int(np.sum(center_theta_ok)),
        "center_median_abs_theta_exit_downstream_rad": median_or_nan(center_theta_abs),
        "center_fraction_theta_exit_within_15deg": float(np.mean(center_theta_abs <= np.deg2rad(15.0))) if center_theta_abs.size else float("nan"),
        "center_fraction_theta_exit_within_30deg": float(np.mean(center_theta_abs <= np.deg2rad(30.0))) if center_theta_abs.size else float("nan"),
        "center_theta_final_censored_count": int(np.sum(center_theta_censored)),
        "center_median_abs_theta_final_censored_downstream_rad": median_or_nan(center_theta_final_abs),
        "mean_abs_yout_center_m": mean_or_nan(abs_center),
        "median_abs_yout_center_m": median_or_nan(abs_center),
        "std_yout_intercepted_m": std_or_nan(yout),
        "fraction_yout_within_center_band": float(np.mean(abs_center <= centerline_band)) if yout.size else float("nan"),
        "center_median_abs_yout_center_m": median_or_nan(center_abs_center),
        "center_fraction_yout_within_center_band": float(np.mean(center_abs_center <= centerline_band)) if center_yout.size else float("nan"),
        "mean_abs_yout_yin_shift_m": mean_or_nan(abs_shift),
        "median_abs_yout_yin_shift_m": median_or_nan(abs_shift),
        "well_gap_lower_nm": float(library.well_gap_lower * 1.0e9),
        "well_gap_upper_nm": float(library.well_gap_upper * 1.0e9),
        "well_gap_minimum_nm": float(library.well_gap_minimum * 1.0e9),
        "well_potential_minimum_kbt": float(library.well_potential_minimum_kbt),
        "well_basin_lower_nm": float(library.well_basin_lower * 1.0e9),
        "well_basin_upper_nm": float(library.well_basin_upper * 1.0e9),
        "well_intercepted_any_count": int(np.sum(well_any)),
        "well_intercepted_mobile_count": int(np.sum(well_mobile)),
        "well_release_count": int(np.sum(well_release)),
        "well_median_abs_theta_exit_downstream_rad": median_or_nan(well_theta_abs),
        "well_fraction_theta_exit_within_30deg": float(np.mean(well_theta_abs <= np.deg2rad(30.0))) if well_theta_abs.size else float("nan"),
        "well_median_angular_travel_rad": median_or_nan(well_travel),
        "well_median_net_angular_travel_rad": median_or_nan(well_net_travel_abs),
        "center_well_intercepted_any_count": int(np.sum(center_well_any)),
        "center_well_release_count": int(np.sum(center_well_release)),
        "center_well_median_abs_theta_exit_downstream_rad": median_or_nan(center_well_theta_abs),
        "center_well_fraction_theta_exit_within_30deg": float(np.mean(center_well_theta_abs <= np.deg2rad(30.0))) if center_well_theta_abs.size else float("nan"),
        "center_well_median_angular_travel_rad": median_or_nan(center_well_travel),
        "center_well_median_net_angular_travel_rad": median_or_nan(center_well_net_travel_abs),
        "well_time_median_s": median_or_nan(well_time[well_any]),
        "center_well_time_median_s": median_or_nan(well_time[center_well_any]),
    }


def save_flow(path: str | Path, flow: FlowField) -> None:
    np.savez_compressed(
        path,
        x=flow.x,
        y=flow.y,
        ux=flow.ux,
        uy=flow.uy,
        solid=flow.solid,
        resolution=flow.resolution,
        iterations=flow.iterations,
        tau=flow.tau,
    )


def load_flow(path: str | Path, params: PhysicalParams | None = None) -> FlowField:
    data = np.load(path)
    if params is None:
        params = PhysicalParams()
    ux = data["ux"]
    uy = data["uy"]
    solid = data["solid"]
    fluid = ~solid
    return FlowField(
        x=data["x"],
        y=data["y"],
        ux=ux,
        uy=uy,
        solid=solid,
        params=params,
        resolution=int(data["resolution"]),
        iterations=int(data["iterations"]),
        tau=float(data["tau"]),
        mean_velocity_before_scale=float(np.mean(ux[fluid])),
        scale_factor=1.0,
    )


def load_library(path: str | Path, params: PhysicalParams | None = None) -> TrajectoryLibrary:
    data = np.load(path)
    if params is None:
        if "param_cell_length" in data.files:
            defaults = PhysicalParams()
            param_kwargs = {}
            for field in fields(PhysicalParams):
                key = f"param_{field.name}"
                if key not in data.files:
                    continue
                value = data[key].item()
                default_value = getattr(defaults, field.name)
                if isinstance(default_value, bool):
                    value = bool(value)
                elif isinstance(default_value, int):
                    value = int(value)
                else:
                    value = float(value)
                param_kwargs[field.name] = value
            params = replace(defaults, **param_kwargs)
        else:
            params = PhysicalParams()
    condition = data["condition"]
    condition_text = str(condition.item() if getattr(condition, "shape", ()) == () else condition)
    n = data["y_in"].size

    def array_or(name: str, default: Array) -> Array:
        return data[name] if name in data.files else default

    surface_mode = "unspecified"
    if "surface_mode" in data.files:
        surface_value = data["surface_mode"]
        surface_mode = str(surface_value.item() if getattr(surface_value, "shape", ()) == () else surface_value)

    return TrajectoryLibrary(
        y_in=data["y_in"],
        y_out=data["y_out"],
        travel_time=data["travel_time"],
        dy=data["dy"],
        attached=data["attached"],
        exited=data["exited"],
        censored=data["censored"],
        interceptions=data["interceptions"],
        near_time=data["near_time"],
        h_min=data["h_min"],
        theta_entry=data["theta_entry"],
        theta_exit=data["theta_exit"],
        theta_final=data["theta_final"],
        collector_entry=array_or("collector_entry", np.full(n, -1, dtype=int)),
        collector_exit=array_or("collector_exit", np.full(n, -1, dtype=int)),
        collector_final=array_or("collector_final", np.full(n, -1, dtype=int)),
        center_interceptions=array_or("center_interceptions", np.zeros(n, dtype=int)),
        corner_interceptions=array_or("corner_interceptions", np.zeros(n, dtype=int)),
        contact_events=array_or("contact_events", np.zeros(n, dtype=int)),
        x_final=data["x_final"],
        y_final=data["y_final"],
        condition=condition_text,
        params=params,
        resolution=int(data["resolution"]),
        surface_mode=surface_mode,
        well_interceptions=array_or("well_interceptions", np.zeros(n, dtype=int)),
        well_time=array_or("well_time", np.zeros(n, dtype=float)),
        theta_well_entry=array_or("theta_well_entry", np.full(n, np.nan)),
        theta_well_exit=array_or("theta_well_exit", np.full(n, np.nan)),
        theta_well_final=array_or("theta_well_final", np.full(n, np.nan)),
        collector_well_entry=array_or("collector_well_entry", np.full(n, -1, dtype=int)),
        collector_well_exit=array_or("collector_well_exit", np.full(n, -1, dtype=int)),
        collector_well_final=array_or("collector_well_final", np.full(n, -1, dtype=int)),
        center_well_interceptions=array_or("center_well_interceptions", np.zeros(n, dtype=int)),
        corner_well_interceptions=array_or("corner_well_interceptions", np.zeros(n, dtype=int)),
        well_angular_travel=array_or("well_angular_travel", np.zeros(n, dtype=float)),
        well_net_angular_travel=array_or("well_net_angular_travel", np.zeros(n, dtype=float)),
        well_gap_lower=float(data["well_gap_lower"]) if "well_gap_lower" in data.files else float("nan"),
        well_gap_upper=float(data["well_gap_upper"]) if "well_gap_upper" in data.files else float("nan"),
        well_gap_minimum=float(data["well_gap_minimum"]) if "well_gap_minimum" in data.files else float("nan"),
        well_potential_minimum_kbt=float(data["well_potential_minimum_kbt"]) if "well_potential_minimum_kbt" in data.files else float("nan"),
        well_basin_lower=float(data["well_basin_lower"]) if "well_basin_lower" in data.files else float("nan"),
        well_basin_upper=float(data["well_basin_upper"]) if "well_basin_upper" in data.files else float("nan"),
    )


def save_library(path: str | Path, library: TrajectoryLibrary) -> None:
    param_payload = {f"param_{key}": value for key, value in asdict(library.params).items()}
    np.savez_compressed(
        path,
        y_in=library.y_in,
        y_out=library.y_out,
        travel_time=library.travel_time,
        dy=library.dy,
        attached=library.attached,
        exited=library.exited,
        censored=library.censored,
        interceptions=library.interceptions,
        near_time=library.near_time,
        h_min=library.h_min,
        theta_entry=library.theta_entry,
        theta_exit=library.theta_exit,
        theta_final=library.theta_final,
        collector_entry=library.collector_entry,
        collector_exit=library.collector_exit,
        collector_final=library.collector_final,
        center_interceptions=library.center_interceptions,
        corner_interceptions=library.corner_interceptions,
        contact_events=library.contact_events,
        x_final=library.x_final,
        y_final=library.y_final,
        condition=library.condition,
        resolution=library.resolution,
        surface_mode=library.surface_mode,
        well_interceptions=library.well_interceptions if library.well_interceptions is not None else np.zeros_like(library.interceptions),
        well_time=library.well_time if library.well_time is not None else np.zeros_like(library.near_time),
        theta_well_entry=library.theta_well_entry if library.theta_well_entry is not None else np.full_like(library.theta_entry, np.nan),
        theta_well_exit=library.theta_well_exit if library.theta_well_exit is not None else np.full_like(library.theta_exit, np.nan),
        theta_well_final=library.theta_well_final if library.theta_well_final is not None else np.full_like(library.theta_final, np.nan),
        collector_well_entry=library.collector_well_entry if library.collector_well_entry is not None else np.full_like(library.collector_entry, -1),
        collector_well_exit=library.collector_well_exit if library.collector_well_exit is not None else np.full_like(library.collector_exit, -1),
        collector_well_final=library.collector_well_final if library.collector_well_final is not None else np.full_like(library.collector_final, -1),
        center_well_interceptions=library.center_well_interceptions if library.center_well_interceptions is not None else np.zeros_like(library.center_interceptions),
        corner_well_interceptions=library.corner_well_interceptions if library.corner_well_interceptions is not None else np.zeros_like(library.corner_interceptions),
        well_angular_travel=library.well_angular_travel if library.well_angular_travel is not None else np.zeros_like(library.near_time),
        well_net_angular_travel=library.well_net_angular_travel if library.well_net_angular_travel is not None else np.zeros_like(library.near_time),
        well_gap_lower=library.well_gap_lower,
        well_gap_upper=library.well_gap_upper,
        well_gap_minimum=library.well_gap_minimum,
        well_potential_minimum_kbt=library.well_potential_minimum_kbt,
        well_basin_lower=library.well_basin_lower,
        well_basin_upper=library.well_basin_upper,
        **param_payload,
    )
