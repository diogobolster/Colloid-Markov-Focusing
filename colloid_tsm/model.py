"""Core models for the single-grain colloid interception-history prototype.

The code in this module is intentionally compact and transparent. It is not a
replacement for high-resolution OpenFOAM/OnePiece simulations; it is a
proof-of-concept sandbox for testing how a DLVO-like near-surface layer and
interception history can be embedded in a trajectory-based Markov model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SingleGrainCell:
    """Dimensionless periodic cell containing one circular collector."""

    length: float = 1.0
    grain_radius: float = 0.22
    colloid_radius: float = 0.01
    mean_velocity: float = 1.0
    diffusion: float = 0.0025
    zoi: float = 0.055
    attach_gap: float = 0.004
    dt: float = 0.0015
    max_time: float = 45.0
    max_speed_factor: float = 5.0

    @property
    def center(self) -> tuple[float, float]:
        half = 0.5 * self.length
        return half, half

    @property
    def exclusion_radius(self) -> float:
        return self.grain_radius + self.colloid_radius

    @property
    def max_steps(self) -> int:
        return int(np.ceil(self.max_time / self.dt))


@dataclass(frozen=True)
class DLVOParams:
    """Dimensionless DLVO-like force parameters.

    ``barrier_height`` controls the repulsive EDL term on the homogeneous
    collector surface. The optional patch parameters are kept as hooks for a
    later heterodomain extension. These are not fitted material constants; they
    are knobs for the toy model and are deliberately labeled as dimensionless.
    """

    barrier_height: float = 0.35
    attraction_strength: float = 5.0e-4
    patch_barrier_height: float = 0.0
    patch_attraction_strength: float = 1.2e-3
    debye_length: float = 0.025
    min_gap: float = 0.002
    mobility_scale: float = 1.0
    max_dlvo_speed: float = 1.5


@dataclass(frozen=True)
class SurfaceChemistry:
    """Homogeneous surface chemistry with optional patch hooks for later work."""

    coverage: float = 0.08
    patch_count: int = 8
    patch_phase: float = 0.19
    base_attachment_rate: float = 45.0
    residual_attachment_rate: float = 0.0
    attachment_decay: float = 0.018

    def is_patch(self, theta: Array | float) -> Array:
        """Return True when a surface angle overlaps an optional attractive patch."""

        theta_arr = np.asarray(theta)
        if self.coverage >= 1.0:
            return np.ones(theta_arr.shape, dtype=bool)
        if self.coverage <= 0.0 or self.patch_count <= 0:
            return np.zeros(theta_arr.shape, dtype=bool)

        width = 2.0 * np.pi * self.coverage / self.patch_count
        centers = (
            self.patch_phase
            + 2.0 * np.pi * np.arange(self.patch_count) / self.patch_count
        )
        # Smallest wrapped angular distance to any patch center.
        diff = np.angle(np.exp(1j * (theta_arr[..., None] - centers)))
        return np.min(np.abs(diff), axis=-1) <= 0.5 * width

    def attachment_rate(self, h: Array, theta: Array, order: Array | int) -> Array:
        """Near-surface attachment hazard per unit dimensionless time."""

        patch = self.is_patch(theta)
        near_factor = np.exp(-np.maximum(h, 0.0) / self.attachment_decay)
        rates = np.where(patch, self.base_attachment_rate, self.residual_attachment_rate)
        return rates * near_factor


@dataclass
class TrainingData:
    """First-interception training library."""

    bulk_dx: Array
    bulk_dy: Array
    bulk_time: Array
    near_dx: Array
    near_dy: Array
    near_time: Array
    total_dx: Array
    total_dy: Array
    total_time: Array
    attached: Array
    h_min: Array
    theta_entry: Array
    completed: Array

    def complete(self) -> "TrainingData":
        mask = self.completed
        return TrainingData(
            bulk_dx=self.bulk_dx[mask],
            bulk_dy=self.bulk_dy[mask],
            bulk_time=self.bulk_time[mask],
            near_dx=self.near_dx[mask],
            near_dy=self.near_dy[mask],
            near_time=self.near_time[mask],
            total_dx=self.total_dx[mask],
            total_dy=self.total_dy[mask],
            total_time=self.total_time[mask],
            attached=self.attached[mask],
            h_min=self.h_min[mask],
            theta_entry=self.theta_entry[mask],
            completed=self.completed[mask],
        )

    @property
    def size(self) -> int:
        return int(self.total_dx.size)


@dataclass
class InterceptionRWResult:
    """Output from the interception-history random walk."""

    x: Array
    y: Array
    t: Array
    interception_order: Array
    attached: Array


def empirical_alpha(data: TrainingData) -> float:
    complete = data.complete()
    if complete.size == 0:
        return float("nan")
    return float(np.mean(complete.attached))


def _local_relative(cell: SingleGrainCell, x: Array, y: Array) -> tuple[Array, Array]:
    cx, cy = cell.center
    return np.mod(x, cell.length) - cx, np.mod(y, cell.length) - cy


def _flow_velocity(cell: SingleGrainCell, x: Array, y: Array) -> tuple[Array, Array]:
    """Potential-flow surrogate around a circular grain.

    It enforces no penetration at the collector surface but not no slip. That is
    sufficient for this first Markov-model prototype and should later be
    replaced by a CFD velocity field.
    """

    rx, ry = _local_relative(cell, x, y)
    a = cell.exclusion_radius
    r2 = np.maximum(rx * rx + ry * ry, (a + 1.0e-6) ** 2)
    r4 = r2 * r2
    u = cell.mean_velocity * (1.0 - a * a * (rx * rx - ry * ry) / r4)
    v = -2.0 * cell.mean_velocity * a * a * rx * ry / r4

    speed = np.sqrt(u * u + v * v)
    cap = cell.max_speed_factor * cell.mean_velocity
    scale = np.where(speed > cap, cap / np.maximum(speed, 1.0e-12), 1.0)
    return u * scale, v * scale


def _surface_state(
    cell: SingleGrainCell, x: Array, y: Array
) -> tuple[Array, Array, Array, Array, Array]:
    rx, ry = _local_relative(cell, x, y)
    r = np.maximum(np.sqrt(rx * rx + ry * ry), 1.0e-12)
    h = r - cell.exclusion_radius
    theta = np.mod(np.arctan2(ry, rx), 2.0 * np.pi)
    nx = rx / r
    ny = ry / r
    return h, theta, nx, ny, r


def _dlvo_velocity(
    cell: SingleGrainCell,
    dlvo: DLVOParams,
    chemistry: SurfaceChemistry,
    x: Array,
    y: Array,
) -> tuple[Array, Array]:
    h, theta, nx, ny, _ = _surface_state(cell, x, y)
    in_layer = h <= cell.zoi
    patch = chemistry.is_patch(theta)
    h_eff = np.maximum(h, dlvo.min_gap)

    barrier = np.where(patch, dlvo.patch_barrier_height, dlvo.barrier_height)
    attraction = np.where(
        patch, dlvo.patch_attraction_strength, dlvo.attraction_strength
    )
    dphi_dh = attraction / (h_eff * h_eff) - (
        barrier / dlvo.debye_length
    ) * np.exp(-h_eff / dlvo.debye_length)
    normal_speed = -cell.diffusion * dlvo.mobility_scale * dphi_dh
    normal_speed = np.clip(normal_speed, -dlvo.max_dlvo_speed, dlvo.max_dlvo_speed)
    normal_speed = np.where(in_layer, normal_speed, 0.0)
    return normal_speed * nx, normal_speed * ny


def _push_out_of_grain(
    cell: SingleGrainCell, x: Array, y: Array
) -> tuple[Array, Array]:
    h, _, nx, ny, _ = _surface_state(cell, x, y)
    inside = h < cell.attach_gap
    if not np.any(inside):
        return x, y

    local_x = np.mod(x, cell.length)
    local_y = np.mod(y, cell.length)
    ix = np.floor(x / cell.length)
    iy = np.floor(y / cell.length)
    cx, cy = cell.center
    target = cell.exclusion_radius + cell.attach_gap
    local_x = np.where(inside, cx + nx * target, local_x)
    local_y = np.where(inside, cy + ny * target, local_y)
    return ix * cell.length + local_x, iy * cell.length + local_y


def simulate_first_interceptions(
    n_particles: int,
    cell: SingleGrainCell,
    dlvo: DLVOParams,
    chemistry: SurfaceChemistry,
    seed: int = 13,
) -> TrainingData:
    """Simulate particles until their first interception ends or attaches."""

    rng = np.random.default_rng(seed)
    x0 = np.zeros(n_particles)
    y0 = rng.uniform(0.0, cell.length, size=n_particles)
    x = x0.copy()
    y = y0.copy()
    t = np.zeros(n_particles)

    in_event = np.zeros(n_particles, dtype=bool)
    done = np.zeros(n_particles, dtype=bool)
    attached = np.zeros(n_particles, dtype=bool)
    completed = np.zeros(n_particles, dtype=bool)

    event_start_x = np.full(n_particles, np.nan)
    event_start_y = np.full(n_particles, np.nan)
    event_start_t = np.full(n_particles, np.nan)
    theta_entry = np.full(n_particles, np.nan)
    near_time = np.zeros(n_particles)
    h_min = np.full(n_particles, np.inf)

    sqrt_2d_dt = np.sqrt(2.0 * cell.diffusion * cell.dt)

    for _ in range(cell.max_steps):
        active = ~done
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        xa = x[idx]
        ya = y[idx]

        ux, uy = _flow_velocity(cell, xa, ya)
        vx_dlvo, vy_dlvo = _dlvo_velocity(cell, dlvo, chemistry, xa, ya)
        noise = rng.normal(size=(idx.size, 2))
        xa = xa + (ux + vx_dlvo) * cell.dt + sqrt_2d_dt * noise[:, 0]
        ya = ya + (uy + vy_dlvo) * cell.dt + sqrt_2d_dt * noise[:, 1]
        xa, ya = _push_out_of_grain(cell, xa, ya)

        x[idx] = xa
        y[idx] = ya
        t[idx] += cell.dt

        h, theta, _, _, _ = _surface_state(cell, xa, ya)
        h_min[idx] = np.minimum(h_min[idx], h)
        in_zoi = h <= cell.zoi

        entering = active.copy()
        entering[idx] = in_zoi & (~in_event[idx])
        if np.any(entering):
            entering_idx = np.flatnonzero(entering)
            local_entering = np.isin(idx, entering_idx)
            event_start_x[entering_idx] = x[entering_idx]
            event_start_y[entering_idx] = y[entering_idx]
            event_start_t[entering_idx] = t[entering_idx]
            theta_entry[entering_idx] = theta[local_entering]
            in_event[entering_idx] = True

        event_idx = np.flatnonzero(active & in_event)
        if event_idx.size:
            near_time[event_idx] += cell.dt
            h_e, theta_e, _, _, _ = _surface_state(cell, x[event_idx], y[event_idx])
            order = np.ones(event_idx.size)
            rates = chemistry.attachment_rate(h_e, theta_e, order)
            p_attach = 1.0 - np.exp(-rates * cell.dt)
            hit = rng.uniform(size=event_idx.size) < p_attach

            # A tiny gap on an attractive patch is treated as primary-minimum contact.
            patch_contact = (h_e <= 1.2 * cell.attach_gap) & chemistry.is_patch(theta_e)
            hit = hit | patch_contact

            escape = (h_e > cell.zoi) & (near_time[event_idx] > cell.dt)
            event_done = hit | escape
            if np.any(event_done):
                done_idx = event_idx[event_done]
                attached[done_idx] = hit[event_done]
                completed[done_idx] = True
                done[done_idx] = True

    total_dx = x - x0
    total_dy = y - y0
    bulk_dx = event_start_x - x0
    bulk_dy = event_start_y - y0
    bulk_time = event_start_t
    near_dx = x - event_start_x
    near_dy = y - event_start_y
    total_time = t
    bulk_dx = np.where(completed, bulk_dx, np.nan)
    bulk_dy = np.where(completed, bulk_dy, np.nan)
    bulk_time = np.where(completed, bulk_time, np.nan)
    near_dx = np.where(completed, near_dx, np.nan)
    near_dy = np.where(completed, near_dy, np.nan)

    return TrainingData(
        bulk_dx=bulk_dx,
        bulk_dy=bulk_dy,
        bulk_time=bulk_time,
        near_dx=near_dx,
        near_dy=near_dy,
        near_time=near_time,
        total_dx=total_dx,
        total_dy=total_dy,
        total_time=total_time,
        attached=attached,
        h_min=h_min,
        theta_entry=theta_entry,
        completed=completed,
    )


def run_interception_rw(
    training: TrainingData,
    n_particles: int = 20_000,
    max_interceptions: int = 200,
    seed: int = 91,
    alpha_schedule: Callable[[Array], Array] | None = None,
) -> InterceptionRWResult:
    """Sample repeated first-interception events until attachment."""

    data = training.complete()
    if data.size == 0:
        raise ValueError("TrainingData contains no completed first interceptions.")

    rng = np.random.default_rng(seed)
    x = np.zeros(n_particles)
    y = np.zeros(n_particles)
    t = np.zeros(n_particles)
    order = np.zeros(n_particles, dtype=int)
    attached = np.zeros(n_particles, dtype=bool)

    for _ in range(max_interceptions):
        active = ~attached
        if not np.any(active):
            break
        active_idx = np.flatnonzero(active)
        draws = rng.integers(0, data.size, size=active_idx.size)
        x[active_idx] += data.total_dx[draws]
        y[active_idx] += data.total_dy[draws]
        t[active_idx] += data.total_time[draws]
        order[active_idx] += 1

        if alpha_schedule is None:
            event_attached = data.attached[draws]
        else:
            p = np.clip(alpha_schedule(order[active_idx]), 0.0, 1.0)
            event_attached = rng.uniform(size=active_idx.size) < p
        attached[active_idx[event_attached]] = True

    return InterceptionRWResult(
        x=x,
        y=y,
        t=t,
        interception_order=order,
        attached=attached,
    )


def run_fixed_interception_sequence(
    training: TrainingData,
    n_particles: int = 20_000,
    n_interceptions: int = 20,
    seed: int = 93,
) -> InterceptionRWResult:
    """Sample a fixed number of non-attaching interception/escape events.

    This is useful for the homogeneous unfavorable starter problem, where a
    repulsive boundary produces interception and reflection but no attachment.
    """

    data = training.complete()
    if data.size == 0:
        raise ValueError("TrainingData contains no completed first interceptions.")

    rng = np.random.default_rng(seed)
    x = np.zeros(n_particles)
    y = np.zeros(n_particles)
    t = np.zeros(n_particles)

    for _ in range(n_interceptions):
        draws = rng.integers(0, data.size, size=n_particles)
        x += data.total_dx[draws]
        y += data.total_dy[draws]
        t += data.total_time[draws]

    return InterceptionRWResult(
        x=x,
        y=y,
        t=t,
        interception_order=np.full(n_particles, n_interceptions, dtype=int),
        attached=np.zeros(n_particles, dtype=bool),
    )


def sample_trajectories(
    n_paths: int,
    cell: SingleGrainCell,
    dlvo: DLVOParams,
    chemistry: SurfaceChemistry,
    seed: int = 23,
    stride: int = 8,
) -> list[dict[str, Array | bool]]:
    """Record a handful of explicit paths for visualization."""

    rng = np.random.default_rng(seed)
    paths: list[dict[str, Array | bool]] = []
    sqrt_2d_dt = np.sqrt(2.0 * cell.diffusion * cell.dt)
    for _ in range(n_paths):
        x = 0.0
        y = float(rng.uniform(0.0, cell.length))
        xs = [x]
        ys = [y]
        in_event = False
        near_time = 0.0
        attached = False
        completed = False
        for step in range(cell.max_steps):
            ux, uy = _flow_velocity(cell, np.array([x]), np.array([y]))
            vx, vy = _dlvo_velocity(cell, dlvo, chemistry, np.array([x]), np.array([y]))
            noise = rng.normal(size=2)
            x = x + float(ux[0] + vx[0]) * cell.dt + sqrt_2d_dt * noise[0]
            y = y + float(uy[0] + vy[0]) * cell.dt + sqrt_2d_dt * noise[1]
            x_arr, y_arr = _push_out_of_grain(cell, np.array([x]), np.array([y]))
            x = float(x_arr[0])
            y = float(y_arr[0])
            h, theta, _, _, _ = _surface_state(cell, np.array([x]), np.array([y]))
            h0 = float(h[0])
            theta0 = float(theta[0])
            if h0 <= cell.zoi and not in_event:
                in_event = True
            if in_event:
                near_time += cell.dt
                rate = float(chemistry.attachment_rate(np.array([h0]), np.array([theta0]), 1)[0])
                p_attach = 1.0 - np.exp(-rate * cell.dt)
                if rng.uniform() < p_attach or (
                    h0 <= 1.2 * cell.attach_gap and bool(chemistry.is_patch(theta0))
                ):
                    attached = True
                    completed = True
                    if step % stride != 0:
                        xs.append(x)
                        ys.append(y)
                    break
                if h0 > cell.zoi and near_time > cell.dt:
                    completed = True
                    if step % stride != 0:
                        xs.append(x)
                        ys.append(y)
                    break
            if step % stride == 0:
                xs.append(x)
                ys.append(y)
        paths.append(
            {
                "x": np.asarray(xs),
                "y": np.asarray(ys),
                "attached": attached,
                "completed": completed,
            }
        )
    return paths
