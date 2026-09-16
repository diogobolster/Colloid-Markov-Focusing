#!/usr/bin/env python3
"""Generate a periodic random disk geometry for the next colloid-focusing stage.

The output is intentionally geometry-first.  It records enough metadata to
build an OpenFOAM case later and to define grain-resolved stagnation points once
the body-fitted flow field has been solved.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


OUT = ROOT / "outputs" / "random_porous_geometry"


@dataclass(frozen=True)
class GeometryConfig:
    name: str = "random_periodic_2d_v1"
    seed: int = 20260505
    length_x: float = 1.20e-3
    length_y: float = 8.00e-4
    n_grains: int = 34
    target_porosity: float = 0.45
    radius_cv: float = 0.16
    minimum_surface_gap: float = 8.0e-6
    particle_radius_reference: float = 0.55e-6
    near_surface_reference: float = 200.0e-9
    raster_nx: int = 900
    raster_ny: int = 600
    pinned_centers_fractional: tuple[tuple[float, float], ...] = (
        (0.150, 0.875),
        (0.917, 0.575),
        (0.667, 0.119),
        (0.760, 0.518),
    )
    pinned_radius_raw: tuple[float, ...] = (0.86, 0.78, 0.78, 0.72)

    @property
    def equivalent_grain_radius(self) -> float:
        solid_area = (1.0 - self.target_porosity) * self.length_x * self.length_y
        return math.sqrt(solid_area / (self.n_grains * math.pi))


def make_radii(cfg: GeometryConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 909)
    sigma = cfg.radius_cv
    raw = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=cfg.n_grains)
    raw = np.clip(raw, 0.72, 1.32)
    n_pinned = len(cfg.pinned_centers_fractional)
    if n_pinned:
        raw[-n_pinned:] = np.array(cfg.pinned_radius_raw, dtype=float)
    solid_area = (1.0 - cfg.target_porosity) * cfg.length_x * cfg.length_y
    scale = math.sqrt(solid_area / (math.pi * float(np.sum(raw * raw))))
    return scale * raw


def pinned_mask(cfg: GeometryConfig) -> np.ndarray:
    mask = np.zeros(cfg.n_grains, dtype=bool)
    n_pinned = len(cfg.pinned_centers_fractional)
    if n_pinned:
        mask[-n_pinned:] = True
    return mask


def periodic_delta(delta: np.ndarray | float, length: float) -> np.ndarray | float:
    return delta - length * np.round(delta / length)


def pair_delta(a: np.ndarray, b: np.ndarray, lx: float, ly: float) -> np.ndarray:
    return np.array(
        [
            periodic_delta(a[0] - b[0], lx),
            periodic_delta(a[1] - b[1], ly),
        ],
        dtype=float,
    )


def initial_centers(cfg: GeometryConfig, rng: np.random.Generator) -> np.ndarray:
    n_pinned = len(cfg.pinned_centers_fractional)
    n_free = cfg.n_grains - n_pinned
    cols = int(round(math.sqrt(n_free * cfg.length_x / cfg.length_y)))
    cols = max(cols, 1)
    rows = int(math.ceil(n_free / cols))
    while cols * rows < n_free:
        cols += 1
        rows = int(math.ceil(n_free / cols))

    slots = []
    for ix in range(cols):
        for iy in range(rows):
            slots.append(((ix + 0.5) * cfg.length_x / cols, (iy + 0.5) * cfg.length_y / rows))
    chosen = rng.choice(len(slots), size=n_free, replace=False)
    centers = np.array([slots[i] for i in chosen], dtype=float)

    jitter = 0.22 * min(cfg.length_x / cols, cfg.length_y / rows)
    centers += rng.normal(0.0, jitter, size=centers.shape)
    centers[:, 0] %= cfg.length_x
    centers[:, 1] %= cfg.length_y
    if n_pinned:
        pinned = np.array(
            [(fx * cfg.length_x, fy * cfg.length_y) for fx, fy in cfg.pinned_centers_fractional],
            dtype=float,
        )
        centers = np.vstack([centers, pinned])
    return centers


def relax_periodic_pack(
    centers: np.ndarray,
    radii: np.ndarray,
    cfg: GeometryConfig,
    *,
    max_sweeps: int = 20000,
    tolerance: float = 5.0e-10,
) -> tuple[np.ndarray, int, float]:
    centers = centers.copy()
    lx = cfg.length_x
    ly = cfg.length_y
    rng = np.random.default_rng(cfg.seed + 17)
    fixed = pinned_mask(cfg)

    for sweep in range(max_sweeps):
        max_overlap = 0.0
        order = rng.permutation(cfg.n_grains)
        for aa in range(cfg.n_grains - 1):
            i = int(order[aa])
            for bb in range(aa + 1, cfg.n_grains):
                j = int(order[bb])
                delta = pair_delta(centers[i], centers[j], lx, ly)
                dist = float(np.hypot(delta[0], delta[1]))
                if dist < 1.0e-18:
                    angle = rng.uniform(0.0, 2.0 * math.pi)
                    target = radii[i] + radii[j] + cfg.minimum_surface_gap
                    delta = np.array([math.cos(angle), math.sin(angle)]) * target
                    dist = target
                target = radii[i] + radii[j] + cfg.minimum_surface_gap
                overlap = target - dist
                if overlap <= 0.0:
                    continue
                max_overlap = max(max_overlap, overlap)
                direction = delta / dist
                if fixed[i] and fixed[j]:
                    continue
                if fixed[i]:
                    centers[j] -= 1.01 * overlap * direction
                elif fixed[j]:
                    centers[i] += 1.01 * overlap * direction
                else:
                    correction = 0.505 * overlap * direction
                    centers[i] += correction
                    centers[j] -= correction
                centers[:, 0] %= lx
                centers[:, 1] %= ly

        if max_overlap < tolerance:
            return centers, sweep + 1, max_overlap

    return centers, max_sweeps, max_overlap


def make_pack(cfg: GeometryConfig) -> tuple[np.ndarray, np.ndarray, int, float]:
    radii = make_radii(cfg)
    best = None
    for attempt in range(40):
        rng = np.random.default_rng(cfg.seed + attempt * 101)
        centers = initial_centers(cfg, rng)
        relaxed, sweeps, overlap = relax_periodic_pack(centers, radii, cfg)
        if best is None or overlap < best[2]:
            best = (relaxed, sweeps, overlap)
        if overlap < 5.0e-10:
            return relaxed, radii, sweeps, overlap
    assert best is not None
    if best[2] > 2.0e-7:
        raise RuntimeError(
            f"Could not create a non-overlapping periodic packing; final overlap {best[2]:.3e} m"
        )
    return best[0], radii, best[1], best[2]


def actual_porosity(radii: np.ndarray, cfg: GeometryConfig) -> float:
    solid_area = math.pi * float(np.sum(radii * radii))
    return 1.0 - solid_area / (cfg.length_x * cfg.length_y)


def surface_gaps(centers: np.ndarray, radii: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    gaps = []
    for i in range(cfg.n_grains - 1):
        for j in range(i + 1, cfg.n_grains):
            delta = pair_delta(centers[i], centers[j], cfg.length_x, cfg.length_y)
            gaps.append(float(np.hypot(delta[0], delta[1]) - radii[i] - radii[j]))
    return np.array(gaps, dtype=float)


def surface_distance_to_grains(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    cfg: GeometryConfig,
) -> np.ndarray:
    d_min = np.full_like(x, np.inf, dtype=float)
    for (cx, cy), radius in zip(centers, radii):
        dx = periodic_delta(x - cx, cfg.length_x)
        dy = periodic_delta(y - cy, cfg.length_y)
        d_min = np.minimum(d_min, np.sqrt(dx * dx + dy * dy) - radius)
    return d_min


def solid_mask(centers: np.ndarray, radii: np.ndarray, cfg: GeometryConfig, clearance: float = 0.0) -> np.ndarray:
    x = (np.arange(cfg.raster_nx) + 0.5) * cfg.length_x / cfg.raster_nx
    y = (np.arange(cfg.raster_ny) + 0.5) * cfg.length_y / cfg.raster_ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return surface_distance_to_grains(xx, yy, centers, radii, cfg) <= clearance


def left_right_connected(fluid: np.ndarray) -> tuple[bool, float]:
    nx, ny = fluid.shape
    visited = np.zeros_like(fluid, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    left_fluid = np.where(fluid[0, :])[0]
    for iy in left_fluid:
        visited[0, iy] = True
        q.append((0, int(iy)))

    reached_right = 0
    while q:
        ix, iy = q.popleft()
        if ix == nx - 1:
            reached_right += 1
        for jx, jy in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
            if jx < 0 or jx >= nx:
                continue
            jy %= ny
            if not fluid[jx, jy] or visited[jx, jy]:
                continue
            visited[jx, jy] = True
            q.append((jx, jy))

    right_fluid_count = max(int(np.sum(fluid[-1, :])), 1)
    return reached_right > 0, reached_right / right_fluid_count


def open_intervals_at_x(
    centers: np.ndarray,
    radii: np.ndarray,
    cfg: GeometryConfig,
    clearance: float,
) -> list[tuple[float, float]]:
    n = 4000
    y = (np.arange(n) + 0.5) * cfg.length_y / n
    x = np.zeros_like(y)
    open_mask = surface_distance_to_grains(x, y, centers, radii, cfg) > clearance

    if np.all(open_mask):
        return [(0.0, cfg.length_y)]
    if not np.any(open_mask):
        return []

    dy = cfg.length_y / n
    starts_list: list[int] = []
    ends_list: list[int] = []
    if open_mask[0]:
        starts_list.append(0)
    starts_list.extend((np.where((~open_mask[:-1]) & open_mask[1:])[0] + 1).astype(int).tolist())
    ends_list.extend((np.where(open_mask[:-1] & (~open_mask[1:]))[0] + 1).astype(int).tolist())
    if open_mask[-1]:
        ends_list.append(n)
    intervals: list[tuple[float, float]] = []
    for start, end in zip(starts_list, ends_list):
        s = start * dy
        e = end * dy
        intervals.append((s, e))

    intervals.sort()
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1] + cfg.length_y / n:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def periodic_images(
    centers: np.ndarray,
    radii: np.ndarray,
    cfg: GeometryConfig,
) -> list[dict[str, float | int | str]]:
    images = []
    for gid, ((cx, cy), r) in enumerate(zip(centers, radii)):
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                x = cx + ix * cfg.length_x
                y = cy + iy * cfg.length_y
                intersects = (-r <= x <= cfg.length_x + r) and (-r <= y <= cfg.length_y + r)
                if not intersects:
                    continue
                images.append(
                    {
                        "name": f"grain_{gid:03d}_img_{ix:+d}_{iy:+d}".replace("+", "p").replace("-", "m"),
                        "grain_id": gid,
                        "image_i": ix,
                        "image_j": iy,
                        "x": float(x),
                        "y": float(y),
                        "radius": float(r),
                    }
                )
    return images


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, centers: np.ndarray, radii: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["grain_id", "x_m", "y_m", "radius_m"])
        writer.writeheader()
        for gid, ((x, y), radius) in enumerate(zip(centers, radii)):
            writer.writerow(
                {
                    "grain_id": gid,
                    "x_m": f"{x:.12e}",
                    "y_m": f"{y:.12e}",
                    "radius_m": f"{radius:.12e}",
                }
            )


def write_openfoam_includes(path: Path, images: list[dict[str, float | int | str]], cfg: GeometryConfig) -> None:
    z0 = -5.0e-6
    z1 = 5.0e-6
    geometry_lines = []
    surface_lines = []
    region_lines = []
    for image in images:
        name = str(image["name"])
        x = float(image["x"])
        y = float(image["y"])
        r = float(image["radius"])
        geometry_lines.append(
            f"""
    {name}
    {{
        type searchableCylinder;
        point1 ({x:.12g} {y:.12g} {z0:.12g});
        point2 ({x:.12g} {y:.12g} {z1:.12g});
        radius {r:.12g};
    }}
"""
        )
        surface_lines.append(
            f"""
        {name}
        {{
            level (2 2);
            patchInfo {{ type wall; }}
        }}
"""
        )
        region_lines.append(
            f"""
        {name}
        {{
            mode distance;
            levels (({0.15 * r:.12g} 2));
        }}
"""
        )

    path.write_text(
        "/* snappyHexMesh geometry entries */\n"
        + "".join(geometry_lines)
        + "\n/* refinementSurfaces entries */\n"
        + "".join(surface_lines)
        + "\n/* refinementRegions entries */\n"
        + "".join(region_lines),
        encoding="utf-8",
    )


def write_report(
    path: Path,
    cfg: GeometryConfig,
    centers: np.ndarray,
    radii: np.ndarray,
    gaps: np.ndarray,
    connected: bool,
    connected_fraction: float,
    intervals: list[tuple[float, float]],
    images: list[dict[str, float | int | str]],
    sweeps: int,
) -> None:
    throat_quantiles = np.quantile(gaps, [0.0, 0.05, 0.25, 0.50, 0.75])
    porosity = actual_porosity(radii, cfg)
    fixed = pinned_mask(cfg)
    pinned_rows = []
    for gid in np.where(fixed)[0]:
        pinned_rows.append(
            f"- Grain {gid}: x={centers[gid, 0] * 1e6:.1f} um, "
            f"y={centers[gid, 1] * 1e6:.1f} um, R={radii[gid] * 1e6:.1f} um"
        )
    pinned_text = "\n".join(pinned_rows) if pinned_rows else "- none"
    interval_text = "\n".join(
        f"- {i + 1}: {s * 1e6:.2f} to {e * 1e6:.2f} um (width {(e - s) * 1e6:.2f} um)"
        for i, (s, e) in enumerate(intervals)
    )
    if not interval_text:
        interval_text = "- none"

    path.write_text(
        f"""# Random periodic porous geometry

This is the first geometry artifact for the larger-domain DLVO focusing problem.
It is a two-dimensional periodic disk packing intended for a body-fitted
OpenFOAM calculation from the start.

## Geometry

- Name: `{cfg.name}`
- Seed: `{cfg.seed}`
- Domain: {cfg.length_x * 1e3:.3f} mm x {cfg.length_y * 1e3:.3f} mm
- Periodicity: x and y
- Grains: {cfg.n_grains} mildly polydisperse circular collectors
- Equivalent monodisperse radius: {cfg.equivalent_grain_radius * 1e6:.2f} um
- Radius range: {float(np.min(radii)) * 1e6:.2f} to {float(np.max(radii)) * 1e6:.2f} um
- Mean radius: {float(np.mean(radii)) * 1e6:.2f} um
- Radius coefficient of variation: {float(np.std(radii) / np.mean(radii)):.3f}
- Target porosity: {cfg.target_porosity:.3f}
- Actual analytic porosity: {porosity:.3f}
- Minimum enforced surface gap: {cfg.minimum_surface_gap * 1e6:.2f} um
- Smallest realized pair gap: {throat_quantiles[0] * 1e6:.2f} um
- Fifth-percentile pair gap: {throat_quantiles[1] * 1e6:.2f} um
- Median pair gap: {throat_quantiles[3] * 1e6:.2f} um
- Relaxation sweeps: {sweeps}
- Periodic OpenFOAM cylinder images intersecting the base cell: {len(images)}

## User-requested void-fill grains

The pinned grains below were added to fill the top-left, mid-right, lower
mid/right, and grain-12-to-grain-3 open regions in the visual packing check.
They participate in the non-overlap quality checks, but their centers are held
fixed during relaxation so the requested voids remain filled.

{pinned_text}

## Connectivity and injection windows

- Raster left-to-right pore connectivity: {'yes' if connected else 'no'}
- Fraction of right-boundary fluid cells reached from the left boundary: {connected_fraction:.3f}
- Open intervals at x=0 for particle centers using a reference clearance of
  one particle radius plus a 1 nm contact gap:

{interval_text}

## Stagnation-point plan after OpenFOAM

For each grain, the upstream and downstream points should be defined from the
resolved flow, not from the global forcing direction.  After OpenFOAM is solved,
sample the velocity on a particle-center shell around each grain,
`r = R_g + a_p + h_probe`, using periodic images.  Compute the surface-tangent
velocity

`u_t(theta) = u(x_g + r n(theta)) dot t(theta)`.

Zeros of `u_t` are surface stagnation candidates for the particle-center
streamline.  Stable zeros, where `d u_t / d theta < 0` in the local
counter-clockwise convention, are downstream release/focusing points.  Unstable
zeros are upstream splitting points.  In a complex pore a grain may have more
than one stable downstream point, so release angles should be measured relative
to the stable point associated with the same tangential basin rather than to a
globally flow-aligned angle.

## Output files

- `random_porous_geometry.json`: full machine-readable geometry and metadata
- `random_porous_geometry.csv`: base-cell grain centers and radii
- `random_porous_geometry_openfoam.inc`: searchable-cylinder snippets for a later snappyHexMesh case
- `random_porous_geometry.png`: visual check of the periodic packing
""",
        encoding="utf-8",
    )


def draw_geometry(
    path: Path,
    centers: np.ndarray,
    radii: np.ndarray,
    cfg: GeometryConfig,
    intervals: list[tuple[float, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=180)
    ax.add_patch(Rectangle((0.0, 0.0), cfg.length_x, cfg.length_y, fill=False, lw=2.0, ec="#111827"))
    fixed = pinned_mask(cfg)

    for gid, ((cx, cy), r) in enumerate(zip(centers, radii)):
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                x = cx + ix * cfg.length_x
                y = cy + iy * cfg.length_y
                if x < -r or x > cfg.length_x + r or y < -r or y > cfg.length_y + r:
                    continue
                alpha = 1.0 if ix == 0 and iy == 0 else 0.45
                edge = "#f97316" if fixed[gid] else "#020617"
                lw = 2.0 if fixed[gid] else 0.8
                ax.add_patch(Circle((x, y), r, fc="#111827", ec=edge, lw=lw, alpha=alpha))
        ax.text(cx, cy, str(gid), color="white", fontsize=6.5, ha="center", va="center")

    for s, e in intervals:
        ax.plot([0.0, 0.0], [s, e], color="#22c55e", lw=5.0, solid_capstyle="butt")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.03 * cfg.length_x, 1.03 * cfg.length_x)
    ax.set_ylim(-0.03 * cfg.length_y, 1.03 * cfg.length_y)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Periodic random disk packing: N={cfg.n_grains}, porosity={actual_porosity(radii, cfg):.2f}, "
        f"R={np.mean(radii) * 1e6:.1f} +/- {np.std(radii) * 1e6:.1f} um"
    )
    ax.text(
        0.012 * cfg.length_x,
        0.985 * cfg.length_y,
        "green: open x=0 injection windows for particle centers",
        ha="left",
        va="top",
        fontsize=9,
        color="#166534",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    cfg = GeometryConfig()
    OUT.mkdir(parents=True, exist_ok=True)

    centers, radii, sweeps, final_overlap = make_pack(cfg)
    gaps = surface_gaps(centers, radii, cfg)
    if final_overlap > 5.0e-10 or float(np.min(gaps)) < cfg.minimum_surface_gap - 5.0e-10:
        raise RuntimeError("packing failed geometry quality checks")

    grain_mask = solid_mask(centers, radii, cfg)
    connected, connected_fraction = left_right_connected(~grain_mask)
    clearance = cfg.particle_radius_reference + 1.0e-9
    intervals = open_intervals_at_x(centers, radii, cfg, clearance=clearance)
    images = periodic_images(centers, radii, cfg)

    grains = [
        {
            "id": int(gid),
            "x": float(x),
            "y": float(y),
            "radius": float(radius),
            "pinned": bool(pinned_mask(cfg)[gid]),
            "role": "user_requested_void_fill" if pinned_mask(cfg)[gid] else "relaxed_random_pack",
        }
        for gid, ((x, y), radius) in enumerate(zip(centers, radii))
    ]
    porosity = actual_porosity(radii, cfg)
    data = {
        "config": asdict(cfg),
        "derived": {
            "equivalent_grain_radius": cfg.equivalent_grain_radius,
            "mean_grain_radius": float(np.mean(radii)),
            "min_grain_radius": float(np.min(radii)),
            "max_grain_radius": float(np.max(radii)),
            "radius_cv": float(np.std(radii) / np.mean(radii)),
            "actual_porosity": porosity,
            "solid_fraction": 1.0 - porosity,
            "minimum_realized_surface_gap": float(np.min(gaps)),
            "fifth_percentile_surface_gap": float(np.quantile(gaps, 0.05)),
            "median_surface_gap": float(np.median(gaps)),
            "left_right_connected": bool(connected),
            "right_boundary_connected_fraction": float(connected_fraction),
            "relaxation_sweeps": int(sweeps),
            "final_overlap": float(final_overlap),
        },
        "domain": {
            "length_x": cfg.length_x,
            "length_y": cfg.length_y,
            "periodic_x": True,
            "periodic_y": True,
        },
        "grains": grains,
        "periodic_images_for_openfoam": images,
        "injection_windows_x0": [
            {"y_min": float(s), "y_max": float(e), "width": float(e - s)} for s, e in intervals
        ],
        "stagnation_detection": {
            "status": "requires_resolved_openfoam_flow",
            "probe_shell": "grain_radius + particle_radius + h_probe",
            "candidate_condition": "zero crossing of tangential velocity on probe shell",
            "downstream_classification": "stable zero with d u_t / d theta < 0",
            "upstream_classification": "unstable zero with d u_t / d theta > 0",
        },
    }

    write_json(OUT / "random_porous_geometry.json", data)
    write_csv(OUT / "random_porous_geometry.csv", centers, radii)
    write_openfoam_includes(OUT / "random_porous_geometry_openfoam.inc", images, cfg)
    write_report(
        OUT / "random_porous_geometry_report.md",
        cfg,
        centers,
        radii,
        gaps,
        connected,
        connected_fraction,
        intervals,
        images,
        sweeps,
    )
    draw_geometry(OUT / "random_porous_geometry.png", centers, radii, cfg, intervals)

    print(OUT / "random_porous_geometry_report.md")
    print(OUT / "random_porous_geometry.png")


if __name__ == "__main__":
    main()
