#!/usr/bin/env python3
"""Create geometry diagnostics for the random periodic porous cell."""

from __future__ import annotations

import json
import math
import os
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
GEOM = OUT / "random_porous_geometry.json"


def periodic_delta(delta: np.ndarray | float, length: float) -> np.ndarray | float:
    return delta - length * np.round(delta / length)


def load_geometry() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(GEOM.read_text(encoding="utf-8"))
    grains = data["grains"]
    centers = np.array([[g["x"], g["y"]] for g in grains], dtype=float)
    radii = np.array([g["radius"] for g in grains], dtype=float)
    pinned = np.array([bool(g.get("pinned", False)) for g in grains], dtype=bool)
    return data, centers, radii, pinned


def surface_distance(
    xx: np.ndarray,
    yy: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    lx: float,
    ly: float,
) -> tuple[np.ndarray, np.ndarray]:
    d_min = np.full_like(xx, np.inf, dtype=float)
    nearest = np.full(xx.shape, -1, dtype=int)
    for gid, ((cx, cy), radius) in enumerate(zip(centers, radii)):
        dx = periodic_delta(xx - cx, lx)
        dy = periodic_delta(yy - cy, ly)
        d = np.sqrt(dx * dx + dy * dy) - radius
        take = d < d_min
        d_min[take] = d[take]
        nearest[take] = gid
    return d_min, nearest


def pair_gap_table(centers: np.ndarray, radii: np.ndarray, lx: float, ly: float) -> list[dict]:
    rows = []
    n = len(centers)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = float(periodic_delta(centers[j, 0] - centers[i, 0], lx))
            dy = float(periodic_delta(centers[j, 1] - centers[i, 1], ly))
            distance = math.hypot(dx, dy)
            gap = distance - radii[i] - radii[j]
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "gap": gap,
                    "distance": distance,
                    "mid_x": (centers[i, 0] + 0.5 * dx) % lx,
                    "mid_y": (centers[i, 1] + 0.5 * dy) % ly,
                }
            )
    rows.sort(key=lambda row: row["gap"])
    return rows


def largest_void_candidates(
    clearance: np.ndarray,
    nearest: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    lx: float,
    ly: float,
    *,
    n_candidates: int = 8,
    exclusion_radius: float = 90.0e-6,
) -> list[dict]:
    fluid = clearance > 0.0
    score = np.where(fluid, clearance, -np.inf)
    candidates = []

    while len(candidates) < n_candidates:
        idx = np.unravel_index(int(np.argmax(score)), score.shape)
        value = float(score[idx])
        if not np.isfinite(value) or value <= 0.0:
            break
        cx = float(x[idx[0]])
        cy = float(y[idx[1]])
        candidates.append(
            {
                "x": cx,
                "y": cy,
                "clearance": value,
                "nearest_grain": int(nearest[idx]),
            }
        )
        dx = periodic_delta(x[:, None] - cx, lx)
        dy = periodic_delta(y[None, :] - cy, ly)
        mask = dx * dx + dy * dy <= exclusion_radius * exclusion_radius
        score[mask] = -np.inf
    return candidates


def draw_disks(ax, centers: np.ndarray, radii: np.ndarray, pinned: np.ndarray, lx: float, ly: float) -> None:
    ax.add_patch(Rectangle((0.0, 0.0), lx, ly, fill=False, lw=1.5, ec="#111827"))
    for gid, ((cx, cy), radius) in enumerate(zip(centers, radii)):
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                x = cx + ix * lx
                y = cy + iy * ly
                if x < -radius or x > lx + radius or y < -radius or y > ly + radius:
                    continue
                edge = "#f97316" if pinned[gid] else "#020617"
                lw = 1.8 if pinned[gid] else 0.6
                alpha = 1.0 if ix == 0 and iy == 0 else 0.35
                ax.add_patch(Circle((x, y), radius, fc="#111827", ec=edge, lw=lw, alpha=alpha))
        ax.text(cx, cy, str(gid), color="white", fontsize=5.5, ha="center", va="center")


def write_report(
    path: Path,
    pair_rows: list[dict],
    candidates: list[dict],
    data: dict,
) -> None:
    narrow = pair_rows[:20]
    narrow_text = "\n".join(
        f"- {row['i']:02d}-{row['j']:02d}: gap={row['gap'] * 1e6:.2f} um, "
        f"mid=({row['mid_x'] * 1e6:.1f}, {row['mid_y'] * 1e6:.1f}) um"
        for row in narrow
    )
    candidate_text = "\n".join(
        f"- C{k + 1}: center=({c['x'] * 1e6:.1f}, {c['y'] * 1e6:.1f}) um, "
        f"clearance radius={c['clearance'] * 1e6:.1f} um, nearest grain={c['nearest_grain']}"
        for k, c in enumerate(candidates)
    )

    path.write_text(
        f"""# Random geometry diagnostics

Geometry source: `random_porous_geometry.json`

## Largest remaining pore void candidates

These points are local high-clearance pore centers on a periodic raster.  They
are not automatic recommendations to add grains; they are a way to guide visual
edits before solving the OpenFOAM flow.

{candidate_text}

## Narrowest grain-pair gaps

{narrow_text}

## Current geometry summary

- Grains: {len(data['grains'])}
- Porosity: {data['derived']['actual_porosity']:.3f}
- Minimum pair gap: {data['derived']['minimum_realized_surface_gap'] * 1e6:.2f} um
- Fifth-percentile pair gap: {data['derived']['fifth_percentile_surface_gap'] * 1e6:.2f} um
- Left-right connected: {data['derived']['left_right_connected']}
""",
        encoding="utf-8",
    )


def main() -> None:
    data, centers, radii, pinned = load_geometry()
    lx = float(data["domain"]["length_x"])
    ly = float(data["domain"]["length_y"])

    nx, ny = 760, 510
    x = (np.arange(nx) + 0.5) * lx / nx
    y = (np.arange(ny) + 0.5) * ly / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    clearance, nearest = surface_distance(xx, yy, centers, radii, lx, ly)
    pair_rows = pair_gap_table(centers, radii, lx, ly)
    candidates = largest_void_candidates(clearance, nearest, x, y, lx, ly)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3), dpi=180)

    ax = axes[0]
    draw_disks(ax, centers, radii, pinned, lx, ly)
    for k, c in enumerate(candidates[:5]):
        ax.plot(c["x"], c["y"], marker="x", ms=8, mew=2.0, color="#22c55e")
        ax.text(c["x"], c["y"], f"C{k + 1}", color="#166534", fontsize=7, ha="left", va="bottom")
    ax.set_title("Geometry with largest void candidates")

    ax = axes[1]
    clearance_um = np.ma.masked_less_equal(clearance * 1e6, 0.0)
    im = ax.imshow(
        clearance_um.T,
        extent=(0.0, lx, 0.0, ly),
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=np.nanpercentile(clearance_um.compressed(), 98),
        aspect="equal",
    )
    for (cx, cy), radius in zip(centers, radii):
        ax.add_patch(Circle((cx, cy), radius, fc="#111827", ec="none"))
    ax.set_title("Pore clearance to nearest grain (um)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("clearance (um)")

    ax = axes[2]
    draw_disks(ax, centers, radii, pinned, lx, ly)
    max_gap = 70.0e-6
    for row in pair_rows:
        if row["gap"] > max_gap:
            break
        x1, y1 = centers[row["i"]]
        x2 = (x1 + periodic_delta(centers[row["j"], 0] - x1, lx)) % lx
        y2 = (y1 + periodic_delta(centers[row["j"], 1] - y1, ly)) % ly
        color = "#dc2626" if row["gap"] < 12.0e-6 else "#f59e0b"
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.4, alpha=0.9)
        ax.text(row["mid_x"], row["mid_y"], f"{row['gap'] * 1e6:.0f}", fontsize=5.5, color=color)
    ax.set_title("Close grain pairs, labeled by gap (um)")

    for ax in axes:
        ax.set_xlim(-0.02 * lx, 1.02 * lx)
        ax.set_ylim(-0.02 * ly, 1.02 * ly)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    fig.tight_layout()
    fig.savefig(OUT / "random_porous_geometry_diagnostics.png")
    plt.close(fig)
    write_report(OUT / "random_porous_geometry_diagnostics.md", pair_rows, candidates, data)
    print(OUT / "random_porous_geometry_diagnostics.png")
    print(OUT / "random_porous_geometry_diagnostics.md")


if __name__ == "__main__":
    main()
