#!/usr/bin/env python3
"""Create a two-panel random-packing stagnation/flow diagnostic figure."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACKING_SCRIPT = ROOT / "scripts" / "run_random_particle_tracking.py"
spec = importlib.util.spec_from_file_location("random_particle_tracking", TRACKING_SCRIPT)
rpt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = rpt
spec.loader.exec_module(rpt)

GEOMETRY_PATH = ROOT / "outputs" / "random_porous_geometry" / "random_porous_geometry.json"
STAGNATION_CSV = ROOT / "outputs" / "random_grain_focusing_production_selected" / "grain_stagnation_points.csv"
OUT = ROOT / "outputs" / "figures" / "random_geometry_stagnation_and_flow.png"


def load_geometry() -> dict:
    return json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))


def load_stagnation_rows() -> list[dict[str, str]]:
    with STAGNATION_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_grains(ax, geometry: dict, *, labels: bool) -> None:
    for gid, grain in enumerate(geometry["grains"]):
        cx = float(grain["x"]) * 1.0e3
        cy = float(grain["y"]) * 1.0e3
        radius = float(grain["radius"]) * 1.0e3
        ax.add_patch(Circle((cx, cy), radius, facecolor="#111827", edgecolor="#111827", linewidth=0.45, zorder=4))
        if labels:
            ax.text(cx, cy, str(gid), color="white", ha="center", va="center", fontsize=5.8, zorder=5)


def solid_mask(geometry: dict, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    mask = np.zeros(xx.shape, dtype=bool)
    for grain in geometry["grains"]:
        dx = rpt.periodic_delta(xx - float(grain["x"]), lx)
        dy = rpt.periodic_delta(yy - float(grain["y"]), ly)
        mask |= np.sqrt(dx * dx + dy * dy) <= float(grain["radius"])
    return mask


def flow_grid(geometry: dict, nx: int = 420, ny: int = 280) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lx = float(geometry["domain"]["length_x"])
    ly = float(geometry["domain"]["length_y"])
    vtu = rpt.latest_vtu(rpt.DEFAULT_FLOW_CASE)
    centers_xy, velocity_xy, _ = rpt.flow_io.parse_xml_vtu(vtu)
    interpolator = rpt.PeriodicIDWFlow(centers_xy, velocity_xy, lx, ly, k=8)
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    ux, uy = interpolator.velocity_at(xx.ravel(), yy.ravel())
    ux = ux.reshape(xx.shape)
    uy = uy.reshape(xx.shape)
    mask = solid_mask(geometry, xx, yy)
    ux = np.where(mask, np.nan, ux)
    uy = np.where(mask, np.nan, uy)
    speed = np.sqrt(ux * ux + uy * uy)
    return xx, yy, ux, uy, speed


def setup_axis(ax, geometry: dict, title: str, panel: str) -> None:
    lx = float(geometry["domain"]["length_x"]) * 1.0e3
    ly = float(geometry["domain"]["length_y"]) * 1.0e3
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"({panel}) {title}", loc="left", fontsize=10.5, weight="bold")
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.9)


def main() -> None:
    geometry = load_geometry()
    rows = load_stagnation_rows()
    xx, yy, ux, uy, speed = flow_grid(geometry)
    extent = (
        0.0,
        float(geometry["domain"]["length_x"]) * 1.0e3,
        0.0,
        float(geometry["domain"]["length_y"]) * 1.0e3,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.15), dpi=260, constrained_layout=True)

    ax = axes[0]
    draw_grains(ax, geometry, labels=True)
    for kind, marker, color, label, size in (
        ("forward", "^", "#2563eb", "forward", 24),
        ("rear", "o", "#ef4444", "rear", 26),
    ):
        pts = np.array(
            [
                (float(row["x_m"]) * 1.0e3, float(row["y_m"]) * 1.0e3)
                for row in rows
                if row["stagnation_type"] == kind
            ],
            dtype=float,
        )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            label=label,
            zorder=6,
        )
    setup_axis(ax, geometry, "grain-local stagnation points", "a")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right", fontsize=8)

    ax = axes[1]
    im = ax.imshow(speed * 1.0e3, origin="lower", extent=extent, cmap="turbo", vmin=0.0, vmax=np.nanpercentile(speed * 1.0e3, 98.5))
    step = 22
    qx = xx[::step, ::step] * 1.0e3
    qy = yy[::step, ::step] * 1.0e3
    qu = ux[::step, ::step]
    qv = uy[::step, ::step]
    finite = np.isfinite(qu) & np.isfinite(qv)
    ax.quiver(
        qx[finite],
        qy[finite],
        qu[finite],
        qv[finite],
        color="white",
        angles="xy",
        scale_units="xy",
        scale=0.045,
        width=0.0021,
        headwidth=3.1,
        headlength=4.0,
        alpha=0.82,
        zorder=3,
    )
    draw_grains(ax, geometry, labels=False)
    setup_axis(ax, geometry, "resolved OpenFOAM speed field", "b")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.015)
    cbar.set_label("speed (mm/s)")

    fig.suptitle("Random periodic porous cell: local stagnation topology and resolved flow", fontsize=12.0, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
