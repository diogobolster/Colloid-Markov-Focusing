#!/usr/bin/env python3
"""Create a compact schematic defining grain-local release angle."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle


OUT = ROOT / "outputs" / "figures" / "release_angle_schematic.png"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=220)
    ax.set_aspect("equal")
    ax.axis("off")
    radius = 1.0
    near = 1.18
    ax.add_patch(Circle((0.0, 0.0), radius, fc="#111827", ec="#111827", zorder=2))
    ax.add_patch(Circle((0.0, 0.0), near, fc="none", ec="#64748b", lw=1.3, ls="--", zorder=1))
    ax.annotate("", xy=(1.72, 0.0), xytext=(-1.72, 0.0), arrowprops=dict(arrowstyle="->", lw=1.6, color="#2563eb"))
    ax.text(-1.68, 0.12, "local flow", color="#2563eb", fontsize=10)

    rear = 0.0
    release = math.radians(42.0)
    ax.scatter([near * math.cos(rear)], [near * math.sin(rear)], s=60, c="#dc2626", zorder=4)
    ax.text(1.26, -0.16, "rear point", fontsize=10, color="#dc2626")
    ax.scatter([near * math.cos(release)], [near * math.sin(release)], s=58, c="#7c3aed", zorder=5)
    ax.text(0.78, 0.91, "release", fontsize=10, color="#7c3aed")
    ax.plot(
        [0.0, near * math.cos(rear)],
        [0.0, near * math.sin(rear)],
        color="#dc2626",
        lw=1.1,
        alpha=0.8,
    )
    ax.plot(
        [0.0, near * math.cos(release)],
        [0.0, near * math.sin(release)],
        color="#7c3aed",
        lw=1.1,
        alpha=0.8,
    )
    arc = Arc((0.0, 0.0), 0.92, 0.92, theta1=0.0, theta2=42.0, color="#111827", lw=1.4)
    ax.add_patch(arc)
    ax.text(0.53, 0.22, r"$\theta_{\rm rel}$", fontsize=12)
    ax.text(-1.23, -1.34, "grain-local frame", fontsize=10)
    ax.text(-1.23, -1.52, r"$F_{30}: |\theta_{\rm rel}|\leq 30^\circ$", fontsize=10)
    ax.set_xlim(-1.8, 1.9)
    ax.set_ylim(-1.65, 1.45)
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
