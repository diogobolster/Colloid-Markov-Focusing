"""Create the focused-cell geometry and flow-field diagnostic figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
FLOW_PATH = ROOT / "outputs" / "openfoam_flow" / "openfoam_flow_N512.npz"
OUT_PATH = ROOT / "outputs" / "figures" / "fig_geometry_flow.png"


def draw_grains(ax: plt.Axes, *, facecolor: str = "black", edgecolor: str = "black") -> None:
    """Draw the center/corner collector geometry in microns."""
    L_um = 400.0
    R_um = 100.0
    for center in [(200.0, 200.0), (0.0, 0.0), (0.0, L_um), (L_um, 0.0), (L_um, L_um)]:
        ax.add_patch(
            patches.Circle(
                center,
                R_um,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.0,
                zorder=3,
            )
        )


def make_geometry_panel(ax: plt.Axes) -> None:
    L_um = 400.0
    R_um = 100.0
    yc = L_um / 2.0
    core_half_width = 12.5

    ax.set_facecolor("#f9fafb")
    ax.add_patch(
        patches.Rectangle(
            (0.0, 0.0),
            L_um,
            L_um,
            facecolor="white",
            edgecolor="#111827",
            linewidth=1.3,
            zorder=1,
        )
    )
    draw_grains(ax)

    ax.add_patch(
        patches.Rectangle(
            (-18.0, yc - core_half_width),
            18.0,
            2.0 * core_half_width,
            facecolor="#38bdf8",
            edgecolor="#0369a1",
            linewidth=1.0,
            alpha=0.85,
            zorder=4,
        )
    )
    ax.annotate(
        "",
        xy=(54.0, yc),
        xytext=(-35.0, yc),
        arrowprops=dict(arrowstyle="->", linewidth=2.0, color="#0369a1"),
        zorder=5,
    )
    ax.text(-39.0, yc + 22.0, r"central $y_{\rm in}$ band", ha="left", va="bottom", fontsize=8.5)

    ax.annotate(
        "",
        xy=(315.0, 335.0),
        xytext=(245.0, 335.0),
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#111827"),
        zorder=5,
    )
    ax.text(280.0, 350.0, r"$\bar u$", ha="center", va="bottom", fontsize=10)

    rear_points = [(200.0 + R_um, 200.0), (R_um, 0.0), (R_um, L_um)]
    for x, y in rear_points:
        ax.plot(x, y, "o", color="#dc2626", markersize=4.5, zorder=6)
    ax.annotate(
        "center rear\nstagnation zone",
        xy=(300.0, 200.0),
        xytext=(318.0, 238.0),
        arrowprops=dict(arrowstyle="->", linewidth=1.0, color="#dc2626"),
        fontsize=8.2,
        ha="left",
        va="bottom",
        color="#7f1d1d",
    )
    ax.annotate(
        "corner-family\nrear zones",
        xy=(100.0, 400.0),
        xytext=(132.0, 362.0),
        arrowprops=dict(arrowstyle="->", linewidth=1.0, color="#dc2626"),
        fontsize=8.2,
        ha="left",
        va="top",
        color="#7f1d1d",
    )

    ax.annotate(
        "",
        xy=(0.0, -19.0),
        xytext=(L_um, -19.0),
        arrowprops=dict(arrowstyle="<->", linewidth=1.0, color="#374151"),
    )
    ax.text(L_um / 2.0, -31.0, r"$L=400\ \mu{\rm m}$", ha="center", va="top", fontsize=8.8)
    ax.annotate(
        "",
        xy=(200.0, 200.0),
        xytext=(300.0, 200.0),
        arrowprops=dict(arrowstyle="<->", linewidth=1.0, color="#374151"),
    )
    ax.text(250.0, 190.0, r"$R=100\ \mu{\rm m}$", ha="center", va="top", fontsize=8.8, color="#374151")

    ax.set_xlim(-45.0, 445.0)
    ax.set_ylim(-45.0, 435.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x\ (\mu{\rm m})$")
    ax.set_ylabel(r"$y\ (\mu{\rm m})$")
    ax.set_title("Periodic center/corner-grain cell", fontsize=11, pad=8)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yticks([0, 100, 200, 300, 400])


def make_flow_panel(ax: plt.Axes) -> None:
    data = np.load(FLOW_PATH)
    x_um = data["x"] * 1.0e6
    y_um = data["y"] * 1.0e6
    ux_um_s = data["ux"] * 1.0e6
    uy_um_s = data["uy"] * 1.0e6
    solid = data["solid"]
    speed_um_s = np.sqrt(ux_um_s**2 + uy_um_s**2)
    speed_masked = np.ma.masked_where(solid, speed_um_s)

    extent = [0.0, 400.0, 0.0, 400.0]
    im = ax.imshow(
        speed_masked,
        origin="lower",
        extent=extent,
        cmap="turbo",
        interpolation="bilinear",
        zorder=1,
    )
    ax.imshow(
        np.ma.masked_where(~solid, solid),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["black"]),
        interpolation="nearest",
        zorder=3,
    )

    xx, yy = np.meshgrid(x_um, y_um)
    skip = 34
    xxq = xx[::skip, ::skip]
    yyq = yy[::skip, ::skip]
    uxq = ux_um_s[::skip, ::skip]
    uyq = uy_um_s[::skip, ::skip]
    open_q = ~solid[::skip, ::skip]
    speed_q = np.sqrt(uxq**2 + uyq**2)
    keep = open_q & (speed_q > 3.0)
    ax.quiver(
        xxq[keep],
        yyq[keep],
        uxq[keep],
        uyq[keep],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=3.8,
        width=0.0022,
        headwidth=3.8,
        headlength=4.8,
        headaxislength=4.2,
        alpha=0.82,
        zorder=4,
    )

    ax.set_xlim(0.0, 400.0)
    ax.set_ylim(0.0, 400.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x\ (\mu{\rm m})$")
    ax.set_ylabel(r"$y\ (\mu{\rm m})$")
    ax.set_title(r"Resolved Stokes flow, $\bar u=4\ {\rm m\,day^{-1}}$", fontsize=11, pad=8)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yticks([0, 100, 200, 300, 400])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.set_label(r"speed $(\mu{\rm m}\ {\rm s}^{-1})$")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.5), constrained_layout=True)
    make_geometry_panel(axes[0])
    make_flow_panel(axes[1])
    for label, ax in zip(["a", "b"], axes):
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
