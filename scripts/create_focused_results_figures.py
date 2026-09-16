"""Create diagnostic Results figures for the focused colloid calculations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import PhysicalParams  # noqa: E402


OUT = ROOT / "outputs" / "figures"
SUITE = ROOT / "outputs" / "openfoam_full_suite"
KB = 1.380649e-23
EPS0 = 8.8541878128e-12


PROFILE_ORDER = [
    "neutral_resolved",
    "unfavorable_6mM_z70",
    "unfavorable_20mM_z70",
    "unfavorable_50mM_z70",
    "unfavorable_100mM_z70",
    "unfavorable_50mM_z50",
    "unfavorable_50mM_z30",
    "unfavorable_100mM_z30",
    "mechanism_50mM_z20_A10x",
    "mechanism_100mM_z20_A10x",
    "neutral_100xD",
    "unfavorable_50mM_z70_100xD",
]

COMPACT_LABEL = {
    "neutral_resolved": "No DLVO",
    "unfavorable_6mM_z70": "6/-70",
    "unfavorable_20mM_z70": "20/-70",
    "unfavorable_50mM_z70": "50/-70",
    "unfavorable_100mM_z70": "100/-70",
    "unfavorable_50mM_z50": "50/-50",
    "unfavorable_50mM_z30": "50/-30",
    "unfavorable_100mM_z30": "100/-30",
    "mechanism_50mM_z20_A10x": "50/-20, 10xA",
    "mechanism_100mM_z20_A10x": "100/-20, 10xA",
    "neutral_100xD": "No DLVO, 100xD",
    "unfavorable_50mM_z70_100xD": "50/-70, 100xD",
}

SHORT_LABEL = {
    "neutral_resolved": "No DLVO",
    "unfavorable_6mM_z70": "6 mM\n-70 mV",
    "unfavorable_20mM_z70": "20 mM\n-70 mV",
    "unfavorable_50mM_z70": "50 mM\n-70 mV",
    "unfavorable_100mM_z70": "100 mM\n-70 mV",
    "unfavorable_50mM_z50": "50 mM\n-50 mV",
    "unfavorable_50mM_z30": "50 mM\n-30 mV",
    "unfavorable_100mM_z30": "100 mM\n-30 mV",
    "mechanism_50mM_z20_A10x": "50 mM\n-20 mV\n10x A",
    "mechanism_100mM_z20_A10x": "100 mM\n-20 mV\n10x A",
    "neutral_100xD": "No DLVO\n100x D",
    "unfavorable_50mM_z70_100xD": "50 mM\n-70 mV\n100x D",
}

PALETTE = {
    "control": "#6b7280",
    "realistic": "#0f766e",
    "focusing": "#2563eb",
    "zeta": "#65a30d",
    "trap": "#7c3aed",
    "diffusion": "#c026d3",
    "warning": "#dc2626",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def params_from_row(row: pd.Series) -> PhysicalParams:
    return replace(
        PhysicalParams(),
        ionic_strength_molar=float(row["ionic_strength_mM"]) * 1.0e-3,
        zeta_collector_unfavorable=float(row["zeta_collector_unfavorable_mV"]) * 1.0e-3,
        hamaker=float(row["hamaker_J"]),
        diffusivity_multiplier=float(row["diffusivity_multiplier"]),
    )


def dlvo_potential(params: PhysicalParams, h: np.ndarray) -> np.ndarray:
    eps = params.relative_permittivity * EPS0
    kappa = 1.0 / params.debye_length
    c_edl = (
        2.0
        * np.pi
        * eps
        * params.particle_radius
        * kappa
        * params.zeta_particle
        * params.zeta_collector_unfavorable
    )
    u = c_edl / kappa * np.exp(-kappa * h) - params.hamaker * params.particle_radius / (6.0 * h)
    return u / (KB * params.temperature)


def finite_series(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def ordered_summary() -> pd.DataFrame:
    df = pd.read_csv(SUITE / "openfoam_full_refinement_summary.csv")
    idx = {profile: i for i, profile in enumerate(PROFILE_ORDER)}
    df = df[df["profile"].isin(PROFILE_ORDER)].copy()
    df["order"] = df["profile"].map(idx)
    return df.sort_values("order").reset_index(drop=True)


def transition_center_rows() -> pd.DataFrame:
    df = pd.read_csv(SUITE / "openfoam_full_transition_rows.csv")
    return df[df["state"].eq("center core")].copy()


def abs_angle_deg(theta: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(np.arctan2(np.sin(theta), np.cos(theta))))


def library(profile: str, ensemble: str = "refinement") -> np.lib.npyio.NpzFile:
    return np.load(SUITE / f"trajectory_library_{ensemble}_{profile}.npz")


def profile_color(profile: str) -> str:
    if profile.startswith("neutral"):
        return PALETTE["control"] if "100xD" not in profile else PALETTE["diffusion"]
    if "100xD" in profile:
        return PALETTE["diffusion"]
    if "mechanism" in profile or profile == "unfavorable_100mM_z30":
        return PALETTE["trap"]
    if profile in {"unfavorable_50mM_z50", "unfavorable_50mM_z30"}:
        return PALETTE["zeta"]
    if profile in {"unfavorable_50mM_z70", "unfavorable_100mM_z70"}:
        return PALETTE["focusing"]
    return PALETTE["realistic"]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def figure_01_dlvo_landscapes(df: pd.DataFrame) -> Path:
    h = np.logspace(np.log10(1.0e-9), np.log10(2.0e-7), 1200)
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.8), constrained_layout=True)
    ax = axes[0]
    salt_profiles = [
        "unfavorable_6mM_z70",
        "unfavorable_20mM_z70",
        "unfavorable_50mM_z70",
        "unfavorable_100mM_z70",
    ]
    for profile in salt_profiles:
        row = df.loc[df["profile"].eq(profile)].iloc[0]
        u = dlvo_potential(params_from_row(row), h)
        ax.plot(h * 1.0e9, np.clip(u, -55.0, 30.0), lw=2.0, label=SHORT_LABEL[profile].replace("\n", ", "))
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.axhline(-1.0, color="#64748b", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(1.0, 200.0)
    ax.set_ylim(-55.0, 30.0)
    ax.set_xlabel(r"separation $h$ (nm)")
    ax.set_ylabel(r"$U(h)/k_BT$")
    ax.set_title(r"DLVO landscapes at $\zeta_c=-70$ mV")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", color="#e5e7eb", lw=0.6)

    ax = axes[1]
    plot_df = df[df["profile"].isin([p for p in PROFILE_ORDER if p != "neutral_resolved" and p != "neutral_100xD"])].copy()
    x = -finite_series(plot_df, "well_minimum_kbt")
    y = finite_series(plot_df, "center_well_time_median_s")
    f30 = finite_series(plot_df, "center_well_release_theta30_fraction")
    colors = [profile_color(p) for p in plot_df["profile"]]
    sizes = np.where(np.isfinite(f30), 60.0 + 160.0 * np.nan_to_num(f30, nan=0.0), 70.0)
    ax.scatter(x, y, s=sizes, c=colors, edgecolor="white", linewidth=0.8, alpha=0.95)
    for _, row in plot_df.iterrows():
        ax.annotate(
            SHORT_LABEL[row["profile"]].replace("\n", " "),
            (-(float(row["well_minimum_kbt"])), float(row["center_well_time_median_s"])),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
            color="#111827",
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"secondary-minimum depth, $-U_{\min}/k_BT$")
    ax.set_ylabel("median center-well residence (s)")
    ax.set_title("Residence grows with secondary-minimum depth")
    ax.grid(True, which="both", color="#e5e7eb", lw=0.6)
    ax.text(
        0.02,
        0.04,
        "marker area scales with F30 when releases occur",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#475569",
    )
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    path = OUT / "focused_results_01_dlvo_regimes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def event_data(profile: str) -> tuple[np.ndarray, np.ndarray, str]:
    d = library(profile)
    if profile.startswith("neutral"):
        mask = (
            d["exited"]
            & ~d["censored"]
            & (d["center_interceptions"] > 0)
            & (d["collector_exit"] == 1)
            & np.isfinite(d["theta_exit"])
        )
        return d["near_time"][mask], abs_angle_deg(d["theta_exit"][mask]), "near-surface release"
    mask = (
        d["exited"]
        & ~d["censored"]
        & (d["center_well_interceptions"] > 0)
        & (d["collector_well_exit"] == 1)
        & np.isfinite(d["theta_well_exit"])
    )
    return d["well_time"][mask], abs_angle_deg(d["theta_well_exit"][mask]), "secondary-minimum release"


def figure_02_event_clouds() -> Path:
    profiles = [
        "neutral_resolved",
        "unfavorable_6mM_z70",
        "unfavorable_50mM_z70",
        "unfavorable_100mM_z70",
        "unfavorable_50mM_z30",
        "unfavorable_50mM_z70_100xD",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.9, 5.7), sharex=False, sharey=True, constrained_layout=True)
    rng = np.random.default_rng(971)
    for ax, profile in zip(axes.ravel(), profiles):
        t, theta, event = event_data(profile)
        if t.size:
            take = rng.choice(t.size, size=min(t.size, 2500), replace=False)
            ax.scatter(t[take], theta[take], s=4, c=profile_color(profile), alpha=0.22, edgecolors="none")
            ax.axvline(np.median(t), color="#111827", lw=1.0, ls="--")
            ax.axhline(np.median(theta), color="#111827", lw=1.0, ls=":")
            ax.text(
                0.04,
                0.06,
                f"N={t.size}\nmedian angle={np.median(theta):.1f} deg",
                transform=ax.transAxes,
                fontsize=7.5,
                color="#111827",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 180)
        ax.set_title(SHORT_LABEL[profile].replace("\n", ", "), fontsize=9)
        ax.grid(True, which="both", color="#e5e7eb", lw=0.6)
        ax.text(0.04, 0.91, event, transform=ax.transAxes, fontsize=7.2, color="#475569")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|\theta_{\rm rel}|$ from rear stagnation (deg)")
    for ax in axes[-1, :]:
        ax.set_xlabel("event residence time (s)")
    add_panel_label(axes[0, 0], "a")
    add_panel_label(axes[0, 1], "b")
    add_panel_label(axes[0, 2], "c")
    add_panel_label(axes[1, 0], "d")
    add_panel_label(axes[1, 1], "e")
    add_panel_label(axes[1, 2], "f")
    path = OUT / "focused_results_02_event_clouds.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_03_focusing_window(df: pd.DataFrame) -> Path:
    labels = [SHORT_LABEL[p] for p in df["profile"]]
    x = np.arange(len(df))
    colors = [profile_color(p) for p in df["profile"]]
    fig, axes = plt.subplots(4, 1, figsize=(8.9, 8.0), sharex=True, constrained_layout=True)

    f30 = finite_series(df, "center_well_release_theta30_fraction")
    err = finite_series(df, "center_well_release_theta30_halfwidth_95")
    axes[0].bar(x, np.nan_to_num(f30, nan=0.0), color=colors, alpha=0.92)
    valid = np.isfinite(f30)
    axes[0].errorbar(x[valid], f30[valid], yerr=err[valid], fmt="none", ecolor="#111827", lw=0.8, capsize=2)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel(r"$F_{30}$")
    axes[0].set_title("Focusing probability for center-grain secondary-minimum releases")
    axes[0].grid(True, axis="y", color="#e5e7eb", lw=0.6)

    angle = finite_series(df, "center_well_release_median_theta_deg")
    axes[1].bar(x, np.nan_to_num(angle, nan=0.0), color=colors, alpha=0.92)
    axes[1].set_ylim(0, 92)
    axes[1].set_ylabel("median angle (deg)")
    axes[1].set_title("Release angle collapses toward the downstream stagnation point")
    axes[1].grid(True, axis="y", color="#e5e7eb", lw=0.6)

    residence = finite_series(df, "center_well_time_median_s")
    axes[2].bar(x, np.nan_to_num(residence, nan=0.0), color=colors, alpha=0.92)
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.001, 1.0e4)
    axes[2].set_ylabel("median residence (s)")
    axes[2].set_title("The same conditions increase secondary-minimum residence")
    axes[2].grid(True, axis="y", which="both", color="#e5e7eb", lw=0.6)

    unresolved = finite_series(df, "center_well_censored_count")
    entries = finite_series(df, "center_well_intercepted_count")
    frac = np.divide(unresolved, entries, out=np.zeros_like(unresolved), where=entries > 0)
    axes[3].bar(x, frac, color=[PALETTE["warning"] if v > 0 else "#94a3b8" for v in frac], alpha=0.92)
    axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("unresolved fraction")
    axes[3].set_title("Completion distinguishes clean focusing from persistent residence")
    axes[3].grid(True, axis="y", color="#e5e7eb", lw=0.6)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=0)
    for ax, label in zip(axes, ["a", "b", "c", "d"]):
        add_panel_label(ax, label)
    path = OUT / "focused_results_03_focusing_window.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def load_payload(profile: str) -> np.lib.npyio.NpzFile:
    return np.load(SUITE / f"openfoam_full_transition_payload_{profile}.npz")


def highres_transition_arrays(profile: str, n_bins: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = library(profile, ensemble="transition")
    y_edges = np.linspace(100.551e-6, 299.449e-6, n_bins + 1)
    theta_edges = np.linspace(0.0, 180.0, n_bins + 1)
    yi = np.digitize(d["y_in"], y_edges) - 1
    yo = np.digitize(d["y_out"], y_edges) - 1
    mobile = d["exited"] & ~d["censored"] & (yi >= 0) & (yi < n_bins) & (yo >= 0) & (yo < n_bins)

    y_counts = np.zeros((n_bins, n_bins), dtype=float)
    np.add.at(y_counts, (yi[mobile], yo[mobile]), 1.0)
    y_rows = y_counts.sum(axis=1)
    y_prob = np.divide(y_counts, y_rows[:, None], out=np.full_like(y_counts, np.nan), where=y_rows[:, None] > 0)

    if profile.startswith("neutral"):
        event = (
            d["exited"]
            & ~d["censored"]
            & (d["center_interceptions"] > 0)
            & (d["collector_exit"] == 1)
            & np.isfinite(d["theta_exit"])
        )
        theta = d["theta_exit"]
    else:
        event = (
            d["exited"]
            & ~d["censored"]
            & (d["center_well_interceptions"] > 0)
            & (d["collector_well_exit"] == 1)
            & np.isfinite(d["theta_well_exit"])
        )
        theta = d["theta_well_exit"]
    theta_abs = abs_angle_deg(theta)
    ti = np.digitize(theta_abs, theta_edges) - 1
    event &= (yi >= 0) & (yi < n_bins) & (ti >= 0) & (ti < n_bins)
    theta_counts = np.zeros((n_bins, n_bins), dtype=float)
    np.add.at(theta_counts, (yi[event], ti[event]), 1.0)
    theta_rows = theta_counts.sum(axis=1)
    theta_prob = np.divide(
        theta_counts,
        theta_rows[:, None],
        out=np.full_like(theta_counts, np.nan),
        where=theta_rows[:, None] >= 10,
    )
    return y_prob, theta_prob, y_edges * 1.0e6, theta_edges


def figure_04_transition_matrices() -> Path:
    profiles = [
        "neutral_resolved",
        "unfavorable_50mM_z70",
        "unfavorable_100mM_z70",
        "unfavorable_50mM_z70_100xD",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 7.2), constrained_layout=False)
    blue = plt.get_cmap("Blues").copy()
    blue.set_bad("#f1f5f9")
    for ax, profile in zip(axes.ravel(), profiles):
        y_prob, _, y_edges_um, _ = highres_transition_arrays(profile, n_bins=100)
        extent_y = [y_edges_um[0], y_edges_um[-1], y_edges_um[0], y_edges_um[-1]]
        im_y = ax.imshow(y_prob, origin="lower", aspect="auto", extent=extent_y, vmin=0.0, vmax=0.45, cmap=blue)
        ax.set_title(SHORT_LABEL[profile].replace("\n", ", "), fontsize=9)
        ax.axhspan(187.5, 212.5, color="#dc2626", alpha=0.10)
        ax.axvspan(187.5, 212.5, color="#dc2626", alpha=0.08)
        ax.plot([y_edges_um[0], y_edges_um[-1]], [y_edges_um[0], y_edges_um[-1]], color="#111827", lw=0.7, alpha=0.6)
        ax.set_xlabel(r"$y_{\rm out}$ ($\mu$m)")
        ax.set_ylabel(r"$y_{\rm in}$ ($\mu$m)")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.13, top=0.88, wspace=0.30, hspace=0.42)
    fig.suptitle(r"100-bin ordinary mobile transition matrices, $P(y_{\rm out}\mid y_{\rm in},{\rm mobile})$", fontsize=12, fontweight="bold", y=0.96)
    cbar_y = fig.colorbar(im_y, cax=fig.add_axes([0.91, 0.22, 0.022, 0.55]))
    cbar_y.set_label("row probability")
    axes[1, 0].text(
        0.02,
        -0.24,
        "Red bands mark the center-grazing inlet and outlet interval. These ordinary matrices preserve transverse transport but only weakly expose focusing.",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        color="#475569",
    )
    for label, ax in zip(["a", "b", "c", "d"], axes.ravel()):
        add_panel_label(ax, label)
    path = OUT / "focused_results_04_highres_y_transition_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_05_focusing_matrices() -> Path:
    profiles = [
        "neutral_resolved",
        "unfavorable_50mM_z70",
        "unfavorable_100mM_z70",
        "unfavorable_50mM_z70_100xD",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 7.2), constrained_layout=False)
    green = plt.get_cmap("YlGnBu").copy()
    green.set_bad("#f1f5f9")
    for ax, profile in zip(axes.ravel(), profiles):
        _, theta_prob, y_edges_um, theta_edges = highres_transition_arrays(profile, n_bins=100)
        extent_theta = [theta_edges[0], theta_edges[-1], y_edges_um[0], y_edges_um[-1]]
        im_t = ax.imshow(theta_prob, origin="lower", aspect="auto", extent=extent_theta, vmin=0.0, vmax=0.45, cmap=green)
        ax.set_title(SHORT_LABEL[profile].replace("\n", ", "), fontsize=9)
        ax.axhspan(187.5, 212.5, color="#dc2626", alpha=0.10)
        ax.axvspan(0.0, 30.0, color="#dc2626", alpha=0.08)
        ax.set_xlabel(r"$|\theta_{\rm rel}|$ (deg)")
        ax.set_ylabel(r"$y_{\rm in}$ ($\mu$m)")
        ax.set_xlim(0, 180)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.13, top=0.88, wspace=0.30, hspace=0.42)
    fig.suptitle(r"100-bin event-conditioned focusing matrices, $P(|\theta_{\rm rel}|\mid y_{\rm in},{\rm event})$", fontsize=12, fontweight="bold", y=0.96)
    cbar_t = fig.colorbar(im_t, cax=fig.add_axes([0.91, 0.22, 0.022, 0.55]))
    cbar_t.set_label("row probability")
    axes[1, 0].text(
        0.02,
        -0.24,
        "Red vertical band is the focusing window, |theta_rel| <= 30 deg. Rows with fewer than 10 release events are masked.",
        transform=axes[1, 0].transAxes,
        fontsize=8,
        color="#475569",
    )
    for label, ax in zip(["a", "b", "c", "d"], axes.ravel()):
        add_panel_label(ax, label)
    path = OUT / "focused_results_05_highres_focusing_matrices.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_99_transition_kernels() -> Path:
    profiles = [
        "neutral_resolved",
        "unfavorable_20mM_z70",
        "unfavorable_50mM_z70",
        "unfavorable_100mM_z70",
        "unfavorable_50mM_z30",
        "unfavorable_50mM_z70_100xD",
    ]
    angle_rows = []
    yout_rows = []
    row_labels = []
    angle_edges = None
    state_edges = None
    center_state = 3
    for profile in profiles:
        p = load_payload(profile)
        angle_edges = p["angle_edges_deg"]
        state_edges = p["state_edges_m"] * 1.0e6
        if profile.startswith("neutral"):
            row = p["center_near_angle_probability"][center_state]
        else:
            row = p["center_well_angle_probability"][center_state]
        angle_rows.append(row)
        yout_rows.append(p["mobile_probability_conditional"][center_state])
        row_labels.append(SHORT_LABEL[profile].replace("\n", ", "))

    angle_matrix = np.vstack(angle_rows)
    yout_matrix = np.vstack(yout_rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.4), constrained_layout=True)
    im0 = axes[0].imshow(angle_matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    axes[0].set_yticks(np.arange(len(row_labels)))
    axes[0].set_yticklabels(row_labels)
    axes[0].set_xticks(np.arange(len(angle_edges) - 1))
    axes[0].set_xticklabels([f"{angle_edges[i]:.0f}-{angle_edges[i+1]:.0f}" for i in range(len(angle_edges) - 1)], rotation=30, ha="right")
    axes[0].set_title("Event-conditioned release-angle kernel")
    axes[0].set_xlabel(r"$|\theta_{\rm rel}|$ bins (deg)")
    for i in range(angle_matrix.shape[0]):
        for j in range(angle_matrix.shape[1]):
            val = angle_matrix[i, j]
            text = "n/a" if not np.isfinite(val) else f"{val:.2f}"
            axes[0].text(j, i, text, ha="center", va="center", fontsize=7, color="white" if np.isfinite(val) and val > 0.55 else "#111827")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02, label="probability")

    im1 = axes[1].imshow(yout_matrix, aspect="auto", vmin=0, vmax=1, cmap="PuBu")
    axes[1].set_yticks(np.arange(len(row_labels)))
    axes[1].set_yticklabels([])
    axes[1].set_xticks(np.arange(len(state_edges) - 1))
    axes[1].set_xticklabels([f"{state_edges[i]:.0f}-{state_edges[i+1]:.0f}" for i in range(len(state_edges) - 1)], rotation=35, ha="right")
    axes[1].set_title(r"Ordinary mobile $y_{\rm out}$ row")
    axes[1].set_xlabel(r"$y_{\rm out}$ bins ($\mu$m)")
    for i in range(yout_matrix.shape[0]):
        for j in range(yout_matrix.shape[1]):
            val = yout_matrix[i, j]
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > 0.55 else "#111827")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02, label="probability")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    path = OUT / "focused_results_05_transition_kernels.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_06_diffusion_control() -> Path:
    profiles = ["neutral_resolved", "unfavorable_50mM_z70", "neutral_100xD", "unfavorable_50mM_z70_100xD"]
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.8), constrained_layout=True)
    bins = np.linspace(0, 180, 19)
    for profile in profiles:
        t, theta, _ = event_data(profile)
        if theta.size == 0:
            continue
        hist, edges = np.histogram(theta, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ls = "--" if "100xD" in profile else "-"
        axes[0].plot(centers, hist, lw=2.0, ls=ls, color=profile_color(profile), label=SHORT_LABEL[profile].replace("\n", ", "))
    axes[0].axvspan(0, 30, color="#dbeafe", alpha=0.45, zorder=0)
    axes[0].set_xlabel(r"$|\theta_{\rm rel}|$ from rear stagnation (deg)")
    axes[0].set_ylabel("probability density")
    axes[0].set_title("High diffusion erases the focusing signature")
    axes[0].legend(frameon=False)
    axes[0].grid(True, color="#e5e7eb", lw=0.6)

    summary = ordered_summary().set_index("profile")
    x = np.arange(2)
    normal = summary.loc["unfavorable_50mM_z70"]
    highd = summary.loc["unfavorable_50mM_z70_100xD"]
    f30 = [normal["center_well_release_theta30_fraction"], highd["center_well_release_theta30_fraction"]]
    time = [normal["center_well_time_median_s"], highd["center_well_time_median_s"]]
    ax2 = axes[1]
    ax2.bar(x - 0.18, f30, width=0.36, color=[profile_color("unfavorable_50mM_z70"), profile_color("unfavorable_50mM_z70_100xD")], label=r"$F_{30}$")
    ax2.set_ylim(0, 0.75)
    ax2.set_ylabel(r"$F_{30}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["normal D", "100x D"])
    ax2.grid(True, axis="y", color="#e5e7eb", lw=0.6)
    ax3 = ax2.twinx()
    ax3.scatter(x + 0.18, time, marker="D", s=55, color="#111827", label="median residence")
    ax3.set_yscale("log")
    ax3.set_ylim(0.001, 100)
    ax3.set_ylabel("median residence (s)")
    ax2.set_title("50 mM, -70 mV unfavorable")
    axes[1].text(
        0.03,
        0.92,
        "100xD samples the well often\nbut residence is too short",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#475569",
    )
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    path = OUT / "focused_results_06_diffusion_control.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_07_completion_outcomes(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 4.0), constrained_layout=True)
    labels = [COMPACT_LABEL[p] for p in df["profile"]]
    y_pos = np.arange(len(df))
    entries = finite_series(df, "center_well_intercepted_count")
    releases = finite_series(df, "center_well_release_count")
    unresolved = finite_series(df, "center_well_censored_count")
    no_well = np.maximum(entries - releases - unresolved, 0.0)
    release_frac = np.divide(releases, entries, out=np.zeros_like(releases), where=entries > 0)
    unresolved_frac = np.divide(unresolved, entries, out=np.zeros_like(unresolved), where=entries > 0)
    other_frac = np.divide(no_well, entries, out=np.zeros_like(no_well), where=entries > 0)
    axes[0].barh(y_pos, release_frac, color="#0f766e", label="released")
    axes[0].barh(y_pos, unresolved_frac, left=release_frac, color="#dc2626", label="unresolved")
    axes[0].barh(y_pos, other_frac, left=release_frac + unresolved_frac, color="#cbd5e1", label="other")
    for yy, n in zip(y_pos, entries):
        if n > 0:
            axes[0].text(1.02, yy, f"N={int(n)}", va="center", ha="left", fontsize=7.5, color="#475569")
    axes[0].set_xlim(0, 1.22)
    axes[0].set_xlabel("fraction of center-well entries")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Release support and unresolved long residence")
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    axes[0].grid(True, axis="x", color="#e5e7eb", lw=0.6)

    y = finite_series(df, "center_well_release_theta30_fraction")
    resid = finite_series(df, "center_well_time_median_s")
    colors = [profile_color(p) for p in df["profile"]]
    size = 60 + 520 * unresolved_frac
    axes[1].scatter(resid, y, s=size, c=colors, edgecolor="white", linewidth=0.8, alpha=0.92)
    for _, row in df.iterrows():
        if np.isfinite(row["center_well_release_theta30_fraction"]):
            axes[1].annotate(
                SHORT_LABEL[row["profile"]].replace("\n", " "),
                (row["center_well_time_median_s"], row["center_well_release_theta30_fraction"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.005, 5000)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("median center-well residence (s)")
    axes[1].set_ylabel(r"$F_{30}$")
    axes[1].set_title("Clean focusing occupies an intermediate residence window")
    axes[1].grid(True, which="both", color="#e5e7eb", lw=0.6)
    axes[1].text(0.04, 0.06, "marker area scales with unresolved fraction", transform=axes[1].transAxes, fontsize=7.5, color="#475569")
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    path = OUT / "focused_results_07_completion_outcomes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    setup()
    df = ordered_summary()
    paths = [
        figure_01_dlvo_landscapes(df),
        figure_02_event_clouds(),
        figure_03_focusing_window(df),
        figure_04_transition_matrices(),
        figure_05_focusing_matrices(),
        figure_06_diffusion_control(),
        figure_07_completion_outcomes(df),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
