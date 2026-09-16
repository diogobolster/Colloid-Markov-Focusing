"""Tiny SVG helpers so the demo has no plotting dependency beyond NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlotStyle:
    color: str
    label: str
    dash: str = ""


def _polyline(points: list[tuple[float, float]], color: str, width: float = 2.0, dash: str = "") -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _split_periodic_path(x: np.ndarray, y: np.ndarray, length: float) -> list[tuple[np.ndarray, np.ndarray]]:
    if x.size <= 1:
        return [(x, y)]
    jumps = (np.abs(np.diff(x)) > 0.5 * length) | (np.abs(np.diff(y)) > 0.5 * length)
    starts = np.r_[0, np.flatnonzero(jumps) + 1]
    stops = np.r_[np.flatnonzero(jumps) + 1, x.size]
    return [(x[s:e], y[s:e]) for s, e in zip(starts, stops) if e - s > 1]


def write_retention_svg(
    path: str,
    series: dict[str, tuple[np.ndarray, np.ndarray, PlotStyle]],
    title: str,
    width: int = 860,
    height: int = 560,
) -> None:
    margin_l, margin_r, margin_t, margin_b = 78, 24, 54, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_max = max(float(np.nanmax(x)) for x, _, _ in series.values())
    positive = np.concatenate([y[y > 0] for _, y, _ in series.values()])
    y_min = max(float(np.nanmin(positive)), 1.0e-5)
    y_max = max(float(np.nanmax(positive)), y_min * 10.0)
    y_min = 10.0 ** np.floor(np.log10(y_min))
    y_max = 10.0 ** np.ceil(np.log10(y_max))

    def sx(x: np.ndarray | float) -> np.ndarray | float:
        return margin_l + np.asarray(x) / x_max * plot_w

    def sy(y: np.ndarray | float) -> np.ndarray | float:
        ly = np.log10(np.maximum(np.asarray(y), y_min))
        return margin_t + (np.log10(y_max) - ly) / (np.log10(y_max) - np.log10(y_min)) * plot_h

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#222"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#222"/>',
    ]

    for tick in np.linspace(0, x_max, 6):
        xpx = float(sx(tick))
        lines.append(f'<line x1="{xpx:.2f}" y1="{margin_t + plot_h}" x2="{xpx:.2f}" y2="{margin_t + plot_h + 6}" stroke="#222"/>')
        lines.append(f'<text x="{xpx:.2f}" y="{margin_t + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.1f}</text>')

    for exp in range(int(np.log10(y_min)), int(np.log10(y_max)) + 1):
        val = 10.0 ** exp
        ypx = float(sy(val))
        lines.append(f'<line x1="{margin_l - 6}" y1="{ypx:.2f}" x2="{margin_l}" y2="{ypx:.2f}" stroke="#222"/>')
        lines.append(f'<line x1="{margin_l}" y1="{ypx:.2f}" x2="{margin_l + plot_w}" y2="{ypx:.2f}" stroke="#e7e7e7"/>')
        lines.append(f'<text x="{margin_l - 10}" y="{ypx + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">1e{exp}</text>')

    for _, (x, y, style) in series.items():
        mask = y > 0
        pts = list(zip(sx(x[mask]).astype(float), sy(y[mask]).astype(float)))
        if len(pts) > 1:
            lines.append(_polyline(pts, style.color, 2.4, style.dash))

    legend_x = margin_l + plot_w - 230
    legend_y = margin_t + 16
    lines.append(f'<rect x="{legend_x - 12}" y="{legend_y - 18}" width="238" height="{24 * len(series) + 14}" fill="white" stroke="#ccc"/>')
    for i, (_, (_, _, style)) in enumerate(series.items()):
        y0 = legend_y + 24 * i
        dash = f' stroke-dasharray="{style.dash}"' if style.dash else ""
        lines.append(f'<line x1="{legend_x}" y1="{y0}" x2="{legend_x + 34}" y2="{y0}" stroke="{style.color}" stroke-width="3"{dash}/>')
        lines.append(f'<text x="{legend_x + 44}" y="{y0 + 4}" font-family="Arial" font-size="13">{style.label}</text>')

    lines.append(f'<text x="{margin_l + plot_w/2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">transport distance, x/L</text>')
    lines.append(f'<text transform="translate(20 {margin_t + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">retained fraction per bin</text>')
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_geometry_svg(
    path: str,
    trajectories: list[dict[str, np.ndarray | bool]],
    length: float,
    grain_radius: float,
    zoi: float,
    title: str = "Single-grain periodic training trajectories",
    width: int = 760,
    height: int = 520,
) -> None:
    margin_l, margin_t = 54, 48
    scale = min((width - 2 * margin_l) / length, (height - 2 * margin_t) / length)
    ox = margin_l
    oy = margin_t

    def sx(x: np.ndarray | float) -> np.ndarray | float:
        return ox + np.asarray(x) * scale

    def sy(y: np.ndarray | float) -> np.ndarray | float:
        return oy + (length - np.asarray(y)) * scale

    cx = float(sx(0.5 * length))
    cy = float(sy(0.5 * length))
    r = grain_radius * scale
    rz = (grain_radius + zoi) * scale

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<rect x="{ox}" y="{oy}" width="{length*scale}" height="{length*scale}" fill="#f7fbff" stroke="#222"/>',
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rz:.2f}" fill="#d7ecff" stroke="#69aeea" stroke-dasharray="5,4"/>',
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="#111"/>',
    ]
    palette = ["#2364aa", "#3da35d", "#f18f01", "#c73e1d", "#6f42c1", "#0f8b8d"]
    for i, tr in enumerate(trajectories):
        x = np.mod(np.asarray(tr["x"], dtype=float), length)
        y = np.mod(np.asarray(tr["y"], dtype=float), length)
        color = "#d7263d" if bool(tr["attached"]) else palette[i % len(palette)]
        for xs, ys in _split_periodic_path(x, y, length):
            pts = list(zip(sx(xs).astype(float), sy(ys).astype(float)))
            lines.append(_polyline(pts, color, 1.9))
    lines.append(f'<text x="{ox}" y="{oy + length*scale + 28}" font-family="Arial" font-size="13">Dashed ring: near-surface interaction zone used to define interception.</text>')
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
