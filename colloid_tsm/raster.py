"""PNG plotting helpers for diagnostic figures."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .svg import PlotStyle


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int) -> None:
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")


def _split_periodic_path(x: np.ndarray, y: np.ndarray, length: float) -> list[tuple[np.ndarray, np.ndarray]]:
    if x.size <= 1:
        return [(x, y)]
    jumps = (np.abs(np.diff(x)) > 0.5 * length) | (np.abs(np.diff(y)) > 0.5 * length)
    starts = np.r_[0, np.flatnonzero(jumps) + 1]
    stops = np.r_[np.flatnonzero(jumps) + 1, x.size]
    return [(x[s:e], y[s:e]) for s, e in zip(starts, stops) if e - s > 1]


def write_retention_png(
    path: str,
    series: dict[str, tuple[np.ndarray, np.ndarray, PlotStyle]],
    title: str,
    width: int = 1720,
    height: int = 1120,
) -> None:
    scale = 2
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    margin_l, margin_r, margin_t, margin_b = 150, 50, 110, 140
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    x_max = max(float(np.nanmax(x)) for x, _, _ in series.values())
    positive = np.concatenate([y[y > 0] for _, y, _ in series.values()])
    y_min = max(float(np.nanmin(positive)), 1.0e-5)
    y_max = max(float(np.nanmax(positive)), y_min * 10.0)
    y_min = 10.0 ** np.floor(np.log10(y_min))
    y_max = 10.0 ** np.ceil(np.log10(y_max))

    def sx(vals):
        return margin_l + np.asarray(vals) / x_max * plot_w

    def sy(vals):
        ly = np.log10(np.maximum(np.asarray(vals), y_min))
        return margin_t + (np.log10(y_max) - ly) / (np.log10(y_max) - np.log10(y_min)) * plot_h

    title_font = _font(38)
    label_font = _font(28)
    tick_font = _font(24)
    legend_font = _font(24)
    draw.text((width / 2, 42), title, fill="#111111", font=title_font, anchor="mm")
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill="#222222", width=2 * scale)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill="#222222", width=2 * scale)

    for tick in np.linspace(0, x_max, 6):
        xpx = float(sx(tick))
        draw.line((xpx, margin_t + plot_h, xpx, margin_t + plot_h + 12), fill="#222222", width=2)
        draw.text((xpx, margin_t + plot_h + 42), f"{tick:.1f}", fill="#222222", font=tick_font, anchor="mm")

    for exp in range(int(np.log10(y_min)), int(np.log10(y_max)) + 1):
        val = 10.0 ** exp
        ypx = float(sy(val))
        draw.line((margin_l - 12, ypx, margin_l, ypx), fill="#222222", width=2)
        draw.line((margin_l, ypx, margin_l + plot_w, ypx), fill="#e5e5e5", width=1)
        draw.text((margin_l - 20, ypx), f"1e{exp}", fill="#222222", font=tick_font, anchor="rm")

    for _, (x, y, style) in series.items():
        mask = y > 0
        pts = list(zip(sx(x[mask]).astype(float), sy(y[mask]).astype(float)))
        _draw_polyline(draw, pts, style.color, 5)

    legend_x = margin_l + plot_w - 470
    legend_y = margin_t + 35
    draw.rectangle(
        (legend_x - 24, legend_y - 32, legend_x + 450, legend_y + 52 * len(series) + 8),
        fill="white",
        outline="#cccccc",
        width=2,
    )
    for i, (_, (_, _, style)) in enumerate(series.items()):
        y0 = legend_y + 52 * i
        draw.line((legend_x, y0, legend_x + 68, y0), fill=style.color, width=7)
        draw.text((legend_x + 88, y0), style.label, fill="#111111", font=legend_font, anchor="lm")

    draw.text((margin_l + plot_w / 2, height - 42), "transport distance, x/L", fill="#111111", font=label_font, anchor="mm")
    y_label = Image.new("RGBA", (520, 60), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((260, 30), "retained fraction per bin", fill="#111111", font=label_font, anchor="mm")
    y_label = y_label.rotate(90, expand=True)
    img.paste(y_label.convert("RGB"), (20, margin_t + plot_h // 2 - y_label.height // 2))
    img = img.resize((width // 2, height // 2), Image.Resampling.LANCZOS)
    img.save(path)


def write_geometry_png(
    path: str,
    trajectories: list[dict[str, np.ndarray | bool]],
    length: float,
    grain_radius: float,
    zoi: float,
    title: str = "Single-grain periodic training trajectories",
    width: int = 1520,
    height: int = 1040,
) -> None:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(38)
    text_font = _font(24)
    margin_l, margin_t = 110, 96
    plot_side = min(width - 2 * margin_l, height - 2 * margin_t - 70)
    scale = plot_side / length
    ox = margin_l + (width - 2 * margin_l - plot_side) / 2
    oy = margin_t

    def sx(vals):
        return ox + np.asarray(vals) * scale

    def sy(vals):
        return oy + (length - np.asarray(vals)) * scale

    draw.text((width / 2, 44), title, fill="#111111", font=title_font, anchor="mm")
    draw.rectangle((ox, oy, ox + plot_side, oy + plot_side), fill="#f7fbff", outline="#222222", width=3)
    cx = float(sx(0.5 * length))
    cy = float(sy(0.5 * length))
    r = grain_radius * scale
    rz = (grain_radius + zoi) * scale
    draw.ellipse((cx - rz, cy - rz, cx + rz, cy + rz), fill="#d7ecff", outline="#69aeea", width=3)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#111111")

    palette = ["#2364aa", "#3da35d", "#f18f01", "#c73e1d", "#6f42c1", "#0f8b8d"]
    for i, tr in enumerate(trajectories):
        x = np.mod(np.asarray(tr["x"], dtype=float), length)
        y = np.mod(np.asarray(tr["y"], dtype=float), length)
        color = "#d7263d" if bool(tr["attached"]) else palette[i % len(palette)]
        for xs, ys in _split_periodic_path(x, y, length):
            pts = list(zip(sx(xs).astype(float), sy(ys).astype(float)))
            _draw_polyline(draw, pts, color, 4)

    draw.text(
        (ox, oy + plot_side + 40),
        "Pale ring: near-surface interaction zone used to define interception.",
        fill="#111111",
        font=text_font,
        anchor="lm",
    )
    img = img.resize((width // 2, height // 2), Image.Resampling.LANCZOS)
    img.save(path)
