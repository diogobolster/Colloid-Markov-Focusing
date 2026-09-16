#!/usr/bin/env python3
"""Draw lightweight PNG diagnostics for the physical-unit pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "physical"


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def ramp(values: np.ndarray, *, low: tuple[int, int, int] = (248, 250, 252)) -> np.ndarray:
    stops = np.array(
        [
            low,
            (147, 197, 253),
            (45, 212, 191),
            (132, 204, 22),
            (250, 204, 21),
            (220, 38, 38),
        ],
        dtype=float,
    )
    v = np.clip(values, 0.0, 1.0)
    scaled = v * (len(stops) - 1)
    i = np.floor(scaled).astype(int)
    i = np.minimum(i, len(stops) - 2)
    f = (scaled - i)[..., None]
    return ((1.0 - f) * stops[i] + f * stops[i + 1]).astype(np.uint8)


def data_to_pixel(x: float, y: float, length: float, ox: int, oy: int, side: int) -> tuple[float, float]:
    return ox + x / length * side, oy + (1.0 - y / length) * side


def draw_arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, dx: float, dy: float, color: str) -> None:
    x1 = x0 + dx
    y1 = y0 - dy
    draw.line((x0, y0, x1, y1), fill=color, width=2)
    angle = np.arctan2(y1 - y0, x1 - x0)
    head = 7.0
    for offset in (0.55, -0.55):
        a = angle + np.pi + offset
        draw.line((x1, y1, x1 + head * np.cos(a), y1 + head * np.sin(a)), fill=color, width=2)


def write_flow(path: Path) -> None:
    flow = np.load(OUT / "flow_N192_4m_per_day.npz")
    ux = flow["ux"]
    uy = flow["uy"]
    solid = flow["solid"]
    speed = np.sqrt(ux * ux + uy * uy)
    length = 4.0e-4
    grain_radius = 1.0e-4
    n = speed.shape[0]
    norm = speed / np.nanmax(speed[~solid])
    rgb = ramp(norm, low=(239, 246, 255))
    rgb[solid] = (8, 8, 8)
    display = np.flip(np.transpose(rgb, (1, 0, 2)), axis=0)

    width, height = 1160, 980
    side = 760
    ox, oy = 130, 104
    img = Image.new("RGB", (width, height), "white")
    heat = Image.fromarray(display, "RGB").resize((side, side), Image.Resampling.BILINEAR)
    img.paste(heat, (ox, oy))
    draw = ImageDraw.Draw(img)
    draw.rectangle((ox, oy, ox + side, oy + side), outline="#111111", width=3)
    draw.text((width / 2, 46), "Resolved no-slip LBM flow, 4 m/day", fill="#111111", font=font(30), anchor="mm")
    draw.text((width / 2, height - 38), "x", fill="#111111", font=font(26), anchor="mm")
    draw.text((44, oy + side / 2), "y", fill="#111111", font=font(26), anchor="mm")

    rr = grain_radius / length * side
    draw.text(
        (ox + side / 2, oy + side + 34),
        "solid mask: one full center disk plus four periodic quarter disks",
        fill="#111111",
        font=font(20),
        anchor="mm",
    )

    stride = 16
    max_speed = float(np.max(speed[~solid]))
    for i in range(stride // 2, n, stride):
        for j in range(stride // 2, n, stride):
            if solid[i, j]:
                continue
            x = (i + 0.5) / n * length
            y = (j + 0.5) / n * length
            px, py = data_to_pixel(x, y, length, ox, oy, side)
            scale = 0.50 * side / n
            draw_arrow(draw, px, py, ux[i, j] / max_speed * scale, uy[i, j] / max_speed * scale, "#111827")

    cb_x, cb_y, cb_w, cb_h = ox + side + 58, oy + 80, 34, side - 160
    vals = np.linspace(1.0, 0.0, cb_h)[:, None]
    cb = Image.fromarray(ramp(vals).reshape(cb_h, 1, 3), "RGB").resize((cb_w, cb_h))
    img.paste(cb, (cb_x, cb_y))
    draw.rectangle((cb_x, cb_y, cb_x + cb_w, cb_y + cb_h), outline="#111111", width=2)
    draw.text((cb_x + cb_w + 14, cb_y), f"{max_speed:.2e}", fill="#111111", font=font(20), anchor="lm")
    draw.text((cb_x + cb_w + 14, cb_y + cb_h), "0", fill="#111111", font=font(20), anchor="lm")
    draw.text((cb_x + cb_w + 54, cb_y + cb_h / 2), "speed (m/s)", fill="#111111", font=font(20), anchor="mm")
    img.save(path)


def write_transition(path: Path, condition: str, velocity: int = 4) -> None:
    matrix = np.loadtxt(OUT / f"transition_probabilities_{condition}_{velocity}m_per_day.csv", delimiter=",")
    finite = np.isfinite(matrix)
    scaled = np.zeros_like(matrix, dtype=float)
    if np.any(finite):
        scaled[finite] = matrix[finite] / max(float(np.max(matrix[finite])), 1.0e-12)
    rgb = ramp(scaled, low=(255, 255, 255))
    rgb[~finite] = (229, 231, 235)
    display = np.flip(rgb, axis=0)

    width, height = 920, 860
    side = 650
    ox, oy = 120, 96
    img = Image.new("RGB", (width, height), "white")
    heat = Image.fromarray(display, "RGB").resize((side, side), Image.Resampling.NEAREST)
    img.paste(heat, (ox, oy))
    draw = ImageDraw.Draw(img)
    draw.rectangle((ox, oy, ox + side, oy + side), outline="#111111", width=3)
    title = f"{condition.capitalize()} transition probabilities, {velocity} m/day"
    draw.text((width / 2, 46), title, fill="#111111", font=font(32), anchor="mm")
    draw.text((ox + side / 2, height - 48), "outlet bin", fill="#111111", font=font(24), anchor="mm")
    y_label = Image.new("RGBA", (190, 44), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((95, 22), "inlet bin", fill="#111111", font=font(24), anchor="mm")
    y_label = y_label.rotate(90, expand=True)
    img.paste(y_label.convert("RGB"), (8, oy + side // 2 - y_label.height // 2))
    for tick in (0, 8, 16, 24, 32):
        x = ox + tick / 32 * side
        y = oy + side - tick / 32 * side
        draw.line((x, oy + side, x, oy + side + 8), fill="#111111", width=2)
        draw.line((ox - 8, y, ox, y), fill="#111111", width=2)
        draw.text((x, oy + side + 28), str(tick), fill="#111111", font=font(18), anchor="mm")
        draw.text((ox - 22, y), str(tick), fill="#111111", font=font(18), anchor="rm")

    cb_x, cb_y, cb_w, cb_h = ox + side + 46, oy + 80, 30, side - 160
    vals = np.linspace(1.0, 0.0, cb_h)[:, None]
    cb = Image.fromarray(ramp(vals, low=(255, 255, 255)).reshape(cb_h, 1, 3), "RGB").resize((cb_w, cb_h))
    img.paste(cb, (cb_x, cb_y))
    draw.rectangle((cb_x, cb_y, cb_x + cb_w, cb_y + cb_h), outline="#111111", width=2)
    draw.text((cb_x + cb_w + 14, cb_y), f"{np.max(matrix):.2f}", fill="#111111", font=font(18), anchor="lm")
    draw.text((cb_x + cb_w + 14, cb_y + cb_h), "0", fill="#111111", font=font(18), anchor="lm")
    img.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_flow(OUT / "flow_field_4m_per_day.png")
    for condition in ("favorable", "unfavorable"):
        write_transition(OUT / f"transition_probabilities_{condition}_4m_per_day.png", condition)
    print("Wrote physical diagnostic figures to", OUT)


if __name__ == "__main__":
    main()
