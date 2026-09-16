#!/usr/bin/env python3
"""Assemble the paired entry-side diagnostic figure (main-text Figure 14 / trimmed Figure 10) from the two
confirmatory analysis panels written by scripts/analyze_entry_ffsz_study.py.

    python scripts/create_entry_side_figure.py [--stage confirmatory]

Reads  outputs/entry_ffsz/<stage>/analysis/entry_ffsz_geometry_summary.png        (panel a: matched direct effect
                                                                                    per domain and per geometry)
       outputs/entry_ffsz/<stage>/analysis/entry_ffsz_selection_decomposition.png  (panel b: population change =
                                                                                    matched direct + completion selection)
Writes outputs/figures/figure_14_entry_side.png

The two panels are placed side by side at a common height with (a)/(b) labels. The submitted figure_14.png was
assembled from the same two panels; only the whitespace and label placement are set here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="confirmatory")
    ap.add_argument("--height", type=int, default=1090, help="common panel height in pixels")
    args = ap.parse_args()

    src = ROOT / "outputs" / "entry_ffsz" / args.stage / "analysis"
    panels = [src / "entry_ffsz_geometry_summary.png", src / "entry_ffsz_selection_decomposition.png"]
    for p in panels:
        if not p.exists():
            raise FileNotFoundError(f"{p} (run scripts/analyze_entry_ffsz_study.py --stage {args.stage} first)")

    images = []
    for p in panels:
        im = Image.open(p).convert("RGB")
        scale = args.height / im.height
        images.append(im.resize((round(im.width * scale), args.height), Image.LANCZOS))

    gap, margin, label_h = 60, 30, 70
    width = sum(im.width for im in images) + gap + 2 * margin
    sheet = Image.new("RGB", (width, args.height + label_h + margin), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
    except OSError:
        font = ImageFont.load_default()
    x = margin
    for label, im in zip("ab", images):
        sheet.paste(im, (x, label_h))
        draw.text((x, 8), label, fill="black", font=font)
        x += im.width + gap

    out = ROOT / "outputs" / "figures" / "figure_14_entry_side.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, dpi=(200, 200))
    print(f"wrote {out} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
