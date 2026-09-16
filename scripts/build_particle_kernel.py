#!/usr/bin/env python3
"""Build the compiled C particle-tracking kernel."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import build_kernel


def main() -> None:
    path = build_kernel(force=True)
    print(path)


if __name__ == "__main__":
    main()
