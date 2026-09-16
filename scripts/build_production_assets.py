#!/usr/bin/env python3
"""Refresh production figures/reports without compiling TeX."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "production.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/codex_mpl_cache")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp/codex_xdg_cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=str(ROOT), env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    production = config["production_run"]
    full_suite = config.get("full_condition_suite", {})
    tracking = config["particle_tracking"]
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_openfoam_focusing_cases.py"),
            "--dt-ms",
            str(tracking["dt_ms"]),
            "--out-name",
            str(production["out_name"]),
            "--figure-prefix",
            str(production["figure_prefix"]),
            "--reuse",
        ]
    )
    run([sys.executable, str(ROOT / "scripts" / "run_openfoam_convergence_validation.py"), "--reuse"])
    run([sys.executable, str(ROOT / "scripts" / "run_timestep_audit.py"), "--reuse"])
    if full_suite:
        suite_args = [
            sys.executable,
            str(ROOT / "scripts" / "run_openfoam_full_condition_suite.py"),
            "--dt-ms",
            str(tracking["dt_ms"]),
            "--out-name",
            str(full_suite["out_name"]),
            "--figure-prefix",
            str(full_suite["figure_prefix"]),
            "--chunk-size",
            str(full_suite.get("chunk_size", 6000)),
            "--workers",
            str(full_suite.get("workers", 4)),
            "--reuse",
        ]
        if full_suite.get("profiles"):
            suite_args.append("--profiles")
            suite_args.extend(str(profile) for profile in full_suite["profiles"])
        run(suite_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
