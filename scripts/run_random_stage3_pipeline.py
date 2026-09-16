#!/usr/bin/env python3
"""Config-driven production pipeline for the random-packing focusing extension."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "random_stage3.json"


def rel(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(cmd: list[str], dry_run: bool, cwd: Path = ROOT) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def geometry_path(candidate: str) -> str:
    return f"outputs/stage1_geometry_screen/geometries/{candidate}/geometry.json"


def flow_case(candidate: str) -> str:
    return f"outputs/stage1_openfoam_flow/{candidate}_gmsh_screen"


def command_production(config: dict, dry_run: bool) -> None:
    production = config["production"]
    cmd = [
        sys.executable,
        "scripts/run_stage2_particle_pilots.py",
        *production["candidates"],
        "--out-root",
        config["out_root"],
        "--particles",
        str(production["particles"]),
        "--chunk-size",
        str(production["chunk_size"]),
        "--max-time",
        str(production["max_time_s"]),
        "--dt",
        str(production["dt_s"]),
        "--matrix-bins",
        str(production["matrix_bins"]),
        "--grid-nx",
        str(production["grid_nx"]),
        "--grid-ny",
        str(production["grid_ny"]),
        "--profiles",
        *production["profiles"],
    ]
    run(cmd, dry_run)


def command_postprocess(config: dict, dry_run: bool) -> None:
    out_root = Path(config["out_root"])
    for candidate in config["production"]["candidates"]:
        cmd = [
            sys.executable,
            "scripts/analyze_random_grain_focusing.py",
            rel(out_root / candidate / "random_condition_screen_payload.npz"),
            "--out-dir",
            rel(out_root / candidate / "grain_focusing"),
            "--geometry-path",
            geometry_path(candidate),
            "--flow-case",
            flow_case(candidate),
        ]
        run(cmd, dry_run)
    run([sys.executable, "scripts/aggregate_stage3_random_production.py"], dry_run)
    run([sys.executable, "scripts/create_stage3_random_mechanism_figure.py"], dry_run)


def command_shadow(config: dict, dry_run: bool) -> None:
    shadow = config["shadow"]
    candidate = shadow["candidate"]
    cmd = [
        sys.executable,
        "scripts/run_transport_shadow_diagnostic.py",
        "--geometry-path",
        geometry_path(candidate),
        "--flow-case",
        flow_case(candidate),
        "--out-dir",
        shadow["out_dir"],
        "--particles",
        str(shadow["particles"]),
        "--chunk-size",
        str(shadow["chunk_size"]),
        "--max-time",
        str(shadow["max_time_s"]),
        "--dt",
        str(shadow["dt_s"]),
        "--pore-nx",
        str(shadow["pore_nx"]),
        "--pore-ny",
        str(shadow["pore_ny"]),
        "--surface-bins",
        str(shadow["surface_bins"]),
        "--sample-stride",
        str(shadow["sample_stride"]),
        "--profiles",
        *shadow["profiles"],
        "--resume",
    ]
    run(cmd, dry_run)


def command_validate(config: dict) -> None:
    required: list[Path] = []
    out_root = ROOT / config["out_root"]
    for candidate in config["production"]["candidates"]:
        required.extend(
            [
                out_root / candidate / "random_condition_screen_payload.npz",
                out_root / candidate / "random_condition_screen_summary.csv",
                out_root / candidate / "grain_focusing" / "grain_local_focusing.csv",
            ]
        )
    required.extend(
            [
                out_root / "stage3_random_focusing_summary.csv",
                ROOT / "outputs" / "figures" / "random_stage3_focusing_summary.png",
                ROOT / "outputs" / "figures" / "random_geometry_mechanism_streamlines.png",
            ]
        )
    shadow = config.get("shadow", {})
    if shadow:
        shadow_out = ROOT / shadow["out_dir"]
        required.extend(
            [
                shadow_out / "transport_shadow_condition_summary.csv",
                shadow_out / "transport_shadow_metrics.csv",
                shadow_out / "transport_shadow_diagnostic_report.md",
                ROOT / "outputs" / "figures" / "transport_shadow_diagnostic.png",
            ]
        )
    missing = [path for path in required if not path.exists()]
    report = ROOT / config["out_root"] / "random_stage3_validation_report.md"
    lines = ["# Random Stage-3 Validation", ""]
    if missing:
        lines.append(f"Missing required artifacts: {len(missing)}")
        lines.extend(f"- `{rel(path)}`" for path in missing)
    else:
        lines.append("All required stage-3 production, focusing, shadow, and diagnostic artifacts are present.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    if missing:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["production", "postprocess", "shadow", "validate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.command == "production":
        command_production(config, args.dry_run)
    elif args.command == "postprocess":
        command_postprocess(config, args.dry_run)
    elif args.command == "shadow":
        command_shadow(config, args.dry_run)
    elif args.command == "validate":
        command_validate(config)


if __name__ == "__main__":
    main()
