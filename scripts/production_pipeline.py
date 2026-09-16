#!/usr/bin/env python3
"""Single-entry production workflow for the periodic-cell DLVO focusing study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "production.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run_command(args: list[str], *, cwd: Path = ROOT, cache_env: bool = True) -> None:
    print("+ " + " ".join(args), flush=True)
    env = os.environ.copy()
    if cache_env:
        env.setdefault("MPLCONFIGDIR", "/private/tmp/codex_mpl_cache")
        env.setdefault("XDG_CACHE_HOME", "/private/tmp/codex_xdg_cache")
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(args, cwd=str(cwd), check=True, env=env)


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, str(ROOT / script), *args]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if not np.isfinite(observed) or abs(observed - expected) > tolerance:
        raise AssertionError(f"{name}: observed {observed:g}, expected {expected:g} +/- {tolerance:g}")


def assert_le(name: str, observed: float, maximum: float) -> None:
    if not np.isfinite(observed) or observed > maximum:
        raise AssertionError(f"{name}: observed {observed:g}, maximum {maximum:g}")


def assert_ge(name: str, observed: float, minimum: float) -> None:
    if not np.isfinite(observed) or observed < minimum:
        raise AssertionError(f"{name}: observed {observed:g}, minimum {minimum:g}")


def command_run(config: dict, reuse: bool) -> None:
    require_file(rel(config["flow"]["path"]))
    run_cfg = config["production_run"]
    dt_ms = str(config["particle_tracking"]["dt_ms"])
    focusing_args = [
        "--dt-ms",
        dt_ms,
        "--out-name",
        str(run_cfg["out_name"]),
        "--figure-prefix",
        str(run_cfg["figure_prefix"]),
    ]
    if reuse:
        focusing_args.append("--reuse")
    run_command(python_cmd("scripts/run_openfoam_focusing_cases.py", *focusing_args))

    audit_args = ["--reuse"] if reuse else []
    run_command(python_cmd("scripts/run_timestep_audit.py", *audit_args))


def command_run_full(config: dict, reuse: bool) -> None:
    require_file(rel(config["flow"]["path"]))
    suite_cfg = config["full_condition_suite"]
    dt_ms = str(config["particle_tracking"]["dt_ms"])
    suite_args = [
        "--dt-ms",
        dt_ms,
        "--out-name",
        str(suite_cfg["out_name"]),
        "--figure-prefix",
        str(suite_cfg["figure_prefix"]),
        "--chunk-size",
        str(suite_cfg.get("chunk_size", 6000)),
        "--workers",
        str(suite_cfg.get("workers", 4)),
    ]
    profiles = suite_cfg.get("profiles", [])
    if profiles:
        suite_args.append("--profiles")
        suite_args.extend(str(profile) for profile in profiles)
    if reuse:
        suite_args.append("--reuse")
    run_command(python_cmd("scripts/run_openfoam_full_condition_suite.py", *suite_args))


def validate_refinement(config: dict) -> list[str]:
    validation = config["validation"]
    rows = read_csv(rel(validation["focusing_refinement_csv"]))
    by_profile = {row["profile"]: row for row in rows}
    messages: list[str] = []
    for profile, expected in validation["expected_refinement"].items():
        row = by_profile[profile]
        f30 = as_float(row, "center_well_release_theta30_fraction")
        angle = as_float(row, "center_well_release_median_theta_deg")
        releases = as_float(row, "center_well_release_count")
        unresolved = as_float(row, "center_well_censored_count")
        assert_close(f"refinement {profile} F30", f30, expected["f30"], expected["f30_abs_tolerance"])
        assert_close(
            f"refinement {profile} median angle",
            angle,
            expected["median_angle_deg"],
            expected["angle_abs_tolerance_deg"],
        )
        assert_ge(f"refinement {profile} releases", releases, expected["min_releases"])
        assert_le(f"refinement {profile} unresolved", unresolved, expected["max_unresolved"])
        messages.append(f"refinement {profile}: F30={f30:.3f}, releases={int(releases)}, unresolved={int(unresolved)}")
    return messages


def validate_transition_rows(config: dict) -> list[str]:
    validation = config["validation"]
    rows = read_csv(rel(validation["focusing_transition_csv"]))
    by_profile = {row["profile"]: row for row in rows if row["state"] == "center core"}
    messages: list[str] = []
    for profile, expected in validation["expected_transition_center_core"].items():
        row = by_profile[profile]
        f30 = as_float(row, "center_well_theta30_fraction")
        releases = as_float(row, "center_well_releases")
        unresolved = as_float(row, "center_well_censored")
        assert_close(f"transition {profile} F30", f30, expected["f30"], expected["f30_abs_tolerance"])
        assert_ge(f"transition {profile} releases", releases, expected["min_releases"])
        assert_le(f"transition {profile} unresolved", unresolved, expected["max_unresolved"])
        messages.append(f"transition {profile}: F30={f30:.3f}, releases={int(releases)}, unresolved={int(unresolved)}")
    return messages


def validate_timestep_audit(config: dict) -> list[str]:
    validation = config["validation"]
    rows = read_csv(rel(validation["timestep_audit_csv"]))
    expected = validation["expected_timestep_audit"]
    row = next(item for item in rows if item["group"] == expected["group"])
    particles = as_float(row, "particles")
    median_substeps = as_float(row, "adaptive_substeps_median")
    guard_hits = as_float(row, "total_guard_hits")
    min_dt_hits = as_float(row, "total_min_dt_hits")
    total_substeps = as_float(row, "total_adaptive_substeps")
    min_dt_fraction = min_dt_hits / total_substeps if total_substeps > 0.0 else float("inf")
    det_step = as_float(row, "max_inward_det_normal_step_nm")
    brownian = as_float(row, "max_brownian_normal_std_nm")
    assert_ge("timestep audit particles", particles, expected["min_particles"])
    assert_ge("timestep audit median adaptive substeps", median_substeps, expected["min_median_adaptive_substeps"])
    assert_le("timestep audit guard hits", guard_hits, expected["max_guard_hits"])
    assert_le("timestep audit min-dt hit fraction", min_dt_fraction, expected["max_min_dt_hit_fraction"])
    assert_le("timestep audit max inward deterministic step", det_step, expected["max_inward_det_normal_step_nm"])
    assert_le("timestep audit max Brownian normal std", brownian, expected["max_brownian_normal_std_nm"])
    return [
        f"timestep audit {expected['group']}: median_substeps={median_substeps:.3g}, "
        f"guard_hits={int(guard_hits)}, min_dt_fraction={min_dt_fraction:.3g}"
    ]


def validate_convergence(config: dict) -> list[str]:
    validation = config["validation"]
    rows = read_csv(rel(validation["convergence_csv"]))
    expected = validation["expected_convergence"]
    deltas = [abs(as_float(row, "delta_f30_vs_baseline")) for row in rows if row.get("delta_f30_vs_baseline", "")]
    unresolved = [as_float(row, "unresolved") for row in rows]
    max_delta = max(deltas)
    max_unresolved = max(unresolved)
    assert_le("convergence max F30 delta", max_delta, expected["max_abs_f30_delta_vs_baseline"])
    assert_le("convergence unresolved", max_unresolved, expected["max_unresolved"])
    return [f"convergence: max_abs_delta={max_delta:.3f}, max_unresolved={int(max_unresolved)}"]


def validate_full_suite(config: dict) -> list[str]:
    if "full_condition_suite" not in config:
        return []
    suite = config["full_condition_suite"]
    profiles = [str(profile) for profile in suite.get("profiles", [])]
    out_dir = ROOT / "outputs" / str(suite["out_name"])
    refinement_csv = out_dir / "openfoam_full_refinement_summary.csv"
    transition_csv = out_dir / "openfoam_full_transition_rows.csv"
    report = out_dir / "openfoam_full_suite_report.md"
    summary_figure = ROOT / "outputs" / "figures" / f"{suite['figure_prefix']}_summary.png"
    transition_figure = ROOT / "outputs" / "figures" / f"{suite['figure_prefix']}_transition_center_rows.png"
    for path in (refinement_csv, transition_csv, report, summary_figure, transition_figure):
        require_file(path)
    refinement = read_csv(refinement_csv)
    transition = read_csv(transition_csv)
    if len(refinement) != len(profiles):
        raise AssertionError(f"full suite refinement rows: observed {len(refinement)}, expected {len(profiles)}")
    if len(transition) != len(profiles) * 7:
        raise AssertionError(f"full suite transition rows: observed {len(transition)}, expected {len(profiles) * 7}")
    by_profile = {row["profile"]: row for row in refinement}
    center_transition = {row["profile"]: row for row in transition if row["state"] == "center core"}
    for profile in profiles:
        if profile not in by_profile:
            raise AssertionError(f"full suite missing refinement profile {profile}")
        if profile not in center_transition:
            raise AssertionError(f"full suite missing transition center row {profile}")
    assert_ge(
        "full suite 50mM -50mV refinement F30",
        as_float(by_profile["unfavorable_50mM_z50"], "center_well_release_theta30_fraction"),
        0.62,
    )
    assert_le(
        "full suite 50mM -50mV unresolved",
        as_float(by_profile["unfavorable_50mM_z50"], "center_well_censored_count"),
        0,
    )
    assert_ge(
        "full suite 100mM -70mV refinement F30",
        as_float(by_profile["unfavorable_100mM_z70"], "center_well_release_theta30_fraction"),
        0.90,
    )
    assert_le(
        "full suite 100xD unfavorable F30",
        as_float(by_profile["unfavorable_50mM_z70_100xD"], "center_well_release_theta30_fraction"),
        0.25,
    )
    return [
        f"full suite: refinement_rows={len(refinement)}, transition_rows={len(transition)}, "
        f"50mM_z50_F30={as_float(by_profile['unfavorable_50mM_z50'], 'center_well_release_theta30_fraction'):.3f}"
    ]


def write_validation_report(config: dict, config_path: Path, messages: Iterable[str]) -> Path:
    out_dir = rel(config["validation"]["regression_output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "production_validation_report.md"
    lines = [
        "# Production validation report",
        "",
        f"Configuration: `{config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path}`",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for message in messages:
        lines.append(f"| {message.split(':', 1)[0]} | {message} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def command_validate(config: dict, config_path: Path) -> None:
    run_command(python_cmd("scripts/run_openfoam_convergence_validation.py", "--reuse"))
    run_command(python_cmd("scripts/run_timestep_audit.py", "--reuse"))
    run_command(python_cmd("scripts/kernel_regression_tests.py", "--config", str(config_path)))
    messages: list[str] = []
    messages.extend(validate_refinement(config))
    messages.extend(validate_transition_rows(config))
    messages.extend(validate_timestep_audit(config))
    messages.extend(validate_convergence(config))
    messages.extend(validate_full_suite(config))
    report = write_validation_report(config, config_path, messages)
    for message in messages:
        print(f"PASS {message}", flush=True)
    print(report, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "run-full", "validate"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse", action="store_true", help="Reuse existing trajectory libraries where supported.")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    try:
        if args.command == "run":
            command_run(config, reuse=args.reuse)
        elif args.command == "run-full":
            command_run_full(config, reuse=args.reuse)
        elif args.command == "validate":
            command_validate(config, config_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
