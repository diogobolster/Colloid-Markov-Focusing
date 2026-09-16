#!/usr/bin/env python3
"""Fast regression checks for the compiled particle kernel."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import build_kernel, simulate_cell_transitions_compiled  # noqa: E402
from colloid_tsm.physical import PhysicalParams, load_flow  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "production.json"


class RegressionFailure(AssertionError):
    """Raised when a regression check fails."""


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise RegressionFailure(message)


def assert_allclose(a: np.ndarray, b: np.ndarray, name: str, atol: float = 1.0e-14) -> None:
    ok = np.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True)
    if not ok:
        diff = np.nanmax(np.abs(a - b))
        raise RegressionFailure(f"{name} is not reproducible; max absolute difference {diff:g}")


def params_from_config(config: dict, *, max_time: float) -> PhysicalParams:
    tracking = config["particle_tracking"]
    return replace(
        PhysicalParams(),
        mean_velocity=4.0 / 86400.0,
        dt=float(tracking["dt_ms"]) * 1.0e-3,
        max_time=max_time,
        resolved_langevin_substeps=int(tracking["near_wall_substeps"]),
        resolved_langevin_substep_cutoff=float(tracking["near_wall_cutoff_nm"]) * 1.0e-9,
        resolved_langevin_normal_step=float(tracking["normal_step_target_nm"]) * 1.0e-9,
        resolved_langevin_min_dt=float(tracking["min_substep_us"]) * 1.0e-6,
    )


def load_production_flow(config: dict, params: PhysicalParams):
    data_root = Path(config.get("_data_root", ROOT))
    flow_path = data_root / config["flow"]["path"]
    if not flow_path.exists():
        raise FileNotFoundError(flow_path)
    return load_flow(flow_path, params=params)


def test_reproducible_neutral_run(config: dict) -> dict[str, float | int | str]:
    params = params_from_config(config, max_time=40.0)
    flow = load_production_flow(config, params)
    y0 = np.linspace(params.inlet_y_min + 20.0e-6, params.inlet_y_min + 70.0e-6, 96)
    kwargs = dict(
        condition="neutral",
        seed=4021,
        allow_attachment=False,
        initial_y=y0,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )
    lib1 = simulate_cell_transitions_compiled(flow, **kwargs)
    lib2 = simulate_cell_transitions_compiled(flow, **kwargs)
    assert_allclose(lib1.y_out, lib2.y_out, "neutral y_out")
    assert_allclose(lib1.travel_time, lib2.travel_time, "neutral travel_time")
    assert_true(int(np.sum(lib1.attached)) == 0, "neutral run produced attachments")
    assert_true(float(np.mean(lib1.exited)) >= 0.98, "neutral run did not mostly exit within 40 s")
    return {
        "test": "reproducible_neutral_run",
        "particles": int(y0.size),
        "exited_fraction": float(np.mean(lib1.exited)),
        "attached": int(np.sum(lib1.attached)),
        "status": "pass",
    }


def test_near_wall_adaptive_diagnostics(config: dict) -> dict[str, float | int | str]:
    params = params_from_config(config, max_time=0.05)
    params = replace(
        params,
        ionic_strength_molar=50.0e-3,
        zeta_collector_unfavorable=-50.0e-3,
    )
    flow = load_production_flow(config, params)
    center = 0.5 * params.cell_length
    radius = params.exclusion_radius + 10.0e-9
    theta = np.linspace(-0.35, 0.35, 24)
    x0 = center + radius * np.cos(theta)
    y0 = center + radius * np.sin(theta)
    lib, diag = simulate_cell_transitions_compiled(
        flow,
        "unfavorable",
        seed=9817,
        allow_attachment=False,
        initial_x=x0,
        initial_y=y0,
        surface_mode="resolved_langevin",
        force_rebuild=False,
        return_timestep_diagnostics=True,
    )
    assert_true(int(np.sum(lib.attached)) == 0, "unfavorable near-wall run attached particles")
    assert_true(int(np.min(diag["adaptive_substeps"])) > 0, "near-wall particles did not use adaptive substeps")
    assert_true(int(np.sum(diag["guard_hits"])) == 0, "near-wall adaptive loop hit max-substep guard")
    max_brownian_std_nm = float(np.max(diag["max_brownian_normal_std"]) * 1.0e9)
    max_det_step_nm = float(np.max(diag["max_inward_det_normal_step"]) * 1.0e9)
    assert_true(max_brownian_std_nm <= 1.6, f"Brownian normal std exceeded expected bound: {max_brownian_std_nm:g} nm")
    assert_true(max_det_step_nm <= 0.25, f"inward deterministic normal step exceeded expected bound: {max_det_step_nm:g} nm")
    return {
        "test": "near_wall_adaptive_diagnostics",
        "particles": int(y0.size),
        "min_adaptive_substeps": int(np.min(diag["adaptive_substeps"])),
        "max_adaptive_substeps": int(np.max(diag["adaptive_substeps"])),
        "guard_hits": int(np.sum(diag["guard_hits"])),
        "max_brownian_normal_std_nm": max_brownian_std_nm,
        "max_inward_det_normal_step_nm": max_det_step_nm,
        "status": "pass",
    }


def test_favorable_contact_attachment(config: dict) -> dict[str, float | int | str]:
    params = params_from_config(config, max_time=0.01)
    flow = load_production_flow(config, params)
    center = 0.5 * params.cell_length
    radius = params.exclusion_radius + 0.5 * params.contact_gap
    theta = np.linspace(-0.2, 0.2, 12)
    x0 = center + radius * np.cos(theta)
    y0 = center + radius * np.sin(theta)
    lib = simulate_cell_transitions_compiled(
        flow,
        "favorable",
        seed=773,
        allow_attachment=True,
        initial_x=x0,
        initial_y=y0,
        surface_mode="resolved_langevin",
        force_rebuild=False,
    )
    attached = int(np.sum(lib.attached))
    assert_true(attached == int(y0.size), f"favorable contact run attached {attached}/{y0.size} particles")
    return {
        "test": "favorable_contact_attachment",
        "particles": int(y0.size),
        "attached": attached,
        "status": "pass",
    }


def write_report(out_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel_regression_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Compiled kernel regression tests",
        "",
        "| test | status | details |",
        "|---|---|---|",
    ]
    for row in rows:
        details = ", ".join(f"{key}={value}" for key, value in row.items() if key not in {"test", "status"})
        lines.append(f"| {row['test']} | {row['status']} | {details} |")
    lines.append("")
    (out_dir / "kernel_regression_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root containing resolved flow outputs when they are stored outside the code checkout.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["_data_root"] = str(args.data_root.resolve())
    out_dir = ROOT / config["validation"]["regression_output"]

    build_kernel(force=True)
    tests: list[Callable[[dict], dict[str, float | int | str]]] = [
        test_reproducible_neutral_run,
        test_near_wall_adaptive_diagnostics,
        test_favorable_contact_attachment,
    ]
    rows: list[dict[str, float | int | str]] = []
    try:
        for test in tests:
            row = test(config)
            rows.append(row)
            print(f"PASS {row['test']}", flush=True)
    except Exception as exc:
        rows.append({"test": getattr(test, "__name__", "unknown"), "status": "fail", "error": str(exc)})
        write_report(out_dir, rows)
        print(f"FAIL {rows[-1]['test']}: {exc}", file=sys.stderr, flush=True)
        return 1
    write_report(out_dir, rows)
    print(out_dir / "kernel_regression_report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
