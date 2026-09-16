#!/usr/bin/env python3
"""Resolved-flow timestep and raster sensitivity for entry-FFSZ metrics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import simulate_cell_transitions_compiled, simulate_random_transitions_compiled  # noqa: E402
from colloid_tsm.physical import load_flow  # noqa: E402


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = import_script("entry_ffsz_runner_for_sensitivity", ROOT / "scripts" / "run_entry_ffsz_study.py")
analysis = import_script("entry_ffsz_analysis_for_sensitivity", ROOT / "scripts" / "analyze_entry_ffsz_study.py")
DEFAULT_CONFIG = ROOT / "config" / "entry_ffsz.json"


def diagnostic_data(library, diagnostics: dict[str, np.ndarray], refs: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    pair = diagnostics["pair_completed"].astype(bool)
    out = {
        "delta_x": analysis.nearest_delta(
            diagnostics["first_entry_theta"][pair],
            diagnostics["first_entry_collector"][pair].astype(int),
            refs,
        ),
        "delta_y": analysis.nearest_delta(
            diagnostics["next_entry_theta"][pair],
            diagnostics["next_entry_collector"][pair].astype(int),
            refs,
        ),
        "next_entry_time": diagnostics["next_entry_time"][pair],
        "first_entry_time": diagnostics["first_entry_time"][pair],
        "first_collector_near_time": diagnostics["first_collector_near_time"][pair],
        "all_pair_completed": pair,
        "all_first_entry": diagnostics["first_entry_collector"] >= 0,
        "all_attached": np.asarray(library.attached, dtype=bool),
        "all_censored": np.asarray(library.censored, dtype=bool),
    }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--periodic-particles", type=int, default=4000)
    parser.add_argument("--random-particles", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026081901)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    out_dir = output / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    window = np.deg2rad(float(config["analysis"]["ffsz_window_degrees"]))
    rows: list[dict] = []

    periodic_refs, _ = analysis.periodic_stagnation(config)
    periodic_injection = np.load(
        output / "confirmatory" / "periodic" / f"seed_{args.seed}" / "injection.npz"
    )
    periodic_x = periodic_injection["initial_x"][: args.periodic_particles]
    periodic_y = periodic_injection["initial_y"][: args.periodic_particles]
    for dt_scale in (0.5, 1.0, 2.0):
        for profile_name in ("no_dlvo", "unfavorable_50mM"):
            profile = config["profiles"][profile_name]
            params = runner.physical_params(
                config,
                profile,
                domain="periodic",
                max_time=float(config["confirmatory"]["periodic_max_time_s"]),
            )
            params = replace(params, dt=params.dt * dt_scale)
            flow = load_flow(
                runner.resolve_data_path(data_root, str(config["periodic"]["flow_path"])),
                params=params,
            )
            library, diagnostics = simulate_cell_transitions_compiled(
                flow,
                str(profile["condition"]),
                seed=args.seed,
                allow_attachment=bool(profile["allow_attachment"]),
                initial_x=periodic_x,
                initial_y=periodic_y,
                stop_at_next_distinct=True,
                return_entry_diagnostics=True,
            )
            data = diagnostic_data(library, diagnostics, periodic_refs)
            rows.append({
                "domain": "periodic",
                "geometry": "center_corner_cell",
                "sensitivity": "timestep",
                "setting": f"dt_scale_{dt_scale:g}",
                "dt_s": params.dt,
                "raster_nx": 512,
                "raster_ny": 512,
                "profile": profile_name,
                **analysis.metrics(data, window),
            })
            print(f"finished periodic {profile_name} dt_scale={dt_scale:g}", flush=True)

    entry = next(item for item in config["random_geometries"] if item["name"] == "through_random_00")
    geometry, interpolator, base_grid, _ = runner.load_random_resolved_flow(
        entry,
        config,
        data_root,
        output / "raster_cache",
    )
    random_refs, _ = analysis.random_stagnation(entry, config, output)
    random_injection = np.load(
        output / "confirmatory" / "random" / "through_random_00" / f"seed_{args.seed}" / "injection.npz"
    )
    random_x = random_injection["initial_x"][: args.random_particles]
    random_y = random_injection["initial_y"][: args.random_particles]

    for dt_scale in (0.5, 1.0, 2.0):
        for profile_name in ("no_dlvo", "unfavorable_50mM"):
            profile = config["profiles"][profile_name]
            params = runner.physical_params(
                config,
                profile,
                domain="random",
                max_time=float(config["confirmatory"]["random_max_time_s"]),
            )
            params = replace(params, dt=params.dt * dt_scale)
            library, diagnostics = simulate_random_transitions_compiled(
                base_grid,
                geometry,
                params,
                str(profile["condition"]),
                seed=args.seed + 10_000_019,
                allow_attachment=bool(profile["allow_attachment"]),
                initial_x=random_x,
                initial_y=random_y,
                stop_at_next_distinct=True,
                return_entry_diagnostics=True,
            )
            data = diagnostic_data(library, diagnostics, random_refs)
            rows.append({
                "domain": "random",
                "geometry": "through_random_00",
                "sensitivity": "timestep",
                "setting": f"dt_scale_{dt_scale:g}",
                "dt_s": params.dt,
                "raster_nx": base_grid.nx,
                "raster_ny": base_grid.ny,
                "profile": profile_name,
                **analysis.metrics(data, window),
            })
            print(f"finished random {profile_name} dt_scale={dt_scale:g}", flush=True)

    profile = config["profiles"]["unfavorable_50mM"]
    params = runner.physical_params(
        config,
        profile,
        domain="random",
        max_time=float(config["confirmatory"]["random_max_time_s"]),
    )
    for nx in (256, 384, 512):
        ny = int(round(nx * int(config["random_flow_raster"]["ny"]) / int(config["random_flow_raster"]["nx"])))
        grid = base_grid if (nx, ny) == (base_grid.nx, base_grid.ny) else runner.rcs.rasterize_flow_grid(
            interpolator,
            base_grid.lx,
            base_grid.ly,
            nx,
            ny,
        )
        library, diagnostics = simulate_random_transitions_compiled(
            grid,
            geometry,
            params,
            str(profile["condition"]),
            seed=args.seed + 10_000_019,
            allow_attachment=False,
            initial_x=random_x,
            initial_y=random_y,
            stop_at_next_distinct=True,
            return_entry_diagnostics=True,
        )
        data = diagnostic_data(library, diagnostics, random_refs)
        rows.append({
            "domain": "random",
            "geometry": "through_random_00",
            "sensitivity": "raster",
            "setting": f"raster_{nx}x{ny}",
            "dt_s": params.dt,
            "raster_nx": nx,
            "raster_ny": ny,
            "profile": "unfavorable_50mM",
            **analysis.metrics(data, window),
        })
        print(f"finished random unfavorable_50mM raster={nx}x{ny}", flush=True)

    path = out_dir / "entry_ffsz_sensitivity.csv"
    write_csv(path, rows)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
