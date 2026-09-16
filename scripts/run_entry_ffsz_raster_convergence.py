#!/usr/bin/env python3
"""Extend random-packing entry metrics to successively finer velocity rasters."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colloid_tsm.compiled import simulate_random_transitions_compiled  # noqa: E402


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = import_script("entry_ffsz_runner_for_raster", ROOT / "scripts" / "run_entry_ffsz_study.py")
analysis = import_script("entry_ffsz_analysis_for_raster", ROOT / "scripts" / "analyze_entry_ffsz_study.py")
DEFAULT_CONFIG = ROOT / "config" / "entry_ffsz.json"


def metric_data(library, diagnostics: dict[str, np.ndarray], refs: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    pair = diagnostics["pair_completed"].astype(bool)
    return {
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


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--geometry", default="through_random_00")
    parser.add_argument("--particles", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026081901)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[384, 512, 768, 1024])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    data_root = Path(config["data_root"])
    entry = next(item for item in config["random_geometries"] if item["name"] == args.geometry)
    geometry, interpolator, base_grid, _ = runner.load_random_resolved_flow(
        entry,
        config,
        data_root,
        output / "raster_cache",
    )
    injection = np.load(
        output / "confirmatory" / "random" / args.geometry / f"seed_{args.seed}" / "injection.npz"
    )
    initial_x = injection["initial_x"][: args.particles]
    initial_y = injection["initial_y"][: args.particles]
    window = np.deg2rad(float(config["analysis"]["ffsz_window_degrees"]))
    out_dir = output / "raster_convergence" / args.geometry
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "entry_ffsz_raster_convergence.csv"
    rows: list[dict] = []
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        requested = set(args.resolutions)
        rows = [row for row in rows if int(row["raster_nx"]) not in requested]

    for nx in args.resolutions:
        ny = int(round(nx * int(config["random_flow_raster"]["ny"]) / int(config["random_flow_raster"]["nx"])))
        grid = base_grid if (nx, ny) == (base_grid.nx, base_grid.ny) else runner.rcs.rasterize_flow_grid(
            interpolator,
            base_grid.lx,
            base_grid.ly,
            nx,
            ny,
        )
        stagnation, stagnation_rows = analysis.grain_analysis.compute_grain_stagnation_points(
            geometry,
            analysis.RasterInterpolator(grid),
            runner.PhysicalParams().particle_radius,
            float(config["analysis"]["stagnation_shell_gap_m"]),
            int(config["analysis"]["stagnation_angle_samples"]),
        )
        refs = {collector: points["forward"] for collector, points in stagnation.items()}
        with (out_dir / f"stagnation_{nx}x{ny}.json").open("w", encoding="utf-8") as handle:
            json.dump(stagnation_rows, handle, indent=2)

        for profile_name in ("no_dlvo", "unfavorable_50mM"):
            profile = config["profiles"][profile_name]
            params = runner.physical_params(
                config,
                profile,
                domain="random",
                max_time=float(config["confirmatory"]["random_max_time_s"]),
            )
            library, diagnostics = simulate_random_transitions_compiled(
                grid,
                geometry,
                params,
                str(profile["condition"]),
                seed=args.seed + 10_000_019,
                allow_attachment=bool(profile["allow_attachment"]),
                initial_x=initial_x,
                initial_y=initial_y,
                stop_at_next_distinct=True,
                return_entry_diagnostics=True,
            )
            data = metric_data(library, diagnostics, refs)
            row = {
                "geometry": args.geometry,
                "raster_nx": nx,
                "raster_ny": ny,
                "profile": profile_name,
                **analysis.metrics(data, window),
                **analysis.bootstrap_intervals(
                    data,
                    window,
                    2000,
                    np.random.default_rng(args.seed + nx),
                ),
            }
            rows.append(row)
            np.savez_compressed(
                out_dir / f"{profile_name}_{nx}x{ny}.npz",
                pair_completed=diagnostics["pair_completed"],
                first_entry_theta=diagnostics["first_entry_theta"],
                next_entry_theta=diagnostics["next_entry_theta"],
                first_entry_collector=diagnostics["first_entry_collector"],
                next_entry_collector=diagnostics["next_entry_collector"],
                first_entry_time=diagnostics["first_entry_time"],
                next_entry_time=diagnostics["next_entry_time"],
                first_collector_near_time=diagnostics["first_collector_near_time"],
                initial_x=initial_x,
                initial_y=initial_y,
            )
            print(f"finished {profile_name} raster={nx}x{ny}: pairs={row['paired_entries']}", flush=True)
        write_csv(summary_path, rows)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
