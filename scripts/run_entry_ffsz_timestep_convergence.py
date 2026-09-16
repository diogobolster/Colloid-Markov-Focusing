#!/usr/bin/env python3
"""Checkpointed high-support timestep convergence for periodic entry focusing."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = import_script(
    "entry_ffsz_runner_for_dt_convergence",
    ROOT / "scripts" / "run_entry_ffsz_study.py",
)
analysis = import_script(
    "entry_ffsz_analysis_for_dt_convergence",
    ROOT / "scripts" / "analyze_entry_ffsz_study.py",
)
DEFAULT_CONFIG = ROOT / "config" / "entry_ffsz.json"


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


def stage_dir_for_scale(output: Path, scale: float) -> Path:
    if scale == 1.0:
        return output / "confirmatory"
    return output / "timestep_convergence" / f"dt_scale_{scale:g}" / "confirmatory"


def run_scale(config: dict, output: Path, data_root: Path, scale: float) -> None:
    if scale == 1.0:
        return
    scaled = copy.deepcopy(config)
    scaled["particle_tracking"]["periodic_dt_s"] = (
        float(config["particle_tracking"]["periodic_dt_s"]) * scale
    )
    scale_root = output / "timestep_convergence" / f"dt_scale_{scale:g}"
    rows = runner.run_periodic(
        scaled,
        "confirmatory",
        ["no_dlvo", "unfavorable_50mM"],
        data_root,
        scale_root,
        True,
    )
    runner.write_manifest(scale_root / "confirmatory" / "run_manifest.json", rows)


def load_scale(
    config: dict,
    output: Path,
    scale: float,
    refs: dict[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    groups = analysis.discover_groups(stage_dir_for_scale(output, scale))
    by_profile: dict[str, list[dict[str, np.ndarray]]] = {
        "no_dlvo": [],
        "unfavorable_50mM": [],
    }
    for domain, geometry, _seed, profile, files in groups:
        if domain != "periodic" or geometry != "center_corner_cell" or profile not in by_profile:
            continue
        by_profile[profile].append(analysis.load_group(files, refs))
    if not by_profile["no_dlvo"] or len(by_profile["no_dlvo"]) != len(by_profile["unfavorable_50mM"]):
        raise RuntimeError(f"incomplete periodic timestep data at dt scale {scale:g}")
    return (
        analysis.pooled_for_matching(by_profile["no_dlvo"]),
        analysis.pooled_for_matching(by_profile["unfavorable_50mM"]),
    )


def plot_summary(rows: list[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["dt_scale"]))
    scales = np.array([float(row["dt_scale"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.4), dpi=220, sharex=True)
    ax = axes[0]
    for name, label, color, marker in (
        ("observed_condition_specific_amplification", "Condition-specific apparent effect", "#8A98A8", "o"),
        ("matched_direct_amplification", "Matched direct effect", "#07867E", "s"),
        ("completion_selection_component", "Completion-selection component", "#D7A72D", "^"),
    ):
        values = np.array([float(row[name]) for row in rows])
        low = np.array([float(row[f"{name}_ci_low"]) for row in rows])
        high = np.array([float(row[f"{name}_ci_high"]) for row in rows])
        ax.errorbar(
            scales,
            values,
            yerr=np.vstack((values - low, high - values)),
            color=color,
            marker=marker,
            capsize=4,
            lw=1.6,
            label=label,
        )
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_ylabel("Unfavorable amplification of Entry F30 change")
    ax.set_title("Threshold metric")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D7DCE2", lw=0.6)

    ax = axes[1]
    for name, label, color, marker in (
        ("mean_additional_contraction_deg", "Mean", "#07867E", "s"),
        ("median_additional_contraction_deg", "Median", "#D7A72D", "o"),
    ):
        values = np.array([float(row[name]) for row in rows])
        low = np.array([float(row[f"{name}_ci_low"]) for row in rows])
        high = np.array([float(row[f"{name}_ci_high"]) for row in rows])
        ax.errorbar(
            scales,
            values,
            yerr=np.vstack((values - low, high - values)),
            color=color,
            marker=marker,
            capsize=4,
            lw=1.6,
            label=label,
        )
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(scales, [f"{scale:g}" for scale in scales])
    ax.set_xlabel("Outer timestep scale relative to 1 ms baseline")
    ax.set_ylabel("Additional contraction (deg)")
    ax.set_title("Threshold-free paired contraction")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D7DCE2", lw=0.6)
    fig.suptitle("Periodic-cell entry-focusing sensitivity to outer timestep")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output

    scales = sorted(set(float(value) for value in args.scales))
    if not args.analyze_only:
        for scale in scales:
            run_scale(config, output, data_root, scale)

    refs, _ = analysis.periodic_stagnation(config)
    window = np.deg2rad(float(config["analysis"]["ffsz_window_degrees"]))
    replicates = int(config["analysis"]["bootstrap_replicates"])
    rng = np.random.default_rng(2026081907)
    rows: list[dict] = []
    for scale in scales:
        no_dlvo, unfavorable = load_scale(config, output, scale, refs)
        matched = analysis.matched_chemistry_metrics(
            no_dlvo,
            unfavorable,
            window,
            replicates,
            rng,
        )
        rows.append({
            "dt_scale": scale,
            "outer_dt_s": float(config["particle_tracking"]["periodic_dt_s"]) * scale,
            **analysis.support_selection_metrics(
                no_dlvo,
                unfavorable,
                window,
                replicates,
                rng,
            ),
            **matched,
        })

    out_dir = output / "timestep_convergence" / "analysis"
    csv_path = out_dir / "periodic_entry_ffsz_timestep_convergence.csv"
    figure_path = out_dir / "periodic_entry_ffsz_timestep_convergence.png"
    write_csv(csv_path, rows)
    plot_summary(rows, figure_path)
    report = {
        "scales": scales,
        "summary_csv": str(csv_path),
        "figure": str(figure_path),
    }
    report_path = out_dir / "periodic_entry_ffsz_timestep_convergence.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
