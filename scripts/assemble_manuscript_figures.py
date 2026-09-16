#!/usr/bin/env python3
"""Collect the generated figures and tables under the file names used by the manuscript and its SI.

    python scripts/assemble_manuscript_figures.py            # copy into outputs/manuscript/
    python scripts/assemble_manuscript_figures.py --check DIR  # compare against a manuscript folder by checksum
    python scripts/assemble_manuscript_figures.py --root OTHER_REPO ...  # use another working tree's outputs/

Main-text figures are renamed figure_01.png ... figure_15.png (numbering of the full-length manuscript;
docs/FIGURE_TABLE_MAP.md gives the trimmed-manuscript numbering). Supplementary figures keep their generated
names. Table .tex files are copied from outputs/tables/ (written by the create_*_tables.py scripts).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"  # overridden by --root

# manuscript name -> generated file name (searched under outputs/, preferring outputs/figures/)
MAIN_FIGURES = {
    "figure_01.png": "fig_geometry_flow.png",                              # create_focused_geometry_flow_figure.py
    "figure_02.png": "release_angle_schematic.png",                        # create_release_angle_schematic.py
    "figure_03.png": "focused_results_01_dlvo_regimes.png",                # create_focused_results_figures.py
    "figure_04.png": "focused_results_02_event_clouds.png",                # create_focused_results_figures.py
    "figure_05.png": "focused_results_03_focusing_window.png",             # create_focused_results_figures.py
    "figure_06.png": "focused_results_04_highres_y_transition_matrices.png",  # create_focused_results_figures.py
    "figure_07.png": "focused_results_05_highres_focusing_matrices.png",   # create_focused_results_figures.py
    "figure_08.png": "science_regime_map.png",                             # create_science_synthesis_figures.py
    "figure_09.png": "random_geometry_stagnation_and_flow.png",            # create_random_geometry_flow_stagnation_figure.py
    "figure_10.png": "random_stage3_focusing_summary.png",                 # aggregate_stage3_random_production.py
    "figure_11.png": "random_geometry_mechanism_streamlines.png",          # create_stage3_random_mechanism_figure.py
    "figure_12.png": "science_next_interception_kernel.png",               # create_science_synthesis_figures.py
    "figure_13.png": "near_wall_boltzmann_validation.png",                 # validate_near_wall_boltzmann.py
    "figure_14.png": "figure_14_entry_side.png",                           # create_entry_side_figure.py
    "figure_15.png": "figure_15_f_alpha.png",                              # compute_f_alpha.py
}

SI_FIGURES = [
    "focused_mobile_state_metrics.png",          # create_focused_mobile_state_metrics.py
    "mechanism_injected_yield_controls.png",     # create_mechanism_revision_tables.py
    "focused_results_06_diffusion_control.png",  # create_focused_results_figures.py
    "focused_results_07_completion_outcomes.png",  # create_focused_results_figures.py
    "center_well_survival_curves.png",           # create_center_well_survival_curves.py
    "center_corner_hns_hc_sensitivity.png",      # run_center_corner_hns_hc_sensitivity.py
    "fig23_openfoam_convergence_validation.png", # run_openfoam_convergence_validation.py
    "random_geometry_focusing_trend.png",        # create_random_geometry_manuscript_figures.py
    "science_random_geometry_controls.png",      # create_science_synthesis_figures.py
    "science_shadow_synthesis.png",              # create_science_synthesis_figures.py
    "pore_shadow_metric_suite.png",              # analyze_pore_shadow_metrics.py
    "shadow_flow_feature_summary.png",           # analyze_shadow_flow_characteristics.py
]

MAIN_TABLES = [
    "boundary_algorithm_table.tex", "dlvo_parameter_table.tex", "event_state_kernel_table.tex",
    "focused_mobile_state_metrics_table.tex", "matched_event_definition_table.tex",
    "mechanism_injected_yield_table.tex", "random_geometry_science_summary.tex",
    "random_stage3_focusing_summary.tex", "science_regime_summary.tex",
]
SI_TABLES = [
    "center_corner_outcome_table.tex", "center_corner_replicate_table.tex", "dlvo_parameter_table.tex",
    "f_alpha_table.tex", "hns_hc_sensitivity_table.tex", "pore_shadow_metric_table.tex",
    "random_packing_outcome_table.tex", "validation_status_table.tex",
]


def find_output(name: str) -> Path | None:
    preferred = OUTPUTS / "figures" / name
    if preferred.exists():
        return preferred
    hits = sorted(OUTPUTS.rglob(name))
    return hits[0] if hits else None


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", type=Path, default=None,
                    help="manuscript folder holding figure_NN.png; report which generated figures match by checksum")
    ap.add_argument("--root", type=Path, default=None,
                    help="repository root whose outputs/ should be used (default: this repository)")
    args = ap.parse_args()
    global ROOT, OUTPUTS
    if args.root is not None:
        ROOT = args.root.resolve()
        OUTPUTS = ROOT / "outputs"

    dest = OUTPUTS / "manuscript"
    if args.check is None:
        (dest / "supplementary_information" / "tables").mkdir(parents=True, exist_ok=True)
        (dest / "tables").mkdir(exist_ok=True)

    missing = []
    for target, generated in MAIN_FIGURES.items():
        src = find_output(generated)
        if src is None:
            missing.append(generated)
            continue
        if args.check is not None:
            ref = args.check / target
            status = "MATCH" if ref.exists() and md5(ref) == md5(src) else "differs"
            print(f"{target:14s} <- {src.relative_to(ROOT)}  [{status}]")
        else:
            shutil.copy2(src, dest / target)
            print(f"{target:14s} <- {src.relative_to(ROOT)}")
    for name in SI_FIGURES:
        src = find_output(name)
        if src is None:
            missing.append(name)
        elif args.check is None:
            shutil.copy2(src, dest / "supplementary_information" / name)
    tables = OUTPUTS / "tables"
    for name in MAIN_TABLES:
        if (tables / name).exists():
            if args.check is None:
                shutil.copy2(tables / name, dest / "tables" / name)
        else:
            missing.append(name)
    for name in SI_TABLES:
        if (tables / name).exists():
            if args.check is None:
                shutil.copy2(tables / name, dest / "supplementary_information" / "tables" / name)
        else:
            missing.append(name)
    if missing:
        print("\nnot generated yet:")
        for m in missing:
            print(f"  {m}")
    if args.check is None:
        print(f"\nassembled under {dest}")


if __name__ == "__main__":
    main()
