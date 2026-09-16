# Colloid-Markov-Focusing (revised code package)

Code for the simulations and analyses in "Focusing Without Attachment: Secondary-Minimum Residence Steers Mobile Colloid Release toward Rear Stagnation Zones" (B. M. Al-Zghoul, W. P. Johnson, D. Bolster, submitted to Advances in Water Resources). This is the content of the revised code repository (https://github.com/diogobolster/Colloid-Markov-Focusing, MIT License). It is code-only: manuscript sources, generated trajectories, OpenFOAM cases, rasters and figures are not versioned; a small set of derived CSV summaries is included under `reference_results/` so that the tabulated numbers can be checked without rerunning the production suites.

The study isolates homogeneous unfavorable secondary-minimum residence in smooth periodic grain-scale geometries. The code covers a center/corner periodic cell and random periodic packings, body-fitted OpenFOAM flow generation and rasterization, resolved-Langevin particle tracking with a compiled C kernel, grain-local focusing diagnostics, transition matrices and event-conditioned kernels, the release-to-next-interception and paired entry-side (grain X to grain Y) diagnostics, the predictive angular residence number, validation tests, and transport-shadow diagnostics.

`docs/FIGURE_TABLE_MAP.md` traces every figure and table of the manuscript and its Supplementary Information to the script that makes it and the data product it reads. Start there if you want to regenerate one item rather than everything.

## What changed relative to the first public version

- Added the post-processing scripts that produce the manuscript's synthesis figures and tables and were missing from the first package: `create_science_synthesis_figures.py` (regime map, next-interception kernel figure, geometry-control and shadow syntheses, `event_state_kernel_table`, `science_regime_summary`, `random_geometry_science_summary`), `create_mechanism_revision_tables.py` (`mechanism_injected_yield_table`, `matched_event_definition_table`, `boundary_algorithm_table`, injected-yield controls figure), `create_submission_readiness_tables.py` (DLVO parameter, outcome, replicate, sensitivity, validation and random-packing outcome tables), `create_random_geometry_manuscript_figures.py` (single-packing trend figure) and `build_production_assets.py`. Their output paths were normalized to `outputs/figures/` and `outputs/tables/`.
- Added `compute_f_alpha.py` (cutoff-free focusing curves F(alpha), rear alignment A, net angular travel; Figure 15 and SI Table S6).
- Added `compute_predictive_lambda.py` (mean first-passage time tau_K, re-entry multiplicity, angular residence number Lambda_theta, and the streamline-nesting argument; main-text Table 3 and Section 3.4).
- Added `create_entry_side_figure.py` (assembles Figure 14 from the confirmatory entry-side analysis panels) and `assemble_manuscript_figures.py` (renames generated files to the manuscript's `figure_NN.png` and collects the tables; `--check` compares against a manuscript folder by checksum).
- Added `entry_side_population/` with the standalone entry-angle analysis (`entry_ffsz_focusing.py`) and the JSON results behind the population rows of the entry-side table.
- Fixed the y-axis label of the five-geometry scatter in `aggregate_stage3_random_production.py` (it rendered as a literal `\(F_{30}\)`).
- `docs/FIGURE_TABLE_MAP.md`, `docs/REPRODUCIBILITY.md`, `Makefile` and this README updated; see `CHANGELOG.md`.

## Repository layout

- `colloid_tsm/`: Python package for physical parameters, DLVO tracking, flow rasters, transition libraries and the compiled-kernel wrapper.
- `colloid_tsm/native/particle_kernel.c`: C implementation of the hot particle-tracking loop (builds to `libparticle_kernel.*`, not versioned).
- `scripts/`: run scripts for flow generation, particle tracking, validation, random-packing production, focusing diagnostics, post-processing and figure/table generation.
- `config/production.json`: center/corner production and validation configuration.
- `config/random_stage3.json`: random-packing production, next-interception and transport-shadow configuration.
- `config/entry_ffsz.json`: paired grain X to distinct grain Y entry-angle study (periodic cell and five random geometries).
- `entry_side_population/`: population-level entry-angle analysis and its archived results.
- `reference_results/`: small derived CSV summaries (F(alpha), predictive Lambda_theta) for checking tabulated numbers.
- `docs/`: `FIGURE_TABLE_MAP.md` and `REPRODUCIBILITY.md`.
- `Makefile`: convenience entry points.

Generated outputs are written under `outputs/` and are ignored by git.

## Requirements

Python 3.11 or newer; dependencies in `requirements.txt` (numpy, scipy, matplotlib, pandas, pillow). A C compiler for the particle kernel (Apple clang or gcc). OpenFOAM v2506 for flow generation, either installed locally or through the Docker helper `scripts/openfoam_docker.sh`; gmsh for the random-packing meshes.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/build_particle_kernel.py
```

## Full reproduction, in order

```bash
# 1. flow fields and rasters
python scripts/run_openfoam_focusing_cases.py            # center/corner OpenFOAM case (Docker helper available)
python scripts/create_openfoam_raster_flow.py            # -> outputs/openfoam_flow/openfoam_flow_N512.npz
python scripts/generate_random_porous_geometry.py        # random packings
python scripts/run_stage1_geometry_screen.py
python scripts/run_random_openfoam_flow.py

# 2. center/corner production, validation and transition matrices
python scripts/production_pipeline.py run-full --config config/production.json --reuse
python scripts/production_pipeline.py validate --config config/production.json
python scripts/run_focusing_transition_matrices.py
python scripts/kernel_regression_tests.py --config config/production.json

# 3. random packings: production, post-processing, shadow diagnostic, validation
python scripts/run_random_stage3_pipeline.py production  --config config/random_stage3.json
python scripts/run_random_stage3_pipeline.py postprocess --config config/random_stage3.json
python scripts/run_random_stage3_pipeline.py shadow      --config config/random_stage3.json
python scripts/run_random_stage3_pipeline.py validate    --config config/random_stage3.json
python scripts/analyze_release_to_next_interception.py

# 4. paired entry-side study
python scripts/validate_entry_ffsz_tracking.py
python scripts/run_entry_ffsz_study.py --config config/entry_ffsz.json --stage confirmatory --domains both --resume
python scripts/analyze_entry_ffsz_study.py --config config/entry_ffsz.json --stage confirmatory

# 5. figures and tables
python scripts/create_focused_geometry_flow_figure.py
python scripts/create_release_angle_schematic.py
python scripts/create_focused_results_figures.py
python scripts/create_focused_mobile_state_metrics.py
python scripts/create_center_well_survival_curves.py
python scripts/create_mechanism_revision_tables.py
python scripts/create_submission_readiness_tables.py
python scripts/create_science_synthesis_figures.py
python scripts/create_random_geometry_flow_stagnation_figure.py
python scripts/create_stage3_random_mechanism_figure.py
python scripts/create_random_geometry_manuscript_figures.py
python scripts/validate_near_wall_boltzmann.py
python scripts/compute_f_alpha.py
python scripts/compute_predictive_lambda.py
python scripts/create_entry_side_figure.py
python scripts/assemble_manuscript_figures.py            # -> outputs/manuscript/figure_01.png ... figure_15.png, tables/
```

The `Makefile` wraps the same steps (`make kernel`, `make run-full`, `make random-production`, `make entry-confirmatory`, `make figures`, `make assemble`). Production tracking runs take hours to days; every long stage is chunked and checkpointed and can be restarted with the same command.

## What is not included

Manuscript TeX and PDF files, generated figures, trajectory libraries, OpenFOAM case outputs, velocity rasters and compiled dynamic libraries. These can be regenerated with the commands above or obtained from the data archive (DOI to be added on deposition). Two calculations quoted in the paper were run interactively rather than from a versioned script and are documented in `docs/FIGURE_TABLE_MAP.md` (last section): the generation-1 first-passage propagation behind the population rows of the entry-side table, and the 90 s horizon-truncation comparison for the five-geometry ensemble.

## Citation and license

MIT License (https://opensource.org/license/mit). Please cite the companion manuscript when using this code.
