# Changelog

## 2026-09-16 (revised package for the AWR submission)

Base: repository state of 2026-08-19 (commit 9335d50, "Add matched high-diffusion entry control").

Added scripts
- `scripts/create_science_synthesis_figures.py`, `scripts/create_mechanism_revision_tables.py`, `scripts/create_submission_readiness_tables.py`, `scripts/create_random_geometry_manuscript_figures.py`, `scripts/build_production_assets.py`: manuscript post-processing scripts from the working tree that the first package omitted. Output paths normalized from `outputs/paper_figures/` and `paper/tables/` to `outputs/figures/` and `outputs/tables/`; no other changes.
- `scripts/compute_f_alpha.py`: cutoff-free focusing curves, rear-alignment statistic A, net angular travel (Figure 15, SI Table S6, the A column of the outcome table).
- `scripts/compute_predictive_lambda.py`: mean first-passage time tau_K from the DLVO potential, re-entry multiplicity from the libraries, Lambda_theta, and streamline outlet offsets (Table 3 and Section 3.4 of the trimmed manuscript). Reproduces the published table to the printed precision.
- `scripts/create_entry_side_figure.py`: assembles the two-panel entry-side figure from the confirmatory analysis panels.
- `scripts/assemble_manuscript_figures.py`: manuscript naming of generated figures and tables; `--check` verifies against a manuscript folder by checksum.
- `entry_side_population/`: `entry_ffsz_focusing.py` and the archived JSON results behind the population rows of the entry-side table, with the production report.
- `reference_results/`: `f_alpha/` and `predictive_lambda/` CSV summaries.

Fixed
- `scripts/aggregate_stage3_random_production.py`: y-axis label of the five-geometry scatter used `\(F_{30}\)` (not mathtext) and rendered literally; now `$F_{30}$`.

Documentation
- `docs/FIGURE_TABLE_MAP.md` (new): every figure and table traced to script, generated file and input data, for both the full-length and the trimmed manuscript numbering.
- `docs/REPRODUCIBILITY.md`, `README.md`, `Makefile` updated for the added steps.

Not in the package (documented in `docs/FIGURE_TABLE_MAP.md`)
- The generation-1 first-passage propagation driver used for the population entry-angle baseline.
- The interactive 90 s horizon-truncation comparison for the five-geometry ensemble.

Excluded on purpose
- Three-dimensional extension scripts from the working tree (`*three_d*`, `generate_periodic_porous_geometry_3d.py`, `run_periodic_openfoam_flow_3d.py`, `run_simple_cubic_contact_mesh_tuning.py`, `config/three_d_stage1.json`, `colloid_tsm/geometry3d.py`); not used in this paper.
