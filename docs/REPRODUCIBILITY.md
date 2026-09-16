# Reproducibility Notes

The code package is organized around three production configurations. `docs/FIGURE_TABLE_MAP.md` maps every manuscript and SI item to its script and data product; this file gives the run order and what each stage expects.

## Center/Corner Periodic Cell

Configuration: `config/production.json`

```bash
python scripts/build_particle_kernel.py
python scripts/production_pipeline.py run --config config/production.json --reuse
python scripts/production_pipeline.py run-full --config config/production.json --reuse
python scripts/production_pipeline.py validate --config config/production.json
python scripts/run_focusing_transition_matrices.py
python scripts/kernel_regression_tests.py --config config/production.json
```

The config expects the center/corner OpenFOAM-derived velocity raster at `outputs/openfoam_flow/openfoam_flow_N512.npz`. If it is absent, regenerate the OpenFOAM case (`scripts/run_openfoam_focusing_cases.py`, OpenFOAM v2506; `scripts/openfoam_docker.sh` runs it in the `openeuler/openfoam:2506` image) and rasterize it with `scripts/create_openfoam_raster_flow.py` (periodic inverse-distance weighting over the 12 nearest cell centers, exponent 2, solid nodes zeroed). The raster-resolution, timestep and normal-step checks are `scripts/run_openfoam_convergence_validation.py` and `scripts/run_timestep_audit.py`; seed replicates are `scripts/run_center_corner_replicate_seeds.py`; event-definition sensitivity is `scripts/run_center_corner_hns_hc_sensitivity.py`; the off-flow Boltzmann test is `scripts/validate_near_wall_boltzmann.py`.

`run-full` writes one library per condition, `outputs/openfoam_full_suite/trajectory_library_refinement_<profile>.npz` (central-core injection, 30,000 particles). The transition ensemble (46,000 particles, seven inlet strata) is written by `run_focusing_transition_matrices.py`, together with `outputs/focusing_transition_matrices/focusing_transition_matrix_rows.csv`, which holds the per-row statistics quoted in the text.

Post-processing that reads only these libraries:

```bash
python scripts/create_focused_results_figures.py       # Figures 3-7 and SI S4, S5
python scripts/create_focused_mobile_state_metrics.py  # focused_mobile_state_metrics table and SI S2
python scripts/create_center_well_survival_curves.py   # SI S6
python scripts/create_mechanism_revision_tables.py     # injected-yield, matched-event and boundary tables; SI S3
python scripts/create_submission_readiness_tables.py   # DLVO, outcome, replicate, sensitivity, validation tables
python scripts/compute_f_alpha.py                      # Figure 15, SI Table S6
python scripts/compute_predictive_lambda.py            # Table 3 (Lambda_theta) and the streamline offsets of Section 3.4
```

## Random Periodic Packing

Configuration: `config/random_stage3.json`

```bash
python scripts/run_random_stage3_pipeline.py production  --config config/random_stage3.json
python scripts/run_random_stage3_pipeline.py postprocess --config config/random_stage3.json
python scripts/run_random_stage3_pipeline.py shadow      --config config/random_stage3.json
python scripts/run_random_stage3_pipeline.py validate    --config config/random_stage3.json
python scripts/analyze_release_to_next_interception.py
python scripts/create_science_synthesis_figures.py
python scripts/create_random_geometry_flow_stagnation_figure.py
python scripts/create_stage3_random_mechanism_figure.py
python scripts/create_random_geometry_manuscript_figures.py
```

The single-packing ionic-strength sweep (three seeds, 1800 s horizon) is run with `scripts/run_random_condition_screen.py --out-dir outputs/random_condition_grainlocal_production_<seed>` and analyzed grain-locally with `scripts/analyze_random_grain_focusing.py <payload>` (shell offset via `--sample-gap-um`; seed spread via `scripts/aggregate_grain_focusing_replicates.py`; particle bootstrap via `scripts/bootstrap_grain_focusing_uncertainty.py`). The five-geometry ensemble (10,000 particles per condition per geometry, 90 s horizon) is the `production`/`postprocess` pair above. The long transport-shadow calculation is checkpointed by particle chunks and restarts with the same `shadow` command.

## Paired entry-side study

Configuration: `config/entry_ffsz.json`

```bash
python scripts/validate_entry_ffsz_tracking.py
python scripts/run_entry_ffsz_study.py --config config/entry_ffsz.json --stage confirmatory --domains both --resume
python scripts/analyze_entry_ffsz_study.py --config config/entry_ffsz.json --stage confirmatory
python scripts/run_entry_ffsz_timestep_convergence.py --config config/entry_ffsz.json
python scripts/create_entry_side_figure.py
```

The analysis separates the condition-specific change in entry-side focusing into a matched-particle direct effect and a completion-selection component, per domain and per geometry. Completed chunks are validated and reused on restart. The population-level entry-angle rows (first-passage baseline and distinct-grain next interceptions) come from `entry_side_population/entry_ffsz_focusing.py` applied to the production restart events and stagnation points; see `docs/FIGURE_TABLE_MAP.md` for the one driver that is not archived.

## Assembling the manuscript items

```bash
python scripts/assemble_manuscript_figures.py                  # outputs/manuscript/figure_01.png ... figure_15.png and tables
python scripts/assemble_manuscript_figures.py --check <folder> # checksum comparison against a manuscript folder
```

## Regression Tests

The compiled-kernel regression tests check deterministic reproducibility for a neutral run, adaptive timestep diagnostics in the near-wall secondary-minimum region, and irreversible favorable attachment at contact:

```bash
python scripts/kernel_regression_tests.py --config config/production.json
```

## Generated Outputs

All generated outputs are ignored by git. `reference_results/` keeps two small CSV sets (F(alpha) and predictive Lambda_theta) so the corresponding tables can be checked without rerunning the production suites.
