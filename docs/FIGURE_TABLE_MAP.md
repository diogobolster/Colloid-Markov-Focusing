# Figure and table map

Every display item of "Focusing Without Attachment: Secondary-Minimum Residence Steers Mobile Colloid Release toward Rear Stagnation Zones" (Al-Zghoul, Johnson, Bolster) traced to the script that produces it and the data product it reads. All commands run from the repository root; `outputs/` holds the generated data products (not versioned).

Two numberings are given: the full-length manuscript (`manuscript.tex`, 15 figures) and the trimmed manuscript (`*_trim.tex` / `*_trim_humanized.tex`, 11 figures; the same 15 files, four of them moved to the SI). `scripts/assemble_manuscript_figures.py` renames the generated files to `figure_NN.png` using the full-length numbering.

## Pipeline order

1. Flow: `scripts/run_openfoam_focusing_cases.py` (center/corner, OpenFOAM v2506 via `scripts/openfoam_docker.sh`) -> `scripts/create_openfoam_raster_flow.py` -> `outputs/openfoam_flow/openfoam_flow_N512.npz`. Random packings: `scripts/generate_random_porous_geometry.py`, `scripts/run_stage1_geometry_screen.py`, `scripts/run_random_openfoam_flow.py`.
2. Kernel: `python scripts/build_particle_kernel.py` (compiles `colloid_tsm/native/particle_kernel.c`).
3. Center/corner production: `python scripts/production_pipeline.py run-full --config config/production.json --reuse` -> `outputs/openfoam_full_suite/trajectory_library_refinement_<profile>.npz` (central-core and transition ensembles), plus the validation targets of `production_pipeline.py validate`.
4. Random packing: `python scripts/run_random_stage3_pipeline.py production|postprocess|shadow|validate --config config/random_stage3.json` -> `outputs/stage3_particle_production/`, `outputs/random_condition_grainlocal_production_*`, `outputs/next_interception_kernel/`, `outputs/stage3_transport_shadow/`.
5. Entry side: `python scripts/run_entry_ffsz_study.py --config config/entry_ffsz.json --stage confirmatory --domains both --resume` then `python scripts/analyze_entry_ffsz_study.py --config config/entry_ffsz.json --stage confirmatory` -> `outputs/entry_ffsz/confirmatory/analysis/`.
6. Post-processing and figures: the scripts listed below, then `python scripts/assemble_manuscript_figures.py`.

## Main-text figures

| Full | Trim | Content | Script | Generated file | Reads |
|---|---|---|---|---|---|
| 1 | 1 | Periodic geometry and resolved flow | `create_focused_geometry_flow_figure.py` | `outputs/figures/fig_geometry_flow.png` | `outputs/openfoam_flow/openfoam_flow_N512.npz` |
| 2 | 2 | Release-angle schematic | `create_release_angle_schematic.py` | `outputs/figures/release_angle_schematic.png` | none (schematic) |
| 3 | 3 | DLVO potentials; well depth vs residence | `create_focused_results_figures.py` | `outputs/figures/focused_results_01_dlvo_regimes.png` | full-suite libraries; `colloid_tsm/physical.py` |
| 4 | 4 | Event clouds (residence vs release angle) | `create_focused_results_figures.py` | `outputs/figures/focused_results_02_event_clouds.png` | full-suite libraries |
| 5 | SI S1 | Focusing-window bar summary | `create_focused_results_figures.py` | `outputs/figures/focused_results_03_focusing_window.png` | full-suite libraries |
| 6 | SI S7 | 100-bin transverse transition matrices | `create_focused_results_figures.py` | `outputs/figures/focused_results_04_highres_y_transition_matrices.png` | transition-ensemble libraries (`run_focusing_transition_matrices.py`) |
| 7 | SI S8 | 100-bin event-conditioned kernels | `create_focused_results_figures.py` | `outputs/figures/focused_results_05_highres_focusing_matrices.png` | transition-ensemble libraries |
| 8 | 6 | Regime synthesis (F30 vs residence, Y30, depth, accounting) | `create_science_synthesis_figures.py` | `outputs/science_synthesis/science_regime_map.png` (copy in `outputs/figures/`) | `outputs/submission_readiness/*.csv`, full-suite libraries |
| 9 | 7 | Random packing: grain-local stagnation points and flow | `create_random_geometry_flow_stagnation_figure.py` | `outputs/figures/random_geometry_stagnation_and_flow.png` | `outputs/random_porous_geometry/`, random OpenFOAM raster, `grain_stagnation_points.csv` |
| 10 | SI S11 | Five-geometry ensemble scatter | `aggregate_stage3_random_production.py` (called by `run_random_stage3_pipeline.py postprocess`) | `outputs/figures/random_stage3_focusing_summary.png` | `outputs/stage3_particle_production/` |
| 11 | 8 | Streamlines with release outcomes (3 chemistries) | `create_stage3_random_mechanism_figure.py` | `outputs/figures/random_geometry_mechanism_streamlines.png` | production packing libraries and raster |
| 12 | 9 | Release-to-next-interception kernel (4 panels) | `create_science_synthesis_figures.py` | `outputs/science_synthesis/science_next_interception_kernel.png` | `outputs/next_interception_kernel/release_to_next_interception_{summary,events}.csv` (from `analyze_release_to_next_interception.py`) |
| 13 | 11 (App. D) | Off-flow Boltzmann validation | `validate_near_wall_boltzmann.py` | `outputs/figures/near_wall_boltzmann_validation.png` | none (self-contained test) |
| 14 | 10 | Paired entry-side diagnostic (2 panels) | `create_entry_side_figure.py` (composites panels from `analyze_entry_ffsz_study.py`) | `outputs/figures/figure_14_entry_side.png` | `outputs/entry_ffsz/confirmatory/analysis/entry_ffsz_{geometry_summary,selection_decomposition}.png` |
| 15 | 5 | Cutoff-free focusing curves F(alpha) | `compute_f_alpha.py` | `outputs/figures/figure_15_f_alpha.png` | full-suite libraries (central-core) |

Known blemish: the y-axis label of the five-geometry scatter (full Figure 10 / SI S11) renders as the literal string `\(F_{30}\)`; fix the label in `aggregate_stage3_random_production.py` (use `$F_{30}$`) and rerun `postprocess`.

## Main-text tables

| Full | Trim | Content | Source |
|---|---|---|---|
| 1 | 1 | Metric definitions | hand-written in the manuscript |
| 2 (`mechanism_injected_yield_table`) | merged into 2 | N_inj, N_event, P_event, q_rel, F30, Y30, P_inj,30 | `create_mechanism_revision_tables.py` -> `outputs/tables/mechanism_injected_yield_table.tex` |
| 3 (`event_state_kernel_table`) | 4 | Event-state variables | `create_science_synthesis_figures.py` -> `outputs/tables/event_state_kernel_table.tex` |
| 4 (`tab:lambda`) | 3 | tau_K, residence per entry, N_e, Lambda_theta, net travel | `compute_predictive_lambda.py` -> `outputs/predictive_lambda/{predictive_lambda.csv,lambda_table.tex}` |
| 5 (`science_regime_summary`) | merged into 2 | Regime synthesis | `create_science_synthesis_figures.py` -> `outputs/tables/science_regime_summary.tex` |
| 6 (`random_stage3_focusing_summary`) | merged into 5 | Five-geometry ensemble | `aggregate_stage3_random_production.py` -> `outputs/tables/random_stage3_focusing_summary.tex` |
| 7 (`tab:entry-side`) | 6 | Entry-side E30 population and paired rows | population rows: `entry_side_population/entry_ffsz_focusing.py` on the production restart events (`entry_ffsz_results.json`, `entry_ffsz_production_gen{1,2}.json`); paired rows: `analyze_entry_ffsz_study.py` -> `outputs/entry_ffsz/confirmatory/analysis/entry_ffsz_{matched_chemistry_summary,support_selection_summary}.csv` |
| 8 (`boundary_algorithm_table`) | 7 | Boundary/barrier rules | `create_mechanism_revision_tables.py` |
| 9 (`tab:model-parameters`) | 8 | Model parameters | hand-written from `colloid_tsm/physical.py` and `config/production.json` |
| 10 (`tab:raster-convergence`) | 13 | Raster convergence | `run_openfoam_convergence_validation.py` (values transcribed) |
| 11 (`dlvo_parameter_table`) | 9 | DLVO parameters and barriers | `create_submission_readiness_tables.py` -> `outputs/tables/dlvo_parameter_table.tex` |
| 12 (`random_packing_parameter_table`) | 10 | Random packing parameters | hand-written from `config/random_stage3.json` and `outputs/random_porous_geometry/` |
| 13 (`focused_mobile_state_metrics_table`) | merged into 2 | q_rel, F30, Y30, G30, Phi30, A | `create_focused_mobile_state_metrics.py` -> `outputs/tables/focused_mobile_state_metrics_table.tex`; A also from `compute_f_alpha.py` |
| 14 (`matched_event_definition_table`) | 11 | Near-surface vs well event definition | `create_mechanism_revision_tables.py` -> `outputs/tables/matched_event_definition_table.tex` |
| 15 (`random_geometry_science_summary`) | 12 | Geometry controls | `create_science_synthesis_figures.py` -> `outputs/tables/random_geometry_science_summary.tex` |
| trim 5 (`tab:random-summary`) | 5 | Single-packing sweep, ensemble, tau_ns, M30, same-grain | single-packing columns: `run_random_condition_screen.py --out-dir outputs/random_condition_grainlocal_production_<seed>` (three seeds) then `analyze_random_grain_focusing.py <payload>` -> `outputs/random_grain_focusing_production_<seed>/` (seed spread via `aggregate_grain_focusing_replicates.py`; bootstrap via `bootstrap_grain_focusing_uncertainty.py`; shell-offset sensitivity via `--sample-gap-um`); tau_ns: near-surface residence in the same payloads; ensemble: `aggregate_stage3_random_production.py`; M30 and same-grain: `analyze_release_to_next_interception.py` -> `outputs/next_interception_kernel/release_to_next_interception_summary.csv` |

Numbers quoted in the trimmed text that are not in a table: streamline outlet offsets (0.13 and 0.26 um) from `compute_predictive_lambda.py` (`outputs/predictive_lambda/streamline_offsets.csv`); transverse-matrix row statistics (0.912/0.913/0.913, 6.30/6.27/6.24 um, 0.308, 23.8 um) and center-grazing-row F30 (0.194/0.572/0.945) from `run_focusing_transition_matrices.py` -> `outputs/focusing_transition_matrices/focusing_transition_matrix_rows.csv` (columns `yout_center_fraction_all_mobile`, `median_abs_yout_center_um`, `center_well_theta30_fraction`, state `center core`); ensemble censoring (2.6-7.4 % of injected 50 mM particles resident at 90 s in the five geometries, and the single-packing 50 mM F30 falling from 0.388 to 0.341 when its releases are truncated at 90 s, the excluded releases having F30 = 0.654) computed ad hoc from the `stage3_particle_production` chunk files and the single-packing release events (see the last section).

## Supplementary Information

| SI item | Content | Script | Generated file |
|---|---|---|---|
| Tables S1, S2 | Ensembles, horizons, inlet strata | `config/production.json`; `create_submission_readiness_tables.py` | hand-transcribed |
| Table S3 | DLVO parameters | `create_submission_readiness_tables.py` | `outputs/tables/dlvo_parameter_table.tex` |
| Table S4 | Center/corner outcome audit | `create_submission_readiness_tables.py` | `outputs/tables/center_corner_outcome_table.tex` |
| Table S5 | Seed replicates | `run_center_corner_replicate_seeds.py`; `create_submission_readiness_tables.py` | `outputs/tables/center_corner_replicate_table.tex` |
| Table S6 | F(alpha) at fixed cutoffs, A, net travel | `compute_f_alpha.py` | `outputs/f_alpha/{f_alpha_summary.csv,table_row_snippets.txt}` -> `f_alpha_table.tex` |
| Fig. S1 | Focusing-window bars | `create_focused_results_figures.py` | `focused_results_03_focusing_window.png` |
| Fig. S2 | Focused-mobile-state support metrics | `create_focused_mobile_state_metrics.py` | `focused_mobile_state_metrics.png` |
| Fig. S3 | Injected-yield controls | `create_mechanism_revision_tables.py` | `mechanism_injected_yield_controls.png` |
| Fig. S4 | High-diffusion control | `create_focused_results_figures.py` | `focused_results_06_diffusion_control.png` |
| Fig. S5 | Completion outcomes | `create_focused_results_figures.py` | `focused_results_07_completion_outcomes.png` |
| Fig. S6 | Survival and release-completion curves | `create_center_well_survival_curves.py` | `center_well_survival_curves.png` |
| Fig. S7, S8 | Transverse matrices; event-conditioned kernels | `create_focused_results_figures.py` | `focused_results_04_*.png`, `focused_results_05_*.png` |
| Table S7 | h_ns / h_c sensitivity | `run_center_corner_hns_hc_sensitivity.py`; `create_submission_readiness_tables.py` | `outputs/tables/hns_hc_sensitivity_table.tex` |
| Table S8 | Validation status | `create_submission_readiness_tables.py` | `outputs/tables/validation_status_table.tex` |
| Fig. S9 | h_ns / h_c sensitivity graphic | `run_center_corner_hns_hc_sensitivity.py` | `center_corner_hns_hc_sensitivity.png` |
| Fig. S10 | OpenFOAM/raster/timestep convergence | `run_openfoam_convergence_validation.py`, `run_timestep_audit.py` | `fig23_openfoam_convergence_validation.png` |
| Table S9 | Random-packing outcome audit | `create_submission_readiness_tables.py` | `outputs/tables/random_packing_outcome_table.tex` |
| Fig. S11 | Five-geometry ensemble scatter | `aggregate_stage3_random_production.py` | `random_stage3_focusing_summary.png` |
| Fig. S12 | Single-packing sweep trend | `create_random_geometry_manuscript_figures.py` | `random_geometry_focusing_trend.png` |
| Fig. S13 | Geometry-control synthesis | `create_science_synthesis_figures.py` | `science_random_geometry_controls.png` |
| Tables S10, S11 | Shadow metrics | `analyze_pore_shadow_metrics.py`; `run_transport_shadow_diagnostic.py` | `outputs/tables/pore_shadow_metric_table.tex`, `transport_shadow_table.tex` |
| Fig. S14 | Shadow synthesis | `create_science_synthesis_figures.py` | `science_shadow_synthesis.png` |
| Fig. S15 | Pore-shadow metric suite | `analyze_pore_shadow_metrics.py` | `pore_shadow_metric_suite.png` |
| Fig. S16 | Shadow flow features | `analyze_shadow_flow_characteristics.py` | `shadow_flow_feature_summary.png` |

## Items whose driver is not in this package

The 90 s horizon-truncation comparison (Appendix C, "Ensemble censoring") was computed interactively from `outputs/stage3_particle_production/<geometry>/` chunk files (fraction of injected and of intercepted 50 mM particles still resident at 90 s) and from the single-packing `grain_local_release_events.csv` (release time filter at 90 s). It is a few lines of pandas on those files and should be folded into `aggregate_stage3_random_production.py` as a reported column.


The generation-1 first-passage propagation behind the population rows of the entry-side table (E30 = 0.292, n = 2810, and the distinct-grain next-interception rows) was run from a handoff folder that is not among the archived working directories; its results are kept verbatim in `entry_side_population/entry_ffsz_production_gen1.json`, `entry_ffsz_production_gen2.json` and `entry_ffsz_results.json`, and the analysis half of that calculation is `entry_side_population/entry_ffsz_focusing.py` (run with `--selftest` to check it). The same quantities can be regenerated from the paired study, whose kernel records `first_entry_theta` and `next_entry_theta` for every particle (`scripts/run_entry_ffsz_study.py`).
