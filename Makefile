CONFIG ?= config/production.json
RANDOM_CONFIG ?= config/random_stage3.json
ENTRY_CONFIG ?= config/entry_ffsz.json
DATA_ROOT ?= .
PY ?= python3

.PHONY: kernel test run run-full validate transition-matrices random-production random-postprocess random-shadow random-validate next-interception entry-validate entry-confirmatory entry-analyze entry-timestep tables figures assemble check

kernel:
	$(PY) scripts/build_particle_kernel.py

test:
	$(PY) scripts/kernel_regression_tests.py --config $(CONFIG) --data-root $(DATA_ROOT)

run:
	$(PY) scripts/production_pipeline.py run --config $(CONFIG)

run-full:
	$(PY) scripts/production_pipeline.py run-full --config $(CONFIG) --reuse

validate:
	$(PY) scripts/production_pipeline.py validate --config $(CONFIG)

transition-matrices:
	$(PY) scripts/run_focusing_transition_matrices.py

random-production:
	$(PY) scripts/run_random_stage3_pipeline.py production --config $(RANDOM_CONFIG)

random-postprocess:
	$(PY) scripts/run_random_stage3_pipeline.py postprocess --config $(RANDOM_CONFIG)

random-shadow:
	$(PY) scripts/run_random_stage3_pipeline.py shadow --config $(RANDOM_CONFIG)

random-validate:
	$(PY) scripts/run_random_stage3_pipeline.py validate --config $(RANDOM_CONFIG)

next-interception:
	$(PY) scripts/analyze_release_to_next_interception.py

entry-validate:
	$(PY) scripts/validate_entry_ffsz_tracking.py

entry-confirmatory:
	$(PY) scripts/run_entry_ffsz_study.py --config $(ENTRY_CONFIG) --stage confirmatory --domains both --resume

entry-analyze:
	$(PY) scripts/analyze_entry_ffsz_study.py --config $(ENTRY_CONFIG) --stage confirmatory

entry-timestep:
	$(PY) scripts/run_entry_ffsz_timestep_convergence.py --config $(ENTRY_CONFIG)

# post-processing that reads existing outputs only
tables:
	$(PY) scripts/create_focused_mobile_state_metrics.py
	$(PY) scripts/create_mechanism_revision_tables.py
	$(PY) scripts/create_submission_readiness_tables.py
	$(PY) scripts/compute_f_alpha.py
	$(PY) scripts/compute_predictive_lambda.py

figures: tables
	$(PY) scripts/create_focused_geometry_flow_figure.py
	$(PY) scripts/create_release_angle_schematic.py
	$(PY) scripts/create_focused_results_figures.py
	$(PY) scripts/create_center_well_survival_curves.py
	$(PY) scripts/create_science_synthesis_figures.py
	$(PY) scripts/create_random_geometry_flow_stagnation_figure.py
	$(PY) scripts/create_stage3_random_mechanism_figure.py
	$(PY) scripts/create_random_geometry_manuscript_figures.py
	$(PY) scripts/validate_near_wall_boltzmann.py
	$(PY) scripts/create_entry_side_figure.py

assemble: figures
	$(PY) scripts/assemble_manuscript_figures.py

# compare generated figures with a manuscript folder: make check MANUSCRIPT=/path/to/folder
check:
	$(PY) scripts/assemble_manuscript_figures.py --check $(MANUSCRIPT)
