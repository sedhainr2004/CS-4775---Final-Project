SHELL := /bin/bash
CONFIG ?= configs/paul15.yaml
VENV := .venv

.PHONY: preprocess graph splits pipeline train-scgnn train-scgnn-paul15 train-scgnn-pbmc

preprocess:
	@source $(VENV)/bin/activate && python preprocess.py --config $(CONFIG)

graph:
	@source $(VENV)/bin/activate && python build_graph.py --config $(CONFIG)

splits:
	@source $(VENV)/bin/activate && python split_data.py --config $(CONFIG)

pipeline: preprocess graph splits

train-scgnn:
	@source $(VENV)/bin/activate && python train_scgnn.py --config $(SCGNN_CONFIG)

train-scgnn-paul15:
	@$(MAKE) train-scgnn SCGNN_CONFIG=configs/scgnn_paul15.yaml

train-scgnn-pbmc:
	@$(MAKE) train-scgnn SCGNN_CONFIG=configs/scgnn_pbmc.yaml

evaluate-scgnn:
	@source $(VENV)/bin/activate && python evaluate_scgnn.py --config $(SCGNN_CONFIG) --results_dir $(RESULTS_DIR)

plot-embeddings:
	@source $(VENV)/bin/activate && python plot_embeddings.py --config $(SCGNN_CONFIG) --results_dir $(RESULTS_DIR) --output_path $(OUTPUT_PATH) --method $(METHOD)

summarize-results:
	@source $(VENV)/bin/activate && python summarize_results.py --results_root results --output_csv reports/metrics_summary.csv
