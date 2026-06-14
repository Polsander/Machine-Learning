
LOAD_DATA_PATH = ./data_processing.py
UNSUPERVISED_PATH = ./unsupervised.py

.PHONY: load_data

help:
# 	@echo "Usage: make [target]"
	@echo ""
	@echo "Make commands:"
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/## //g' | awk -F ': ' '{printf "  %-25s %s\n", $$1, $$2}'

## load_data: Load and process the data and save to matrix
load_data:
	@echo "Loading in csv data..."
	@bash -c "python3 $(LOAD_DATA_PATH)"

## run_unsupervised: Run unsupervised training methods and data visualization
run_unsupervised:
	@echo "Running unsupervised learning script"
	@bash -c "python3 $(UNSUPERVISED_PATH)"