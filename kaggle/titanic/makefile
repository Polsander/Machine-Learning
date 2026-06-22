
LOAD_DATA_PATH = ./data_processing.py
UNSUPERVISED_PATH = ./unsupervised.py
SUPERVISED_PATH = ./supervised.py
TRAIN_PATH = ./train_model.py

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

## run_supervised: Run supervised training methods
run_supervised:
	@echo "Running supervised learning script"
	@bash -c "python3 $(SUPERVISED_PATH)"

## run_train: Run the training model script and save the csv submission locally for contest submission
run_train:
	@echo "Running training script"
	@bash -c "python3 $(TRAIN_PATH)"