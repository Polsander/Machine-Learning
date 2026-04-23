# Any python script should be run from this make file if possible
R-SUPERVISED_LEARNING_PATH = ./src/regression/supervised_learning.py
R-UNSUPERVISED_LEARNING_PATH = ./src/regression/unsupervised_learning.py
C-SUPERVISED_LEARNING_PATH = ./src/classification/supervised_learning.py
C-UNSUPERVISED_LEARNING_PATH = ./src/classification/unsupervised_learning.py
DATA_PROCESSING_PATH = ./src/data_processing.py
FRAMES_PROCESSING_PATH = ./src/convert_frames.py
GENERATE_FEATURE_LABELS_PATH = ./src/generate_features_labels.py

# Variables
VENV = che-500-venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.PHONY: help
.PHONY: run-setup
.PHONY: run-regression-supervised_learning
.PHONY: run-regression-unsupervised_learning
.PHONY: run-classification-supervised_learning
.PHONY: run-classification-unsupervised_learning
.PHONY: run-data_processing
.PHONY: create-frames
.PHONY: generate_feature_labels

help:
# 	@echo "Usage: make [target]"
	@echo ""
	@echo "Make commands:"
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/## //g' | awk -F ': ' '{printf "  %-25s %s\n", $$1, $$2}'

## run-supervised: Launch the supervised learning regression script
run-regression-supervised_learning:
	@echo "Launching python script..."
	@bash -c "python3 $(R-SUPERVISED_LEARNING_PATH)"

## run-unsupervised: Launch the unsupervised learning regression script
run-regression-unsupervised_learning:
	@echo "Launching unsupervised learning..."
	@bash -c "python $(R-UNSUPERVISED_LEARNING_PATH)"

## run-supervised: Launch the supervised learning classification script
run-classification-supervised_learning:
	@echo "Launching python script..."
	@bash -c "python3 $(C-SUPERVISED_LEARNING_PATH)"

## run-unsupervised: Launch the unsupervised learning classification script
run-classification-unsupervised_learning:
	@echo "Launching unsupervised learning..."
	@bash -c "python $(C-UNSUPERVISED_LEARNING_PATH)"

## run-data_processing: Launch the data processing file
run-data_processing:
	@echo "Running data processing..."
	@bash -c "python $(DATA_PROCESSING_PATH)"

## create frames
create-frames:
	@echo "generating frames from mp4 video from data folder..."
	@bash -c "python $(FRAMES_PROCESSING_PATH)"

## generate feature labels
generate_feature_labels:
	@echo "generating feature labels for data ...."
	@bash -c "python $(GENERATE_FEATURE_LABELS_PATH)"

## setup: Installs your packages and sets you up with a virtual environment you can enable.
setup:
	@echo "Make sure you have python 3.10 version installed!!!"
	@echo "Checking for virtual environment..."
	@test -d $(VENV) || python3.10 -m venv $(VENV)
	@echo "Installing package in editable mode..."
	@$(PIP) install --upgrade pip
	@$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu	
	@$(PIP) install -e .
	@echo ""
	@echo "✅ Setup complete."
	@echo "Run 'source $(VENV)/bin/activate' to enable the environment in your terminal."
	@echo "You need to run this command prior to running any makefile scripts"