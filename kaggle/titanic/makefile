
LOAD_DATA_PATH = ./data_processing.py

.PHONY: load_data

help:
# 	@echo "Usage: make [target]"
	@echo ""
	@echo "Make commands:"
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/## //g' | awk -F ': ' '{printf "  %-25s %s\n", $$1, $$2}'

## load_data: Launch the supervised learning regression script
load_data:
	@echo "Loading in csv data..."
	@bash -c "python3 $(LOAD_DATA_PATH)"