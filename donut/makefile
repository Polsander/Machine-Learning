INFERENCE_TEST_PATH_DONUT = ./donut_model/inference_test.py
INFERENCE_TEST_PATH_QWEN = ./qwen_model/inference_test.py

.PHONY: help
.PHONY: run-inference_test_donut
.PHONY: run-inference_test_qwen

help:
# 	@echo "Usage: make [target]"
	@echo ""
	@echo "Make commands:"
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/## //g' | awk -F ': ' '{printf "  %-25s %s\n", $$1, $$2}'



## run-inference_test_donut: Infer the model and test with what is defined in the inference script
run-inference_test_donut:
	@echo "Launching python script..."
	@bash -c "python $(INFERENCE_TEST_PATH_DONUT)"

## run-inference_test_qwen: Infer the model and test with what is defined in the inference script
run-inference_test_qwen:
	@echo "Launching python script..."
	@bash -c "python $(INFERENCE_TEST_PATH_QWEN)"