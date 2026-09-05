INFERENCE_TEST_PATH_DONUT = ./donut_model/inference_test.py
INFERENCE_TEST_PATH_QWEN = ./qwen_model/inference_test.py

LABEL_APP_PATH = ./image_labeler/app.py

.PHONY: help
.PHONY: run-inference_test_donut
.PHONY: run-inference_test_qwen

.PHONY: run-labelling

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

## run-labelling: launch GUI to help lavel images defined in the configurations
run-labelling:
	@echo "Launching python labelling app..."
	@bash -c "python ${LABEL_APP_PATH}"