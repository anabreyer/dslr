# ========= DSLR Makefile (simplified) =========
# Usage:
#   make help
#   make describe
#   make histogram
#   make scatter
#   make pair
#   make train
#   make predict
#   make test
#   make fclean

# -------- Paths --------
PY        ?= python3
SRC_DIR   ?= src
DATA_DIR  ?= data
OUT_DIR   ?= outputs
MODEL_DIR ?= models

TRAIN_CSV ?= $(DATA_DIR)/dataset_train.csv
TEST_CSV  ?= $(DATA_DIR)/dataset_test.csv

# -------- Phony --------
.PHONY: help dirs describe histogram scatter pair train predict test clean fclean

help:
	@echo ""
	@echo "DSLR Makefile — commands"
	@echo "  make describe    -> $(PY) $(SRC_DIR)/describe.py $(TRAIN_CSV)"
	@echo "  make histogram   -> $(PY) $(SRC_DIR)/histogram.py $(TRAIN_CSV) --show"
	@echo "  make scatter     -> $(PY) $(SRC_DIR)/scatter_plot.py $(TRAIN_CSV) --show"
	@echo "  make pair        -> $(PY) $(SRC_DIR)/pair_plot.py $(TRAIN_CSV) --show"
	@echo "  make train       -> $(PY) $(SRC_DIR)/logreg_train.py $(TRAIN_CSV)"
	@echo "  make predict     -> $(PY) $(SRC_DIR)/logreg_predict.py $(TEST_CSV)"
	@echo "  make test        -> $(PY) $(SRC_DIR)/test.py"
	@echo "  make fclean      -> delete ALL files in '$(OUT_DIR)' and '$(MODEL_DIR)'"
	@echo ""

dirs:
	@mkdir -p "$(OUT_DIR)" "$(OUT_DIR)/figures" "$(MODEL_DIR)"

# -------- Exact commands you requested --------
describe: dirs
	$(PY) $(SRC_DIR)/describe.py $(TRAIN_CSV)

histogram: dirs
	$(PY) $(SRC_DIR)/histogram.py $(TRAIN_CSV) --show

scatter: dirs
	$(PY) $(SRC_DIR)/scatter_plot.py $(TRAIN_CSV) --show

pair: dirs
	$(PY) $(SRC_DIR)/pair_plot.py $(TRAIN_CSV) --show

train: dirs
	$(PY) $(SRC_DIR)/logreg_train.py $(TRAIN_CSV)

predict: dirs
	$(PY) $(SRC_DIR)/logreg_predict.py $(TEST_CSV)

test: dirs
	$(PY) $(SRC_DIR)/test.py

# -------- Cleanup --------
# 'fclean' should delete all files inside outputs/ and models/
# but keep the directories themselves for convenience.
fclean:
	@echo "Removing all generated files in '$(OUT_DIR)' and '$(MODEL_DIR)'..."
	@rm -rf $(OUT_DIR)/* $(MODEL_DIR)/* || true
	@mkdir -p "$(OUT_DIR)" "$(OUT_DIR)/figures" "$(MODEL_DIR)"
	@echo "Done."
