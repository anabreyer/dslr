# ========= DSLR Makefile =========
# Usage examples (from project root):
#   make help
#   make dirs
#   make deps
#   make histogram SHOW=1 BINS=30
#   make scatter SHOW=1 RANK=10
#   make pairplot SHOW=1 FEATURES="Astronomy,Defense Against the Dark Arts,Herbology"
#   make train ALPHA=0.1 MAX_ITER=6000 LAMBDA=0.001 FEATURES="Astronomy,Defense Against the Dark Arts,Herbology,Charms,Transfiguration"
#   make predict PROBA_OUT=outputs/probas.csv
#
# Variables you can override on the command line are documented per target.

# -------- Paths --------
PY          ?= python3
SRC_DIR     ?= src
DATA_DIR    ?= data
OUT_DIR     ?= outputs
FIG_DIR     ?= $(OUT_DIR)/figures
MODEL_DIR   ?= models

TRAIN_CSV   ?= $(DATA_DIR)/dataset_train.csv
TEST_CSV    ?= $(DATA_DIR)/dataset_test.csv
MODEL_JSON  ?= $(MODEL_DIR)/logreg_model.json
PRED_CSV    ?= $(OUT_DIR)/houses.csv

# -------- Defaults (overridable) --------
# histogram
BINS        ?= 20
SHOW        ?= 0                  # pass SHOW=1 to open windows
MULTI       ?= 0                  # for histogram: SHOW=1 MULTI=1 -> legacy multi-windows

# scatter
RANK        ?= 0                  # e.g., RANK=10 to print top-10 pairs
X           ?=
Y           ?=

# pairplot
FEATURES    ?=                    # comma-separated list; if empty, auto-picks by variance
MAXF        ?= 6                  # max auto-selected features

# training
ALPHA       ?= 0.05
MAX_ITER    ?= 4000
LAMBDA      ?= 0.0
NO_STD      ?= 0                  # set NO_STD=1 to disable standardization (not recommended)

# prediction
PROBA_OUT   ?=                    # e.g., outputs/probas.csv

# -------- Derived flags --------
ifeq ($(SHOW),1)
SHOW_FLAG := --show
else
SHOW_FLAG :=
endif

ifeq ($(MULTI),1)
MULTI_FLAG := --multi-windows
else
MULTI_FLAG :=
endif

ifeq ($(strip $(FEATURES)),)
FEATURES_FLAG :=
else
FEATURES_FLAG := --features "$(FEATURES)"
endif

ifeq ($(strip $(X)$(Y)),)
XY_FLAG :=
else
XY_FLAG := --x "$(X)" --y "$(Y)"
endif

ifeq ($(strip $(RANK)),)
RANK_FLAG :=
else
RANK_FLAG := --rank $(RANK)
endif

ifeq ($(NO_STD),1)
NOSTD_FLAG := --no-standardize
else
NOSTD_FLAG :=
endif

ifeq ($(strip $(PROBA_OUT)),)
PROBA_FLAG :=
else
PROBA_FLAG := --proba-out "$(PROBA_OUT)"
endif

# -------- Phony --------
.PHONY: help dirs deps clean veryclean \
        histogram scatter pairplot \
        train predict

help:
	@echo ""
	@echo "DSLR Makefile — targets:"
	@echo "  make dirs                   # create outputs/ and models/ folders"
	@echo "  make deps                   # install python deps (matplotlib)"
	@echo ""
	@echo "  make histogram [SHOW=1] [BINS=30] [MULTI=1]"
	@echo "  make scatter [SHOW=1] [RANK=10] [X='Astronomy'] [Y='Defense Against the Dark Arts']"
	@echo "  make pairplot [SHOW=1] [FEATURES='A,B,C'] [MAXF=6]"
	@echo ""
	@echo "  make train [ALPHA=0.05] [MAX_ITER=4000] [LAMBDA=0.0] [NO_STD=0] [FEATURES='A,B,...']"
	@echo "  make predict [PROBA_OUT=outputs/probas.csv]"
	@echo ""
	@echo "Paths:"
	@echo "  TRAIN_CSV=$(TRAIN_CSV)"
	@echo "  TEST_CSV=$(TEST_CSV)"
	@echo "  MODEL_JSON=$(MODEL_JSON)"
	@echo "  PRED_CSV=$(PRED_CSV)"
	@echo ""

dirs:
	@mkdir -p "$(FIG_DIR)" "$(MODEL_DIR)"

deps:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install matplotlib

# -------- Visualization --------

histogram: dirs
	$(PY) $(SRC_DIR)/histogram.py "$(TRAIN_CSV)" --bins $(BINS) --outdir "$(FIG_DIR)" $(SHOW_FLAG) $(MULTI_FLAG)

scatter: dirs
	$(PY) $(SRC_DIR)/scatter_plot.py "$(TRAIN_CSV)" --outdir "$(FIG_DIR)" $(SHOW_FLAG) $(RANK_FLAG) $(XY_FLAG)

pairplot: dirs
	$(PY) $(SRC_DIR)/pair_plot.py "$(TRAIN_CSV)" --outdir "$(FIG_DIR)" $(SHOW_FLAG) $(FEATURES_FLAG) --max $(MAXF)

# -------- Model --------

train: dirs
	@mkdir -p "$(MODEL_DIR)"
	$(PY) $(SRC_DIR)/logreg_train.py "$(TRAIN_CSV)" $(FEATURES_FLAG) \
	    --alpha $(ALPHA) --max_iter $(MAX_ITER) --lambda $(LAMBDA) $(NOSTD_FLAG) \
	    --model "$(MODEL_JSON)"

predict: dirs
	$(PY) $(SRC_DIR)/logreg_predict.py "$(TEST_CSV)" --model "$(MODEL_JSON)" --out "$(PRED_CSV)" $(PROBA_FLAG)

# -------- Cleanup --------

clean:
	@echo "Removing generated figures and predictions..."
	@rm -f $(FIG_DIR)/*.png
	@rm -f $(OUT_DIR)/houses.csv
	@rm -f $(OUT_DIR)/probas.csv

veryclean: clean
	@echo "Removing saved models..."
	@rm -f $(MODEL_DIR)/*.json
