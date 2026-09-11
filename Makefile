# Figures regenerate from committed results - never hand-edit a figure.
PY := ./.venv/bin/python

.PHONY: all setup figures data baselines lora active clean help

all: setup figures

setup: .venv/bin/python  ## create the virtualenv and install dependencies
.venv/bin/python:
	python3 -m venv .venv
	./.venv/bin/pip install -q --upgrade pip
	./.venv/bin/pip install -q -r requirements.txt

figures:            ## regenerate every figure from results/ (fast, no GPU)
	$(PY) src/make_figures.py

data:               ## download GSE104878, parse to parquet, freeze splits (~1GB)
	$(PY) src/parse_raw.py
	$(PY) src/build_exclusions.py
	$(PY) src/build_splits.py

baselines:          ## ridge, LightGBM, CNN across all n x 3 seeds x 2 splits
	$(PY) src/run_baselines.py --model ridge
	$(PY) src/run_baselines.py --model lgbm
	$(PY) src/run_baselines.py --model cnn

lora:               ## DNABERT-2 and NT LoRA sweeps (needs a CUDA GPU)
	$(PY) src/run_lora.py --model dnabert
	$(PY) src/run_lora.py --model nt

active:             ## D2 selection curves
	$(PY) src/run_active.py

clean:              ## remove generated figures only; never touches frozen splits or results
	rm -f figures/*.png figures/*.pdf

help:
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'
