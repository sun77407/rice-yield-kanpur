# Rice Yield Kanpur — pipeline entry points. Requires GNU make + uv.
# Windows: install make via `winget install ezwinports.make` or `choco install make`.
PY := uv run python
SEED ?= 42

.PHONY: install data features train rag app test lint paper all clean

install:            ## create venv and install locked deps
	uv sync --all-groups

data:               ## Phase 1: acquire raw data (GEE steps are quota-guarded, see logs/gee_quota.jsonl)
	$(PY) -m src.data.run_all

features:           ## Phase 2: build feature matrices and targets
	$(PY) -m src.features.run_all

train:              ## Phase 3: train baselines, tree ensembles, Bayesian models
	$(PY) -m src.models.run_all

rag:                ## Phase 4: build KB chunks and the nomic-embed-text Chroma index
	$(PY) -m src.rag.build_index

app:                ## Phase 5: launch Streamlit on localhost:8501
	uv run streamlit run src/app/app.py --server.address 127.0.0.1

test:               ## ruff + mypy --strict + pytest (coverage >= 80%)
	uv run ruff check .
	uv run mypy
	uv run pytest

paper:              ## Phase 7: regenerate figures/tables, build manuscript
	$(PY) -m src.app.paper_assets
	cd paper && latexmk -pdf main.tex

all: install data features train rag test paper

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
