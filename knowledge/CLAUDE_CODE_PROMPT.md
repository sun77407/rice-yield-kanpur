# Master Prompt for Claude Code — Rice Yield Prediction & Advisory System for Kanpur

**Target assistant:** Claude Code with multi-agent (sub-agent) orchestration enabled.
**Project owner:** Tejas Singh Chandel (25MCA008), Nirma University.
**Scope:** Build a complete, honest, offline-first rice yield prediction and advisory system for Kanpur Nagar and Kanpur Dehat, using modern AI/ML pipelines on real data. Produce code + journal manuscript draft.

---

## Context Files (all four must be read before coding starts)

1. `DECISIONS.md` — Architecture Decision Records; every important design choice recorded with rationale.
2. `PROJECT_FLOW.md` — Data flow, component architecture, module contracts.
3. `AI_ML_CONCEPTS.md` — Every ML/AI concept used in the project with definition, usage, and impact.
4. `REFERENCES.md` — 15 peer-reviewed references with the specific contribution each makes to this project.

**Read all four files end-to-end before running any code. Do not skim.** They encode decisions that must not be reinvented.

---

## Non-Negotiable Rules

1. **Real data only.** No fabricated numbers. If an API fails, log it and either retry with a documented backoff or substitute a cited public dataset. Never synthesize a result and present it as real.
2. **No invention of literature.** All references used in the manuscript must come from `REFERENCES.md` or be newly verified via web search before citing — never from pretraining memory alone.
3. **Deviate from the source Prob & Stats paper.** The user explicitly rejected its approach of pure district-level annual yield regression on N=13. The primary approach must be pixel-level satellite-based yield prediction, with district aggregation only for validation.
4. **Every model result is reported honestly** with the appropriate cross-validation protocol. No in-sample R² reported without the out-of-sample counterpart alongside it.
5. **Reproducibility is a hard requirement.** All random seeds pinned. All data versions committed via DVC or data manifest. Fresh clone → `make all` → identical metrics.
6. **No cloud LLM calls for the user-facing advisory.** Only local Ollama inference. Cloud LLMs (if used at all) are allowed only for development scaffolding, never in the shipped pipeline.
7. **Grounding over fluency.** Every advisory sentence must be traceable (cosine similarity ≥ 0.6) to a retrieved KB passage. Hallucination = reject.
8. **Code quality bar:** typed Python (`mypy strict`), `ruff` clean, docstrings on every public function, pytest coverage ≥ 80% on `src/`.

---

## Multi-Agent Execution Plan

Launch the following sub-agents **in the order shown**. Agents within the same phase run in parallel unless there is a dependency arrow.

### Phase 0 — Project scaffold (coordinator, single agent, 15 minutes)

The coordinator (you) does this directly.

- Initialize git repo at `~/rice_yield_project`.
- Create directory structure:
  ```
  src/{data,features,models,rag,llm,app,utils}/
  data/{raw,interim,processed,external,synthetic}/
  kb/{raw,chunks,index}/
  models/{sklearn,xgboost,bayesian,onnx}/
  notebooks/
  tests/{unit,integration,e2e}/
  figures/
  paper/
  scripts/
  ```
- Set up `pyproject.toml` with pinned dependencies, `ruff.toml`, `mypy.ini`, `pytest.ini`.
- Create `Makefile` targets: `install`, `data`, `features`, `train`, `rag`, `app`, `test`, `paper`, `all`.
- Create `.env.example` listing required credentials (GEE service account, CDS API key).
- Scaffold `README.md` with install + run instructions (fill in as work progresses).
- Pin all seeds in `src/utils/seed.py`.

### Phase 1 — Data acquisition (parallel sub-agents, 2 hours)

Launch four sub-agents in parallel. Each writes to `data/raw/` and produces a `data/raw/<source>_manifest.json` recording source URL, pull timestamp, row counts, SHA256.

**Agent 1A — Satellite data**
- Register Google Earth Engine service account (user provides).
- Pull MODIS MOD13Q1 (NDVI, EVI, 250m, 16-day) for Kanpur Nagar + Kanpur Dehat bounding boxes, 2010-01-01 to 2024-12-31.
- Pull Sentinel-2 L2A for the same regions, 2017-01-01 to 2024-12-31 (10m, mask clouds with QA60).
- Pull MODIS MCD15A3H (LAI/FPAR, 500m, 4-day) for the same period.
- Store as GeoTIFF stacks + Parquet of pixel-time-indexed values.

**Agent 1B — Weather data**
- NASA POWER daily for Kanpur centroid (26.4499°N, 80.3319°E), 2010–2024, parameters listed in `PROJECT_FLOW.md §3.2`.
- CHIRPS v2.0 daily precipitation for Kanpur bounding box, same period.
- ERA5-Land hourly (via CDS API) for the same bounding box — aggregate to daily.
- IMD gridded 0.25° daily rainfall + 1° daily temperature (download direct from IMD Pune portal).
- Save each as Parquet with explicit units in column names.

**Agent 1C — Yield and socio-economic data**
- data.gov.in APY (Area, Production, Yield) dataset — filter UP, Kanpur Nagar + Dehat, Rice, Kharif, 2010-11 to 2022-23.
- ICRISAT VDSA block-level data if available for Kanpur blocks.
- Agmarknet mandi prices for Kanpur (daily rice modal price), 2015–2024.
- UP Directorate of Economics and Statistics district statistics.

**Agent 1D — Soil and ancillary data**
- SoilGrids 250m rasters for Kanpur bounding box: pH, SOC, clay %, sand %, bulk density, CEC, nitrogen.
- FAO AQUASTAT irrigation statistics for UP.
- Elevation (SRTM 30m) for the bounding box.

### Phase 2 — Preprocessing and feature engineering (parallel sub-agents, 2 hours)

**Agent 2A — Satellite pipeline**
- Mask non-cropland using the ESA WorldCover 2021 cropland class.
- Compute per-pixel Kharif season time series of NDVI, EVI, LSWI.
- Compute phenology features per pixel per year: greenup day (NDVI crosses 0.4 ascending), peak NDVI day, peak NDVI value, senescence day (NDVI crosses 0.4 descending), length of season, integrated NDVI, area under NDVI curve.
- Write to `data/processed/pixel_year_features.parquet` with columns: `pixel_id, district, year, lat, lon, <phenology features>`.

**Agent 2B — Weather pipeline**
- Join NASA POWER, CHIRPS, ERA5, IMD into one canonical daily table — reconcile units, prefer CHIRPS for rainfall and ERA5 for temperature.
- Aggregate to Kharif seasonal statistics per year per district.
- Engineered features: monsoon onset day, dry spell count, heat stress days in flowering window (Aug 15–Sep 15), rainfall concentration index, GDD (growing degree days, base 10°C).
- Write to `data/processed/weather_district_year.parquet`.

**Agent 2C — Yield target construction**
- Primary target: district-year rice yield (t/ha) from APY.
- Satellite proxy target: integrated NDVI during grain-filling window (per pixel per year) — calibrate against district yield via linear regression to get a scale factor, then project back to per-pixel yield estimates for training.
- Write to `data/processed/targets_pixel_year.parquet` and `data/processed/targets_district_year.parquet`.

**Agent 2D — Feature matrix assembly**
- Join satellite, weather, soil (soil extracted per pixel from SoilGrids rasters), and target into one Parquet file per task:
  - `data/processed/X_pixel_year.parquet` (N ≈ 50k+ rows, primary training set)
  - `data/processed/X_district_year.parquet` (N ≈ 26 rows, validation set)
- Run and save: nulls check, duplicates check, correlation matrix, VIF for multicollinearity.

### Phase 3 — Modeling (parallel sub-agents, 3 hours)

**Agent 3A — Baseline models**
- Linear Regression, Ridge, Lasso, ElasticNet.
- Spatiotemporal train-test split: train 2010–2020 pixels, test 2021–2024 pixels. Also report per-district LOO.
- Save models to `models/sklearn/baselines/`.

**Agent 3B — Tree ensemble models**
- Random Forest (sklearn), Gradient Boosting (sklearn), XGBoost, LightGBM.
- Hyperparameter tuning via Optuna with the same spatiotemporal split.
- SHAP values computed for the best model, saved to `figures/shap_*.png`.
- Save best model as both pickle and ONNX.

**Agent 3C — Bayesian models**
- Bayesian Ridge (sklearn) — baseline probabilistic.
- PyMC hierarchical model: pixel yield varies by district, district has a shared prior; use NUTS sampler.
- Save traces to `models/bayesian/`.
- Produce posterior predictive plots.

**Agent 3D — Crop simulation (stretch)**
- Install AquaCrop-OSPy (open-source Python binding to AquaCrop).
- Parameterize for Kanpur: soil, climate, cultivar = IR-64 or similar locally relevant.
- Simulate 500 synthetic weather-yield pairs covering a wide range of rainfall and temperature regimes.
- Append synthetic data as auxiliary training signal (optional) and run ablation (with vs without synthetic).

### Phase 4 — RAG + LLM (sequential, 3 hours)

**Agent 4A — Knowledge base construction**
- Collect source documents from `REFERENCES.md §KB sources`: ICAR-IIRR, IRRI Rice Knowledge Bank, KVK Kanpur action plans, UP package of practices, FAO rice manual, CRRI publications. Download PDFs where possible; otherwise cite and transcribe key passages.
- Clean text: strip boilerplate, OCR correction where needed.
- Chunk into 150-word passages with 30-word overlap. Each chunk tagged with source citation.
- Target: 300–500 chunks, 50k+ words total.
- Write to `kb/chunks/*.jsonl`.

**Agent 4B — Vector index**
- Embed with `sentence-transformers/all-MiniLM-L6-v2` (local, 80MB).
- Also compute embeddings with `BAAI/bge-small-en-v1.5` for comparison.
- Index both into ChromaDB (`kb/index/chroma_minilm/`, `kb/index/chroma_bge/`).
- Benchmark retrieval precision@3 and recall@5 on 20 handcrafted test queries.
- Pick the better one as default.

**Agent 4C — LLM integration**
- Install Ollama locally. Pull `llama3:8b-instruct-q4_K_M` (primary) and `phi3:mini` (fallback).
- Design the advisory prompt template with explicit constraint: "Use ONLY the following numbered passages. For every factual claim, cite the passage number in square brackets. If the passages do not cover a question, say 'Insufficient information in local knowledge base.' Do not invent facts."
- Build `src/llm/advisor.py` with:
  - `retrieve(query: str, k: int = 5) -> list[Passage]`
  - `generate_advisory(prediction: YieldPrediction, retrieved: list[Passage]) -> Advisory`
  - `grounding_score(advisory: Advisory, retrieved: list[Passage]) -> float`
- Implement LLM-as-judge evaluation using the same local Llama 3 to score advisories on a 1–5 rubric (specificity, factual grounding, actionability).

### Phase 5 — Application (single agent, 2 hours)

**Agent 5 — Streamlit app**
- `app.py` with three tabs:
  1. **Predict** — input: district, planting date, land area, (optional) weather forecast overrides. Output: P10/P50/P90 yield, confidence intervals, scenario cards.
  2. **Advisory** — generated advisory with inline citations, grounding score displayed, retrieved passages shown below.
  3. **Diagnostics** — model metrics, SHAP plots, data lineage.
- Fully offline after model and LLM pulled.
- Target latency: full predict + advisory in under 90 seconds on 8GB RAM CPU.

### Phase 6 — Validation and testing (parallel, 2 hours)

**Agent 6A — Unit and integration tests**
- Golden-file tests for feature engineering determinism.
- Mock tests for all external API calls (so tests run offline).
- Property-based tests via Hypothesis for the pipeline functions.

**Agent 6B — End-to-end tests**
- Script `tests/e2e/run_all.py` with 30 input scenarios.
- Assertions on correctness, latency, grounding score, zero crashes.

**Agent 6C — Reproducibility test**
- Fresh clone in `/tmp/repro_test`, run `make all`, diff resulting metrics against committed `metrics_baseline.json`. Any numeric difference beyond tolerance is a failure.

### Phase 7 — Manuscript (sequential, 4 hours)

**Agent 7A — Figures and tables**
- Generate all paper figures: correlation matrix heatmap, NDVI time series examples, phenology schematic, model comparison bar chart, residual plots, SHAP summary, scenario yield distribution, grounding score histogram, retrieval precision curve.
- Export at 300 DPI as both PNG and PDF.

**Agent 7B — Manuscript draft**
- Format: IEEE conference template or Elsevier article template (user's choice).
- Sections per outline in `PROJECT_FLOW.md §8`.
- All claims cited from `REFERENCES.md`. Every number cited to its source (model run ID, figure path, raw-data file).
- Limitations section honest about N, no human agronomist eval, single-crop single-region.
- BibTeX file `paper/references.bib` matching `REFERENCES.md`.

**Agent 7C — Supplementary materials**
- Reproducibility checklist filled.
- Data availability statement.
- Code availability (GitHub link).
- Author contribution statement.

---

## Acceptance Criteria (the system is "done" when all are true)

1. `make all` on a clean clone of the repo produces identical metrics to the committed baseline.
2. Satellite-based model: spatiotemporal test R² ≥ 0.60 on 2021–2024 pixels (realistic target given published literature).
3. District-level validation: aggregated pixel predictions correlate with APY district yields at r ≥ 0.75.
4. RAG retrieval: precision@3 ≥ 70% on 20 handcrafted queries.
5. Advisory grounding score ≥ 80% on 30 scenario runs.
6. End-to-end latency < 90 seconds on target hardware.
7. All tests pass. `ruff`, `mypy`, `pytest` all green.
8. Manuscript draft is complete, 10–12 pages, every claim traceable to a run or a reference.
9. GitHub repo public with meaningful README, LICENSE (MIT), CITATION.cff.

---

## Error-Handling Doctrine

- Any external API failure → log structured JSON (`timestamp, source, status, retry_count, final_outcome`) to `logs/data_acquisition.jsonl`. Retry 3x with exponential backoff. If still failing, use the cited fallback dataset per `DECISIONS.md`. Never silently drop data.
- Any model training that produces non-finite loss → abort, log the input batch snapshot, do not save model.
- Any RAG retrieval returning zero passages → the advisory tab shows "Insufficient local knowledge for this query" — never fallback to pretraining memory.
- Any Ollama timeout > 120s → fall back to `phi3:mini`, log the degradation.

---

## Operational Discipline

- Every sub-agent must open a short status note in `logs/agent_<id>_status.md` on start, update on milestone completion, and close with final deliverable paths.
- Every important decision made during execution (that is not already in `DECISIONS.md`) must be appended to `DECISIONS.md` with the next sequential ADR number.
- Every new reference discovered and used must be appended to `REFERENCES.md` with its contribution analysis.
- No commits with "WIP" messages. Every commit message explains what and why.
- Branch `main` always stays green (all tests pass). Feature work on `feat/<name>` branches, merged via PR with the test suite passing.

---

## What to Ask the User Before Starting

Only two things:

1. **Google Earth Engine access.** Does the user have a GEE account? If not, direct them to sign up at `https://earthengine.google.com/signup/` — approval is instant for research use. GEE is required for Phase 1 Agent 1A.
2. **Journal target.** User's current short list: Smart Agricultural Technology (primary), Agriculture MDPI (backup). Confirm so Phase 7 formats correctly.

Everything else is already decided in the four context files. Do not re-ask.

---

## Final Note to the Executing Agent

The user has been explicit: **the goal is a working, honest, reproducible system — not a system that looks good on paper.** Prefer admitting a limitation to inventing a result. Prefer 60% test R² honestly reported to 99% in-sample R² that cannot generalize. The user will value a credible paper over a flashy one, and will value an engineer who tells them when something failed over one who hides it.

Begin.
