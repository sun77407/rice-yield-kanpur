# Project Flow — Data, Components, and Interfaces

This document is the single source of truth for how the system is wired. Every module must honor the contracts described here.

---

## 1. System Overview (One Picture)

```
                    ┌───────────────────────────────────────────┐
                    │          EXTERNAL DATA SOURCES            │
                    │                                           │
                    │  Google Earth Engine ─ MODIS, Sentinel-2  │
                    │  NASA POWER ─ daily weather point         │
                    │  CHIRPS v2.0 ─ daily rainfall grid        │
                    │  Copernicus CDS ─ ERA5-Land hourly        │
                    │  IMD Pune ─ gridded rainfall & temp       │
                    │  data.gov.in ─ APY yield statistics       │
                    │  ICRISAT VDSA ─ block-level yields        │
                    │  ISRIC SoilGrids ─ soil rasters 250m      │
                    │  Agmarknet ─ mandi prices                 │
                    │  IRRI/ICAR/KVK PDFs ─ RAG corpus          │
                    └───────────────────────┬───────────────────┘
                                            │
                                            ▼
          ┌─────────────────────────────────────────────────────────┐
          │                data/raw/ (immutable)                    │
          │       + manifest.json  (source, time, hash)             │
          └─────────────────────────────────────────────────────────┘
                                            │
                   ┌────────────────────────┼────────────────────────┐
                   ▼                        ▼                        ▼
         ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
         │ src/data/        │   │ src/data/        │   │ src/data/        │
         │ satellite.py     │   │ weather.py       │   │ ancillary.py     │
         │ (GEE pulls, EE)  │   │ (API fetches)    │   │ (soil, prices)   │
         └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
                  │                      │                      │
                  ▼                      ▼                      ▼
            data/interim/*.parquet (per-source, cleaned)
                                        │
                                        ▼
         ┌───────────────────────────────────────────────────────┐
         │               src/features/                           │
         │  phenology.py │ weather_seasonal.py │ soil_perpixel.py│
         │                                                       │
         │  Produces: pixel-year feature matrix + district-year  │
         └───────────────────────────────┬───────────────────────┘
                                         ▼
                     data/processed/X_pixel_year.parquet
                     data/processed/X_district_year.parquet
                     data/processed/targets_*.parquet
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   ▼                     ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │ src/models/      │  │ src/models/      │  │ src/models/      │
         │ baselines.py     │  │ xgb_lgb.py       │  │ bayesian.py      │
         │                  │  │                  │  │ (PyMC + bayesian │
         │ Linear/Ridge/    │  │ XGBoost +        │  │  ridge)          │
         │ Lasso/ElasticNet │  │ LightGBM + GBR   │  │                  │
         └──────────────────┘  └──────────────────┘  └──────────────────┘
                                         │
                                         ▼
                       models/*.pkl, models/*.onnx, models/traces/
                                         │
                                         ▼
                      ┌──────────────────────────────┐
                      │ src/models/predict.py        │
                      │ (P10/P50/P90 scenarios +     │
                      │  Bayesian HDI + bootstrap CI)│
                      └──────────────┬───────────────┘
                                     │
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ kb/raw/*.pdf     │      │ src/rag/         │      │ src/llm/         │
│   (sources)      │────▶ │  index.py        │────▶ │  advisor.py      │
└──────────────────┘      │  retriever.py    │      │  (Ollama client) │
                          └──────────────────┘      └──────────────────┘
                                     │                         │
                                     ▼                         ▼
                           kb/chunks/*.jsonl         src/llm/grounding.py
                           kb/index/chroma_*/
                                                               │
                                     ┌─────────────────────────┘
                                     ▼
                          ┌──────────────────────┐
                          │     app.py           │
                          │     (Streamlit)      │
                          │                      │
                          │  Predict | Advisory  │
                          │     | Diagnostics    │
                          └──────────────────────┘
                                     │
                                     ▼
                                 User
```

---

## 2. Data Flow — Step by Step

### 2.1 Satellite ingestion (daily, per-pixel)
1. GEE service account authenticates (`.env`).
2. For each year 2010–2024, pull MOD13Q1 16-day NDVI composites filtered to Kharif window (DOY 150–300 ≈ June–Oct).
3. Mask by ESA WorldCover 2021 cropland (class 40).
4. Export per-pixel time series as Parquet: `data/interim/modis_ndvi_pixel.parquet`.
5. Columns: `pixel_id, lat, lon, district, year, doy, ndvi, evi, qa_pass`.

### 2.2 Weather fusion
1. Pull NASA POWER, CHIRPS, ERA5-Land, IMD gridded.
2. Spatially subset each to Kanpur bounding box.
3. Temporally align to daily UTC.
4. Fusion rule: rainfall = CHIRPS (fallback IMD if CHIRPS NaN), temperature = ERA5-Land (fallback IMD), humidity = ERA5-Land, soil moisture = SMAP where available else GWETROOT (NASA POWER).
5. Write `data/interim/weather_daily_canonical.parquet`.

### 2.3 Feature engineering
1. **Phenology features (per pixel per year):**
   - `greenup_day` — first DOY in Kharif when NDVI crosses 0.4 ascending.
   - `peak_ndvi_day` — DOY of max NDVI.
   - `peak_ndvi` — max NDVI value.
   - `senescence_day` — first DOY after peak when NDVI crosses 0.4 descending.
   - `length_of_season` — senescence_day − greenup_day.
   - `integrated_ndvi` — area under NDVI curve during Kharif.
   - `grain_filling_ndvi` — mean NDVI in Aug 15–Sep 15 window.
2. **Weather features (per district per year):**
   - `kharif_rainfall_mm` — sum of daily rainfall June 1–Oct 31.
   - `monsoon_onset_day` — first day with ≥5mm rain in June.
   - `dry_spell_count` — count of consecutive 7+ day no-rain windows during Kharif.
   - `kharif_tmax_mean`, `kharif_tmin_mean`.
   - `heat_stress_days` — count of days with Tmax > 35°C in Aug 15–Sep 15.
   - `kharif_rh_mean`.
   - `rain_cv` — coefficient of variation of monthly rainfall.
   - `gdd_kharif` — growing degree days (base 10°C) accumulated June–Oct.
3. **Soil features (per pixel):** `ph`, `soc`, `clay_pct`, `sand_pct`, `cec`, `nitrogen` — extracted from SoilGrids rasters, constant across years (soil changes slowly).
4. **District one-hot** encoding.

### 2.4 Target construction
- **District-year target (26 rows):** APY yield in t/ha.
- **Pixel-year target (50k+ rows):** calibrated NDVI → yield.
  - For each district, fit `apy_yield_year = α + β × mean_district_grain_filling_ndvi_year` on 13 years. Record α, β.
  - For each pixel-year: `pixel_yield_estimate = α_district + β_district × pixel_grain_filling_ndvi`.
  - These calibrated per-pixel yields become the training targets.

### 2.5 Train/test split
- **Train:** all pixels, years 2010–2020 (≈ 70% of years).
- **Test:** all pixels, years 2021–2024 (≈ 30% of years).
- No pixel-year appears in both. No same-year appears in both.

### 2.6 Model training
- Linear, Ridge, Lasso, ElasticNet → simple fit, no HP tuning.
- RF, sklearn GBR, XGBoost, LightGBM → Optuna HP tuning on the train split with 5-fold cross-validation within train (never touching test).
- Bayesian Ridge → fit on train.
- PyMC hierarchical: `yield ~ Normal(mu_district + β × features, sigma)` with district-level priors. NUTS 2 chains × 2000 samples.

### 2.7 Prediction and uncertainty
- **Point prediction:** XGBoost `.predict(X)` → yhat.
- **Scenario prediction (P10/P50/P90):**
  - Compute P10/P50/P90 of each weather feature across historical years.
  - Build three input vectors (bad/avg/good year weather + current soil + current phenology).
  - Predict yhat for each.
- **Confidence interval per scenario:**
  - Bayesian 95% HDI from PyMC posterior predictive.
  - Bootstrap 95% CI from 1000 XGBoost resamples.
- **Grounding score for advisory:** cosine similarity of each advisory sentence to retrieved passages, averaged.

### 2.8 Advisory generation
1. User inputs district + planting date + land area in Streamlit.
2. Predict yields and scenarios (above).
3. Build query string: "Rice advisory for {district}, planting {date}, expected {P50_yield} t/ha. What should the farmer do about fertilizer, irrigation, pest watch?"
4. Retriever returns top-5 passages from ChromaDB.
5. Build prompt from passages (numbered 1–5) + strict instruction (cite every claim).
6. Call Ollama `llama3:8b-instruct-q4_K_M` with temperature 0.2.
7. Parse response, split into sentences.
8. For each sentence, compute max cosine similarity to the 5 passages. Average = grounding score.
9. Display: advisory text with inline citations, grounding score, retrieved passages below.

---

## 3. Module Contracts

### 3.1 `src/data/satellite.py`
```python
def fetch_modis_ndvi(
    bbox: BoundingBox,
    start: date,
    end: date,
    out_path: Path,
) -> DataManifest: ...

def fetch_sentinel2(
    bbox: BoundingBox,
    start: date,
    end: date,
    out_path: Path,
    cloud_mask: bool = True,
) -> DataManifest: ...
```

### 3.2 `src/data/weather.py`
```python
def fetch_nasa_power(
    lat: float, lon: float,
    start: date, end: date,
    params: list[str],
    out_path: Path,
) -> DataManifest: ...

def fetch_chirps(bbox, start, end, out_path) -> DataManifest: ...
def fetch_era5_land(bbox, start, end, variables, out_path) -> DataManifest: ...
def fetch_imd_gridded(variable, start, end, out_path) -> DataManifest: ...

def fuse_weather(sources: list[Path]) -> pd.DataFrame: ...
```

### 3.3 `src/features/phenology.py`
```python
def extract_phenology(
    ndvi_timeseries: pd.DataFrame,  # columns: pixel_id, doy, ndvi
    threshold: float = 0.4,
) -> pd.DataFrame:
    """Returns one row per pixel-year with all phenology features."""
```

### 3.4 `src/features/weather_seasonal.py`
```python
def aggregate_kharif(
    daily: pd.DataFrame,
    start_doy: int = 152,  # June 1
    end_doy: int = 304,    # October 31
) -> pd.DataFrame: ...
```

### 3.5 `src/models/predict.py`
```python
@dataclass
class YieldPrediction:
    district: str
    year: int
    scenarios: dict[Literal["P10","P50","P90"], float]
    bayesian_hdi_95: tuple[float, float]
    bootstrap_ci_95: tuple[float, float]
    feature_importance: dict[str, float]  # SHAP values

def predict_scenarios(
    model_path: Path,
    district: str,
    historical_features: pd.DataFrame,
) -> YieldPrediction: ...
```

### 3.6 `src/rag/retriever.py`
```python
@dataclass
class Passage:
    chunk_id: str
    text: str
    source_citation: str
    score: float

def retrieve(query: str, k: int = 5) -> list[Passage]: ...
```

### 3.7 `src/llm/advisor.py`
```python
@dataclass
class Advisory:
    text: str
    sentences: list[str]
    citations: dict[int, str]  # sentence_idx -> passage_id
    grounding_score: float

def generate_advisory(
    prediction: YieldPrediction,
    retrieved: list[Passage],
    model: str = "llama3:8b-instruct-q4_K_M",
) -> Advisory: ...
```

### 3.8 `src/llm/grounding.py`
```python
def grounding_score(
    advisory_sentences: list[str],
    retrieved_passages: list[Passage],
    threshold: float = 0.6,
) -> float: ...
```

---

## 4. Testing Flow

```
tests/
├── unit/
│   ├── test_phenology.py        # golden-file determinism
│   ├── test_weather_fusion.py   # fusion logic correctness
│   ├── test_calibration.py      # NDVI→yield calibration math
│   ├── test_retriever.py        # mocked ChromaDB behavior
│   └── test_grounding.py        # cosine math edge cases
├── integration/
│   ├── test_data_pipeline.py    # end-to-end from raw → processed
│   ├── test_model_training.py   # train tiny model, assert finite
│   └── test_rag_pipeline.py     # build index → query → retrieve
├── e2e/
│   └── test_full_pipeline.py    # 30 scripted inputs, assert all pass
└── repro/
    └── test_fresh_clone.py      # metric baseline comparison
```

---

## 5. Error Flow

Each layer's failures handled explicitly:

| Failure | Detection | Action |
|---------|-----------|--------|
| GEE auth failure | API returns 401 | Retry with refreshed token; if still fails, abort with instructions |
| NASA POWER timeout | HTTP timeout 60s | Retry 3x with exponential backoff; fallback to Open-Meteo |
| CHIRPS NaN patches | NaN count > 5% | Log; fallback to IMD for those cells |
| XGBoost NaN loss | Train metric non-finite | Abort training, dump input batch for inspection |
| Ollama timeout > 120s | Timeout exception | Fallback to phi3:mini; log degradation |
| Zero RAG hits | Retriever returns empty | Advisory shows "Insufficient local knowledge"; log query |
| Grounding score < 0.5 | Post-hoc check | Flag advisory as low-confidence in UI |

---

## 6. Logging Flow

All structured JSON logs land in `logs/`:
- `logs/data_acquisition.jsonl` — one line per API call
- `logs/model_training.jsonl` — one line per training run with final metrics
- `logs/advisory_generation.jsonl` — one line per advisory with grounding score and latency
- `logs/agent_<id>_status.md` — markdown status note per sub-agent

---

## 7. Deployment Flow (For User)

```
# first time
git clone <repo>
cd rice_yield_project
make install           # creates venv, installs pinned deps
make auth              # prompts for GEE creds, CDS API key
ollama pull llama3:8b-instruct-q4_K_M
ollama pull bge-small     # or let sentence-transformers download
make data              # pulls all data sources (~2 hours first time)
make features          # builds feature matrices
make train             # trains all models, saves artefacts
make rag               # builds vector index
make test              # runs full test suite
make app               # launches Streamlit on localhost:8501

# subsequent runs
make app               # everything cached
```

---

## 8. Manuscript Flow

```
paper/
├── main.tex              # Elsevier or IEEE class, user picks
├── sections/
│   ├── 01_abstract.tex
│   ├── 02_introduction.tex
│   ├── 03_related_work.tex
│   ├── 04_data.tex
│   ├── 05_methodology.tex
│   ├── 06_experiments.tex
│   ├── 07_results.tex
│   ├── 08_discussion.tex
│   ├── 09_limitations.tex
│   └── 10_conclusion.tex
├── figures/*.pdf         # 300 DPI PDF + PNG
├── tables/*.tex          # auto-generated from metrics JSON
├── references.bib        # BibTeX, matches REFERENCES.md
└── cover_letter.tex
```

Every figure and table is **auto-generated from a notebook or script** committed to the repo. No manual figure editing. Reproducibility proven on fresh clone.

---

## 9. Interfaces to the Four Context Files

- `DECISIONS.md` → referenced in code comments as `# See ADR-NNN` whenever a non-obvious choice appears.
- `AI_ML_CONCEPTS.md` → cited in paper methodology prose, and linked to from docstrings.
- `REFERENCES.md` → every citation key in `paper/references.bib` matches an entry here.
- `CLAUDE_CODE_PROMPT.md` → the executing agent's primary instruction set; this document is subservient to it.
