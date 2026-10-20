# Architecture Decision Records — Rice Yield Prediction & Advisory for Kanpur

Each entry follows the ADR format: **Context → Options → Decision → Rationale → Consequences**. New decisions made during execution are appended with the next sequential ID.

---

## ADR-001: Primary training unit is pixel-year, not district-year

**Context.** The official APY dataset gives only 13 yearly observations per district. Any model trained on N=13 is overfitting with near-certainty, regardless of regularization.

**Options.**
- (a) District-year regression with heavy Bayesian regularization (N=13).
- (b) Pixel-year regression using satellite NDVI/EVI as the target proxy (N ≈ 50,000 per district).
- (c) Block-year regression using ICRISAT VDSA (N ≈ 100 if available).

**Decision.** Primary approach is (b). Keep (a) as a validation step: aggregate pixel predictions to district and compare against the 13 APY numbers.

**Rationale.** Pixel-year gives the model enough data to actually learn weather × soil → yield relationships. This is the dominant approach in the modern yield-prediction literature (Lobell 2013, You et al. 2017, Khaki & Wang 2019). The district-level numbers remain the ground truth for the aggregation validation.

**Consequences.** We must handle spatial autocorrelation in train/test splits (same-pixel samples across years are not independent). We use **spatiotemporal blocking**: train on 2010–2020, test on 2021–2024, and never let the same pixel appear in both.

---

## ADR-002: Yield proxy is calibrated NDVI, not NDVI itself

**Context.** Pixel-level yield labels do not exist. We need a proxy.

**Options.**
- (a) Use integrated NDVI directly as yield.
- (b) Calibrate: fit a linear model APY_yield = α + β × district_mean_integrated_NDVI, then back-project per pixel.
- (c) Use a published NDVI-to-yield formula.

**Decision.** (b). Calibrate per-district using the 13 APY points per district as calibration anchors.

**Rationale.** Raw NDVI is a vegetation index, not a yield. Calibration ensures our pixel-level targets have correct magnitude and are reconcilable with the official statistics. Published formulas (c) assume cultivar and management practices that may not match Kanpur.

**Consequences.** Calibration error becomes a floor on prediction accuracy. We report calibration quality (R² of the APY vs mean-NDVI fit) as a separate metric.

---

## ADR-003: Weather data is a fusion of four sources

**Context.** No single weather source is optimal for all variables in India.

**Options.**
- (a) NASA POWER alone (0.5° grid).
- (b) IMD alone (0.25° rainfall, 1° temp).
- (c) ERA5-Land (0.1° hourly) alone.
- (d) Fusion: CHIRPS for rainfall, ERA5-Land for temperature, SMAP for soil moisture, IMD as validation.

**Decision.** (d), with NASA POWER retained only as a secondary cross-check.

**Rationale.** CHIRPS is the gold standard for Indian rainfall at pixel scale (Funk et al. 2015). ERA5-Land has the best temperature reanalysis globally. IMD carries authority with Indian reviewers and lets us cross-validate.

**Consequences.** More engineering work in the preprocessing agent. Mitigated by writing a single canonical daily table that all downstream code reads.

---

## ADR-004: Primary ML model is XGBoost, not sklearn GBR

**Context.** The source Prob & Stats paper used sklearn GradientBoostingRegressor. Modern practice uses XGBoost or LightGBM.

**Options.**
- (a) Keep sklearn GBR (matches source paper).
- (b) XGBoost.
- (c) LightGBM.
- (d) CatBoost.

**Decision.** XGBoost as primary. LightGBM as speed comparison. sklearn GBR retained as baseline.

**Rationale.** XGBoost has native SHAP support, better regularization defaults, categorical handling, and GPU acceleration if ever needed. Also the most-cited model in agricultural ML papers 2020–2026.

**Consequences.** Reviewers familiar with XGBoost will accept the methodology without question. The citation count for XGBoost (>60k) carries weight.

---

## ADR-005: LOO-CV is kept as a secondary metric, not primary

**Context.** The source paper used LOO-CV. For pixel-year data with N=50k+, LOO is computationally infeasible and statistically meaningless (one pixel's leave-out barely moves the model).

**Options.**
- (a) LOO on all pixels (infeasible).
- (b) Spatial LOO at district level.
- (c) Spatiotemporal blocking.
- (d) Both (b) and (c).

**Decision.** (d). Primary metric is 2010–2020 train / 2021–2024 test. Secondary is leave-one-district-out (train on Nagar, test on Dehat, and vice versa).

**Rationale.** Realistic deployment is future-time prediction in a known region. Spatiotemporal split models that. Spatial LOO tests transfer to new districts. LOO-pixel is noise.

**Consequences.** We must report the metric name clearly in every table caption and figure. No ambiguity about which split is being reported.

---

## ADR-006: Streamlit over Tkinter

**Context.** Source paper used Tkinter. User asked for the better choice.

**Options.** Tkinter (desktop, matches source) vs Streamlit (web-local, modern UI).

**Decision.** Streamlit, served on `localhost` with no outbound network.

**Rationale.** 10x less GUI code to write. Cleaner screenshots for the paper. Still fully offline. Same reproducibility guarantee. Faster iteration during the build.

**Consequences.** Need to document in the README that the user accesses via `http://localhost:8501` after `streamlit run app.py`. Mentioning in the paper that offline property holds because Streamlit binds to loopback only.

---

## ADR-007: Local LLM is Qwen2.5 14B instruct (Q6_K), fallback Llama 3 8B (Q4_K_M)

**Status.** REVISED (hardware correction by user; supersedes the original Llama 3 8B primary / Phi-3 mini fallback decision).

**Context.** Need a local, offline, instruction-following LLM. Development and target hardware is 32GB RAM + RTX 5070 Ti with 16GB VRAM, not the 8GB-RAM CPU box originally assumed.

**Options.**
- (a) Qwen2.5 14B instruct, Q6_K (~12GB; fits fully in 16GB VRAM). Already pulled in Ollama as `qwen2.5:14b-instruct-q6_K`.
- (b) Llama 3 8B instruct, Q4_K_M (~4.7GB). Already pulled as `llama3:latest`.
- (c) Phi-3 mini (3.8B, ~2.3GB). Original fallback; dropped.
- (d) Mistral 7B / Gemma 2 9B. Not pulled; not considered further.

**Decision.** Primary = `qwen2.5:14b-instruct-q6_K`, used everywhere (advisory generation, LLM-as-judge). Fallback = `llama3:latest` (Q4_K_M, 4.7GB), used on Ollama timeout > 120s. Phi-3 mini is no longer part of the pipeline; every reference to `llama3:8b-instruct-q4_K_M` or `phi3:mini` in the other context files is overridden by this ADR.

**Rationale.** The larger model with a high-fidelity quant is better at instruction/citation following, which the RAG constraint depends on, and the available GPU can hold it entirely. Both models are already local, so no new downloads. Llama 3 as fallback is a quality-comparable smaller model, faster to load.

**Consequences.** (1) The original "<90 s on 8GB RAM CPU" latency criterion is re-baselined to the stated 32GB/16GB-VRAM GPU hardware and must be stated as such in the paper. (2) The LLM-as-judge uses the same family as the generator (self-preference bias) and this is disclosed in Limitations. (3) The 8GB-RAM deployability claim is dropped; README documents the minimum hardware honestly. (4) The offline claim (ADR-018) is unaffected.

---

## ADR-008: Embedding model is nomic-embed-text via Ollama

**Status.** REVISED (user decision; supersedes the original MiniLM-vs-BGE benchmark plan).

**Context.** Need a local sentence embedding model for RAG and for the grounding check (ADR-010). `nomic-embed-text` is already pulled in Ollama.

**Options.**
- (a) sentence-transformers/all-MiniLM-L6-v2. Rejected by user.
- (b) BAAI/bge-small-en-v1.5. Rejected by user.
- (c) nomic-embed-text (via Ollama, 137M, 274MB).

**Decision.** (c). All embeddings (KB chunks, queries, advisory sentences for grounding) come from Ollama's `nomic-embed-text`. `sentence-transformers` is NOT a dependency. The MiniLM-vs-BGE comparison in Phase 4B is cancelled; Phase 4B instead builds a single ChromaDB index (`kb/index/chroma_nomic/`) and still reports precision@3 / recall@5 on the 20 handcrafted queries.

**Rationale.** Single runtime (Ollama) for both generation and embeddings; fewer dependencies (no torch-based sentence-transformers install); the model is already local. Retrieval quality is measured empirically rather than assumed.

**Consequences.** (1) Reimers & Gurevych 2019 (R12, Sentence-BERT) is no longer a methodological citation for our embedding stack; it will only be cited, if at all, as background after verification. A nomic-embed-text technical report must be located and verified via web search before it is cited. (2) nomic-embed-text expects task prefixes (`search_query: ` / `search_document: `); the retriever must apply them. (3) The 0.6 cosine grounding threshold (ADR-010) was a heuristic calibrated on other embedders; it must be re-checked against nomic similarity distributions on a small labeled sample rather than assumed.

---

## ADR-009: Vector store is ChromaDB with HNSW

**Context.** Need an offline, file-backed vector store.

**Options.** ChromaDB, FAISS, LanceDB, SQLite + sqlite-vss.

**Decision.** ChromaDB with HNSW index.

**Rationale.** Zero-config, pure Python, file-backed, HNSW gives sub-millisecond retrieval on our KB size (<1000 chunks). FAISS is faster at scale but more fragile to install on Windows.

**Consequences.** Index files stored in `kb/index/` committed to the repo (small, <10MB).

---

## ADR-010: Advisory grounding is enforced via post-hoc cosine check

**Context.** Even with a strict prompt, LLMs can inject un-grounded sentences.

**Options.**
- (a) Trust the prompt alone.
- (b) Post-hoc check: for each advisory sentence, compute max cosine similarity to any retrieved passage. If < 0.6, flag.
- (c) Rewrite flagged sentences via a second LLM pass.
- (d) Both (b) and (c).

**Decision.** (b) initially. If grounding score < 80% consistently, add (c).

**Rationale.** Cosine threshold at 0.6 is a published heuristic for semantic grounding (Es et al. 2023 — RAGAS). We display the grounding score in the UI, which pressures the system toward honesty rather than hiding failures.

**Consequences.** Grounding score becomes a first-class user-facing metric. We cite the methodology.

---

## ADR-011: Confidence intervals are both Bayesian and bootstrap

**Context.** User asked for "scenario-based probabilistic forecasting with Bayesian confidence intervals."

**Options.**
- (a) Bayesian Ridge posterior intervals only.
- (b) Bootstrap intervals from the XGBoost model only.
- (c) Both.

**Decision.** (c). Report Bayesian 95% HDI from a PyMC hierarchical model AND bootstrap 95% CI from 1000 resamples of XGBoost.

**Rationale.** They answer slightly different questions. Bayesian gives us "given the data and priors, what is the posterior over yield?" Bootstrap gives us "if we resampled the training set, how stable would the point estimate be?" Both are defensible. Showing both and discussing when they diverge is a methodological strength.

**Consequences.** More compute, more code. Benefit: a paper figure comparing the two intervals is distinctive.

---

## ADR-012: Synthetic data augmentation via AquaCrop (stretch goal)

**Context.** Even with pixel-level data, out-of-distribution weather scenarios (worst-case drought, worst-case flood) are rare. Simulated data covers them.

**Options.**
- (a) Skip synthetic augmentation.
- (b) Gaussian noise on existing features (weak).
- (c) AquaCrop-OSPy process-based simulation.
- (d) DSSAT (more detailed, harder to install).

**Decision.** Attempt (c) as a stretch goal. If time runs out, document as future work. Do not use (b).

**Rationale.** Process-based crop models encode agronomic physics. (b) only amplifies noise and reviewers will dismiss it. AquaCrop-OSPy is pip-installable and well-documented.

**Consequences.** If completed, a paper subsection on "process-informed data augmentation" is a distinctive methodological contribution.

---

## ADR-013: Spatiotemporal train-test split, pixel-blocked

**Context.** Random pixel splits leak information because adjacent pixels and consecutive years are correlated.

**Decision.** Train on all pixels in years 2010–2020. Test on all pixels in years 2021–2024. No pixel repetition logic needed because the time boundary is the block.

**Rationale.** Mimics real deployment: predicting future years for the same region.

**Consequences.** Reported R² will be lower than random splits — that is the honest number.

---

## ADR-014: SHAP for feature importance, not permutation

**Context.** Permutation importance is noisy for correlated features. SHAP handles correlation better and gives per-sample explanations.

**Decision.** Report SHAP summary plot as the primary feature importance figure. Permutation importance as a sanity check in supplementary.

**Rationale.** SHAP is the de facto standard in ML interpretability since Lundberg & Lee 2017. Per-sample explanations also enable the Diagnostics tab in the Streamlit app.

**Consequences.** Must pin `shap==0.44.x` because older versions have XGBoost incompatibility.

---

## ADR-015: Reference list is finalized before manuscript writing

**Context.** Hallucinated citations are a known LLM failure mode and would torpedo the paper.

**Decision.** All references used in the manuscript must appear in `REFERENCES.md` and must be verified via web search or direct DOI resolution before citing. Any new reference discovered mid-write is added to `REFERENCES.md` first, then cited.

**Rationale.** Non-negotiable. One fabricated citation can get a paper retracted.

**Consequences.** Slower writing. Worth it.

---

## ADR-016: Weather forecast integration for live mode (post-sprint)

**Context.** The system currently predicts using historical scenarios. For deployment, farmers need predictions using actual forecasts for the coming season.

**Decision.** Post-sprint extension: integrate Open-Meteo historical+forecast API. Feature-flag it behind `USE_LIVE_FORECAST=true`. Document as future work in the paper.

**Rationale.** Live forecasting adds dependency and failure modes. Keep the sprint target as the honest offline-first baseline.

**Consequences.** README mentions the flag. Paper mentions it as near-term extension, not as part of the submitted contribution.

---

## ADR-017: Public GitHub repo with MIT license and DOI via Zenodo

**Context.** Reproducibility statement in the paper needs a stable archive.

**Decision.** Public GitHub repo. On submission, archive a release to Zenodo for a permanent DOI. Cite the DOI in the paper.

**Rationale.** Journals increasingly require this. Zenodo is free and GitHub-integrated.

**Consequences.** No private forks after submission (public from day one). All credentials in `.env.example`, never committed.

---

## ADR-018: No cloud LLM in the shipped pipeline

**Context.** Reviewers will test the offline claim.

**Decision.** The shipped pipeline uses Ollama only. Cloud LLMs are explicitly forbidden in `src/llm/`. If a future developer adds one, CI fails on a lint rule that blocks `openai`, `anthropic`, `google.generativeai` imports in that directory.

**Rationale.** The offline-first claim is a core contribution. Protecting it in code is more credible than protecting it in prose.

**Consequences.** Development scaffolding (not shipped) may use cloud LLMs for scaffolding. Developer convenience not degraded.

---

## ADR-019: Honest limitations section is required

**Context.** The paper must survive a hostile peer review.

**Decision.** Limitations section lists at minimum: (a) absence of human agronomist evaluation of advisories; (b) single-crop single-region; (c) satellite proxy introduces calibration error floor; (d) RAG KB is 300–500 chunks, far smaller than production systems; (e) local LLM hallucination risk not eliminated, only reduced.

**Rationale.** Every reviewer will spot these anyway. Pre-empting them signals methodological maturity.

**Consequences.** Discussion section loses pure celebration tone. Gains credibility.

---

## ADR-020: Decision deferral protocol

**Context.** Mid-build, new decisions will arise.

**Decision.** When a decision cannot be resolved immediately: file a new ADR with status DEFERRED, document options and the information needed to resolve, continue with the most reversible option.

**Rationale.** Decisions deferred without record become hidden technical debt.

**Consequences.** `DECISIONS.md` becomes the single source of truth for the project's design rationale.

---

## ADR-021: SHAP and dependency versions are pinned via uv.lock, not hand-pinned

**Context.** ADR-014 pinned `shap==0.44.x` for XGBoost compatibility. The dev machine runs Python 3.13, for which 0.44 very likely has no wheel.

**Decision.** Declare ranges in `pyproject.toml` (`shap>=0.46`) and pin exact versions in the committed `uv.lock`. Verify SHAP-on-XGBoost works against the locked versions in Phase 3 before relying on it.

**Consequences.** ADR-014's intent (SHAP must work with XGBoost) is kept; its literal version pin is superseded.

---

## ADR-022: GEE access via service account; MODIS design

**Status.** ACTIVE for auth, spatial scope and MODIS design. Item 2 (Cloud Storage export) is SUPERSEDED by ADR-024. Replaces the earlier draft (user OAuth + Export to Google Drive), which never ran.

**Context.** A service account cannot write to a personal Google Drive, and a user OAuth token cannot be refreshed unattended. The Community-tier EECU budget is 150 h with a 100 h usable ceiling (user rule).

**Decision.**
1. Auth: `google.oauth2.service_account.Credentials.from_service_account_file()` with `rice-yield-worker@notifyproj-a28b2.iam.gserviceaccount.com`; key path from `GEE_SERVICE_ACCOUNT_KEY` (default `.secrets/gee-key.json`, gitignored). IAM roles granted by the user: Earth Engine Resource Writer, Service Usage Consumer, Storage Admin. APIs enabled: Earth Engine, Cloud Storage.
2. Export route: originally `Export.image.toCloudStorage` into bucket `rice-yield-kanpur-exports` (asia-south1, NEARLINE). Superseded by ADR-024 (Earth Engine assets).
3. Spatial scope: GADM 4.1 level-2 polygons for Kanpur Nagar and Kanpur Dehat (not a bounding box).
4. Native MOD13Q1 (v061) 16-day composites on the native 231.66 m sinusoidal grid, Kharif window DOY 150-300, bands NDVI/EVI/SummaryQA/DayOfYear. Monthly aggregation was rejected for phenology (it blurs the 0.4-threshold crossings) and would not reduce EECU, since no reducers run.
5. Cropland (WorldCover 2021 class 40) is exported once as a fraction raster on the MODIS grid and applied locally.
6. EECU: a-priori estimates are logged before each export and the metered `batch_eecu_usage_seconds` after; the guard in `src/data/gee_quota.py` stops at 100 h.
7. `force_ipv4()` is applied before any Google client is built: the dev machine's IPv6 route is black-holed and stalled every connection.

**Consequences.** (1) Creating the bucket failed with `403: The billing account for the owning project is disabled`, which led to ADR-024. (2) Key hygiene: `.secrets/` is gitignored; the key must never be committed. Verified before the first key-related commit: `git log --all --full-history -- .secrets/gee-key.json` and `git ls-files .secrets/` are both empty, and no object named like the key exists in any ref.

---

## ADR-023: Force IPv4 for all Google client traffic

**Context.** The dev machine's IPv6 route is black-holed: connecting to any Google API over IPv6 times out (about 8 s measured with raw sockets) before IPv4 succeeds. `ee.Initialize` makes several sequential connections (including google-auth's regional-access-boundary lookup) and appeared to hang indefinitely.

**Decision.** `src/utils/net.force_ipv4()` sets `urllib3.util.connection.HAS_IPV6 = False`; it is called from `satellite.init_ee` before any client is created.

**Consequences.** Affects every library built on urllib3/requests in-process (earthengine-api, google-auth, requests). Harmless on dual-stack machines. Remove if IPv6 is repaired and this shim is no longer needed.

---

## ADR-024: Export GEE results to Earth Engine assets, not Cloud Storage

**Context.** `Export.image.toCloudStorage` needs a GCS bucket, and creating one failed: the project `notifyproj-a28b2` has no billing account. Service accounts also cannot export to a personal Drive (ADR-022).

**Options.** (a) Link a billing account and use GCS. (b) `Export.image.toAsset` into the project's own asset store, then pull pixels with `Image.getDownloadURL`. (c) Skip batch export and call `getDownloadURL` on the computed image directly.

**Decision.** (b). Assets live at `projects/notifyproj-a28b2/assets/kanpur_rice/<source>/<year>/<name>`. Folders are created idempotently. The `google-cloud-storage` dependency is kept for now but is unused (marked in `pyproject.toml`).

**Rationale.** No billing account is attached, which keeps the project on the free Community tier and strengthens the zero-cost claim of the paper. A batch export (unlike (c)) reports a metered `batch_eecu_usage_seconds`, which the quota ledger needs (the 100 EECU-hour ceiling).

**Consequences.** (1) The download from the asset is an interactive request and is not metered in batch EECU; it is limited to about 48 MB per request, which is why exports are per year. (2) Asset storage counts against the Earth Engine asset quota (verify the quota before the full-range export). (3) Manifest entries record the `ee_asset_id` instead of a `gs://` URL.

**Dry-run findings (2020, appended to ADR-024).**
- Asset root `projects/notifyproj-a28b2/assets` did not exist for the service account; it was created once with `ee.data.createAssetHome`. Quota shown by EE: 250 GB / 50,000 assets.
- `reduceResolution` of 10 m WorldCover fails ("Reprojection output too large") when the export footprint exceeds roughly 20M source pixels (probed: 0.15 x 0.15 degrees works, 0.3 x 0.5 fails). The cropland-fraction export is therefore tiled 4 x 4 (16 assets, `worldcover/2021/cropland_frac_t<i>`) and mosaicked locally.
- A lon/lat rectangle is slanted on the MODIS sinusoidal grid, so each tile's bounding box carries ~40% masked padding (-32768). The mosaic must paste only valid pixels.
- `getDownloadURL` returns one unnamed multi-band GeoTIFF; band names are saved in a `<zip>.bands.json` sidecar at download time.
- Measured cost: year stack 11.2 EECU-s; one-off cropland tiles 104 EECU-s total.

## ADR-025: MODIS QA policy and temporal gap-filling (Savitzky-Golay)

**Status:** Accepted (user-approved 2026-10-20). Implemented in Phase 2.

**Context.** The 2020 dry run showed SummaryQA shares of 18.4% good (0), 37.4% marginal (1) and 44.2% cloudy (3) over the Kharif window. Keeping only QA = 0 leaves ~18% of observations and breaks the phenology curve.

**Decision.**
1. Accept QA <= 1 (good + marginal). Observations with QA = 3, or NDVI missing, are set to NaN.
2. Gap-fill NaNs by linear interpolation in time (edge gaps by nearest valid value), then smooth with a Savitzky-Golay filter, polynomial order 2, window 5 composites (`scipy.signal.savgol_filter`).
3. Pixels with fewer than 5 valid raw observations in the window are dropped (the filter would be mostly interpolation).
4. Raw values and a `was_filled` flag are kept next to the smoothed series.

**Why Savitzky-Golay over Whittaker.** Both are defensible. SG is in SciPy (no extra dependency, tested), is vectorisable over ~100k pixels, and the window/order are the user's specified values. Whittaker handles gaps natively through weights but needs a custom or third-party solver. SG has no native gap handling, hence the interpolation step in (2).

**Caveats.** With 16-day composites and a window of 5 (~80 days) the filter can flatten a short peak and lower the peak NDVI slightly. This is checked in Phase 2 by comparing raw and smoothed peak NDVI. Window 5 / order 2 is fixed by the user, not tuned.

**References (verified by web search 2026-10-20).**
- Chen, J., Jonsson, P., Tamura, M., Gu, Z., Matsushita, B., Eklundh, L. (2004). A simple method for reconstructing a high-quality NDVI time-series data set based on the Savitzky-Golay filter. *Remote Sensing of Environment* 91(3-4), 332-344. doi:10.1016/j.rse.2004.03.014.
- Son, N.-T., Chen, C.-F., Chen, C.-R., Duc, H.-N., Chang, L.-Y. (2014). A phenology-based classification of time-series MODIS data for rice crop monitoring in Mekong Delta, Vietnam. *Remote Sensing* 6(1), 135-156. doi:10.3390/rs6010135.

**Citation note.** Son et al. (2014) smooths MODIS EVI with *empirical mode decomposition*, not Savitzky-Golay. It is cited here only as precedent for smoothing MODIS series before rice-phenology analysis; the SG method itself rests on Chen et al. (2004). The request to cite Son et al. for the SG methodology could not be honoured as stated.

### ADR-025 addendum (2026-10-20): minimum valid observations

After the QA <= 1 rule and before gap-filling, each pixel-year must have **at least 4 valid observations** (of 9 composites). Pixel-years with 0-3 are dropped *for that year only*; the pixel stays in other years. This replaces the "fewer than 5" floor stated in item 3 above. Dropped counts per year are reported in the Phase 2 report.

---

## ADR-026: CHIRPS is the primary rainfall source; IMD kept as a secondary column

**Status:** Accepted (user decision, 2026-10-20).

**Context.** Over 2010-2024 the district-mean annual rainfall is 884-918 mm in CHIRPS (Nagar/Dehat) and 863 mm at the NASA POWER point, but 629 mm in the IMD 0.25 deg grid (5x5 cells over the box), about 30% below CHIRPS. Year totals diverge most in 2014 (466 mm), 2015 (446 mm) and 2017 (453 mm).

**Decision.** All rainfall features use CHIRPS (0.05 deg). IMD rainfall stays in the feature matrix as a secondary column (`imd_*`), not dropped, so reviewers can see the sensitivity. IMD Tmax/Tmin remain secondary too.

**Rationale.** CHIRPS' totals agree with the Kanpur climatological normal range (820-900 mm, as stated by the project owner; not independently re-checked here) and with NASA POWER, and CHIRPS is validated over South Asia (Funk et al. 2015, R15). The suggested mechanism for the IMD shortfall (0.25 deg gridding under-representing convective peaks) is a hypothesis; it was not tested.

**Consequences.** IMD-based sensitivity features can be added later without re-acquisition. The CHIRPS-vs-IMD gap is disclosed in the manuscript's limitations.

---

## ADR-027: Phase 2 conventions (cropland mask, time axis, phenology edge cases)

**Status:** Accepted by the implementer in Phase 2 (self-decision, flagged for review).

- **Cropland mask:** pixels with WorldCover-2021 cropland fraction >= 0.5 (92,969 of 113,733). 0.75 would keep 76,412; 0.25 would keep 102,630. Not tuned against any outcome.
- **Time axis:** each 16-day composite is placed at its centre (start DOY + 8); the filled series is linearly interpolated to daily values for threshold crossings, peak, integral and the Aug 15 - Sep 15 window (DOY 227-258, 228-259 in leap years).
- **Edge cases:** `greenup_day` is NaN if the smoothed curve is already >= 0.4 at the first composite or never reaches it; `senescence_day` is NaN if it is still >= 0.4 at the window end (DOY 297). `length_of_season` is NaN if either is NaN. Censoring flags `greenup_censored` / `senescence_censored` are kept. The Kharif window ends at DOY 300, so senescence is censored for most pixels (see Phase 2 report).
- **Weather deviations from the spec text:** `monsoon_onset_day` = first day >= 5 mm on or after Jun 1 (all 30 district-years fell in June); dry spell = run of >= 7 days with < 1 mm (each run counted once); `kharif_gwetroot_mean` and IMD secondary columns are added. Temperature and humidity are one NASA POWER point shared by both districts.
- **SoilGrids:** 0-5 cm mean, nearest-neighbour onto the MODIS grid; 3.8% of pixels have no soil value (masked in SoilGrids). The `nitrogen` column (catalog unit cg/kg divided by 100) has an implausible magnitude (mean 15 g/kg against SOC 18 g/kg) and is unverified; use it only as a relative covariate until checked.
- **Circularity warning:** the pixel target is alpha + beta * `grain_filling_ndvi` per district, so a model that sees `grain_filling_ndvi` can recover the target exactly. Phase 3 must exclude it (and review `integrated_ndvi`, `peak_ndvi`) from the feature set.

---

## ADR-025 addendum 2 (2026-10-20): Kharif window to DOY 330, SG window sensitivity

**Window.** The composite window now ends at start-DOY 330 (composites starting DOY 161 ... 321, 11 per year; the last is centred on DOY 329). Two extra composites per year were pulled from Earth Engine as new assets `.../modis_mod13q1/<year>/kharif_stack_ext330` (about 3-7 EECU-s each; the original 9-composite assets are kept). Result: `senescence_day` null fell from 67% to 19% of pixel-years; `length_of_season` from 74% to 38%; `greenup_day` null is 24% (pixels already >= 0.4 at the first composite).

**Side effect on the 4-observation rule.** With 11 composites, the minimum-4-valid rule drops almost nothing (e.g. 2018: 10 pixels instead of 40,720), because the two added Oct/Nov composites are mostly clear. A pixel can now pass the rule with valid points only after the monsoon, so its peak is interpolated across the cloudy period. The rule is kept as specified; median valid observations per pixel-year is 7 (min 4).

**Sensitivity test (pre-registered rule: switch to window 3 only if the mean over years of the yearly median smoothed peak NDVI improves by >= 0.05 and median roughness, the mean absolute second difference of the smoothed series, rises by no more than 50%).**

| Filter | Mean of yearly median peak NDVI | Median roughness | Years with median peak in 0.65-0.85 |
|---|---|---|---|
| SG window 5, order 2 (current) | 0.627 | 0.035 | 4 of 15 |
| SG window 3, order 2 | 0.645 | 0.057 | 6 of 15 |
| SG window 3, order 1 (3-point linear fit) | 0.605 | 0.028 | 2 of 15 |

Window 3 with order 2 is the identity filter (a parabola through 3 points reproduces them), so that row is the gap-filled series with no smoothing. It raises the peak by only 0.018 (< 0.05) and roughness by 63%. **Decision: keep window 5, order 2.** The raw (unsmoothed, QA <= 1) peak median equals the window-3/order-2 numbers (identity filter), so the low peak NDVI is mostly in the data (cloud/marginal-QA contamination, mixed cropland, the 16-day composite peak), not an artifact of smoothing. Median peak day is DOY 249 in 8 years and 265 in 7 (composite-centre granularity).

---

## ADR-026 correction (2026-10-20)

The IMD shortfall is about **30%** below CHIRPS (629 mm vs 884-918 mm), not the 25% quoted in the original request. ADR-026 already states 30%; this note records that the 25% figure was a misstatement.

---

## ADR-028: Exclude calibration-derived and target-correlated NDVI features from model inputs

**Status:** Accepted (user decision 2026-10-20). Most important methodological ADR of the project.

**Context.** Per ADR-002 the pixel-level training target is `yield_est = alpha_d + beta_d * grain_filling_ndvi` (district-specific alpha, beta fitted on 13 APY years). The target is therefore an exact linear function of `grain_filling_ndvi` within each district. A model given that feature can recover the target with R^2 = 1.0 without learning anything about yield.

**Decision.**
1. `grain_filling_ndvi` is never a model input.
2. `integrated_ndvi` and `peak_ndvi` (smoothed) are removed too. Measured Pearson r with `grain_filling_ndvi` over 1,394,144 pixel-years: `peak_ndvi` 0.876, `integrated_ndvi` 0.866 (0.872 and 0.871 after removing district-year means). These are below the "0.9" quoted in the review request; they are removed on the grounds that they are the same smoothed-curve signal, not because they cross 0.9.
3. Allowed phenology inputs: `greenup_day`, `peak_ndvi_day`, `senescence_day`, `length_of_season`, `raw_peak_ndvi`. All weather features and all soil features are allowed.
4. Enforced in code: `src/models/feature_sets.py` defines the forbidden set and a test fails if any model's features include it.

**Known residual leakage (not resolved by this ADR).** `raw_peak_ndvi` has r = 0.848 with `grain_filling_ndvi` (0.844 within district-year), i.e. a model using it alone can explain roughly 70% of the target variance. The timing features correlate at -0.50 (`greenup_day`), -0.33 (`peak_ndvi_day`), 0.42 (`senescence_day`) and 0.51 (`length_of_season`). Keeping `raw_peak_ndvi` was the user's decision; it is noisier than the smoothed peak but it is not independent of the target. Consequences for reporting:
- Pixel-level R^2 measures how well NDVI-derived phenology plus weather and soil predict a calibrated NDVI quantity. It is **not** evidence of yield-prediction skill. The target contains no information that is not in the satellite signal.
- The independent evidence is the district-level comparison against the 13 APY years per district (aggregate predictions vs APY, leave-one-year-out). That is the headline metric.
- Phase 3 must report an ablation with and without `raw_peak_ndvi` (and a satellite-free weather+soil model) so the leakage contribution is visible.

**Per-model feature policy (weather collinearity).** Linear, Lasso and ElasticNet drop `gdd_kharif` (r = 0.99 with `kharif_tmax_mean`) and keep `kharif_tmax_mean` and `kharif_rh_mean`; Ridge, RF, XGBoost and the other models use the full allowed set. Note `kharif_tmax_mean` and `kharif_rh_mean` correlate at about -0.99 in the district-year table, so VIF stays high for the linear baselines; this was accepted. The weather table has 9 engineered features (the request said 8), plus `kharif_gwetroot_mean` and the IMD secondary columns.

**Rationale.** Evaluating a model on a target that is a deterministic function of its inputs inflates every metric and would invalidate the manuscript. Removing the source feature is necessary; because other NDVI features carry most of its information, the reporting rules above are also necessary.

---

## ADR-027 update (2026-10-20): SoilGrids nitrogen scale

SoilGrids' catalog labels nitrogen as cg/kg, but the raw median over Kanpur is 1,465. Taken as cg/kg that is 14.7 g/kg, which with SOC at 17.4 g/kg gives C:N of about 1.2 (real topsoil is about 8-15). Dividing by 100 (the first fix) therefore left an implausible value. The column now holds raw / 10,000 = **0.15** (mean over pixels, range 0.12-0.30), which equals 1.5 g/kg or 0.15% N (C:N about 12). **The unit of the `nitrogen` column is percent N (g per 100 g), not g/kg**; the request's expectation of "0.1-0.3 g/kg" matches the number but 0.15 g/kg would be implausibly low for a topsoil, so the label is % N. The catalog unit/scale discrepancy was not independently confirmed with ISRIC.
