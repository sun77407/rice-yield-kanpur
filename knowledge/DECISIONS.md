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
