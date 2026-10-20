# AI & ML Concepts — Definitions, Usage, and Impact

Every concept used in this project. Each entry answers four questions: **What is it? Where in our project? Why this and not alternatives? What impact?**

---

## Section A — Foundations

### A1. Supervised Learning

**Definition.** A learning paradigm where the algorithm is given a labeled training set `{(x_i, y_i)}` and must learn a function `f: X → Y` that generalizes to unseen inputs.

**In our project.** Pixel-year rice yield prediction. `x` = phenology + weather + soil features. `y` = calibrated NDVI-derived yield.

**Why.** Our problem has clear inputs and labels. Unsupervised methods (clustering) cannot produce calibrated predictions.

**Impact.** Enables point predictions with quantifiable error. Allows comparison across model families on identical splits.

---

### A2. Regression (vs Classification)

**Definition.** Supervised learning where `y` is continuous (regression) rather than categorical (classification).

**In our project.** Yield in tonnes per hectare is continuous. Scenario bucketing (bad/avg/good) happens post-hoc on continuous outputs, not during training.

**Why.** Treating yield as classes (low/med/high) would discard information and make calibration meaningless.

**Impact.** Enables scenario forecasting via quantile outputs instead of fixed buckets.

---

### A3. Train / Validation / Test Split

**Definition.** Partitioning data into disjoint sets: training (fit parameters), validation (tune hyperparameters), test (final honest metric).

**In our project.** Spatiotemporal: 2010–2020 for train+val (5-fold CV within), 2021–2024 held out as test.

**Why.** Random pixel splits leak information because of spatial and temporal correlation. Time-forward splits mimic deployment.

**Impact.** The test R² we report is a realistic estimate of future performance, not an optimistic one.

---

### A4. Cross-Validation (K-fold, LOO, Spatial LOO)

**Definition.**
- **K-fold:** split train into K folds, train on K-1, validate on 1, rotate.
- **LOO:** leave one sample out, repeat N times.
- **Spatial LOO:** leave one spatial unit (district) out.

**In our project.** 5-fold K-fold inside the 2010–2020 training window for HP tuning. Spatial LOO (leave Kanpur Nagar out, test on Dehat; and vice versa) as a secondary generalization metric.

**Why.** 5-fold CV is the standard HP tuning protocol. Spatial LOO probes transferability.

**Impact.** Prevents HP tuning from leaking test information. Spatial LOO result is the headline for "will this work in a new district?"

---

## Section B — Classical ML Models

### B1. Linear Regression

**Definition.** `y = Xβ + ε` with closed-form OLS solution `β = (XᵀX)⁻¹Xᵀy`.

**In our project.** Baseline floor. If a complex model does not beat Linear, the complexity has not earned its place.

**Why.** Interpretable, fast, cited.

**Impact.** Defines the minimum defensible performance. Serves as a sanity check that features carry signal.

---

### B2. Ridge Regression (L2 regularization)

**Definition.** Minimizes `||y − Xβ||² + α||β||²`. The α term shrinks coefficients, trading bias for variance reduction.

**In our project.** Our baseline for small-N district model (N=26 pooled). Also included in the pixel-year baseline suite.

**Why.** Handles multicollinearity (precipitation and soil moisture correlate at r=0.96). Does not drop features (unlike Lasso).

**Impact.** Produces stable coefficients when features are correlated, which is the norm in weather data.

---

### B3. Lasso Regression (L1 regularization)

**Definition.** Minimizes `||y − Xβ||² + α||β||₁`. The L1 penalty forces some coefficients to exactly zero → feature selection.

**In our project.** Baseline with built-in feature selection; auxiliary for understanding which features matter.

**Why.** Interpretability. If Lasso zeroes a feature, that is evidence the feature does not help linear modeling of the problem.

**Impact.** Produces a sparse coefficient vector that is easier to communicate to agronomists.

---

### B4. ElasticNet

**Definition.** Combines L1 and L2: `||y − Xβ||² + α(ρ||β||₁ + (1−ρ)||β||²)`.

**In our project.** Baseline suite.

**Why.** When correlated features exist and sparsity is desired, ElasticNet typically outperforms either penalty alone.

**Impact.** Robust baseline for correlated weather features.

---

### B5. Random Forest

**Definition.** Ensemble of decision trees trained on bootstrap samples with feature subsampling at each split. Predictions averaged across trees.

**In our project.** Non-linear baseline.

**Why.** Captures interactions linear models miss (e.g., high rainfall AND high temperature causes different stress than either alone). Minimal HP tuning needed. Native handling of missing values.

**Impact.** Establishes a strong non-linear baseline before investing in boosting.

---

### B6. Gradient Boosting (sklearn GBR)

**Definition.** Sequentially fit trees to the residuals of previous trees. Each new tree corrects errors of the ensemble so far.

**In our project.** Baseline matching the source Prob & Stats paper (so we can show the delta from using modern alternatives).

**Why. Direct comparability** to the source paper's approach.

**Impact.** Reviewers can see where our methodological upgrades (XGBoost) produced measurable improvement.

---

### B7. XGBoost

**Definition.** Extreme Gradient Boosting. A regularized, histogram-based implementation of gradient boosting with second-order loss approximation, parallel tree construction, and built-in SHAP support.

**In our project.** **Primary production model.**

**Why.** State of the art on tabular agricultural data per van Klompenburg et al. 2020. SHAP integration is native. Robust to hyperparameters. Fast CPU inference.

**Impact.** The headline model result in the paper. Enables SHAP-based feature importance figures.

---

### B8. LightGBM

**Definition.** Microsoft's gradient boosting framework using leaf-wise tree growth (vs level-wise in XGBoost) and histogram binning.

**In our project.** Speed comparison and secondary candidate.

**Why.** Often faster than XGBoost at similar accuracy. If LightGBM beats XGBoost on our data, we switch; otherwise it is reported in the appendix.

**Impact.** Provides a second ensemble opinion, strengthens the methodological claim.

---

## Section C — Bayesian Methods

### C1. Bayesian Inference

**Definition.** Posterior ∝ Likelihood × Prior. Instead of a point estimate for parameters, we maintain a probability distribution reflecting uncertainty.

**In our project.** For scenario forecasting with honest uncertainty quantification. Classical ML gives point predictions; Bayesian gives posterior predictive distributions.

**Why.** Our dataset is small (district level) and uncertainty matters more than point accuracy. Priors encode agronomic knowledge (e.g., rice yield in UP is centered around 2.5 t/ha with SD 0.7).

**Impact.** Enables P10/P50/P90 scenarios as proper quantiles of the posterior predictive, not post-hoc assumptions.

---

### C2. Bayesian Ridge Regression

**Definition.** Linear regression with Gaussian priors on coefficients. Posterior is analytically tractable.

**In our project.** Fast Bayesian baseline. Produces 95% credible intervals natively.

**Why.** sklearn implementation is one line. Serves as a smoke test before the heavier PyMC model.

**Impact.** First-pass uncertainty quantification without MCMC compute cost.

---

### C3. Hierarchical Bayesian Model

**Definition.** Model with multiple levels of parameters: individual (pixel), group (district), and population (crop). Lower levels share information upward.

**In our project.** `yield_pixel ~ Normal(mu_district + Xβ, σ_pixel)` with `mu_district ~ Normal(mu_global, σ_district)`.

**Why.** Districts share information (both grow rice in similar climate), but have different baseline yields. A pooled model loses district effects; separate models cannot share information. Hierarchy solves this.

**Impact.** Better estimates for the district with less data (Dehat), via regularization toward the global mean.

---

### C4. NUTS (No-U-Turn Sampler) / HMC

**Definition.** A gradient-based MCMC algorithm that explores the posterior efficiently. NUTS auto-tunes the step size.

**In our project.** PyMC's default sampler for our hierarchical model.

**Why.** Dominant modern MCMC method. Converges on complex geometries where Metropolis-Hastings and Gibbs fail.

**Impact.** Reliable posterior samples with reproducible seeds.

---

### C5. Highest Density Interval (HDI) vs Credible Interval

**Definition.** HDI is the smallest interval containing a given probability mass (e.g., 95%); credible interval is the central interval between quantiles. For skewed posteriors they differ.

**In our project.** We report HDI, because yield distributions are often right-skewed (bad years can be very bad, good years are capped by biology).

**Why.** HDI is the more defensible choice for skewed distributions.

**Impact.** More honest intervals than naive 2.5–97.5 percentile cuts.

---

### C6. Bootstrap Confidence Intervals

**Definition.** Resample the training set with replacement, retrain, predict. Repeat 1000 times. The empirical distribution of predictions gives a CI.

**In our project.** Non-Bayesian uncertainty for XGBoost (which has no closed-form posterior).

**Why.** Model-agnostic, well-understood, publishable. Complements Bayesian intervals.

**Impact.** When Bayesian and bootstrap intervals agree, confidence is high. When they diverge, we discuss why in the paper.

---

## Section D — Feature Engineering

### D1. Phenology Features from NDVI Time Series

**Definition.** Metrics describing the vegetation growth curve: greenup day, peak, senescence, length of season, area under curve.

**In our project.** The core novel features. Extracted per pixel per year from MODIS NDVI.

**Why.** Rice yield depends on when the crop grew, how long, and how vigorously. These metrics capture that directly. Published rice yield correlations with grain-filling NDVI exceed 0.75 (Son et al. 2014).

**Impact.** Transforms NDVI from a vegetation signal into an agronomically meaningful feature set.

---

### D2. Growing Degree Days (GDD)

**Definition.** Cumulative daily `max(0, T_mean − T_base)` where `T_base = 10°C` for rice.

**In our project.** Seasonal GDD aggregated June–October.

**Why.** Captures total thermal time, which drives crop development stages.

**Impact.** Explains inter-annual variation that calendar-based features miss.

---

### D3. Heat Stress Days

**Definition.** Count of days with `T_max > 35°C` during the flowering window (Aug 15–Sep 15 for Kharif rice in UP).

**Why.** Rice spikelet sterility rises sharply above 35°C during flowering. One day of extreme heat can tank yield.

**Impact.** Non-linear feature that linear models can use directly. XGBoost finds the threshold interaction automatically.

---

### D4. Dry Spell Count

**Definition.** Count of consecutive 7+ day windows with no rainfall during the Kharif season.

**In our project.** Weather feature.

**Why.** Even in a wet year, a long mid-season dry spell can kill the crop. The seasonal total hides this.

**Impact.** Captures risk that averages miss.

---

### D5. Rainfall Concentration Index (CV of monthly rainfall)

**Definition.** Coefficient of variation of monthly rainfall during Kharif.

**Why.** Even distribution of rainfall across months matters more than total.

**Impact.** Distinguishes a 1000mm season spread evenly from a 1000mm season concentrated in September.

---

### D6. Standardization (Z-score)

**Definition.** `x_std = (x − μ) / σ` using training-set statistics.

**In our project.** All features standardized before Ridge/Lasso/ElasticNet. Tree models are scale-invariant and skip this step.

**Why.** Regularization penalizes coefficients proportional to feature magnitude. Without standardization, large-scale features (rainfall in mm) are penalized more than small-scale features (NDVI 0–1).

**Impact.** Fair regularization across features. Reproducible via saved scaler artefact.

---

### D7. Variance Inflation Factor (VIF)

**Definition.** VIF_j = 1 / (1 − R²_j) where R²_j is from regressing feature j on all others. VIF > 10 flags high multicollinearity.

**In our project.** Run during feature engineering as a diagnostic. Features above threshold → consider dropping one.

**Impact.** Honest reporting of correlated features. Prevents linear model instability.

---

## Section E — Natural Language and Retrieval

### E1. Word Embeddings and Sentence Embeddings

**Definition.** Dense vector representations of text where semantic similarity corresponds to geometric proximity.

**In our project.** `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2` for sentence-level embeddings.

**Why.** Enables semantic search over the KB. Keyword search would miss "fertilizer" when the KB uses "nutrient management."

**Impact.** Users can ask questions in natural language and retrieve conceptually relevant passages.

---

### E2. Transformer Architecture

**Definition.** Neural network architecture built on self-attention. Replaces RNNs/CNNs for most NLP tasks. Multi-head attention lets each position attend to all other positions.

**In our project.** Underlies both the embedding model and the Llama 3 LLM.

**Why.** State of the art for text understanding and generation.

**Impact.** Enables high-quality retrieval and advisory generation on commodity hardware.

---

### E3. Large Language Model (LLM)

**Definition.** A transformer with billions of parameters trained on massive text. Capable of instruction-following, reasoning, and generation.

**In our project.** Llama 3 8B instruct, 4-bit quantized, running locally via Ollama.

**Why.** Converts structured yield predictions + retrieved passages into a natural-language advisory a non-expert can read.

**Impact.** The usability layer of the system. Without this, the output is a table of numbers.

---

### E4. Quantization (4-bit, GGUF format)

**Definition.** Compressing model weights from 16-bit to 4-bit by grouping and scaling. Trades minor accuracy loss for 4x less memory.

**In our project.** Llama 3 8B Q4_K_M = 4.7GB instead of 16GB.

**Why.** Fits in 8GB RAM. Enables CPU-only inference at usable speeds.

**Impact.** Makes offline deployment feasible on consumer hardware.

---

### E5. Retrieval-Augmented Generation (RAG)

**Definition.** Augment an LLM with an external knowledge base: at inference, retrieve relevant passages and inject them into the prompt. The LLM generates grounded in these passages rather than pretraining memory alone.

**In our project.** The advisory layer. Query → embed → cosine search → top-5 passages → prompt-stuffed into Llama 3.

**Why.**
1. Grounds the LLM in curated, region-specific agronomy.
2. Updatable: swap KB files and the advisory changes immediately, no retraining.
3. Auditable: every claim traces back to a passage.

**Impact.** Reduces hallucination. Enables localized advice without fine-tuning. Enables honest grounding score reporting.

---

### E6. Vector Database (ChromaDB)

**Definition.** A specialized database that indexes high-dimensional embeddings for fast approximate nearest neighbor search.

**In our project.** ChromaDB stores KB passage embeddings, serves top-k retrieval in sub-millisecond.

**Why.** Zero-config, file-backed, offline. Alternatives (Pinecone, Weaviate cloud) violate offline requirement.

**Impact.** Enables production RAG without cloud dependency.

---

### E7. HNSW (Hierarchical Navigable Small World)

**Definition.** An ANN (approximate nearest neighbor) algorithm. Builds a layered graph; search traverses from coarse to fine layers.

**In our project.** ChromaDB's default index for the KB.

**Why.** Sub-millisecond queries at our KB size. Scales to millions of vectors if needed.

**Impact.** Fast enough that retrieval latency is dominated by embedding the query, not searching.

---

### E8. Prompt Engineering (Constraint Prompting)

**Definition.** Designing input text to steer LLM behavior. Constraint prompting adds explicit rules the model should follow.

**In our project.** The advisory prompt specifies: "Use ONLY the following passages. For every claim, cite [n]. If passages do not cover a question, say 'Insufficient information.'"

**Why.** Even strong LLMs drift. Explicit constraints reduce hallucination.

**Impact.** Combined with post-hoc grounding verification, keeps the advisory honest.

---

### E9. Grounding / Faithfulness Metric

**Definition.** The fraction of generated statements that can be traced to the retrieved context, typically via cosine similarity threshold.

**In our project.** Each advisory sentence embedded, max cosine to any retrieved passage computed, threshold 0.6 for grounded.

**Why.** Prompt alone cannot guarantee grounding; post-hoc verification can measure it.

**Impact.** First-class user-facing metric. Low grounding is surfaced instead of hidden.

---

### E10. LLM-as-Judge

**Definition.** Using an LLM to evaluate the quality of another LLM's output on a rubric.

**In our project.** Secondary evaluation: ask Llama 3 to score 20 advisories on specificity, factual grounding, and actionability (1–5).

**Why.** No human agronomist evaluation available in a one-day sprint. LLM-as-judge is a published surrogate (Zheng et al. 2023).

**Impact.** Provides a quality signal beyond grounding score, with explicit acknowledgment of its limitations.

---

## Section F — Interpretability

### F1. SHAP (SHapley Additive exPlanations)

**Definition.** Attribution method based on Shapley values from cooperative game theory. For each prediction, assigns each feature a value representing its contribution.

**In our project.** Primary feature importance method on the final XGBoost model.

**Why.** Theoretically grounded (unique solution satisfying efficiency, symmetry, dummy, additivity). Per-sample explanations enable the Diagnostics tab.

**Impact.** Reviewers accept SHAP; agronomists can inspect specific predictions.

---

### F2. Permutation Importance

**Definition.** Shuffle one feature, measure the drop in model performance. Repeat per feature.

**In our project.** Sanity check alongside SHAP.

**Why.** Model-agnostic, no assumptions.

**Impact.** Agreement with SHAP strengthens the importance claims. Disagreement surfaces interesting cases.

---

### F3. Partial Dependence Plots (PDP)

**Definition.** For a given feature, show the model's average prediction as that feature varies, marginalizing the others.

**In our project.** Supplementary figures for the top-5 features by SHAP importance.

**Why.** Shows the functional shape (linear, threshold, saturating) the model learned.

**Impact.** Agronomists can validate the learned relationships against domain knowledge.

---

## Section G — Validation and Metrics

### G1. R² (Coefficient of Determination)

**Definition.** `1 − SS_res/SS_tot`. Fraction of variance explained.

**In our project.** Primary regression metric. Reported for both train and test.

**Why.** Standard, intuitive, directly comparable across papers.

**Impact.** The headline number for the yield prediction result.

---

### G2. RMSE (Root Mean Squared Error)

**Definition.** `sqrt(mean((y_true − y_pred)²))`. In units of y.

**In our project.** Secondary metric. Reported in t/ha.

**Why.** Directly interpretable in yield units.

**Impact.** Allows agronomists to assess if the error is practically acceptable (sub-tonne per ha is good).

---

### G3. MAE (Mean Absolute Error)

**Definition.** `mean(|y_true − y_pred|)`. Robust to outliers.

**In our project.** Secondary metric alongside RMSE.

**Why.** When RMSE and MAE diverge, outlier years are dominating RMSE.

**Impact.** Helps diagnose years the model handled poorly.

---

### G4. Precision@K / Recall@K (Retrieval)

**Definition.** Precision@K = fraction of top-K retrieved that are relevant. Recall@K = fraction of all relevant that are in top-K.

**In our project.** On 20 handcrafted test queries with marked relevant passages.

**Why.** Standard IR metrics. Directly measure retrieval quality.

**Impact.** Justifies the embedding model choice. Baseline for future retrieval improvements.

---

## Section H — Advanced Topics (Used or Discussed)

### H1. Hyperparameter Optimization (Optuna / Bayesian Optimization)

**Definition.** Searching HP space more efficiently than grid or random search by modeling the objective and selecting promising points.

**In our project.** Optuna with TPE sampler for XGBoost HPs.

**Why.** Finds good HPs in 50 trials where grid search needs thousands.

**Impact.** Faster, more thorough tuning on compute budget.

---

### H2. ONNX (Open Neural Network Exchange)

**Definition.** Open format for storing trained models in a framework-agnostic way.

**In our project.** Export the final XGBoost model to ONNX for portable, fast CPU inference.

**Why.** 10x faster than Python-side predict for batch scoring. Deployable to edge devices as future work.

**Impact.** Opens the door to mobile deployment.

---

### H3. MLflow or DVC (Experiment Tracking / Data Versioning)

**Definition.** MLflow tracks runs, metrics, and artefacts. DVC versions data files like git versions code.

**In our project.** DVC for data versioning. MLflow optional for experiment tracking.

**Why.** Reproducibility. Collaboration if the project expands.

**Impact.** Fresh clone can replay any historical experiment exactly.

---

### H4. Data Augmentation (via Crop Simulation)

**Definition.** Synthetic data generated by a process-based model (AquaCrop, DSSAT) to supplement observations.

**In our project.** Stretch goal. 500 AquaCrop runs with varied weather for Kanpur.

**Why.** Covers scenarios absent from history (worst-case droughts under climate change).

**Impact.** Potential test-R² improvement + a distinctive methodological contribution for the paper.

---

### H5. Transfer Learning (Discussed, Not Used)

**Definition.** Pretrain on a large related dataset, fine-tune on target.

**In our project.** Discussed in future work. Could pretrain on all-India satellite+weather and fine-tune for Kanpur.

**Impact.** Suggests a path to generalization across districts and crops.

---

### H6. Fine-Tuning LLMs (Discussed, Not Used)

**Definition.** Continuing to train an LLM on a task-specific dataset.

**In our project.** Not used. RAG is cheaper, interpretable, and updatable.

**Why.** Fine-tuning on 300 KB passages would overfit catastrophically. RAG is the right tool.

**Impact.** Justifies the architectural choice in the methodology section.

---

### H7. Reinforcement Learning (Discussed, Not Used)

**Definition.** Learning via rewards from an environment.

**In our project.** Not used. We have no reward signal from the environment.

**Why.** Supervised learning fits the data we have.

**Impact.** Noted in future work: an RL agent recommending irrigation given weather forecasts could be a follow-up.

---

### H8. Climate Change Projections (Future Work)

**Definition.** CMIP6 downscaled scenarios project Kanpur temperature up by 2–3°C by 2050.

**In our project.** Discussed in future work. The current model, if applied with projected climate, forecasts yield declines.

**Impact.** Policy-relevant extension in the paper's conclusion.

---

## Section I — Why these specific concepts matter together

The system succeeds because these concepts combine, not because any one is unique:

- **Satellite data + phenology features** solve the small-N problem.
- **XGBoost + spatiotemporal CV + SHAP** give an honest, interpretable predictor.
- **Bayesian + bootstrap intervals** give defensible uncertainty.
- **RAG + grounding score** give honest LLM output.
- **Local quantized LLM + offline vector store** keep the system deployable rurally.
- **Decision log + reference discipline + reproducibility tests** keep the paper credible.

The novelty is **not** in any individual technique. It is in combining proven techniques into a honest, reproducible, offline-first pipeline that most agricultural AI papers claim but do not deliver.
