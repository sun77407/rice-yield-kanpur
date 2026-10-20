# References — Peer-Reviewed Sources Used in This Project

Fifteen peer-reviewed papers, each with (a) full citation, (b) key finding, (c) specific contribution to our project, and (d) BibTeX key.

**Verification protocol.** Every entry below has been curated from well-known works in the field. The executing Claude Code agent **MUST** verify each DOI/URL via web search before citing in the manuscript. Do not cite from this file blindly; treat it as a starting roster and confirm each entry is real and accurately summarized before using it. If verification fails for any entry, drop it and find a replacement via web search.

Also: the agent **MUST** append any additional reference discovered during the build to this file with the same format, before citing it in the paper.

---

## 1. Yield prediction methodology — satellite-based

### R1 — van Klompenburg et al. 2020

**Citation.** van Klompenburg, T., Kassahun, A., & Catal, C. (2020). *Crop yield prediction using machine learning: A systematic literature review.* Computers and Electronics in Agriculture, 177, 105709. https://doi.org/10.1016/j.compag.2020.105709

**Key finding.** A systematic review of 50 papers 2008–2019 on crop yield ML. CNN and LSTM dominated recent deep learning approaches; XGBoost and Random Forest dominated tabular. NDVI/EVI were the most used satellite features. Lack of standard datasets and reproducibility was the field's major gap.

**Our use.** Cited in Introduction and Related Work. Justifies our tabular approach over deep learning (small-N, interpretable). Supports NDVI as feature. Motivates our reproducibility focus as a contribution.

**BibTeX key.** `vanKlompenburg2020`

---

### R2 — Khaki & Wang 2019

**Citation.** Khaki, S., & Wang, L. (2019). *Crop yield prediction using deep neural networks.* Frontiers in Plant Science, 10, 621. https://doi.org/10.3389/fpls.2019.00621

**Key finding.** DNN beat Lasso, shallow NN, and regression trees on US maize yield prediction. Showed that feature engineering (soil, weather, management) matters as much as model choice.

**Our use.** Related Work. Shows that deep learning is possible but requires N in the thousands. Supports our pixel-level approach to achieve similar N.

**BibTeX key.** `khaki2019`

---

### R3 — You et al. 2017

**Citation.** You, J., Li, X., Low, M., Lobell, D., & Ermon, S. (2017). *Deep Gaussian Process for crop yield prediction based on remote sensing data.* In AAAI 2017. https://doi.org/10.1609/aaai.v31i1.11172

**Key finding.** Combined CNN on MODIS surface reflectance histograms with Gaussian Process for county-level US soybean yield. Beat USDA baseline with months lead time.

**Our use.** Related Work. Precedent for satellite-only yield prediction. Our method is simpler (phenology features vs CNN) but same spirit.

**BibTeX key.** `you2017`

---

### R4 — Son et al. 2014

**Citation.** Son, N.-T., Chen, C.-F., Chen, C.-R., Chang, L.-Y., Duc, H.-N., & Nguyen, L.-D. (2014). *Prediction of rice crop yield using MODIS EVI-LAI data in the Mekong Delta, Vietnam.* International Journal of Remote Sensing, 34(20), 7275–7292. https://doi.org/10.1080/01431161.2013.818258

**Key finding.** EVI during grain-filling correlated with rice yield at r > 0.75 across multiple years in the Mekong Delta. LAI less useful than EVI due to saturation.

**Our use.** Cited as justification for grain-filling NDVI/EVI as yield proxy. Our calibration ADR (ADR-002) is grounded here.

**BibTeX key.** `son2014`

---

### R5 — Lobell 2013

**Citation.** Lobell, D. B. (2013). *The use of satellite data for crop yield gap analysis.* Field Crops Research, 143, 56–64. https://doi.org/10.1016/j.fcr.2012.08.008

**Key finding.** Satellite data can estimate yield gaps (potential minus actual) at field scale, enabling targeted intervention. NDVI, EVI, and biomass-based indices all work; choice depends on cloud frequency and growing region.

**Our use.** Related Work. Foundational for the satellite-based yield prediction tradition.

**BibTeX key.** `lobell2013`

---

## 2. ML methodology

### R6 — Chen & Guestrin 2016

**Citation.** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* In KDD '16, 785–794. https://doi.org/10.1145/2939672.2939785

**Key finding.** Introduced XGBoost: regularized gradient boosting with sparse-aware splitting, out-of-core training, and parallel tree construction. SOTA on many Kaggle competitions.

**Our use.** Primary citation for XGBoost (our main model). Methodology section.

**BibTeX key.** `chen2016xgboost`

---

### R7 — Lundberg & Lee 2017

**Citation.** Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* In NeurIPS 2017, 4765–4774.

**Key finding.** SHAP values are the unique attribution method satisfying efficiency, symmetry, dummy, and additivity. Model-agnostic in principle; fast exact computation for tree ensembles.

**Our use.** Methodology section — our interpretability approach. SHAP summary and dependence plots in Results.

**BibTeX key.** `lundberg2017`

---

### R8 — Breiman 2001

**Citation.** Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

**Key finding.** Random Forests combine bootstrap aggregation with random feature subset selection at splits. Robust to overfitting and tuning.

**Our use.** Baseline model citation.

**BibTeX key.** `breiman2001`

---

## 3. Bayesian methods

### R9 — Gelman et al. 2013 (BDA3)

**Citation.** Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall/CRC. ISBN 9781439840955.

**Key finding.** Standard reference for applied Bayesian analysis including hierarchical models, prior choice, and posterior predictive checks.

**Our use.** Methodology citation for hierarchical model (ADR C3). Prior sensitivity discussion.

**BibTeX key.** `gelman2013`

---

### R10 — Hoffman & Gelman 2014

**Citation.** Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo.* Journal of Machine Learning Research, 15, 1593–1623.

**Key finding.** NUTS auto-tunes HMC step sizes, eliminating a key tuning parameter. Dominant modern MCMC algorithm.

**Our use.** Methodology — our MCMC algorithm for the PyMC hierarchical model.

**BibTeX key.** `hoffman2014nuts`

---

## 4. NLP, RAG, and LLMs

### R11 — Lewis et al. 2020

**Citation.** Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* In NeurIPS 2020.

**Key finding.** Introduced RAG: parametric (seq2seq) + non-parametric (dense retrieval) memory. Outperformed parametric-only on knowledge-intensive QA.

**Our use.** Primary citation for RAG. Introduction and Methodology.

**BibTeX key.** `lewis2020rag`

---

### R12 — Reimers & Gurevych 2019

**Citation.** Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* In EMNLP 2019, 3982–3992. https://doi.org/10.18653/v1/D19-1410

**Key finding.** Sentence-BERT produces semantically meaningful sentence embeddings 1000x faster than BERT cross-encoding. Enables semantic search at scale.

**Our use.** Methodology citation for sentence-transformers (our embedding model library).

**BibTeX key.** `reimers2019sbert`

---

### R13 — Es et al. 2023 (RAGAS)

**Citation.** Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217. https://arxiv.org/abs/2309.15217

**Key finding.** Framework for evaluating RAG systems on four dimensions: faithfulness, answer relevance, context precision, context recall — all computed via LLM judgments without ground truth.

**Our use.** Methodology — grounding score (faithfulness) methodology. ADR-010 derives from this.

**BibTeX key.** `es2023ragas`

---

### R14 — Zheng et al. 2023 (LLM-as-Judge)

**Citation.** Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* In NeurIPS 2023 Datasets and Benchmarks Track. https://arxiv.org/abs/2306.05685

**Key finding.** Strong LLMs can grade responses from other LLMs with Spearman correlation ≈ 0.8 against human judgments, enabling scalable evaluation.

**Our use.** Justifies our LLM-as-judge rubric for advisory evaluation (ADR Section E10).

**BibTeX key.** `zheng2023llmjudge`

---

## 5. Data and tooling

### R15 — Funk et al. 2015 (CHIRPS)

**Citation.** Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S., Husak, G., Rowland, J., Harrison, L., Hoell, A., & Michaelsen, J. (2015). *The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes.* Scientific Data, 2, 150066. https://doi.org/10.1038/sdata.2015.66

**Key finding.** CHIRPS v2.0: 0.05° daily precipitation combining satellite estimates with ground stations, 1981–present. Validated extensively over Africa and South Asia.

**Our use.** Data section citation for CHIRPS (our primary rainfall source per ADR-003).

**BibTeX key.** `funk2015chirps`

---

### R16 — Chen et al. 2004 (Savitzky-Golay NDVI smoothing)

**Citation.** Chen, J., Jönsson, P., Tamura, M., Gu, Z., Matsushita, B., & Eklundh, L. (2004). *A simple method for reconstructing a high-quality NDVI time-series data set based on the Savitzky-Golay filter.* Remote Sensing of Environment, 91(3-4), 332-344. https://doi.org/10.1016/j.rse.2004.03.014

**Key finding.** An iterative Savitzky-Golay filter that pulls cloud-depressed NDVI toward the upper envelope of the annual cycle outperformed BISE and Fourier fitting on 10-day SPOT-VGT composites. (Verified via web search, 2026-10-20.)

**Our use.** Methodology citation for Savitzky-Golay NDVI smoothing in Phase 2 feature engineering (ADR-025). We use the plain (non-iterative) filter, order 2, window 5, after linear gap interpolation.

**BibTeX key.** `chen2004sg`

---


## Additional sources for the RAG knowledge base (not cited in paper, but included in KB corpus)

### KB-1 — IRRI Rice Knowledge Bank
URL: `http://www.knowledgebank.irri.org/`. Comprehensive rice cultivation protocols from the International Rice Research Institute.

### KB-2 — ICAR-IIRR Hyderabad
URL: `https://icar-iirr.org/`. Indian Institute of Rice Research publications.

### KB-3 — ICAR-IIFSR Modipuram
URL: `https://iifsr.icar.gov.in/`. Rice-wheat system research for Indian plains.

### KB-4 — KVK Kanpur Annual Reports
Krishi Vigyan Kendra Kanpur annual action plans and farmer advisory bulletins.

### KB-5 — UP Agricultural Department Package of Practices
URL: `http://upagriculture.com/`. State-level cultivation recommendations.

### KB-6 — FAO Rice Production Manual
FAO rice cultivation guide for South Asia; URL: `https://www.fao.org/`.

### KB-7 — ICAR-CRRI Cuttack
Central Rice Research Institute publications.

---

## Reference discipline

Per ADR-015: every reference above must be verified before citation in the manuscript. The executing agent will:

1. Search for each DOI/URL via web search before including the citation.
2. If a paper is found but with a different title/year/author, update this file.
3. If a paper cannot be found, mark it STATUS: UNVERIFIED and find a replacement.
4. Any new reference discovered during the build is added here with the same four fields before citation.
5. The `paper/references.bib` file is generated from this document — never written independently.

---

## Expected contribution of this reference list to the project

- **R1, R2, R3, R4, R5** — establish the satellite-based yield prediction tradition we build on and extend to Kanpur rice.
- **R6, R7, R8** — justify our ML model choices and interpretability method.
- **R9, R10** — justify our Bayesian uncertainty quantification.
- **R11, R12, R13, R14** — justify our RAG architecture, embeddings, grounding metric, and LLM-as-judge evaluation.
- **R15** — justifies our weather data choice.
- **KB-1 through KB-7** — populate the knowledge base; cited in the advisory outputs, not in the paper body.

Together these provide complete methodological cover for every claim in the manuscript. No claim should appear in the paper without a reference from this list (or a verified addition to it).
