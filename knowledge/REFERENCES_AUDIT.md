# References audit (2026-10-21)

Method: one web search per entry for title, authors, venue, year, volume and pages, plus the DOI/ISBN/arXiv id where the entry gives one. "Verified" means the search returned a record (publisher, ACL/NeurIPS/JMLR/arXiv/USGS/PMC or similar) matching the entry. "Abstract-level" means I read search-result summaries, not the full text; claims about what a paper concludes must be re-checked against the full text in Phase 7.

| ID | Entry | Status | Notes |
|---|---|---|---|
| R1 | van Klompenburg et al. 2020, Comput. Electron. Agric. 177, 105709 | Verified | DOI matches; CC BY 4.0. 50 studies analysed; temperature, rainfall, soil type most-used features; ANN most-used algorithm. |
| R2 | Khaki & Wang 2019, Front. Plant Sci. 10, 621 | Verified | DOI matches (PMC6540942). Maize hybrids, Syngenta Crop Challenge, not rice. |
| R3 | You et al. 2017, AAAI | Verified | DOI matches; AAAI vol. 31 no. 1; pages in some listings 4559-4566. Soybean, US counties. |
| R4 | Son et al. 2014, Int. J. Remote Sens. 34(20), 7275-7292 | Verified | DOI matches. MODIS EVI + LAI, Mekong Delta; correlations 0.70/0.74 in ripening period; EMD noise filtering. |
| R5 | Lobell 2013, Field Crops Res. 143, 56-64 | **Partly verified** | Authors, venue, volume and pages confirmed (NASA MODIS pubs list, Stanford FSI). **DOI 10.1016/j.fcr.2012.08.008 not confirmed** by the search. |
| R6 | Chen & Guestrin 2016, KDD '16, 785-794 | Verified | DOI matches; arXiv 1603.02754. |
| R7 | Lundberg & Lee 2017, NeurIPS 30 | **Partly verified** | Paper confirmed (NeurIPS, arXiv 1705.07874). **Pages 4765-4774 not confirmed.** |
| R8 | Breiman 2001, Mach. Learn. 45(1), 5-32 | Verified | DOI matches. |
| R9 | Gelman et al. 2013, BDA3 | Verified | ISBN 9781439840955 is the hardback of the 3rd edition, Chapman & Hall/CRC. |
| R10 | Hoffman & Gelman 2014, JMLR 15, 1593-1623 | Verified | JMLR record matches; arXiv 1111.4246. |
| R11 | Lewis et al. 2020, NeurIPS | Verified (pages unverified) | arXiv 2005.11401. NeurIPS vol. 33 pages 9459-9474 appear only on a third-party page. Entry gives no pages; keep it that way. |
| R12 | Reimers & Gurevych 2019, EMNLP, 3982-3992 | Verified, **but no longer the embedding model used** | ACL Anthology confirms DOI and pages. The project embeds with `nomic-embed-text` (ADR-008). Keep R12 only for the general idea of sentence embeddings compared by cosine similarity; cite R18 for the model actually used. |
| R13 | Es et al. 2023, RAGAS, arXiv 2309.15217 | Verified | Title appears as "Ragas" in the arXiv listing. |
| R14 | Zheng et al. 2023, LLM-as-a-Judge, arXiv 2306.05685 | Verified (venue unverified) | arXiv confirmed. The NeurIPS 2023 Datasets and Benchmarks venue was not confirmed by this search; cite as arXiv unless checked. |
| R15 | Funk et al. 2015, Sci. Data 2, 150066 | Verified | USGS/NOAA records match; DOI matches. |
| R16 | Chen et al. 2004, Remote Sens. Environ. 91(3-4), 332-344 | Verified | Confirmed earlier (Lund University record, NASA MODIS pubs list). |

## Additions (appended to REFERENCES.md as R17-R24)

| ID | Entry | Status | Used for |
|---|---|---|---|
| R17 | Son et al. 2014, Remote Sens. 6(1), 135-156 (phenology-based rice classification, EMD smoothing) | Verified | Precedent for smoothing MODIS series (ADR-025). Distinct paper from R4. |
| R18 | Nussbaum et al. 2024, Nomic Embed, arXiv 2402.01613 (accepted at TMLR) | Verified | The embedding model (ADR-008). |
| R19 | Qwen Team 2024, Qwen2.5 Technical Report, arXiv 2412.15115 | Verified | The primary LLM family (ADR-007). The model used is a 14B instruct variant run in q6_K quantisation through Ollama. |
| R20 | Salvatier, Wiecki & Fonnesbeck 2016, PeerJ Comput. Sci. 2, e55, doi 10.7717/peerj-cs.55 | Verified | PyMC. The code uses PyMC 6 (not PyMC3); cite the software version in the methods. |
| R21 | Akiba et al. 2019, Optuna, KDD 2019 (arXiv 1907.10902) | **Partly verified** | Hyperparameter tuning. KDD DOI 10.1145/3292500.3330701 not confirmed; cite the arXiv id until it is. |
| R22 | Poggio et al. 2021, SOIL 7, 217-240, doi 10.5194/soil-7-217-2021 | Verified | SoilGrids 2.0. |
| R23 | Zanaga et al. 2022, ESA WorldCover 10 m 2021 v200, doi 10.5281/zenodo.7254221 | Verified | Cropland mask. |
| R24 | Savitzky & Golay 1964, Anal. Chem. 36(8), 1627-1639 | Verified (DOI from a listing: 10.1021/ac60214a047, not independently confirmed) | The filter itself. |

## Findings that affect the paper
1. Two different "Son 2014" papers: R4 (yield, IJRS) and R17 (phenology, Remote Sensing). ADR-025 uses R17; the Discussion subsection in `paper/FRAMING.md` uses R4.
2. R12 is no longer the right embedding citation (ADR-008 override); R18 replaces it for the model.
3. The decision text mapped van Klompenburg 2020, Son 2014 and Lobell 2013 onto the three hypotheses. Abstract-level support: only H3 has a direct citation (R4). H1 has only indirect support (R5). H2 has none among R1-R24, and the data contradict its wording (see `paper/FRAMING.md`). Missing citations for H1 and H2 should be searched for in Phase 7 and verified before use; one search each this session found no usable source (a 2024 *Earth's Future* study of rice in Uttar Pradesh surfaced; its details are unverified and not added).
4. PyMC, Optuna, XGBoost and SHAP are cited by their papers; the exact software versions used are in `uv.lock`.
5. Unconfirmed items to resolve from the publisher pages before submission: R5 DOI, R7 pages, R14 venue, R21 DOI.
