# Paper framing (ADR-032)

## Claim
Spatial disaggregation of district rice-yield statistics to MODIS pixels for Kanpur Nagar and Kanpur Dehat, plus a fully offline (local LLM + local embeddings) advisory system. Not temporal yield forecasting.

## Primary results table (headline: district-aggregated, leave-one-year-out vs APY, N = 26)

| Predictor | Pooled R^2 | Pooled r | Pooled RMSE (t/ha) | Nagar R^2 | Dehat R^2 |
|---|---|---|---|---|---|
| XGBoost, full features | 0.379 | 0.62 | 0.319 | 0.47 | 0.18 |
| XGBoost without `raw_peak_ndvi` | 0.136 | 0.40 | 0.377 | 0.13 | 0.01 |
| Per-district linear time trend | **0.464** | 0.69 | 0.297 | 0.38 | 0.47 |
| District mean of other years | -0.091 | n/a | 0.423 | -0.17 | -0.17 |

Caveats that must sit under the table: the calibration constants were fitted on all 13 APY years (no nested re-calibration); N = 13 per district; the pixel-level targets are calibrated NDVI (ADR-028, ADR-031). Source numbers: `models/results/phase3_report.md`.

## Limitations to state (user-specified)
- Kanpur Dehat calibration is much weaker than Nagar (R^2 0.19, leave-one-year-out R^2 -0.09 vs 0.47 / 0.32). Likely causes to discuss: cropland-mask heterogeneity, noise in reported yields, N = 13.
- Peak NDVI 0.56-0.73 is a ceiling set by 16-day MODIS composites and mixed cropland; not fixable with current data.

## Features considered but not used (one sentence in the main text)
AquaCrop-simulated yield: uncalibrated, and perfectly correlated with calendar year, so uninterpretable (supplementary appendix "Exploratory, not recommended").

## Discussion subsection: "Why satellite phenology does not improve on a time trend in this region"

Evidence status below separates what is verified from what is not. Every citation was checked to exist (see `knowledge/REFERENCES_AUDIT.md`); **support** was judged from abstracts only and needs a full-text read in Phase 7.

### H1. Varietal and management intensification is invisible to NDVI
- Own evidence: APY correlates with calendar year at r = 0.74-0.78 in both districts; a linear trend predicts held-out years better (R^2 0.46) than the satellite/weather model (0.38).
- Candidate citation: Lobell (2013, R5), a review of satellite-based yield-gap analysis (abstract: persistence of yield-controlling factors across years). Support is **indirect**; it does not state that NDVI misses intensification.
- Gap: no verified source in REFERENCES.md states the trend/technology argument directly. A search for one found nothing usable.

### H2. Irrigation buffers rainfall variability
- Own evidence: Kharif rainfall CV is 0.20-0.21 (537-1,110 mm) with two drought years (2014, 2015: about 550 mm, 17-24 heat-stress days). APY dipped in those years in Nagar (2.32, 2.27) and Dehat (2.47, 2.63) against 2.5-3.0 around them, a small dip next to the rainfall swing; this is consistent with, but does not prove, irrigation buffering. (The earlier wording "moderate inter-annual climate variability" was dropped: the data show large rainfall variability.)
- **[CITATION NEEDED]** Phase 7 full-text search for Indo-Gangetic Plain irrigation-mediated yield-stability references (see `knowledge/PHASE7_AUDIT.md`). No verified source among R1-R24 supports this hypothesis. van Klompenburg et al. (2020, R1) reviews features used (temperature, rainfall, soil type most common) and does not support it. A 2024 *Earth's Future* crop-model study of rice in Uttar Pradesh surfaced in a search; its details are unverified and it is not cited.

### H3. MODIS 250 m resolution and cloud contamination set a resolution floor
- Own evidence: SummaryQA = 3 (cloudy) is 45% of observations on average over 2010-2024 in the DOY 150-300 window (range 34-58% by year) and 37% in the extended DOY 150-330 window (28-48%); QA <= 1 is 55% and 63% respectively. The "54% cloud contamination" figure in the decision text matches no measured average (it is close to the single worst years, 2018 at 58% and 2024 at 55%, in the 9-composite window) and should be replaced by these numbers in the manuscript. Other evidence: 16-day composites; peak NDVI medians 0.56-0.73; pixel R^2 falls from 0.90 to 0.60 without the unsmoothed peak.
- Citation: Son et al. 2014 (R4, *Int. J. Remote Sens.* 34(20)) used MODIS EVI/LAI for rice yield in the Mekong Delta, filtered noise with empirical mode decomposition, and found the strongest correlations in the ripening period (0.70 and 0.74). Support is **direct but partial** (MODIS noise and filtering needs; yield correlations of that size). Lobell (2013, R5) for resolution is **unverified**.

## Notes for Phase 7
- "Son 2014" is two different papers: R4 (yield prediction, IJRS) and R17 (phenology classification with EMD, *Remote Sensing*). Cite with explicit keys.
- Replace the 54% cloud figure with the measured values in H3.
