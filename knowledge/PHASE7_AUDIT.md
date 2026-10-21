# Phase 7 pre-submission audit checklist

Items verified only partly (see `REFERENCES_AUDIT.md`). None blocks earlier phases.

- [ ] (a) **R5 DOI.** Lobell (2013), *Field Crops Research* 143, 56-64: confirm DOI 10.1016/j.fcr.2012.08.008 from the publisher page.
- [ ] (b) **R7 page range.** Lundberg & Lee (2017), NeurIPS 30: confirm pages 4765-4774 from the NeurIPS proceedings or DBLP.
- [ ] (c) **R14 venue.** Zheng et al. (2023), "Judging LLM-as-a-Judge": confirm the NeurIPS 2023 Datasets and Benchmarks Track listing (otherwise cite as arXiv:2306.05685).
- [ ] (d) **H2 citation hunt.** Full-text search for Indo-Gangetic Plain, irrigation-mediated yield stability references to support "irrigation buffers rainfall variability" (marked [CITATION NEEDED] in `paper/FRAMING.md`). Also look for a direct source for H1 (technology/management trend not visible to NDVI).
- [ ] (e) **R21 DOI.** Optuna KDD 2019 DOI 10.1145/3292500.3330701 (currently cited by arXiv id).
- [ ] (f) **Full-text check of support.** Re-read R4, R5 and R17 in full to confirm the claims `paper/FRAMING.md` attributes to them (only abstract-level support was checked).
- [ ] (g) **ADR-010 / RAGAS.** ADR-010 says a 0.6 cosine threshold is a "published heuristic (Es et al. 2023)". RAGAS faithfulness uses LLM claim verification, not a cosine threshold; do not cite R13 for the 0.6 value (see ADR-034).
- [ ] (h) **Cloud-contamination figures.** Use the measured values in the Discussion: 45% cloudy (QA = 3) in DOY 150-300, 37% in DOY 150-330.

- (i) Verify the "3-4 t/ha typical grain yield for Pusa Basmati 1 in UP" sentence added to KB passage A6 (SOURCES.md corpus corrections) against a citable source, or reword it.
