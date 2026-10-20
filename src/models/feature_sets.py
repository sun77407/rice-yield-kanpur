"""Model feature sets (ADR-028: no calibration-derived NDVI features as inputs)."""

from __future__ import annotations

# Target = alpha_d + beta_d * grain_filling_ndvi, so these carry the target (ADR-028).
FORBIDDEN = frozenset({"grain_filling_ndvi", "integrated_ndvi", "peak_ndvi"})

PHENOLOGY = ["greenup_day", "peak_ndvi_day", "senescence_day", "length_of_season", "raw_peak_ndvi"]
WEATHER = [
    "kharif_rainfall_mm",
    "monsoon_onset_day",
    "dry_spell_count",
    "kharif_tmax_mean",
    "kharif_tmin_mean",
    "heat_stress_days",
    "kharif_rh_mean",
    "rain_cv",
    "gdd_kharif",
]
SECONDARY = ["kharif_gwetroot_mean", "imd_kharif_rainfall_mm"]  # IMD kept per ADR-026
SOIL = ["ph", "soc", "clay_pct", "sand_pct", "cec", "nitrogen"]
DISTRICT = ["district_kanpur_dehat"]  # one dummy only: the pair is perfectly collinear

FULL = [*PHENOLOGY, *WEATHER, *SECONDARY, *SOIL, *DISTRICT]
# Linear / Lasso / ElasticNet: gdd_kharif is r=0.99 with kharif_tmax_mean, so drop it and keep
# tmax and rh (ADR-028). Ridge, RF, XGBoost, LightGBM, GBR, Bayesian models use FULL.
LINEAR = [f for f in FULL if f != "gdd_kharif"]
# Ablations required by ADR-028: without the noisiest NDVI proxy, and with no satellite at all.
NO_RAW_PEAK = [f for f in FULL if f != "raw_peak_ndvi"]
NO_SATELLITE = [f for f in FULL if f not in PHENOLOGY]

FEATURE_SETS = {
    "linear": LINEAR,
    "full": FULL,
    "no_raw_peak": NO_RAW_PEAK,
    "no_satellite": NO_SATELLITE,
}
