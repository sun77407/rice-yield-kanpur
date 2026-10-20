## Clipping (ADR-029)
| subset | n | pct_below_0.5 | pct_above_5.0 | pct_clipped | raw_min | raw_p01 | raw_median | raw_p99 | raw_max |
|---|---|---|---|---|---|---|---|---|---|
| all | 1394144 | 0.254 | 0.072 | 0.326 | -2.416 | 1.066 | 2.935 | 4.447 | 6.110 |
| train_2010_2020 | 1022366 | 0.290 | 0.050 | 0.340 | -2.416 | 1.014 | 2.906 | 4.395 | 5.705 |
| test_2021_2024 | 371778 | 0.157 | 0.131 | 0.288 | -2.316 | 1.225 | 3.019 | 4.559 | 6.110 |
| KanpurDehat | 750634 | 0.013 | 0.000 | 0.013 | -0.539 | 1.916 | 3.004 | 4.040 | 4.810 |
| KanpurNagar | 643510 | 0.536 | 0.155 | 0.692 | -2.416 | 0.770 | 2.793 | 4.642 | 6.110 |

## A. Pixel-level test R2 (2021-2024)
| model | R2 all | RMSE all | MAE all | R2 Nagar | RMSE Nagar | MAE Nagar | R2 Dehat | RMSE Dehat | MAE Dehat | CV R2 |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.724 | 0.351 | 0.262 | 0.746 | 0.413 | 0.319 | 0.669 | 0.288 | 0.213 | nan |
| ridge | 0.724 | 0.351 | 0.262 | 0.746 | 0.413 | 0.319 | 0.669 | 0.288 | 0.213 | nan |
| lasso | 0.702 | 0.365 | 0.272 | 0.725 | 0.430 | 0.332 | 0.646 | 0.298 | 0.220 | nan |
| elasticnet | 0.694 | 0.370 | 0.278 | 0.717 | 0.436 | 0.338 | 0.636 | 0.302 | 0.226 | nan |
| bayesian_ridge | 0.724 | 0.351 | 0.262 | 0.746 | 0.413 | 0.320 | 0.669 | 0.288 | 0.212 | nan |
| random_forest | 0.897 | 0.214 | 0.153 | 0.901 | 0.258 | 0.187 | 0.887 | 0.168 | 0.123 | 0.929 |
| gbr | 0.890 | 0.221 | 0.160 | 0.896 | 0.264 | 0.194 | 0.874 | 0.177 | 0.132 | 0.925 |
| xgboost | 0.897 | 0.214 | 0.154 | 0.901 | 0.257 | 0.189 | 0.887 | 0.168 | 0.124 | 0.935 |
| lightgbm | 0.894 | 0.218 | 0.156 | 0.898 | 0.262 | 0.191 | 0.883 | 0.171 | 0.127 | 0.934 |
| pymc_hierarchical | 0.720 | 0.354 | 0.264 | 0.748 | 0.411 | 0.317 | 0.651 | 0.296 | 0.219 | nan |

## B. raw_peak_ndvi sensitivity (XGBoost)
| model | R2 all | RMSE all | MAE all | R2 Nagar | RMSE Nagar | MAE Nagar | R2 Dehat | RMSE Dehat | MAE Dehat | CV R2 |
|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | 0.897 | 0.214 | 0.154 | 0.901 | 0.257 | 0.189 | 0.887 | 0.168 | 0.124 | 0.935 |
| xgb_no_raw_peak | 0.596 | 0.425 | 0.317 | 0.609 | 0.512 | 0.392 | 0.560 | 0.332 | 0.253 | nan |
| xgb_no_satellite | 0.069 | 0.645 | 0.505 | 0.087 | 0.782 | 0.626 | 0.015 | 0.497 | 0.402 | nan |
| xgb_with_aquacrop | 0.892 | 0.220 | 0.156 | 0.900 | 0.259 | 0.188 | 0.871 | 0.180 | 0.129 | nan |
| xgb_no_raw_peak_with_aquacrop | 0.621 | 0.411 | 0.313 | 0.645 | 0.488 | 0.379 | 0.561 | 0.332 | 0.257 | nan |

## C. District-aggregated vs APY: leave-one-year-out (HEADLINE)

| model | N | pooled r | pooled R2 | pooled RMSE | Nagar r | Nagar R2 | Nagar RMSE | Dehat r | Dehat R2 | Dehat RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 26 | 0.565 | 0.308 | 0.337 | 0.619 | 0.366 | 0.316 | 0.404 | 0.138 | 0.357 |
| ridge | 26 | 0.565 | 0.308 | 0.337 | 0.619 | 0.366 | 0.316 | 0.404 | 0.138 | 0.357 |
| xgboost | 26 | 0.618 | 0.379 | 0.319 | 0.689 | 0.471 | 0.289 | 0.433 | 0.183 | 0.347 |
| xgb_no_raw_peak | 26 | 0.397 | 0.136 | 0.377 | 0.390 | 0.128 | 0.371 | 0.230 | 0.009 | 0.382 |
| xgb_no_satellite | 26 | 0.441 | 0.190 | 0.365 | 0.321 | 0.063 | 0.385 | 0.454 | 0.199 | 0.344 |
| xgb_with_aquacrop | 26 | 0.620 | 0.381 | 0.319 | 0.694 | 0.478 | 0.287 | 0.430 | 0.181 | 0.348 |
| xgb_no_raw_peak_with_aquacrop | 26 | 0.468 | 0.206 | 0.361 | 0.465 | 0.204 | 0.354 | 0.327 | 0.085 | 0.367 |

Reference (predict the district mean of the other APY years): pooled R2 -0.091, RMSE 0.423; Nagar R2 -0.174, Dehat R2 -0.174

Reference (per-district linear time trend, leave-one-year-out): pooled R2 0.464, RMSE 0.297, r 0.692; Nagar R2 0.379, Dehat R2 0.472

### C2. All 13 APY years, in-sample for 2010-2020 (not a test)

| model | N | pooled r | pooled R2 | pooled RMSE | Nagar r | Nagar R2 | Nagar RMSE | Dehat r | Dehat R2 | Dehat RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 26 | nan | 0.377 | 0.320 | 0.665 | 0.430 | 0.300 | 0.473 | 0.223 | 0.339 |
| ridge | 26 | nan | 0.377 | 0.320 | 0.665 | 0.430 | 0.300 | 0.473 | 0.223 | 0.339 |
| lasso | 26 | nan | 0.383 | 0.318 | 0.665 | 0.430 | 0.300 | 0.486 | 0.236 | 0.336 |
| elasticnet | 26 | nan | 0.378 | 0.320 | 0.663 | 0.427 | 0.301 | 0.479 | 0.229 | 0.337 |
| bayesian_ridge | 26 | 0.615 | 0.377 | 0.320 | 0.665 | 0.430 | 0.300 | 0.473 | 0.223 | 0.339 |
| random_forest | 26 | nan | 0.390 | 0.317 | 0.682 | 0.465 | 0.291 | 0.465 | 0.214 | 0.340 |
| gbr | 26 | nan | 0.389 | 0.317 | 0.683 | 0.467 | 0.290 | 0.460 | 0.210 | 0.341 |
| xgboost | 26 | nan | 0.390 | 0.316 | 0.683 | 0.467 | 0.290 | 0.463 | 0.213 | 0.341 |
| lightgbm | 26 | nan | 0.389 | 0.317 | 0.684 | 0.467 | 0.290 | 0.460 | 0.210 | 0.341 |
| pymc_hierarchical | 26 | 0.608 | 0.369 | 0.322 | 0.665 | 0.431 | 0.300 | 0.455 | 0.203 | 0.343 |
| xgb_no_raw_peak | 26 | 0.592 | 0.349 | 0.327 | 0.672 | 0.449 | 0.295 | 0.381 | 0.140 | 0.356 |
| xgb_no_satellite | 26 | 0.636 | 0.403 | 0.313 | 0.682 | 0.461 | 0.292 | 0.508 | 0.248 | 0.333 |
| xgb_with_aquacrop | 26 | 0.632 | 0.398 | 0.314 | 0.684 | 0.468 | 0.290 | 0.483 | 0.230 | 0.337 |
| xgb_no_raw_peak_with_aquacrop | 26 | 0.630 | 0.396 | 0.315 | 0.684 | 0.467 | 0.290 | 0.478 | 0.226 | 0.338 |

### C3. Out-of-sample years only (2021-2022, N=4)

| model | N | pooled r | pooled R2 | pooled RMSE |
|---|---|---|---|---|
| ridge | 4 | nan | -1.172 | 0.228 |
| xgboost | 4 | nan | -0.844 | 0.210 |
| xgb_no_raw_peak | 4 | 0.489 | -2.565 | 0.293 |
| xgb_no_satellite | 4 | -0.015 | -0.337 | 0.179 |
| xgb_with_aquacrop | 4 | 0.540 | -0.480 | 0.189 |
| xgb_no_raw_peak_with_aquacrop | 4 | 0.565 | -0.476 | 0.188 |

## Leave-one-district-out (pixel level)
| model | split | r2 | rmse | mae |
|---|---|---|---|---|
| linear | train_KanpurNagar_test_KanpurDehat_spatial | 0.499 | 0.337 | 0.256 |
| linear | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.338 | 0.407 | 0.318 |
| linear | train_KanpurDehat_test_KanpurNagar_spatial | 0.682 | 0.471 | 0.365 |
| linear | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.260 | 0.704 | 0.563 |
| ridge | train_KanpurNagar_test_KanpurDehat_spatial | 0.499 | 0.337 | 0.256 |
| ridge | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.338 | 0.407 | 0.319 |
| ridge | train_KanpurDehat_test_KanpurNagar_spatial | 0.682 | 0.471 | 0.365 |
| ridge | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.268 | 0.701 | 0.560 |
| lasso | train_KanpurNagar_test_KanpurDehat_spatial | 0.504 | 0.335 | 0.253 |
| lasso | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.335 | 0.408 | 0.319 |
| lasso | train_KanpurDehat_test_KanpurNagar_spatial | 0.679 | 0.473 | 0.367 |
| lasso | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.582 | 0.529 | 0.414 |
| elasticnet | train_KanpurNagar_test_KanpurDehat_spatial | 0.503 | 0.335 | 0.254 |
| elasticnet | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.236 | 0.438 | 0.349 |
| elasticnet | train_KanpurDehat_test_KanpurNagar_spatial | 0.680 | 0.473 | 0.366 |
| elasticnet | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.578 | 0.532 | 0.417 |
| xgboost | train_KanpurNagar_test_KanpurDehat_spatial | 0.575 | 0.310 | 0.249 |
| xgboost | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.552 | 0.335 | 0.270 |
| xgboost | train_KanpurDehat_test_KanpurNagar_spatial | 0.764 | 0.406 | 0.319 |
| xgboost | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.747 | 0.412 | 0.328 |
| lightgbm | train_KanpurNagar_test_KanpurDehat_spatial | 0.564 | 0.314 | 0.253 |
| lightgbm | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.538 | 0.340 | 0.275 |
| lightgbm | train_KanpurDehat_test_KanpurNagar_spatial | 0.762 | 0.408 | 0.319 |
| lightgbm | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.748 | 0.411 | 0.326 |
| random_forest | train_KanpurNagar_test_KanpurDehat_spatial | 0.560 | 0.315 | 0.254 |
| random_forest | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.542 | 0.339 | 0.273 |
| random_forest | train_KanpurDehat_test_KanpurNagar_spatial | 0.763 | 0.406 | 0.320 |
| random_forest | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.746 | 0.413 | 0.327 |
| gbr | train_KanpurNagar_test_KanpurDehat_spatial | 0.564 | 0.321 | 0.259 |
| gbr | train_KanpurNagar_test_KanpurDehat_spatiotemporal | 0.520 | 0.347 | 0.278 |
| gbr | train_KanpurDehat_test_KanpurNagar_spatial | 0.764 | 0.405 | 0.316 |
| gbr | train_KanpurDehat_test_KanpurNagar_spatiotemporal | 0.749 | 0.410 | 0.327 |

## E. SHAP top-10 (XGBoost, 5,000 test pixel-years)
| feature | mean_abs_shap |
|---|---|
| raw_peak_ndvi | 0.3934 |
| greenup_day | 0.1357 |
| peak_ndvi_day | 0.1042 |
| district_kanpur_dehat | 0.0602 |
| senescence_day | 0.0242 |
| length_of_season | 0.0109 |
| rain_cv | 0.0079 |
| ph | 0.0067 |
| kharif_tmin_mean | 0.0053 |
| monsoon_onset_day | 0.0045 |

## F. Bayesian 95% HDI widths, 20 random test pixels
| pixel_id | district | year | y_true | pred_mean | hdi95_mean_width | hdi95_predictive_width |
|---|---|---|---|---|---|---|
| 30596_120874 | KanpurNagar | 2022 | 2.133 | 2.086 | 0.053 | 1.155 |
| 30360_120716 | KanpurNagar | 2024 | 3.542 | 3.406 | 0.129 | 1.173 |
| 30631_120861 | KanpurDehat | 2022 | 3.300 | 3.169 | 0.060 | 1.121 |
| 30336_120673 | KanpurNagar | 2021 | 3.734 | 3.866 | 0.043 | 1.226 |
| 30459_120840 | KanpurNagar | 2022 | 3.745 | 3.188 | 0.050 | 1.113 |
| 30479_120654 | KanpurDehat | 2021 | 2.651 | 2.384 | 0.041 | 1.113 |
| 30447_120766 | KanpurDehat | 2021 | 3.058 | 2.810 | 0.042 | 1.081 |
| 30405_120687 | KanpurDehat | 2021 | 2.362 | 2.864 | 0.034 | 1.134 |
| 30707_121010 | KanpurNagar | 2024 | 2.599 | 2.746 | 0.133 | 1.158 |
| 30445_120747 | KanpurDehat | 2023 | 3.686 | 3.861 | 0.063 | 1.100 |
| 30590_120800 | KanpurDehat | 2024 | 2.484 | 2.163 | 0.159 | 1.184 |
| 30659_121013 | KanpurNagar | 2021 | 2.396 | 2.617 | 0.046 | 1.141 |
| 30389_120775 | KanpurNagar | 2024 | 4.215 | 3.527 | 0.134 | 1.120 |
| 30648_120889 | KanpurNagar | 2022 | 3.171 | 2.908 | 0.049 | 1.029 |
| 30553_120773 | KanpurDehat | 2024 | 3.254 | 3.158 | 0.155 | 1.141 |
| 30431_120839 | KanpurNagar | 2021 | 2.985 | 2.794 | 0.041 | 1.054 |
| 30545_120828 | KanpurDehat | 2024 | 3.612 | 3.314 | 0.157 | 1.149 |
| 30451_120609 | KanpurDehat | 2023 | 2.985 | 2.778 | 0.070 | 1.174 |
| 30555_120932 | KanpurNagar | 2024 | 2.942 | 2.709 | 0.134 | 1.171 |
| 30304_120643 | KanpurNagar | 2022 | 2.283 | 2.679 | 0.053 | 1.087 |

{
 "sampler": {
  "seconds": 689.7864289283752,
  "divergences": 0,
  "max_rhat": 1.004215589146039,
  "min_ess_bulk": 1583.9885356753825
 },
 "coverage_95_predictive_test": 0.8985,
 "coverage_95_mean_only_test": 0.10715,
 "mean_ppc_width_test": 1.1472174931394652
}

## G. AquaCrop standalone
```
{
 "n": 26,
 "pooled": {
  "r2": -113.90224723680024,
  "rmse": 4.343259889341549,
  "mae": 4.3320121702102155
 },
 "KanpurDehat": {
  "pearson_r": 0.7806647108958478,
  "mean_sim": 7.209474175569274
 },
 "KanpurNagar": {
  "pearson_r": 0.7457425251053874,
  "mean_sim": 7.209165549466543
 }
}
```

## H. Residuals (XGBoost)
```
Overall mean residual (target - prediction): -0.0160 t/ha

By district:
               mean     std
district                   
KanpurDehat -0.0225  0.1669
KanpurNagar -0.0083  0.2572

By year x district:
district  KanpurDehat  KanpurNagar
year                              
2021          -0.0633      -0.0037
2022           0.0238       0.0440
2023          -0.0940      -0.1227
2024           0.0434       0.0493

Mean residual by prediction decile (0=lowest):
decile
0    0.0030
1   -0.0070
2   -0.0164
3   -0.0183
4   -0.0232
5   -0.0281
6   -0.0375
7   -0.0336
8   -0.0163
9    0.0177
```