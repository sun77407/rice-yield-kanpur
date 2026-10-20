"""Agent 3D (stretch): AquaCrop-OSPy PaddyRice per Kanpur district-year, compared with APY.

Uncalibrated: library 'PaddyRice' parameters, a library soil chosen from mean SoilGrids texture,
FAO-56 Penman-Monteith ET0 from NASA POWER, CHIRPS rain per district. Not tuned to APY.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src.features.targets import load_apy
from src.models import common as c

warnings.filterwarnings("ignore")


def et0_pm(p: pd.DataFrame, lat: float = 26.45, elev: float = 122.5) -> pd.Series:
    """FAO-56 daily reference ET (mm/day) from NASA POWER columns."""
    tmean = p["T2M"]
    es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    ea = es * p["RH2M"] / 100
    delta = 4098 * es / (tmean + 237.3) ** 2
    pres = p["PS"]  # kPa
    gamma = 0.000665 * pres
    rn = (
        0.77 * p["ALLSKY_SFC_SW_DWN"]
    )  # net shortwave only (net longwave ignored: slight overestimate)
    u2 = p["WS2M"]
    et = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea)) / (
        delta + gamma * (1 + 0.34 * u2)
    )
    return et.clip(lower=0)


def main() -> None:
    """Run AquaCrop for 2010-2022 in both districts and compare with APY."""
    from aquacrop import (
        AquaCropModel,
        Crop,
        FieldMngt,
        InitialWaterContent,
        IrrigationManagement,
        Soil,
    )

    p = pd.read_parquet("data/raw/nasa_power/daily.parquet")
    rain = pd.read_parquet("data/raw/chirps/daily.parquet")
    p["ET0"] = et0_pm(p)
    soil = Soil("ClayLoam")  # mean topsoil clay 23%, sand 37% -> loam/clay loam
    apy = load_apy()
    rows = []
    for dist in c.DISTRICTS:
        r = rain[rain.district == dist][["date", "precip_mm"]]
        w = p.merge(r, on="date")
        wdf = pd.DataFrame(
            {
                "MinTemp": w["T2M_MIN"],
                "MaxTemp": w["T2M_MAX"],
                "Precipitation": w["precip_mm"],
                "ReferenceET": w["ET0"],
                "Date": w["date"],
            }
        )
        for year in range(2010, 2025):
            crop = Crop("PaddyRice", planting_date="07/01")
            model = AquaCropModel(
                f"{year}/06/01",
                f"{year}/11/30",
                wdf,
                soil,
                crop,
                InitialWaterContent(value=["FC"]),
                field_management=FieldMngt(bunds=True, z_bund=0.2),
                irrigation_management=IrrigationManagement(irrigation_method=1, SMT=[90] * 4),
            )
            model.run_model(till_termination=True)
            fin = model._outputs.final_stats
            if not rows:
                print(list(fin.columns), fin.iloc[0].to_dict())
            rows.append(
                {
                    "district": dist,
                    "year": year,
                    "sim_yield_t_ha": float(fin["Dry yield (tonne/ha)"].iloc[0]),
                }
            )
            print(rows[-1], flush=True)
    sim_all = pd.DataFrame(rows).rename(columns={"sim_yield_t_ha": "aquacrop_sim_yield"})
    sim_all.to_csv(c.PROCESSED / "aquacrop_district_year.csv", index=False)
    sim = pd.DataFrame(rows).merge(apy, on=["district", "year"])
    out: dict[str, object] = {
        "n": len(sim),
        "pooled": c.metrics(sim.yield_t_ha, sim.sim_yield_t_ha),
    }
    for d in c.DISTRICTS:
        s = sim[sim.district == d]
        out[d] = {
            "pearson_r": float(np.corrcoef(s.yield_t_ha, s.sim_yield_t_ha)[0, 1]),
            "mean_sim": float(s.sim_yield_t_ha.mean()),
        }
    sim.to_csv(c.RESULTS / "aquacrop_vs_apy.csv", index=False)
    (c.RESULTS / "aquacrop.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
