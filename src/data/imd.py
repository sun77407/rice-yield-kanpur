"""IMD gridded rainfall (0.25 deg) and Tmax/Tmin (1 deg) via ``imdlib``; box-mean for Kanpur."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data import weather as w

VARS = {"rain": "precip_mm", "tmax": "tmax_c", "tmin": "tmin_c"}


def fetch_imd() -> pd.DataFrame:
    """Download IMD yearly grids, average over the Kanpur box, write daily parquet + manifest."""
    import imdlib as imd

    out_dir = w.RAW / "imd"
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, so, ea, no = w.BBOX
    cols: dict[str, pd.Series] = {}
    for var, name in VARS.items():

        def call(v: str = var) -> Any:
            return imd.get_data(v, 2010, 2024, fn_format="yearwise", file_dir=str(out_dir))

        data = w.with_retries("imd", f"download {var}", call)
        ds = data.get_xarray()
        ds = ds.where(ds != -999.0)
        sub = ds.sel(lat=slice(so, no), lon=slice(lo, ea))
        if sub.sizes["lat"] == 0 or sub.sizes["lon"] == 0:  # 1 deg grid may miss the box
            sub = ds.sel(lat=w.KANPUR_LAT, lon=w.KANPUR_LON, method="nearest")
        series = sub[next(iter(sub.data_vars))].mean(
            dim=[d for d in ("lat", "lon") if d in sub.dims]
        )
        cols[name] = series.to_series()
    df = pd.DataFrame(cols).rename_axis("date").reset_index()
    out = out_dir / "daily.parquet"
    df.to_parquet(out, index=False)
    raw = sorted(p for p in out_dir.rglob("*.GRD"))
    w._write_manifest(
        "imd",
        "https://www.imdpune.gov.in/lrfindex.php (via imdlib)",
        [out, *raw],
        len(df),
        {"bbox_WSEN": w.BBOX, "failures": []},
    )
    return df
