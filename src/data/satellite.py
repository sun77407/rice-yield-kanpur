"""MODIS MOD13Q1 acquisition from Google Earth Engine (see ADR-022 and ADR-024).

Pipeline per year: server-side filter (Kharif window, district polygons) ->
batch ``Export.image.toAsset`` (service account, no billing) of a multi-band image on the
native MODIS grid -> direct ``getDownloadURL`` pull of the asset -> local conversion to a
pixel-time Parquet. The batch task is what gives a metered EECU figure.
Cropland masking (WorldCover 2021 class 40) is applied locally from a single static
fraction raster, so it costs EECU once instead of once per year.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd
import rasterio
import requests
from google.oauth2 import service_account
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.io import MemoryFile

from src.data.gee_quota import log_actual, log_estimate
from src.data.manifest import DataManifest
from src.utils.net import force_ipv4

GADM_ZIP = Path("data/raw/gadm/gadm41_IND_2.json.zip")
DISTRICTS = {"KanpurNagar": 1, "KanpurDehat": 2}
MODIS_COLLECTION = "MODIS/061/MOD13Q1"
WORLDCOVER = "ESA/WorldCover/v200"
KHARIF_DOY = (150, 300)  # PROJECT_FLOW.md section 2.1
BANDS = ("NDVI", "EVI", "SummaryQA", "DayOfYear")
NODATA = -9999
ASSET_ROOT = "kanpur_rice"  # projects/<project>/assets/kanpur_rice/<source>/<year>/<id>
DEFAULT_KEY_PATH = ".secrets/gee-key.json"
SCOPES = ("https://www.googleapis.com/auth/earthengine",)
SINU_PROJ4 = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
# MODIS sinusoidal global grid: upper-left corner and native pixel size (m)
_PIX = 231.6563582639
_X0, _Y0 = -20015109.354, 10007554.677


@dataclass(frozen=True)
class ExportJob:
    """A started GEE batch export to an Earth Engine asset."""

    export_id: str
    task_id: str
    asset_id: str


def service_credentials() -> service_account.Credentials:
    """Load service-account credentials from ``GEE_SERVICE_ACCOUNT_KEY`` (see ADR-022)."""
    key = Path(os.environ.get("GEE_SERVICE_ACCOUNT_KEY", DEFAULT_KEY_PATH))
    # google-auth ships no type information for this classmethod, hence the narrow ignores
    creds: service_account.Credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
        str(key), scopes=list(SCOPES)
    )
    return creds


def init_ee(project: str) -> None:
    """Initialize Earth Engine with the service account."""
    force_ipv4()
    ee.Initialize(credentials=service_credentials(), project=project)


def asset_id(project: str, source: str, year: int, name: str) -> str:
    """Asset path ``projects/<project>/assets/kanpur_rice/<source>/<year>/<name>``."""
    return f"projects/{project}/assets/{ASSET_ROOT}/{source}/{year}/{name}"


def ensure_folders(asset: str) -> None:
    """Create every missing parent folder of ``asset`` (idempotent)."""
    parts = asset.split("/")  # projects, <p>, assets, kanpur_rice, <source>, <year>, <name>
    for i in range(4, len(parts)):
        folder = "/".join(parts[:i])
        try:
            ee.data.getAsset(folder)
        except ee.EEException:
            ee.data.createAsset({"type": "FOLDER"}, folder)


def load_districts(zip_path: Path = GADM_ZIP) -> dict[str, dict[str, Any]]:
    """Return GADM level-2 GeoJSON geometries for Kanpur Nagar and Kanpur Dehat."""
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read(z.namelist()[0]))
    out = {
        f["properties"]["NAME_2"]: f["geometry"]
        for f in data["features"]
        if f["properties"]["NAME_1"] == "UttarPradesh" and f["properties"]["NAME_2"] in DISTRICTS
    }
    missing = set(DISTRICTS) - set(out)
    if missing:
        raise ValueError(f"GADM districts not found: {missing}")
    return out


def ee_region(districts: dict[str, dict[str, Any]]) -> ee.Geometry:
    """Union of the district polygons as a planar ``ee.Geometry``."""
    polys: list[Any] = []
    for g in districts.values():
        polys.extend(g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]])
    geom: ee.Geometry = ee.Geometry.MultiPolygon(polys, geodesic=False)  # ee is untyped (Any)
    return geom


def modis_grid() -> tuple[str, list[float]]:
    """Native MODIS sinusoidal CRS and affine transform (avoids resampling to 250 m)."""
    proj = ee.ImageCollection(MODIS_COLLECTION).first().select("NDVI").projection().getInfo()
    return str(proj["wkt"] if "wkt" in proj else proj["crs"]), list(proj["transform"])


def kharif_composites(year: int) -> ee.ImageCollection:
    """Server-side filter: MOD13Q1 composites whose start DOY lies in the Kharif window."""
    lo, hi = KHARIF_DOY
    col: ee.ImageCollection = (  # ee is untyped (Any); annotate so callers get a real type
        ee.ImageCollection(MODIS_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filter(ee.Filter.calendarRange(lo, hi, "day_of_year"))
        .select(list(BANDS))
    )
    return col


def year_stack(year: int, region: ee.Geometry) -> tuple[ee.Image, list[int]]:
    """Stack one year's composites into a single int16 multi-band image clipped to ``region``."""
    col = kharif_composites(year)
    times: list[int] = col.aggregate_array("system:time_start").getInfo() or []  # getInfo: Any|None
    doys = [datetime.fromtimestamp(t / 1000, UTC).timetuple().tm_yday for t in times]
    stack: ee.Image | None = None
    for t, doy in zip(times, doys, strict=True):
        im = col.filter(ee.Filter.eq("system:time_start", t)).first().toInt16()
        im = im.rename([f"{b}_{doy:03d}" for b in BANDS])
        stack = im if stack is None else stack.addBands(im)
    if stack is None:
        raise RuntimeError(f"No MOD13Q1 composites found for {year}")
    return stack.clip(region).unmask(NODATA), doys


def cropland_fraction(tile: ee.Geometry) -> ee.Image:
    """Fraction (permille) of 10 m WorldCover-2021 cropland (class 40) per MODIS pixel in ``tile``.

    The source is clipped to the tile *before* reduceResolution so EE only has to aggregate
    that tile's 10 m pixels (the full region exceeds its reprojection limit). The aggregation
    itself happens at the export grid (crs/crsTransform on the export).
    """
    wc = ee.ImageCollection(WORLDCOVER).first().select("Map").clip(tile)
    frac = wc.eq(40).reduceResolution(ee.Reducer.mean(), maxPixels=1024).rename("cropland_frac")
    out: ee.Image = frac.multiply(1000).unmask(NODATA).toInt16()  # ee is untyped (Any)
    return out


def tile_regions(region: ee.Geometry, nx: int, ny: int) -> list[ee.Geometry]:
    """Split the bounding box of ``region`` into an ``nx`` x ``ny`` grid of rectangles."""
    coords = region.bounds().coordinates().getInfo()  # getInfo is typed Any | None
    if not coords:
        raise RuntimeError("could not read region bounds from Earth Engine")
    ring = coords[0]
    xs, ys = [c[0] for c in ring], [c[1] for c in ring]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    return [
        ee.Geometry.Rectangle(
            [x0 + i * dx, y0 + j * dy, x0 + (i + 1) * dx, y0 + (j + 1) * dy], geodesic=False
        )
        for j in range(ny)
        for i in range(nx)
    ]


def start_export(
    image: ee.Image,
    export_id: str,
    asset: str,
    region: ee.Geometry,
    grid_crs: str,
    transform: list[float],
    estimated_eecu_hours: float,
    basis: str,
    details: dict[str, Any],
) -> ExportJob:
    """Log the EECU estimate (raises ``QuotaExceeded`` over the ceiling), then start the task."""
    ensure_folders(asset)
    log_estimate(export_id, estimated_eecu_hours, basis, {**details, "asset": asset})
    task = ee.batch.Export.image.toAsset(
        image=image,
        description=export_id,
        assetId=asset,
        region=region,
        crs=grid_crs,
        crsTransform=transform,
        maxPixels=int(1e9),
    )
    task.start()
    return ExportJob(export_id, str(task.id), asset)


def adopt_running(export_id: str, asset: str) -> ExportJob | None:
    """Return a handle for an already submitted, unfinished task named ``export_id``, if any."""
    for st in ee.data.getTaskList():
        if st.get("description") == export_id and st["state"] in {"READY", "RUNNING"}:
            return ExportJob(export_id, str(st["id"]), asset)
    return None


def wait_and_log(job: ExportJob, poll_s: int = 10, timeout_s: int = 3600) -> float | None:
    """Poll until the task ends, log the metered EECU-seconds, return them (None if unreported)."""
    t0 = time.time()
    while True:
        st = ee.data.getTaskStatus(job.task_id)[0]
        if st["state"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"{job.export_id} still {st['state']} after {timeout_s}s")
        time.sleep(poll_s)
    eecu = st.get("batch_eecu_usage_seconds")
    log_actual(job.export_id, job.task_id, None if eecu is None else float(eecu), st["state"])
    if st["state"] != "COMPLETED":
        raise RuntimeError(f"{job.export_id} {st['state']}: {st.get('error_message')}")
    return None if eecu is None else float(eecu)


def download_asset(
    asset: str, region: ee.Geometry, grid_crs: str, transform: list[float], out_zip: Path
) -> Path:
    """Pull an exported asset via ``getDownloadURL`` as a zip with one GeoTIFF per band."""
    url = ee.Image(asset).getDownloadURL(
        {
            "name": out_zip.stem,
            "format": "ZIPPED_GEO_TIFF",
            "region": region,
            "crs": grid_crs,
            "crs_transform": transform,
        }
    )
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    out_zip.write_bytes(resp.content)
    save_band_names(asset, out_zip)
    return out_zip


def _sidecar(zip_path: Path) -> Path:
    return zip_path.with_name(zip_path.stem + ".bands.json")


def save_band_names(asset: str, zip_path: Path) -> None:
    """Persist the asset's band order next to the download (the GeoTIFF carries no names)."""
    names = ee.Image(asset).bandNames().getInfo()
    _sidecar(zip_path).write_text(json.dumps(names), encoding="utf-8")


def read_band_zip(path: Path) -> tuple[dict[str, np.ndarray], rasterio.Affine]:
    """Read a download into ``{band_name: 2-D array}`` plus the affine transform.

    Band names come from the ``.bands.json`` sidecar written at download time.
    """
    names: list[str] = json.loads(_sidecar(path).read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as z:
        tifs = [n for n in z.namelist() if n.endswith(".tif")]
        if len(tifs) != 1:
            raise ValueError(f"{path.name}: expected one GeoTIFF, found {tifs}")
        with MemoryFile(z.read(tifs[0])) as mf, mf.open() as src:
            if src.count != len(names):
                raise ValueError(f"{path.name}: {src.count} bands but {len(names)} names")
            data = src.read()
            transform = src.transform
    return {n: data[i] for i, n in enumerate(names)}, transform


def _district_raster(
    districts: dict[str, dict[str, Any]], transform: rasterio.Affine, shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize district polygons (projected to MODIS sinusoidal) onto the grid; 0 = outside."""
    to_sinu = Transformer.from_crs("EPSG:4326", SINU_PROJ4, always_xy=True)
    shapes = []
    for name, geom in districts.items():
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        sinu = {
            "type": "MultiPolygon",
            "coordinates": [
                [[list(to_sinu.transform(x, y)) for x, y in ring] for ring in poly]
                for poly in polys
            ],
        }
        shapes.append((sinu, DISTRICTS[name]))
    return np.asarray(
        rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8")
    )


def mosaic_tiles(
    tile_zips: list[Path], shape: tuple[int, int], transform: rasterio.Affine, band: str
) -> np.ndarray:
    """Paste grid-aligned tile downloads into one array on the year grid (``NODATA`` elsewhere)."""
    out = np.full(shape, NODATA, dtype=float)
    for zp in tile_zips:
        bands, tf = read_band_zip(zp)
        arr = bands[band]
        col_off = (tf.c - transform.c) / transform.a
        row_off = (tf.f - transform.f) / transform.e
        if abs(col_off - round(col_off)) > 1e-3 or abs(row_off - round(row_off)) > 1e-3:
            raise ValueError(f"{zp.name} is not aligned to the MODIS grid")
        r0, c0 = round(row_off), round(col_off)
        # edge tiles can spill a pixel or two past the year grid; paste only the overlap
        rs, cs = max(r0, 0), max(c0, 0)
        re_, ce = min(r0 + arr.shape[0], shape[0]), min(c0 + arr.shape[1], shape[1])
        if re_ > rs and ce > cs:
            piece = arr[rs - r0 : re_ - r0, cs - c0 : ce - c0]
            # A lon/lat tile is a slanted shape on the sinusoidal grid: its bbox is padded with
            # masked (-32768) pixels that belong to neighbouring tiles, so never paste those.
            valid = piece != np.iinfo(np.int16).min
            view = out[rs:re_, cs:ce]
            view[valid] = piece[valid]
    return out


def stack_to_frame(
    year_zip: Path, crop_zips: list[Path], districts: dict[str, dict[str, Any]], year: int
) -> pd.DataFrame:
    """Convert the year stack to long-format pixel-time rows (one row per pixel per composite)."""
    bands, transform = read_band_zip(year_zip)
    h, w = next(iter(bands.values())).shape
    crop = mosaic_tiles(crop_zips, (h, w), transform, "cropland_frac")
    rows, cols = np.mgrid[0:h, 0:w]
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    xs, ys = np.asarray(xs), np.asarray(ys)
    lon, lat = Transformer.from_crs(SINU_PROJ4, "EPSG:4326", always_xy=True).transform(xs, ys)
    dist = _district_raster(districts, transform, (h, w))
    inv = {v: k for k, v in DISTRICTS.items()}
    col_g = np.floor((xs - _X0) / _PIX).astype(np.int64)
    row_g = np.floor((_Y0 - ys) / _PIX).astype(np.int64)
    m = (dist > 0) & (crop != NODATA)

    by_doy: dict[int, dict[str, np.ndarray]] = {}
    for n, arr in bands.items():
        band, doy_s = n.rsplit("_", 1)
        by_doy.setdefault(int(doy_s), {})[band] = arr
    pid = np.array([f"{r}_{c}" for r, c in zip(row_g[m], col_g[m], strict=True)])
    district = np.array([inv[d] for d in dist[m]])
    frames = []
    for doy, b in sorted(by_doy.items()):
        ndvi = np.where(b["NDVI"] <= -3000, np.nan, b["NDVI"] * 1e-4)[m]
        evi = np.where(b["EVI"] <= -3000, np.nan, b["EVI"] * 1e-4)[m]
        qa = b["SummaryQA"][m]
        frames.append(
            pd.DataFrame(
                {
                    "pixel_id": pid,
                    "lat": lat[m],
                    "lon": lon[m],
                    "district": district,
                    "year": year,
                    "doy": doy,
                    "obs_doy": b["DayOfYear"][m],
                    "ndvi": ndvi,
                    "evi": evi,
                    "summary_qa": qa,
                    "qa_pass": (qa == 0) & ~np.isnan(ndvi),
                    "cropland_frac": crop[m] / 1000.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def write_modis_manifest(
    raw_dir: Path,
    files: list[Path],
    rows: int | None,
    notes: dict[str, Any],
    asset_ids: dict[str, str],
) -> Path:
    """Write ``manifest.json`` for ``raw_dir``."""
    m = DataManifest.build(
        source="Google Earth Engine: MOD13Q1 v061 + ESA WorldCover v200",
        source_url="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1",
        files=files,
        row_count=rows,
        notes=notes,
        asset_ids=asset_ids,
    )
    return m.write(raw_dir / "manifest.json")
