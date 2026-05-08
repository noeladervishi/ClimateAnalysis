import os
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import geopandas as gpd
from shapely.geometry import box
import h5py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    RAW_DIR, ALBANIA_BBOX, CRS_WGS84, NASA_EARTHDATA_TOKEN, ALBANIA_CITIES
)

# NDVI Calculator (from Sentinel-2 bands)
class NDVIProcessor:
    def __init__(self):
        self.output_dir = RAW_DIR / "sentinel2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_ndvi(self, red_band_path: Path, nir_band_path: Path) -> np.ndarray:
        """Compute NDVI from red and NIR raster files."""
        with rasterio.open(red_band_path) as red_src:
            red = red_src.read(1).astype(np.float32)
            profile = red_src.profile.copy()

        with rasterio.open(nir_band_path) as nir_src:
            nir = nir_src.read(1).astype(np.float32)

        # Avoid division by zero
        denominator = nir + red
        ndvi = np.where(denominator == 0, np.nan, (nir - red) / denominator)

        logger.info(f"NDVI computed | mean={np.nanmean(ndvi):.3f} | "
                    f"min={np.nanmin(ndvi):.3f} | max={np.nanmax(ndvi):.3f}")
        return ndvi, profile

    def save_ndvi(self, ndvi: np.ndarray, profile: dict, output_name: str) -> Path:
        """Save NDVI array as a GeoTIFF."""
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        output_path = self.output_dir / f"{output_name}.tif"

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndvi.astype(np.float32), 1)

        logger.success(f"NDVI saved: {output_path}")
        return output_path

    def compute_ndvi_timeseries(
        self, scene_pairs: list[dict], output_prefix: str = "albania_ndvi"
    ) -> list[dict]:

        results = []
        for scene in scene_pairs:
            date_str = scene["date"]
            ndvi, profile = self.compute_ndvi(scene["red"], scene["nir"])
            out_name = f"{output_prefix}_{date_str}"
            path = self.save_ndvi(ndvi, profile, out_name)
            results.append({
                "date": date_str,
                "mean_ndvi": float(np.nanmean(ndvi)),
                "std_ndvi": float(np.nanstd(ndvi)),
                "path": path,
            })
        return results

    def mask_to_albania(self, raster_path: Path, albania_gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Clip a raster to Albania boundary polygon."""
        with rasterio.open(raster_path) as src:
            albania_proj = albania_gdf.to_crs(src.crs)
            geoms = [geom.__geo_interface__ for geom in albania_proj.geometry]
            masked, _ = rio_mask(src, geoms, crop=True, nodata=np.nan)
        return masked[0]

# MODIS LST Downloader
class MODISDownloader:
    BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives"
    SEARCH_URL = "https://cmr.earthdata.nasa.gov/search"

    PRODUCTS = {
        "LST":        "MOD11A2.061",
        "NDVI":       "MOD13A3.061",
        "SNOW_COVER": "MOD10A2.061",
    }

    # Albania MODIS tiles (h20v04, h21v04)
    ALBANIA_TILES = ["h20v04", "h21v04"]

    def __init__(self):
        self.output_dir = RAW_DIR / "modis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = NASA_EARTHDATA_TOKEN
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def search_granules(
        self,
        product: str,
        start_date: str,
        end_date: str,
        tiles: list[str] | None = None,
    ) -> list[dict]:

        tiles = tiles or self.ALBANIA_TILES
        short_name = product.split(".")[0]
        version = product.split(".")[1] if "." in product else "061"
        params = {
            "short_name": short_name,
            "version": [version, str(int(version))] if version.isdigit() else version,
            "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
            "bounding_box": (
                f"{ALBANIA_BBOX['min_lon']},{ALBANIA_BBOX['min_lat']},"
                f"{ALBANIA_BBOX['max_lon']},{ALBANIA_BBOX['max_lat']}"
            ),
            "provider": "LPCLOUD",
            "page_size": 100,
        }

        url = f"{self.SEARCH_URL}/granules.json"
        resp = requests.get(url, params=params, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            logger.error(f"CMR granule search failed ({resp.status_code}): {resp.text[:500]}")
            raise

        granules = resp.json().get("feed", {}).get("entry", [])
        # Filter by known Albania tile identifiers in granule metadata when possible.
        filtered = []
        for g in granules:
            tile_blob = " ".join(
                [
                    str(g.get("title", "")),
                    str(g.get("producer_granule_id", "")),
                    str(g.get("id", "")),
                ]
            ).lower()
            if any(tile.lower() in tile_blob for tile in tiles):
                filtered.append(g)
        if filtered:
            granules = filtered
        logger.info(f"Found {len(granules)} granules for {product} ({start_date}–{end_date})")
        return granules

    def download_granule(self, download_url: str, filename: str) -> Path:
        """Download a single MODIS granule HDF file."""
        output_path = self.output_dir / filename
        if output_path.exists():
            logger.info(f"Already downloaded: {filename}")
            return output_path

        logger.info(f"Downloading: {filename}")
        resp = requests.get(download_url, headers=self.headers, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.success(f"Saved: {output_path}")
        return output_path

    def download_lst_series(self, start_date: str, end_date: str) -> list[Path]:
        """Download MODIS 8-day LST granules for Albania."""
        granules = self.search_granules(self.PRODUCTS["LST"], start_date, end_date)
        paths = []
        for g in granules:
            links = [
                l for l in g.get("links", [])
                if "data#" in l.get("rel", "") and str(l.get("href", "")).lower().endswith(".hdf")
            ]
            if links:
                url = links[0]["href"]
                fname = Path(url).name
                path = self.download_granule(url, fname)
                paths.append(path)
        return paths

    def extract_lst_from_hdf(self, hdf_path: Path) -> np.ndarray:
        
        subdataset_name = "LST_Day_1km"
        scale_factor = 0.02

        with h5py.File(hdf_path, "r") as f:
            # MODIS HDF structure varies; try common paths
            for key in f.keys():
                if "LST" in key:
                    data = f[key][:].astype(np.float32)
                    data = np.where(data == 0, np.nan, data * scale_factor - 273.15)
                    logger.info(f"LST extracted: mean={np.nanmean(data):.2f}°C")
                    return data

        logger.warning(f"LST data not found in {hdf_path}")
        return None

# NOAA / Open-Meteo Fallback Downloader
class OpenMeteoDownloader:
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    VARIABLES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "windspeed_10m_max",
        "et0_fao_evapotranspiration",
        "shortwave_radiation_sum",
        "relative_humidity_2m_mean",
    ]

    def __init__(self):
        self.output_dir = RAW_DIR / "open_meteo"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_city(
        self,
        city_name: str,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: list[str] | None = None,
    ) -> Path:
       
        variables = variables or self.VARIABLES
        output_path = self.output_dir / f"{city_name.lower()}_{start_date[:4]}_{end_date[:4]}.csv"

        if output_path.exists():
            logger.info(f"Already exists: {output_path}")
            return output_path

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(variables),
            "timezone": "Europe/Tirane",
        }

        logger.info(f"Downloading Open-Meteo data for {city_name} ({start_date}–{end_date}) …")
        resp = requests.get(self.BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data["daily"])
        df["city"] = city_name
        df["latitude"] = lat
        df["longitude"] = lon
        df.to_csv(output_path, index=False)

        logger.success(f"Saved {len(df)} rows → {output_path}")
        return output_path

    def download_all_cities(
        self,
        cities: dict,
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
    ) -> dict[str, Path]:
        """Download data for all Albania cities."""
        cities = cities or ALBANIA_CITIES
        results = {}
        for city, (lon, lat) in cities.items():
            try:
                path = self.download_city(city, lat, lon, start_date, end_date)
                results[city] = path
            except Exception as e:
                logger.error(f"Failed to download {city}: {e}")
        return results


if __name__ == "__main__":
    dl = OpenMeteoDownloader()
    results = dl.download_all_cities(ALBANIA_CITIES, "2020-01-01", "2025-12-31")
    for city, path in results.items():
        print(f"  {city}: {path}")