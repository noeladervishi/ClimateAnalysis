import cdsapi
import xarray as xr
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    RAW_DIR, ALBANIA_BBOX, ERA5_VARIABLES, CDS_API_KEY, CDS_API_URL
)


class ERA5Downloader:
    DATASET_MONTHLY = "reanalysis-era5-single-levels-monthly-means"
    DATASET_HOURLY  = "reanalysis-era5-single-levels"
    DATASET_PRESSURE = "reanalysis-era5-pressure-levels"

    def __init__(self):
        self.output_dir = RAW_DIR / "era5"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_client()

    def _init_client(self):
        try:
            self.client = cdsapi.Client(url=CDS_API_URL, key=CDS_API_KEY)
            logger.info("CDS API client initialised.")
        except Exception as e:
            logger.warning(f"CDS client init failed (no API key?): {e}")
            self.client = None

    # Public API
    def download_monthly(
        self,
        variables: list[str] | None = None,
        start_year: int = 2020,
        end_year: int = 2025,
        months: list[str] | None = None,
    ) -> Path:
        """Download ERA5 monthly mean data for Albania."""
        variables = variables or ERA5_VARIABLES
        months = months or [f"{m:02d}" for m in range(1, 13)]

        filename = f"era5_monthly_{start_year}_{end_year}.nc"
        output_path = self.output_dir / filename

        if output_path.exists():
            logger.info(f"ERA5 monthly data already exists: {output_path}")
            return output_path

        if self.client is None:
            logger.error("CDS client not available — cannot download ERA5 data.")
            return output_path

        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variables,
            "year": [str(y) for y in range(start_year, end_year + 1)],
            "month": months,
            "time": "00:00",
            "area": self._bbox_to_area(),
            "format": "netcdf",
        }

        logger.info(f"Requesting ERA5 monthly data ({start_year}–{end_year}) …")
        self.client.retrieve(self.DATASET_MONTHLY, request, str(output_path))
        logger.success(f"Saved: {output_path}")
        return output_path

    def download_daily(
        self,
        variable: str,
        year: int,
        months: list[str] | None = None,
    ) -> Path:
        """Download ERA5 hourly data for a single variable/year and compute daily means."""
        months = months or [f"{m:02d}" for m in range(1, 13)]
        filename = f"era5_daily_{variable}_{year}.nc"
        output_path = self.output_dir / filename

        if output_path.exists():
            logger.info(f"Already exists: {output_path}")
            return output_path

        if self.client is None:
            logger.error("CDS client not available.")
            return output_path

        request = {
            "product_type": "reanalysis",
            "variable": [variable],
            "year": str(year),
            "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "area": self._bbox_to_area(),
            "format": "netcdf",
        }

        logger.info(f"Requesting ERA5 hourly {variable} for {year} …")
        tmp_path = self.output_dir / f"tmp_{variable}_{year}.nc"
        self.client.retrieve(self.DATASET_HOURLY, request, str(tmp_path))

        # Resample to daily
        logger.info("Resampling hourly → daily …")
        ds = xr.open_dataset(tmp_path)
        ds_daily = ds.resample(time="1D").mean()
        ds_daily.to_netcdf(output_path)
        tmp_path.unlink(missing_ok=True)

        logger.success(f"Daily means saved: {output_path}")
        return output_path

    def download_pressure_levels(
        self,
        variables: list[str],
        pressure_levels: list[int],
        year: int,
        months: list[str] | None = None,
    ) -> Path:
        """Download ERA5 pressure-level data (geopotential, temperature, humidity)."""
        months = months or [f"{m:02d}" for m in range(1, 13)]
        filename = f"era5_pressure_{year}.nc"
        output_path = self.output_dir / filename

        if output_path.exists():
            logger.info(f"Already exists: {output_path}")
            return output_path

        if self.client is None:
            logger.error("CDS client not available.")
            return output_path

        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "pressure_level": [str(p) for p in pressure_levels],
            "year": str(year),
            "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": self._bbox_to_area(),
            "format": "netcdf",
        }

        logger.info(f"Requesting ERA5 pressure-level data for {year} …")
        self.client.retrieve(self.DATASET_PRESSURE, request, str(output_path))
        logger.success(f"Saved: {output_path}")
        return output_path

    # Loaders (after download)
    def load_dataset(self, path: Path) -> xr.Dataset:
        """Load a NetCDF ERA5 file as an xarray Dataset."""
        logger.info(f"Loading ERA5 dataset: {path}")
        ds = xr.open_dataset(path, engine="netcdf4")
        return ds

    def to_dataframe(self, ds: xr.Dataset, variable: str) -> pd.DataFrame:
        """Convert an ERA5 DataArray for a single variable to a tidy DataFrame."""
        da = ds[variable]
        df = da.to_dataframe().reset_index()
        df.columns = [c.lower() for c in df.columns]
        return df

    def get_albania_timeseries(self, path: Path, variable: str) -> pd.Series:
        """Spatially average a variable over Albania and return a time series."""
        ds = self.load_dataset(path)
        da = ds[variable].mean(dim=["latitude", "longitude"])
        series = da.to_series()
        series.name = variable
        logger.info(f"Extracted Albania-mean time series for '{variable}'.")
        return series

    # Helpers
    def _bbox_to_area(self) -> list[float]:
        """CDS expects [North, West, South, East]."""
        return [
            ALBANIA_BBOX["max_lat"],
            ALBANIA_BBOX["min_lon"],
            ALBANIA_BBOX["min_lat"],
            ALBANIA_BBOX["max_lon"],
        ]

    def list_downloaded(self) -> list[Path]:
        """Return all downloaded ERA5 NetCDF files."""
        files = sorted(self.output_dir.glob("*.nc"))
        logger.info(f"Found {len(files)} ERA5 files in {self.output_dir}")
        return files

# CLI convenience
if __name__ == "__main__":
    dl = ERA5Downloader()
    path = dl.download_monthly(
        variables=["2m_temperature", "total_precipitation"],
        start_year=2020,
        end_year=2025,
    )
    print(f"Downloaded to: {path}")
