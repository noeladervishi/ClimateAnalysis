import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    RAW_DIR, ALBANIA_BBOX, ALBANIA_CITIES, HISTORICAL_START, HISTORICAL_END
)


def _normalise_name(value: str) -> str:
    """Lightweight normalisation for city/station name matching."""
    if not isinstance(value, str):
        return ""
    text = value.lower()
    replacements = {
        "ë": "e",
        "ç": "c",
        "ş": "s",
        "’": "'",
        "`": "'",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join(text.replace("_", " ").replace("-", " ").split())

# Known Albanian GHCND station IDs
ALBANIA_GHCND_STATIONS = {
    "Tirana_Rinas": {"id": "ALB000013600", "lat": 41.41, "lon": 19.72, "elev_m": 38},
    "Shkodër":      {"id": "ALB000013601", "lat": 42.07, "lon": 19.51, "elev_m": 15},
    "Durrës":       {"id": "ALB000013603", "lat": 41.33, "lon": 19.47, "elev_m": 10},
    "Tirana":       {"id": "ALB000013612", "lat": 41.33, "lon": 19.83, "elev_m": 110},
    "Vlorë":        {"id": "ALB000013618", "lat": 40.47, "lon": 19.49, "elev_m": 11},
    "Gjirokastër":  {"id": "ALB000013621", "lat": 40.07, "lon": 20.14, "elev_m": 300},
    "Korçë":        {"id": "ALB000013626", "lat": 40.61, "lon": 20.78, "elev_m": 869},
    "Lezhe":        {"id": "ALB000013631", "lat": 41.78, "lon": 19.64, "elev_m": 7},
}

# GHCND element codes we want
GHCND_ELEMENTS = {
    "TMAX": "max_temp_tenths_C",       # Daily max temp (tenths of °C)
    "TMIN": "min_temp_tenths_C",       # Daily min temp (tenths of °C)
    "PRCP": "precip_tenths_mm",        # Daily precipitation (tenths of mm)
    "SNOW": "snowfall_mm",             # Snowfall (mm)
    "SNWD": "snow_depth_mm",           # Snow depth (mm)
    "AWND": "avg_wind_speed_tenths_ms",# Avg wind speed (tenths of m/s)
    "TAVG": "avg_temp_tenths_C",       # Daily average temp (tenths of °C)
    "EVAP": "evaporation_tenths_mm",   # Pan evaporation
}

# NOAA CDO API Downloader
class NOAACDODownloader:
    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
    MAX_RESULTS_PER_PAGE = 1000
    REQUEST_DELAY_S = 0.25   # stay within rate limit

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("NOAA_CDO_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No NOAA CDO API key set. "
                "Register at https://www.ncdc.noaa.gov/cdo-web/token "
                "and add NOAA_CDO_API_KEY to your .env file."
            )
        self.headers = {"token": self.api_key}
        self.output_dir = RAW_DIR / "noaa" / "cdo"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Station Discovery
    def find_albania_stations(
        self,
        dataset_id: str = "GHCND",
        data_category: str = "TEMP",
    ) -> pd.DataFrame:
        
        params = {
            "datasetid":    dataset_id,
            "datacategoryid": data_category,
            "extent":       (
                f"{ALBANIA_BBOX['min_lat']},{ALBANIA_BBOX['min_lon']},"
                f"{ALBANIA_BBOX['max_lat']},{ALBANIA_BBOX['max_lon']}"
            ),
            "limit":        1000,
            "offset":       1,
        }

        logger.info(f"Searching NOAA CDO for {dataset_id} stations in Albania …")
        resp = self._get("stations", params)
        if not resp or "results" not in resp:
            logger.warning("No stations found or API error.")
            return pd.DataFrame()

        df = pd.DataFrame(resp["results"])
        logger.success(f"Found {len(df)} stations in Albania ({dataset_id})")
        return df

    def get_station_info(self, station_id: str) -> dict:
        """Fetch metadata for a single station."""
        result = self._get(f"stations/{station_id}")
        return result or {}

    # Data Download
    def download_station_daily(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        datatypes: list[str] | None = None,
        dataset_id: str = "GHCND",
        city_name: str = "unknown",
    ) -> pd.DataFrame:
        
        datatypes = datatypes or list(GHCND_ELEMENTS.keys())

        # Format station ID
        if not station_id.startswith(dataset_id + ":"):
            station_id_full = f"{dataset_id}:{station_id}"
        else:
            station_id_full = station_id

        output_path = self.output_dir / f"{city_name.lower()}_{dataset_id.lower()}_daily.csv"
        if output_path.exists():
            logger.info(f"Already downloaded: {output_path}")
            return pd.read_csv(output_path, parse_dates=["date"])

        all_records = []
        # Chunk by year (CDO API max range is 1 year for daily data)
        years = range(
            int(start_date[:4]),
            int(end_date[:4]) + 1
        )

        for year in tqdm(years, desc=f"Downloading {city_name} ({dataset_id})"):
            year_start = f"{year}-01-01"
            year_end   = f"{year}-12-31"
            # Clamp to requested range
            year_start = max(year_start, start_date)
            year_end   = min(year_end, end_date)

            params = {
                "datasetid":  dataset_id,
                "stationid":  station_id_full,
                "datatypeid": ",".join(datatypes),
                "startdate":  year_start,
                "enddate":    year_end,
                "units":      "metric",
                "limit":      self.MAX_RESULTS_PER_PAGE,
                "offset":     1,
            }

            while True:
                resp = self._get("data", params)
                if not resp or "results" not in resp:
                    break
                all_records.extend(resp["results"])
                # Check if more pages remain
                meta = resp.get("metadata", {}).get("resultset", {})
                count = meta.get("count", 0)
                offset = meta.get("offset", 1)
                limit = meta.get("limit", self.MAX_RESULTS_PER_PAGE)
                if offset + limit > count:
                    break
                params["offset"] = offset + limit

        if not all_records:
            logger.warning(f"No records returned for {station_id_full} ({start_date}–{end_date})")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.rename(columns={"date": "datetime"})
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        df["station_id"] = station_id
        df["city"] = city_name

        # Pivot to wide format (one column per datatype)
        df_wide = df.pivot_table(
            index=["date", "city", "station_id"],
            columns="datatype",
            values="value",
            aggfunc="first",
        ).reset_index()
        df_wide.columns.name = None

        # Apply scale factors (GHCND uses tenths for temp/precip)
        for col in df_wide.columns:
            if col in ["TMAX", "TMIN", "TAVG"]:
                df_wide[col] = df_wide[col] / 10.0    # tenths°C → °C
            if col in ["PRCP"]:
                df_wide[col] = df_wide[col] / 10.0    # tenths mm → mm

        df_wide.to_csv(output_path, index=False)
        logger.success(f"Saved {len(df_wide)} days → {output_path}")
        return df_wide

    def download_all_albania_stations(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2025-12-31",
        datatypes: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
       
        results = {}
        resolved = NOAAGHCNDDirectDownloader().resolve_city_station_map(country_code="AL")
        for city, meta in ALBANIA_GHCND_STATIONS.items():
            station_id = resolved.get(city, {}).get("id", meta["id"])
            logger.info(f"Downloading NOAA GHCND data for {city} …")
            try:
                df = self.download_station_daily(
                    station_id=station_id,
                    start_date=start_date,
                    end_date=end_date,
                    datatypes=datatypes,
                    city_name=city,
                )
                results[city] = df
            except Exception as e:
                logger.error(f"Failed {city}: {e}")
        return results

    def download_monthly_summaries(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        city_name: str = "unknown",
    ) -> pd.DataFrame:
        
        return self.download_station_daily(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            dataset_id="GSOM",
            datatypes=["TMAX", "TMIN", "TAVG", "PRCP", "SNOW"],
            city_name=city_name,
        )

    # Available Data Types
    def list_datatypes(self, dataset_id: str = "GHCND") -> pd.DataFrame:
        """List all available data types for a dataset."""
        resp = self._get("datatypes", {"datasetid": dataset_id, "limit": 1000})
        if resp and "results" in resp:
            return pd.DataFrame(resp["results"])
        return pd.DataFrame()

    def check_data_coverage(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> dict:
       
        resp = self._get(
            f"stations/{station_id}",
            params={"datasetid": "GHCND"},
        )
        if not resp:
            return {}
        coverage = {
            "station_id":   station_id,
            "name":         resp.get("name", ""),
            "mindate":      resp.get("mindate", ""),
            "maxdate":      resp.get("maxdate", ""),
            "datacoverage": resp.get("datacoverage", 0),
            "elevation_m":  resp.get("elevation", None),
            "latitude":     resp.get("latitude", None),
            "longitude":    resp.get("longitude", None),
        }
        logger.info(
            f"Coverage for {station_id}: "
            f"{coverage['mindate']} → {coverage['maxdate']} "
            f"({coverage['datacoverage']*100:.0f}% complete)"
        )
        return coverage

    # Internal HTTP helper
    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Make a GET request to the CDO API with rate limiting."""
        if not self.api_key:
            logger.error("CDO API key not set. Cannot make API requests.")
            return None

        url = f"{self.BASE_URL}/{endpoint}"
        try:
            time.sleep(self.REQUEST_DELAY_S)
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP {resp.status_code} for {url}: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

# NOAA GHCND FTP/Direct Downloader
class NOAAGHCNDDirectDownloader:
    BASE_URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"
    INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"

    def __init__(self):
        self.output_dir = RAW_DIR / "noaa" / "ghcnd_direct"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search_station_ids(self, country_code: str = "AL") -> pd.DataFrame:
        
        cc = country_code.upper()
        inventory_cache = self.output_dir / f"ghcnd_stations_{cc}.csv"

        if inventory_cache.exists():
            logger.info(f"Loading cached station inventory: {inventory_cache}")
            return pd.read_csv(inventory_cache)

        logger.info(f"Downloading GHCND station inventory from NOAA …")
        try:
            resp = requests.get(self.INVENTORY_URL, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Could not download station inventory: {e}")
            return pd.DataFrame()

        records = []
        for line in resp.text.splitlines():
            if not line.startswith(cc):
                continue
            try:
                station_id = line[0:11].strip()
                lat        = float(line[12:20].strip())
                lon        = float(line[21:30].strip())
                elev_m     = float(line[31:37].strip())
                name       = line[41:71].strip()
                records.append({
                    "id": station_id, "lat": lat, "lon": lon,
                    "elev_m": elev_m, "name": name,
                })
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning(f"No stations found for country code '{cc}'.")
        else:
            df.to_csv(inventory_cache, index=False)
            logger.success(f"Found {len(df)} {cc} stations → {inventory_cache}")
            print(df.to_string(index=False))

        return df

    def resolve_city_station_map(
        self,
        cities_dict: dict | None = None,
        country_code: str = "AL",
    ) -> dict[str, dict]:
        """
        Resolve best station ID for each Albanian city using nearest inventory station.
        Returns mapping: city -> {"id", "lat", "lon", "name", "distance_deg"}.
        """
        cities_dict = cities_dict or ALBANIA_CITIES
        stations = self.search_station_ids(country_code=country_code)
        if stations.empty:
            return {}

        required_cols = {"id", "lat", "lon", "name"}
        if not required_cols.issubset(set(stations.columns)):
            return {}

        mapping = {}
        station_df = stations.dropna(subset=["id", "lat", "lon"]).copy()
        station_df["name_norm"] = station_df["name"].map(_normalise_name)

        for city, (lon, lat) in cities_dict.items():
            city_norm = _normalise_name(city)
            candidates = station_df.copy()

            # Prefer textual match when available.
            text_match = candidates[candidates["name_norm"].str.contains(city_norm, regex=False)]
            if not text_match.empty:
                candidates = text_match

            distances = (candidates["lat"] - lat) ** 2 + (candidates["lon"] - lon) ** 2
            idx = distances.idxmin()
            best = candidates.loc[idx]
            mapping[city] = {
                "id": str(best["id"]),
                "lat": float(best["lat"]),
                "lon": float(best["lon"]),
                "name": str(best.get("name", city)),
                "distance_deg": float(np.sqrt(distances.loc[idx])),
            }
        return mapping

    def download_station(
        self,
        station_id: str,
        city_name: str = "unknown",
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        
        output_path = self.output_dir / f"{city_name.lower()}_ghcnd.csv"

        if output_path.exists():
            logger.info(f"Already downloaded: {output_path}")
            return self._load_and_clean(output_path, start_year, end_year)

        url = f"{self.BASE_URL}/{station_id}.csv"
        logger.info(f"Downloading GHCND: {url}")

        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with open(output_path, "wb") as f:
                for chunk in tqdm(
                    resp.iter_content(chunk_size=8192),
                    total=max(1, total // 8192),
                    desc=f"GHCND {city_name}",
                    unit="KB",
                ):
                    f.write(chunk)

            logger.success(f"Saved: {output_path}")

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"Station {station_id} not found on NOAA GHCND (HTTP {resp.status_code}). "
                f"Run search_station_ids() to discover correct IDs for Albania."
            )
            return pd.DataFrame()

        return self._load_and_clean(output_path, start_year, end_year)

    def _load_and_clean(
        self,
        path: Path,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """Load and standardise a GHCND CSV file."""
        try:
            df = pd.read_csv(path, parse_dates=["DATE"], low_memory=False)
        except Exception as e:
            logger.error(f"Could not read {path}: {e}")
            return pd.DataFrame()

        df = df.rename(columns={"DATE": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Filter years
        if start_year:
            df = df[df.index.year >= start_year]
        if end_year:
            df = df[df.index.year <= end_year]

        # Select and rename key columns
        col_map = {}
        for ghcnd_col, nice_name in {
            "TMAX": "tmax_c",
            "TMIN": "tmin_c",
            "TAVG": "tavg_c",
            "PRCP": "precip_mm",
            "SNOW": "snowfall_mm",
            "SNWD": "snow_depth_mm",
            "AWND": "wind_speed_ms",
        }.items():
            if ghcnd_col in df.columns:
                col_map[ghcnd_col] = nice_name

        df = df[list(col_map.keys())].rename(columns=col_map)

        # Apply GHCND scale factors
        for col in ["tmax_c", "tmin_c", "tavg_c"]:
            if col in df.columns:
                df[col] = df[col] / 10.0     # tenths °C → °C

        if "precip_mm" in df.columns:
            df["precip_mm"] = df["precip_mm"] / 10.0   # tenths mm → mm

        if "wind_speed_ms" in df.columns:
            df["wind_speed_ms"] = df["wind_speed_ms"] / 10.0

        # Replace GHCND missing flag (-9999)
        df = df.replace(-9999 / 10, np.nan).replace(-9999.0, np.nan)

        logger.info(
            f"GHCND data: {len(df)} days | "
            f"{df.index.min().date()} → {df.index.max().date()} | "
            f"columns: {list(df.columns)}"
        )
        return df

    def download_all_albania(
        self,
        start_year: int = 2020,
        end_year: int = 2025,
    ) -> dict[str, pd.DataFrame]:
        """Download GHCND data for all known Albanian stations."""
        results = {}
        resolved = self.resolve_city_station_map(country_code="AL")
        for city, meta in ALBANIA_GHCND_STATIONS.items():
            station_id = resolved.get(city, {}).get("id", meta["id"])
            logger.info(f"Downloading GHCND for {city} ({station_id}) …")
            df = self.download_station(
                station_id=station_id,
                city_name=city,
                start_year=start_year,
                end_year=end_year,
            )
            if not df.empty:
                results[city] = df
                logger.success(f"{city}: {len(df)} records")
            else:
                logger.warning(f"{city}: no data returned")
        return results

# NAO Index Downloader
class NAOIndexDownloader:
    
    MONTHLY_URL = (
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
        "pna/norm.nao.monthly.b5001.current.ascii.table"
    )
    DAILY_URLS = [
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.daily.b5001.current.ascii.table",
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.daily.b500101.current.ascii.table",
    ]

    def __init__(self):
        self.output_dir = RAW_DIR / "noaa" / "nao"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_monthly_nao(self, force: bool = False) -> pd.Series:
        output_path = self.output_dir / "nao_monthly.csv"

        if output_path.exists() and not force:
            logger.info(f"Loading cached NAO monthly: {output_path}")
            df = pd.read_csv(output_path, index_col=0, parse_dates=True)
            return df["nao_index"]

        logger.info("Downloading monthly NAO index from NOAA CPC …")
        try:
            resp = requests.get(self.MONTHLY_URL, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"NAO download failed: {e}")
            return pd.Series(dtype=float, name="nao_index")

        records = []
        for line in resp.text.strip().split("\n"):
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            year = int(parts[0])
            for month_idx, val in enumerate(parts[1:13], start=1):
                try:
                    records.append({
                        "date": pd.Timestamp(year=year, month=month_idx, day=1),
                        "nao_index": float(val),
                    })
                except (ValueError, IndexError):
                    continue

        if not records:
            logger.error("Could not parse NAO data.")
            return pd.Series(dtype=float, name="nao_index")

        df = pd.DataFrame(records).set_index("date").sort_index()
        # Replace NOAA missing value flag
        df["nao_index"] = df["nao_index"].replace(-99.9, np.nan)

        df.to_csv(output_path)
        logger.success(
            f"NAO monthly index saved: {len(df)} months "
            f"({df.index.min().year}–{df.index.max().year}) → {output_path}"
        )
        return df["nao_index"]

    def download_daily_nao(self, force: bool = False) -> pd.Series:
        
        output_path = self.output_dir / "nao_daily.csv"

        if output_path.exists() and not force:
            logger.info(f"Loading cached NAO daily: {output_path}")
            df = pd.read_csv(output_path, index_col=0, parse_dates=True)
            return df["nao_index"]

        logger.info("Downloading daily NAO index from NOAA CPC …")
        resp = None
        last_error = None
        for url in self.DAILY_URLS:
            try:
                candidate = requests.get(url, timeout=30)
                candidate.raise_for_status()
                resp = candidate
                break
            except Exception as e:
                last_error = e

        if resp is None:
            logger.error(f"Daily NAO download failed: {last_error}")
            monthly = self.download_monthly_nao()
            if monthly.empty:
                return pd.Series(dtype=float, name="nao_index")
            # Fallback: provide daily index via linear interpolation of monthly values.
            daily = monthly.resample("D").interpolate("linear")
            daily.name = "nao_index"
            daily.to_frame().to_csv(output_path)
            logger.warning(
                f"Daily NAO endpoint unavailable; saved interpolated daily series ({len(daily)} days) → {output_path}"
            )
            return daily

        records = []
        for line in resp.text.strip().split("\n"):
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                val = float(parts[3])
                records.append({
                    "date": pd.Timestamp(year=year, month=month, day=day),
                    "nao_index": val,
                })
            except (ValueError, IndexError):
                continue

        if not records:
            logger.error("Could not parse daily NAO data.")
            return pd.Series(dtype=float, name="nao_index")

        df = pd.DataFrame(records).set_index("date").sort_index()
        df["nao_index"] = df["nao_index"].replace(-99.9, np.nan)

        df.to_csv(output_path)
        logger.success(f"Daily NAO index saved: {len(df)} days → {output_path}")
        return df["nao_index"]

    def nao_albania_correlation(
        self,
        nao_series: pd.Series,
        climate_series: pd.Series,
        variable: str = "precipitation",
        season: str = "DJF",
    ) -> dict:
        
        from scipy.stats import pearsonr

        # Guard: both series must have a DatetimeIndex
        if climate_series.empty:
            logger.warning("NAO correlation skipped: climate_series is empty.")
            return {}

        if not isinstance(climate_series.index, pd.DatetimeIndex):
            logger.warning(
                "NAO correlation skipped: climate_series does not have a "
                "DatetimeIndex (got %s). Ensure the series is indexed by date "
                "before calling nao_albania_correlation().",
                type(climate_series.index).__name__,
            )
            return {}

        if not isinstance(nao_series.index, pd.DatetimeIndex):
            logger.warning("NAO correlation skipped: nao_series does not have a DatetimeIndex.")
            return {}

        SEASON_MONTHS = {
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "JJA": [6, 7, 8],
            "SON": [9, 10, 11],
            "annual": list(range(1, 13)),
        }
        months = SEASON_MONTHS.get(season, list(range(1, 13)))

        nao_filtered = nao_series[nao_series.index.month.isin(months)]
        clim_filtered = climate_series[climate_series.index.month.isin(months)]

        # Align by index
        aligned = pd.concat(
            [nao_filtered.rename("nao"), clim_filtered.rename("climate")], axis=1
        ).dropna()

        if len(aligned) < 10:
            logger.warning(f"Too few overlapping records for NAO correlation ({len(aligned)}).")
            return {}

        r, p = pearsonr(aligned["nao"], aligned["climate"])
        result = {
            "season":    season,
            "variable":  variable,
            "pearson_r": round(r, 4),
            "p_value":   round(p, 4),
            "n":         len(aligned),
            "significant": p < 0.05,
        }
        logger.info(
            f"NAO–{variable} correlation ({season}): "
            f"r={r:.3f}, p={p:.4f} "
            f"({'significant' if p < 0.05 else 'not significant'})"
        )
        return result

# NOAA ISD (Integrated Surface Database)
class NOAAISDDownloader:
    BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"

    ALBANIA_ISD_STATIONS = {
        "Tirana_Rinas":  {"wmo": "13612", "usaf": "135950", "wban": "99999"},
        "Durrës":        {"wmo": "13603", "usaf": "135910", "wban": "99999"},
        "Shkodër":       {"wmo": "13601", "usaf": "135850", "wban": "99999"},
        "Vlorë":         {"wmo": "13618", "usaf": "136000", "wban": "99999"},
        "Korçë":         {"wmo": "13626", "usaf": "136020", "wban": "99999"},
        "Gjirokastër":   {"wmo": "13625", "usaf": "136010", "wban": "99999"},
    }

    def __init__(self):
        self.output_dir = RAW_DIR / "noaa" / "isd"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _station_file_id(self, station: dict) -> str:
        """Construct the ISD filename from USAF and WBAN codes."""
        return f"{station['usaf']}-{station['wban']}"

    def download_station_year(
        self,
        station_name: str,
        year: int,
    ) -> pd.DataFrame:
        
        if station_name not in self.ALBANIA_ISD_STATIONS:
            logger.error(f"Unknown station: {station_name}")
            return pd.DataFrame()

        station = self.ALBANIA_ISD_STATIONS[station_name]
        file_id = self._station_file_id(station)
        output_path = self.output_dir / f"{station_name.lower()}_{year}_isd.csv"

        if output_path.exists():
            logger.info(f"Already downloaded: {output_path}")
            return pd.read_csv(output_path, parse_dates=["DATE"])

        url = f"{self.BASE_URL}/{year}/{file_id}.csv"
        logger.info(f"Downloading ISD: {url}")

        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.success(f"Saved ISD: {output_path}")
        except Exception as e:
            logger.error(f"ISD download failed for {station_name} {year}: {e}")
            return pd.DataFrame()

        return self._parse_isd_csv(output_path)

    def _parse_isd_csv(self, path: Path) -> pd.DataFrame:
        """Parse and clean an ISD CSV file into a usable DataFrame."""
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return pd.DataFrame()

        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.rename(columns={"DATE": "datetime"})
        df = df.set_index("datetime").sort_index()

        # ISD columns of interest (with their scale factors)
        # TMP = air temperature (tenths of °C, e.g. "+0150" = 15.0°C)
        # DEW = dew point temperature
        # SLP = sea-level pressure (tenths of hPa)
        # WND = wind direction + speed (m/s tenths)

        parsed = {}
        if "TMP" in df.columns:
            parsed["temp_c"] = (
                df["TMP"]
                .astype(str)
                .str.extract(r"([+-]\d+)")[0]
                .astype(float, errors="ignore") / 10.0
            )
            parsed["temp_c"] = parsed["temp_c"].replace(999.9, np.nan)

        if "DEW" in df.columns:
            parsed["dew_c"] = (
                df["DEW"]
                .astype(str)
                .str.extract(r"([+-]\d+)")[0]
                .astype(float, errors="ignore") / 10.0
            )
            parsed["dew_c"] = parsed["dew_c"].replace(999.9, np.nan)

        if "SLP" in df.columns:
            parsed["slp_hpa"] = (
                pd.to_numeric(df["SLP"], errors="coerce") / 10.0
            )
            parsed["slp_hpa"] = parsed["slp_hpa"].replace(9999.9, np.nan)

        result = pd.DataFrame(parsed, index=df.index)
        logger.info(f"ISD parsed: {len(result)} hourly records.")
        return result

    def download_station_range(
        self,
        station_name: str,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """Download multiple years of ISD data and concatenate."""
        frames = []
        for year in tqdm(range(start_year, end_year + 1), desc=f"ISD {station_name}"):
            df = self.download_station_year(station_name, year)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        # Remove duplicates from overlapping downloads
        combined = combined[~combined.index.duplicated(keep="first")]
        logger.success(
            f"ISD combined for {station_name}: {len(combined)} hourly records "
            f"({start_year}–{end_year})"
        )
        return combined

    def to_daily(self, hourly_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ISD hourly data to daily statistics."""
        agg = {}
        if "temp_c" in hourly_df.columns:
            agg["tmax_c"] = hourly_df["temp_c"].resample("D").max()
            agg["tmin_c"] = hourly_df["temp_c"].resample("D").min()
            agg["tmean_c"] = hourly_df["temp_c"].resample("D").mean()
        if "slp_hpa" in hourly_df.columns:
            agg["slp_hpa"] = hourly_df["slp_hpa"].resample("D").mean()
        if "dew_c" in hourly_df.columns:
            agg["dew_mean_c"] = hourly_df["dew_c"].resample("D").mean()

        return pd.DataFrame(agg)

# Convenience wrapper
class NOAADownloader:
    def __init__(self, cdo_api_key: str | None = None):
        self.cdo = NOAACDODownloader(api_key=cdo_api_key)
        self.ghcnd = NOAAGHCNDDirectDownloader()
        self.isd = NOAAISDDownloader()
        self.nao = NAOIndexDownloader()

    def download_best(
        self,
        city: str,
        start: str = "2020-01-01",
        end: str = "2025-12-31",
    ) -> pd.DataFrame:
        
        if city not in ALBANIA_GHCND_STATIONS:
            logger.warning(
                f"'{city}' not in ALBANIA_GHCND_STATIONS. "
                f"Available: {list(ALBANIA_GHCND_STATIONS.keys())}"
            )
            return pd.DataFrame()

        station_meta = ALBANIA_GHCND_STATIONS[city]
        resolved = self.ghcnd.resolve_city_station_map(country_code="AL")
        station_id = resolved.get(city, {}).get("id", station_meta["id"])

        # Try CDO API first if key available
        if self.cdo.api_key:
            logger.info(f"Using CDO API for {city} …")
            df = self.cdo.download_station_daily(
                station_id=station_id,
                start_date=start,
                end_date=end,
                city_name=city,
            )
            if not df.empty:
                return df

        # Fallback: direct GHCND download (no API key required)
        logger.info(f"Using GHCND direct download for {city} …")
        return self.ghcnd.download_station(
            station_id=station_id,
            city_name=city,
            start_year=int(start[:4]),
            end_year=int(end[:4]),
        )

    def download_all(
        self,
        start: str = "2020-01-01",
        end: str = "2025-12-31",
    ) -> dict[str, pd.DataFrame]:
        """Download data for all Albanian stations."""
        results = {}
        for city in ALBANIA_GHCND_STATIONS:
            df = self.download_best(city, start, end)
            if not df.empty:
                results[city] = df
        logger.success(f"NOAA download complete: {len(results)}/{len(ALBANIA_GHCND_STATIONS)} cities")
        return results

    def download_nao(self, daily: bool = False) -> pd.Series:
        """Download NAO index (monthly by default, or daily)."""
        if daily:
            return self.nao.download_daily_nao()
        return self.nao.download_monthly_nao()

    def combined_station_dataset(
        self,
        city: str,
        start: str = "2020-01-01",
        end: str = "2025-12-31",
    ) -> pd.DataFrame:
       
        logger.info(f"Building combined NOAA dataset for {city} …")
        station_meta = ALBANIA_GHCND_STATIONS.get(city, {})

        # GHCND data
        df_ghcnd = self.download_best(city, start, end)

        # ISD data (if available for this city)
        isd_name = f"{city.replace('ë','e').replace('ç','c')}"
        isd_match = next(
            (k for k in self.isd.ALBANIA_ISD_STATIONS if city.lower() in k.lower()), None
        )
        df_isd_daily = pd.DataFrame()
        if isd_match:
            hourly = self.isd.download_station_range(
                isd_match, int(start[:4]), int(end[:4])
            )
            if not hourly.empty:
                df_isd_daily = self.isd.to_daily(hourly)

        # NAO index
        nao = self.nao.download_monthly_nao()
        nao_daily = nao.resample("D").interpolate("linear")

        # Merge
        frames = [df for df in [df_ghcnd, df_isd_daily] if not df.empty]
        if not frames:
            logger.warning(f"No data to combine for {city}.")
            return pd.DataFrame()

        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.join(frame, how="outer", rsuffix="_isd")

        if not nao_daily.empty:
            combined["nao_index"] = nao_daily.reindex(combined.index)

        combined.index.name = "date"
        combined = combined.loc[start:end]
        logger.success(
            f"Combined dataset for {city}: {len(combined)} days, "
            f"{combined.shape[1]} variables"
        )
        return combined

# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NOAA data downloader for Albania")
    parser.add_argument("--city", default="Tirana",
                        help=f"City: {list(ALBANIA_GHCND_STATIONS.keys())}")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--nao", action="store_true", help="Download NAO index only")
    parser.add_argument("--all", action="store_true", help="Download all cities")
    parser.add_argument("--search", action="store_true",
                        help="Search and print all Albanian GHCND station IDs from NOAA inventory")
    args = parser.parse_args()

    dl = NOAADownloader()

    if args.search:
        # Discover correct station IDs from the NOAA inventory file
        stations_df = dl.ghcnd.search_station_ids("ALB")
        if not stations_df.empty:
            print(f"\nFound {len(stations_df)} Albanian GHCND stations:")
            print(stations_df.to_string(index=False))

    elif args.nao:
        nao = dl.download_nao()
        print(f"\nNAO index: {len(nao)} months | mean={nao.mean():.3f}")

    elif args.all:
        results = dl.download_all(args.start, args.end)
        for city, df in results.items():
            print(f"  {city}: {len(df)} records")

    else:
        df = dl.download_best(args.city, args.start, args.end)
        if not df.empty:
            print(f"\n{args.city} NOAA data ({len(df)} records):")
            print(df.head(10).to_string())

            # NAO correlation — only possible when df has a DatetimeIndex
            nao = dl.download_nao()
            if "precip_mm" in df.columns:
                corr = dl.nao.nao_albania_correlation(
                    nao, df["precip_mm"], season="DJF"
                )
                if corr:
                    print(
                        f"\nNAO–Precipitation correlation (DJF): "
                        f"r={corr['pearson_r']}, p={corr['p_value']}"
                    )
            else:
                print("\nNo precip_mm column available for NAO correlation.")
        else:
            print(
                f"\nNo data returned for {args.city}. "
                f"Try running with --search to verify station IDs."
            )