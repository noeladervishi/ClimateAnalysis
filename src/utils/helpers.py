import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from tqdm import tqdm
from typing import Any, Callable
import time
from config.settings import BASE_DIR, ALBANIA_CITIES, ALBANIA_BBOX

# Logging Setup
def setup_logging(log_dir: Path | None = None, level: str = "INFO") -> None:
    logger.remove()  # Remove default handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"albania_climate_{datetime.now():%Y%m%d}.log"
        logger.add(
            log_path,
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        )
        logger.info(f"Logging to: {log_path}")

# Date / Time Utilities
def date_range_years(start: int, end: int) -> list[str]:
    return [str(y) for y in range(start, end + 1)]


def split_date_range(
    start: str, end: str, chunk_years: int = 5
) -> list[tuple[str, str]]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(
            current.replace(year=current.year + chunk_years) - timedelta(days=1),
            end_dt,
        )
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    return chunks


def month_name(month_int: int, lang: str = "en") -> str:
    en = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    sq = ["Jan","Shk","Mar","Pri","Maj","Qer","Kor","Gus","Sht","Tet","Nën","Dhj"]  # Albanian
    if lang == "sq":
        return sq[month_int - 1]
    return en[month_int - 1]


def season_from_month(month: int) -> str:
    return {12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer",
             9: "Autumn", 10: "Autumn", 11: "Autumn"}[month]

# Unit Conversions
def kelvin_to_celsius(k: float | np.ndarray) -> float | np.ndarray:
    return k - 273.15


def celsius_to_fahrenheit(c: float | np.ndarray) -> float | np.ndarray:
    return c * 9 / 5 + 32


def m_to_mm(m: float | np.ndarray) -> float | np.ndarray:
    return m * 1000


def ms_to_kmh(ms: float | np.ndarray) -> float | np.ndarray:
    return ms * 3.6


def pa_to_hpa(pa: float | np.ndarray) -> float | np.ndarray:
    return pa / 100


def dew_point_to_rh(temp_c: float, dew_c: float) -> float:
    rh = 100 * np.exp((17.625 * dew_c) / (243.04 + dew_c)) / \
         np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return float(np.clip(rh, 0, 100))


def heat_index(temp_c: float, rh: float) -> float:
    t = temp_c * 9 / 5 + 32  # to Fahrenheit for formula
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 0.00683783 * t**2
        - 0.05481717 * rh**2
        + 0.00122874 * t**2 * rh
        + 0.00085282 * t * rh**2
        - 0.00000199 * t**2 * rh**2
    )
    hi_c = (hi - 32) * 5 / 9  # back to Celsius
    return round(hi_c, 1)


def wind_chill(temp_c: float, wind_kmh: float) -> float:
    wc = (
        13.12 + 0.6215 * temp_c
        - 11.37 * wind_kmh**0.16
        + 0.3965 * temp_c * wind_kmh**0.16
    )
    return round(wc, 1)

# File I/O
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Path, pattern: str = "*", recursive: bool = False) -> list[Path]:
    if recursive:
        return sorted(directory.rglob(pattern))
    return sorted(directory.glob(pattern))


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)

# Data Utilities
def describe_climate_df(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.describe().T
    summary["skewness"] = numeric.skew()
    summary["kurtosis"] = numeric.kurtosis()
    summary["missing_pct"] = (numeric.isna().mean() * 100).round(2)
    summary["p5"] = numeric.quantile(0.05)
    summary["p95"] = numeric.quantile(0.95)
    return summary


def reindex_to_complete_dates(
    df: pd.DataFrame, freq: str = "D", fill_value: float = np.nan
) -> pd.DataFrame:
    
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return df.reindex(full_range, fill_value=fill_value)


def smooth_series(series: pd.Series, window: int = 12, min_periods: int = 1) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=min_periods).mean()


def percent_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods) * 100


def top_n_events(df: pd.DataFrame, value_col: str, n: int = 10, largest: bool = True) -> pd.DataFrame:
    return df.nlargest(n, value_col) if largest else df.nsmallest(n, value_col)

# Progress & Retry
def with_progress(iterable, description: str = "Processing") -> tqdm:
    return tqdm(iterable, desc=description, unit="item", ncols=80)


def retry(func: Callable, max_attempts: int = 3, delay_s: float = 2.0) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay_s)
            else:
                logger.error(f"All {max_attempts} attempts failed.")
                raise

# Albania-Specific
def is_within_albania(lat: float, lon: float) -> bool:
    return (
        ALBANIA_BBOX["min_lat"] <= lat <= ALBANIA_BBOX["max_lat"]
        and ALBANIA_BBOX["min_lon"] <= lon <= ALBANIA_BBOX["max_lon"]
    )


def nearest_city(lat: float, lon: float) -> str:
    min_dist = float("inf")
    nearest = "Unknown"
    for city, (c_lon, c_lat) in ALBANIA_CITIES.items():
        dist = np.sqrt((lat - c_lat) ** 2 + (lon - c_lon) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest = city
    return nearest


def albania_region_from_coords(lat: float, lon: float) -> str:
    if lon < 19.8 and lat > 40.5:
        return "Northern Coast (Adriatic)"
    if lon < 19.8 and lat <= 40.5:
        return "Southern Coast (Ionian)"
    if lat > 41.5:
        return "Albanian Alps (North)"
    if lon > 20.5 and lat < 41.5:
        return "Eastern Highlands"
    return "Central Lowlands"

def print_project_summary() -> None:
    print("=" * 60)
    print("Albania Geospatial Climate Analysis Project")
    print("=" * 60)
    print(f"  Base directory  : {BASE_DIR}")
    print(f"  Cities tracked  : {len(ALBANIA_CITIES)}")
    print(f"  Bounding box    : "
          f"{ALBANIA_BBOX['min_lat']}°N–{ALBANIA_BBOX['max_lat']}°N, "
          f"{ALBANIA_BBOX['min_lon']}°E–{ALBANIA_BBOX['max_lon']}°E")
    print("=" * 60)