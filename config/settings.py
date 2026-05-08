from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SHAPEFILE_DIR = DATA_DIR / "shapefiles"
OUTPUT_DIR = BASE_DIR / "outputs"
MAPS_DIR = OUTPUT_DIR / "maps"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Raw/processed structure declared in README
RAW_ERA5_DIR = RAW_DIR / "era5"
RAW_OPEN_METEO_DIR = RAW_DIR / "open_meteo"
RAW_NOAA_DIR = RAW_DIR / "noaa"
RAW_NOAA_CDO_DIR = RAW_NOAA_DIR / "cdo"
RAW_NOAA_GHCND_DIR = RAW_NOAA_DIR / "ghcnd_direct"
RAW_NOAA_ISD_DIR = RAW_NOAA_DIR / "isd"
RAW_NOAA_NAO_DIR = RAW_NOAA_DIR / "nao"
RAW_SENTINEL2_DIR = RAW_DIR / "sentinel2"
RAW_MODIS_DIR = RAW_DIR / "modis"

PROCESSED_TIMESERIES_DIR = PROCESSED_DIR / "timeseries"
PROCESSED_RASTERS_DIR = PROCESSED_DIR / "rasters"
PROCESSED_VECTORS_DIR = PROCESSED_DIR / "vectors"
PROCESSED_INDICES_DIR = PROCESSED_DIR / "indices"
PROCESSED_EXTREME_EVENTS_DIR = PROCESSED_DIR / "extreme_events"
PROCESSED_SPATIAL_DIR = PROCESSED_DIR / "spatial"
PROCESSED_MODELS_DIR = PROCESSED_DIR / "models"
PROCESSED_EVAL_RESULTS_DIR = PROCESSED_DIR / "eval_results"

# Create all directories if they don't exist.
for d in [
    RAW_DIR,
    PROCESSED_DIR,
    SHAPEFILE_DIR,
    MAPS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    RAW_ERA5_DIR,
    RAW_OPEN_METEO_DIR,
    RAW_NOAA_DIR,
    RAW_NOAA_CDO_DIR,
    RAW_NOAA_GHCND_DIR,
    RAW_NOAA_ISD_DIR,
    RAW_NOAA_NAO_DIR,
    RAW_SENTINEL2_DIR,
    RAW_MODIS_DIR,
    PROCESSED_TIMESERIES_DIR,
    PROCESSED_RASTERS_DIR,
    PROCESSED_VECTORS_DIR,
    PROCESSED_INDICES_DIR,
    PROCESSED_EXTREME_EVENTS_DIR,
    PROCESSED_SPATIAL_DIR,
    PROCESSED_MODELS_DIR,
    PROCESSED_EVAL_RESULTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Albania Bounding Box & CRS
ALBANIA_BBOX = {
    "min_lon": 19.1,
    "max_lon": 21.1,
    "min_lat": 39.6,
    "max_lat": 42.7,
}

# Geographic extent as list [West, East, South, North]
ALBANIA_EXTENT = [
    ALBANIA_BBOX["min_lon"],
    ALBANIA_BBOX["max_lon"],
    ALBANIA_BBOX["min_lat"],
    ALBANIA_BBOX["max_lat"],
]

ALBANIA_CENTER = {"lat": 41.15, "lon": 20.17}

# WGS84 geographic CRS
CRS_WGS84 = "EPSG:4326"
# Albanian projected CRS (Balkans-appropriate UTM zone 34N)
CRS_PROJECTED = "EPSG:32634"

# Time Periods
HISTORICAL_START = "2020-01-01"
HISTORICAL_END   = "2025-12-31"
FORECAST_HORIZON_DAYS = 30

# ERA5 Variables
ERA5_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "2m_dewpoint_temperature",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_cloud_cover",
]

ERA5_PRESSURE_LEVELS = [500, 700, 850, 925, 1000]

# Climate Thresholds
THRESHOLDS = {
    "heatwave_temp_c": 35.0,          # °C above which it's a heatwave day
    "extreme_precip_mm": 50.0,         # mm/day for extreme precipitation
    "drought_spi_threshold": -1.5,     # SPI below this = severe drought
    "frost_temp_c": 0.0,               # °C below which frost occurs
    "hot_day_c": 30.0,                 # °C threshold for a "hot day"
    "cold_day_c": 5.0,                 # °C threshold for a "cold day"
    "ndvi_vegetation_min": 0.3,        # Min NDVI for healthy vegetation
}

# Albania Climate Zones
CLIMATE_ZONES = {
    "coastal_mediterranean": {
        "description": "Adriatic coast — hot dry summers, mild wet winters",
        "cities": ["Durrës", "Vlorë", "Sarandë", "Shkodër (coast)"],
    },
    "continental_highland": {
        "description": "Interior highlands — cold winters, warm summers",
        "cities": ["Korçë", "Peshkopi", "Ersekë"],
    },
    "transitional": {
        "description": "Transition zone — moderate climate",
        "cities": ["Tirana", "Elbasan", "Berat"],
    },
    "alpine": {
        "description": "Albanian Alps — cold, heavy snowfall",
        "cities": ["Kukës", "Bajram Curri"],
    },
}

# Major Albanian Cities (lon, lat)
ALBANIA_CITIES = {
    "Tirana":    (19.8187, 41.3275),
    "Durrës":    (19.4565, 41.3246),
    "Vlorë":     (19.4914, 40.4660),
    "Shkodër":   (19.5125, 42.0685),
    "Elbasan":   (20.0820, 41.1125),
    "Korçë":     (20.7819, 40.6186),
    "Gjirokastër": (20.1389, 40.0758),
    "Sarandë":   (20.0053, 39.8756),
    "Berat":     (19.9520, 40.7058),
    "Kukës":     (20.4219, 42.0781),
    "Lushnjë":   (19.7050, 40.9419),
    "Fier":      (19.5564, 40.7239),
}

# API Keys (from .env file)
CDS_API_KEY = os.getenv("CDS_API_KEY", "")           # Copernicus CDS
CDS_API_URL = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "")

# Model Hyperparameters
MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "prophet": {
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "seasonality_mode": "multiplicative",
    },
    "train_test_split": 0.8,
    "cv_folds": 5,
}

# Visualization Settings
VIZ_CONFIG = {
    "figure_dpi": 150,
    "figure_size": (12, 8),
    "colormap_temperature": "RdYlBu_r",
    "colormap_precipitation": "Blues",
    "colormap_ndvi": "RdYlGn",
    "colormap_anomaly": "RdBu_r",
    "map_tiles": "OpenStreetMap",
    "font_size": 12,
}