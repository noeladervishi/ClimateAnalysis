import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ALBANIA_CITIES, RAW_DIR, PROCESSED_DIR,
    HISTORICAL_START, HISTORICAL_END,
)
from src.utils.helpers import setup_logging, print_project_summary

# Pipeline Steps
def step_download(cities: list[str] | None = None, start: str = "2020-01-01", end: str = "2025-12-31"):
    import pandas as pd
    from src.data_collection.era5_downloader import ERA5Downloader
    from src.data_collection.noaa_downloader import NOAADownloader
    from src.data_collection.satellite_downloader import OpenMeteoDownloader

    logger.info("═══ STEP 1: Data Download ═══")
    open_meteo_dl = OpenMeteoDownloader()
    noaa_dl = NOAADownloader()
    era5_dl = ERA5Downloader()

    target_cities = {k: v for k, v in ALBANIA_CITIES.items() if k in cities} \
                    if cities else ALBANIA_CITIES

    results = {
        "open_meteo": {},
        "noaa": {},
        "era5": None,
        "nao_monthly": None,
    }

    # Open-Meteo city-scale daily features (no API key required)
    results["open_meteo"] = open_meteo_dl.download_all_cities(target_cities, start, end)

    # NOAA station products per city (CDO key optional; direct GHCND fallback enabled)
    for city in target_cities:
        try:
            noaa_df = noaa_dl.download_best(city, start, end)
            if isinstance(noaa_df, pd.DataFrame) and not noaa_df.empty:
                results["noaa"][city] = noaa_df
        except Exception as e:
            logger.warning(f"NOAA download skipped for {city}: {e}")

    # ERA5 monthly reanalysis over Albania (requires CDS credentials)
    try:
        results["era5"] = era5_dl.download_monthly(
            start_year=int(start[:4]),
            end_year=int(end[:4]),
        )
    except Exception as e:
        logger.warning(f"ERA5 download skipped: {e}")

    # NAO monthly index from NOAA CPC
    try:
        results["nao_monthly"] = noaa_dl.download_nao(daily=False)
    except Exception as e:
        logger.warning(f"NAO download skipped: {e}")

    logger.success(
        "Download step complete | "
        f"Open-Meteo cities: {len(results['open_meteo'])}, "
        f"NOAA cities: {len(results['noaa'])}, "
        f"ERA5: {'yes' if results['era5'] else 'no'}, "
        f"NAO: {'yes' if results['nao_monthly'] is not None else 'no'}"
    )
    return results


def step_validate(city: str = "Tirana"):
    from src.utils.validators import ClimateDataValidator
    import pandas as pd

    logger.info("═══ STEP 2: Data Validation ═══")

    csv_files = sorted((RAW_DIR / "open_meteo").glob(f"{city.lower()}_*.csv"))
    if not csv_files:
        logger.warning(f"No data found for {city}. Run download step first.")
        return False

    df = pd.read_csv(csv_files[0], parse_dates=["time"]).set_index("time")
    validator = ClimateDataValidator()
    all_passed, summary = validator.run_full_validation(df)
    print(summary.to_string())
    return all_passed


def step_preprocess(city: str = "Tirana"):
    import pandas as pd
    from src.preprocessing.time_series_processor import TimeSeriesProcessor

    logger.info(f"═══ STEP 3: Preprocessing — {city} ═══")

    csv_files = sorted((RAW_DIR / "open_meteo").glob(f"{city.lower()}_*.csv"))
    if not csv_files:
        logger.warning(f"No raw data for {city}.")
        return None

    tsp = TimeSeriesProcessor(city=city)
    df = tsp.load_csv(csv_files[0], date_col="time")

    # Remove outliers
    numeric_cols = df.select_dtypes("number").columns.tolist()
    df = tsp.remove_outliers(df, numeric_cols, method="zscore")

    # Impute missing values
    df = tsp.impute_missing(df, method="time")

    # Resample to monthly
    df_monthly = tsp.resample(df, freq="MS", agg="mean")
    df_monthly["precipitation_sum"] = df.resample("MS")["precipitation_sum"].sum()

    # Feature engineering
    df_features = tsp.create_ml_features(df_monthly, "temperature_2m_mean")

    # Save
    tsp.save(df_monthly, f"{city.lower()}_monthly")
    tsp.save(df_features, f"{city.lower()}_features")

    logger.success(f"Preprocessing complete: {df_monthly.shape}")
    return df_monthly, df_features


def step_climate_indices(city: str = "Tirana"):
    import pandas as pd
    from src.analysis.climate_indices import ClimateIndices
    from src.preprocessing.time_series_processor import TimeSeriesProcessor

    logger.info(f"═══ STEP 4: Climate Indices — {city} ═══")

    tsp = TimeSeriesProcessor(city=city)
    try:
        df = tsp.load(f"{city.lower()}_monthly")
    except Exception:
        logger.error("Processed data not found. Run preprocess step first.")
        return

    ci = ClimateIndices()
    report = ci.full_index_report(
        tmax=df["temperature_2m_max"],
        tmin=df["temperature_2m_min"],
        tmean=df["temperature_2m_mean"],
        precip=df["precipitation_sum"],
    )
    print(f"\nAnnual Climate Indices for {city}:")
    print(report.round(2).to_string())
    return report


def step_trend_analysis(city: str = "Tirana"):
    import pandas as pd
    from src.analysis.trend_analysis import TrendAnalyser
    from src.preprocessing.time_series_processor import TimeSeriesProcessor

    logger.info(f"═══ STEP 5: Trend Analysis — {city} ═══")

    tsp = TimeSeriesProcessor(city=city)
    try:
        df = tsp.load(f"{city.lower()}_monthly")
    except Exception:
        logger.error("Run preprocess step first.")
        return

    ta = TrendAnalyser()
    key_vars = ["temperature_2m_mean", "precipitation_sum"]
    available = [v for v in key_vars if v in df.columns]

    report = ta.generate_trend_report(df[available], city)
    print(report)

    # Seasonal trends for temperature
    if "temperature_2m_mean" in df.columns:
        seasonal = ta.season_trends(df["temperature_2m_mean"])
        print(f"\nSeasonal temperature trends:\n{seasonal[['trend','slope','p_value']].to_string()}")

    return report


def step_extreme_events(city: str = "Tirana"):
    import pandas as pd
    from src.analysis.extreme_events import ExtremeEventDetector
    from src.preprocessing.time_series_processor import TimeSeriesProcessor

    logger.info(f"═══ STEP 6: Extreme Events — {city} ═══")

    tsp = TimeSeriesProcessor(city=city)
    try:
        df_monthly = tsp.load(f"{city.lower()}_monthly")
    except Exception:
        logger.error("Run preprocess step first.")
        return

    # Load daily for heatwave detection
    csv_files = sorted((RAW_DIR / "open_meteo").glob(f"{city.lower()}_*.csv"))
    if csv_files:
        df_daily = pd.read_csv(csv_files[0], parse_dates=["time"]).set_index("time")
    else:
        logger.warning("Daily data not found for extreme event detection.")
        return

    eed = ExtremeEventDetector()

    # Heatwaves
    if "temperature_2m_max" in df_daily.columns and "temperature_2m_mean" in df_daily.columns:
        heatwaves = eed.detect_heatwaves(
            df_daily["temperature_2m_max"],
            df_daily["temperature_2m_mean"],
        )
        logger.info(f"Heatwaves detected: {len(heatwaves)}")
        if heatwaves:
            hw_df = eed.events_to_dataframe(heatwaves)
            eed.save_events(heatwaves, f"{city.lower()}_heatwaves")
            print(f"\nTop 5 heatwaves in {city}:")
            print(hw_df.nlargest(5, "duration_days")[
                ["start","end","duration_days","max_intensity","severity"]
            ].to_string())

    # Drought
    if "precipitation_sum" in df_monthly.columns:
        from src.analysis.climate_indices import ClimateIndices
        ci = ClimateIndices()
        spi3 = ci.compute_spi(df_monthly["precipitation_sum"], 3)
        droughts = eed.detect_droughts(spi3)
        logger.info(f"Drought episodes detected: {len(droughts)}")
        eed.save_events(droughts, f"{city.lower()}_droughts")

    return heatwaves, droughts


def step_train_models(city: str = "Tirana"):
    from src.modeling.temperature_model import TemperatureModel
    from src.modeling.precipitation_model import MonthlyPrecipitationModel
    from src.modeling.model_evaluation import ModelEvaluator
    from src.preprocessing.time_series_processor import TimeSeriesProcessor

    logger.info(f"═══ STEP 7–8: Model Training — {city} ═══")

    tsp = TimeSeriesProcessor(city=city)
    try:
        df_features = tsp.load(f"{city.lower()}_features")
        df_monthly  = tsp.load(f"{city.lower()}_monthly")
    except Exception:
        logger.error("Run preprocess step first.")
        return

    evaluator = ModelEvaluator()

    # ── Temperature model (XGBoost)
    logger.info("Training temperature model …")
    train, test = tsp.train_test_split_temporal(df_features)
    temp_model = TemperatureModel(city=city, model_type="xgboost")
    temp_model.fit(train)
    temp_metrics = temp_model.evaluate(test)
    evaluator.save_metrics(temp_metrics, f"{city.lower()}_temp_metrics")

    logger.info(
        f"Temperature model | RMSE={temp_metrics['rmse']:.2f}°C | "
        f"R²={temp_metrics['r2']:.3f}"
    )

    # Feature importance
    fi_df = temp_model.feature_importance(top_n=10)
    print(f"\nTop 10 features for temperature prediction in {city}:")
    print(fi_df.to_string(index=False))

    # ── Temperature Prophet model
    if "temperature_2m_mean" in df_monthly.columns:
        logger.info("Training Prophet temperature model …")
        try:
            temp_model.fit_prophet(df_monthly["temperature_2m_mean"])
            fc_df = temp_model.forecast_prophet(periods=12)
            print(f"\n12-month temperature forecast for {city}:")
            print(fc_df.round(2).to_string())
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")

    # ── Precipitation model
    if "precipitation_sum" in df_monthly.columns:
        logger.info("Training precipitation (Prophet) model …")
        try:
            precip_model = MonthlyPrecipitationModel(city=city)
            precip_model.fit_prophet(df_monthly["precipitation_sum"])
            precip_fc = precip_model.forecast_prophet(periods=12)
            print(f"\n12-month precipitation forecast for {city}:")
            print(precip_fc.round(1).to_string())
        except Exception as e:
            logger.warning(f"Precipitation Prophet failed: {e}")

    # Save temperature model
    temp_model.save()
    return temp_model, temp_metrics


def step_generate_outputs(city: str = "Tirana"):
    import pandas as pd
    from src.visualization.climate_plots import ClimateVisualiser
    from src.preprocessing.time_series_processor import TimeSeriesProcessor

    logger.info(f"═══ STEP 9: Generating Outputs — {city} ═══")

    tsp = TimeSeriesProcessor(city=city)
    try:
        df_monthly = tsp.load(f"{city.lower()}_monthly")
    except Exception:
        logger.warning("No processed data found. Skipping output generation.")
        return

    viz = ClimateVisualiser()

    if "temperature_2m_mean" in df_monthly.columns:
        # Seasonal climatology
        if "precipitation_sum" in df_monthly.columns:
            viz.seasonal_climatology(
                df_monthly["temperature_2m_mean"],
                df_monthly["precipitation_sum"],
                city=city,
                output_name=f"{city.lower()}_climatology",
            )

        # Warming stripes
        annual_t = df_monthly["temperature_2m_mean"].resample("YS").mean()
        viz.warming_stripes(annual_t, title=f"{city} — Warming Stripes",
                            output_name=f"{city.lower()}_warming_stripes")

        # Full time series plot
        from src.analysis.climate_indices import ClimateIndices
        ci = ClimateIndices()
        climatology = df_monthly["temperature_2m_mean"].groupby(df_monthly.index.month).mean()
        anomaly = df_monthly["temperature_2m_mean"] - df_monthly.index.month.map(climatology)
        viz.temperature_timeseries(
            df_monthly["temperature_2m_mean"],
            city=city,
            anomaly=anomaly,
            output_name=f"{city.lower()}_temperature_timeseries",
        )

    logger.success(f"Output plots saved to: {viz.output_dir}")


def step_launch_dashboard():
    import subprocess
    dashboard_path = PROJECT_ROOT / "src" / "visualization" / "dashboard.py"
    logger.info("Launching Streamlit dashboard …")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )

# Full Pipeline
def run_full_pipeline(city: str = "Tirana", start: str = "2020-01-01", end: str = "2025-12-31"):
    logger.info(f"\n{'═'*60}")
    logger.info(f"  ALBANIA CLIMATE ANALYSIS — {city}")
    logger.info(f"  Period: {start} → {end}")
    logger.info(f"{'═'*60}\n")

    try:
        step_download(cities=[city], start=start, end=end)
    except Exception as e:
        logger.error(f"Download failed: {e}")

    step_validate(city)

    result = step_preprocess(city)
    if result is None:
        logger.error("Pipeline aborted: preprocessing failed.")
        return

    step_climate_indices(city)
    step_trend_analysis(city)

    try:
        step_extreme_events(city)
    except Exception as e:
        logger.warning(f"Extreme events step failed: {e}")

    try:
        step_train_models(city)
    except Exception as e:
        logger.warning(f"Model training failed: {e}")

    step_generate_outputs(city)

    logger.success(f"\n Pipeline complete for {city}!")

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Albania Geospatial Climate Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                             # Full pipeline, all cities
  python main.py --city Tirana              # Full pipeline for Tirana only
  python main.py --step download --city Vlorë
  python main.py --step indices --city Korçë
  python main.py --dashboard                 # Launch Streamlit dashboard
        """,
    )
    parser.add_argument(
        "--city", type=str, default="Tirana",
        help=f"Albanian city ({', '.join(ALBANIA_CITIES.keys())})",
    )
    parser.add_argument(
        "--all-cities", action="store_true",
        help="Run pipeline for all Albanian cities",
    )
    parser.add_argument(
        "--step", type=str,
        choices=["download", "validate", "preprocess", "indices",
                 "trends", "events", "models", "outputs", "all"],
        default="all",
        help="Pipeline step to run",
    )
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    print_project_summary()

    if args.dashboard:
        step_launch_dashboard()
        return

    cities = list(ALBANIA_CITIES.keys()) if args.all_cities else [args.city]

    for city in cities:
        if city not in ALBANIA_CITIES:
            logger.error(f"Unknown city: {city}. Choose from: {list(ALBANIA_CITIES.keys())}")
            continue

        if args.step == "all":
            run_full_pipeline(city, args.start, args.end)
        elif args.step == "download":
            step_download([city], args.start, args.end)
        elif args.step == "validate":
            step_validate(city)
        elif args.step == "preprocess":
            step_preprocess(city)
        elif args.step == "indices":
            step_climate_indices(city)
        elif args.step == "trends":
            step_trend_analysis(city)
        elif args.step == "events":
            step_extreme_events(city)
        elif args.step == "models":
            step_train_models(city)
        elif args.step == "outputs":
            step_generate_outputs(city)


if __name__ == "__main__":
    main()
