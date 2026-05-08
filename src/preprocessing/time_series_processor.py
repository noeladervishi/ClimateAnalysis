import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
from loguru import logger
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import PROCESSED_DIR, HISTORICAL_START, HISTORICAL_END


class TimeSeriesProcessor:
    def __init__(self, city: str = "Albania"):
        self.city = city
        self.output_dir = PROCESSED_DIR / "timeseries"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()

    # Loading
    def load_csv(
        self,
        path: Path,
        date_col: str = "time",
        value_col: str | None = None,
    ) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=[date_col])
        df = df.set_index(date_col).sort_index()
        df.index.name = "date"

        if value_col:
            df = df[[value_col]]

        logger.info(f"Loaded {len(df)} rows from {path.name} ({df.index[0]} → {df.index[-1]})")
        return df

    def load_openmeteo_cities(self, raw_dir: Path) -> pd.DataFrame:
        frames = []
        for csv_file in sorted(raw_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, parse_dates=["time"])
                frames.append(df)
            except Exception as e:
                logger.warning(f"Could not load {csv_file.name}: {e}")

        if not frames:
            logger.warning("No Open-Meteo CSV files found.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["time"] = pd.to_datetime(combined["time"])
        logger.info(f"Loaded {len(combined)} rows across {combined['city'].nunique()} cities.")
        return combined

    # Quality Control
    def detect_outliers_zscore(
        self, series: pd.Series, threshold: float = 3.5
    ) -> pd.Series:
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / (mad + 1e-9)
        outliers = np.abs(modified_z) > threshold
        logger.info(f"Outliers detected: {outliers.sum()} ({outliers.mean()*100:.1f}%)")
        return outliers

    def detect_outliers_iqr(
        self, series: pd.Series, multiplier: float = 1.5
    ) -> pd.Series:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outliers = (series < q1 - multiplier * iqr) | (series > q3 + multiplier * iqr)
        return outliers

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str],
        method: str = "zscore",
        replace_with: str = "interpolate",
    ) -> pd.DataFrame:
        
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            outliers_mask = (
                self.detect_outliers_zscore(series)
                if method == "zscore"
                else self.detect_outliers_iqr(series)
            )
            df.loc[outliers_mask[outliers_mask].index, col] = np.nan

            if replace_with == "interpolate":
                df[col] = df[col].interpolate(method="time")
            elif replace_with == "median":
                df[col] = df[col].fillna(df[col].median())

        return df

    # Missing Value Imputation
    def impute_missing(
        self,
        df: pd.DataFrame,
        method: str = "linear",
        max_gap_days: int = 7,
    ) -> pd.DataFrame:
        
        df = df.copy()
        num_missing_before = df.isnull().sum().sum()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if method in ("linear", "time"):
                df[col] = df[col].interpolate(method=method, limit=max_gap_days)
            elif method == "seasonal":
                df[col] = self._seasonal_impute(df[col])

        num_missing_after = df.isnull().sum().sum()
        logger.info(
            f"Imputation: {num_missing_before} → {num_missing_after} missing values "
            f"({num_missing_before - num_missing_after} filled)"
        )
        return df

    def _seasonal_impute(self, series: pd.Series) -> pd.Series:
        series = series.copy()
        for idx in series[series.isna()].index:
            month = idx.month
            same_month = series[(series.index.month == month) & series.notna()]
            if not same_month.empty:
                series[idx] = same_month.median()
        return series

    # Resampling
    def resample(
        self,
        df: pd.DataFrame,
        freq: str = "MS",           # "D"=daily, "MS"=month-start, "YS"=year-start
        agg: str = "mean",
    ) -> pd.DataFrame:
        if agg == "mean":
            return df.resample(freq).mean(numeric_only=True)
        elif agg == "sum":
            return df.resample(freq).sum()
        elif agg == "max":
            return df.resample(freq).max()
        elif agg == "min":
            return df.resample(freq).min()
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    # Anomaly & Standardisation
    def compute_anomalies(
        self,
        series: pd.Series,
        baseline_start: str = "2020-01-01",
        baseline_end: str = "2025-12-31",
    ) -> pd.Series:
        
        baseline = series.loc[baseline_start:baseline_end]
        baseline_mean = baseline.groupby(baseline.index.month).mean()

        anomaly = series.copy()
        for month in range(1, 13):
            mask = series.index.month == month
            anomaly.loc[mask] = series.loc[mask] - baseline_mean.get(month, 0)

        logger.info(f"Anomalies computed vs {baseline_start[:4]}–{baseline_end[:4]} baseline.")
        return anomaly

    def standardise(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def normalise(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        df = df.copy()
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    # Decomposition
    def decompose_seasonal(
        self, series: pd.Series, period: int = 12, model: str = "additive"
    ) -> object:
        
        series_clean = series.dropna()
        decomp = seasonal_decompose(series_clean, model=model, period=period, extrapolate_trend="freq")
        logger.info(f"Seasonal decomposition done (model={model}, period={period})")
        return decomp

    def stl_decompose(self, series: pd.Series, period: int = 12) -> STL:
        series_clean = series.dropna()
        stl = STL(series_clean, period=period, robust=True).fit()
        logger.info("STL decomposition complete.")
        return stl

    # Feature Engineering (for ML)
    def create_ml_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        
        df = df.copy()

        # Calendar features
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["year"] = df.index.year
        df["day_of_year"] = df.index.day_of_year
        df["season"] = df["month"].map({
            12: 0, 1: 0, 2: 0,   # Winter
            3: 1, 4: 1, 5: 1,    # Spring
            6: 2, 7: 2, 8: 2,    # Summer
            9: 3, 10: 3, 11: 3,  # Autumn
        })

        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        # Rolling statistics
        for window in [3, 6, 12]:
            df[f"{target_col}_roll_mean_{window}"] = (
                df[target_col].shift(1).rolling(window).mean()
            )
            df[f"{target_col}_roll_std_{window}"] = (
                df[target_col].shift(1).rolling(window).std()
            )
            df[f"{target_col}_roll_max_{window}"] = (
                df[target_col].shift(1).rolling(window).max()
            )

        # Year-over-year difference
        df[f"{target_col}_yoy_diff"] = df[target_col].diff(12)

        # Long-term trend (detrended)
        df["trend_index"] = np.arange(len(df))

        df = df.dropna()
        logger.info(f"Feature engineering: {df.shape[1]} features, {len(df)} rows.")
        return df

    def train_test_split_temporal(
        self, df: pd.DataFrame, split_ratio: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        n = int(len(df) * split_ratio)
        train = df.iloc[:n]
        test = df.iloc[n:]
        logger.info(f"Train: {len(train)} | Test: {len(test)} | Split: {df.index[n].date()}")
        return train, test

    # Save / Load
    def save(self, df: pd.DataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.success(f"Saved processed time series: {path}")
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = self.output_dir / f"{name}.parquet"
        df = pd.read_parquet(path)
        logger.info(f"Loaded: {path}")
        return df