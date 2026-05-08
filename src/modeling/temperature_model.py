import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from src.preprocessing.time_series_processor import TimeSeriesProcessor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MODEL_CONFIG, PROCESSED_DIR
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet

HAS_XGB = True
HAS_LGB = True
HAS_PROPHET = True


class TemperatureModel:
    
    def __init__(self, city: str = "Tirana", model_type: str = "xgboost"):
        self.city = city
        self.model_type = model_type
        self.model = None
        self.feature_cols: list[str] = []
        self.target_col: str = "temperature_2m_mean"
        self.model_dir = PROCESSED_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # Model Initialisation
    def _build_model(self):
        cfg = MODEL_CONFIG

        if self.model_type == "xgboost" and HAS_XGB:
            self.model = xgb.XGBRegressor(**cfg["xgboost"])
        elif self.model_type == "lightgbm" and HAS_LGB:
            self.model = lgb.LGBMRegressor(**cfg["lightgbm"])
        else:
            # Fallback to sklearn GBM
            logger.warning(f"{self.model_type} not available — using sklearn GBM.")
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            )

    # Training
    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        self.target_col = "temperature_2m_mean"
        if self.target_col not in df.columns:
            # Try alternative column names
            for alt in ["temperature_mean", "t2m_mean", "tmean"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: self.target_col})
                    break

        exclude = [self.target_col, "city", "latitude", "longitude", "time"]
        self.feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        X = df[self.feature_cols]
        y = df[self.target_col]
        return X, y

    def fit(self, train_df: pd.DataFrame) -> "TemperatureModel":
        X_train, y_train = self.prepare_features(train_df)
        self._build_model()

        logger.info(f"Training {self.model_type} temperature model for {self.city} "
                    f"({len(X_train)} samples, {len(self.feature_cols)} features) …")
        self.model.fit(X_train, y_train)
        logger.success(f"Training complete.")
        return self

    def fit_with_validation(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> dict:
        self.fit(train_df)
        metrics = self.evaluate(val_df)
        logger.info(f"Validation | RMSE={metrics['rmse']:.2f}°C | MAE={metrics['mae']:.2f}°C | R²={metrics['r2']:.3f}")
        return metrics

    # Prediction
    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X, _ = self.prepare_features(df)
        preds = self.model.predict(X[self.feature_cols])
        return pd.Series(preds, index=df.index, name="predicted_temp_c")

    def forecast_future(
        self,
        last_known_df: pd.DataFrame,
        n_months: int = 12,
    ) -> pd.DataFrame:

        tsp = TimeSeriesProcessor()

        historical = last_known_df.copy()
        forecasts = []

        for step in range(n_months):
            featured = tsp.create_ml_features(historical, self.target_col)
            last_row = featured.iloc[[-1]].copy()

            # Advance date
            next_date = last_row.index[-1] + pd.DateOffset(months=1)
            last_row.index = [next_date]
            last_row["month"] = next_date.month
            last_row["year"] = next_date.year
            last_row["month_sin"] = np.sin(2 * np.pi * next_date.month / 12)
            last_row["month_cos"] = np.cos(2 * np.pi * next_date.month / 12)

            pred = self.model.predict(last_row[self.feature_cols])[0]
            forecasts.append({"date": next_date, "forecast_temp_c": pred})

            # Append prediction back to historical
            new_row = pd.DataFrame(
                {self.target_col: [pred]}, index=[next_date]
            )
            historical = pd.concat([historical, new_row])

        forecast_df = pd.DataFrame(forecasts).set_index("date")
        logger.info(f"Generated {n_months}-month temperature forecast.")
        return forecast_df

    # Evaluation
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        X_test, y_test = self.prepare_features(test_df)
        y_pred = self.model.predict(X_test[self.feature_cols])

        metrics = {
            "rmse":  float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae":   float(mean_absolute_error(y_test, y_pred)),
            "r2":    float(r2_score(y_test, y_pred)),
            "bias":  float(np.mean(y_pred - y_test)),
            "n":     len(y_test),
        }
        return metrics

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
        results = []
        n = len(df)
        min_train = int(n * 0.5)
        fold_size = (n - min_train) // n_splits

        for fold in range(n_splits):
            train_end = min_train + fold * fold_size
            test_end = min(train_end + fold_size, n)
            if test_end >= n:
                break

            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]

            self.fit(train_df)
            metrics = self.evaluate(test_df)
            metrics["fold"] = fold + 1
            metrics["train_size"] = train_end
            metrics["test_size"] = test_end - train_end
            results.append(metrics)

        cv_df = pd.DataFrame(results)
        logger.info(
            f"CV Results | RMSE={cv_df['rmse'].mean():.2f}±{cv_df['rmse'].std():.2f}°C | "
            f"R²={cv_df['r2'].mean():.3f}"
        )
        return cv_df

    # Feature Importance
    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            logger.warning("Model does not provide feature importances.")
            return pd.DataFrame()

        fi_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)
        return fi_df

    # Prophet Wrapper
    def fit_prophet(
        self, series: pd.Series, changepoint_prior_scale: float = 0.05
    ) -> "TemperatureModel":
        
        if not HAS_PROPHET:
            logger.error("prophet not installed. Run: pip install prophet")
            return self

        prophet_df = pd.DataFrame({
            "ds": series.index,
            "y": series.values,
        }).reset_index(drop=True)

        cfg = MODEL_CONFIG["prophet"]
        self.prophet_model = Prophet(
            yearly_seasonality=cfg["yearly_seasonality"],
            weekly_seasonality=cfg["weekly_seasonality"],
            daily_seasonality=cfg["daily_seasonality"],
            seasonality_mode=cfg["seasonality_mode"],
            changepoint_prior_scale=changepoint_prior_scale,
        )
        self.prophet_model.fit(prophet_df)
        logger.success("Prophet temperature model fitted.")
        return self

    def forecast_prophet(self, periods: int = 24) -> pd.DataFrame:
        if not hasattr(self, "prophet_model"):
            raise RuntimeError("Prophet model not fitted.")

        future = self.prophet_model.make_future_dataframe(periods=periods, freq="MS")
        forecast = self.prophet_model.predict(future)
        forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        forecast.columns = ["date", "forecast", "lower_95", "upper_95"]
        return forecast

    # Persistence
    def save(self) -> Path:
        path = self.model_dir / f"temp_model_{self.city}_{self.model_type}.pkl"
        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "city": self.city,
            "model_type": self.model_type,
        }, path)
        logger.success(f"Model saved: {path}")
        return path

    def load(self, city: str | None = None, model_type: str | None = None) -> "TemperatureModel":
        city = city or self.city
        model_type = model_type or self.model_type
        path = self.model_dir / f"temp_model_{city}_{model_type}.pkl"
        obj = joblib.load(path)
        self.model = obj["model"]
        self.feature_cols = obj["feature_cols"]
        self.target_col = obj["target_col"]
        logger.success(f"Model loaded: {path}")
        return self


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    dates = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    np.random.seed(42)
    # Synthetic monthly temperatures for Tirana (seasonal pattern + trend)
    t = np.arange(len(dates))
    seasonal = 10 * np.sin(2 * np.pi * t / 12 - np.pi / 2)
    trend = 0.015 * t
    noise = np.random.normal(0, 1.5, len(dates))
    temps = 12 + seasonal + trend + noise

    df = pd.DataFrame({"temperature_2m_mean": temps}, index=dates)

    tsp = TimeSeriesProcessor()
    df_feat = tsp.create_ml_features(df, "temperature_2m_mean")
    train, test = tsp.train_test_split_temporal(df_feat)

    model = TemperatureModel(city="Tirana", model_type="xgboost")
    model.fit(train)
    metrics = model.evaluate(test)
    print(f"Test RMSE: {metrics['rmse']:.2f}°C | R²: {metrics['r2']:.3f}")