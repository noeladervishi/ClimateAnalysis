import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import joblib
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MODEL_CONFIG, PROCESSED_DIR

HAS_XGB = True
HAS_LGB = True
HAS_PROPHET = True
HAS_SARIMA = True

class HurdlePrecipitationModel:
    def __init__(self, city: str = "Tirana"):
        self.city = city
        self.classifier = None    # Stage 1: occurrence
        self.regressor = None     # Stage 2: amount
        self.feature_cols: list[str] = []
        self.model_dir = PROCESSED_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        
        target = "precipitation_sum"
        if target not in df.columns:
            for alt in ["total_precipitation", "rain_sum", "precip"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: target})
                    break

        exclude = [target, "city", "latitude", "longitude"]
        self.feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        X = df[self.feature_cols]
        y_occ = (df[target] >= 1.0).astype(int)    # 1 = wet day
        y_amt = df[target].where(df[target] >= 1.0)  # NaN on dry days

        return X, y_occ, y_amt

    def fit(self, train_df: pd.DataFrame) -> "HurdlePrecipitationModel":
        X, y_occ, y_amt = self.prepare_features(train_df)

        # Stage 1: Occurrence
        cfg = MODEL_CONFIG
        if HAS_XGB:
            self.classifier = xgb.XGBClassifier(
                **{k: v for k, v in cfg["xgboost"].items() if k != "n_estimators"},
                n_estimators=300,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            self.classifier = GradientBoostingClassifier(n_estimators=200, random_state=42)

        logger.info(f"Fitting Stage 1 (occurrence) for {self.city} …")
        self.classifier.fit(X, y_occ)

        # Stage 2: Amount (only on wet days)
        wet_mask = y_occ == 1
        X_wet = X[wet_mask]
        y_wet = y_amt[wet_mask].dropna()

        if HAS_LGB:
            self.regressor = lgb.LGBMRegressor(**cfg["lightgbm"])
        else:
            self.regressor = GradientBoostingRegressor(n_estimators=200, random_state=42)

        logger.info(f"Fitting Stage 2 (amount) on {wet_mask.sum()} wet days …")
        self.regressor.fit(X_wet, y_wet)

        logger.success("Hurdle model training complete.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _, _ = self.prepare_features(df)

        p_wet = self.classifier.predict_proba(X)[:, 1]
        amount_given_wet = self.regressor.predict(X).clip(min=0)
        predicted = p_wet * amount_given_wet

        return pd.DataFrame({
            "p_wet_day": p_wet,
            "predicted_amount_mm": predicted,
            "predicted_wet_day": (p_wet >= 0.5).astype(int),
        }, index=df.index)

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        X, y_occ, y_amt = self.prepare_features(test_df)
        preds = self.predict(test_df)

        # Stage 1 metrics
        accuracy = accuracy_score(y_occ, preds["predicted_wet_day"])
        try:
            auc = roc_auc_score(y_occ, preds["p_wet_day"])
        except Exception:
            auc = float("nan")

        # Stage 2 metrics (total precipitation)
        actual_total = y_amt.fillna(0)
        pred_total = preds["predicted_amount_mm"]

        rmse = float(np.sqrt(mean_squared_error(actual_total, pred_total)))
        mae = float(mean_absolute_error(actual_total, pred_total))

        return {
            "occurrence_accuracy": round(accuracy, 3),
            "occurrence_auc": round(auc, 3),
            "amount_rmse_mm": round(rmse, 2),
            "amount_mae_mm": round(mae, 2),
            "n_test": len(test_df),
        }

    def save(self) -> Path:
        path = self.model_dir / f"precip_hurdle_{self.city}.pkl"
        joblib.dump({
            "classifier": self.classifier,
            "regressor": self.regressor,
            "feature_cols": self.feature_cols,
        }, path)
        logger.success(f"Hurdle model saved: {path}")
        return path

    def load(self, city: str | None = None) -> "HurdlePrecipitationModel":
        city = city or self.city
        path = self.model_dir / f"precip_hurdle_{city}.pkl"
        obj = joblib.load(path)
        self.classifier = obj["classifier"]
        self.regressor = obj["regressor"]
        self.feature_cols = obj["feature_cols"]
        return self


class MonthlyPrecipitationModel:
    
    def __init__(self, city: str = "Tirana"):
        self.city = city
        self.model = None
        self.model_dir = PROCESSED_DIR / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit_prophet(
        self,
        monthly_precip: pd.Series,
        nao_index: pd.Series | None = None,
    ) -> "MonthlyPrecipitationModel":
        
        if not HAS_PROPHET:
            logger.error("prophet not installed.")
            return self

        df = pd.DataFrame({
            "ds": monthly_precip.index,
            "y": monthly_precip.values,
        }).reset_index(drop=True)

        cfg = MODEL_CONFIG["prophet"]
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
        )

        # Add NAO regressor if available
        if nao_index is not None:
            nao_aligned = nao_index.reindex(monthly_precip.index, method="nearest")
            df["nao"] = nao_aligned.values
            self.model.add_regressor("nao", standardize=True)
            logger.info("NAO index added as regressor.")

        self.model.fit(df)
        logger.success(f"Prophet precipitation model fitted for {self.city}.")
        return self

    def forecast_prophet(
        self, periods: int = 24, future_nao: pd.Series | None = None
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        future = self.model.make_future_dataframe(periods=periods, freq="MS")

        if future_nao is not None:
            future["nao"] = future_nao.reindex(
                pd.DatetimeIndex(future["ds"]), method="nearest"
            ).values

        forecast = self.model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        result.columns = ["date", "forecast_mm", "lower_95_mm", "upper_95_mm"]
        result["forecast_mm"] = result["forecast_mm"].clip(lower=0)
        result["lower_95_mm"] = result["lower_95_mm"].clip(lower=0)
        return result


class SARIMAModel:
    
    def __init__(self, city: str = "Tirana"):
        self.city = city
        self.model = None
        self.result = None
        # SARIMA(1,0,1)(1,1,1)[12] — common for monthly precip
        self.order = (1, 0, 1)
        self.seasonal_order = (1, 1, 1, 12)

    def fit(self, series: pd.Series) -> "SARIMAModel":
        if not HAS_SARIMA:
            logger.error("statsmodels not installed.")
            return self

        series_clean = series.dropna()
        logger.info(f"Fitting SARIMA{self.order}×{self.seasonal_order} for {self.city} …")
        self.model = SARIMAX(
            series_clean,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.result = self.model.fit(disp=False)
        logger.success(f"SARIMA fitted. AIC={self.result.aic:.2f}")
        return self

    def forecast(self, steps: int = 24) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Model not fitted.")

        forecast_obj = self.result.get_forecast(steps=steps)
        forecast_df = forecast_obj.summary_frame(alpha=0.05)
        forecast_df = forecast_df[["mean", "mean_ci_lower", "mean_ci_upper"]]
        forecast_df.columns = ["forecast_mm", "lower_95_mm", "upper_95_mm"]
        forecast_df["forecast_mm"] = forecast_df["forecast_mm"].clip(lower=0)
        return forecast_df

    def residual_diagnostics(self) -> None:
        if self.result:
            print(self.result.summary())


if __name__ == "__main__":
    dates = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    np.random.seed(99)
    t = np.arange(len(dates))
    seasonal = 50 + 40 * np.cos(2 * np.pi * t / 12)   # peak in winter
    noise = np.random.exponential(10, len(t))
    precip = np.maximum(seasonal + noise - 20, 0)

    series = pd.Series(precip, index=dates, name="precip_mm")

    model = MonthlyPrecipitationModel("Tirana")
    model.fit_prophet(series)
    fc = model.forecast_prophet(periods=12)
    print(fc.to_string())
