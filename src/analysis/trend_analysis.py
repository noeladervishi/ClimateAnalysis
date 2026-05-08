import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import pymannkendall as mk
import ruptures as rpt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm

HAS_MK = True

class TrendAnalyser:

    # Mann-Kendall Tests
    def mann_kendall(self, series: pd.Series, alpha: float = 0.05) -> dict:
        data = series.dropna().values

        if HAS_MK:
            result = mk.original_test(data, alpha=alpha)
            return {
                "trend":     result.trend,
                "p_value":   result.p,
                "z_score":   result.z,
                "tau":       result.Tau,
                "slope":     result.slope,        # Sen's slope
                "intercept": result.intercept,
                "significant": result.h,
            }
        else:
            # Fallback: manual Mann-Kendall
            return self._manual_mann_kendall(data, alpha)

    def _manual_mann_kendall(self, data: np.ndarray, alpha: float = 0.05) -> dict:
        n = len(data)
        S = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                diff = data[j] - data[k]
                if diff > 0:
                    S += 1
                elif diff < 0:
                    S -= 1

        var_S = n * (n - 1) * (2 * n + 5) / 18
        if S > 0:
            z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            z = (S + 1) / np.sqrt(var_S)
        else:
            z = 0

        p_value = 2 * (1 - norm.cdf(abs(z)))
        trend = "increasing" if S > 0 and p_value < alpha else \
                "decreasing" if S < 0 and p_value < alpha else "no trend"

        # Sen's slope
        n_obs = len(data)
        slopes = []
        for k in range(n_obs - 1):
            for j in range(k + 1, n_obs):
                if j != k:
                    slopes.append((data[j] - data[k]) / (j - k))
        sen_slope = np.median(slopes) if slopes else 0.0

        return {
            "trend": trend,
            "p_value": round(p_value, 4),
            "z_score": round(z, 4),
            "tau": None,
            "slope": round(sen_slope, 6),
            "intercept": None,
            "significant": p_value < alpha,
        }

    def seasonal_mann_kendall(self, series: pd.Series, alpha: float = 0.05) -> dict:
        if not HAS_MK:
            logger.warning("pymannkendall required for seasonal MK test.")
            return self.mann_kendall(series, alpha)

        data = series.dropna().values
        result = mk.seasonal_test(data, period=12, alpha=alpha)
        return {
            "trend":     result.trend,
            "p_value":   result.p,
            "slope":     result.slope,
            "significant": result.h,
        }

    def trend_all_variables(
        self, df: pd.DataFrame, variables: list[str] | None = None, alpha: float = 0.05
    ) -> pd.DataFrame:
        
        variables = variables or df.select_dtypes(include=[np.number]).columns.tolist()
        results = []
        for var in variables:
            if var not in df.columns:
                continue
            mk_result = self.mann_kendall(df[var], alpha=alpha)
            mk_result["variable"] = var
            results.append(mk_result)

        summary = pd.DataFrame(results).set_index("variable")
        logger.info(f"Trend test complete for {len(summary)} variables.")
        return summary

    # Linear Regression Trend
    def linear_trend(self, series: pd.Series) -> dict:
        series_clean = series.dropna()
        x = np.arange(len(series_clean))
        y = series_clean.values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        fitted = slope * x + intercept

        logger.info(
            f"Linear trend: slope={slope:.4f}/unit | R²={r_value**2:.3f} | "
            f"p={p_value:.4f} | {'Significant' if p_value < 0.05 else 'Not significant'}"
        )

        return {
            "slope":     slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value":   p_value,
            "std_err":   std_err,
            "fitted":    pd.Series(fitted, index=series_clean.index),
        }

    def decadal_change_rate(
        self, series: pd.Series, units_per_decade: bool = True
    ) -> float:
        
        mk_result = self.mann_kendall(series)
        slope_per_step = mk_result["slope"]

        # Determine the time step in the series
        if hasattr(series.index, "freq") and series.index.freq:
            freq = series.index.freq.name
        else:
            delta = (series.index[1] - series.index[0]).days
            freq = "monthly" if delta < 40 else "annual"

        steps_per_decade = 120 if "M" in str(freq) or "monthly" in str(freq) else 10
        rate_per_decade = slope_per_step * steps_per_decade

        logger.info(f"Rate of change: {rate_per_decade:.4f} units/decade")
        return rate_per_decade

    # Change Point Detection
    def detect_changepoints(
        self, series: pd.Series, n_changepoints: int = 3
    ) -> list[pd.Timestamp]:
        
        signal = series.dropna().values.reshape(-1, 1)
        model = rpt.Pelt(model="rbf").fit(signal)
        breakpoints_idx = model.predict(n_bkps=n_changepoints)[:-1]  # exclude last

        timestamps = [series.dropna().index[i] for i in breakpoints_idx if i < len(series.dropna())]
        logger.info(f"Change points detected at: {[str(t.date()) for t in timestamps]}")
        return timestamps

    def cusum(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        cusum = (series - mean).cumsum()
        cusum.name = f"CUSUM_{series.name}"
        return cusum

    # Smoothing
    def moving_average(
        self, series: pd.Series, window: int = 12, center: bool = True
    ) -> pd.Series:
        smoothed = series.rolling(window=window, center=center, min_periods=1).mean()
        smoothed.name = f"{series.name}_MA{window}"
        return smoothed

    def loess_smooth(self, series: pd.Series, frac: float = 0.1) -> pd.Series:
        x = np.arange(len(series))
        y = series.fillna(method="ffill").values
        smoothed = lowess(y, x, frac=frac, return_sorted=False)
        result = pd.Series(smoothed, index=series.index, name=f"{series.name}_LOESS")
        return result

    # Decadal & Period Comparisons
    def decadal_means(self, series: pd.Series) -> pd.DataFrame:
        df = series.to_frame()
        df["decade"] = (df.index.year // 10) * 10
        decadal = df.groupby("decade")[series.name].agg(["mean", "std", "count"])
        decadal.columns = ["mean", "std", "n_years"]
        baseline_mean = decadal.loc[1981:2010, "mean"].mean() if 1981 in decadal.index else decadal["mean"].mean()
        decadal["anomaly_vs_baseline"] = decadal["mean"] - baseline_mean
        return decadal

    def season_trends(self, series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
        seasons = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
        results = []
        for season_name, months in seasons.items():
            subset = series[series.index.month.isin(months)]
            result = self.mann_kendall(subset, alpha)
            result["season"] = season_name
            results.append(result)
        df = pd.DataFrame(results).set_index("season")
        logger.info(f"Seasonal trend analysis complete:\n{df[['trend','slope','p_value']]}")
        return df

    def generate_trend_report(
        self,
        df: pd.DataFrame,
        city_name: str = "Albania",
    ) -> str:
        report_lines = [
            f"═══════════════════════════════════════════",
            f"  Climate Trend Report — {city_name}",
            f"  Period: {df.index[0]} – {df.index[-1]}",
            f"═══════════════════════════════════════════\n",
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mk = self.mann_kendall(df[col])
            decade_rate = self.decadal_change_rate(df[col])
            sig = "✓ SIGNIFICANT" if mk["significant"] else "  not significant"
            report_lines.append(
                f"  {col:35s}: {mk['trend']:12s} | slope={decade_rate:+.3f}/decade | "
                f"p={mk['p_value']:.3f} {sig}"
            )

        report = "\n".join(report_lines)
        logger.info("Trend report generated.")
        return report
