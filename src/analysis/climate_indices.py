import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from scipy.special import gammainc
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import THRESHOLDS, PROCESSED_DIR


class ClimateIndices:
    def __init__(self):
        self.output_dir = PROCESSED_DIR / "indices"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # SPI — Standardised Precipitation Index
    def compute_spi(
        self,
        precip_series: pd.Series,
        timescale_months: int = 3,
    ) -> pd.Series:
        
        rolling = precip_series.rolling(window=timescale_months, min_periods=1).sum()
        spi = pd.Series(index=precip_series.index, dtype=float, name=f"SPI_{timescale_months}")

        for month in range(1, 13):
            idx = rolling.index.month == month
            data = rolling.loc[idx].dropna()

            if len(data) < 10:
                continue

            # Gamma distribution fit (skip zeros)
            non_zero = data[data > 0]
            if len(non_zero) < 5:
                continue

            zero_prob = (data == 0).sum() / len(data)

            try:
                shape, loc, scale = gamma.fit(non_zero, floc=0)
                # CDF with adjustment for zero values
                cdf = zero_prob + (1 - zero_prob) * gamma.cdf(data, shape, loc, scale)
                cdf = np.clip(cdf, 1e-4, 1 - 1e-4)   # avoid ±inf
                spi.loc[idx] = norm.ppf(cdf)
            except Exception as e:
                logger.warning(f"SPI fit failed for month {month}: {e}")

        logger.info(f"SPI-{timescale_months} computed. Drought months: "
                    f"{(spi < THRESHOLDS['drought_spi_threshold']).sum()}")
        return spi

    def compute_spi_multi(
        self,
        precip_series: pd.Series,
        timescales: list[int] = [1, 3, 6, 12],
    ) -> pd.DataFrame:
        """Compute SPI at multiple timescales simultaneously."""
        results = {}
        for ts in timescales:
            results[f"SPI_{ts}"] = self.compute_spi(precip_series, ts)
        return pd.DataFrame(results)

    # Temperature-Based Indices
    def heat_days(
        self,
        tmax_series: pd.Series,
        threshold_c: float | None = None,
    ) -> pd.Series:
        """Count days per month where Tmax exceeds heatwave threshold."""
        threshold_c = threshold_c or THRESHOLDS["heatwave_temp_c"]
        hot = (tmax_series > threshold_c).astype(int)
        monthly = hot.resample("MS").sum()
        monthly.name = f"heat_days_above_{int(threshold_c)}C"
        logger.info(f"Heat days (>{threshold_c}°C) — annual mean: {monthly.resample('YS').sum().mean():.1f}")
        return monthly

    def frost_days(
        self,
        tmin_series: pd.Series,
        threshold_c: float | None = None,
    ) -> pd.Series:
        """Count days per month where Tmin < frost threshold."""
        threshold_c = threshold_c or THRESHOLDS["frost_temp_c"]
        frost = (tmin_series < threshold_c).astype(int)
        monthly = frost.resample("MS").sum()
        monthly.name = f"frost_days_below_{int(threshold_c)}C"
        return monthly

    def growing_degree_days(
        self,
        tmean_series: pd.Series,
        base_temp_c: float = 10.0,
        max_temp_c: float = 30.0,
    ) -> pd.Series:
        
        effective_temp = tmean_series.clip(upper=max_temp_c)
        gdd = (effective_temp - base_temp_c).clip(lower=0)
        gdd.name = f"GDD_base{int(base_temp_c)}"
        logger.info(f"Annual GDD (base {base_temp_c}°C): {gdd.resample('YS').sum().mean():.1f}")
        return gdd

    def diurnal_temperature_range(
        self,
        tmax_series: pd.Series,
        tmin_series: pd.Series,
    ) -> pd.Series:
        """Diurnal Temperature Range (DTR) = Tmax - Tmin per day."""
        dtr = (tmax_series - tmin_series).clip(lower=0)
        dtr.name = "DTR"
        return dtr

    def tropical_nights(self, tmin_series: pd.Series, threshold_c: float = 20.0) -> pd.Series:
        """Count 'tropical nights' (Tmin > threshold). Increasing in Albania."""
        nights = (tmin_series > threshold_c).astype(int)
        return nights.resample("YS").sum().rename("tropical_nights")

    # Precipitation Indices
    def consecutive_dry_days(self, precip_daily: pd.Series, threshold_mm: float = 1.0) -> pd.Series:
       
        dry = precip_daily < threshold_mm

        def max_run(group):
            runs = (dry.loc[group.index] != dry.loc[group.index].shift()).cumsum()
            run_lengths = dry.loc[group.index].groupby(runs).transform("sum")
            return run_lengths[dry.loc[group.index]].max() if dry.loc[group.index].any() else 0

        cdd = precip_daily.groupby(precip_daily.index.year).apply(
            lambda g: self._max_consecutive(g < threshold_mm)
        )
        cdd.name = "CDD"
        logger.info(f"Mean CDD (consecutive dry days): {cdd.mean():.1f}")
        return cdd

    def consecutive_wet_days(self, precip_daily: pd.Series, threshold_mm: float = 1.0) -> pd.Series:
        """Longest run of consecutive wet days per year (CWD)."""
        cwd = precip_daily.groupby(precip_daily.index.year).apply(
            lambda g: self._max_consecutive(g >= threshold_mm)
        )
        cwd.name = "CWD"
        return cwd

    def extreme_precipitation_days(
        self, precip_daily: pd.Series, threshold_mm: float | None = None
    ) -> pd.Series:
        """Count days per year with extreme precipitation (> threshold)."""
        threshold_mm = threshold_mm or THRESHOLDS["extreme_precip_mm"]
        extreme = (precip_daily >= threshold_mm).astype(int)
        return extreme.resample("YS").sum().rename(f"R{int(threshold_mm)}days")

    def precipitation_percentile_threshold(
        self, precip_daily: pd.Series, percentile: float = 95.0
    ) -> float:
        """Return the Nth percentile of wet-day precipitation (R95p threshold)."""
        wet_days = precip_daily[precip_daily >= 1.0]
        threshold = np.percentile(wet_days, percentile)
        logger.info(f"R{percentile}p threshold for Albania: {threshold:.1f} mm/day")
        return threshold

    # Aridity Indices
    def de_martonne_aridity(
        self,
        annual_precip_mm: float | pd.Series,
        mean_temp_c: float | pd.Series,
    ) -> float | pd.Series:
        
        index = annual_precip_mm / (mean_temp_c + 10)
        if isinstance(index, pd.Series):
            index.name = "de_martonne_AI"
        logger.info(f"De Martonne AI computed.")
        return index

    def unep_aridity_index(
        self,
        annual_precip_mm: float | pd.Series,
        annual_pet_mm: float | pd.Series,
    ) -> float | pd.Series:
        
        ai = annual_precip_mm / (annual_pet_mm + 1e-9)
        if isinstance(ai, pd.Series):
            ai.name = "UNEP_AI"
        return ai

    def hargreaves_pet(
        self,
        tmax: pd.Series,
        tmin: pd.Series,
        tmean: pd.Series,
        latitude_deg: float = 41.15,
    ) -> pd.Series:
        
        doy = tmean.index.day_of_year
        lat_rad = np.radians(latitude_deg)
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        Ra = (24 * 60 / np.pi) * 0.082 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta)
            + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )  # MJ/m²/day
        Ra_mm = Ra * 0.408  # convert to mm/day equivalent

        pet = 0.0023 * Ra_mm * (tmean + 17.8) * np.sqrt((tmax - tmin).clip(lower=0))
        pet = pet.clip(lower=0)
        pet.name = "PET_hargreaves_mm"
        return pet

    # Utility
    @staticmethod
    def _max_consecutive(bool_series: pd.Series) -> int:
        """Find the length of the longest True run in a boolean Series."""
        max_run = 0
        current_run = 0
        for val in bool_series:
            if val:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def classify_spi(self, spi_value: float) -> str:
        """Return human-readable SPI category."""
        if spi_value >= 2.0:   return "Extremely Wet"
        if spi_value >= 1.5:   return "Very Wet"
        if spi_value >= 1.0:   return "Moderately Wet"
        if spi_value >= -1.0:  return "Near Normal"
        if spi_value >= -1.5:  return "Moderately Dry"
        if spi_value >= -2.0:  return "Severely Dry"
        return "Extremely Dry"

    def full_index_report(
        self,
        tmax: pd.Series,
        tmin: pd.Series,
        tmean: pd.Series,
        precip: pd.Series,
    ) -> pd.DataFrame:
        
        annual = {}

        # Temperature
        annual["mean_temp_c"]      = tmean.resample("YS").mean()
        annual["max_temp_c"]       = tmax.resample("YS").max()
        annual["min_temp_c"]       = tmin.resample("YS").min()
        annual["heat_days"]        = self.heat_days(tmax).resample("YS").sum()
        annual["frost_days"]       = self.frost_days(tmin).resample("YS").sum()
        annual["tropical_nights"]  = self.tropical_nights(tmin)
        annual["gdd"]              = self.growing_degree_days(tmean).resample("YS").sum()
        annual["dtr_mean"]         = self.diurnal_temperature_range(tmax, tmin).resample("YS").mean()

        # Precipitation
        annual["total_precip_mm"]  = precip.resample("YS").sum()
        annual["extreme_precip_days"] = self.extreme_precipitation_days(precip)
        annual["cdd"]              = self.consecutive_dry_days(precip)
        annual["cwd"]              = self.consecutive_wet_days(precip)

        # SPI
        spi3 = self.compute_spi(precip, 3)
        annual["spi3_mean"]        = spi3.resample("YS").mean()

        # Aridity
        pet = self.hargreaves_pet(tmax, tmin, tmean)
        annual["pet_mm"]           = pet.resample("YS").sum()
        annual["de_martonne_AI"]   = self.de_martonne_aridity(
            annual["total_precip_mm"], annual["mean_temp_c"]
        )

        df = pd.DataFrame(annual)
        df.index = pd.to_datetime(df.index)
        df.index = df.index.year
        df.index.name = "year"

        path = self.output_dir / "annual_climate_indices.csv"
        df.to_csv(path)
        logger.success(f"Annual index report saved: {path}")
        return df