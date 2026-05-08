import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import THRESHOLDS, PROCESSED_DIR


@dataclass
class ExtremeEvent:
    event_type: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    max_intensity: float
    mean_intensity: float
    location: str = "Albania"
    notes: str = ""

    @property
    def severity(self) -> str:
        if self.duration_days >= 14:  return "Extreme"
        if self.duration_days >= 7:   return "Severe"
        if self.duration_days >= 3:   return "Moderate"
        return "Mild"


class ExtremeEventDetector:
    def __init__(self):
        self.output_dir = PROCESSED_DIR / "extreme_events"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Heatwaves
    def detect_heatwaves(
        self,
        tmax: pd.Series,
        tmean: pd.Series,
        threshold_tmax: float | None = None,
        min_duration_days: int = 3,
        method: str = "fixed",
    ) -> list[ExtremeEvent]:
        threshold_tmax = threshold_tmax or THRESHOLDS["heatwave_temp_c"]

        if method == "fixed":
            return self._detect_runs(
                condition=(tmax > threshold_tmax),
                intensity_series=tmax,
                min_days=min_duration_days,
                event_type="Heatwave",
            )
        elif method == "euro":
            condition = (tmax > 30.0) & (tmean > 25.0)
            return self._detect_runs(condition, tmax, min_duration_days, "Heatwave (Euro)")
        elif method == "ehf":
            return self._detect_heatwaves_ehf(tmax, tmean, min_duration_days)
        else:
            raise ValueError(f"Unknown heatwave method: {method}")

    def _detect_heatwaves_ehf(
        self,
        tmax: pd.Series,
        tmean: pd.Series,
        min_duration: int = 3,
    ) -> list[ExtremeEvent]:
       
        # 3-day rolling mean
        t3d = tmean.rolling(3, min_periods=1).mean()
        # 30-day rolling mean (climatological normal)
        t30d = tmean.rolling(30, min_periods=10).mean()
        # 90th percentile baseline (per calendar day)
        t90 = tmean.groupby(tmean.index.day_of_year).transform(lambda x: x.quantile(0.9))

        ehi_sig = t3d - t90
        ehi_accl = (t3d - t30d) / 3.0
        ehf = ehi_sig * np.maximum(1, ehi_accl)

        condition = ehf > 0
        events = self._detect_runs(condition, tmax, min_duration, "Heatwave (EHF)")
        logger.info(f"EHF heatwaves detected: {len(events)}")
        return events

    # Cold Spells
    def detect_cold_spells(
        self,
        tmin: pd.Series,
        threshold_c: float | None = None,
        min_duration_days: int = 3,
    ) -> list[ExtremeEvent]:
        threshold_c = threshold_c or THRESHOLDS["cold_day_c"]
        condition = tmin < threshold_c
        return self._detect_runs(
            condition, tmin, min_duration_days, "Cold Spell",
            intensity_is_min=True,
        )

    # Droughts (SPI-based)
    def detect_droughts(
        self,
        spi_series: pd.Series,
        threshold: float | None = None,
        min_duration_months: int = 2,
    ) -> list[ExtremeEvent]:
        threshold = threshold or THRESHOLDS["drought_spi_threshold"]
        condition = spi_series < threshold
        events = self._detect_runs(
            condition, spi_series, min_duration_months,
            "Drought", intensity_is_min=True,
        )
        logger.info(f"Drought events detected: {len(events)} (SPI < {threshold})")
        return events

    # Extreme Precipitation / Floods
    def detect_extreme_precipitation(
        self,
        precip_daily: pd.Series,
        threshold_mm: float | None = None,
        min_duration_days: int = 1,
    ) -> list[ExtremeEvent]:
        threshold_mm = threshold_mm or THRESHOLDS["extreme_precip_mm"]
        condition = precip_daily >= threshold_mm
        events = self._detect_runs(
            condition, precip_daily, min_duration_days, "Extreme Precipitation"
        )
        logger.info(f"Extreme precipitation events (≥{threshold_mm}mm): {len(events)}")
        return events

    def flood_risk_index(
        self,
        precip_5day: pd.Series,
        antecedent_30d: pd.Series,
    ) -> pd.Series:
        p5d = precip_5day.rolling(5, min_periods=1).sum()
        p30d_norm = (antecedent_30d.rolling(30).sum() - antecedent_30d.rolling(30).sum().min()) / \
                    (antecedent_30d.rolling(30).sum().max() - antecedent_30d.rolling(30).sum().min() + 1e-9)
        fri = p5d * (1 + p30d_norm)
        fri.name = "FRI"
        return fri
    
    # Wildfire Weather (Fire Weather Index)
    def fire_weather_index_simplified(
        self,
        tmax: pd.Series,
        rh: pd.Series,          # relative humidity (%)
        wind_speed: pd.Series,  # km/h
        precip: pd.Series,      # mm/day
    ) -> pd.Series:
        drought_factor = 1 / (precip.rolling(7).sum().clip(lower=0.1))
        fwi = (tmax * wind_speed * drought_factor) / (rh.clip(lower=1))
        fwi = fwi.clip(lower=0)
        fwi.name = "FWI_simplified"

        # Log high-risk days
        high_risk = (fwi > 34).sum()
        logger.info(f"Fire Weather Index — high/extreme risk days: {high_risk}")
        return fwi

    # Snowfall / Snow Cover (Alpine Albania)
    def detect_heavy_snowfall(
        self,
        snowfall_series: pd.Series,
        threshold_cm: float = 30.0,
        min_duration_days: int = 1,
    ) -> list[ExtremeEvent]:
        condition = snowfall_series >= threshold_cm
        events = self._detect_runs(condition, snowfall_series, min_duration_days, "Heavy Snowfall")
        logger.info(f"Heavy snowfall events (≥{threshold_cm}cm): {len(events)}")
        return events

    # Core Helper: Run Detection
    def _detect_runs(
        self,
        condition: pd.Series,
        intensity_series: pd.Series,
        min_days: int,
        event_type: str,
        intensity_is_min: bool = False,
    ) -> list[ExtremeEvent]:
        events = []
        in_event = False
        start = None

        for date, val in condition.items():
            if val and not in_event:
                in_event = True
                start = date
            elif not val and in_event:
                end = date - pd.Timedelta(days=1)
                duration = (end - start).days + 1
                if duration >= min_days:
                    segment = intensity_series.loc[start:end]
                    events.append(ExtremeEvent(
                        event_type=event_type,
                        start_date=start,
                        end_date=end,
                        duration_days=duration,
                        max_intensity=float(segment.min() if intensity_is_min else segment.max()),
                        mean_intensity=float(segment.mean()),
                    ))
                in_event = False

        # Close final event if it extends to end of series
        if in_event and start is not None:
            end = condition.index[-1]
            duration = (end - start).days + 1
            if duration >= min_days:
                segment = intensity_series.loc[start:end]
                events.append(ExtremeEvent(
                    event_type=event_type,
                    start_date=start,
                    end_date=end,
                    duration_days=duration,
                    max_intensity=float(segment.min() if intensity_is_min else segment.max()),
                    mean_intensity=float(segment.mean()),
                ))

        return events
    
    # Summary & Export
    def events_to_dataframe(self, events: list[ExtremeEvent]) -> pd.DataFrame:
        records = [
            {
                "type": e.event_type,
                "start": e.start_date,
                "end": e.end_date,
                "duration_days": e.duration_days,
                "max_intensity": e.max_intensity,
                "mean_intensity": e.mean_intensity,
                "severity": e.severity,
                "location": e.location,
            }
            for e in events
        ]
        return pd.DataFrame(records)

    def annual_event_summary(self, events: list[ExtremeEvent]) -> pd.DataFrame:
        df = self.events_to_dataframe(events)
        if df.empty:
            return pd.DataFrame()
        df["year"] = pd.to_datetime(df["start"]).dt.year
        summary = df.groupby("year").agg(
            count=("type", "count"),
            total_days=("duration_days", "sum"),
            max_intensity=("max_intensity", "max"),
        )
        return summary

    def save_events(self, events: list[ExtremeEvent], name: str) -> Path:
        df = self.events_to_dataframe(events)
        path = self.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.success(f"Saved {len(events)} events → {path}")
        return path