import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from typing import Any
from config.settings import ALBANIA_BBOX

# Plausible Ranges for Albania
ALBANIA_CLIMATE_BOUNDS = {
    # Variable              (absolute_min, absolute_max, warn_min, warn_max)
    "temperature_2m_mean":   (-25.0, 45.0,  -15.0, 40.0),
    "temperature_2m_max":    (-20.0, 48.0,  -10.0, 45.0),
    "temperature_2m_min":    (-30.0, 35.0,  -20.0, 30.0),
    "precipitation_sum":     (0.0, 300.0,    0.0, 200.0),
    "relative_humidity_2m_mean": (5.0, 100.0, 10.0, 98.0),
    "windspeed_10m_max":     (0.0, 150.0,    0.0, 100.0),
    "snowfall_sum":          (0.0, 200.0,    0.0, 100.0),
    "spi":                   (-4.0, 4.0,    -3.5, 3.5),
    "ndvi":                  (-1.0, 1.0,    -0.2, 0.95),
    "lst_celsius":           (-25.0, 65.0,  -15.0, 60.0),
}


@dataclass
class ValidationReport:
    variable: str
    n_records: int
    n_missing: int
    n_below_absolute: int
    n_above_absolute: int
    n_warnings: int
    error_indices: list = field(default_factory=list)
    warning_indices: list = field(default_factory=list)
    passed: bool = True

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} | {self.variable} | "
            f"n={self.n_records} | missing={self.n_missing} | "
            f"errors={self.n_below_absolute + self.n_above_absolute} | "
            f"warnings={self.n_warnings}"
        )


class ClimateDataValidator:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.reports: list[ValidationReport] = []

    # Column-Level Validation
    def validate_series(
        self,
        series: pd.Series,
        variable: str,
        max_missing_pct: float = 10.0,
    ) -> ValidationReport:
        
        n = len(series)
        n_missing = series.isna().sum()
        missing_pct = n_missing / n * 100

        bounds = ALBANIA_CLIMATE_BOUNDS.get(variable, (-1e9, 1e9, -1e9, 1e9))
        abs_min, abs_max, warn_min, warn_max = bounds

        non_null = series.dropna()
        below_abs = (non_null < abs_min).sum()
        above_abs = (non_null > abs_max).sum()
        below_warn = ((non_null >= abs_min) & (non_null < warn_min)).sum()
        above_warn = ((non_null <= abs_max) & (non_null > warn_max)).sum()

        error_idx = list(non_null[(non_null < abs_min) | (non_null > abs_max)].index[:10])
        warn_idx  = list(non_null[(non_null < warn_min) | (non_null > warn_max)].index[:10])

        passed = (
            missing_pct <= max_missing_pct
            and below_abs == 0
            and above_abs == 0
        )

        report = ValidationReport(
            variable=variable,
            n_records=n,
            n_missing=int(n_missing),
            n_below_absolute=int(below_abs),
            n_above_absolute=int(above_abs),
            n_warnings=int(below_warn + above_warn),
            error_indices=error_idx,
            warning_indices=warn_idx,
            passed=passed,
        )

        if not passed:
            logger.error(str(report))
        elif report.n_warnings > 0:
            logger.warning(str(report))
        else:
            logger.success(str(report))

        self.reports.append(report)
        return report

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        variables: list[str] | None = None,
        max_missing_pct: float = 10.0,
    ) -> dict[str, ValidationReport]:

        variables = variables or [
            col for col in df.columns
            if col in ALBANIA_CLIMATE_BOUNDS
        ]
        if not variables:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {}
        for var in variables:
            if var not in df.columns:
                logger.warning(f"Variable '{var}' not found in DataFrame.")
                continue
            results[var] = self.validate_series(df[var], var, max_missing_pct)

        n_pass = sum(1 for r in results.values() if r.passed)
        logger.info(f"Validation complete: {n_pass}/{len(results)} variables passed.")
        return results

    # Temporal Validation
    def check_temporal_gaps(
        self,
        df: pd.DataFrame,
        expected_freq: str = "D",
        max_gap_tolerance: int = 7,
    ) -> dict:
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"error": "DataFrame index is not a DatetimeIndex"}

        expected_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
        missing_dates = expected_range.difference(df.index)

        if len(missing_dates) == 0:
            logger.success("No temporal gaps detected.")
            return {"gaps": 0, "missing_dates": []}

        # Find consecutive runs
        gaps = []
        run_start = missing_dates[0]
        run_end = missing_dates[0]
        for d in missing_dates[1:]:
            delta = (d - run_end).days if expected_freq == "D" else 1
            if delta <= 1:
                run_end = d
            else:
                gaps.append((run_start, run_end, (run_end - run_start).days + 1))
                run_start = run_end = d
        gaps.append((run_start, run_end, (run_end - run_start).days + 1))

        large_gaps = [(s, e, n) for s, e, n in gaps if n > max_gap_tolerance]

        logger.warning(
            f"Temporal gaps: {len(missing_dates)} missing {expected_freq} steps "
            f"in {len(gaps)} runs ({len(large_gaps)} large gaps > {max_gap_tolerance} units)"
        )

        return {
            "total_missing": len(missing_dates),
            "n_runs": len(gaps),
            "n_large_gaps": len(large_gaps),
            "large_gaps": [(str(s.date()), str(e.date()), n) for s, e, n in large_gaps[:5]],
        }

    def check_duplicate_timestamps(self, df: pd.DataFrame) -> int:
        dupes = df.index.duplicated().sum()
        if dupes:
            logger.warning(f"Duplicate timestamps: {dupes}")
        else:
            logger.success("No duplicate timestamps.")
        return int(dupes)

    def check_monotonic_index(self, df: pd.DataFrame) -> bool:
        is_mono = df.index.is_monotonic_increasing
        if not is_mono:
            logger.error("Index is NOT monotonically increasing. Sort by index.")
        return is_mono

    # Coordinate Validation
    def validate_coordinates(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        strict_albania: bool = True,
    ) -> pd.Series:

        bbox = ALBANIA_BBOX

        valid_lat = df[lat_col].between(bbox["min_lat"], bbox["max_lat"])
        valid_lon = df[lon_col].between(bbox["min_lon"], bbox["max_lon"])
        valid = valid_lat & valid_lon

        n_invalid = (~valid).sum()
        if n_invalid:
            logger.warning(
                f"{n_invalid} records have coordinates outside Albania bounding box."
            )
            if not strict_albania:
                # Allow ±0.5° tolerance (border areas)
                valid_lat_loose = df[lat_col].between(bbox["min_lat"] - 0.5, bbox["max_lat"] + 0.5)
                valid_lon_loose = df[lon_col].between(bbox["min_lon"] - 0.5, bbox["max_lon"] + 0.5)
                valid = valid_lat_loose & valid_lon_loose
        else:
            logger.success("All coordinates are within Albania bounds.")

        return valid

    # Physical Consistency
    def check_tmax_gt_tmin(
        self,
        df: pd.DataFrame,
        tmax_col: str = "temperature_2m_max",
        tmin_col: str = "temperature_2m_min",
    ) -> int:
       
        if tmax_col not in df.columns or tmin_col not in df.columns:
            return 0
        violations = (df[tmax_col] < df[tmin_col]).sum()
        if violations:
            logger.error(f"Tmax < Tmin violations: {violations}")
        else:
            logger.success("Tmax >= Tmin check passed.")
        return int(violations)

    def check_precip_non_negative(
        self,
        df: pd.DataFrame,
        precip_col: str = "precipitation_sum",
    ) -> int:
        if precip_col not in df.columns:
            return 0
        negatives = (df[precip_col] < 0).sum()
        if negatives:
            logger.error(f"Negative precipitation values: {negatives}")
        return int(negatives)

    # Summary
    def summary_report(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "variable":     r.variable,
                "n_records":    r.n_records,
                "n_missing":    r.n_missing,
                "missing_pct":  round(r.n_missing / r.n_records * 100, 2),
                "errors":       r.n_below_absolute + r.n_above_absolute,
                "warnings":     r.n_warnings,
                "passed":       r.passed,
            }
            for r in self.reports
        ])

    def run_full_validation(
        self,
        df: pd.DataFrame,
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> tuple[bool, pd.DataFrame]:

        logger.info("═══ Running Full Dataset Validation ═══")

        # Range validation
        self.validate_dataframe(df)

        # Temporal checks
        if isinstance(df.index, pd.DatetimeIndex):
            self.check_temporal_gaps(df)
            self.check_duplicate_timestamps(df)
            self.check_monotonic_index(df)

        # Physical consistency
        self.check_tmax_gt_tmin(df)
        self.check_precip_non_negative(df)

        # Coordinates
        if lat_col and lon_col:
            self.validate_coordinates(df, lat_col, lon_col)

        summary = self.summary_report()
        all_passed = all(r.passed for r in self.reports)
        logger.info(f"Validation complete. All passed: {all_passed}")
        return all_passed, summary
