import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import VIZ_CONFIG, PLOTS_DIR, ALBANIA_CITIES


class ClimateVisualiser:
    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        self.output_dir = PLOTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-whitegrid")

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
        logger.success(f"Plot saved: {path}")
        return path

    # Time Series
    def temperature_timeseries(
        self,
        series: pd.Series,
        city: str = "Albania",
        trend_line: pd.Series | None = None,
        anomaly: pd.Series | None = None,
        output_name: str | None = None,
    ) -> plt.Figure:
        n_panels = 2 if anomaly is not None else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        # Panel 1: Raw series + trend
        ax = axes[0]
        ax.plot(series.index, series.values, color="#D73027", lw=1.2,
                alpha=0.7, label="Monthly mean Temp (°C)")

        if trend_line is not None:
            ax.plot(trend_line.index, trend_line.values, "k--", lw=2,
                    label="Trend (Sen's slope)")

        # Rolling 12-month mean
        smooth = series.rolling(12, center=True).mean()
        ax.plot(smooth.index, smooth.values, color="#1A237E", lw=2.5,
                label="12-month rolling mean")

        ax.set_ylabel("Temperature (°C)", fontsize=11)
        ax.set_title(f"Temperature — {city} (2020–2025)", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.axhline(series.mean(), color="grey", lw=0.8, ls="--", alpha=0.5)

        # Panel 2: Anomaly
        if anomaly is not None:
            ax2 = axes[1]
            pos = anomaly.clip(lower=0)
            neg = anomaly.clip(upper=0)
            ax2.bar(anomaly.index, pos.values, width=20, color="#D73027", alpha=0.8, label="Warm anomaly")
            ax2.bar(anomaly.index, neg.values, width=20, color="#4575B4", alpha=0.8, label="Cold anomaly")
            ax2.axhline(0, color="black", lw=0.8)
            ax2.set_ylabel("Anomaly (°C)", fontsize=11)
            ax2.set_title("Temperature Anomaly vs 2020-2025 Baseline", fontsize=12)
            ax2.legend(fontsize=9)

        # X-axis formatting
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30)

        plt.tight_layout()
        if output_name:
            self._save(fig, output_name)
        return fig

    def precipitation_timeseries(
        self,
        monthly_precip: pd.Series,
        city: str = "Albania",
        spi_series: pd.Series | None = None,
        output_name: str | None = None,
    ) -> plt.Figure:
        n = 2 if spi_series is not None else 1
        fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), sharex=True)
        if n == 1:
            axes = [axes]

        ax = axes[0]
        ax.bar(monthly_precip.index, monthly_precip.values,
               width=20, color="#2166AC", alpha=0.75, label="Precipitation (mm)")
        smooth = monthly_precip.rolling(12, center=True).mean()
        ax.plot(smooth.index, smooth.values, color="#B22222", lw=2, label="12-month mean")
        ax.set_ylabel("Precipitation (mm)", fontsize=11)
        ax.set_title(f"Monthly Precipitation — {city}", fontsize=13, fontweight="bold")
        ax.legend()

        if spi_series is not None:
            ax2 = axes[1]
            colors = ["#D73027" if v < 0 else "#4575B4" for v in spi_series.values]
            ax2.bar(spi_series.index, spi_series.values, width=20, color=colors, alpha=0.85)
            ax2.axhline(0, color="black", lw=0.8)
            ax2.axhline(-1.5, color="red", lw=1.2, ls="--", label="Severe drought (SPI=-1.5)")
            ax2.axhline(1.5, color="blue", lw=1.2, ls="--", label="Very wet (SPI=+1.5)")
            ax2.set_ylabel("SPI-3", fontsize=11)
            ax2.set_title("Standardised Precipitation Index (SPI-3)", fontsize=12)
            ax2.legend(fontsize=9)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
        plt.tight_layout()

        if output_name:
            self._save(fig, output_name)
        return fig

    # Seasonal Climatology
    def seasonal_climatology(
        self,
        temp_series: pd.Series,
        precip_series: pd.Series,
        city: str = "Tirana",
        output_name: str | None = None,
    ) -> plt.Figure:
        
        monthly_temp = temp_series.groupby(temp_series.index.month).mean()
        monthly_precip = precip_series.groupby(precip_series.index.month).mean()

        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        x = np.arange(12)

        fig, ax1 = plt.subplots(figsize=(11, 6), dpi=VIZ_CONFIG["figure_dpi"])
        ax2 = ax1.twinx()

        # Temperature line
        ax1.plot(x, monthly_temp.values, "r-o", lw=2.5, markersize=7,
                 label="Mean Temperature (°C)", zorder=5)
        ax1.fill_between(x, monthly_temp.values,
                          alpha=0.15, color="red")
        ax1.set_ylabel("Temperature (°C)", color="red", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="red")

        # Precipitation bars
        ax2.bar(x, monthly_precip.values, 0.6, color="#2166AC",
                alpha=0.6, label="Mean Precipitation (mm)", zorder=2)
        ax2.set_ylabel("Precipitation (mm)", color="#2166AC", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#2166AC")

        # Formatting
        ax1.set_xticks(x)
        ax1.set_xticklabels(months, fontsize=11)
        ax1.set_title(f"Climate Diagram — {city}, Albania", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

        # Annual stats annotation
        ann_temp = temp_series.mean()
        ann_precip = precip_series.sum() / len(temp_series.index.year.unique())
        ax1.text(
            0.98, 0.95,
            f"Annual mean T: {ann_temp:.1f}°C\nAnnual mean P: {ann_precip:.0f} mm",
            transform=ax1.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

        plt.tight_layout()
        if output_name:
            self._save(fig, output_name)
        return fig

    # Warming Stripes
    def warming_stripes(
        self,
        annual_temp_series: pd.Series,
        title: str = "Albania Warming Stripes",
        output_name: str | None = None,
    ) -> plt.Figure:
        
        series = annual_temp_series.dropna()
        mean = series.mean()
        anomaly = series - mean

        vmax = abs(anomaly).max()
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")

        fig, ax = plt.subplots(figsize=(14, 4), dpi=VIZ_CONFIG["figure_dpi"])

        for i, (year, val) in enumerate(anomaly.items()):
            color = cmap(norm(val))
            ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="none"))

        # Colourbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                             fraction=0.04, pad=0.15, shrink=0.5)
        cbar.set_label("Temperature anomaly (°C) vs baseline", fontsize=10)

        # Year labels
        years = list(anomaly.index)
        tick_positions = [i for i, y in enumerate(pd.to_datetime(years).year) if y % 10 == 0]
        tick_labels = [str(years[i]) for i in tick_positions]
        ax.set_xticks([p + 0.5 for p in tick_positions])
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_yticks([])
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlim(0, len(series))
        ax.set_ylim(0, 1)

        plt.tight_layout()
        if output_name:
            self._save(fig, output_name)
        return fig

    # Multi-City Comparison
    def city_comparison_boxplot(
        self,
        df: pd.DataFrame,
        value_col: str,
        city_col: str = "city",
        title: str = "Temperature Distribution by City",
        output_name: str | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 7), dpi=VIZ_CONFIG["figure_dpi"])

        city_order = df.groupby(city_col)[value_col].median().sort_values().index.tolist()
        df_plot = df[df[city_col].isin(city_order)]

        palette = sns.color_palette("RdYlBu_r", n_colors=len(city_order))
        sns.boxplot(
            data=df_plot, x=city_col, y=value_col,
            order=city_order, palette=palette, ax=ax,
            flierprops={"marker": ".", "markerfacecolor": "grey", "markersize": 3},
        )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("City", fontsize=11)
        ax.set_ylabel(value_col, fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        if output_name:
            self._save(fig, output_name)
        return fig

    def trend_comparison_by_season(
        self,
        seasonal_trends_df: pd.DataFrame,
        output_name: str | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=VIZ_CONFIG["figure_dpi"])

        seasons = seasonal_trends_df.index.tolist()
        slopes = seasonal_trends_df["slope"].values * 120  # monthly slope → /decade
        colors = ["#FF7043" if s > 0 else "#42A5F5" for s in slopes]

        bars = ax.bar(seasons, slopes, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)

        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Trend (°C/decade)", fontsize=12)
        ax.set_title("Seasonal Temperature Trends — Albania", fontsize=13, fontweight="bold")

        for bar, slope, sig in zip(
            bars, slopes, seasonal_trends_df["significant"].values
        ):
            marker = "**" if sig else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * (1 if slope >= 0 else -1),
                f"{slope:+.2f}{marker}", ha="center", fontsize=10,
            )

        ax.text(
            0.99, 0.02, "** p < 0.05",
            transform=ax.transAxes, ha="right", fontsize=9, style="italic"
        )

        plt.tight_layout()
        if output_name:
            self._save(fig, output_name)
        return fig

    def forecast_plot(
        self,
        historical: pd.Series,
        forecast_df: pd.DataFrame,
        forecast_col: str = "forecast",
        lower_col: str | None = None,
        upper_col: str | None = None,
        title: str = "Temperature Forecast — Albania",
        output_name: str | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(13, 6), dpi=VIZ_CONFIG["figure_dpi"])

        # Historical
        ax.plot(historical.index, historical.values, color="#1A237E", lw=1.5,
                label="Historical", alpha=0.85)

        # Forecast
        fc_dates = pd.DatetimeIndex(forecast_df.index if "date" not in forecast_df.columns
                                    else forecast_df["date"])
        ax.plot(fc_dates, forecast_df[forecast_col].values,
                color="#D73027", lw=2, label="Forecast", ls="--")

        # Confidence interval
        if lower_col and upper_col and lower_col in forecast_df.columns:
            ax.fill_between(
                fc_dates,
                forecast_df[lower_col].values,
                forecast_df[upper_col].values,
                alpha=0.2, color="#D73027", label="95% CI",
            )

        # Vertical line at forecast start
        ax.axvline(fc_dates[0], color="grey", lw=1, ls=":", label="Forecast start")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=11)
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        plt.tight_layout()

        if output_name:
            self._save(fig, output_name)
        return fig
