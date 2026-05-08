import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from src.modeling.model_evaluation import ModelEvaluator
    from config.settings import (
        ALBANIA_CITIES, ALBANIA_CENTER, PROCESSED_DIR, RAW_DIR,
        THRESHOLDS, PLOTS_DIR
    )
except ImportError:
    ALBANIA_CITIES = {
        "Tirana": (19.8189, 41.3275), "Durres": (19.4565, 41.3246),
        "Vlore": (19.4914, 40.4661), "Shkoder": (19.5126, 42.0683),
        "Elbasan": (20.0822, 41.1125), "Korce": (20.7752, 40.6186),
        "Gjirokaster": (20.1389, 40.0758), "Sarande": (20.0053, 39.8752),
        "Berat": (19.9522, 40.7058), "Kukes": (20.4158, 42.0781),
        "Lushnje": (19.7058, 40.9419), "Fier": (19.5569, 40.7239),
    }
    ALBANIA_CENTER = {"lat": 41.1533, "lon": 20.1683}
    PROCESSED_DIR = Path("data/processed")
    RAW_DIR = Path("data/raw")
    THRESHOLDS = {"heatwave_temp_c": 35, "frost_temp_c": 0, "drought_spi_threshold": -1.5}
    PLOTS_DIR = Path("plots")

# Page config
st.set_page_config(
    page_title="Albania Climate Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Data loaders
@st.cache_data(ttl=3600)
def load_city_data(city: str) -> pd.DataFrame:
    try:
        raw_dir = RAW_DIR / "open_meteo"
        pattern = list(raw_dir.glob(f"{city.lower()}_*.csv"))
        if pattern:
            df = pd.read_csv(pattern[0], parse_dates=["time"])
            return df.set_index("time").sort_index()
    except Exception:
        pass
    return _synthetic(city)

@st.cache_data
def _synthetic(city: str) -> pd.DataFrame:
    np.random.seed(hash(city) % 1000)
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    t = np.arange(len(dates))
    bases = {"Tirana":15.5,"Durres":16.2,"Vlore":17.1,"Shkoder":14.8,"Elbasan":15.0,
             "Korce":11.5,"Gjirokaster":14.0,"Sarande":17.8,"Berat":15.3,"Kukes":12.0,
             "Lushnje":15.8,"Fier":16.0}
    b = bases.get(city, 15.0)
    tmean = b + 10*np.sin(2*np.pi*t/365-np.pi/2) + 0.0004*t + np.random.normal(0,2.5,len(t))
    tmax  = tmean + np.random.uniform(4,7,len(t))
    tmin  = tmean - np.random.uniform(4,7,len(t))
    sp    = 2 + 2*np.cos(2*np.pi*t/365)
    prec  = np.random.exponential(sp, len(t))
    prec[sp<1.5] *= 0.2
    df = pd.DataFrame({
        "temperature_2m_mean": tmean, "temperature_2m_max": tmax,
        "temperature_2m_min": tmin, "precipitation_sum": prec,
        "windspeed_10m_max": np.random.exponential(15,len(t)),
        "relative_humidity_2m_mean": np.random.normal(65,12,len(t)).clip(10,100),
    }, index=dates)
    df.index.name = "time"
    return df

@st.cache_data
def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("MS").agg({
        "temperature_2m_mean":"mean","temperature_2m_max":"max",
        "temperature_2m_min":"min","precipitation_sum":"sum",
        "windspeed_10m_max":"mean","relative_humidity_2m_mean":"mean",
    })

@st.cache_resource(show_spinner=False)
def compute_shap_bundle(_model, X):
    try:
        from src.modeling.model_evaluation import ModelEvaluator
        ev = ModelEvaluator()
        sv = ev.shap_values(_model, X, sample_size=300)
        im = ev.shap_mean_importance(_model, X, sample_size=300)
        return sv, im
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def build_shap_ctx(city: str, monthly_df: pd.DataFrame):
    try:
        from src.modeling.temperature_model import TemperatureModel
        from src.preprocessing.time_series_processor import TimeSeriesProcessor
    except Exception as e:
        return {"available": False, "reason": str(e)}
    if "temperature_2m_mean" not in monthly_df.columns:
        return {"available": False, "reason": "No temperature column."}
    try:
        tsp = TimeSeriesProcessor(city=city)
        fdf = tsp.create_ml_features(monthly_df, "temperature_2m_mean")
        if len(fdf) < 18:
            return {"available": False, "reason": "Insufficient rows."}
        train, test = tsp.train_test_split_temporal(fdf)
        model = TemperatureModel(city=city, model_type="xgboost")
        model.fit(train)
        leakage = {"temperature_2m_max","temperature_2m_min"}
        feats = [c for c in model.feature_cols if c not in leakage
                 and not c.endswith("_roll_max_3") and not c.endswith("_roll_max_6")
                 and not c.endswith("_roll_max_12")]
        X = fdf[feats].select_dtypes(include=["number"]).dropna()
        if X.empty:
            return {"available": False, "reason": "No numeric features."}
        y = fdf.loc[X.index, model.target_col]
        model.model.fit(X, y)
        model.feature_cols = list(X.columns)
        sv, imp = compute_shap_bundle(model.model, X)
        if sv is None:
            return {"available": False, "reason": "SHAP computation failed."}
        mean_shap = sv.values.mean(axis=0)
        direction = pd.Series(mean_shap, index=X.columns)
        return {"available": True, "model": model.model, "X": X,
                "shap_values": sv, "importance": imp, "direction": direction,
                "causal_features": list(X.columns)}
    except Exception as e:
        return {"available": False, "reason": str(e)}

@st.cache_data(ttl=3600)
def city_map_data() -> pd.DataFrame:
    rows = []
    for name, (lon, lat) in ALBANIA_CITIES.items():
        df = load_city_data(name)
        rows.append({
            "city": name, "lat": lat, "lon": lon,
            "mean_temp": float(df["temperature_2m_mean"].mean()),
            "annual_precip": float(df["precipitation_sum"].resample("YS").sum().mean()),
        })
    return pd.DataFrame(rows)

# FIX: dedicated loader for SPI data that does not pollute the main df namespace
@st.cache_data(ttl=3600)
def load_spi_data(city: str) -> pd.DataFrame:
    try:
        spi_path = Path("data/processed/indices/albania_spi3.csv")
        spi_df = pd.read_csv(spi_path)
        spi_df["date"] = pd.to_datetime(spi_df["date"])
        if "city" in spi_df.columns:
            spi_df = spi_df[spi_df["city"] == city]
        spi_df = spi_df.sort_values("date").set_index("date")
        return spi_df
    except Exception:
        return pd.DataFrame()

# SHAP helpers: human-readable English labels
FEATURE_LABELS = {
    "month_sin":                               "Season (sine component)",
    "month_cos":                               "Season (cosine component)",
    "month":                                   "Month of year",
    "year":                                    "Calendar year",
    "day_of_year":                             "Day of year",
    "season":                                  "Season",
    "trend_index":                             "Long-term time trend",
    "precipitation_sum":                       "Monthly rainfall (mm)",
    "windspeed_10m_max":                       "Maximum wind speed",
    "relative_humidity_2m_mean":               "Average relative humidity",
    "temperature_2m_mean_lag_1":               "Temperature last month",
    "temperature_2m_mean_lag_2":               "Temperature 2 months ago",
    "temperature_2m_mean_lag_3":               "Temperature 3 months ago",
    "temperature_2m_mean_roll_mean_3":         "3-month average temperature",
    "temperature_2m_mean_roll_mean_6":         "6-month average temperature",
    "temperature_2m_mean_roll_mean_12":        "12-month average temperature",
    "temperature_2m_mean_roll_std_3":          "3-month temperature variability",
    "temperature_2m_mean_roll_std_6":          "6-month temperature variability",
    "temperature_2m_mean_roll_std_12":         "12-month temperature variability",
    "temperature_2m_mean_yoy_diff":            "Year-over-year temperature change",
}

FEATURE_GROUPS = {
    "month":           "Seasonal Forcing",
    "month_sin":       "Seasonal Forcing",
    "month_cos":       "Seasonal Forcing",
    "season":          "Seasonal Forcing",
    "day_of_year":     "Seasonal Forcing",
    "year":            "Long-term Trend",
    "trend_index":     "Long-term Trend",
    "precipitation_sum":             "Weather Covariate",
    "relative_humidity_2m_mean":     "Weather Covariate",
    "windspeed_10m_max":             "Weather Covariate",
}

PANEL_CONTEXT = {
    "Monthly Temperature": (
        "This chart shows how the model predicts monthly average temperature. "
        "The top drivers listed below are the variables that most strongly pushed "
        "predictions up (warmer) or down (cooler) across all months in the dataset."
    ),
    "Monthly Climatology": (
        "This chart shows the typical temperature range for each calendar month. "
        "The SHAP values here highlight which seasonal features (e.g. the time of year) "
        "explain the difference between the warmest and coolest months."
    ),
    "Warming Trend": (
        "This chart visualises year-by-year temperature anomalies as colour stripes. "
        "The SHAP values reveal which long-term trend features (e.g. calendar year, "
        "12-month rolling average) are driving the observed warming or cooling signal."
    ),
    "Monthly Precipitation": (
        "This chart shows monthly rainfall totals over time. "
        "The SHAP drivers here explain how rainfall and humidity influence the "
        "model's temperature predictions month by month."
    ),
    "Annual Precipitation": (
        "This chart shows total annual rainfall for each year. "
        "The SHAP values identify whether long-term changes in precipitation or "
        "seasonal patterns are shaping year-to-year temperature variability."
    ),
    "SPI Drought": (
        "The SPI-3 index measures how unusual the last 3 months of rainfall are "
        "compared to the historical average. Negative values indicate drier-than-normal "
        "conditions; the SHAP drivers show how drought and humidity features "
        "influence model temperature predictions during dry spells."
    ),
    "Heat & Frost": (
        "This chart counts the number of heatwave days (above threshold) and frost days "
        "(below 0 °C) per year. The SHAP values show which seasonal and lagged "
        "temperature features are most responsible for the model predicting extreme temperatures."
    ),
    "De Martonne Aridity": (
        "The De Martonne index combines annual rainfall and mean temperature into a single "
        "aridity score. Values below 20 indicate semi-arid conditions. The SHAP drivers "
        "here show how precipitation and long-term temperature trends interact to "
        "determine climate zone classification."
    ),
    "Extreme Events": (
        "This panel summarises detected heatwaves, droughts, and heavy rainfall events. "
        "The SHAP values highlight which seasonal and lagged temperature features make "
        "the model most confident when predicting temperatures near extreme thresholds."
    ),
    "Temperature Exceedances": (
        "This table lists days when the maximum temperature exceeded the heatwave threshold. "
        "The SHAP drivers explain which features (season, recent temperatures) contributed "
        "most to the model predicting those high values."
    ),
    "City Climate Map": (
        "This map shows mean temperature or annual precipitation across Albanian cities. "
        "The SHAP values indicate which geographic and seasonal features explain "
        "differences in predicted temperatures between cities and across the year."
    ),
    "Forecast Drivers": (
        "These are the top features driving the 12-month temperature forecast. "
        "A positive SHAP value (warmer direction) means the feature pushed the "
        "prediction higher; a negative value means it pulled the prediction lower."
    ),
    "Global Feature Importance": (
        "This bar chart ranks every model feature by its average absolute SHAP value,"
        "i.e. how much, on average, each feature moves the temperature prediction "
        "away from the baseline. Taller bars mean greater overall influence."
    ),
    "SHAP Distribution": (
        "Each dot represents one month in the dataset. The horizontal position shows "
        "the SHAP value (positive = pushes temperature prediction up; negative = down). "
        "Colour indicates whether the feature value was high or low that month."
    ),
    "Single Prediction": (
        "This waterfall chart explains a single monthly prediction step by step. "
        "Starting from the average predicted temperature, each bar shows how much "
        "a specific feature raised or lowered the final prediction for that month."
    ),
}

def _label(f: str) -> str:
    if f in FEATURE_LABELS:
        return FEATURE_LABELS[f]
    return f.replace("temperature_2m_mean_", "Temp").replace("_", " ").title()

def _group(f: str) -> str:
    if f in FEATURE_GROUPS:
        return FEATURE_GROUPS[f]
    if any(x in f for x in ["lag", "roll_mean", "roll_std", "yoy_diff"]):
        return "Temperature Persistence"
    return "Other Feature"

def shap_panel(title: str, ctx: dict, features=None):
    with st.expander(f"What drives this chart? (SHAP explanation)", expanded=True):
        context_text = PANEL_CONTEXT.get(
            title,
            "The SHAP values below show which features most influenced the model's "
            "temperature predictions for this view. Positive values push predictions "
            "warmer; negative values push them cooler."
        )
        st.caption(context_text)
        if not ctx.get("available"):
            st.info(f"Explainability unavailable: {ctx.get('reason', 'No reason given.')}")
            return
        imp       = ctx["importance"]
        direction = ctx["direction"]
        rows      = imp[imp["feature"].isin(features)].head(3) if features else imp.head(3)
        if rows.empty:
            rows = imp.head(3)
        st.markdown("**Top drivers for this chart:**")
        for _, row in rows.iterrows():
            f         = row["feature"]
            d         = direction.get(f, 0.0)
            effect    = "tends to raise temperature predictions" if d >= 0 else "tends to lower temperature predictions"
            shap_val  = f"{row['mean_abs_shap']:.2f}"
            group     = _group(f)
            label     = _label(f)
            st.markdown(
                f"**{label}** ({group})  \n"
                f"This feature *{effect}* on average.  \n"
                f"Average absolute influence: **{shap_val} °C**"
            )
            st.divider()
        st.caption(
            "Model: XGBoost trained on monthly climate data. Same-month max/min temperatures "
            "are excluded to prevent data leakage. Features include seasonality, long-term trend, "
            "lagged temperatures, precipitation, humidity, and wind speed."
        )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.title("Albania Climate Monitor")
sb.header("Location")
city = sb.selectbox("City", list(ALBANIA_CITIES.keys()), index=0)
sb.header("Time Period")
date_range = sb.date_input(
    "Date range",
    value=[pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")],
    min_value=pd.Timestamp("2020-01-01"),
    max_value=pd.Timestamp("2025-12-31"),
)
show_trend     = True
show_anomaly   = True
show_forecast  = False
baseline_start = 2020
baseline_end   = 2025
sb.header("Event Thresholds")
heat_thr   = sb.slider("Heatwave threshold (°C)", 25, 45, int(THRESHOLDS["heatwave_temp_c"]))
frost_thr  = sb.slider("Frost threshold (°C)", -10, 5, int(THRESHOLDS["frost_temp_c"]))
drought_sp = sb.slider("Drought SPI threshold", -3.0, -0.5, float(THRESHOLDS["drought_spi_threshold"]))

# ---------------------------------------------------------------------------
# Load data — single canonical df / dfm used throughout all tabs
# ---------------------------------------------------------------------------
df_raw = load_city_data(city)
start  = pd.Timestamp(date_range[0]) if len(date_range) > 0 else pd.Timestamp("2020-01-01")
end    = pd.Timestamp(date_range[1]) if len(date_range) > 1 else pd.Timestamp("2025-12-31")
df     = df_raw.loc[start:end]
dfm    = monthly_agg(df)
ctx    = build_shap_ctx(city, dfm)

# Convenience alias used in several tabs (monthly precipitation series)
mp = dfm["precipitation_sum"]

lon, lat = ALBANIA_CITIES[city]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"Albania Climate Monitor for {city}")
st.caption(
    f"Coordinates: {lat:.4f}°N, {lon:.4f}°E  ·  "
    f"Period: {start.strftime('%d %b %Y')} → {end.strftime('%d %b %Y')}  ·  "
    f"{(end-start).days:,} days"
)

# KPI row
mean_t  = df["temperature_2m_mean"].mean()
max_t   = df["temperature_2m_max"].max()
min_t   = df["temperature_2m_min"].min()
ann_p   = df["precipitation_sum"].sum() / max(1, (end-start).days/365)
h_days  = (df["temperature_2m_max"] > heat_thr).sum()
dry_d   = (df["precipitation_sum"] < 1.0).sum()
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Mean Temperature", f"{mean_t:.1f} °C")
k2.metric("Peak Temperature", f"{max_t:.1f} °C")
k3.metric("Min Temperature",  f"{min_t:.1f} °C")
k4.metric("Annual Precipitation", f"{ann_p:.0f} mm")
k5.metric("Heat Days", f"{h_days} d")
k6.metric("Dry Days",  f"{dry_d} d")
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Temperature", "Rainfall", "Indices",
    "Events", "Map", "Forecast", "XAI"
])

# ---------------------------------------------------------------------------
# TAB 1: Temperature
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Temperature Analysis")
    mt = dfm["temperature_2m_mean"]
    bl = mt.loc[f"{baseline_start}":f"{baseline_end}"]
    bl_monthly = bl.groupby(bl.index.month).mean()
    anom = mt.copy()
    for m in range(1, 13):
        mask = mt.index.month == m
        anom.loc[mask] = mt.loc[mask] - bl_monthly.get(m, mt.mean())
    fig = go.Figure()
    if show_anomaly:
        fig.add_trace(go.Bar(
            x=anom.index, y=anom.values,
            marker_color=["red" if v > 0 else "steelblue" for v in anom.values],
            opacity=0.50, name="Temperature Anomaly (°C)",
        ))
    fig.add_trace(go.Scatter(
        x=mt.index, y=mt.values, mode="lines",
        name="Monthly Mean Temperature", line=dict(color="orange", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=mt.index, y=mt.rolling(12, center=True).mean().values,
        mode="lines", name="12-month Moving Average", line=dict(color="white", width=2.5),
    ))
    if show_trend:
        from scipy import stats as sp
        xn = np.arange(len(mt))
        sl, ic, *_ = sp.linregress(xn, mt.fillna(0))
        fig.add_trace(go.Scatter(
            x=mt.index, y=sl*xn+ic, mode="lines",
            name=f"Linear Trend ({sl*120:+.3f} °C/decade)",
            line=dict(dash="dot", width=1.5),
        ))
    fig.update_layout(title=f"Monthly Mean Temperature for {city}", height=360,
                      xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)
    shap_panel("Monthly Temperature", ctx,
               ["month_sin", "month_cos", "temperature_2m_mean_lag_1",
                "temperature_2m_mean_roll_mean_12", "year", "trend_index"])
    st.divider()
    clim = df["temperature_2m_mean"].groupby(df.index.month).agg(["mean", "min", "max"])
    mlbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=mlbl, y=clim["max"].values, fill=None,
                              mode="lines", line_color="red", name="Monthly Maximum"))
    fig2.add_trace(go.Scatter(x=mlbl, y=clim["min"].values, fill="tonexty",
                              mode="lines", line_color="steelblue",
                              fillcolor="rgba(70,130,180,0.15)", name="Monthly Minimum"))
    fig2.add_trace(go.Scatter(x=mlbl, y=clim["mean"].values, mode="lines+markers",
                              line=dict(color="orange", width=2),
                              marker=dict(size=5, color="orange"), name="Monthly Mean"))
    fig2.update_layout(title="Average Temperature by Month (Climatology)", height=320,
                       xaxis_title="Month", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig2, use_container_width=True)
    shap_panel("Monthly Climatology", ctx,
               ["month_sin", "month_cos", "month", "season"])
    st.divider()
    st.subheader("Warming Stripes")
    ann_t  = df["temperature_2m_mean"].resample("YS").mean()
    anom_a = ann_t - ann_t.mean()
    vmax   = abs(anom_a).max()
    import matplotlib
    import matplotlib.colors as mcolors
    cmap_s = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["steelblue", "white", "red"], N=256)
    norm_s = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig_s, ax_s = plt.subplots(figsize=(12, 1.8), dpi=130)
    for i, (yr, val) in enumerate(anom_a.items()):
        ax_s.add_patch(plt.Rectangle((i, 0), 1, 1,
                                     facecolor=cmap_s(norm_s(val)), edgecolor="none"))
    yrs  = list(anom_a.index.year)
    tpos = [i for i, y in enumerate(yrs) if y % 5 == 0]
    ax_s.set_xticks([p + 0.5 for p in tpos])
    ax_s.set_xticklabels([str(yrs[i]) for i in tpos], fontsize=8)
    ax_s.set_yticks([])
    ax_s.set_xlim(0, len(anom_a))
    for sp_ in ax_s.spines.values():
        sp_.set_visible(False)
    st.pyplot(fig_s, use_container_width=True)
    plt.close(fig_s)
    st.caption(
        "Blue stripes = cooler-than-average years; red stripes = warmer-than-average years. "
        "Colour intensity reflects how far each year's mean temperature deviates from the long-run average."
    )
    shap_panel("Warming Trend", ctx,
               ["year", "trend_index", "temperature_2m_mean_roll_mean_12",
                "temperature_2m_mean_yoy_diff"])

# ---------------------------------------------------------------------------
# TAB 2: Precipitation
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Precipitation Analysis")
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        x=mp.index, y=mp.values,
        marker_color="steelblue", opacity=0.70, name="Monthly Total (mm)",
    ))
    fig_p.add_trace(go.Scatter(
        x=mp.index, y=mp.rolling(12, center=True).mean(),
        mode="lines", name="12-month Moving Average",
        line=dict(color="orange", width=2.5),
    ))
    fig_p.update_layout(title=f"Monthly Precipitation for {city}", height=360,
                        xaxis_title="Date", yaxis_title="Precipitation (mm)")
    st.plotly_chart(fig_p, use_container_width=True)
    shap_panel("Monthly Precipitation", ctx,
               ["precipitation_sum", "relative_humidity_2m_mean", "month_sin", "month_cos"])
    st.divider()
    ann_p_s = df["precipitation_sum"].resample("YS").sum()
    fig_a = px.bar(
        x=ann_p_s.index.year, y=ann_p_s.values,
        labels={"x": "Year", "y": "Annual Precipitation (mm)"},
        color=ann_p_s.values,
        color_continuous_scale="Blues",
        title=f"Annual Total Precipitation for {city}",
    )
    fig_a.update_layout(coloraxis_showscale=False, height=320)
    st.plotly_chart(fig_a, use_container_width=True)
    shap_panel("Annual Precipitation", ctx,
               ["precipitation_sum", "year", "trend_index", "month_sin", "month_cos"])

# ---------------------------------------------------------------------------
# TAB 3: Indices
# FIX: load SPI into a local variable (spi_df) — never overwrite the global df.
# FIX: use sidebar `city` throughout; no second city selectbox inside the tab.
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Climate Indices")

    spi_df = load_spi_data(city)

    # Compute SPI-3 series
    spi3 = None
    try:
        from src.analysis.climate_indices import ClimateIndices
        ci = ClimateIndices()
        # Prefer recomputing from raw precipitation if available in the daily df
        if "precipitation_sum" in df.columns:
            spi3 = ci.compute_spi(df["precipitation_sum"], timescale_months=3)
        elif not spi_df.empty and "spi3_mean" in spi_df.columns:
            spi3 = spi_df["spi3_mean"]
        else:
            raise ValueError("No suitable precipitation or SPI column found.")
    except Exception as e:
        st.warning(f"SPI fallback used: {e}")
        if not spi_df.empty and "spi3_mean" in spi_df.columns:
            spi3 = spi_df["spi3_mean"]
        elif "precipitation_sum" in df.columns:
            # Simple z-score fallback from monthly precipitation
            mp_fallback = dfm["precipitation_sum"]
            spi3 = (mp_fallback - mp_fallback.mean()) / mp_fallback.std()
        else:
            st.error("Cannot compute SPI-3: no precipitation data available.")
            spi3 = pd.Series(dtype=float)

    spi3.name = "SPI-3"

    # SPI figure
    if not spi3.empty:
        fig_spi = go.Figure()
        fig_spi.add_trace(
            go.Bar(
                x=spi3.index,
                y=spi3.values,
                marker_color=[
                    "red" if v < drought_sp
                    else ("steelblue" if v > 1.5 else "grey")
                    for v in spi3.values
                ],
                name=f"{city} SPI-3",
            )
        )
        fig_spi.add_hline(
            y=drought_sp, line_dash="dash", line_color="red", line_width=1,
            annotation_text="Severe drought threshold", annotation_font=dict(size=9),
        )
        fig_spi.add_hline(
            y=1.5, line_dash="dash", line_color="steelblue", line_width=1,
            annotation_text="Very wet threshold", annotation_font=dict(size=9),
        )
        fig_spi.add_hline(y=0, line_color="black", line_width=1)
        fig_spi.update_layout(
            title=f"{city} — SPI-3 Standardised Precipitation Index",
            height=350, xaxis_title="Date", yaxis_title="SPI-3", bargap=0.1,
        )
        st.plotly_chart(fig_spi, use_container_width=True)

        st.caption(
            "SPI-3 compares precipitation over the previous 3 months against historical "
            "climatology. Negative values indicate drought conditions; positive values "
            "indicate wetter-than-normal conditions."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean SPI-3",    f"{spi3.mean():.2f}")
        col2.metric("Minimum SPI-3", f"{spi3.min():.2f}")
        col3.metric("Maximum SPI-3", f"{spi3.max():.2f}")

        # Drought events — always built from spi3 directly to avoid cross-index mismatches
        st.subheader(f"{city} Severe Drought Events")
        drought_mask = spi3 < drought_sp
        if drought_mask.any():
            # Start from the filtered SPI series — this is always correct
            drought_display = spi3[drought_mask].rename("SPI-3").to_frame().sort_index()
            # Optionally join extra metadata columns (lon/lat) from spi_df using index alignment
            if not spi_df.empty:
                meta_cols = [c for c in ["city", "longitude", "latitude"] if c in spi_df.columns]
                if meta_cols:
                    drought_display = drought_display.join(spi_df[meta_cols], how="left")
            st.dataframe(drought_display, use_container_width=True)
        else:
            st.info("No severe drought events detected.")

    shap_panel("SPI Drought", ctx, ["precipitation_sum", "relative_humidity_2m_mean"])

# ---------------------------------------------------------------------------
# TAB 4: Extreme Events
# FIX: use the original `df` and the module-level `mp` alias — not the Tab 3 variable.
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Extreme Weather Events")
    try:
        from src.analysis.extreme_events import ExtremeEventDetector
        from src.analysis.climate_indices import ClimateIndices as CI2
        eed  = ExtremeEventDetector()
        ci2  = CI2()
        hwe  = eed.detect_heatwaves(df["temperature_2m_max"], df["temperature_2m_mean"],
                                    threshold_tmax=heat_thr, min_duration_days=3)
        dre  = eed.detect_droughts(ci2.compute_spi(mp, 3), threshold=drought_sp)
        pre  = eed.detect_extreme_precipitation(df["precipitation_sum"], threshold_mm=50)
        c1, c2, c3 = st.columns(3)
        c1.metric("Heatwave Events Detected", len(hwe))
        c2.metric("Drought Episodes Detected", len(dre))
        c3.metric("Extreme Rainfall Events", len(pre))
        if hwe:
            hw_df = eed.events_to_dataframe(hwe)
            st.markdown("**Heatwave Events**")
            st.dataframe(hw_df.sort_values("start", ascending=False), use_container_width=True)
        if pre:
            pe_df = eed.events_to_dataframe(pre)
            st.markdown("**Extreme Precipitation Events**")
            st.dataframe(pe_df.sort_values("max_intensity", ascending=False).head(20),
                         use_container_width=True)
        shap_panel("Extreme Events", ctx,
                   ["month_sin", "month_cos", "temperature_2m_mean_lag_1",
                    "precipitation_sum"])
    except Exception as e:
        st.warning(f"Full extreme event module unavailable ({e}). Showing basic exceedance table.")
        exc = df[df["temperature_2m_max"] > heat_thr][
            ["temperature_2m_max", "temperature_2m_mean"]].head(30)
        st.markdown("**Days where maximum temperature exceeded the heatwave threshold**")
        st.dataframe(exc, use_container_width=True)
        shap_panel("Temperature Exceedances", ctx)

# ---------------------------------------------------------------------------
# TAB 5: Map
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Albania City Climate Map")
    cdf     = city_map_data()
    map_var = st.radio(
        "Colour cities by:",
        ["mean_temp", "annual_precip"],
        horizontal=True,
        format_func=lambda x: "Mean Temperature (°C)" if x == "mean_temp" else "Annual Precipitation (mm)",
    )
    cscale = "RdYlBu_r" if map_var == "mean_temp" else "Blues"
    fig_m = px.scatter_mapbox(
        cdf, lat="lat", lon="lon",
        size=[14]*len(cdf), color=map_var, text="city",
        color_continuous_scale=cscale, zoom=6.3,
        center={"lat": ALBANIA_CENTER["lat"], "lon": ALBANIA_CENTER["lon"]},
        mapbox_style="carto-positron",
        hover_data={"city": True, "mean_temp": ":.1f", "annual_precip": ":.0f"},
        title="Mean Temperature and Annual Precipitation Across Albanian Cities",
    )
    fig_m.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0),
                        coloraxis_colorbar_title="°C" if map_var == "mean_temp" else "mm")
    st.plotly_chart(fig_m, use_container_width=True)
    shap_panel("City Climate Map", ctx,
               ["precipitation_sum", "month_sin", "month_cos", "year", "trend_index"])

# ---------------------------------------------------------------------------
# TAB 6: Forecast
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("12-Month Temperature Forecast")
    if show_forecast or st.button("Generate Forecast"):
        with st.spinner("Training forecast model: this may take a moment …"):
            try:
                from src.modeling.temperature_model import TemperatureModel
                from src.preprocessing.time_series_processor import TimeSeriesProcessor
                tsp  = TimeSeriesProcessor()
                mts  = dfm["temperature_2m_mean"].dropna()
                tm   = TemperatureModel(city=city, model_type="xgboost")
                tm.fit_prophet(mts)
                fc   = tm.forecast_prophet(periods=12)
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=mts.index, y=mts.values, mode="lines",
                    name="Historical Monthly Mean",
                    line=dict(color="orange", width=1.8)))
                fig_fc.add_trace(go.Scatter(
                    x=fc["date"], y=fc["forecast"], mode="lines+markers",
                    name="12-month Forecast",
                    line=dict(color="red", dash="dash", width=2),
                    marker=dict(size=5)))
                fig_fc.add_trace(go.Scatter(
                    x=pd.concat([fc["date"], fc["date"][::-1]]),
                    y=pd.concat([fc["upper_95"], fc["lower_95"][::-1]]),
                    fill="toself", fillcolor="rgba(255,0,0,0.08)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Confidence Interval"))
                fig_fc.update_layout(
                    title=f"Temperature Forecast for {city} : Next 12 Months",
                    height=400, xaxis_title="Date", yaxis_title="Temperature (°C)",
                )
                st.plotly_chart(fig_fc, use_container_width=True)
                st.markdown("**Forecast values (monthly)**")
                st.dataframe(fc.round(2), use_container_width=True)
                shap_panel("Forecast Drivers", ctx,
                           ["temperature_2m_mean_lag_1",
                            "temperature_2m_mean_roll_mean_12",
                            "month_sin", "month_cos", "trend_index"])
            except Exception as e:
                st.error(f"Forecast error: {e}")
                st.info("Ensure Prophet is installed: `pip install prophet`")
    else:
        st.info("Enable the forecast toggle in the sidebar, or click the button above to generate a 12-month outlook.")

# ---------------------------------------------------------------------------
# TAB 7: XAI / SHAP
# ---------------------------------------------------------------------------
with tab7:
    st.subheader("Model Explainability: SHAP Analysis")
    st.caption(
        "SHAP (SHapley Additive exPlanations) measures how much each input feature "
        "contributed to each individual temperature prediction. Positive values indicate "
        "the feature pushed the prediction warmer; negative values pushed it cooler."
    )
    if not ctx.get("available"):
        st.info(f"SHAP analysis unavailable: {ctx.get('reason', 'Unknown reason.')}")
    else:
        import shap
        sv     = ctx["shap_values"]
        imp    = ctx["importance"]
        X_shap = ctx["X"].iloc[:sv.values.shape[0]]
        st.markdown("### Overall Feature Importance")
        st.caption(
            "The bar chart below ranks features by their average absolute SHAP value, "
            "i.e. how much, on average, each feature moves the temperature prediction "
            "away from the model baseline. Longer bars = more influential features."
        )
        imp_display = imp.head(15).copy()
        imp_display["Feature (plain English)"] = imp_display["feature"].apply(_label)
        fig_i = px.bar(
            imp_display,
            x="mean_abs_shap",
            y="Feature (plain English)",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
            labels={"mean_abs_shap": "Average |SHAP| value (°C)"},
            title="Top Features Driving Temperature Predictions",
        )
        fig_i.update_layout(yaxis=dict(autorange="reversed"),
                            coloraxis_showscale=False, height=380)
        st.plotly_chart(fig_i, use_container_width=True)
        shap_panel("Global Feature Importance", ctx)
        st.divider()
        st.markdown("### SHAP Summary Plot: Distribution of Feature Impacts")
        st.caption(
            "Each dot represents one month in the dataset. The horizontal axis shows the SHAP value "
            "for that month: positive = the feature pushed the prediction warmer that month; negative = cooler. "
            "Dot colour shows whether the feature value was high (red) or low (blue) that month."
        )
        plt.figure(figsize=(7, 5))
        shap.summary_plot(sv, X_shap, show=False,
                          max_display=len(X_shap.columns), color_bar=True)
        plt.tight_layout()
        fig_sh = plt.gcf()
        st.pyplot(fig_sh, use_container_width=True)
        plt.close(fig_sh)
        shap_panel("SHAP Distribution", ctx,
                   ["month_sin", "month_cos", "temperature_2m_mean_lag_1", "year"])
        st.divider()
        st.markdown("### SHAP Dependence Plot: Single Feature Deep Dive")
        st.caption(
            "Select a feature to see how its values relate to its SHAP impact. "
            "The x-axis shows the raw feature value; the y-axis shows how much that "
            "value raised or lowered the temperature prediction."
        )
        feat_options = {_label(f): f for f in X_shap.columns}
        feat_display = st.selectbox("Select feature to examine", list(feat_options.keys()))
        feat         = feat_options[feat_display]
        fig_d, ax_d = plt.subplots(figsize=(7, 4))
        shap.dependence_plot(feat, sv.values, X_shap, ax=ax_d,
                             show=False, interaction_index=None)
        ax_d.set_xlabel(f"{feat_display} (feature value)")
        ax_d.set_ylabel("SHAP value: impact on temperature prediction (°C)")
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)
        st.divider()
        st.markdown("### Waterfall Chart: Explaining a Single Monthly Prediction")
        st.caption(
            "Select a specific month to see a step-by-step breakdown of its prediction. "
            "Starting from the baseline (average prediction), each bar shows how much "
            "a single feature raised (red, upward) or lowered (blue, downward) "
            "the final forecast for that month."
        )
        idx = st.slider("Select month index (0 = oldest month in dataset)", 0, len(sv)-1, 0)
        shap.plots.waterfall(sv[idx], max_display=len(X_shap.columns), show=False)
        fig_w = plt.gcf()
        fig_w.set_size_inches(7, 5)
        plt.tight_layout()
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)
        shap_panel("Single Prediction", ctx,
                   ["temperature_2m_mean_lag_1", "month_sin",
                    "temperature_2m_mean_roll_mean_12"])

# Footer
st.divider()
st.caption("Albania Geospatial Climate Analysis · © 2026")