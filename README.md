# Albania Geospatial Climate Analysis & Monitoring

A Python application for climate data analysis, trends, extreme events, and forecasting focused on Albania.


## Overview

End-to-end pipeline for:

* Data collection
* Preprocessing & validation
* Climate indices & trend analysis
* Extreme event detection
* Machine learning forecasts
* Visualization & dashboard


## Key Features

* Multi-source climate data ingestion
* Cleaning, imputation, feature engineering
* Climate indices (SPI, GDD, heat/frost days)
* Trend analysis (Mann-Kendall, Sen’s slope)
* Extreme events (heatwaves, droughts, floods)
* ML models (XGBoost, Prophet, SARIMA)
* Maps + Streamlit dashboard


## Installation

```bash
git clone <repo link>
cd climateanalysis

py -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

> *Add API keys in `.env` for ERA5 (CDS), NOAA CDO (optional, direct NOAA fallback exists), and NASA.*

---

## Usage

Run full pipeline:

```bash
py main.py --all-cities
```

Dashboard:

```bash
py -m streamlit run src/visualization/dashboard.py
```


## Pipeline

1. Download data
2. Validate & preprocess
3. Compute indices
4. Analyze trends
5. Detect extreme events
6. Train models
7. Generate outputs & dashboard


## Outputs

* Climate maps
* Time series plots
* Extreme event reports
* Model forecasts


## License

MIT