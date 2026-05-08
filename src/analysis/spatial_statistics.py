import sys
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.transform import from_bounds
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from pathlib import Path
from loguru import logger
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
from shapely.geometry import box as shapely_box
from config.settings import ALBANIA_BBOX, CRS_PROJECTED, CRS_WGS84, PROCESSED_DIR


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

class SpatialClimateAnalyser:
    def __init__(self):
        self.output_dir = PROCESSED_DIR / "spatial"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Spatial Interpolation (Kriging / RBF)
    def interpolate_to_grid(
        self,
        stations_df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        value_col: str,
        grid_resolution: float = 0.05,
        method: str = "rbf",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        df = stations_df.dropna(subset=[lat_col, lon_col, value_col])

        # Source points
        lats = df[lat_col].values
        lons = df[lon_col].values
        values = df[value_col].values

        # Target grid
        lon_range = np.arange(ALBANIA_BBOX["min_lon"], ALBANIA_BBOX["max_lon"], grid_resolution)
        lat_range = np.arange(ALBANIA_BBOX["min_lat"], ALBANIA_BBOX["max_lat"], grid_resolution)
        grid_lons, grid_lats = np.meshgrid(lon_range, lat_range)
        grid_points = np.column_stack([grid_lons.ravel(), grid_lats.ravel()])
        source_points = np.column_stack([lons, lats])

        if method == "rbf":
            interpolator = RBFInterpolator(source_points, values, kernel="thin_plate_spline")
            grid_values = interpolator(grid_points).reshape(grid_lons.shape)

        elif method == "idw":
            grid_values = self._idw_interpolate(source_points, values, grid_points)
            grid_values = grid_values.reshape(grid_lons.shape)

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        logger.info(
            f"Interpolated {value_col} to {grid_lons.shape} grid "
            f"({grid_resolution}° resolution, method={method})"
        )
        return grid_lons, grid_lats, grid_values

    def _idw_interpolate(
        self,
        source_xy: np.ndarray,
        values: np.ndarray,
        target_xy: np.ndarray,
        power: float = 2.0,
    ) -> np.ndarray:
        distances = cdist(target_xy, source_xy)
        distances = np.where(distances == 0, 1e-10, distances)
        weights = 1.0 / (distances ** power)
        weights_norm = weights / weights.sum(axis=1, keepdims=True)
        return (weights_norm * values).sum(axis=1)

    def save_interpolated_geotiff(
        self,
        grid_lons: np.ndarray,
        grid_lats: np.ndarray,
        grid_values: np.ndarray,
        output_name: str,
        crs: str = CRS_WGS84,
    ) -> Path:
        output_path = self.output_dir / f"{output_name}.tif"
        transform = from_bounds(
            grid_lons.min(), grid_lats.min(),
            grid_lons.max(), grid_lats.max(),
            grid_lons.shape[1], grid_lons.shape[0],
        )
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": grid_lons.shape[1],
            "height": grid_lons.shape[0],
            "count": 1,
            "crs": crs,
            "transform": transform,
            "nodata": np.nan,
            "compress": "lzw",
        }
        data = np.flipud(grid_values.astype(np.float32))
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

        logger.success(f"Interpolated GeoTIFF saved: {output_path}")
        return output_path

    # Spatial Autocorrelation (Moran's I)
    def morans_i(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        k_neighbours: int = 5,
    ) -> dict:
        
        gdf_proj = gdf.to_crs(CRS_PROJECTED).dropna(subset=[value_col])
        w = KNN.from_dataframe(gdf_proj, k=k_neighbours)
        w.transform = "R"  # Row standardise

        y = gdf_proj[value_col].values
        moran = Moran(y, w)

        result = {
            "I":          round(moran.I, 4),
            "expected_I": round(moran.EI, 4),
            "variance":   round(moran.VI_norm, 6),
            "p_value":    round(moran.p_norm, 4),
            "z_score":    round(moran.z_norm, 4),
            "significant": moran.p_norm < 0.05,
        }
        logger.info(
            f"Moran's I for '{value_col}': I={result['I']}, "
            f"p={result['p_value']} ({'clustered' if result['I'] > result['expected_I'] else 'dispersed'})"
        )
        return result

    def local_morans_i(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        k_neighbours: int = 5,
    ) -> gpd.GeoDataFrame:
       
        gdf_proj = gdf.to_crs(CRS_PROJECTED).copy()
        valid = gdf_proj.dropna(subset=[value_col])
        w = KNN.from_dataframe(valid, k=k_neighbours)
        w.transform = "R"

        y = valid[value_col].values
        local_moran = Moran_Local(y, w)

        valid["lisa_I"] = local_moran.Is
        valid["lisa_p"] = local_moran.p_sim
        valid["lisa_cluster"] = np.where(
            local_moran.p_sim < 0.05,
            np.where(
                local_moran.q == 1, "HH",
                np.where(local_moran.q == 3, "LL",
                np.where(local_moran.q == 2, "LH", "HL"))
            ),
            "Not significant"
        )

        logger.info(f"LISA clusters: {valid['lisa_cluster'].value_counts().to_dict()}")
        return valid

    # Climate Zone Classification
    def classify_climate_zones(
        self,
        stations_df: pd.DataFrame,
        feature_cols: list[str],
        n_zones: int = 4,
        random_state: int = 42,
    ) -> pd.DataFrame:

        df = stations_df.copy().dropna(subset=feature_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])

        kmeans = KMeans(n_clusters=n_zones, random_state=random_state, n_init=15)
        df["climate_zone"] = kmeans.fit_predict(X_scaled)

        # Label zones by mean temperature (ascending = alpine→coastal)
        zone_temps = df.groupby("climate_zone")[feature_cols[0]].mean().sort_values()
        zone_labels = {
            zone: label for zone, label in zip(
                zone_temps.index,
                ["Alpine", "Continental Highland", "Transitional", "Coastal Mediterranean"][:n_zones]
            )
        }
        df["climate_zone_name"] = df["climate_zone"].map(zone_labels)

        logger.info(f"Climate zones classified:\n{df['climate_zone_name'].value_counts()}")
        return df

    def koppen_classify(
        self,
        annual_temp_c: float,
        annual_precip_mm: float,
        driest_month_precip_mm: float,
        wettest_month_precip_mm: float,
        coldest_month_temp_c: float,
        hottest_month_temp_c: float,
    ) -> str:
        
        # Check for Mediterranean (Cs): dry summer
        summer_months_dry = driest_month_precip_mm < 40
        summer_precip_ratio = driest_month_precip_mm < wettest_month_precip_mm / 3

        if coldest_month_temp_c > -3 and hottest_month_temp_c >= 10:
            if summer_months_dry and summer_precip_ratio:
                return "Csa" if hottest_month_temp_c >= 22 else "Csb"
            return "Cfb"  # Oceanic
        elif coldest_month_temp_c <= -3:
            if annual_precip_mm > 300:
                return "Dfb" if hottest_month_temp_c < 22 else "Dfa"
            return "Dsa"
        elif annual_precip_mm < 500:
            return "BSk"  # Semi-arid steppe
        return "C??"  # Unclassified

    # Elevation-Temperature Lapse Rate
    def elevation_lapse_rate(
        self,
        stations_df: pd.DataFrame,
        temp_col: str,
        elevation_col: str,
    ) -> dict:
        
        df = stations_df.dropna(subset=[temp_col, elevation_col])
        elev = df[elevation_col].values
        temp = df[temp_col].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(elev, temp)

        result = {
            "lapse_rate_C_per_100m": round(slope * 100, 3),
            "lapse_rate_C_per_1000m": round(slope * 1000, 3),
            "r_squared": round(r_value**2, 3),
            "p_value": round(p_value, 4),
            "intercept_at_sea_level": round(intercept, 2),
        }

        logger.info(
            f"Albania lapse rate: {result['lapse_rate_C_per_1000m']:.2f}°C/1000m "
            f"(standard = -6.5°C/1000m) | R²={result['r_squared']:.3f}"
        )
        return result

    # Point-in-Polygon Queries
    def assign_districts(
        self,
        points_gdf: gpd.GeoDataFrame,
        districts_gdf: gpd.GeoDataFrame,
        district_col: str = "name",
    ) -> gpd.GeoDataFrame:
        
        points_proj = points_gdf.to_crs(CRS_PROJECTED)
        districts_proj = districts_gdf.to_crs(CRS_PROJECTED)[[district_col, "geometry"]]

        joined = gpd.sjoin(
            points_proj, districts_proj,
            how="left", predicate="within"
        )
        joined = joined.drop(columns=["index_right"], errors="ignore")
        joined = joined.rename(columns={district_col: "district"})
        logger.info(f"Assigned {len(joined)} points to districts.")
        return joined

    def create_albania_grid_gdf(
        self, resolution_deg: float = 0.1
    ) -> gpd.GeoDataFrame:
        
        lons = np.arange(ALBANIA_BBOX["min_lon"], ALBANIA_BBOX["max_lon"], resolution_deg)
        lats = np.arange(ALBANIA_BBOX["min_lat"], ALBANIA_BBOX["max_lat"], resolution_deg)

        cells = []
        for lon in lons:
            for lat in lats:
                cells.append({
                    "geometry": shapely_box(lon, lat, lon + resolution_deg, lat + resolution_deg),
                    "center_lon": lon + resolution_deg / 2,
                    "center_lat": lat + resolution_deg / 2,
                })

        gdf = gpd.GeoDataFrame(cells, crs=CRS_WGS84)
        logger.info(f"Created {len(gdf)}-cell grid over Albania ({resolution_deg}° resolution)")
        return gdf

    # Semivariogram
    def empirical_semivariogram(
        self,
        stations_df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        value_col: str,
        n_lags: int = 15,
        max_lag_km: float = 300,
    ) -> pd.DataFrame:
        
        df = stations_df.dropna(subset=[lat_col, lon_col, value_col])
        coords = df[[lon_col, lat_col]].values
        values = df[value_col].values
        n = len(df)

        # Compute pairwise distances (approximate km using haversine-like scaling)
        dists = cdist(
            np.radians(coords), np.radians(coords),
            lambda u, v: 6371 * np.sqrt(
                ((u[1] - v[1]) * np.cos((u[0] + v[0]) / 2))**2 + (u[0] - v[0])**2
            )
        )

        # Pairwise squared differences
        sq_diff = (values[:, None] - values[None, :]) ** 2

        lag_edges = np.linspace(0, max_lag_km, n_lags + 1)
        records = []
        for i in range(len(lag_edges) - 1):
            mask = (dists > lag_edges[i]) & (dists <= lag_edges[i + 1]) & (dists > 0)
            if mask.sum() > 0:
                records.append({
                    "lag_km": (lag_edges[i] + lag_edges[i + 1]) / 2,
                    "semivariance": 0.5 * sq_diff[mask].mean(),
                    "n_pairs": mask.sum() // 2,
                })

        svario = pd.DataFrame(records)
        logger.info(f"Empirical semivariogram computed ({len(svario)} lag classes)")
        return svario
