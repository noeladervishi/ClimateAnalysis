import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.geometry import box as shp_box
from pathlib import Path
from loguru import logger
import requests
from config.settings import CRS_WGS84, CRS_PROJECTED, ALBANIA_CITIES
import rasterio
from rasterio.sample import sample_gen

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ALBANIA_BBOX, CRS_WGS84, CRS_PROJECTED,
    SHAPEFILE_DIR, PROCESSED_DIR
)


class VectorProcessor:
    def __init__(self):
        self.output_dir = PROCESSED_DIR / "vectors"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Loading
    def load_albania_boundary(self) -> gpd.GeoDataFrame:
        local = SHAPEFILE_DIR / "albania_boundary.shp"
        if local.exists():
            gdf = gpd.read_file(local)
            logger.info(f"Loaded Albania boundary: {local}")
            return gdf.to_crs(CRS_WGS84)

        # Try GADM level-0 via URL
        try:
            url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ALB_0.json"
            gdf = gpd.read_file(url)
            gdf = gdf.to_crs(CRS_WGS84)
            gdf.to_file(local)
            logger.success("Albania boundary downloaded from GADM.")
            return gdf
        except Exception as e:
            logger.warning(f"Could not download boundary: {e}")
            # Fallback: bounding box polygon
            geom = shp_box(
                ALBANIA_BBOX["min_lon"], ALBANIA_BBOX["min_lat"],
                ALBANIA_BBOX["max_lon"], ALBANIA_BBOX["max_lat"],
            )
            gdf = gpd.GeoDataFrame([{"name": "Albania", "geometry": geom}], crs=CRS_WGS84)
            logger.warning("Using bounding box as fallback boundary.")
            return gdf

    def load_albania_districts(self, level: int = 2) -> gpd.GeoDataFrame:
        local = SHAPEFILE_DIR / f"albania_admin{level}.shp"
        if local.exists():
            gdf = gpd.read_file(local)
            return gdf.to_crs(CRS_WGS84)

        try:
            url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ALB_{level}.json"
            gdf = gpd.read_file(url)
            gdf = gdf.to_crs(CRS_WGS84)
            gdf.to_file(local)
            logger.success(f"Albania admin level {level} downloaded from GADM.")
            return gdf
        except Exception as e:
            logger.error(f"Could not load districts: {e}")
            return gpd.GeoDataFrame()

    def load_shapefile(self, path: Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_WGS84)
        else:
            gdf = gdf.to_crs(CRS_WGS84)
        logger.info(f"Loaded shapefile: {path} | {len(gdf)} features | CRS: {gdf.crs}")
        return gdf

    def cities_to_geodataframe(self, cities_dict: dict | None = None) -> gpd.GeoDataFrame:
        cities_dict = cities_dict or ALBANIA_CITIES

        records = [
            {"city": name, "longitude": lon, "latitude": lat,
             "geometry": Point(lon, lat)}
            for name, (lon, lat) in cities_dict.items()
        ]
        gdf = gpd.GeoDataFrame(records, crs=CRS_WGS84)
        return gdf

    # Cleaning & Validation
    def fix_invalid_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            logger.warning(f"Fixing {invalid.sum()} invalid geometries …")
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
        return gdf

    def remove_slivers(
        self, gdf: gpd.GeoDataFrame, min_area_m2: float = 1000
    ) -> gpd.GeoDataFrame:
        gdf_proj = gdf.to_crs(CRS_PROJECTED)
        mask = gdf_proj.geometry.area >= min_area_m2
        removed = (~mask).sum()
        if removed:
            logger.info(f"Removed {removed} sliver polygons (< {min_area_m2} m²)")
        return gdf[mask].to_crs(CRS_WGS84)

    def clip_to_albania(
        self, gdf: gpd.GeoDataFrame, albania_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        albania_union = unary_union(albania_gdf.to_crs(gdf.crs).geometry)
        clipped = gdf.clip(albania_union)
        logger.info(f"Clipped to Albania: {len(clipped)} features (was {len(gdf)})")
        return clipped

    def explode_multipolygons(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return gdf.explode(index_parts=False).reset_index(drop=True)

    # Spatial Operations
    def compute_area_km2(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf_proj = gdf.to_crs(CRS_PROJECTED)
        gdf["area_km2"] = gdf_proj.geometry.area / 1e6
        return gdf

    def compute_centroid(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf_proj = gdf.to_crs(CRS_PROJECTED)
        centroids = gdf_proj.geometry.centroid.to_crs(CRS_WGS84)
        gdf["centroid_lon"] = centroids.x
        gdf["centroid_lat"] = centroids.y
        return gdf

    def spatial_join_climate(
        self,
        districts_gdf: gpd.GeoDataFrame,
        climate_df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        value_cols: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
       
        # Create point GeoDataFrame
        geometry = gpd.points_from_xy(climate_df[lon_col], climate_df[lat_col])
        points_gdf = gpd.GeoDataFrame(climate_df, geometry=geometry, crs=CRS_WGS84)

        joined = gpd.sjoin(
            points_gdf, districts_gdf.to_crs(CRS_WGS84),
            how="inner", predicate="within",
        )

        value_cols = value_cols or [c for c in climate_df.columns
                                     if c not in [lat_col, lon_col]]
        agg = joined.groupby("index_right")[value_cols].mean()
        result = districts_gdf.copy()
        for col in value_cols:
            result[col] = agg[col].reindex(result.index)

        logger.info(f"Spatial join complete. {result[value_cols[0]].notna().sum()} districts have data.")
        return result

    def buffer_around_cities(
        self, cities_gdf: gpd.GeoDataFrame, buffer_km: float = 50
    ) -> gpd.GeoDataFrame:
        cities_proj = cities_gdf.to_crs(CRS_PROJECTED)
        cities_proj["geometry"] = cities_proj.geometry.buffer(buffer_km * 1000)
        return cities_proj.to_crs(CRS_WGS84)

    def elevation_from_dem(
        self,
        points_gdf: gpd.GeoDataFrame,
        dem_path: Path,
    ) -> gpd.GeoDataFrame:
    
        points_wgs = points_gdf.to_crs(CRS_WGS84)
        coords = [(geom.x, geom.y) for geom in points_wgs.geometry]

        with rasterio.open(dem_path) as dem:
            elevations = [v[0] for v in dem.sample(coords)]

        points_gdf = points_gdf.copy()
        points_gdf["elevation_m"] = elevations
        logger.info(f"Elevation sampled for {len(points_gdf)} points.")
        return points_gdf

    # Export
    def save(
        self,
        gdf: gpd.GeoDataFrame,
        name: str,
        format: str = "gpkg",
    ) -> Path:
        ext_map = {"gpkg": ".gpkg", "shp": ".shp", "geojson": ".geojson"}
        ext = ext_map.get(format, ".gpkg")
        path = self.output_dir / f"{name}{ext}"
        gdf.to_file(path, driver=format.upper() if format != "gpkg" else "GPKG")
        logger.success(f"Saved vector dataset: {path}")
        return path

    def to_geojson(self, gdf: gpd.GeoDataFrame, name: str) -> Path:
        path = self.output_dir / f"{name}.geojson"
        gdf.to_crs(CRS_WGS84).to_file(path, driver="GeoJSON")
        logger.success(f"GeoJSON exported: {path}")
        return path