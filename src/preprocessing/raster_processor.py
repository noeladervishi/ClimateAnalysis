import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
import geopandas as gpd
from rasterstats import zonal_stats
from pathlib import Path
from loguru import logger
import xarray as xr
import pandas as pd
from rasterio.windows import from_bounds

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ALBANIA_BBOX, CRS_WGS84, CRS_PROJECTED, PROCESSED_DIR
)


class RasterProcessor:
    def __init__(self):
        self.output_dir = PROCESSED_DIR / "rasters"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Clipping
    def clip_to_albania(
        self,
        raster_path: Path,
        albania_gdf: gpd.GeoDataFrame,
        output_name: str | None = None,
    ) -> Path:
        output_name = output_name or f"clipped_{raster_path.stem}.tif"
        output_path = self.output_dir / output_name

        with rasterio.open(raster_path) as src:
            albania_proj = albania_gdf.to_crs(src.crs)
            geoms = [geom.__geo_interface__ for geom in albania_proj.geometry]
            clipped, transform = rio_mask(src, geoms, crop=True, nodata=np.nan)
            profile = src.profile.copy()
            profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=transform,
                nodata=np.nan,
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(clipped)

        logger.success(f"Clipped to Albania: {output_path}")
        return output_path

    def clip_bbox(self, raster_path: Path, output_name: str | None = None) -> Path:
        output_name = output_name or f"bbox_{raster_path.stem}.tif"
        output_path = self.output_dir / output_name

        with rasterio.open(raster_path) as src:
            window = from_bounds(
                ALBANIA_BBOX["min_lon"],
                ALBANIA_BBOX["min_lat"],
                ALBANIA_BBOX["max_lon"],
                ALBANIA_BBOX["max_lat"],
                src.transform,
            )
            data = src.read(window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        logger.success(f"BBox clip saved: {output_path}")
        return output_path

    # Reprojection & Resampling
    def reproject_raster(
        self,
        raster_path: Path,
        target_crs: str = CRS_PROJECTED,
        resampling_method: Resampling = Resampling.bilinear,
        output_name: str | None = None,
    ) -> Path:
        output_name = output_name or f"reproj_{raster_path.stem}.tif"
        output_path = self.output_dir / output_name

        with rasterio.open(raster_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(crs=target_crs, transform=transform, width=width, height=height)

            with rasterio.open(output_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=resampling_method,
                    )

        logger.success(f"Reprojected to {target_crs}: {output_path}")
        return output_path

    def resample_raster(
        self,
        raster_path: Path,
        target_resolution_m: float,
        output_name: str | None = None,
    ) -> Path:
        output_name = output_name or f"resampled_{int(target_resolution_m)}m_{raster_path.stem}.tif"
        output_path = self.output_dir / output_name

        with rasterio.open(raster_path) as src:
            scale_x = src.res[0] / target_resolution_m
            scale_y = src.res[1] / target_resolution_m
            new_width = int(src.width * scale_x)
            new_height = int(src.height * scale_y)

            data = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.bilinear,
            )
            new_transform = src.transform * src.transform.scale(
                (src.width / new_width), (src.height / new_height)
            )
            profile = src.profile.copy()
            profile.update(width=new_width, height=new_height, transform=new_transform)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        logger.success(f"Resampled to {target_resolution_m}m: {output_path}")
        return output_path

    # Unit Conversions
    @staticmethod
    def kelvin_to_celsius(raster_path: Path, output_path: Path) -> Path:
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32) - 273.15
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

        logger.info(f"K→°C conversion: {output_path}")
        return output_path

    @staticmethod
    def era5_precip_to_mm(raster_path: Path, output_path: Path) -> Path:
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32) * 1000.0
            profile = src.profile.copy()
            profile.update(dtype=rasterio.float32)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

        logger.info(f"m→mm precipitation: {output_path}")
        return output_path

    # Zonal Statistics
    def zonal_stats_by_district(
        self,
        raster_path: Path,
        districts_gdf: gpd.GeoDataFrame,
        variable_name: str,
        stats: list[str] = ["mean", "min", "max", "std", "median"],
    ) -> gpd.GeoDataFrame:
        
        districts_proj = districts_gdf.to_crs(CRS_WGS84)

        result = zonal_stats(
            vectors=districts_proj,
            raster=str(raster_path),
            stats=stats,
            nodata=np.nan,
            geojson_out=False,
        )

        df = pd.DataFrame(result)
        df.columns = [f"{variable_name}_{c}" for c in df.columns]
        result_gdf = districts_gdf.copy()
        for col in df.columns:
            result_gdf[col] = df[col].values

        logger.info(f"Zonal stats for {len(result_gdf)} districts computed.")
        return result_gdf

    # xarray / NetCDF Utilities
    def xarray_to_geotiff(self, da: xr.DataArray, output_path: Path, crs: str = CRS_WGS84):
        if "time" in da.dims:
            da = da.isel(time=0)

        lat = da.coords.get("latitude", da.coords.get("lat")).values
        lon = da.coords.get("longitude", da.coords.get("lon")).values

        transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(),
                                len(lon), len(lat))

        data = da.values.astype(np.float32)
        if lat[0] < lat[-1]:          # south-up → flip to north-up
            data = np.flipud(data)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "crs": crs,
            "transform": transform,
            "nodata": np.nan,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

        logger.success(f"Exported to GeoTIFF: {output_path}")
        return output_path

    def merge_rasters(self, raster_paths: list[Path], output_path: Path) -> Path:
        datasets = [rasterio.open(p) for p in raster_paths]
        merged, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            height=merged.shape[1],
            width=merged.shape[2],
            transform=transform,
        )
        for ds in datasets:
            ds.close()

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(merged)

        logger.success(f"Merged {len(raster_paths)} rasters → {output_path}")
        return output_path
