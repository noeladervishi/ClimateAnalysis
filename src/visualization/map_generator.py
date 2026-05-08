import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from loguru import logger
import folium
from folium import plugins
import contextily as ctx
import rasterio
from rasterio.plot import show as rasterio_show

HAS_FOLIUM = True
HAS_CTX = True

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ALBANIA_CENTER, ALBANIA_CITIES, VIZ_CONFIG, MAPS_DIR,
    CRS_WGS84, CRS_PROJECTED, ALBANIA_BBOX
)


class AlbaniaMapGenerator:
    def __init__(self):
        self.output_dir = MAPS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.center = [ALBANIA_CENTER["lat"], ALBANIA_CENTER["lon"]]

    # Interactive Maps (Folium)
    def base_map(
        self,
        zoom_start: int = 7,
        tiles: str = "OpenStreetMap",
    ) -> "folium.Map":
        if not HAS_FOLIUM:
            raise ImportError("folium not installed. Run: pip install folium")

        m = folium.Map(
            location=self.center,
            zoom_start=zoom_start,
            tiles=tiles,
            control_scale=True,
        )
        # Add fullscreen button
        try:
            plugins.Fullscreen().add_to(m)
            plugins.MiniMap().add_to(m)
        except Exception:
            pass

        return m

    def choropleth_map(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        name_col: str = "name",
        title: str = "Albania Climate Map",
        colormap: str = "RdYlBu_r",
        output_name: str | None = None,
    ) -> "folium.Map":
        
        m = self.base_map()

        # Reproject to WGS84 for Folium
        gdf_wgs = gdf.to_crs(CRS_WGS84)

        # Colour scale
        vmin = gdf_wgs[value_col].min()
        vmax = gdf_wgs[value_col].max()
        cmap = plt.get_cmap(colormap)

        def style_function(feature):
            val = feature["properties"].get(value_col, 0)
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            r, g, b, _ = cmap(norm)
            return {
                "fillColor": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}",
                "fillOpacity": 0.7,
                "color": "#333",
                "weight": 1,
            }

        tooltip_fields = [name_col, value_col]
        tooltip_aliases = ["District:", f"{value_col}:"]

        folium.GeoJson(
            gdf_wgs.__geo_interface__,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
            ),
            name=title,
        ).add_to(m)

        # Add city markers
        self._add_city_markers(m)

        # Add title
        title_html = f"""
        <div style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                    background:rgba(255,255,255,0.9); padding:10px 20px;
                    border-radius:8px; font-size:16px; font-weight:bold;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3); z-index:9999;">
            🇦🇱 {title}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        if output_name:
            path = self.output_dir / f"{output_name}.html"
            m.save(str(path))
            logger.success(f"Choropleth map saved: {path}")

        return m

    def temperature_heatmap(
        self,
        stations_df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        value_col: str = "temperature_c",
        output_name: str = "albania_temp_heatmap",
    ) -> "folium.Map":
        m = self.base_map(zoom_start=7)

        points = [
            [row[lat_col], row[lon_col], max(0, row[value_col])]
            for _, row in stations_df.dropna(subset=[lat_col, lon_col, value_col]).iterrows()
        ]

        plugins.HeatMap(
            points,
            radius=25,
            blur=20,
            gradient={0.2: "blue", 0.5: "yellow", 0.8: "orange", 1.0: "red"},
            min_opacity=0.4,
        ).add_to(m)

        self._add_city_markers(m)

        path = self.output_dir / f"{output_name}.html"
        m.save(str(path))
        logger.success(f"Temperature heatmap saved: {path}")
        return m

    def extreme_events_map(
        self,
        events_df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        output_name: str = "extreme_events",
    ) -> "folium.Map":
        
        m = self.base_map()

        COLOR_MAP = {
            "Heatwave":              "#FF4500",
            "Drought":               "#8B4513",
            "Extreme Precipitation": "#0000CD",
            "Cold Spell":            "#00CED1",
            "Heavy Snowfall":        "#9370DB",
            "Wildfire":              "#FF6347",
        }

        for _, row in events_df.iterrows():
            event_type = row.get("type", "Unknown")
            color = COLOR_MAP.get(event_type, "#666666")
            lat = row.get(lat_col, ALBANIA_CENTER["lat"])
            lon = row.get(lon_col, ALBANIA_CENTER["lon"])

            popup_html = f"""
            <b>{event_type}</b><br>
             {row.get('start', '')} → {row.get('end', '')}<br>
            Duration: {row.get('duration_days', '?')} days<br>
            Severity: {row.get('severity', '?')}<br>
            Max intensity: {row.get('max_intensity', '?'):.1f}
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, row.get("duration_days", 5) / 2),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

        # Add legend
        legend_html = self._build_legend_html(COLOR_MAP, "Extreme Event Types")
        m.get_root().html.add_child(folium.Element(legend_html))

        path = self.output_dir / f"{output_name}.html"
        m.save(str(path))
        logger.success(f"Extreme events map saved: {path}")
        return m

    # Static Maps (Matplotlib)
    def static_climate_map(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        title: str,
        colormap: str = "RdYlBu_r",
        output_name: str | None = None,
        add_basemap: bool = False,
        units: str = "",
    ) -> plt.Figure:

        fig, ax = plt.subplots(
            figsize=VIZ_CONFIG["figure_size"],
            dpi=VIZ_CONFIG["figure_dpi"],
        )

        gdf_proj = gdf.to_crs(CRS_PROJECTED)
        gdf_proj.plot(
            column=value_col,
            cmap=colormap,
            legend=True,
            legend_kwds={
                "label": f"{value_col} {units}",
                "orientation": "vertical",
                "shrink": 0.5,
            },
            ax=ax,
            edgecolor="#333",
            linewidth=0.5,
            missing_kwds={"color": "lightgrey"},
        )

        if add_basemap and HAS_CTX:
            ctx.add_basemap(ax, crs=gdf_proj.crs.to_string(), source=ctx.providers.CartoDB.Positron)

        # Add city markers
        for city_name, (lon, lat) in ALBANIA_CITIES.items():
            gdf_point = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy([lon], [lat]), crs=CRS_WGS84
            ).to_crs(CRS_PROJECTED)
            x, y = gdf_point.geometry.iloc[0].x, gdf_point.geometry.iloc[0].y
            ax.scatter(x, y, c="black", s=25, zorder=5)
            ax.annotate(
                city_name, (x, y),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7, color="#222",
            )

        # ── Style
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)

        # North arrow
        ax.annotate("N", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=14, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color="black"))
        ax.annotate("", xy=(0.05, 0.97), xytext=(0.05, 0.93),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

        plt.tight_layout()

        if output_name:
            path = self.output_dir / f"{output_name}.png"
            fig.savefig(path, dpi=VIZ_CONFIG["figure_dpi"], bbox_inches="tight")
            logger.success(f"Static map saved: {path}")

        return fig

    def spatial_anomaly_map(
        self,
        gdf: gpd.GeoDataFrame,
        value_col: str,
        baseline_mean: float | None = None,
        title: str = "Temperature Anomaly",
        output_name: str | None = None,
    ) -> plt.Figure:
       
        gdf = gdf.copy()
        if baseline_mean is not None:
            gdf["anomaly"] = gdf[value_col] - baseline_mean
        else:
            gdf["anomaly"] = gdf[value_col] - gdf[value_col].mean()

        vabs = max(abs(gdf["anomaly"].min()), abs(gdf["anomaly"].max()))
        return self.static_climate_map(
            gdf, "anomaly", title,
            colormap=VIZ_CONFIG["colormap_anomaly"],
            output_name=output_name,
            units="(anomaly)",
        )

    def raster_map(
        self,
        raster_path: Path,
        title: str,
        colormap: str = "RdYlBu_r",
        output_name: str | None = None,
        albania_boundary: gpd.GeoDataFrame | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=VIZ_CONFIG["figure_size"], dpi=VIZ_CONFIG["figure_dpi"])

        with rasterio.open(raster_path) as src:
            data = src.read(1)
            data = np.where(data == src.nodata, np.nan, data)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

            im = ax.imshow(data, extent=extent, cmap=colormap, origin="upper")
            plt.colorbar(im, ax=ax, shrink=0.5, label=title)

        if albania_boundary is not None:
            albania_boundary.boundary.plot(ax=ax, color="black", linewidth=1.0)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        if output_name:
            path = self.output_dir / f"{output_name}.png"
            fig.savefig(path, bbox_inches="tight")
            logger.success(f"Raster map saved: {path}")

        return fig

    # Helpers
    def _add_city_markers(self, m: "folium.Map"):
        for city_name, (lon, lat) in ALBANIA_CITIES.items():
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#333",
                fill=True,
                fill_color="#FFF",
                fill_opacity=0.9,
                popup=city_name,
                tooltip=city_name,
            ).add_to(m)

    def _build_legend_html(self, color_map: dict, title: str) -> str:
        items = "".join(
            f'<div style="display:flex; align-items:center; margin:4px 0;">'
            f'<div style="width:16px;height:16px;background:{color};margin-right:8px;'
            f'border-radius:3px;"></div>{label}</div>'
            for label, color in color_map.items()
        )
        return f"""
        <div style="position:fixed; bottom:40px; right:20px; background:rgba(255,255,255,0.95);
                    padding:12px; border-radius:8px; box-shadow:2px 2px 8px rgba(0,0,0,0.3);
                    z-index:9999; font-size:12px;">
            <b>{title}</b><br>{items}
        </div>
        """
