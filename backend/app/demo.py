"""生成无需外部数据的 GIS 演示数据。"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point


def seed_demo(application) -> dict[str, str]:
    root = application.workspace.input_dir
    roads_path = root / "roads.geojson"
    population_path = root / "population.geojson"
    dem_path = root / "dem.tif"
    roads = gpd.GeoDataFrame({"road_id": ["r1", "r2", "r3"], "class": ["primary", "secondary", "local"]}, geometry=[LineString([(114.00, 22.50), (114.08, 22.54)]), LineString([(114.02, 22.52), (114.10, 22.52)]), LineString([(114.01, 22.56), (114.08, 22.58)])], crs="EPSG:4326")
    population = gpd.GeoDataFrame({"zone": ["a", "b", "c", "d"], "POP2025": [1200, 2500, 800, 3100]}, geometry=[Point(114.02, 22.52), Point(114.05, 22.54), Point(114.08, 22.56), Point(114.06, 22.50)], crs="EPSG:4326")
    roads.to_file(roads_path, driver="GeoJSON")
    population.to_file(population_path, driver="GeoJSON")
    data = np.add.outer(np.linspace(20, 80, 80), np.linspace(0, 40, 80)).astype("float32")
    transform = from_origin(113.98, 22.62, 0.002, 0.002)
    with rasterio.open(dem_path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0) as destination:
        destination.write(data, 1)
    result = {}
    for name, path in (("roads", roads_path), ("population", population_path), ("dem", dem_path)):
        result[name] = application.register_dataset(path, name=name).id
    return result

