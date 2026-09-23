"""无浏览器依赖的结果地图渲染器。

输出一个包含数据摘要和 GeoJSON 的 HTML，前端可以直接下载或在自己的地图组件
中读取。它不是 GUI 自动化，也不依赖第三方地图服务。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import geopandas as gpd

from app.core.models import Dataset
from app.gis.errors import GISFailure


class MapRenderer:
    def render(self, dataset: Dataset, output_path: str | Path, *, title: str | None = None) -> Path:
        try:
            frame = gpd.read_file(dataset.path)
            geojson = frame.to_json()
        except Exception as exc:
            raise GISFailure("MAP_RENDER_FAILED", f"地图结果渲染失败：{exc}") from exc
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        label = html.escape(title or dataset.name)
        dataset_name = html.escape(dataset.name)
        safe_geojson = (
            json.dumps(json.loads(geojson), ensure_ascii=False, separators=(",", ":"))
            .replace("<", r"\u003c")
            .replace(">", r"\u003e")
            .replace("&", r"\u0026")
        )
        map_svg = _svg_for_frame(frame)
        document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{label}</title>
<style>body{{font:14px system-ui;margin:2rem;color:#16324f}}pre{{background:#f3f7fa;padding:1rem;overflow:auto}}.badge{{background:#d9f2e6;padding:.3rem .6rem;border-radius:1rem}}.map-shell{{max-width:900px;background:#eef7f3;border:1px solid #c9e4d8;border-radius:12px;padding:.5rem}}.map-shell svg{{display:block;width:100%;height:auto}}</style>
</head><body><h1>{label}</h1><p><span class="badge">GeoAgent 地图产物</span> 数据集：{dataset_name} · 要素：{len(frame)}</p>
<p>该产物包含可供地图组件载入的 GeoJSON 数据。</p><div class="map-shell">{map_svg}</div><script type="application/json" id="geojson-data">{safe_geojson}</script>
<pre id="preview"></pre><script>const data=JSON.parse(document.querySelector('#geojson-data').textContent || "{{}}");document.querySelector('#preview').textContent=JSON.stringify(data,null,2).slice(0,12000);</script>
</body></html>"""
        target.write_text(document, encoding="utf-8")
        return target


def _svg_for_frame(frame: gpd.GeoDataFrame, *, width: int = 900, height: int = 520) -> str:
    """生成不依赖外部脚本的只读 SVG，几何数值来自已读取的数据而非用户 HTML。"""

    if frame.empty:
        return f'<svg class="map-canvas" viewBox="0 0 {width} {height}" role="img" aria-label="空地图"><rect width="100%" height="100%" fill="#eef7f3"/><text x="24" y="48" fill="#537064">没有可显示的要素</text></svg>'
    min_x, min_y, max_x, max_y = (float(value) for value in frame.total_bounds)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    padding = 24.0

    def project(position: list[float] | tuple[float, ...]) -> tuple[float, float]:
        x, y = float(position[0]), float(position[1])
        return (padding + (x - min_x) / span_x * (width - padding * 2), height - padding - (y - min_y) / span_y * (height - padding * 2))

    shapes: list[str] = []
    for geometry in frame.geometry:
        if geometry is None or geometry.is_empty:
            continue
        data = geometry.__geo_interface__
        geometry_type = data["type"]
        coordinates = data["coordinates"]
        if geometry_type == "Point":
            x, y = project(coordinates)
            shapes.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#348166" stroke="#fff" stroke-width="1.5"/>')
            continue
        lines = _coordinate_lines(coordinates)
        for line in lines:
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (project(position) for position in line))
            if not points:
                continue
            if "Polygon" in geometry_type:
                shapes.append(f'<polygon points="{points}" fill="#7bbda0" fill-opacity=".35" stroke="#348166" stroke-width="1.2"/>')
            else:
                shapes.append(f'<polyline points="{points}" fill="none" stroke="#348166" stroke-width="1.5"/>')
    return f'<svg class="map-canvas" viewBox="0 0 {width} {height}" role="img" aria-label="数据地图"><rect width="100%" height="100%" rx="12" fill="#eef7f3"/>{"".join(shapes)}</svg>'


def _coordinate_lines(value: object) -> list[list[list[float]]]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    if isinstance(value[0], (int, float)):
        return [[value]]  # type: ignore[list-item]
    if isinstance(value[0], list) and value[0] and isinstance(value[0][0], (int, float)):
        return [value]  # type: ignore[return-value]
    lines: list[list[list[float]]] = []
    for item in value:
        lines.extend(_coordinate_lines(item))
    return lines
