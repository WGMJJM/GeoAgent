import json

from app.core.models import Dataset, DatasetKind
from app.gis.visualization import MapRenderer


def test_map_renderer_escapes_labels_and_embeds_geojson_safely(tmp_path):
    source = tmp_path / "roads.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "</script><script>alert(1)</script>"},
                        "geometry": {"type": "Point", "coordinates": [116.3, 39.9]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = Dataset(name="道路 <script>", kind=DatasetKind.VECTOR, path=str(source), format="geojson")

    target = MapRenderer().render(dataset, tmp_path / "map.html", title="标题 <script>")
    content = target.read_text(encoding="utf-8")

    assert "<title>标题 &lt;script&gt;</title>" in content
    assert "GeoAgent 地图产物" in content
    assert '<svg class="map-canvas"' in content
    assert "window.GEOJSON" not in content
    assert "</script><script>alert(1)</script>" not in content
    assert "\\u003c/script\\u003e" in content
