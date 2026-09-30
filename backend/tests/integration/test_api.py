from pathlib import Path

import geopandas as gpd


def test_api_uploads_shapefile_components_as_one_dataset(application, authenticated_client, tmp_path):
    source_dir = tmp_path / "shapefile"
    source_dir.mkdir()
    source = source_dir / "roads.shp"
    frame = gpd.GeoDataFrame(
        {"name": ["主路", "支路"]},
        geometry=gpd.points_from_xy([114.0, 114.1], [22.5, 22.6]),
        crs="EPSG:4326",
    )
    frame.to_file(source)
    components = [
        path
        for path in source_dir.iterdir()
        if path.suffix.casefold() in {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"}
    ]

    with authenticated_client as client:
        incomplete = client.post(
            "/api/v1/attachments/shapefile",
            files=[("files", ("roads.shp", b"not-a-complete-shapefile", "application/octet-stream"))],
        )
        assert incomplete.status_code == 400
        assert ".dbf" in incomplete.json()["detail"]
        assert ".shx" in incomplete.json()["detail"]
        response = client.post(
            "/api/v1/attachments/shapefile",
            files=[("files", (path.name, path.read_bytes(), "application/octet-stream")) for path in components],
        )

    assert response.status_code == 200, response.text
    dataset = response.json()["dataset"]
    assert dataset["kind"] == "VECTOR"
    assert dataset["format"] == "shp"
    assert dataset["schema"]["feature_count"] == 2
    saved = application.store.get_dataset(dataset["id"])
    assert saved is not None
    assert {path.suffix.casefold() for path in Path(saved.path).parent.iterdir()} >= {".shp", ".shx", ".dbf"}
