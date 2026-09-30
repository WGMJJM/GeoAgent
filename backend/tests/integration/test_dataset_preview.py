import json
from io import BytesIO

from docx import Document
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfWriter

from app.api import create_app


def _register(client, username: str) -> dict:
    response = client.post("/api/v1/auth/register", json={"username": username, "password": "password123", "display_name": username})
    assert response.status_code == 200, response.text
    return response.json()


def _geojson(count: int) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": f"road-{index}", "description": "x" * 300}, "geometry": {"type": "Point", "coordinates": [index, index]}}
            for index in range(count)
        ],
    }


def test_dataset_preview_is_limited_and_user_scoped(application):
    with TestClient(create_app(application)) as client_a:
        user_a = _register(client_a, "preview-owner")
        path = application.workspace.for_user(user_a["id"]).input_dir / "roads.geojson"
        path.write_text(json.dumps(_geojson(3)), encoding="utf-8")
        dataset = application.registry.for_user(user_a["id"]).register_path(path, name="道路")
        application.settings.max_preview_features = 2
        response = client_a.get(f"/api/v1/datasets/{dataset.id}/preview")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["dataset_id"] == dataset.id
        assert len(payload["geojson"]["features"]) == 2
        assert payload["truncated"] is True
        assert payload["source_crs"] in {None, "EPSG:4326"}
        assert len(payload["geojson"]["features"][0]["properties"]["description"]) < 300

    with TestClient(create_app(application)) as client_b:
        _register(client_b, "preview-other")
        assert client_b.get(f"/api/v1/datasets/{dataset.id}/preview").status_code == 404


def test_common_document_and_image_attachments_are_inspected(application):
    document_buffer = BytesIO()
    document = Document()
    document.add_paragraph("GeoAgent Word 文档内容")
    document.save(document_buffer)

    pdf_buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(pdf_buffer)

    image_buffer = BytesIO()
    Image.new("RGB", (12, 8), color=(20, 80, 120)).save(image_buffer, format="PNG")

    uploads = {
        "notes.txt": "GeoAgent 文本内容".encode(),
        "report.docx": document_buffer.getvalue(),
        "legacy.doc": b"legacy-word-document",
        "report.pdf": pdf_buffer.getvalue(),
        "preview.png": image_buffer.getvalue(),
    }
    with TestClient(create_app(application)) as client:
        _register(client, "common-files-owner")
        datasets = {}
        for filename, content in uploads.items():
            response = client.post("/api/v1/attachments", files={"file": (filename, content, "application/octet-stream")})
            assert response.status_code == 200, response.text
            datasets[filename] = response.json()["dataset"]

        text_preview = client.get(f"/api/v1/datasets/{datasets['notes.txt']['id']}/preview").json()
        word_preview = client.get(f"/api/v1/datasets/{datasets['report.docx']['id']}/preview").json()
        pdf_preview = client.get(f"/api/v1/datasets/{datasets['report.pdf']['id']}/preview").json()
        image_preview = client.get(f"/api/v1/datasets/{datasets['preview.png']['id']}/preview").json()

    assert datasets["notes.txt"]["kind"] == "DOCUMENT"
    assert text_preview["text"] == "GeoAgent 文本内容"
    assert "GeoAgent Word 文档内容" in word_preview["text"]
    assert datasets["legacy.doc"]["metadata"]["text_extractable"] is False
    assert pdf_preview["page_count"] == 1
    assert image_preview["kind"] == "IMAGE"
    assert (image_preview["width"], image_preview["height"], image_preview["bands"]) == (12, 8, 3)
