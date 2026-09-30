from __future__ import annotations

import pytest

from app.auth.policy import PermissionPolicy, ToolDiscoveryContext
from app.core.models import RiskLevel, ToolMetadata
from app.execution.tools import RegisteredTool, ToolCatalog, ToolRegistry


def _metadata(
    name: str,
    description: str,
    *,
    properties: dict | None = None,
    scopes: list[str] | None = None,
    envs: list[str] | None = None,
    risk: RiskLevel = RiskLevel.READ,
) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties or {}, "additionalProperties": False},
        required_scopes=scopes if scopes is not None else [],
        required_envs=envs if envs is not None else [],
        risk_level=risk,
    )


def _register(registry: ToolRegistry, metadata: ToolMetadata, *, deferred: bool = True) -> None:
    registry.register(metadata, lambda _arguments, _context: {}, deferred=deferred)


def _tool_catalog(registry: ToolRegistry, is_discoverable=None, *, providers=()) -> ToolCatalog:
    return ToolCatalog(
        registry,
        is_discoverable,
        providers=providers,
        regex_results=2,
        chinese_bm25_results=1,
        english_bm25_results=3,
    )


def _context(
    *,
    scopes: set[str] | None = None,
    envs: set[str] | None = None,
) -> ToolDiscoveryContext:
    return ToolDiscoveryContext(
        frozenset(scopes if scopes is not None else {"dataset.read", "dataset.write", "workspace.read", "workspace.write", "artifact.create"}),
        frozenset(envs if envs is not None else {"gis.dataset", "gis.vector", "gis.raster", "gis.crs", "gis.visualization", "workspace"}),
    )


def _catalog(*metadata: ToolMetadata) -> tuple[ToolRegistry, ToolCatalog]:
    registry = ToolRegistry()
    for item in metadata:
        _register(registry, item)
    return registry, _tool_catalog(registry, PermissionPolicy().is_discoverable)


def test_search_matches_exact_name_substring_english_description_chinese_and_parameters():
    _, catalog = _catalog(
        _metadata("vector.buffer", "按距离生成矢量缓冲区 / Create vector buffers by distance"),
        _metadata("crs.project", "Transform coordinates to a projected coordinate reference system"),
        _metadata(
            "analysis.custom",
            "Run a spatial operation",
            properties={"target_crs": {"type": "string", "description": "Output reference altitude"}},
        ),
    )
    context = _context()

    for query, name in (
        ("vector.buffer", "vector.buffer"),
        ("buffer", "vector.buffer"),
        ("projected", "crs.project"),
        ("矢量缓冲区", "vector.buffer"),
        ("vector 缓冲区", "vector.buffer"),
        ("target_crs", "analysis.custom"),
        ("altitude", "analysis.custom"),
    ):
        assert catalog.tool_search({"query": query}, context)["tools"][0]["name"] == name


@pytest.mark.parametrize("overlap", [False, True])
def test_one_bilingual_call_returns_one_chinese_three_english_and_deduplicates(overlap):
    metadata = (
        [
            _metadata("test.shared", "唯一中文能力 quantum_marker"),
            _metadata("test.en_1", "quantum_marker"),
            _metadata("test.en_2", "quantum_marker"),
        ]
        if overlap
        else [
            _metadata("test.zh_0", "唯一中文能力"),
            *[_metadata(f"test.en_{index}", "quantum_marker") for index in range(3)],
        ]
    )
    metadata.append(_metadata("test.forbidden", "唯一中文能力 quantum_marker", scopes=["system.admin"]))
    _, catalog = _catalog(*metadata)
    response = catalog.tool_search({"query": "唯一中文能力", "english_query": "quantum_marker"}, _context())
    expected = ["test.shared", "test.en_1", "test.en_2"] if overlap else ["test.zh_0", "test.en_0", "test.en_1", "test.en_2"]
    assert [item["name"] for item in response["tools"]] == expected
    assert len(response["tools"]) == 4 - int(overlap)
    assert "test.forbidden" not in expected
    assert sum("bm25_zh" in item["matched_by"] for item in response["tools"]) == 1
    assert sum("bm25_en" in item["matched_by"] for item in response["tools"]) == 3


def test_regex_branch_returns_at_most_two_candidates_while_english_bm25_returns_three():
    _, catalog = _catalog(
        *[_metadata(f"vector.buffer_{index}", "Buffer geometry operation") for index in range(4)]
    )

    tools = catalog.tool_search({"query": "buffer"}, _context())["tools"]

    assert sum("regex" in item["matched_by"] for item in tools) == 2
    assert sum("bm25_en" in item["matched_by"] for item in tools) == 3


def test_exact_name_does_not_bypass_scope_or_environment_filtering():
    _, catalog = _catalog(
        _metadata("system.delete_database", "Delete system database", scopes=["system.admin"]),
        _metadata("raster.slope", "Calculate slope from DEM", envs=["gis.raster"]),
    )

    assert catalog.tool_search({"query": "system.delete_database"}, _context())["tools"] == []
    assert catalog.tool_search({"query": "raster.slope"}, _context(envs={"gis.vector", "workspace"}))["tools"] == []


def test_dynamic_provider_participates_in_bm25_without_eagerly_materializing_every_tool():
    class Provider:
        def __init__(self):
            self.summary_calls = 0
            self.materialized = []

        def summaries(self):
            self.summary_calls += 1
            return (
                _metadata(
                    "arcpy.slope_sa",
                    "ArcPy slope geoprocessing tool from a raster elevation surface",
                    envs=["gis.arcpy", "workspace"],
                ),
            )

        def materialize(self, name):
            self.materialized.append(name)
            metadata = _metadata(
                name,
                "Calculate slope with installed ArcPy",
                properties={"in_raster": {"type": "string", "description": "GeoAgent Dataset ID"}},
                envs=["gis.arcpy", "workspace"],
            )
            return RegisteredTool(metadata, lambda _arguments, _context: {})

        def close(self):
            pass

    registry = ToolRegistry()
    _register(registry, _metadata("raster.inspect", "Raster elevation inspection"))
    _register(registry, _metadata("raster.statistics", "Raster elevation statistics"))
    provider = Provider()
    catalog = _tool_catalog(registry, providers=(provider,))
    context = _context(envs={"gis.raster", "gis.arcpy", "workspace"})

    assert catalog.tool_search({"query": "inspection"}, context)["tools"][0]["name"] == "raster.inspect"
    assert provider.summary_calls == 1
    assert provider.materialized == []

    cards = catalog.tool_search({"query": "slope"}, context)["tools"]
    assert cards[0]["name"] == "arcpy.slope_sa"
    assert cards[0]["parameter_names"] == ["in_raster"]
    assert provider.summary_calls == 2
    assert provider.materialized == ["arcpy.slope_sa"]
    assert registry.is_deferred("arcpy.slope_sa") is True

    exact = catalog.tool_search({"query": "ARCPY.SLOPE_SA"}, context)["tools"]
    assert [item["name"] for item in exact] == ["arcpy.slope_sa"]

    catalog.tool_search({"query": "slope"}, context)
    assert provider.materialized == ["arcpy.slope_sa"]

    registry.unregister("arcpy.slope_sa")
    assert catalog.ensure_registered("arcpy.slope_sa", context) is True
    assert registry.is_deferred("arcpy.slope_sa") is True
    assert provider.materialized == ["arcpy.slope_sa", "arcpy.slope_sa"]
