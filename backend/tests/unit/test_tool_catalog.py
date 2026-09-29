from __future__ import annotations

import pytest

from app.auth.policy import PermissionPolicy, ToolDiscoveryContext
from app.core.models import RiskLevel, ToolMetadata
from app.execution.tools import TOOL_SEARCH_DEFINITION, RegisteredTool, ToolCatalog, ToolRegistry
from app.tools.gis import register_gis_tools
from app.tools.runtime import register_runtime_tools


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


def test_english_query_keeps_tool_names_case_and_whitespace_support():
    _, catalog = _catalog(_metadata("raster.inspect", "Inspect raster metadata CRS"))
    tools = catalog.tool_search({"query": "  RASTER.INSPECT\tmetadata CRS  "}, _context())["tools"]
    assert tools[0]["name"] == "raster.inspect"


def test_regex_identifier_matches_precede_bm25_description_matches():
    _, catalog = _catalog(
        _metadata("vector.buffer", "Other operation"),
        _metadata("analysis.nearby", "Buffer operation"),
        _metadata("z.shared", "Shared capability"),
        _metadata("a.shared", "Shared capability"),
    )

    assert catalog.tool_search({"query": "buffer"}, _context())["tools"][0]["name"] == "vector.buffer"
    tied = catalog.tool_search({"query": "capability"}, _context())["tools"]
    assert tied[0]["name"] == "a.shared"


def test_exact_identifier_term_ranks_above_substring_and_complete_query_coverage():
    _, catalog = _catalog(
        _metadata("raster.slope", "Calculate terrain slope", properties={}),
        _metadata("arcpy.generate_hachures_for_defined_slopes", "Terrain cartography"),
        _metadata("analysis.terrain", "Terrain operation"),
    )

    assert catalog.tool_search({"query": "slope"}, _context())["tools"][0]["name"] == "raster.slope"
    combined = catalog.tool_search({"query": "terrain slope"}, _context())["tools"]
    slope = next(item for item in combined if item["name"] == "raster.slope")
    assert "bm25_en" in slope["matched_by"]


def test_single_english_query_returns_three_bm25_candidates():
    registry = ToolRegistry()
    for index in range(12):
        _register(registry, _metadata(f"analysis.operation_{index:02d}", "Geometry analysis utility"))
    catalog = _tool_catalog(registry, PermissionPolicy().is_discoverable)

    tools = catalog.tool_search({"query": "geometry"}, _context())["tools"]
    assert len(tools) == 3
    assert [item["name"] for item in tools] == [
        "analysis.operation_00",
        "analysis.operation_01",
        "analysis.operation_02",
    ]


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


def test_exact_tool_name_returns_only_that_tool_without_approximate_candidates():
    _, catalog = _catalog(
        _metadata("vector.buffer", "Create a vector buffer"),
        _metadata("analysis.buffer_summary", "Summarize buffer distances"),
    )

    tools = catalog.tool_search({"query": "vector.buffer"}, _context())["tools"]

    assert [item["name"] for item in tools] == ["vector.buffer"]
    assert tools[0]["matched_by"] == ["regex"]


@pytest.mark.parametrize("query,english_query", [("无关中文", "buffer"), ("缓冲区", "unknown_marker"), ("无关中文", "unknown_marker")])
def test_bilingual_call_preserves_nonempty_branch_and_reports_only_total_miss(query, english_query):
    _, catalog = _catalog(_metadata("vector.buffer", "矢量缓冲区 / Create vector buffers"))
    response = catalog.tool_search({"query": query, "english_query": english_query}, _context())
    expected = [] if english_query == "unknown_marker" and query == "无关中文" else ["vector.buffer"]
    assert [item["name"] for item in response["tools"]] == expected
    assert ("message" in response) == (not expected)


@pytest.mark.parametrize("field", ["regex_results", "chinese_bm25_results", "english_bm25_results"])
def test_invalid_branch_limit_has_a_clear_error(field):
    registry = ToolRegistry()
    limits = {"regex_results": 2, "chinese_bm25_results": 1, "english_bm25_results": 3}
    limits[field] = 0
    with pytest.raises(ValueError, match=field):
        ToolCatalog(registry, **limits)


@pytest.mark.parametrize("query", ["", " \n\t", "x" * 161, None])
def test_empty_or_excessively_long_query_is_rejected(query):
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))
    with pytest.raises(ValueError, match="query"):
        catalog.tool_search({"query": query}, _context())


def test_no_match_returns_empty_list_without_falling_back_to_registry_contents():
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))

    response = catalog.tool_search({"query": "unrelated capability"}, _context())
    assert response["tools"] == []
    assert "重新搜索" in response["message"]


def test_exact_name_does_not_bypass_scope_or_environment_filtering():
    _, catalog = _catalog(
        _metadata("system.delete_database", "Delete system database", scopes=["system.admin"]),
        _metadata("raster.slope", "Calculate slope from DEM", envs=["gis.raster"]),
    )

    assert catalog.tool_search({"query": "system.delete_database"}, _context())["tools"] == []
    assert catalog.tool_search({"query": "raster.slope"}, _context(envs={"gis.vector", "workspace"}))["tools"] == []


def test_write_tools_are_discoverable_when_declared_user_scopes_and_environment_exist():
    _, catalog = _catalog(
        _metadata(
            "vector.buffer",
            "Create vector buffers",
            scopes=["dataset.read", "dataset.write", "workspace.write"],
            envs=["gis.vector", "workspace"],
            risk=RiskLevel.WRITE,
        )
    )

    assert catalog.tool_search({"query": "buffer"}, _context())["tools"][0]["name"] == "vector.buffer"


def test_registry_add_and_remove_immediately_changes_search_results():
    registry = ToolRegistry()
    catalog = _tool_catalog(registry, PermissionPolicy().is_discoverable)
    context = _context()
    first = _metadata("crs.reproject", "Reproject a dataset")

    _register(registry, first)
    assert catalog.tool_search({"query": "reproject"}, context)["tools"][0]["name"] == "crs.reproject"
    _register(registry, _metadata("raster.reproject", "Reproject a raster"))
    assert catalog.tool_search({"query": "reproject"}, context)["tools"][0]["name"] == "crs.reproject"
    assert registry.unregister("crs.reproject") is True
    assert catalog.tool_search({"query": "reproject"}, context)["tools"][0]["name"] == "raster.reproject"
    assert registry.unregister("crs.reproject") is False
    assert registry.is_deferred("crs.reproject") is False


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


def test_public_tool_card_contains_no_score_or_schema_details():
    description = "长描述 " * 80
    _, catalog = _catalog(
        _metadata(
            "vector.buffer",
            description,
            properties={"dataset_id": {"type": "string", "description": "Private schema detail"}, "distance": {"type": "number"}},
        )
    )

    card = catalog.card("vector.buffer")
    public = card.public()
    assert set(public) == {"name", "description", "parameter_names"}
    assert public["parameter_names"] == ["dataset_id", "distance"]
    assert len(public["description"]) <= 300
    assert "score" not in public
    assert "input_schema" not in public
    assert "Private schema detail" not in str(public)


def test_discovery_context_uses_authenticated_identity_and_verified_isolation_flags():
    services = {
        "registry": object(),
        "inspector": object(),
        "vectors": object(),
        "rasters": object(),
        "workspace": object(),
        "python": object(),
        "shell": object(),
        "allow_unsafe_python": True,
    }
    context = PermissionPolicy.discovery_context(authenticated_user=True, services=services)
    anonymous = PermissionPolicy.discovery_context(authenticated_user=False, services=services)

    assert {"dataset.read", "dataset.write", "workspace.write"}.issubset(context.granted_scopes)
    assert "system.admin" not in context.granted_scopes
    assert {"gis.dataset", "gis.vector", "gis.raster", "workspace"}.issubset(context.available_envs)
    assert "isolated_python" not in context.available_envs
    assert "isolated_shell" not in context.available_envs
    assert anonymous.granted_scopes == frozenset()


def test_registered_tools_are_split_into_two_resident_and_deferred_capabilities():
    registry = ToolRegistry()
    register_gis_tools(registry)
    register_runtime_tools(registry)

    assert registry.deferred_names() == tuple(name for name in registry.names() if name not in {"dataset.list", "dataset.inspect"})
    assert registry.is_deferred("dataset.list") is False
    assert registry.is_deferred("dataset.inspect") is False
    assert registry.is_deferred("vector.buffer") is True
    assert registry.is_deferred("raster.slope") is True
    assert registry.get("python.execute").metadata.required_envs == ["isolated_python", "workspace"]
    assert registry.get("shell.execute").metadata.required_envs == ["isolated_shell", "workspace"]
    assert all(registry.get(name).metadata.required_envs for name in registry.deferred_names())


@pytest.mark.parametrize("query", ["栅格统计 最小值 最大值", "raster statistics min max"])
def test_raster_sample_statistics_are_discoverable_with_honest_limits(query):
    registry = ToolRegistry()
    register_gis_tools(registry)
    cards = _tool_catalog(registry).tool_search({"query": query}, _context())["tools"]
    inspected = next(item for item in cards if item["name"] == "raster.inspect")
    assert "first-band sampled statistics" in inspected["description"]
    assert "512x512" in inspected["description"]
    assert "no histogram" in inspected["description"]
    assert registry.is_deferred("raster.inspect")


def test_tool_search_protocol_supports_one_bilingual_call_with_server_side_branch_limits():
    function = TOOL_SEARCH_DEFINITION["function"]
    assert function["name"] == "tool.search"
    assert function["parameters"]["required"] == ["query"]
    assert set(function["parameters"]["properties"]) == {"query", "english_query"}
    assert "BM25" in function["description"]
    query = function["parameters"]["properties"]["query"]
    assert query["maxLength"] == 160
    english_query = function["parameters"]["properties"]["english_query"]
    assert english_query["minLength"] == 1
    assert english_query["maxLength"] == 160
    assert "granted_scopes" not in function["parameters"]["properties"]
    assert "available_envs" not in function["parameters"]["properties"]
