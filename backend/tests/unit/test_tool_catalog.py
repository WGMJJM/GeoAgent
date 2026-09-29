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
    return registry, ToolCatalog(registry, PermissionPolicy().is_discoverable)


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
        assert [item.name for item in catalog.search(query, context)] == [name]
        assert [item["name"] for item in catalog.tool_search({"query": query}, context)["tools"]] == [name]


def test_english_query_keeps_tool_names_case_and_whitespace_support():
    _, catalog = _catalog(_metadata("raster.inspect", "Inspect raster metadata CRS"))
    assert [item.name for item in catalog.search("  RASTER.INSPECT\tmetadata CRS  ", _context())] == ["raster.inspect"]


def test_name_matches_rank_above_description_and_ties_use_name_order():
    _, catalog = _catalog(
        _metadata("vector.buffer", "Other operation"),
        _metadata("analysis.nearby", "Buffer operation"),
        _metadata("z.shared", "Shared capability"),
        _metadata("a.shared", "Shared capability"),
    )

    assert [item.name for item in catalog.search("buffer", _context())] == ["vector.buffer"]
    tied = catalog.search("capability", _context())
    assert [item.name for item in tied] == ["a.shared"]


def test_exact_identifier_term_ranks_above_substring_and_complete_query_coverage():
    _, catalog = _catalog(
        _metadata("raster.slope", "Calculate terrain slope", properties={}),
        _metadata("arcpy.generate_hachures_for_defined_slopes", "Terrain cartography"),
        _metadata("analysis.terrain", "Terrain operation"),
    )

    assert [item.name for item in catalog.search("slope", _context())] == ["raster.slope"]
    assert catalog.search("terrain slope", _context())[0].name == "raster.slope"


def test_search_returns_only_the_best_result():
    registry = ToolRegistry()
    for index in range(12):
        _register(registry, _metadata(f"analysis.operation_{index:02d}", "Geometry analysis utility"))
    catalog = ToolCatalog(registry, PermissionPolicy().is_discoverable)

    assert len(catalog.search("geometry", _context(), limit=1)) == 1
    assert [item.name for item in catalog.search("geometry", _context())] == ["analysis.operation_00"]


@pytest.mark.parametrize("overlap", [False, True])
def test_one_bilingual_call_returns_deduplicated_union_with_per_query_limit(overlap):
    metadata = [
        *[_metadata(f"test.zh_{index}", "唯一中文能力") for index in range(3)],
        *[_metadata(f"test.en_{index}", "quantum_marker") for index in range(3)],
        _metadata("test.forbidden", "唯一中文能力 quantum_marker", scopes=["system.admin"]),
    ]
    if overlap:
        metadata.append(_metadata("test.a_shared", "唯一中文能力 quantum_marker"))
    _, catalog = _catalog(*metadata)
    chinese = catalog.search("唯一中文能力", _context())
    english = catalog.search("quantum_marker", _context())
    expected = list(dict.fromkeys(item.name for item in [*chinese, *english]))
    response = catalog.tool_search({"query": "唯一中文能力", "english_query": "quantum_marker"}, _context())
    assert [item["name"] for item in response["tools"]] == expected
    assert len(response["tools"]) == 2 - int(overlap)
    assert "test.forbidden" not in expected


@pytest.mark.parametrize("query,english_query", [("无关中文", "buffer"), ("缓冲区", "unknown_marker"), ("无关中文", "unknown_marker")])
def test_bilingual_call_preserves_nonempty_branch_and_reports_only_total_miss(query, english_query):
    _, catalog = _catalog(_metadata("vector.buffer", "矢量缓冲区 / Create vector buffers"))
    response = catalog.tool_search({"query": query, "english_query": english_query}, _context())
    expected = [] if english_query == "unknown_marker" and query == "无关中文" else ["vector.buffer"]
    assert [item["name"] for item in response["tools"]] == expected
    assert ("message" in response) == (not expected)


@pytest.mark.parametrize("limit", [0, 2, 3, True, 1.5, "2"])
def test_invalid_limit_has_a_clear_error(limit):
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))
    with pytest.raises(ValueError, match="limit"):
        catalog.search("buffer", _context(), limit=limit)


@pytest.mark.parametrize("query", ["", " \n\t", "x" * 161, None])
def test_empty_or_excessively_long_query_is_rejected(query):
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))
    with pytest.raises(ValueError, match="query"):
        catalog.search(query, _context())


def test_no_match_returns_empty_list_without_falling_back_to_registry_contents():
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))

    assert catalog.search("unrelated capability", _context()) == []
    response = catalog.tool_search({"query": "unrelated capability"}, _context())
    assert response["tools"] == []
    assert "重新搜索" in response["message"]


def test_exact_name_does_not_bypass_scope_or_environment_filtering():
    _, catalog = _catalog(
        _metadata("system.delete_database", "Delete system database", scopes=["system.admin"]),
        _metadata("raster.slope", "Calculate slope from DEM", envs=["gis.raster"]),
    )

    assert catalog.search("system.delete_database", _context()) == []
    assert catalog.search("raster.slope", _context(envs={"gis.vector", "workspace"})) == []


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

    assert [item.name for item in catalog.search("buffer", _context())] == ["vector.buffer"]


def test_registry_add_and_remove_immediately_changes_search_results():
    registry = ToolRegistry()
    catalog = ToolCatalog(registry, PermissionPolicy().is_discoverable)
    context = _context()
    first = _metadata("crs.reproject", "Reproject a dataset")

    _register(registry, first)
    assert [item.name for item in catalog.search("reproject", context)] == ["crs.reproject"]
    _register(registry, _metadata("raster.reproject", "Reproject a raster"))
    assert [item.name for item in catalog.search("reproject", context)] == ["crs.reproject"]
    assert registry.unregister("crs.reproject") is True
    assert [item.name for item in catalog.search("reproject", context)] == ["raster.reproject"]
    assert registry.unregister("crs.reproject") is False
    assert registry.is_deferred("crs.reproject") is False


def test_dynamic_provider_is_loaded_only_when_registered_results_do_not_fill_limit():
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
    catalog = ToolCatalog(registry, providers=(provider,))
    context = _context(envs={"gis.raster", "gis.arcpy", "workspace"})

    assert len(catalog.search("raster elevation", context)) == 1
    assert provider.summary_calls == 0

    cards = catalog.search("slope", context)
    assert [item.name for item in cards] == ["arcpy.slope_sa"]
    assert cards[0].parameter_names == ("in_raster",)
    assert provider.summary_calls == 1
    assert provider.materialized == ["arcpy.slope_sa"]
    assert registry.is_deferred("arcpy.slope_sa") is True

    exact = catalog.search("ARCPY.SLOPE_SA", context)
    assert [item.name for item in exact] == ["arcpy.slope_sa"]

    catalog.search("slope", context)
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

    card = catalog.search("buffer", _context())[0]
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
    cards = ToolCatalog(registry).search(query, _context())
    assert cards[0].name == "raster.inspect"
    assert "first-band sampled statistics" in cards[0].description
    assert "512x512" in cards[0].description
    assert "no histogram" in cards[0].description
    assert registry.is_deferred("raster.inspect")


def test_tool_search_protocol_supports_one_bilingual_call_and_caps_each_query_at_one():
    function = TOOL_SEARCH_DEFINITION["function"]
    assert function["name"] == "tool.search"
    assert function["parameters"]["required"] == ["query"]
    assert function["parameters"]["properties"]["limit"]["maximum"] == 1
    assert function["parameters"]["properties"]["limit"]["default"] == 1
    query = function["parameters"]["properties"]["query"]
    assert query["maxLength"] == 160
    english_query = function["parameters"]["properties"]["english_query"]
    assert english_query["minLength"] == 1
    assert english_query["maxLength"] == 160
    assert "granted_scopes" not in function["parameters"]["properties"]
    assert "available_envs" not in function["parameters"]["properties"]
