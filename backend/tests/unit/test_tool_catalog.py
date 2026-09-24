from __future__ import annotations

import pytest

from app.auth.policy import PermissionPolicy, ToolDiscoveryContext
from app.core.models import RiskLevel, ToolMetadata
from app.execution.tools import TOOL_SEARCH_DEFINITION, ToolCatalog, ToolRegistry
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
    registry, catalog = _catalog(
        _metadata("vector.buffer", "按距离生成矢量缓冲区 / Create vector buffers by distance"),
        _metadata("crs.project", "Transform coordinates to a projected coordinate reference system"),
        _metadata(
            "analysis.custom",
            "Run a spatial operation",
            properties={"target_crs": {"type": "string", "description": "Output reference altitude"}},
        ),
    )
    context = _context()

    assert [item.name for item in catalog.search("vector.buffer", context)] == ["vector.buffer"]
    assert [item.name for item in catalog.search("buffer", context)] == ["vector.buffer"]
    assert [item.name for item in catalog.search("projected", context)] == ["crs.project"]
    assert [item.name for item in catalog.search("矢量缓冲区", context)] == ["vector.buffer"]
    assert [item.name for item in catalog.search("target_crs", context)] == ["analysis.custom"]
    assert [item.name for item in catalog.search("altitude", context)] == ["analysis.custom"]


def test_name_matches_rank_above_description_and_ties_use_name_order():
    _, catalog = _catalog(
        _metadata("vector.buffer", "Other operation"),
        _metadata("analysis.nearby", "Buffer operation"),
        _metadata("z.shared", "Shared capability"),
        _metadata("a.shared", "Shared capability"),
    )

    assert [item.name for item in catalog.search("buffer", _context())] == ["vector.buffer", "analysis.nearby"]
    tied = catalog.search("capability", _context())
    assert [item.name for item in tied] == ["a.shared", "z.shared"]
    assert tied[0].score == tied[1].score


def test_search_caps_results_at_five_and_honors_smaller_limit():
    registry = ToolRegistry()
    for index in range(12):
        _register(registry, _metadata(f"analysis.operation_{index:02d}", "Geometry analysis utility"))
    catalog = ToolCatalog(registry, PermissionPolicy().is_discoverable)

    assert len(catalog.search("geometry", _context(), limit=5)) == 5
    assert len(catalog.search("geometry", _context(), limit=1)) == 1
    assert [item.name for item in catalog.search("geometry", _context())] == [
        "analysis.operation_00",
        "analysis.operation_01",
        "analysis.operation_02",
        "analysis.operation_03",
        "analysis.operation_04",
    ]
    assert len(catalog.search("geometry", _context(), limit=5)) <= 5


@pytest.mark.parametrize("limit", [0, 6, True, 1.5, "2"])
def test_invalid_limit_has_a_clear_error(limit):
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))
    with pytest.raises(ValueError, match="limit"):
        catalog.search("buffer", _context(), limit=limit)


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "x" * 161, None])
def test_empty_or_excessively_long_query_is_rejected(query):
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))
    with pytest.raises(ValueError, match="query"):
        catalog.search(query, _context())


def test_no_match_returns_empty_list_without_falling_back_to_registry_contents():
    _, catalog = _catalog(_metadata("vector.buffer", "Create vector buffers"))

    assert catalog.search("unrelated capability", _context()) == []


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
    assert [item.name for item in catalog.search("reproject", context)] == ["crs.reproject", "raster.reproject"]
    assert registry.unregister("crs.reproject") is True
    assert [item.name for item in catalog.search("reproject", context)] == ["raster.reproject"]
    assert registry.unregister("crs.reproject") is False
    assert registry.is_deferred("crs.reproject") is False


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

    assert len(registry.names()) == 21
    assert registry.deferred_names() == tuple(name for name in registry.names() if name not in {"dataset.list", "dataset.inspect"})
    assert registry.is_deferred("dataset.list") is False
    assert registry.is_deferred("dataset.inspect") is False
    assert registry.is_deferred("vector.buffer") is True
    assert registry.is_deferred("raster.slope") is True
    assert registry.get("python.execute").metadata.required_envs == ["isolated_python", "workspace"]
    assert registry.get("shell.execute").metadata.required_envs == ["isolated_shell", "workspace"]
    assert all(registry.get(name).metadata.required_envs for name in registry.deferred_names())


def test_tool_search_protocol_is_structured_and_capped_at_five():
    function = TOOL_SEARCH_DEFINITION["function"]
    assert function["name"] == "tool.search"
    assert function["parameters"]["required"] == ["query"]
    assert function["parameters"]["properties"]["limit"]["maximum"] == 5
    assert "granted_scopes" not in function["parameters"]["properties"]
    assert "available_envs" not in function["parameters"]["properties"]
