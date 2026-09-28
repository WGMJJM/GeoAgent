from __future__ import annotations

from pathlib import Path

import pytest

from app.core.models import Dataset, DatasetKind, RiskLevel
from app.execution.arcpy.provider import ArcPyProvider
from app.execution.arcpy.schema import (
    ArcPyParameter,
    ArcPyToolSpec,
    UnsupportedArcPyTool,
    build_tool_definition,
)
from app.execution.sandbox import WorkspaceManager
from app.execution.tools import ToolContext


def _slope_description(version: str = "3.4.3") -> dict:
    return {
        "name": "Slope_sa",
        "version": version,
        "usage": "Slope_sa(in_raster, out_raster, {output_measurement}, {z_factor})",
        "parameters": [
            {
                "name": "in_raster",
                "display_name": "输入栅格",
                "direction": "Input",
                "datatype": "复合地理数据集",
                "parameter_type": "Required",
                "multi_value": False,
            },
            {
                "name": "out_raster",
                "display_name": "输出栅格",
                "direction": "Output",
                "datatype": "栅格数据集",
                "parameter_type": "Required",
                "multi_value": False,
            },
            {
                "name": "output_measurement",
                "display_name": "输出测量单位",
                "direction": "Input",
                "datatype": "字符串",
                "parameter_type": "Optional",
                "multi_value": False,
                "filter_type": "ValueList",
                "filter_list": ["度", "增量百分比"],
            },
            {
                "name": "z_factor",
                "display_name": "Z 因子",
                "direction": "Input",
                "datatype": "双精度型",
                "parameter_type": "Optional",
                "multi_value": False,
            },
        ],
    }


class FakeWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def request(self, action, payload=None, **_kwargs):
        payload = payload or {}
        self.calls.append((action, payload))
        if action == "ping":
            return {"version": "3.4.3", "product": "ArcGIS Pro"}
        if action == "catalog":
            return {"version": "3.4.3", "tools": ["Slope_sa", "Buffer_analysis"]}
        if action == "describe":
            return _slope_description()
        if action == "execute":
            output = Path(payload["values"][1])
            output.write_bytes(b"fake raster")
            return {"outputs": [str(output)], "messages": "Succeeded"}
        raise AssertionError(action)

    def close(self):
        self.closed = True


class FakeRegistry:
    def __init__(self, source: Dataset) -> None:
        self.source = source
        self.registered: list[dict] = []

    def resolve(self, identifier, *, user_id=None):
        return self.source if identifier == self.source.id and user_id == self.source.owner_user_id else None

    def register_path(self, path, **kwargs):
        self.registered.append({"path": Path(path), **kwargs})
        return Dataset(
            id="ds_result",
            name=Path(path).stem,
            kind=DatasetKind.RASTER,
            path=str(path),
            format="tif",
            owner_user_id=self.source.owner_user_id,
            created_by_run_id=kwargs["run_id"],
            source_dataset_ids=kwargs["source_dataset_ids"],
        )


def test_localized_arcpy_parameters_become_dataset_scalar_and_generated_output_schema():
    spec = ArcPyToolSpec.from_dict(_slope_description(), public_name="arcpy.slope_sa")
    definition = build_tool_definition(spec)

    assert definition.metadata.name == "arcpy.slope_sa"
    assert definition.metadata.risk_level is RiskLevel.WRITE
    assert definition.metadata.required_envs == ["gis.arcpy", "workspace"]
    assert definition.metadata.required_scopes == ["dataset.read", "dataset.write", "workspace.write"]
    assert definition.metadata.input_schema["required"] == ["in_raster"]
    assert definition.metadata.input_schema["properties"]["in_raster"]["type"] == "string"
    assert "Dataset ID" in definition.metadata.input_schema["properties"]["in_raster"]["description"]
    assert definition.metadata.input_schema["properties"]["z_factor"]["type"] == "number"
    assert "enum" not in definition.metadata.input_schema["properties"]["output_measurement"]
    assert "out_raster" not in definition.metadata.input_schema["properties"]
    assert definition.generated_outputs == {"out_raster": ".tif"}


def test_unknown_required_parameter_rejects_tool_instead_of_guessing_schema():
    spec = ArcPyToolSpec(
        actual_name="Complex_toolbox",
        public_name="arcpy.complex_toolbox",
        version="3.4",
        usage="Complex_toolbox(value_table)",
        parameters=(
            ArcPyParameter(
                name="value_table",
                display_name="值表",
                direction="Input",
                datatype="值表",
                parameter_type="Required",
            ),
        ),
    )

    with pytest.raises(UnsupportedArcPyTool, match="value_table"):
        build_tool_definition(spec)


def test_provider_caches_light_catalog_materializes_one_schema_and_executes_with_dataset_ids(tmp_path):
    executable = tmp_path / "propy.bat"
    executable.write_text("test", encoding="utf-8")
    source_path = tmp_path / "source.tif"
    source_path.write_bytes(b"source")
    source = Dataset(
        id="ds_dem",
        name="dem",
        kind=DatasetKind.RASTER,
        path=str(source_path),
        format="tif",
        owner_user_id="user-1",
    )
    worker = FakeWorker()
    provider = ArcPyProvider(executable, tmp_path / "cache", worker=worker)

    summaries = provider.summaries()
    assert [item.name for item in summaries] == ["arcpy.buffer_analysis", "arcpy.slope_sa"]
    assert [action for action, _ in worker.calls] == ["ping", "catalog"]
    provider.summaries()
    assert [action for action, _ in worker.calls] == ["ping", "catalog"]

    registered = provider.materialize("arcpy.slope_sa")
    assert [action for action, _ in worker.calls].count("describe") == 1
    provider.materialize("arcpy.slope_sa")
    assert [action for action, _ in worker.calls].count("describe") == 1

    registry = FakeRegistry(source)
    context = ToolContext(
        run_id="run-1",
        agent_id="agent-loop",
        call_id="call-1",
        services={
            "workspace": WorkspaceManager(tmp_path / "workspace"),
            "registry": registry,
            "user_id": "user-1",
        },
    )
    result = registered.handler({"in_raster": source.id, "z_factor": 2.0}, context)

    execute_payload = next(payload for action, payload in worker.calls if action == "execute")
    assert execute_payload["name"] == "Slope_sa"
    assert execute_payload["values"][0] == str(source_path.resolve())
    assert execute_payload["values"][1].endswith(".tif")
    assert execute_payload["values"][2] is None
    assert execute_payload["values"][3] == 2.0
    assert result["datasets"] == ["ds_result"]
    assert registry.registered[0]["source_dataset_ids"] == [source.id]
    assert registry.registered[0]["operation"] == "arcpy.slope_sa"

    provider.close()
    assert worker.closed is True


def test_new_provider_reuses_versioned_catalog_and_tool_schema_cache(tmp_path):
    executable = tmp_path / "propy.bat"
    executable.write_text("test", encoding="utf-8")
    cache = tmp_path / "cache"
    first_worker = FakeWorker()
    first = ArcPyProvider(executable, cache, worker=first_worker)
    first.summaries()
    first.materialize("arcpy.slope_sa")

    second_worker = FakeWorker()
    second = ArcPyProvider(executable, cache, worker=second_worker)
    second.summaries()
    second.materialize("arcpy.slope_sa")

    actions = [action for action, _ in second_worker.calls]
    assert actions == ["ping"]
    assert not list(cache.rglob("*.tmp"))
