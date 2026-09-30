from __future__ import annotations

from pathlib import Path

from app.core.models import Dataset, DatasetKind
from app.execution.arcpy.provider import ArcPyProvider
from app.execution.arcpy.schema import (
    ArcPyToolSpec,
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


def _feature_analysis_description(version: str = "3.4.3") -> dict:
    return {
        "name": "FeatureAnalysis_stats",
        "version": version,
        "usage": "FeatureAnalysis_stats(Input_Features, Output_Features, Analysis_Field)",
        "parameters": [
            {
                "name": "Input_Features",
                "display_name": "输入要素",
                "direction": "Input",
                "datatype": "要素图层",
                "parameter_type": "Required",
                "multi_value": False,
                "filter_type": "Feature",
                "filter_list": ["Point", "Multipoint", "Polygon"],
            },
            {
                "name": "Output_Features",
                "display_name": "输出要素",
                "direction": "Output",
                "datatype": "要素类",
                "parameter_type": "Required",
                "multi_value": False,
            },
            {
                "name": "Analysis_Field",
                "display_name": "分析字段",
                "direction": "Input",
                "datatype": "字段",
                "parameter_type": "Required",
                "multi_value": False,
                "dependencies": [0],
                "filter_type": "Field",
                "filter_list": ["Short", "Long", "Float", "Double", "BigInteger"],
            },
        ],
    }


class FakeWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.messages = "Succeeded"

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
            return {"outputs": [str(output)], "messages": self.messages}
        raise AssertionError(action)

    def close(self):
        self.closed = True


class FeatureAnalysisWorker(FakeWorker):
    def request(self, action, payload=None, **_kwargs):
        payload = payload or {}
        self.calls.append((action, payload))
        if action == "ping":
            return {"version": "3.4.3", "product": "ArcGIS Pro"}
        if action == "catalog":
            return {"version": "3.4.3", "tools": ["FeatureAnalysis_stats"]}
        if action == "describe":
            return _feature_analysis_description()
        if action == "execute":
            output = Path(payload["values"][1])
            output.write_bytes(b"fake vector")
            return {"outputs": [str(output)], "messages": self.messages}
        raise AssertionError(action)


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


def test_feature_and_field_filters_describe_constraints_without_replacing_real_values():
    spec = ArcPyToolSpec.from_dict(_feature_analysis_description(), public_name="arcpy.featureanalysis_stats")
    definition = build_tool_definition(spec)
    properties = definition.metadata.input_schema["properties"]

    assert definition.input_kinds == {"Input_Features": "dataset", "Analysis_Field": "field"}
    assert properties["Input_Features"]["type"] == "string"
    assert "enum" not in properties["Input_Features"]
    assert "Dataset ID" in properties["Input_Features"]["description"]
    assert "Point, Multipoint, Polygon" in properties["Input_Features"]["description"]
    assert properties["Analysis_Field"]["type"] == "string"
    assert "enum" not in properties["Analysis_Field"]
    assert "真实字段名" in properties["Analysis_Field"]["description"]
    assert "Short, Long, Float, Double, BigInteger" in properties["Analysis_Field"]["description"]


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
    assert result["warnings"] == []
    assert registry.registered[0]["source_dataset_ids"] == [source.id]
    assert registry.registered[0]["operation"] == "arcpy.slope_sa"

    worker.messages = "\ufffd\ufffd\ufffd"
    unreadable = registered.handler({"in_raster": source.id}, context)
    assert unreadable["output"]["messages"] == ""
    assert unreadable["warnings"] == ["ArcPy 已完成执行，但本机 ArcGIS Pro 的本地化消息无法正确解码。"]

    provider.close()
    assert worker.closed is True


def test_provider_resolves_dataset_id_and_passes_real_field_name(tmp_path):
    executable = tmp_path / "propy.bat"
    executable.write_text("test", encoding="utf-8")
    source_path = tmp_path / "provinces.shp"
    source_path.write_bytes(b"source")
    source = Dataset(
        id="ds_provinces",
        name="provinces",
        kind=DatasetKind.VECTOR,
        path=str(source_path),
        format="shp",
        owner_user_id="user-1",
    )
    worker = FeatureAnalysisWorker()
    provider = ArcPyProvider(executable, tmp_path / "cache", worker=worker)
    registered = provider.materialize("arcpy.featureanalysis_stats")
    context = ToolContext(
        run_id="run-1",
        agent_id="agent-loop",
        call_id="call-1",
        services={
            "workspace": WorkspaceManager(tmp_path / "workspace"),
            "registry": FakeRegistry(source),
            "user_id": "user-1",
        },
    )

    registered.handler(
        {"Input_Features": source.id, "Analysis_Field": "pm2_5_2020"},
        context,
    )

    execute_payload = next(payload for action, payload in worker.calls if action == "execute")
    assert execute_payload["values"][0] == str(source_path.resolve())
    assert execute_payload["values"][1].endswith(".shp")
    assert execute_payload["values"][2] == "pm2_5_2020"
