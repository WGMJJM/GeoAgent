"""把已安装 ArcPy 工具的参数元数据转换为 GeoAgent Tool Schema。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.models import DatasetOutputPolicy, RiskLevel, ToolMetadata
from app.execution.tools.provider import UnsupportedToolDefinition


class UnsupportedArcPyTool(UnsupportedToolDefinition):
    """工具含有当前转换层无法安全表达的必填参数。"""


@dataclass(frozen=True)
class ArcPyParameter:
    name: str
    display_name: str
    direction: str
    datatype: str | tuple[str, ...]
    parameter_type: str
    multi_value: bool = False
    enabled: bool = True
    filter_type: str | None = None
    filter_list: tuple[Any, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArcPyParameter:
        datatype = value.get("datatype", "")
        if isinstance(datatype, list):
            datatype = tuple(str(item) for item in datatype)
        else:
            datatype = str(datatype)
        return cls(
            name=str(value["name"]),
            display_name=str(value.get("display_name") or value["name"]),
            direction=str(value.get("direction") or "Input"),
            datatype=datatype,
            parameter_type=str(value.get("parameter_type") or "Optional"),
            multi_value=bool(value.get("multi_value")),
            enabled=bool(value.get("enabled", True)),
            filter_type=str(value["filter_type"]) if value.get("filter_type") else None,
            filter_list=tuple(value.get("filter_list") or ()),
        )

    @property
    def datatypes(self) -> tuple[str, ...]:
        return self.datatype if isinstance(self.datatype, tuple) else (self.datatype,)

    @property
    def required(self) -> bool:
        return self.parameter_type.casefold() == "required"

    @property
    def output(self) -> bool:
        return self.direction.casefold() == "output"

    @property
    def derived(self) -> bool:
        return self.parameter_type.casefold() == "derived"


@dataclass(frozen=True)
class ArcPyToolSpec:
    actual_name: str
    public_name: str
    version: str
    usage: str
    parameters: tuple[ArcPyParameter, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, public_name: str) -> ArcPyToolSpec:
        return cls(
            actual_name=str(value["name"]),
            public_name=public_name,
            version=str(value.get("version") or "unknown"),
            usage=str(value.get("usage") or value["name"]),
            parameters=tuple(ArcPyParameter.from_dict(item) for item in value.get("parameters", ())),
        )


@dataclass(frozen=True)
class ArcPyToolDefinition:
    metadata: ToolMetadata
    input_kinds: dict[str, str]
    generated_outputs: dict[str, str]
    mutates_inputs: bool


_STRING_TYPES = {
    "string",
    "field",
    "linear unit",
    "sql expression",
    "where clause",
    "expression",
    "coordinate system",
    "spatial reference",
    "字符串",
    "字段",
    "线性单位",
    "sql 表达式",
    "where 子句",
    "表达式",
    "坐标系",
    "空间参考",
}
_INTEGER_TYPES = {"long", "short", "long integer", "short integer", "长整型", "短整型"}
_NUMBER_TYPES = {"double", "float", "双精度型", "浮点型"}
_BOOLEAN_TYPES = {"boolean", "bool", "布尔", "布尔型"}
_SPATIAL_REFERENCE_TYPES = {"coordinate system", "spatial reference", "坐标系", "空间参考"}
_RASTER_TYPES = {"raster dataset", "raster layer", "栅格数据集", "栅格图层"}
_FEATURE_TYPES = {
    "feature layer",
    "feature class",
    "feature dataset",
    "scene layer",
    "building scene layer",
    "要素图层",
    "要素类",
    "要素数据集",
    "场景图层",
    "构建场景图层",
}
_TABLE_TYPES = {"table", "table view", "表", "表视图"}
_DATASET_TYPES = {
    *_RASTER_TYPES,
    *_FEATURE_TYPES,
    *_TABLE_TYPES,
    "composite geodataset",
    "file",
    "mosaic layer",
    "复合地理数据集",
    "文件",
    "镶嵌图层",
}
_OUTPUT_SUFFIXES = {
    "raster": ".tif",
    "feature": ".shp",
}


def public_tool_name(actual_name: str) -> str:
    return f"arcpy.{actual_name.casefold()}"


def catalog_description(actual_name: str) -> str:
    stem, _, toolbox = actual_name.rpartition("_")
    words = " ".join(_name_words(stem or actual_name))
    toolbox_text = toolbox or "unknown"
    return f"ArcPy {words} geoprocessing tool; toolbox alias {toolbox_text}; installed name {actual_name}."


def build_tool_definition(spec: ArcPyToolSpec) -> ArcPyToolDefinition:
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    input_kinds: dict[str, str] = {}
    generated_outputs: dict[str, str] = {}
    has_derived_output = False

    for parameter in spec.parameters:
        kind = parameter_kind(parameter)
        if parameter.output:
            if parameter.derived:
                has_derived_output = True
                continue
            suffix = _OUTPUT_SUFFIXES.get(kind)
            if suffix is None:
                if parameter.required:
                    raise UnsupportedArcPyTool(
                        f"{spec.actual_name} 的必填输出参数 {parameter.name} 类型尚不支持：{_datatype_text(parameter)}"
                    )
                continue
            generated_outputs[parameter.name] = suffix
            continue

        if kind is None:
            if parameter.required:
                raise UnsupportedArcPyTool(
                    f"{spec.actual_name} 的必填参数 {parameter.name} 类型尚不支持：{_datatype_text(parameter)}"
                )
            continue

        input_kinds[parameter.name] = kind
        property_schema = _property_schema(parameter, kind)
        properties[parameter.name] = property_schema
        if parameter.required:
            required.append(parameter.name)

    mutates_inputs = has_derived_output and not generated_outputs
    risk = RiskLevel.DESTRUCTIVE if mutates_inputs else RiskLevel.WRITE if generated_outputs else RiskLevel.READ
    scopes = ["dataset.read"] if any(kind == "dataset" for kind in input_kinds.values()) else []
    if generated_outputs:
        scopes.extend(["dataset.write", "workspace.write"])
    description = (
        f"ArcPy installed geoprocessing tool {spec.actual_name}. "
        f"Output paths are allocated inside the current GeoAgent workspace. Installed usage: {spec.usage}"
    )
    metadata = ToolMetadata(
        name=spec.public_name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        required_scopes=list(dict.fromkeys(scopes)),
        required_envs=["gis.arcpy", "workspace"],
        risk_level=risk,
        supports_retry=False,
        dataset_output_policy=DatasetOutputPolicy.REQUIRED if generated_outputs else DatasetOutputPolicy.NONE,
        tags=["gis", "arcpy", _toolbox_alias(spec.actual_name)],
    )
    return ArcPyToolDefinition(metadata, input_kinds, generated_outputs, mutates_inputs)


def parameter_kind(parameter: ArcPyParameter) -> str | None:
    values = {item.strip().casefold() for item in parameter.datatypes if item.strip()}
    if values & _DATASET_TYPES:
        if values & _RASTER_TYPES:
            return "raster" if parameter.output else "dataset"
        if values & _FEATURE_TYPES:
            return "feature" if parameter.output else "dataset"
        return "dataset" if not parameter.output else None
    if values & _BOOLEAN_TYPES:
        return "boolean"
    if values & _INTEGER_TYPES:
        return "integer"
    if values & _NUMBER_TYPES:
        return "number"
    if values & _SPATIAL_REFERENCE_TYPES:
        return "spatial_reference"
    if values & _STRING_TYPES:
        return "string"
    return None


def _property_schema(parameter: ArcPyParameter, kind: str) -> dict[str, Any]:
    value_type = {
        "dataset": "string",
        "spatial_reference": "string",
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
    }[kind]
    datatype = _datatype_text(parameter)
    prefix = "GeoAgent Dataset ID; " if kind == "dataset" else ""
    item: dict[str, Any] = {
        "type": value_type,
        "description": f"{prefix}{parameter.display_name} (ArcPy: {datatype})",
    }
    enum = [value for value in parameter.filter_list if isinstance(value, (str, int, float, bool))]
    if value_type == "string" and enum and all(isinstance(value, str) and value.isascii() for value in enum):
        item["enum"] = enum
    if parameter.multi_value:
        return {
            "type": "array",
            "items": item,
            "minItems": 1,
            "description": item["description"],
        }
    return item


def _datatype_text(parameter: ArcPyParameter) -> str:
    return " | ".join(parameter.datatypes)


def _name_words(name: str) -> list[str]:
    return [item.casefold() for item in re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", name)] or [name.casefold()]


def _toolbox_alias(actual_name: str) -> str:
    _, _, alias = actual_name.rpartition("_")
    return alias.casefold() or "unknown"


__all__ = [
    "ArcPyParameter",
    "ArcPyToolDefinition",
    "ArcPyToolSpec",
    "UnsupportedArcPyTool",
    "build_tool_definition",
    "catalog_description",
    "parameter_kind",
    "public_tool_name",
]
