"""地名查询的只读 HTTP 工具，不选择候选或串联天气工作流。"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.models import ErrorCategory
from app.execution.tools import ToolContext, ToolRegistry
from app.gis.errors import GISFailure
from app.run.recovery import retry_after, transient_error
from app.tools.gis import metadata


def register(registry: ToolRegistry) -> None:
    registry.register(
        metadata(
            "location.geocode",
            "中文地名先转换为英文标准名，无通用英文名则使用无声调拼音，直接查询候选经纬度与行政区，不先用中文试查；结合上下文选择，无法确定时询问用户，不能默认选第一项；仅返回代表点，不含区域边界 / Geocode English place names or postal codes to WGS84 coordinates. Returns candidates, not region boundaries.",
            tags=["gis", "location", "geocoding", "remote"],
            required_scopes=["location.read"],
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "查询用的英文标准地名或邮编。用户提供中文地名时，在生成本次参数时转换为英文标准名，无通用英文名则使用无声调拼音；保留原地点含义，避免逐字直译行政区后缀。直接提交英文查询词，不先用中文试查，不猜测坐标。可在逗号后附英文国家或一级行政区以缩小范围；转换无法确定时先询问用户。",
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 10,
                        "description": "最多返回的候选数。返回顺序不表示地点已确定。",
                    },
                    "language": {
                        "type": "string",
                        "default": "zh",
                        "description": "返回名称的语言代码，例如 zh 或 en；不改变搜索词。",
                    },
                    "countryCode": {
                        "type": "string",
                        "pattern": "^[A-Za-z]{2}$",
                        "description": "可选 ISO 3166-1 alpha-2 国家代码，用于筛选候选。",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
        geocode,
        deferred=True,
    )


async def geocode(arguments: dict[str, Any], context: ToolContext) -> dict:
    settings = context.services["settings"]
    parameters = {
        "name": arguments["name"],
        "count": arguments.get("count", 10),
        "language": arguments.get("language", "zh"),
        "format": "json",
    }
    if "countryCode" in arguments:
        parameters["countryCode"] = arguments["countryCode"]
    try:
        async with httpx.AsyncClient(timeout=settings.tool_timeout_seconds) as client:
            response = await client.get(settings.geocoding_url, params=parameters)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise GISFailure(
            "GEOCODING_REQUEST_FAILED", "地名查询服务请求失败。",
            category=ErrorCategory.EXTERNAL,
            retryable=transient_error(exc), details={"retry_after_seconds": retry_after(exc)},
        ) from exc
    data = response.json()
    if data.get("error"):
        raise GISFailure(
            "GEOCODING_SERVICE_ERROR", data["reason"], category=ErrorCategory.EXTERNAL,
        )
    return {
        "output": {
            "query": parameters,
            "results": data.get("results", []),
            "crs": "EPSG:4326",
            "coordinate_kind": "representative_point",
            "source": {
                "provider": "Open-Meteo",
                "url": settings.geocoding_url,
                "attribution": "Location data: GeoNames; CC BY 4.0",
                "attribution_url": "https://www.geonames.org/",
            },
        },
    }
