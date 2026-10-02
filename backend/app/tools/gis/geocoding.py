"""地名查询的只读 HTTP 工具，不选择候选或串联天气工作流。"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.models import ErrorCategory
from app.execution.tools import ToolContext, ToolRegistry
from app.gis.errors import GISFailure
from app.tools.gis import metadata


def register(registry: ToolRegistry) -> None:
    registry.register(
        metadata(
            "location.geocode",
            "查询地名、城市或邮编的候选经纬度与行政区；结合上下文选择，无法确定时询问用户，不能默认选第一项；仅返回代表点，不含区域边界 / Geocode place names, cities or postal codes to WGS84 coordinates with administrative areas. Returns candidates, not region boundaries.",
            tags=["gis", "location", "geocoding", "remote"],
            required_scopes=["location.read"],
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "地点名称或邮编，可在逗号后附国家或一级行政区以缩小范围。保留用户明确的地名，不猜测坐标。",
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
