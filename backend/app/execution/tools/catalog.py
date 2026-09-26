"""对延迟工具做权限与环境过滤后的中英文关键词发现。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.auth.policy import PermissionPolicy, ToolDiscoveryContext
from app.core.models import ToolMetadata

from .registry import ToolRegistry

MAX_QUERY_LENGTH = 160
MAX_RESULTS = 2
MAX_DESCRIPTION_LENGTH = 300

_ASCII_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")

TOOL_SEARCH_DEFINITION = {
    "type": "function",
    "function": {
        "name": "tool.search",
        "description": "Search accessible tools using Chinese or English capability keywords or tool names. Each search returns at most two tools. Submit Chinese and English searches together in one batch; their results are deduplicated by tool name and merged for the next model turn. Do not call newly discovered tools in the search batch.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_LENGTH,
                    "description": "Chinese or English capability keywords or tool names, e.g. 栅格检查, raster metadata CRS or raster.inspect.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": MAX_RESULTS},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class ToolCard:
    """精简的模型可见结果；score 只用于服务端排序和测试。"""

    name: str
    description: str
    parameter_names: tuple[str, ...]
    score: int

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameter_names": list(self.parameter_names),
        }


DiscoverabilityCheck = Callable[[ToolMetadata, ToolDiscoveryContext], bool]


class ToolCatalog:
    def __init__(
        self,
        registry: ToolRegistry,
        is_discoverable: DiscoverabilityCheck | None = None,
    ) -> None:
        self.registry = registry
        policy = PermissionPolicy()
        self.is_discoverable = is_discoverable or policy.is_discoverable

    def search(self, query: str, context: ToolDiscoveryContext, limit: int = MAX_RESULTS) -> list[ToolCard]:
        normalized = _normalize_query(query)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit 必须是 1 到 {MAX_RESULTS} 之间的整数。")

        matches: list[ToolCard] = []
        # 每次从 Registry 读取最新延迟工具集合，不缓存索引，注册/注销立即生效。
        for name in self.registry.deferred_names():
            try:
                registered = self.registry.get(name)
            except KeyError:
                continue
            metadata = registered.metadata
            if not self.is_discoverable(metadata, context):
                continue
            score = relevance(normalized, metadata)
            if score <= 0:
                continue
            description = " ".join(metadata.description.split())
            if len(description) > MAX_DESCRIPTION_LENGTH:
                description = description[: MAX_DESCRIPTION_LENGTH - 3].rstrip() + "..."
            properties = metadata.input_schema.get("properties", {})
            parameter_names = tuple(str(key) for key in properties) if isinstance(properties, dict) else ()
            matches.append(ToolCard(name, description, parameter_names, score))

        matches.sort(key=lambda item: (-item.score, item.name))
        return matches[: min(limit, MAX_RESULTS)]

    def tool_search(self, arguments: dict[str, Any], context: ToolDiscoveryContext) -> dict[str, Any]:
        """tool.search 的轻量响应包装，供 AgentLoop 作为标准工具观察返回。"""

        cards = self.search(arguments.get("query"), context, arguments.get("limit", MAX_RESULTS))
        response: dict[str, Any] = {"tools": [item.public() for item in cards]}
        if not cards:
            response["message"] = "未找到匹配工具，可改用工具名称或简短中文/英文能力词重新搜索。"
        return response


def tokenize(query: str) -> tuple[str, ...]:
    terms = list(_ASCII_TOKEN.findall(query.casefold()))
    for text in _CJK_RUN.findall(query):
        if len(text) <= 2:
            terms.append(text)
        else:
            terms.extend(text[index : index + 2] for index in range(len(text) - 1))
    return tuple(dict.fromkeys(term for term in terms if len(term) >= 2))


def relevance(query: str, metadata: ToolMetadata) -> int:
    name = metadata.name.casefold()
    description = " ".join(metadata.description.casefold().split())
    properties = metadata.input_schema.get("properties", {})
    argument_parts: list[str] = []
    if isinstance(properties, dict):
        for key, value in properties.items():
            details = value.get("description", "") if isinstance(value, dict) else ""
            argument_parts.append(f"{key} {details}")
    arguments = " ".join(argument_parts).casefold()
    all_text = f"{name} {description} {arguments}"
    normalized_query = " ".join(query.casefold().split())
    score = 30 if normalized_query in all_text else 0
    for term in tokenize(normalized_query):
        if term in name:
            score += 12
        elif term in description:
            score += 6
        elif term in arguments:
            score += 3
    return score


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query 必须是字符串。")
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query 不能为空。")
    if len(normalized) > MAX_QUERY_LENGTH:
        raise ValueError(f"query 不能超过 {MAX_QUERY_LENGTH} 个字符。")
    return normalized


__all__ = [
    "MAX_QUERY_LENGTH",
    "MAX_RESULTS",
    "TOOL_SEARCH_DEFINITION",
    "ToolCard",
    "ToolCatalog",
    "relevance",
    "tokenize",
]
