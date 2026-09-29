"""对延迟工具做权限与环境过滤后的中英文关键词发现。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from app.auth.policy import PermissionPolicy, ToolDiscoveryContext
from app.core.models import ToolMetadata

from .provider import DynamicToolProvider, UnsupportedToolDefinition
from .registry import ToolRegistry

MAX_QUERY_LENGTH = 160
MAX_RESULTS = 1
MAX_DESCRIPTION_LENGTH = 300

_ASCII_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")

TOOL_SEARCH_DEFINITION = {
    "type": "function",
    "function": {
        "name": "tool.search",
        "description": "Discover a necessary capability missing from the tools currently provided. The current tool visibility state lists callable tools with complete schemas and cached cards without schemas; historical search results do not establish current availability. Use callable tools directly, without searching or restoring them; when their results satisfy the user's goal, answer without searching. Express the missing capability in query and issue the tool call without a user-facing preamble. Search accessible tools using Chinese or English capability keywords or tool names. For bilingual discovery, make ONE tool.search call with Chinese query and english_query for the same capability; do not make separate Chinese and English tool calls. The server searches both internally, returns one tool per query, and returns their deduplicated union (up to two tools) for the next model turn, not automatically executed. Candidates may receive full schemas on that next turn within budget, without a separate selection call. After the complete tool batch, uncalled candidates become cached cards; called tools may retain schemas within budget even when arguments need repair, execution fails, or approval is pending. For a cached card or previously discovered tool without a currently provided schema, send only query with its exact tool name once to restore it from this Run's cache, subject to the context budget; omit english_query. Do not call newly discovered or restored tools in the search batch. No match does not prove a capability is absent.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_LENGTH,
                    "description": "Chinese or English capability keywords or an exact tool name. For bilingual discovery, put Chinese keywords here and English keywords in english_query in the SAME call; for cache restoration, use only the exact tool name here.",
                },
                "english_query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_LENGTH,
                    "description": "English keywords for the same capability as query; the server merges both searches within this one tool call. Omit for exact-name cache restoration.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS, "default": MAX_RESULTS, "description": "Maximum results PER query, before union and deduplication."},
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
        providers: tuple[DynamicToolProvider, ...] = (),
    ) -> None:
        self.registry = registry
        policy = PermissionPolicy()
        self.is_discoverable = is_discoverable or policy.is_discoverable
        self.providers = providers
        self._provider_lock = RLock()

    def card(self, name: str, *, score: int = 0) -> ToolCard:
        metadata = self.registry.get(name).metadata
        description = " ".join(metadata.description.split())
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[: MAX_DESCRIPTION_LENGTH - 3].rstrip() + "..."
        return ToolCard(name, description, tuple(metadata.input_schema.get("properties", {})), score)

    def ensure_registered(self, name: str, context: ToolDiscoveryContext) -> bool:
        """按 Checkpoint 中的精确名称恢复动态工具，不重新做关键词检索。"""

        try:
            registered = self.registry.get(name)
        except KeyError:
            registered = None
        if registered is not None:
            return self.registry.is_deferred(name) and self.is_discoverable(registered.metadata, context)

        for provider in self.providers:
            summary = next((item for item in provider.summaries() if item.name == name), None)
            if summary is None or not self.is_discoverable(summary, context):
                continue
            try:
                with self._provider_lock:
                    try:
                        registered = self.registry.get(name)
                    except KeyError:
                        registered = provider.materialize(name)
                        self.registry.register(registered.metadata, registered.handler, deferred=True)
            except UnsupportedToolDefinition:
                return False
            return self.is_discoverable(registered.metadata, context)
        return False

    def search(self, query: str, context: ToolDiscoveryContext, limit: int = MAX_RESULTS) -> list[ToolCard]:
        normalized = _normalize_query(query)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit 必须是 1 到 {MAX_RESULTS} 之间的整数。")

        matches: list[tuple[int, str, DynamicToolProvider | None]] = []
        # 每次从 Registry 读取最新延迟工具集合，不缓存索引，注册/注销立即生效。
        for name in self.registry.deferred_names():
            try:
                registered = self.registry.get(name)
            except KeyError:
                continue
            metadata = registered.metadata
            if not self.is_discoverable(metadata, context):
                continue
            if name.casefold() == normalized.casefold():
                return [self.card(name, score=100)]
            score = relevance(normalized, metadata)
            if score <= 0:
                continue
            matches.append((score, name, None))

        matches.sort(key=lambda item: (-item[0], item[1]))
        if len(matches) < limit:
            known = {name for _, name, _ in matches}
            for provider in self.providers:
                for metadata in provider.summaries():
                    if metadata.name in known or not self.is_discoverable(metadata, context):
                        continue
                    if metadata.name.casefold() == normalized.casefold():
                        try:
                            with self._provider_lock:
                                try:
                                    registered = self.registry.get(metadata.name)
                                except KeyError:
                                    registered = provider.materialize(metadata.name)
                                    self.registry.register(registered.metadata, registered.handler, deferred=True)
                        except UnsupportedToolDefinition:
                            return []
                        if not self.is_discoverable(registered.metadata, context):
                            return []
                        return [self.card(metadata.name, score=100)]
                    score = relevance(normalized, metadata)
                    if score <= 0:
                        continue
                    matches.append((score, metadata.name, provider))
                    known.add(metadata.name)
            matches.sort(key=lambda item: (-item[0], item[2] is not None, item[1]))

        cards: list[ToolCard] = []
        for score, name, provider in matches:
            if len(cards) >= min(limit, MAX_RESULTS):
                break
            if provider is not None:
                try:
                    with self._provider_lock:
                        try:
                            registered = self.registry.get(name)
                        except KeyError:
                            registered = provider.materialize(name)
                            self.registry.register(registered.metadata, registered.handler, deferred=True)
                except UnsupportedToolDefinition:
                    continue
                if not self.is_discoverable(registered.metadata, context):
                    continue
            cards.append(self.card(name, score=score))
        return cards

    def tool_search(self, arguments: dict[str, Any], context: ToolDiscoveryContext) -> dict[str, Any]:
        """tool.search 的轻量响应包装，供 AgentLoop 作为标准工具观察返回。"""

        queries = [arguments.get("query")]
        if "english_query" in arguments:
            queries.append(arguments["english_query"])
        cards: dict[str, ToolCard] = {}
        for query in queries:
            for card in self.search(query, context, arguments.get("limit", MAX_RESULTS)):
                cards.setdefault(card.name, card)
        response: dict[str, Any] = {"tools": [item.public() for item in cards.values()]}
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
    name_terms = set(_identifier_terms(metadata.name))
    tag_terms = {term for tag in metadata.tags for term in _identifier_terms(tag)}
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
    matched = 0
    for term in tokenize(normalized_query):
        if term in name_terms:
            score += 16
            matched += 1
        elif term in name:
            score += 12
            matched += 1
        elif term in tag_terms:
            score += 8
            matched += 1
        elif term in description:
            score += 6
            matched += 1
        elif term in arguments:
            score += 3
            matched += 1
    query_terms = tokenize(normalized_query)
    if query_terms and matched == len(query_terms):
        score += 20
    return score


def _identifier_terms(value: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return tuple(item.casefold() for item in re.findall(r"[A-Za-z0-9]+", separated))


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
