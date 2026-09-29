"""对延迟工具做权限与环境过滤后的 Regex＋BM25 发现。"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import RLock
from typing import Any

import bm25s
from bm25s.stopwords import STOPWORDS_EN

from app.auth.policy import PermissionPolicy, ToolDiscoveryContext
from app.core.models import ToolMetadata

from .provider import DynamicToolProvider, UnsupportedToolDefinition
from .registry import ToolRegistry

MAX_QUERY_LENGTH = 160
MAX_DESCRIPTION_LENGTH = 300

_ASCII_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")

TOOL_SEARCH_DEFINITION = {
    "type": "function",
    "function": {
        "name": "tool.search",
        "description": "Discover a necessary capability missing from the tools currently provided. The current tool visibility state lists callable tools with complete schemas and cached cards without schemas; historical search results do not establish current availability. Use callable tools directly, without searching or restoring them; when their results satisfy the user's goal, answer without searching. Express the missing capability in query and issue the tool call without a user-facing preamble. For bilingual discovery, make ONE tool.search call with the original Chinese capability in query and one English equivalent in english_query; do not make separate tool calls. The server combines exact identifier matching with Chinese and English BM25 retrieval, merges candidates by tool name, and returns a small deduplicated set for the next model turn, not automatically executed. Candidates may receive full schemas on that next turn within budget, without a separate selection call. After the complete tool batch, uncalled candidates become cached cards; called tools may retain schemas within budget even when arguments need repair, execution fails, or approval is pending. For a cached card or previously discovered tool without a currently provided schema, send only query with its exact tool name once to restore it from this Run's cache, subject to the context budget; omit english_query. Do not call newly discovered or restored tools in the search batch. No match does not prove a capability is absent.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_LENGTH,
                    "description": "Original capability request or exact tool name. For bilingual discovery, keep the Chinese capability here and put one English equivalent in english_query; for cache restoration, use only the exact tool name here.",
                },
                "english_query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_LENGTH,
                    "description": "One English equivalent of the capability in query. The server searches both within this single tool call. Omit for exact-name cache restoration.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class ToolCard:
    """精简的模型可见结果。"""

    name: str
    description: str
    parameter_names: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameter_names": list(self.parameter_names),
        }


@dataclass(frozen=True)
class _CatalogEntry:
    metadata: ToolMetadata
    provider: DynamicToolProvider | None = None


DiscoverabilityCheck = Callable[[ToolMetadata, ToolDiscoveryContext], bool]


class ToolCatalog:
    def __init__(
        self,
        registry: ToolRegistry,
        is_discoverable: DiscoverabilityCheck | None = None,
        providers: tuple[DynamicToolProvider, ...] = (),
        *,
        regex_results: int,
        chinese_bm25_results: int,
        english_bm25_results: int,
    ) -> None:
        self.registry = registry
        policy = PermissionPolicy()
        self.is_discoverable = is_discoverable or policy.is_discoverable
        self.providers = providers
        self.regex_results = _positive_limit("regex_results", regex_results)
        self.chinese_bm25_results = _positive_limit("chinese_bm25_results", chinese_bm25_results)
        self.english_bm25_results = _positive_limit("english_bm25_results", english_bm25_results)
        self._provider_lock = RLock()

    def card(self, name: str) -> ToolCard:
        metadata = self.registry.get(name).metadata
        description = " ".join(metadata.description.split())
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[: MAX_DESCRIPTION_LENGTH - 3].rstrip() + "..."
        return ToolCard(name, description, tuple(metadata.input_schema.get("properties", {})))

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
            return self._materialize(_CatalogEntry(summary, provider), context)
        return False

    def tool_search(self, arguments: dict[str, Any], context: ToolDiscoveryContext) -> dict[str, Any]:
        """一次完成 Regex、中文 BM25 和英文 BM25，并按工具名去重。"""

        query = _normalize_query(arguments.get("query"))
        english_query = _normalize_query(arguments["english_query"]) if "english_query" in arguments else None
        entries = self._entries(context)
        all_queries = (query, english_query) if english_query is not None else (query,)
        exact_name = next((name for name in entries if name.casefold() == query.casefold()), None)
        if english_query is None and exact_name is not None:
            cards = self._cards((exact_name,), entries, context)
            return {"tools": [{**card.public(), "matched_by": ["regex"]} for card in cards]}
        bm25_index = self._bm25_index(entries)
        branches: list[tuple[str, list[str]]] = [
            ("regex", self._regex_matches(all_queries, entries, self.regex_results)),
        ]
        if english_query is None:
            source = "bm25_zh" if _CJK_RUN.search(query) else "bm25_en"
            limit = self.chinese_bm25_results if source == "bm25_zh" else self.english_bm25_results
            branches.append((source, self._bm25_matches(query, bm25_index, limit)))
        else:
            branches.extend(
                (
                    ("bm25_zh", self._bm25_matches(query, bm25_index, self.chinese_bm25_results)),
                    ("bm25_en", self._bm25_matches(english_query, bm25_index, self.english_bm25_results)),
                )
            )

        names: list[str] = []
        matched_by: dict[str, list[str]] = {}
        for source, branch_names in branches:
            for name in branch_names:
                if name not in matched_by:
                    names.append(name)
                    matched_by[name] = []
                matched_by[name].append(source)

        cards = self._cards(names, entries, context)
        tools = []
        for card in cards:
            public = card.public()
            public["matched_by"] = matched_by[card.name]
            tools.append(public)
        response: dict[str, Any] = {"tools": tools}
        if not tools:
            response["message"] = "未找到匹配工具，可改用工具名称或简短中文/英文能力词重新搜索。"
        return response

    def _entries(self, context: ToolDiscoveryContext) -> dict[str, _CatalogEntry]:
        entries: dict[str, _CatalogEntry] = {}
        for name in self.registry.deferred_names():
            try:
                metadata = self.registry.get(name).metadata
            except KeyError:
                continue
            if self.is_discoverable(metadata, context):
                entries[name] = _CatalogEntry(metadata)
        for provider in self.providers:
            for metadata in provider.summaries():
                if metadata.name not in entries and self.is_discoverable(metadata, context):
                    entries[metadata.name] = _CatalogEntry(metadata, provider)
        return dict(sorted(entries.items()))

    @staticmethod
    def _regex_matches(queries: Iterable[str], entries: dict[str, _CatalogEntry], limit: int) -> list[str]:
        ranked: list[tuple[tuple[int, int, int, int, int], str]] = []
        for name, entry in entries.items():
            ranks = tuple(_regex_rank(query, entry.metadata) for query in queries)
            best = max((rank for rank in ranks if rank is not None), default=None)
            if best is not None:
                ranked.append((best, name))
        ranked.sort(key=lambda item: (tuple(-part for part in item[0]), item[1]))
        return [name for _, name in ranked[:limit]]

    @staticmethod
    def _bm25_index(entries: dict[str, _CatalogEntry]) -> tuple[list[str], bm25s.BM25] | None:
        if not entries:
            return None
        names = list(entries)
        corpus = [list(_metadata_terms(entries[name].metadata)) for name in names]
        retriever = bm25s.BM25(method="lucene")
        retriever.index(corpus, show_progress=False)
        return names, retriever

    @staticmethod
    def _bm25_matches(
        query: str,
        index: tuple[list[str], bm25s.BM25] | None,
        limit: int,
    ) -> list[str]:
        query_terms = list(_tokenize(query))
        if not query_terms or index is None:
            return []
        names, retriever = index
        results = retriever.retrieve(
            [query_terms],
            corpus=list(range(len(names))),
            k=len(names),
            show_progress=False,
        )
        ranked = [
            (float(score), names[int(index)])
            for index, score in zip(results.documents[0], results.scores[0], strict=True)
            if float(score) > 0
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in ranked[:limit]]

    def _cards(
        self,
        names: Iterable[str],
        entries: dict[str, _CatalogEntry],
        context: ToolDiscoveryContext,
    ) -> list[ToolCard]:
        cards: list[ToolCard] = []
        for name in names:
            entry = entries.get(name)
            if entry is None or not self._materialize(entry, context):
                continue
            cards.append(self.card(name))
        return cards

    def _materialize(self, entry: _CatalogEntry, context: ToolDiscoveryContext) -> bool:
        try:
            registered = self.registry.get(entry.metadata.name)
        except KeyError:
            if entry.provider is None:
                return False
            try:
                with self._provider_lock:
                    try:
                        registered = self.registry.get(entry.metadata.name)
                    except KeyError:
                        registered = entry.provider.materialize(entry.metadata.name)
                        self.registry.register(registered.metadata, registered.handler, deferred=True)
            except UnsupportedToolDefinition:
                return False
        return self.registry.is_deferred(registered.metadata.name) and self.is_discoverable(registered.metadata, context)


def _regex_rank(query: str, metadata: ToolMetadata) -> tuple[int, int, int, int, int] | None:
    normalized = query.casefold()
    query_terms = set(_tokenize(query))
    name_terms = set(_identifier_terms(metadata.name))
    tag_terms = {term for tag in metadata.tags for term in _tokenize(tag)}
    properties = metadata.input_schema.get("properties", {})
    parameter_names = tuple(str(key) for key in properties) if isinstance(properties, dict) else ()
    parameter_terms = {term for key in parameter_names for term in _identifier_terms(key)}
    name_matches = query_terms & name_terms
    tag_matches = query_terms & tag_terms
    parameter_matches = query_terms & parameter_terms
    exact_name = int(normalized == metadata.name.casefold())
    exact_parameter = int(normalized in {name.casefold() for name in parameter_names})
    if not exact_name and not exact_parameter and not name_matches and not tag_matches and not parameter_matches:
        return None
    return (
        exact_name,
        exact_parameter,
        int(bool(name_matches)),
        len(name_matches),
        len(tag_matches) + len(parameter_matches),
    )


def _metadata_terms(metadata: ToolMetadata) -> tuple[str, ...]:
    parts = [metadata.name, metadata.description, *metadata.tags]
    properties = metadata.input_schema.get("properties", {})
    if isinstance(properties, dict):
        for key, value in properties.items():
            parts.append(str(key))
            if isinstance(value, dict):
                parts.append(str(value.get("description", "")))
    return _tokenize(" ".join(parts))


def _tokenize(text: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    terms = [
        item.casefold()
        for item in _ASCII_WORD.findall(separated)
        if len(item) >= 2 and item.casefold() not in STOPWORDS_EN
    ]
    for value in _CJK_RUN.findall(text):
        if len(value) <= 2:
            terms.append(value)
        else:
            terms.extend(value[index : index + 2] for index in range(len(value) - 1))
    return tuple(terms)


def _identifier_terms(value: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return tuple(item.casefold() for item in _ASCII_WORD.findall(separated))


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query 必须是字符串。")
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("query 不能为空。")
    if len(normalized) > MAX_QUERY_LENGTH:
        raise ValueError(f"query 不能超过 {MAX_QUERY_LENGTH} 个字符。")
    return normalized


def _positive_limit(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是正整数。")
    return value


__all__ = [
    "MAX_QUERY_LENGTH",
    "TOOL_SEARCH_DEFINITION",
    "ToolCard",
    "ToolCatalog",
]
