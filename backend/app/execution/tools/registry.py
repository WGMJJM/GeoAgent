"""Tool Registry：工具的发现、描述和可用性边界。"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.models import ToolMetadata

from .model import RegisteredTool, ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._deferred: set[str] = set()

    def register(self, metadata: ToolMetadata, handler: ToolHandler, *, deferred: bool = False) -> None:
        if metadata.name in self._tools:
            raise ValueError(f"Tool 已注册：{metadata.name}")
        self._tools[metadata.name] = RegisteredTool(metadata, handler)
        if deferred:
            self._deferred.add(metadata.name)

    def unregister(self, name: str) -> bool:
        if name not in self._tools:
            return False
        del self._tools[name]
        self._deferred.discard(name)
        return True

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"未知 Tool：{name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def deferred_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._deferred))

    def is_deferred(self, name: str) -> bool:
        return name in self._deferred

    def definitions(self, *, tags: Iterable[str] | None = None) -> list[ToolMetadata]:
        requested = set(tags or ())
        tools = self._tools.values()
        if requested:
            tools = (tool for tool in tools if requested.intersection(tool.metadata.tags))
        return [tool.metadata for tool in tools]

