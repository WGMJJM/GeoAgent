"""可按需物化工具 Schema 的轻量目录协议。"""

from __future__ import annotations

from typing import Protocol

from app.core.models import ToolMetadata

from .model import RegisteredTool


class ToolProviderError(RuntimeError):
    pass


class UnsupportedToolDefinition(ToolProviderError):
    pass


class DynamicToolProvider(Protocol):
    def summaries(self) -> tuple[ToolMetadata, ...]: ...

    def materialize(self, name: str) -> RegisteredTool: ...

    def close(self) -> None: ...


__all__ = ["DynamicToolProvider", "ToolProviderError", "UnsupportedToolDefinition"]
