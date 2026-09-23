"""Tool Runtime 的局部协议。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Event
from typing import Any

from app.core.models import ToolMetadata

ToolHandler = Callable[[dict[str, Any], "ToolContext"], Any | Awaitable[Any]]


@dataclass(frozen=True)
class RegisteredTool:
    metadata: ToolMetadata
    handler: ToolHandler


@dataclass
class ToolContext:
    run_id: str
    agent_id: str
    services: dict[str, Any]
    cancel_event: Event | None = None
