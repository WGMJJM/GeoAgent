"""MCP 服务生命周期及宿主授权；不接收模型提供的连接配置。"""

from __future__ import annotations

import asyncio

from app.auth.policy import PermissionPolicy, ToolDiscoveryContext
from app.execution.tools import ToolRegistry

from .config import MCPConfig
from .provider import MCPProvider


class MCPManager:
    def __init__(self, config: MCPConfig, registry: ToolRegistry, *, timeout_seconds: int) -> None:
        self.providers = tuple(
            MCPProvider(server, registry, timeout_seconds=timeout_seconds)
            for server in config.servers if server.enabled
        )

    async def start(self) -> None:
        await asyncio.gather(*(provider.start() for provider in self.providers))

    async def close(self) -> None:
        await asyncio.gather(*(provider.aclose() for provider in self.providers))

    def granted_scopes(self, user_id: str | None) -> frozenset[str]:
        return frozenset(
            provider.config.scope for provider in self.providers
            if user_id and (provider.config.allowed_users is None or user_id in provider.config.allowed_users)
        )

    def available_envs(self) -> frozenset[str]:
        return frozenset(provider.config.environment for provider in self.providers if provider.available)

    def capabilities(self, context: ToolDiscoveryContext) -> list[dict[str, str]]:
        policy = PermissionPolicy()
        return [
            {"id": provider.config.id, "description": provider.config.description}
            for provider in self.providers
            if any(policy.is_discoverable(metadata, context) for metadata in provider.summaries())
        ]
