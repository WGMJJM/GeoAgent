"""MCP 连接声明；能力简介由配置提供，不由模型猜测。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import ENV_FILE
from app.core.models import RiskLevel


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    enabled: bool = False
    description: str = Field(min_length=1)
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    url: str | None = None
    env_refs: dict[str, str] = Field(default_factory=dict)
    header_refs: dict[str, str] = Field(default_factory=dict)
    allowed_users: list[str] | None = None
    risk_level: RiskLevel = RiskLevel.EXTERNAL
    tool_risks: dict[str, RiskLevel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command or self.url or self.header_refs:
                raise ValueError("stdio 需要 command，不能配置 url 或 header_refs")
        elif not self.url or self.command or self.args or self.cwd or self.env_refs:
            raise ValueError("streamable_http 需要 url，不能配置本地进程参数")
        return self

    @property
    def scope(self) -> str:
        return f"mcp.{self.id}.execute"

    @property
    def environment(self) -> str:
        return f"mcp.{self.id}"

    def resolve_refs(self, refs: dict[str, str]) -> dict[str, str]:
        # 只读取声明的凭据，不将配置中的密钥转存到工具定义或 Checkpoint。
        environment = {**dotenv_values(ENV_FILE), **os.environ}
        resolved = {name: environment[variable] for name, variable in refs.items()}
        if any(value is None for value in resolved.values()):
            raise ValueError("MCP 凭据环境变量未赋值")
        return resolved


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    servers: list[MCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_servers(self) -> MCPConfig:
        identifiers = [server.id for server in self.servers]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("MCP 服务 id 不能重复")
        return self


def load_mcp_config(path: Path | None) -> MCPConfig:
    return MCPConfig() if path is None else MCPConfig.model_validate_json(path.read_text(encoding="utf-8-sig"))
