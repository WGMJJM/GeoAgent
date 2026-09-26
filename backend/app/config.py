"""GeoAgent 配置。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "backend" / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEOAGENT_", env_file=ENV_FILE, extra="ignore", protected_namespaces=())

    root: Path = Field(default=PROJECT_ROOT)
    database: Path = Field(default=Path("state/geoagent.sqlite3"))
    workspace: Path = Field(default=Path("workspace"))
    default_crs: str = "EPSG:3857"
    max_agent_turns: int = Field(default=20, ge=1)
    max_tool_calls: int = Field(default=40, ge=1)
    max_subagents: int = Field(default=5, ge=1, le=20)
    max_parallel_agents: int = Field(default=3, ge=1, le=20)
    max_preview_features: int = Field(default=200, ge=1, le=1000)
    max_preview_fields: int = Field(default=32, ge=1, le=128)
    max_preview_property_length: int = Field(default=160, ge=16, le=2000)
    max_tokens: int = Field(default=3200, ge=1)
    tool_context_tokens: int = Field(default=2500, ge=1)
    tool_context_max_cards: int = Field(default=8, ge=1)
    max_execution_seconds: int = Field(default=300, ge=1)
    tool_timeout_seconds: int = Field(default=120, ge=1)
    enable_unsafe_python: bool = False
    model_profiles: str | None = None
    auth_cookie_name: str = "geoagent_session"
    auth_cookie_secure: bool = False
    auth_session_ttl_hours: int = Field(default=168, ge=1)
    bootstrap_user: str = ""
    bootstrap_password: str = ""
    bootstrap_email: str | None = None
    bootstrap_display_name: str | None = None

    @property
    def database_path(self) -> Path:
        return self._absolute(self.database)

    @property
    def workspace_path(self) -> Path:
        return self._absolute(self.workspace)

    def _absolute(self, value: Path) -> Path:
        return value if value.is_absolute() else (self.root / value).resolve()
