"""GeoAgent 配置。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.tokens import DEFAULT_TOKENIZER_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "backend" / ".env"
DEFAULT_SUMMARY_RECENT_MESSAGES = 16
DEFAULT_SUMMARY_TRIGGER_MESSAGES = 24
DEFAULT_SUMMARY_TRIGGER_TOKENS = 51200
DEFAULT_SUMMARY_MESSAGE_MAX_CHARS = 10000
DEFAULT_EMERGENCY_RECENT_MESSAGES = 8
DEFAULT_CONVERSATION_TOOL_INDEX_LIMIT = 8


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEOAGENT_", env_file=ENV_FILE, extra="ignore", protected_namespaces=())

    root: Path = Field(default=PROJECT_ROOT)
    database: Path = Field(default=Path("state/geoagent.sqlite3"))
    workspace: Path = Field(default=Path("workspace"))
    skills_directory: Path = Field(default=Path("backend/skills"))
    default_crs: str = "EPSG:3857"
    max_agent_turns: int = Field(default=20, ge=1)
    max_empty_response_retries: int = Field(default=1, ge=0)
    max_tool_retries: int = Field(default=2, ge=0)
    max_model_retries: int = Field(default=1, ge=0)
    max_run_retries: int = Field(default=6, ge=0)
    retry_delay_seconds: float = Field(default=1, ge=0)
    retry_max_delay_seconds: float = Field(default=8, ge=0)
    max_tool_calls: int = Field(default=40, ge=1)
    max_subagents: int = Field(default=5, ge=1, le=20)
    max_parallel_agents: int = Field(default=3, ge=1, le=20)
    max_preview_features: int = Field(default=200, ge=1, le=1000)
    max_preview_fields: int = Field(default=32, ge=1, le=128)
    max_preview_property_length: int = Field(default=160, ge=16, le=2000)
    max_tokens: int = Field(default=12800, ge=1)
    completion_review_enabled: bool = True
    completion_review_max_tokens: int = Field(default=3200, ge=1)
    completion_review_timeout_seconds: float = Field(default=20, gt=0)
    tokenizer_file: Path = DEFAULT_TOKENIZER_FILE
    model_input_tokens: int = Field(default=128000, ge=1)
    summary_recent_messages: int = Field(default=DEFAULT_SUMMARY_RECENT_MESSAGES, ge=1)
    summary_trigger_messages: int = Field(default=DEFAULT_SUMMARY_TRIGGER_MESSAGES, ge=1)
    summary_trigger_tokens: int = Field(default=DEFAULT_SUMMARY_TRIGGER_TOKENS, ge=1)
    summary_message_max_chars: int = Field(default=DEFAULT_SUMMARY_MESSAGE_MAX_CHARS, ge=1)
    emergency_recent_messages: int = Field(default=DEFAULT_EMERGENCY_RECENT_MESSAGES, ge=1)
    tool_result_recent_full: int = Field(default=16, ge=1)
    conversation_tool_index_limit: int = Field(default=DEFAULT_CONVERSATION_TOOL_INDEX_LIMIT, ge=1)
    tool_result_emergency_fraction: float = Field(default=0.5, gt=0, le=1)
    tool_context_tokens: int = Field(default=25600, ge=1)
    tool_context_max_cards: int = Field(default=8, ge=1)
    tool_search_regex_results: int = Field(default=2, ge=1)
    tool_search_chinese_results: int = Field(default=1, ge=1)
    tool_search_english_results: int = Field(default=3, ge=1)
    max_execution_seconds: int = Field(default=300, ge=1)
    tool_timeout_seconds: int = Field(default=120, ge=1)
    enable_unsafe_python: bool = False
    enable_arcpy: bool = True
    arcpy_executable: Path | None = None
    arcpy_cache: Path = Field(default=Path("state/arcpy"))
    mcp_config: Path | None = None
    geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    model_profiles: str | None = None
    additional_model_profiles: str | None = None
    model_reasoning_config: str | None = None
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

    @property
    def skills_path(self) -> Path:
        return self._absolute(self.skills_directory)

    @property
    def arcpy_cache_path(self) -> Path:
        return self._absolute(self.arcpy_cache)

    @property
    def mcp_config_path(self) -> Path | None:
        return self._absolute(self.mcp_config) if self.mcp_config is not None else None

    def _absolute(self, value: Path) -> Path:
        return value if value.is_absolute() else (self.root / value).resolve()
