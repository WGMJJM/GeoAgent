"""模型配置。"""

from pathlib import Path
from pydantic import BaseModel, Field, model_validator

from app.core.models import ReasoningEffort
from app.core.tokens import DEFAULT_TOKENIZER_FILE


class ModelConfig(BaseModel):
    provider: str = "openai-compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    tokenizer_file: Path = DEFAULT_TOKENIZER_FILE
    timeout_seconds: int = Field(default=90, ge=1)
    temperature: float = Field(default=0.1, ge=0, le=2)
    supports_stream: bool = True
    supports_tools: bool = True
    supports_json_object: bool = True
    supports_json_schema: bool = False
    reasoning_efforts: list[ReasoningEffort] = Field(default_factory=list)
    default_reasoning_effort: ReasoningEffort | None = None


class ModelProfile(BaseModel):
    """启动时预加载的模型配置；API 返回时只暴露脱敏后的字段。"""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    provider: str = "openai-compatible"
    base_url: str | None = None
    api_key: str | None = None
    model: str = Field(min_length=1)
    tokenizer_file: Path | None = None
    timeout_seconds: int = Field(default=90, ge=1)
    temperature: float = Field(default=0.1, ge=0, le=2)
    supports_stream: bool = True
    supports_tools: bool = True
    supports_json_object: bool = True
    supports_json_schema: bool = False
    reasoning_efforts: list[ReasoningEffort] = Field(default_factory=list)
    default_reasoning_effort: ReasoningEffort | None = None
    default: bool = False

    @model_validator(mode="after")
    def validate_reasoning_default(self):
        if self.default_reasoning_effort and self.default_reasoning_effort not in self.reasoning_efforts:
            raise ValueError("default_reasoning_effort 必须包含在 reasoning_efforts 中")
        return self

    def as_config(self, *, tokenizer_file: Path = DEFAULT_TOKENIZER_FILE) -> ModelConfig:
        return ModelConfig(
            provider=self.provider,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            tokenizer_file=self.tokenizer_file or tokenizer_file,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
            supports_stream=self.supports_stream,
            supports_tools=self.supports_tools,
            supports_json_object=self.supports_json_object,
            supports_json_schema=self.supports_json_schema,
            reasoning_efforts=self.reasoning_efforts,
            default_reasoning_effort=self.default_reasoning_effort,
        )
