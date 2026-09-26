"""LLM Provider 的最小适配协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from app.core.tokens import estimate_tokens


class ModelRequest(BaseModel):
    """一次模型请求；``max_tokens`` 表示模型输出上限，不是输入上下文预算。"""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int = 3200
    response_format: dict[str, Any] | None = None


class ModelResponse(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    finish_reason: str | None = None


class ModelStreamChunk(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    finish_reason: str | None = None
    done: bool = False


class ModelAdapter(ABC):
    supports_stream = False
    supports_tools = False
    supports_json_object = False
    supports_structured_output = False
    supports_json_schema = False

    def count_tokens(self, value: str) -> int:
        """本地预算分词；适配器可配置对应模型的词表，不进行网络请求。"""
        return estimate_tokens(value)

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """执行一次模型请求。"""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        """流式执行一次模型请求；未实现流式接口的适配器退化为单片段。"""
        response = await self.complete(request)
        yield ModelStreamChunk(
            content=response.content,
            tool_calls=response.tool_calls,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            finish_reason=response.finish_reason or "stop",
            done=True,
        )

    async def close(self) -> None:
        return None
