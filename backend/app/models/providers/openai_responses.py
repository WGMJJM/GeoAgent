"""OpenAI Responses API 适配器。"""

from __future__ import annotations

import asyncio
import ssl
from typing import Any

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from app.core.tokens import estimate_tokens
from app.models.adapter import ModelAdapter, ModelRequest, ModelResponse, ModelStreamChunk
from app.models.config import ModelConfig
from app.models.providers.openai_compatible import (
    _capability,
    _provider_tool_name,
    _reasoning_effort,
    _restore_tool_names,
)


class OpenAIResponsesAdapter(ModelAdapter):
    """在 GeoAgent 的模型协议与 Responses API 之间转换。"""

    def __init__(self, config: ModelConfig) -> None:
        if not config.model or not config.model.strip():
            raise ValueError("Responses API 需要填写模型名称。")
        self.config = config
        self.count_tokens("")
        self.timeout_seconds = config.timeout_seconds
        self.minimum_reasoning_effort = config.minimum_reasoning_effort
        self.completion_review_extra_body = config.completion_review_extra_body
        self.supports_stream = config.supports_stream
        self.supports_tools = config.supports_tools
        self.supports_json_object = config.supports_json_object
        self.supports_structured_output = config.supports_json_object
        self.supports_json_schema = config.supports_json_schema
        base_url = config.base_url.strip() if config.base_url and config.base_url.strip() else None
        http_client = DefaultAsyncHttpxClient(verify=ssl.create_default_context()) if config.use_system_certificates else None
        self.client = AsyncOpenAI(api_key=config.api_key or "local", base_url=base_url, timeout=config.timeout_seconds, max_retries=0, http_client=http_client)

    def count_tokens(self, value: str) -> int:
        return estimate_tokens(value, self.config.tokenizer_file)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs, alias_to_name = _response_request(self, request)
        async with asyncio.timeout(self.config.timeout_seconds):
            response = await self.client.responses.create(**kwargs)
        return _model_response(response, alias_to_name)

    async def stream(self, request: ModelRequest):
        if not _capability(self, "supports_stream", True):
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
            return

        kwargs, alias_to_name = _response_request(self, request)
        stream = None
        pending_calls: dict[int, dict[str, Any]] = {}
        terminal_sent = False
        async with asyncio.timeout(self.config.timeout_seconds):
            stream = await self.client.responses.create(**kwargs, stream=True)
            try:
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            yield ModelStreamChunk(content=delta)
                    elif event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if getattr(item, "type", None) == "function_call":
                            pending_calls[getattr(event, "output_index", len(pending_calls))] = _tool_call(item)
                    elif event_type == "response.function_call_arguments.delta":
                        index = getattr(event, "output_index", len(pending_calls))
                        call = pending_calls.setdefault(index, _empty_tool_call(getattr(event, "item_id", "")))
                        call["function"]["arguments"] += getattr(event, "delta", "")
                    elif event_type == "response.function_call_arguments.done":
                        index = getattr(event, "output_index", len(pending_calls))
                        call = pending_calls.setdefault(index, _empty_tool_call(getattr(event, "item_id", "")))
                        call["function"]["name"] = getattr(event, "name", "")
                        call["function"]["arguments"] = getattr(event, "arguments", "")
                    elif event_type in {"response.completed", "response.incomplete"}:
                        response = getattr(event, "response")
                        result = _model_response(response, alias_to_name, content="")
                        yield ModelStreamChunk(**result.model_dump(), done=True)
                        terminal_sent = True
                        break
                    elif event_type in {"response.failed", "error"}:
                        raise RuntimeError(_response_error(event))
            finally:
                await stream.close()

        if not terminal_sent:
            yield ModelStreamChunk(
                tool_calls=_restore_tool_names([pending_calls[index] for index in sorted(pending_calls)], alias_to_name),
                done=True,
            )

    async def close(self) -> None:
        await self.client.close()


def _response_request(adapter: OpenAIResponsesAdapter, request: ModelRequest) -> tuple[dict[str, Any], dict[str, str]]:
    tools, name_to_alias, alias_to_name = _response_tools(request.tools)
    effort = _reasoning_effort(adapter.config, request)
    kwargs: dict[str, Any] = {
        "model": adapter.config.model,
        "input": _response_input(request.messages, name_to_alias),
        "max_output_tokens": request.max_tokens,
    }
    if request.extra_body is not None:
        kwargs["extra_body"] = request.extra_body
    if tools and _capability(adapter, "supports_tools", True):
        kwargs["tools"] = tools
    if request.response_format and (
        _capability(adapter, "supports_json_object", True)
        or _capability(adapter, "supports_json_schema", False)
    ):
        kwargs["text"] = {"format": request.response_format}
    if effort:
        kwargs["reasoning"] = {"effort": effort}
    else:
        kwargs["temperature"] = request.temperature if request.temperature is not None else adapter.config.temperature
    return kwargs, alias_to_name


def _response_tools(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    converted: list[dict[str, Any]] = []
    name_to_alias: dict[str, str] = {}
    alias_to_name: dict[str, str] = {}
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        name = function["name"]
        alias = _provider_tool_name(name)
        existing = alias_to_name.get(alias)
        if existing is not None and existing != name:
            raise ValueError(f"工具名称映射冲突：{existing} 与 {name}")
        name_to_alias[name] = alias
        alias_to_name[alias] = name
        converted.append({
            "type": "function",
            "name": alias,
            "description": function.get("description"),
            "parameters": function.get("parameters", {}),
            "strict": function.get("strict"),
        })
    return converted, name_to_alias, alias_to_name


def _response_input(messages: list[dict[str, Any]], name_to_alias: dict[str, str]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            converted.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id", ""),
                "output": message.get("content", ""),
            })
            continue
        converted.append({"role": role, "content": message.get("content", "")})
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "")
                converted.append({
                    "type": "function_call",
                    "call_id": call.get("id", ""),
                    "name": name_to_alias.get(name, _provider_tool_name(name)),
                    "arguments": function.get("arguments", ""),
                })
    return converted


def _model_response(response: Any, alias_to_name: dict[str, str], *, content: str | None = None) -> ModelResponse:
    tool_calls = [
        _tool_call(item)
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "function_call"
    ]
    usage = getattr(response, "usage", None)
    return ModelResponse(
        content=getattr(response, "output_text", "") if content is None else content,
        tool_calls=_restore_tool_names(tool_calls, alias_to_name),
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        model=getattr(response, "model", None),
        finish_reason=_finish_reason(response, tool_calls),
    )


def _tool_call(item: Any) -> dict[str, Any]:
    return {
        "id": getattr(item, "call_id", ""),
        "type": "function",
        "function": {
            "name": getattr(item, "name", ""),
            "arguments": getattr(item, "arguments", ""),
        },
    }


def _empty_tool_call(call_id: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": "", "arguments": ""}}


def _finish_reason(response: Any, tool_calls: list[dict[str, Any]]) -> str | None:
    if tool_calls:
        return "tool_calls"
    status = getattr(response, "status", None)
    if status == "completed":
        return "stop"
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    return "length" if reason == "max_output_tokens" else reason or status


def _response_error(event: Any) -> str:
    response = getattr(event, "response", None)
    error = getattr(response, "error", None) or getattr(event, "error", None)
    return getattr(error, "message", None) or getattr(event, "message", None) or "Responses API 调用失败"
