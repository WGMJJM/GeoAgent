import asyncio
from types import SimpleNamespace

import pytest

from app.models import ModelRequest
from app.models.config import ModelConfig
from app.models.providers.openai_compatible import OpenAICompatibleAdapter
from app.models.providers.openai_responses import OpenAIResponsesAdapter


def _tool_definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "测试工具",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _choice(*, finish_reason=None, content="", tool_calls=None):
    return SimpleNamespace(
        finish_reason=finish_reason,
        delta=SimpleNamespace(content=content, tool_calls=tool_calls or []),
    )


def _tool_fragment(*, index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.requests = []
        self.closed = False

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        owner = self

        class Stream:
            async def __aiter__(self):
                for chunk in owner.chunks:
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield chunk

            async def close(self):
                owner.closed = True

        return Stream()


def _adapter(chunks):
    adapter = object.__new__(OpenAICompatibleAdapter)
    adapter.config = ModelConfig(model="fake", timeout_seconds=1)
    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(chunks)))
    return adapter


@pytest.mark.asyncio
async def test_stream_restores_aliased_tool_name_after_fragment_assembly():
    adapter = _adapter([
        SimpleNamespace(
            model="fake",
            usage=None,
            choices=[_choice(tool_calls=[_tool_fragment(index=0, call_id="call-1", name="dataset__", arguments="{")])],
        ),
        SimpleNamespace(
            model="fake",
            usage=None,
            choices=[_choice(finish_reason="tool_calls", tool_calls=[_tool_fragment(index=0, name="inspect", arguments="}")])],
        ),
    ])

    request = ModelRequest(
        messages=[{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "old", "type": "function", "function": {"name": "dataset.inspect", "arguments": "{}"}}],
        }],
        tools=[_tool_definition("dataset.inspect")],
    )
    chunks = [chunk async for chunk in adapter.stream(request)]
    sent = adapter.client.chat.completions.requests[0]

    assert sent["tools"][0]["function"]["name"] == "dataset__inspect"
    assert sent["messages"][0]["tool_calls"][0]["function"]["name"] == "dataset__inspect"
    assert chunks[-1].tool_calls[0]["function"]["name"] == "dataset.inspect"


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [(123, 45)])
async def test_stream_reads_usage_packet_after_finish_without_buffering_text(tokens):
    adapter = _adapter([
        SimpleNamespace(model="fake", usage=None, choices=[_choice(content="第一段")]),
        SimpleNamespace(model="fake", usage=None, choices=[_choice(content="第二段", finish_reason="stop")]),
        SimpleNamespace(model="fake", usage=SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1]), choices=[]),
    ])
    stream = adapter.stream(ModelRequest(messages=[], response_format={"type": "json_object"}))
    assert (await anext(stream)).content == "第一段"
    assert (await anext(stream)).content == "第二段"
    terminal = await anext(stream)
    await stream.aclose()

    assert terminal.done and terminal.finish_reason == "stop"
    assert (terminal.input_tokens, terminal.output_tokens) == tokens
    completions = adapter.client.chat.completions
    assert completions.closed
    assert len(completions.requests) == 1
    assert completions.requests[0]["stream_options"] == {"include_usage": True}
    assert completions.requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_stream_keeps_interleaved_tool_arguments_separate():
    adapter = _adapter([
        SimpleNamespace(model="fake", usage=None, choices=[_choice(tool_calls=[
            _tool_fragment(index=0, call_id="a", name="dataset.inspect", arguments='{"dataset_id":"'),
            _tool_fragment(index=1, call_id="b", name="dataset.inspect", arguments='{"dataset_id":"'),
        ])]),
        SimpleNamespace(model="fake", usage=None, choices=[_choice(finish_reason="tool_calls", tool_calls=[
            _tool_fragment(index=1, arguments='raster"}'),
            _tool_fragment(index=0, arguments='roads"}'),
        ])]),
    ])
    chunks = [chunk async for chunk in adapter.stream(ModelRequest(messages=[]))]
    assert [call["id"] for call in chunks[-1].tool_calls] == ["a", "b"]
    assert [call["function"]["arguments"] for call in chunks[-1].tool_calls] == ['{"dataset_id":"roads"}', '{"dataset_id":"raster"}']


@pytest.mark.asyncio
async def test_stream_timeout_closes_connection_and_never_emits_terminal(monkeypatch):
    adapter = _adapter([])
    adapter.config.timeout_seconds = 0.05
    closed = []

    class HangingStream:
        async def __aiter__(self):
            yield SimpleNamespace(model="fake", usage=None, choices=[_choice(content="部分")])
            await asyncio.Event().wait()

        async def close(self):
            closed.append(True)

    async def create(**_kwargs):
        return HangingStream()

    monkeypatch.setattr(adapter.client.chat.completions, "create", create)
    stream = adapter.stream(ModelRequest(messages=[]))
    assert (await anext(stream)).content == "部分"
    with pytest.raises(TimeoutError):
        await anext(stream)
    assert closed == [True]


@pytest.mark.asyncio
async def test_responses_adapter_converts_tools_history_reasoning_and_json_format():
    requests = []
    response = SimpleNamespace(
        model="gpt-6-sol",
        status="completed",
        incomplete_details=None,
        output_text="",
        output=[SimpleNamespace(type="function_call", call_id="call-new", name="dataset__inspect", arguments='{"dataset_id":"ds_1"}')],
        usage=SimpleNamespace(input_tokens=321, output_tokens=45),
    )

    async def create(**kwargs):
        requests.append(kwargs)
        return response

    adapter = object.__new__(OpenAIResponsesAdapter)
    adapter.config = ModelConfig(
        model="gpt-6-sol",
        wire_api="responses",
        reasoning_efforts=["low", "medium", "high", "xhigh", "max"],
    )
    adapter.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = await adapter.complete(ModelRequest(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-old", "type": "function", "function": {"name": "dataset.inspect", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-old", "content": "已完成"},
        ],
        tools=[_tool_definition("dataset.inspect")],
        response_format={"type": "json_object"},
        reasoning_effort="xhigh",
    ))

    sent = requests[0]
    assert sent["tools"][0]["name"] == "dataset__inspect"
    assert "function" not in sent["tools"][0]
    assert sent["input"][1] == {"type": "function_call", "call_id": "call-old", "name": "dataset__inspect", "arguments": "{}"}
    assert sent["input"][2] == {"type": "function_call_output", "call_id": "call-old", "output": "已完成"}
    assert sent["reasoning"] == {"effort": "xhigh"}
    assert sent["text"] == {"format": {"type": "json_object"}}
    assert "temperature" not in sent
    assert result.tool_calls[0]["function"]["name"] == "dataset.inspect"
    assert (result.input_tokens, result.output_tokens, result.finish_reason) == (321, 45, "tool_calls")


@pytest.mark.asyncio
async def test_responses_adapter_streams_text_then_publishes_terminal_usage():
    final = SimpleNamespace(
        model="gpt-6-sol",
        status="completed",
        incomplete_details=None,
        output_text="你好",
        output=[],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="你"),
        SimpleNamespace(type="response.output_text.delta", delta="好"),
        SimpleNamespace(type="response.completed", response=final),
    ]

    class Stream:
        closed = False

        async def __aiter__(self):
            for event in events:
                yield event

        async def close(self):
            self.closed = True

    stream = Stream()
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return stream

    adapter = object.__new__(OpenAIResponsesAdapter)
    adapter.config = ModelConfig(model="gpt-6-sol", wire_api="responses", timeout_seconds=1)
    adapter.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    chunks = [chunk async for chunk in adapter.stream(ModelRequest(messages=[{"role": "user", "content": "你好"}]))]

    assert "".join(chunk.content for chunk in chunks) == "你好"
    assert requests[0]["stream"] is True
    assert chunks[-1].done and chunks[-1].finish_reason == "stop"
    assert (chunks[-1].input_tokens, chunks[-1].output_tokens) == (12, 3)
    assert stream.closed
