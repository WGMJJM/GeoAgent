import asyncio
from types import SimpleNamespace

import pytest

from app.models import ModelRequest, ModelResponse
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


class _ToolCall:
    def __init__(self, name):
        self.name = name

    def model_dump(self):
        return {"id": "call-1", "type": "function", "function": {"name": self.name, "arguments": "{}"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, SimpleNamespace(prompt_tokens=0, completion_tokens=0), SimpleNamespace(prompt_tokens=123, completion_tokens=45)])
async def test_complete_reads_existing_response_usage_without_an_extra_request(usage):
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(model="fake", usage=usage, choices=[SimpleNamespace(
            message=SimpleNamespace(content="完成", tool_calls=[]), finish_reason="stop",
        )])

    adapter = object.__new__(OpenAICompatibleAdapter)
    adapter.config = ModelConfig(model="fake")
    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = await adapter.complete(ModelRequest(messages=[{"role": "user", "content": "你好"}]))
    assert len(requests) == 1
    assert response.input_tokens == (usage.prompt_tokens if usage else None)
    assert response.output_tokens == (usage.completion_tokens if usage else None)


@pytest.mark.asyncio
async def test_reasoning_effort_is_forwarded_only_for_declared_levels():
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(model="fake", usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content="完成", tool_calls=[]), finish_reason="stop",
        )])

    adapter = object.__new__(OpenAICompatibleAdapter)
    adapter.config = ModelConfig(model="fake", reasoning_efforts=["low", "medium", "high", "xhigh", "max"])
    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    await adapter.complete(ModelRequest(messages=[], reasoning_effort="xhigh"))
    assert requests[0]["reasoning_effort"] == "xhigh"

    with pytest.raises(ValueError, match="不支持思考程度"):
        adapter.config = ModelConfig(model="fake")
        await adapter.complete(ModelRequest(messages=[], reasoning_effort="high"))


@pytest.mark.asyncio
async def test_complete_maps_provider_safe_tool_names_and_restores_internal_names():
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(model="fake", usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[_ToolCall("dataset__inspect")]),
            finish_reason="tool_calls",
        )])

    adapter = object.__new__(OpenAICompatibleAdapter)
    adapter.config = ModelConfig(model="fake")
    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    response = await adapter.complete(ModelRequest(
        messages=[{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "old", "type": "function", "function": {"name": "dataset.inspect", "arguments": "{}"}},
                {"id": "older", "type": "function", "function": {"name": "raster.slope", "arguments": "{}"}},
            ],
        }],
        tools=[_tool_definition("dataset.inspect")],
    ))

    assert requests[0]["tools"][0]["function"]["name"] == "dataset__inspect"
    assert requests[0]["messages"][0]["tool_calls"][0]["function"]["name"] == "dataset__inspect"
    assert requests[0]["messages"][0]["tool_calls"][1]["function"]["name"] == "raster__slope"
    assert response.tool_calls[0]["function"]["name"] == "dataset.inspect"


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
@pytest.mark.parametrize("finish_reason", ["stop", "length", "content_filter"])
async def test_openai_stream_publishes_protocol_terminal_reasons(finish_reason):
    adapter = _adapter([
        SimpleNamespace(model="fake", usage=None, choices=[_choice(content="正文")]),
        SimpleNamespace(model="fake", usage=None, choices=[_choice(finish_reason=finish_reason)]),
    ])

    chunks = [chunk async for chunk in adapter.stream(ModelRequest(messages=[]))]

    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == finish_reason
    assert "正文" == "".join(chunk.content for chunk in chunks)


@pytest.mark.asyncio
async def test_openai_stream_assembles_tool_call_fragments_at_finish_reason():
    adapter = _adapter([
        SimpleNamespace(
            model="fake",
            usage=None,
            choices=[_choice(tool_calls=[_tool_fragment(index=0, call_id="call-1", name="dataset.", arguments='{"dataset_id":"')])],
        ),
        SimpleNamespace(
            model="fake",
            usage=None,
            choices=[_choice(finish_reason="tool_calls", tool_calls=[_tool_fragment(index=0, name="inspect", arguments="roads" + '"}')])],
        ),
    ])

    chunks = [chunk async for chunk in adapter.stream(ModelRequest(messages=[]))]
    terminal = chunks[-1]

    assert terminal.done is True
    assert terminal.finish_reason == "tool_calls"
    assert terminal.tool_calls == [{"id": "call-1", "type": "function", "function": {"name": "dataset.inspect", "arguments": '{"dataset_id":"roads"}'}}]


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
async def test_openai_stream_emits_terminal_chunk_when_provider_closes_without_finish_reason():
    adapter = _adapter([SimpleNamespace(model="fake", usage=None, choices=[_choice(content="尾部已关闭")])])

    chunks = [chunk async for chunk in adapter.stream(ModelRequest(messages=[]))]

    assert chunks[-1].done is True
    assert chunks[-1].finish_reason is None
    assert "尾部已关闭" in "".join(chunk.content for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [(123, 45), (0, 0)])
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
@pytest.mark.parametrize("abort", ["consumer", "provider"])
async def test_stream_closes_connection_on_abort_without_retry(abort):
    adapter = _adapter([
        SimpleNamespace(model="fake", usage=None, choices=[_choice(content="部分正文")]),
        RuntimeError("连接中断"),
    ])
    stream = adapter.stream(ModelRequest(messages=[]))
    assert (await anext(stream)).content == "部分正文"
    if abort == "consumer":
        await stream.aclose()
    else:
        with pytest.raises(RuntimeError, match="连接中断"):
            await anext(stream)
    assert adapter.client.chat.completions.closed
    assert len(adapter.client.chat.completions.requests) == 1


@pytest.mark.asyncio
async def test_declared_non_stream_provider_still_uses_one_completion(monkeypatch):
    adapter = _adapter([])
    adapter.config.supports_stream = False
    requests = []

    async def complete(request):
        requests.append(request)
        return ModelResponse(content="完整回答", input_tokens=100, output_tokens=20)

    monkeypatch.setattr(adapter, "complete", complete)
    chunks = [chunk async for chunk in adapter.stream(ModelRequest(messages=[]))]
    assert len(requests) == len(chunks) == 1
    assert chunks[0].done and chunks[0].content == "完整回答"
    assert chunks[0].output_tokens == 20
    assert adapter.client.chat.completions.requests == []


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
