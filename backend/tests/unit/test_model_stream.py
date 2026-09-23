from types import SimpleNamespace

import pytest

from app.models import ModelRequest
from app.models.config import ModelConfig
from app.models.providers.openai_compatible import OpenAICompatibleAdapter


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

    async def create(self, **kwargs):
        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


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
async def test_openai_stream_emits_terminal_chunk_when_provider_closes_without_finish_reason():
    adapter = _adapter([SimpleNamespace(model="fake", usage=None, choices=[_choice(content="尾部已关闭")])])

    chunks = [chunk async for chunk in adapter.stream(ModelRequest(messages=[]))]

    assert chunks[-1].done is True
    assert chunks[-1].finish_reason is None
    assert "尾部已关闭" in "".join(chunk.content for chunk in chunks)
