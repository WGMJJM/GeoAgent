from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.loop import AgentLoop
from app.core.models import AgentRequest, AgentResultStatus, Dataset, DatasetKind, Message, RunStatus
from app.execution.tools import ToolExecutor, ToolRegistry
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.observability import TraceRecorder
from app.state import StateStore
from app.tools.gis import register_gis_tools


class SequenceAdapter(ModelAdapter):
    supports_tools = True

    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class DatasetView:
    def __init__(self, rows: list[Dataset]) -> None:
        self.rows = rows

    def list(self, kind=None):
        return [row for row in self.rows if kind is None or row.kind is kind]

    def resolve(self, identifier, **_kwargs):
        return next((row for row in self.rows if identifier in {row.id, row.name}), None)


def _loop(tmp_path, adapter: ModelAdapter | None, datasets: list[Dataset] | None = None):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    registry = ToolRegistry()
    register_gis_tools(registry)
    trace = TraceRecorder(store)
    executor = ToolExecutor(registry, store, trace)
    settings = SimpleNamespace(max_agent_turns=6, max_tool_calls=8, max_tokens=256)
    dataset_view = DatasetView(datasets or [])
    loop = AgentLoop(
        store,
        registry,
        executor,
        trace,
        settings,
        lambda _profile: adapter,
        lambda _user: {"registry": dataset_view},
    )
    return store, loop


async def _run(loop: AgentLoop, store: StateStore, text: str):
    conversation = store.create_conversation("测试")
    request = AgentRequest(conversation_id=conversation.id, user_input=text)
    store.save_message(Message(conversation_id=conversation.id, role="user", content=text))
    prepared = await loop.prepare_request(request)
    return request, await loop.run(request, prepared=prepared)


@pytest.mark.asyncio
async def test_dataset_list_uses_tool_result_in_same_model_loop(tmp_path):
    dataset = Dataset(id="ds_roads", name="roads", kind=DatasetKind.VECTOR, path="roads.gpkg", format="GPKG")
    adapter = SequenceAdapter(
        ModelResponse(
            content="我先查询已登记数据。",
            tool_calls=[{"id": "call_list", "function": {"name": "dataset.list", "arguments": "{}"}}],
        ),
        ModelResponse(content="当前有 1 个数据集 roads。"),
    )
    store, loop = _loop(tmp_path, adapter, [dataset])
    request, result = await _run(loop, store, "查看已上传的数据")

    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "当前有 1 个数据集 roads。"
    assert store.get_run(result.trace_id).status is RunStatus.COMPLETED
    assert store.get_run(result.trace_id).tool_call_count == 1
    assert "roads" in adapter.requests[1].messages[-1]["content"]
    assert all(tool["function"]["name"] in {"dataset.list", "dataset.inspect", "agent.ask_user", "conversation.search_history"} for tool in adapter.requests[0].tools)


@pytest.mark.asyncio
async def test_plain_question_goes_directly_to_the_same_model(tmp_path):
    adapter = SequenceAdapter(ModelResponse(content="栅格数据以规则网格组织像元。"))
    store, loop = _loop(tmp_path, adapter)
    _, result = await _run(loop, store, "什么是栅格数据？")

    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "栅格数据以规则网格组织像元。"
    assert len(adapter.requests) == 1
    assert adapter.requests[0].tools


@pytest.mark.asyncio
async def test_ask_user_persists_waiting_state_and_checkpoint(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(
            tool_calls=[{"id": "call_question", "function": {"name": "agent.ask_user", "arguments": '{"question":"请确认要检查哪个图层？"}'}}]
        )
    )
    store, loop = _loop(tmp_path, adapter)
    _, result = await _run(loop, store, "检查这个图层")

    assert result.status is AgentResultStatus.BLOCKED
    assert result.error == "WAITING_USER"
    assert store.get_run(result.trace_id).status is RunStatus.WAITING_USER
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert checkpoint is not None
    assert checkpoint.state["request"]["user_input"] == "检查这个图层"
    assert checkpoint.state["protocol_messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_missing_model_fails_honestly_without_running_tools(tmp_path):
    store, loop = _loop(tmp_path, None)
    _, result = await _run(loop, store, "查看数据")

    assert result.status is AgentResultStatus.FAILED
    assert result.error == "MODEL_NOT_CONFIGURED"
    assert store.get_run(result.trace_id).status is RunStatus.FAILED


@pytest.mark.asyncio
async def test_tool_budget_allows_final_answer_from_last_observation(tmp_path):
    dataset = Dataset(id="ds_roads", name="roads", kind=DatasetKind.VECTOR, path="roads.gpkg", format="GPKG")
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "call_list", "function": {"name": "dataset.list", "arguments": "{}"}}]),
        ModelResponse(content="已到工具上限；查询结果中有 roads 数据集。"),
    )
    store, loop = _loop(tmp_path, adapter, [dataset])
    loop.settings.max_tool_calls = 1
    _, result = await _run(loop, store, "列出数据集")

    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "已到工具上限；查询结果中有 roads 数据集。"
    assert adapter.requests[1].tools == []


@pytest.mark.asyncio
async def test_waiting_run_resumes_from_checkpoint_with_user_reply(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "call_question", "function": {"name": "agent.ask_user", "arguments": '{"question":"请确认要检查哪个图层？"}'}}]),
        ModelResponse(content="收到，后续按道路图层处理。"),
    )
    store, loop = _loop(tmp_path, adapter)
    request, waiting = await _run(loop, store, "检查这个图层")
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    store.save_message(Message(conversation_id=request.conversation_id, role="user", content="道路图层"))
    resumed_request = request.model_copy(update={"user_input": "道路图层"})
    resumed = await loop.run(
        resumed_request,
        prepared=loop.prepare_resume(resumed_request, store.get_run(waiting.trace_id)),
        resume_from=checkpoint,
        continuation={"type": "user_input", "content": "道路图层"},
    )

    assert resumed.status is AgentResultStatus.SUCCESS
    assert resumed.trace_id == waiting.trace_id
    assert "道路图层" in adapter.requests[1].messages[-1]["content"]


@pytest.mark.asyncio
async def test_model_can_search_original_messages_on_demand(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(
            tool_calls=[
                {
                    "id": "call_history",
                    "function": {"name": "conversation.search_history", "arguments": '{"query":"道路缓冲距离"}'},
                }
            ]
        ),
        ModelResponse(content="历史记录里提到 500 米。"),
    )
    store, loop = _loop(tmp_path, adapter)
    conversation = store.create_conversation("历史检索")
    store.save_message(Message(conversation_id=conversation.id, role="user", content="道路缓冲距离采用 500 米。"))
    request = AgentRequest(conversation_id=conversation.id, user_input="之前的距离是多少？")
    store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))
    prepared = await loop.prepare_request(request)

    result = await loop.run(request, prepared=prepared)

    assert result.status is AgentResultStatus.SUCCESS
    assert "500 米" in adapter.requests[1].messages[-1]["content"]
    assert store.get_run(result.trace_id).tool_call_count == 1
