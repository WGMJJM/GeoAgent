from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.context import (
    TOOL_HISTORY_PREFIX,
    TOOL_VISIBILITY_PREFIX,
    compact_model_input,
    model_input_tokens,
    narrow_model_input,
    prepare_model_messages,
)
from app.agent.loop import AgentLoop
from app.auth.approval import ApprovalService
from app.core.models import (
    AgentRequest,
    AgentResultStatus,
    Dataset,
    DatasetKind,
    Message,
    RiskLevel,
    RunStatus,
    ToolMetadata,
)
from app.core.tokens import estimate_tokens
from app.execution.tools import ToolExecutor, ToolRegistry
from app.memory import ConversationMemoryService
from app.models import ModelAdapter, ModelRequest, ModelResponse, ModelStreamChunk
from app.observability import TraceRecorder
from app.run.lifecycle import record_approval_decision
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
    settings = SimpleNamespace(max_agent_turns=6, max_tool_calls=8, max_tokens=256,
                               model_input_tokens=128000, tool_result_recent_full=16,
                               tool_result_emergency_fraction=0.5,
                               tool_context_tokens=12800, tool_context_max_cards=8,
                               emergency_recent_messages=8)
    dataset_view = DatasetView(datasets or [])
    loop = AgentLoop(
        store,
        registry,
        executor,
        trace,
        settings,
        lambda _profile: adapter,
        lambda user_id: {
            "registry": dataset_view,
            "inspector": object(),
            "vectors": object(),
            "rasters": object(),
            "workspace": object(),
            "user_id": user_id,
        },
        context_services={"conversation_memory": ConversationMemoryService(store)},
    )
    return store, loop


def _assert_tool_visibility(loop, request):
    messages = [item for item in request.messages if item["role"] == "system" and item["content"].startswith(TOOL_VISIBILITY_PREFIX)]
    assert len(messages) == 1
    visibility = json.loads(messages[0]["content"].removeprefix(TOOL_VISIBILITY_PREFIX))
    assert set(visibility) == {"callable", "cached"}
    assert visibility["callable"] == [item["function"]["name"] for item in request.tools]
    card_names = {item["name"] for item in visibility["cached"]}
    assert not (set(visibility["callable"]) & card_names)
    assert len(card_names) + sum(loop.registry.is_deferred(name) for name in visibility["callable"]) <= loop.settings.tool_context_max_cards
    tokens = estimate_tokens(json.dumps(request.tools, ensure_ascii=False, separators=(",", ":")))
    tokens += estimate_tokens(messages[0]["content"])
    assert tokens == loop._tool_context_tokens(request.tools, visibility["cached"])
    assert tokens <= loop.settings.tool_context_tokens
    return visibility


async def _run(loop: AgentLoop, store: StateStore, text: str, on_model_delta=None):
    conversation = store.create_conversation("测试", user_id="test-user")
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input=text)
    store.save_message(Message(conversation_id=conversation.id, role="user", content=text))
    prepared = await loop.prepare_request(request)
    return request, await loop.run(request, prepared=prepared, on_model_delta=on_model_delta)


@pytest.mark.asyncio
async def test_streamed_tool_batch_waits_for_terminal_and_does_not_duplicate_answer(tmp_path):
    fragments = []

    class StreamingAdapter(SequenceAdapter):
        async def complete(self, request):
            raise AssertionError("不能另发非流式请求")

        async def stream(self, request):
            self.requests.append(request)
            run = store.list_runs()[0]
            if len(self.requests) == 1:
                yield ModelStreamChunk(content="先查询数据。", done=True, finish_reason="tool_calls", input_tokens=100, output_tokens=10,
                                       tool_calls=[{"id": "list", "function": {"name": "dataset.list", "arguments": "{}"}}])
            else:
                assert store.get_run(run.id).tool_call_count == 1
                yield ModelStreamChunk(content="没有")
                yield ModelStreamChunk(content="数据。")
                yield ModelStreamChunk(done=True, finish_reason="stop", input_tokens=200, output_tokens=20)

    adapter = StreamingAdapter()
    store, loop = _loop(tmp_path, adapter)

    live_usage = []

    async def capture(content, token_usage):
        fragments.append(content)
        live_usage.append(token_usage)

    _, result = await _run(loop, store, "查看数据", capture)
    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "没有数据。"
    assert fragments == ["", "", "没有", "数据。"]
    assert live_usage[0].local_input_tokens > 0
    assert live_usage[-1].local_output_tokens > live_usage[0].local_output_tokens
    run = store.get_run(result.trace_id)
    assert run.token_usage.reported_input_tokens == 300
    assert run.token_usage.reported_output_tokens == 30
    assert run.token_usage.model_calls == 2
    events = store.list_events(run.id)
    assert sum(event.event_type == "ModelResponseStarted" for event in events) == 2
    preparing = next(index for index, event in enumerate(events) if event.event_type == "ToolPreparing")
    started = next(index for index, event in enumerate(events) if event.event_type == "ToolStarted")
    assert preparing < started
    assert events[preparing].payload == {"tools": ["dataset.list"], "tool_count": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "missing_terminal"])
async def test_incomplete_stream_does_not_execute_tools_or_report_success(tmp_path, failure):
    class BrokenAdapter(SequenceAdapter):
        async def stream(self, request):
            self.requests.append(request)
            yield ModelStreamChunk(content="未完成", tool_calls=[{"id": "list", "function": {"name": "dataset.list", "arguments": "{}"}}])
            if failure == "connection":
                raise RuntimeError("断流")

    adapter = BrokenAdapter()
    store, loop = _loop(tmp_path, adapter)
    _, result = await _run(loop, store, "查看数据")
    run = store.get_run(result.trace_id)
    assert result.status is AgentResultStatus.FAILED
    assert result.error == "MODEL_UNAVAILABLE"
    assert run.tool_call_count == 0
    assert run.token_usage is None
    assert len(adapter.requests) == 1


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
    assert {tool["function"]["name"] for tool in adapter.requests[0].tools} == {
        "tool.search",
        "dataset.list",
        "dataset.inspect",
        "agent.ask_user",
        "conversation.search_history",
    }
    for model_request in adapter.requests:
        assert _assert_tool_visibility(loop, model_request)["cached"] == []


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
@pytest.mark.parametrize("second_usage", [(200, 30), (0, 0), (None, None), (200, None)])
async def test_usage_counts_each_response_without_extra_model_requests(tmp_path, second_usage):
    first = ModelResponse(tool_calls=[{"id": "list", "function": {"name": "dataset.list", "arguments": "{}"}}],
                          input_tokens=100, output_tokens=20)
    second = ModelResponse(content="当前没有数据集。", input_tokens=second_usage[0], output_tokens=second_usage[1])
    adapter = SequenceAdapter(first, second)
    store, loop = _loop(tmp_path, adapter)
    seen = []

    async def capture(event):
        seen.append(event)

    loop.trace.bus.subscribe(capture)
    _, result = await _run(loop, store, "列出数据集")
    assert len(adapter.requests) == 2
    usage = store.get_run(result.trace_id).token_usage
    assert usage.model_calls == 2
    complete_usage = second_usage[0] is not None and second_usage[1] is not None
    assert usage.reported_calls == 1 + complete_usage
    assert usage.reported_input_tokens == 100 + (second_usage[0] if complete_usage else 0)
    assert usage.reported_output_tokens == 20 + (second_usage[1] if complete_usage else 0)
    assert usage.local_input_tokens == sum(adapter.count_tokens(json.dumps(
        {"messages": request.messages, "tools": request.tools}, ensure_ascii=False, separators=(",", ":"),
    )) for request in adapter.requests)
    assert usage.local_output_tokens == adapter.count_tokens(second.content) + adapter.count_tokens(json.dumps(first.tool_calls, ensure_ascii=False, separators=(",", ":")))
    events = [event for event in seen if event.event_type == "TokenUsageUpdated"]
    assert [event.payload["token_usage"]["model_calls"] for event in events] == [1, 2]
    assert events[-1].payload["token_usage"] == usage.model_dump()
    assert seen.index(events[-1]) < next(index for index, event in enumerate(seen) if event.event_type == "RunCompleted")


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
    assert _assert_tool_visibility(loop, adapter.requests[1]) == {"callable": [], "cached": []}


@pytest.mark.asyncio
async def test_waiting_run_resumes_from_checkpoint_with_user_reply(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "call_question", "function": {"name": "agent.ask_user", "arguments": '{"question":"请确认要检查哪个图层？"}'}}]),
        ModelResponse(content="收到，后续按道路图层处理。"),
    )
    store, loop = _loop(tmp_path, adapter)
    request, waiting = await _run(loop, store, "检查这个图层")
    assert store.get_run(waiting.trace_id).token_usage.model_calls == 1
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    store.save_message(Message(conversation_id=request.conversation_id, role="user", content="这是另一项新任务，不应混入旧 Run。"))
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
    assert "另一项新任务" not in str(adapter.requests[1].messages)
    assert store.get_run(resumed.trace_id).token_usage.model_calls == 2


@pytest.mark.asyncio
async def test_search_injects_deferred_tool_on_next_turn_and_executes_it(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "search", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}}]),
        ModelResponse(tool_calls=[{"id": "buffer", "function": {"name": "vector.buffer", "arguments": '{"dataset_id":"ds_roads","distance":25}'}}]),
        ModelResponse(content="缓冲区已生成。"),
    )
    store, loop = _loop(tmp_path, adapter, [Dataset(id="ds_roads", name="roads", kind=DatasetKind.VECTOR, path="roads.gpkg", format="GPKG")])
    calls: list[dict] = []
    metadata = loop.registry.get("vector.buffer").metadata
    loop.registry.unregister("vector.buffer")
    loop.registry.register(metadata, lambda arguments, _context: calls.append(arguments) or {"output": {"created": True}}, deferred=True)

    _, result = await _run(loop, store, "为道路生成缓冲区")

    names = [{tool["function"]["name"] for tool in item.tools} for item in adapter.requests]
    assert "vector.buffer" not in names[0]
    assert "vector.buffer" in names[1]
    assert len(names[0]) < len(loop.registry.names())
    assert result.status is AgentResultStatus.SUCCESS
    assert calls == [{"dataset_id": "ds_roads", "distance": 25}]
    assert store.get_run(result.trace_id).tool_call_count == 2
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert checkpoint.state["activated_tool_names"] == ["vector.buffer"]
    assert "vector.buffer" in _assert_tool_visibility(loop, adapter.requests[1])["callable"]
    assert not any(item["role"] == "system" for item in checkpoint.state["protocol_messages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_count", [1, 2])
@pytest.mark.parametrize("first_result", ["success", "failure", "invalid_arguments"])
async def test_unused_candidates_become_cards_and_used_schemas_survive_resume(tmp_path, selected_count, first_result):
    names = [f"test.candidate_{index}" for index in range(2)]

    def call(name, arguments, call_id):
        return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}

    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[call("tool.search", {"query": "primary_marker", "english_query": "secondary_marker"}, "find")]),
        ModelResponse(tool_calls=[call(name, {} if first_result == "invalid_arguments" else {"value": 1}, f"selected_{index}")
                                  for index, name in enumerate(names[:selected_count])]),
        ModelResponse(tool_calls=[call("agent.ask_user", {"question": "继续使用已选择工具吗？"}, "pause")]),
        ModelResponse(tool_calls=[call(names[0], {"value": 2}, "reuse")]),
        ModelResponse(content="已直接复用完整 Schema，未调用候选保留卡片。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []

    def execute(arguments, context):
        executed.append(context.call_id)
        if first_result == "failure" and arguments["value"] == 1:
            raise ValueError("模拟执行失败")
        return {"output": {"value": arguments["value"]}}

    for index, name in enumerate(names):
        loop.registry.register(ToolMetadata(name=name, description="primary_marker" if index == 0 else "secondary_marker",
                                           input_schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}),
                               execute, deferred=True)
    request, waiting = await _run(loop, store, "查找候选后只使用选中的工具")
    assert waiting.error == "WAITING_USER"
    assert set(names) <= set(_assert_tool_visibility(loop, adapter.requests[1])["callable"])
    selected = set(names[:selected_count])
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    assert set(checkpoint.state["used_tool_names"]) == selected
    assert set(checkpoint.state["activated_tool_names"]) == selected
    visibility = _assert_tool_visibility(loop, adapter.requests[2])
    assert selected <= set(visibility["callable"])
    assert set(names) - selected == {item["name"] for item in visibility["cached"]}
    observations = [json.loads(item["content"]) for item in adapter.requests[2].messages
                    if str(item.get("tool_call_id", "")).startswith("selected_")]
    assert {item["status"] for item in observations} == ({"SUCCESS"} if first_result == "success" else {"FAILED"})

    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(waiting.trace_id)),
                            resume_from=checkpoint, continuation={"type": "user_input", "content": "继续"})
    assert result.status is AgentResultStatus.SUCCESS
    assert executed == ([] if first_result == "invalid_arguments" else [f"{result.trace_id}:selected_{index}" for index in range(selected_count)]) + [f"{result.trace_id}:reuse"]
    assert len(adapter.requests) == 5
    assert store.get_run(result.trace_id).tool_call_count == selected_count + 3
    assert selected <= set(_assert_tool_visibility(loop, adapter.requests[3])["callable"])
    for model_request in adapter.requests:
        _assert_tool_visibility(loop, model_request)


@pytest.mark.asyncio
async def test_interrupted_tool_batch_keeps_schemas_until_all_saved_calls_finish(tmp_path):
    names = [f"test.batch_{index}" for index in range(2)]
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"batch_primary","english_query":"batch_secondary"}'}}]),
        ModelResponse(tool_calls=[{"id": f"execute_{index}", "function": {"name": name, "arguments": "{}"}} for index, name in enumerate(names[:2])]),
        ModelResponse(content="已恢复剩余调用，未用候选已降为卡片。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []
    for index, name in enumerate(names):
        loop.registry.register(ToolMetadata(name=name, description="batch_primary" if index == 0 else "batch_secondary", input_schema={"type": "object"}),
                               lambda _args, context: executed.append(context.call_id) or {"output": "ok"}, deferred=True)
    conversation = store.create_conversation("批次恢复", user_id="test-user")
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="调用两个候选")
    prepared = await loop.prepare_request(request)
    original = loop._save_checkpoint

    def interrupt_after_first_call(*args, **kwargs):
        original(*args, **kwargs)
        pending = kwargs.get("pending_tool_calls", [])
        if args[4] == "tool_observation" and pending and pending[0][0] == "execute_1":
            raise RuntimeError("模拟批次中断")

    loop._save_checkpoint = interrupt_after_first_call
    with pytest.raises(RuntimeError, match="批次中断"):
        await loop.run(request, prepared=prepared)
    checkpoint = store.latest_checkpoint(prepared.run.id)
    assert set(checkpoint.state["activated_tool_names"]) == set(names)
    assert checkpoint.state["used_tool_names"] == [names[0]]
    assert checkpoint.state["pending_tool_calls"][0][0] == "execute_1"
    loop._save_checkpoint = original
    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(prepared.run.id)),
                            resume_from=checkpoint, continuation={"type": "technical_resume"})
    assert result.status is AgentResultStatus.SUCCESS
    assert executed == [f"{result.trace_id}:execute_0", f"{result.trace_id}:execute_1"]
    assert len(adapter.requests) == 3
    assert store.get_run(result.trace_id).tool_call_count == 3
    visibility = _assert_tool_visibility(loop, adapter.requests[2])
    assert set(names[:2]) <= set(visibility["callable"])
    assert visibility["cached"] == []


@pytest.mark.asyncio
async def test_unused_card_restores_from_cache_without_research_or_same_batch_execution(tmp_path):
    names = [f"test.restore_{index}" for index in range(2)]
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"restore_primary","english_query":"restore_secondary"}'}}]),
        ModelResponse(tool_calls=[{"id": "first", "function": {"name": names[0], "arguments": "{}"}}]),
        ModelResponse(tool_calls=[
            {"id": "restore", "function": {"name": "tool.search", "arguments": json.dumps({"query": names[1]})}},
            {"id": "premature", "function": {"name": names[1], "arguments": "{}"}},
        ]),
        ModelResponse(tool_calls=[{"id": "second", "function": {"name": names[1], "arguments": "{}"}}]),
        ModelResponse(content="只恢复卡片，不重复语义检索。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []
    for index, name in enumerate(names):
        loop.registry.register(ToolMetadata(name=name, description="restore_primary" if index == 0 else "restore_secondary", input_schema={"type": "object"}),
                               lambda _args, context: executed.append(context.call_id) or {"output": "ok"}, deferred=True)
    searches = []
    original_search = loop.catalog.tool_search

    def record_search(arguments, context):
        searches.append(arguments)
        return original_search(arguments, context)

    loop.catalog.tool_search = record_search
    _, result = await _run(loop, store, "先使用一个工具，再恢复另一个候选")
    assert result.status is AgentResultStatus.SUCCESS
    assert searches == [{"query": "restore_primary", "english_query": "restore_secondary"}]
    assert {item["name"] for item in _assert_tool_visibility(loop, adapter.requests[2])["cached"]} == set(names[1:])
    observations = {item["tool_call_id"]: json.loads(item["content"]) for item in adapter.requests[3].messages if item["role"] == "tool"}
    assert observations["restore"]["output"]["source"] == "run_cache"
    assert observations["restore"]["output"]["already_callable"] is False
    assert observations["premature"]["error"]["code"] == "DEFERRED_TOOL_NOT_ACTIVE"
    assert executed == [f"{result.trace_id}:first", f"{result.trace_id}:second"]
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert set(checkpoint.state["used_tool_names"]) == set(names)
    assert set(checkpoint.state["activated_tool_names"]) == set(names)
    for model_request in adapter.requests:
        _assert_tool_visibility(loop, model_request)


@pytest.mark.asyncio
async def test_visible_schema_cache_query_reports_availability_without_search_or_execution(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}}]),
        ModelResponse(tool_calls=[{"id": "redundant", "function": {"name": "tool.search", "arguments": '{"query":"vector.buffer"}'}}]),
        ModelResponse(tool_calls=[{"id": "execute", "function": {"name": "vector.buffer", "arguments": '{"dataset_id":"ds_roads","distance":25}'}}]),
        ModelResponse(content="直接使用已提供 Schema 的工具。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []
    metadata = loop.registry.get("vector.buffer").metadata
    loop.registry.unregister("vector.buffer")
    loop.registry.register(metadata, lambda arguments, _context: executed.append(arguments) or {"output": {"checked": True}}, deferred=True)
    searches = []
    original_search = loop.catalog.tool_search

    def record_search(arguments, context):
        searches.append(arguments)
        return original_search(arguments, context)

    loop.catalog.tool_search = record_search
    _, result = await _run(loop, store, "复用缓冲工具")
    assert result.status is AgentResultStatus.SUCCESS
    assert searches == [{"query": "buffer"}]
    assert executed == [{"dataset_id": "ds_roads", "distance": 25}]
    assert store.get_run(result.trace_id).tool_call_count == 3
    observation = json.loads(next(item["content"] for item in adapter.requests[2].messages if item.get("tool_call_id") == "redundant"))
    assert observation["output"]["source"] == "run_cache"
    assert observation["output"]["already_callable"] is True
    assert "本轮已提供完整 Schema" in observation["output"]["message"]
    for model_request in adapter.requests:
        _assert_tool_visibility(loop, model_request)
    decisions = [item for item in store.list_events(result.trace_id) if item.event_type == "DecisionMade"]
    assert decisions[1].payload["tool_visibility"]["callable"] == [item["function"]["name"] for item in adapter.requests[1].tools]
    assert decisions[1].payload["searches"] == [{"call_id": f"{result.trace_id}:redundant", "source": "run_cache", "already_callable": True, "tools": ["vector.buffer"]}]


@pytest.mark.asyncio
async def test_search_does_not_activate_tool_earlier_in_same_batch(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(
            tool_calls=[
                {"id": "search", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}},
                {"id": "forged", "function": {"name": "vector.buffer", "arguments": '{"dataset_id":"ds_roads","distance":25}'}},
            ]
        ),
        ModelResponse(content="工具未在调用前激活，因此没有执行。"),
    )
    store, loop = _loop(tmp_path, adapter)
    calls: list[dict] = []
    metadata = loop.registry.get("vector.buffer").metadata
    loop.registry.unregister("vector.buffer")
    loop.registry.register(metadata, lambda arguments, _context: calls.append(arguments), deferred=True)

    _, result = await _run(loop, store, "生成缓冲区")

    assert result.status is AgentResultStatus.SUCCESS
    assert calls == []
    assert store.get_tool_call(f"{result.trace_id}:forged") is None
    observation = adapter.requests[1].messages[-1]["content"]
    assert "DEFERRED_TOOL_NOT_ACTIVE" in observation
    assert {message.get("tool_call_id") for message in adapter.requests[1].messages if message.get("role") == "tool"} == {"search", "forged"}
    assert "vector.buffer" in {tool["function"]["name"] for tool in adapter.requests[1].tools}


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ['{"query":""}', '{"query":"buffer","limit":3}', '{"query":"buffer","english_query":""}', '{"query":"buffer","english_query":123}'])
async def test_invalid_tool_search_arguments_return_a_tool_observation(tmp_path, arguments):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "invalid_search", "function": {"name": "tool.search", "arguments": arguments}}]),
        ModelResponse(content="搜索参数无效，已安全恢复。"),
    )
    store, loop = _loop(tmp_path, adapter)

    _, result = await _run(loop, store, "搜索工具")

    assert result.status is AgentResultStatus.SUCCESS
    tool_messages = [item for item in adapter.requests[1].messages if item.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "invalid_search"
    assert "INVALID_TOOL_ARGUMENTS" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_one_bilingual_search_deduplicates_shared_tools_and_counts_once(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[
            {"id": "bilingual", "function": {"name": "tool.search", "arguments": '{"query":"缓冲区","english_query":"buffer"}'}},
        ]),
        ModelResponse(tool_calls=[{"id": "buffer", "function": {"name": "vector.buffer", "arguments": '{"dataset_id":"ds_roads","distance":25}'}}]),
        ModelResponse(content="检查完成，中文提问和回复不受限制。"),
    )
    store, loop = _loop(tmp_path, adapter)
    writes = []
    metadata = loop.registry.get("vector.buffer").metadata
    loop.registry.unregister("vector.buffer")
    loop.registry.register(metadata, lambda arguments, _context: writes.append(arguments) or {"output": {"created": True}}, deferred=True)
    _, result = await _run(loop, store, "检查这个中文请求")

    assert result.status is AgentResultStatus.SUCCESS
    assert writes == [{"dataset_id": "ds_roads", "distance": 25}]
    assert store.latest_checkpoint(result.trace_id).state["activated_tool_names"] == ["vector.buffer"]
    assert "去重取并集" in adapter.requests[0].messages[0]["content"]
    observations = {item["tool_call_id"]: item["content"] for item in adapter.requests[1].messages if item["role"] == "tool"}
    assert set(observations) == {"bilingual"}
    assert '"status": "SUCCESS"' in observations["bilingual"]
    assert store.get_run(result.trace_id).tool_call_count == 2
    assert "只调用一次 tool.search" in adapter.requests[0].messages[0]["content"]
    names = [item["function"]["name"] for item in adapter.requests[1].tools]
    assert names.count("vector.buffer") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["distinct", "overlap", "empty"])
async def test_one_bilingual_search_preserves_union_and_executes_each_tool_once(tmp_path, mode):
    english_names = {"test.en_a"}
    chinese_names = {"test.zh_a"}
    if mode == "overlap":
        english_names = {"test.shared"}
        chinese_names = {"test.shared"}
    if mode == "empty":
        chinese_names = set()
    expected = english_names | chinese_names
    chinese_query = "唯一中文能力" if mode != "empty" else "虚空玄冥"
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[
            {"id": "bilingual", "function": {"name": "tool.search", "arguments": json.dumps({"query": chinese_query, "english_query": "quantum_marker"})}},
            {"id": "premature", "function": {"name": next(iter(english_names)), "arguments": "{}"}},
        ]),
        ModelResponse(tool_calls=[{"id": name, "function": {"name": name, "arguments": "{}"}} for name in sorted(expected)]),
        ModelResponse(content="已统一调用合并后的工具。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []
    for name in sorted(expected):
        description = " ".join([
            "quantum_marker" if name in english_names else "",
            "唯一中文能力" if name in chinese_names else "",
        ])
        loop.registry.register(
            ToolMetadata(name=name, description=description, input_schema={"type": "object", "additionalProperties": False}),
            lambda _args, context: executed.append(context.call_id) or {"output": {"checked": True}},
            deferred=True,
        )
    _, result = await _run(loop, store, "合并中英文工具检索")
    assert result.status is AgentResultStatus.SUCCESS
    names = [item["function"]["name"] for item in adapter.requests[1].tools]
    assert all(names.count(name) == 1 for name in expected)
    observations = {item["tool_call_id"]: json.loads(item["content"]) for item in adapter.requests[1].messages if item["role"] == "tool"}
    assert {item["name"] for item in observations["bilingual"]["output"]["tools"]} == expected
    assert len(observations["bilingual"]["output"]["tools"]) == len(expected)
    assert observations["premature"]["error"]["code"] == "DEFERRED_TOOL_NOT_ACTIVE"
    assert executed == [f"{result.trace_id}:{name}" for name in sorted(expected)]
    assert set(store.latest_checkpoint(result.trace_id).state["activated_tool_names"]) == expected
    assert store.get_run(result.trace_id).tool_call_count == 2 + len(expected)


@pytest.mark.asyncio
async def test_bilingual_query_does_not_skip_english_branch_when_primary_name_is_cached(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "first", "function": {"name": "tool.search", "arguments": '{"query":"slope"}'}}]),
        ModelResponse(tool_calls=[{"id": "bilingual", "function": {"name": "tool.search", "arguments": '{"query":"raster.slope","english_query":"buffer"}'}}]),
        ModelResponse(content="已合并两路发现结果。"),
    )
    store, loop = _loop(tmp_path, adapter)
    _, result = await _run(loop, store, "检查组合查询与缓存边界")
    observation = json.loads(next(item["content"] for item in adapter.requests[2].messages if item.get("tool_call_id") == "bilingual"))
    assert "source" not in observation["output"]
    assert {"raster.slope", "vector.buffer"} <= {item["name"] for item in observation["output"]["tools"]}
    assert {"raster.slope", "vector.buffer"} <= {item["function"]["name"] for item in adapter.requests[2].tools}
    assert store.get_run(result.trace_id).tool_call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", [False, True])
async def test_interrupted_search_restores_union_without_double_counting(tmp_path, combined):
    search_calls = [
        {"id": "english", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}},
        {"id": "chinese", "function": {"name": "tool.search", "arguments": '{"query":"坡度"}'}},
    ]
    if combined:
        search_calls = [{"id": "bilingual", "function": {"name": "tool.search", "arguments": '{"query":"坡度","english_query":"buffer"}'}}]
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=search_calls),
        ModelResponse(content="已获得缓冲和坡度工具。"),
    )
    store, loop = _loop(tmp_path, adapter)
    conversation = store.create_conversation("检索恢复", user_id="test-user")
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="查找缓冲和坡度工具")
    prepared = await loop.prepare_request(request)
    original = loop._save_checkpoint
    interrupted = False

    def interrupt_between_searches(*args, **kwargs):
        nonlocal interrupted
        original(*args, **kwargs)
        pending = kwargs.get("pending_tool_calls", [])
        if not combined and not interrupted and args[4] == "tool_pending" and pending and pending[0][0] == "chinese":
            interrupted = True
            raise RuntimeError("模拟两次检索之间进程中断")

    loop._save_checkpoint = interrupt_between_searches
    original_search = loop.catalog.search

    def interrupt_internal_search(query, context, limit):
        nonlocal interrupted
        if combined and not interrupted and query == "buffer":
            interrupted = True
            raise RuntimeError("模拟内部两路检索之间进程中断")
        return original_search(query, context, limit)

    loop.catalog.search = interrupt_internal_search
    with pytest.raises(RuntimeError, match="进程中断"):
        await loop.run(request, prepared=prepared)
    checkpoint = store.latest_checkpoint(prepared.run.id)
    assert checkpoint.state["batch_next_activations"] == (None if combined else ["vector.buffer"])
    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(prepared.run.id)),
                            resume_from=checkpoint, continuation={"type": "technical_resume"})
    assert result.status is AgentResultStatus.SUCCESS
    names = {item["function"]["name"] for item in adapter.requests[1].tools}
    assert {"vector.buffer", "raster.slope"}.issubset(names)
    assert store.get_run(prepared.run.id).tool_call_count == (1 if combined else 2)
    assert len(adapter.requests) == 2
    for model_request in adapter.requests:
        _assert_tool_visibility(loop, model_request)


@pytest.mark.asyncio
async def test_later_search_preserves_used_schema_and_empty_search_keeps_unused_card(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "search_buffer", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}}]),
        ModelResponse(
            tool_calls=[
                {"id": "search_slope", "function": {"name": "tool.search", "arguments": '{"query":"slope"}'}},
                {"id": "active_buffer", "function": {"name": "vector.buffer", "arguments": '{"dataset_id":"ds_roads","distance":15}'}},
            ]
        ),
        ModelResponse(content="已重新搜索。"),
    )
    store, loop = _loop(tmp_path, adapter)
    calls: list[dict] = []
    metadata = loop.registry.get("vector.buffer").metadata
    loop.registry.unregister("vector.buffer")
    loop.registry.register(metadata, lambda arguments, _context: calls.append(arguments) or {"output": "ok"}, deferred=True)
    _, result = await _run(loop, store, "查询两个能力")
    names = [{tool["function"]["name"] for tool in item.tools} for item in adapter.requests]
    assert "vector.buffer" not in names[0]
    assert "vector.buffer" in names[1]
    assert "vector.buffer" in names[2]
    assert "raster.slope" in names[2]
    assert calls == [{"dataset_id": "ds_roads", "distance": 15}]
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert checkpoint.state["activated_tool_names"] == ["raster.slope", "vector.buffer"]

    empty_adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "search_buffer", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}}]),
        ModelResponse(tool_calls=[{"id": "search_empty", "function": {"name": "tool.search", "arguments": '{"query":"not_a_real_capability_938"}'}}]),
        ModelResponse(content="没有找到匹配工具。"),
    )
    empty_store, empty_loop = _loop(tmp_path / "empty", empty_adapter)
    _, empty_result = await _run(empty_loop, empty_store, "查询能力")
    assert "vector.buffer" not in {tool["function"]["name"] for tool in empty_adapter.requests[2].tools}
    assert {item["name"] for item in _assert_tool_visibility(empty_loop, empty_adapter.requests[2])["cached"]} == {"vector.buffer"}
    assert empty_store.latest_checkpoint(empty_result.trace_id).state["activated_tool_names"] == []
    assert empty_store.latest_checkpoint(empty_result.trace_id).state["used_tool_names"] == []


@pytest.mark.asyncio
async def test_eight_tool_limit_keeps_discovery_records_and_restores_from_cache(tmp_path):
    names = [f"test.operation_{index}" for index in range(9)]
    adapter = SequenceAdapter(
        *[ModelResponse(tool_calls=[{"id": f"search_{index}", "function": {"name": "tool.search", "arguments": json.dumps({"query": f"unique_capability_{index}"})}}]) for index in range(9)],
        ModelResponse(tool_calls=[{"id": "pause", "function": {"name": "agent.ask_user", "arguments": '{"question":"接下来使用第一个工具吗？"}'}}]),
        ModelResponse(tool_calls=[
            {"id": "evicted", "function": {"name": names[0], "arguments": "{}"}},
            {"id": "restore", "function": {"name": "tool.search", "arguments": json.dumps({"query": names[0]})}},
            {"id": "premature", "function": {"name": names[0], "arguments": "{}"}},
        ]),
        ModelResponse(tool_calls=[{"id": "execute", "function": {"name": names[0], "arguments": "{}"}}]),
        ModelResponse(content="已从发现缓存恢复并执行。"),
    )
    store, loop = _loop(tmp_path, adapter)
    loop.settings.max_agent_turns = 13
    loop.settings.max_tool_calls = 16
    executed = []
    for index, name in enumerate(names):
        loop.registry.register(ToolMetadata(name=name, description=f"unique_capability_{index}", input_schema={"type": "object"}),
                               lambda _args, context: executed.append(context.call_id) or {"output": {"checked": True}}, deferred=True)
    searches = []
    original_search = loop.catalog.tool_search

    def record_search(arguments, context):
        searches.append(arguments["query"])
        return original_search(arguments, context)

    loop.catalog.tool_search = record_search
    request, waiting = await _run(loop, store, "逐步发现工具后复用第一个")
    assert waiting.error == "WAITING_USER"
    saved = store.latest_checkpoint(waiting.trace_id)
    assert set(saved.state["discovered_tool_names"]) == set(names)
    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(waiting.trace_id)),
                            resume_from=saved, continuation={"type": "user_input", "content": "使用第一个工具"})
    assert result.status is AgentResultStatus.SUCCESS
    assert executed == [f"{result.trace_id}:execute"]
    assert searches == [f"unique_capability_{index}" for index in range(9)]
    assert names[0] not in {item["function"]["name"] for item in adapter.requests[9].tools}
    assert names[0] in {item["function"]["name"] for item in adapter.requests[11].tools}
    observations = {item["tool_call_id"]: json.loads(item["content"]) for item in adapter.requests[11].messages if item["role"] == "tool"}
    assert observations["restore"]["output"]["source"] == "run_cache"
    assert observations["restore"]["output"]["already_callable"] is False
    assert observations["evicted"]["error"]["code"] == "DEFERRED_TOOL_NOT_ACTIVE"
    assert observations["premature"]["error"]["code"] == "DEFERRED_TOOL_NOT_ACTIVE"
    for request in adapter.requests:
        _assert_tool_visibility(loop, request)
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert set(checkpoint.state["discovered_tool_names"]) == set(names)
    assert checkpoint.state["activated_tool_names"] == [names[0]]
    assert checkpoint.state["used_tool_names"] == [names[0]]
    raw = next(item for item in saved.state["protocol_messages"] if item.get("tool_call_id") == "search_0")
    assert "description" in json.loads(raw["content"])["output"]["tools"][0]
    assert observations["search_8"]["output"]["tools"] == [{"name": names[8]}]


@pytest.mark.parametrize("budget_delta", [0, -1])
def test_tool_schema_budget_boundary_uses_cards_without_truncating_schema(tmp_path, budget_delta):
    _, loop = _loop(tmp_path, None)
    name = "test.large_schema"
    schema = {"type": "object", "properties": {"mode": {"type": "string", "enum": [f"choice_{index}" for index in range(120)]}}}
    loop.registry.register(ToolMetadata(name=name, description="Large parameters", input_schema=schema), lambda *_: {}, deferred=True)
    context = loop._discovery_context(AgentRequest(conversation_id="test", user_id="test-user", user_input="test"), loop.services_factory("test-user"))
    full_definitions = loop._tool_definitions(context, {name})
    loop.settings.tool_context_tokens = loop._tool_context_tokens(full_definitions, []) + budget_delta
    definitions, cards, active = loop._tool_context(context, [name], {name})
    assert loop._tool_context_tokens(definitions, cards) <= loop.settings.tool_context_tokens
    if budget_delta == 0:
        assert active == {name}
        assert cards == []
        assert next(item["function"]["parameters"] for item in definitions if item["function"]["name"] == name) == schema
    else:
        assert active == set()
        assert [item["name"] for item in cards] == [name]
        messages = prepare_model_messages([{"role": "system", "content": "original"}], definitions, cards)
        assert "精确查询工具名称" in messages[1]["content"]
    original = [{"role": "system", "content": "original"}, {"role": "user", "content": "需要这个工具"}]
    messages = prepare_model_messages(original, definitions, cards)
    visibility = _assert_tool_visibility(loop, ModelRequest(messages=messages, tools=definitions))
    assert visibility["cached"] == cards
    assert original == [{"role": "system", "content": "original"}, {"role": "user", "content": "需要这个工具"}]


def test_cards_and_schemas_share_one_budget_and_permission_filter(tmp_path):
    _, loop = _loop(tmp_path, None)
    loop.settings.tool_context_tokens //= 2
    names = [f"test.detailed_{index}" for index in range(18)]
    for name in names:
        loop.registry.register(ToolMetadata(name=name, description="Detailed operation", input_schema={
            "type": "object", "properties": {"mode": {"type": "string", "enum": [f"choice_{index}" for index in range(160)]}},
        }), lambda *_: {}, deferred=True)
    loop.registry.register(ToolMetadata(name="test.forbidden", description="Unavailable", required_scopes=["system.admin"]), lambda *_: {}, deferred=True)
    context = loop._discovery_context(AgentRequest(conversation_id="test", user_id="test-user", user_input="test"), loop.services_factory("test-user"))
    definitions, cards, active = loop._tool_context(context, [*names, "test.forbidden"], set(names) | {"test.forbidden"})
    assert cards and active
    visible = active | {item["name"] for item in cards}
    assert len(visible) <= 16
    assert not (active & {item["name"] for item in cards})
    assert "test.forbidden" not in visible
    assert loop._tool_context_tokens(definitions, cards) <= loop.settings.tool_context_tokens
    assert visible <= set(names[-16:])
    messages = prepare_model_messages([{"role": "system", "content": "original"}], definitions, cards)
    visibility = _assert_tool_visibility(loop, ModelRequest(messages=messages, tools=definitions))
    assert "test.forbidden" not in visibility["callable"]
    assert "test.forbidden" not in {item["name"] for item in visibility["cached"]}


@pytest.mark.asyncio
async def test_multi_tool_call_batch_keeps_all_results_before_waiting_for_user(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(
            tool_calls=[
                {"id": "search", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}},
                {"id": "question", "function": {"name": "agent.ask_user", "arguments": '{"question":"要使用多少米？"}'}},
            ]
        )
    )
    store, loop = _loop(tmp_path, adapter)
    _, result = await _run(loop, store, "生成缓冲区")
    assert result.error == "WAITING_USER"
    checkpoint = store.latest_checkpoint(result.trace_id)
    messages = checkpoint.state["protocol_messages"]
    tool_results = [item["tool_call_id"] for item in messages if item.get("role") == "tool"]
    assert tool_results == ["search", "question"]
    assert checkpoint.state["activated_tool_names"] == ["vector.buffer"]


@pytest.mark.asyncio
async def test_active_tools_are_run_local_and_inspect_is_read_only(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "search", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}}]),
        ModelResponse(content="第一轮结束。"),
        ModelResponse(content="新运行没有继承旧工具。"),
    )
    store, loop = _loop(tmp_path, adapter)
    _, first = await _run(loop, store, "搜索缓冲工具")
    _, second = await _run(loop, store, "另一个独立问题")
    assert first.status is AgentResultStatus.SUCCESS
    assert second.status is AgentResultStatus.SUCCESS
    third_names = {tool["function"]["name"] for tool in adapter.requests[2].tools}
    assert "vector.buffer" not in third_names
    assert "vector.buffer" not in _assert_tool_visibility(loop, adapter.requests[2])["callable"]

    inspect_schema = loop.registry.get("dataset.inspect").metadata.input_schema
    assert inspect_schema["required"] == ["dataset_id"]
    assert set(inspect_schema["properties"]) == {"dataset_id"}


@pytest.mark.asyncio
async def test_checkpoint_restore_revalidates_current_environment(tmp_path):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "search", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}}]),
        ModelResponse(tool_calls=[{"id": "question", "function": {"name": "agent.ask_user", "arguments": '{"question":"采用多少米？"}'}}]),
        ModelResponse(tool_calls=[
            {"id": "cached", "function": {"name": "tool.search", "arguments": '{"query":"vector.buffer"}'}},
            {"id": "buffer", "function": {"name": "vector.buffer", "arguments": '{"dataset_id":"ds_roads","distance":25}'}},
        ]),
        ModelResponse(content="当前环境不支持该操作，因此没有执行。"),
    )
    store, loop = _loop(tmp_path, adapter)
    services = loop.services_factory("test-user")
    loop.services_factory = lambda _user: services
    calls: list[dict] = []
    metadata = loop.registry.get("vector.buffer").metadata
    loop.registry.unregister("vector.buffer")
    loop.registry.register(metadata, lambda arguments, _context: calls.append(arguments), deferred=True)
    conversation = store.create_conversation("环境恢复", user_id="test-user")
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="生成道路缓冲区")
    store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))

    waiting = await loop.run(request, prepared=await loop.prepare_request(request))
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    assert checkpoint.state["activated_tool_names"] == []
    assert checkpoint.state["discovered_tool_names"] == ["vector.buffer"]
    services.pop("vectors")
    store.save_message(Message(conversation_id=conversation.id, role="user", content="采用25米"))
    resumed_request = request.model_copy(update={"user_input": "采用25米"})
    resumed = await loop.run(
        resumed_request,
        prepared=loop.prepare_resume(resumed_request, store.get_run(waiting.trace_id)),
        resume_from=checkpoint,
        continuation={"type": "user_input", "content": "采用25米"},
    )

    assert resumed.status is AgentResultStatus.SUCCESS
    assert "vector.buffer" not in {tool["function"]["name"] for tool in adapter.requests[2].tools}
    assert calls == []
    assert "DEFERRED_TOOL_NOT_ACTIVE" in adapter.requests[3].messages[-1]["content"]
    cached = next(item for item in adapter.requests[3].messages if item.get("tool_call_id") == "cached")
    output = json.loads(cached["content"])["output"]
    assert "vector.buffer" not in {item["name"] for item in output["tools"]}
    assert "source" not in output
    visibility = _assert_tool_visibility(loop, adapter.requests[2])
    assert "vector.buffer" not in visibility["callable"]
    assert "vector.buffer" not in {item["name"] for item in visibility["cached"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("approved_by_user", [True, False])
async def test_approval_pauses_and_resumes_saved_call_once(tmp_path, approved_by_user):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "search", "function": {"name": "tool.search", "arguments": '{"query":"敏感"}'}}]),
        ModelResponse(tool_calls=[{"id": "danger", "function": {"name": "test.sensitive_write", "arguments": '{"value":7}'}}]),
        ModelResponse(content="审批结果已处理。"),
    )
    store, loop = _loop(tmp_path, adapter)
    loop.executor.approval_service = ApprovalService(store)
    side_effects: list[dict] = []
    loop.registry.register(
        ToolMetadata(
            name="test.sensitive_write",
            description="敏感测试写入操作",
            input_schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False},
            required_scopes=["dataset.write"],
            risk_level=RiskLevel.DESTRUCTIVE,
        ),
        lambda arguments, _context: side_effects.append(arguments) or {"output": {"written": True}},
        deferred=True,
    )
    conversation = store.create_conversation("审批恢复", user_id="test-user")
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="执行敏感写入")
    store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))
    prepared = await loop.prepare_request(request)
    waiting = await loop.run(request, prepared=prepared)

    assert waiting.error == "APPROVAL_REQUIRED"
    assert store.get_run(waiting.trace_id).status is RunStatus.WAITING_APPROVAL
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    pending = checkpoint.state["pending_approvals"][0]
    assert checkpoint.state["used_tool_names"] == ["test.sensitive_write"]
    assert checkpoint.state["activated_tool_names"] == ["test.sensitive_write"]
    assert pending["tool_name"] == "test.sensitive_write"
    assert pending["arguments"] == {"value": 7}
    assert store.get_tool_call(pending["call_id"]) is None

    approval = store.get_approval(pending["approval_id"], user_id="test-user")
    decision = (
        loop.executor.approval_service.approve(approval.id, user_id="test-user")
        if approved_by_user
        else loop.executor.approval_service.deny(approval.id, user_id="test-user")
    )
    resumed_run, _ = record_approval_decision(store, decision)
    result = await loop.run(
        request,
        prepared=loop.prepare_resume(request, resumed_run),
        resume_from=checkpoint,
        continuation={"type": "approval_result", "approval_id": approval.id, "approved": approved_by_user},
    )

    assert result.status is AgentResultStatus.SUCCESS
    assert side_effects == ([{"value": 7}] if approved_by_user else [])
    final_approval = store.get_approval(approval.id, user_id="test-user")
    assert final_approval.status.value == ("CONSUMED" if approved_by_user else "DENIED")
    expected_observation = "written" if approved_by_user else "APPROVAL_DENIED"
    assert expected_observation in adapter.requests[-1].messages[-1]["content"]


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
    conversation = store.create_conversation("历史检索", user_id="test-user")
    store.save_message(Message(conversation_id=conversation.id, role="user", content="道路缓冲距离采用 500 米。"))
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="之前的距离是多少？")
    store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))
    prepared = await loop.prepare_request(request)

    result = await loop.run(request, prepared=prepared)

    assert result.status is AgentResultStatus.SUCCESS
    assert "500 米" in adapter.requests[1].messages[-1]["content"]
    assert store.get_run(result.trace_id).tool_call_count == 1


def test_prepared_messages_keep_full_tool_observations():
    observation = {"role": "tool", "tool_call_id": "inspect", "content": json.dumps({"status": "SUCCESS", "output": {"body": "x" * 4000}})}
    original = [
        {"role": "system", "content": "规则"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "inspect", "type": "function", "function": {"name": "dataset.inspect", "arguments": "{}"}}]},
        observation,
    ]
    prepared = prepare_model_messages(original, [], [])
    assert prepared[2:] == original[1:]
    assert original[-1] == observation


def test_tool_results_slide_to_sixteen_and_emergency_summarizes_older_batches():
    calls = [
        {"id": call_id, "type": "function", "function": {"name": "test.report", "arguments": "{}"}}
        for call_id in (f"call_{index}" for index in range(18))
    ]
    original = [
        {"role": "system", "content": "固定规则"},
        {"role": "user", "content": "保留当前目标"},
    ]
    for call in calls:
        original.append({"role": "assistant", "content": "", "tool_calls": [call]})
        original.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"status": "SUCCESS", "output": {"body": "x" * 4000}})})
    marked: set[str] = set()
    summarized: set[str] = set()
    options = {"run_id": "run-test", "compacted_ids": marked, "summarized_ids": summarized,
               "recent_full": 16, "emergency_fraction": 0.5}
    view = compact_model_input(original, **options)
    assert marked == {"call_0", "call_1"}
    assert summarized == set()
    assert json.loads(next(item["content"] for item in view if item.get("tool_call_id") == "call_0"))["result_reference"] == {
        "run_id": "run-test", "tool_call_id": "call_0", "source": "checkpoint.protocol_messages",
    }
    assert json.loads(next(item["content"] for item in view if item.get("tool_call_id") == "call_17"))["output"] == {"body": "x" * 4000}
    emergency = compact_model_input(original, emergency=True, **options)
    assert marked == {f"call_{index}" for index in range(10)}
    assert summarized == {"call_0", "call_1"}
    assert sum(item.get("role") == "system" and "执行历史摘要" in item.get("content", "") for item in emergency) == 1
    assert not any(item.get("tool_call_id") in summarized for item in emergency)
    declared = {call["id"] for item in emergency for call in item.get("tool_calls", [])}
    observed = {item["tool_call_id"] for item in emergency if item.get("role") == "tool"}
    assert declared == observed
    assert compact_model_input(original, **options) == emergency
    assert all(json.loads(item["content"])["output"] == {"body": "x" * 4000} for item in original if item.get("role") == "tool")


@pytest.mark.parametrize(
    ("result_count", "recent_full"),
    [(1, 16), (9, 16), (10, 16), (16, 16), (20, 20), (24, 16)],
)
def test_emergency_compacts_half_of_current_full_results(result_count, recent_full):
    messages = [{"role": "system", "content": "固定规则"}]
    for index in range(result_count):
        call_id = f"call_{index}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {"name": "test.report", "arguments": "{}"},
        }]})
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": '{"status":"SUCCESS","output":{"value":1}}'})

    marked, summarized = set(), set()
    compact_model_input(messages, run_id="run-test", compacted_ids=marked,
                        summarized_ids=summarized, recent_full=recent_full,
                        emergency_fraction=0.5, emergency=True)

    routine_count = max(0, result_count - recent_full)
    emergency_count = (result_count - routine_count) // 2
    assert marked == {f"call_{index}" for index in range(routine_count + emergency_count)}
    assert summarized == {f"call_{index}" for index in range(routine_count)}


def test_emergency_keeps_latest_complete_batch_even_when_half_would_reach_it():
    calls = [{"id": f"call_{index}", "type": "function",
              "function": {"name": "test.report", "arguments": "{}"}} for index in range(10)]
    messages = [{"role": "system", "content": "固定规则"}]
    for call in calls[:4]:
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": '{"status":"SUCCESS","output":{"value":1}}'})
    messages.append({"role": "assistant", "content": "", "tool_calls": calls[4:]})
    messages.extend({"role": "tool", "tool_call_id": call["id"],
                     "content": '{"status":"SUCCESS","output":{"value":1}}'} for call in calls[4:])

    marked, summarized = set(), set()
    compact_model_input(messages, run_id="run-test", compacted_ids=marked,
                        summarized_ids=summarized, recent_full=10,
                        emergency_fraction=0.5, emergency=True)

    assert marked == {f"call_{index}" for index in range(4)}
    assert summarized == set()


def test_tool_history_summary_keeps_remaining_calls_in_mixed_batch():
    calls = [
        {"id": name, "type": "function", "function": {"name": "test.report", "arguments": "{}"}}
        for name in ("old", "still_visible", "latest")
    ]
    original = [
        {"role": "system", "content": "固定规则"},
        {"role": "assistant", "content": "", "tool_calls": calls[:2]},
        {"role": "tool", "tool_call_id": "old", "content": '{"status":"SUCCESS","output":{"value":1}}'},
        {"role": "tool", "tool_call_id": "still_visible", "content": '{"status":"SUCCESS","output":{"value":2}}'},
        {"role": "assistant", "content": "", "tool_calls": calls[2:]},
        {"role": "tool", "tool_call_id": "latest", "content": '{"status":"SUCCESS","output":{"value":3}}'},
    ]
    marked, summarized = {"old"}, set()
    view = compact_model_input(original, run_id="run-test", compacted_ids=marked,
                               summarized_ids=summarized, recent_full=16, emergency_fraction=0.5, emergency=True)
    assert summarized == {"old"}
    assert marked == {"old", "still_visible"}
    assert [call["id"] for item in view for call in item.get("tool_calls", [])] == ["still_visible", "latest"]
    assert [item["tool_call_id"] for item in view if item.get("role") == "tool"] == ["still_visible", "latest"]
    assert json.loads(next(item["content"] for item in view if item.get("tool_call_id") == "latest"))["output"] == {"value": 3}


def test_final_budget_view_removes_oldest_messages_and_keeps_tool_pairs():
    original = [{"role": "system", "content": "固定规则"}]
    original.extend({"role": "user", "content": f"消息 {index}：" + "内容" * 300} for index in range(10))
    original.append({"role": "system", "content": TOOL_HISTORY_PREFIX + "旧状态"})
    for index in range(17):
        calls = [{"id": f"call_{index}", "type": "function", "function": {"name": "test.report", "arguments": "{}"}}]
        original.append({"role": "assistant", "content": "", "tool_calls": calls})
        original.append({"role": "tool", "tool_call_id": f"call_{index}", "content": '{"status":"SUCCESS","output":{"value":1}}'})

    first = narrow_model_input(original, [], input_budget_tokens=1_000_000, recent_messages=8, recent_results=16)
    dialogue = [index for index, item in enumerate(first) if item["role"] == "user"]
    removed_calls = {f"call_{index}" for index in range(1, 5)}
    expected = [
        item for index, item in enumerate(first)
        if index not in dialogue[:2]
        and item.get("tool_call_id") not in removed_calls
        and not any(call["id"] in removed_calls for call in item.get("tool_calls", []))
    ]
    budget = model_input_tokens(expected, [])
    reduced = narrow_model_input(original, [], input_budget_tokens=budget, recent_messages=8, recent_results=16)

    assert [item["content"] for item in reduced if item["role"] == "user"] == [
        f"消息 {index}：" + "内容" * 300 for index in range(4, 10)
    ]
    declared = {call["id"] for item in reduced for call in item.get("tool_calls", [])}
    observed = {item["tool_call_id"] for item in reduced if item["role"] == "tool"}
    assert declared == observed == {f"call_{index}" for index in range(5, 17)}
    assert not any("执行历史摘要" in item.get("content", "") for item in reduced)
    assert len(original) == 46

    later_replies = [{"role": "user", "content": "当前请求"}]
    later_replies.extend({"role": "assistant", "content": f"后续回复 {index}"} for index in range(9))
    protected = narrow_model_input(later_replies, [], input_budget_tokens=1, recent_messages=8, recent_results=16)
    assert [item["content"] for item in protected if item["role"] == "user"] == ["当前请求"]

    no_history = narrow_model_input(original, [], input_budget_tokens=1, recent_messages=8, recent_results=16)
    assert [item["content"] for item in no_history if item["role"] == "user"] == [original[10]["content"]]
    assert not any(item["role"] == "tool" for item in no_history)

    mixed_batch = [original[0], original[1], {
        "role": "assistant", "content": "", "tool_calls": [item["tool_calls"][0] for item in original if item.get("tool_calls")],
    }]
    mixed_batch.extend(item for item in original if item["role"] == "tool")
    mixed_view = narrow_model_input(mixed_batch, [], input_budget_tokens=1_000_000, recent_messages=8, recent_results=16)
    assert {call["id"] for item in mixed_view for call in item.get("tool_calls", [])} == {
        item["tool_call_id"] for item in mixed_view if item["role"] == "tool"
    } == {f"call_{index}" for index in range(1, 17)}


@pytest.mark.asyncio
async def test_tool_history_summary_survives_checkpoint_restore(tmp_path):
    store, loop = _loop(tmp_path, None)
    conversation = store.create_conversation("恢复执行摘要", user_id="test-user")
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="检查历史")
    store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))
    run = (await loop.prepare_request(request)).run
    messages = loop.context.build(request, run=run)
    for index in range(18):
        call_id = f"result_{index}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {"name": "test.report", "arguments": "{}"},
        }]})
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps({"status": "SUCCESS", "output": {"value": index}})})
    marked, summarized = set(), set()
    options = {"run_id": run.id, "compacted_ids": marked, "summarized_ids": summarized,
               "recent_full": 16, "emergency_fraction": 0.5}
    compact_model_input(messages, **options)
    compact_model_input(messages, emergency=True, **options)
    loop._save_checkpoint(request, run, messages, None, "context_compacted", set(), [],
                          compacted_ids=marked, summarized_ids=summarized)
    checkpoint = store.latest_checkpoint(run.id)
    assert checkpoint.state["summarized_tool_call_ids"] == ["result_0", "result_1"]
    raw = {item["tool_call_id"]: json.loads(item["content"])["output"]
           for item in checkpoint.state["protocol_messages"] if item["role"] == "tool"}
    assert raw == {f"result_{index}": {"value": index} for index in range(18)}
    restored = loop.context.build(request, run=run, protocol_messages=checkpoint.state["protocol_messages"], append_request=False)
    replay = compact_model_input(restored, run_id=run.id,
                                 compacted_ids=set(checkpoint.state["compacted_tool_call_ids"]),
                                 summarized_ids=set(checkpoint.state["summarized_tool_call_ids"]),
                                 recent_full=16, emergency_fraction=0.5)
    assert [item["tool_call_id"] for item in replay if item["role"] == "tool"] == [f"result_{index}" for index in range(2, 18)]
    assert any("执行历史摘要" in item.get("content", "") for item in replay if item["role"] == "system")


def test_trusted_state_is_not_cut_mid_json(tmp_path, monkeypatch):
    _, loop = _loop(tmp_path, None)
    payload = {"selected_datasets": [{"id": "ds_large", "schema": {"field": "值" * 20000}}]}
    monkeypatch.setattr(loop.context, "_trusted_context", lambda *_: payload.copy())
    messages = loop.context.build(AgentRequest(conversation_id="test", user_id="test-user", user_input="检查结构"))
    assert json.loads(messages[1]["content"].split("\n", 1)[1]) == payload


@pytest.mark.asyncio
async def test_total_input_limit_blocks_oversized_request_without_model_call(tmp_path):
    adapter = SequenceAdapter(ModelResponse(content="不应调用模型"))
    store, loop = _loop(tmp_path, adapter)
    loop.settings.model_input_tokens = 1
    _, result = await _run(loop, store, "当前请求必须保持完整")
    assert result.error == "BUDGET_EXCEEDED"
    assert store.get_run(result.trace_id).status is RunStatus.BUDGET_EXCEEDED
    assert adapter.requests == []
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert checkpoint.state["request"]["user_input"] == "当前请求必须保持完整"


@pytest.mark.asyncio
async def test_over_budget_run_summarizes_old_conversation_and_keeps_latest_eight(tmp_path):
    class SummaryAwareAdapter(ModelAdapter):
        supports_tools = True
        supports_json_object = True

        def __init__(self):
            self.summary_requests = []
            self.requests = []

        async def complete(self, model_request):
            if model_request.response_format:
                self.summary_requests.append(model_request)
                return ModelResponse(content=json.dumps({
                    "summary": "先前讨论了研究区域与分析条件。",
                    "key_facts": [], "decisions": [], "unresolved_topics": [], "references": [],
                }, ensure_ascii=False))
            self.requests.append(model_request)
            return ModelResponse(content="已读取最近的讨论。")

    adapter = SummaryAwareAdapter()
    store, loop = _loop(tmp_path, adapter)
    conversation = store.create_conversation("紧急摘要", user_id="test-user")
    old = []
    for index in range(10):
        content = f"历史消息 {index}：" + ("完整研究条件" * 1100 if index < 3 else "保留原文")
        item = Message(conversation_id=conversation.id, role="user" if index % 2 == 0 else "assistant", content=content)
        store.save_message(item)
        old.append(item)
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="请继续当前分析")
    current = Message(conversation_id=conversation.id, role="user", content=request.user_input)
    store.save_message(current)
    prepared = await loop.prepare_request(request)
    initial = prepare_model_messages(loop.context.build(request, run=prepared.run), [], [])
    loop.settings.model_input_tokens = model_input_tokens(initial, [], adapter.count_tokens) - 1

    result = await loop.run(request, prepared=prepared)
    assert result.status is AgentResultStatus.SUCCESS
    assert adapter.summary_requests
    assert len(adapter.requests) == 1
    memory, tail = loop.context.conversation_memory.load_context(conversation.id, user_id="test-user", recent_message_limit=24)
    assert memory is not None and memory.summary_version >= 1
    assert tail == [*old[-7:], current]
    sent_history = [item["content"] for item in adapter.requests[0].messages if item["role"] in {"user", "assistant"}]
    assert sent_history == [item.content for item in tail]
    assert model_input_tokens(adapter.requests[0].messages, adapter.requests[0].tools, adapter.count_tokens) <= loop.settings.model_input_tokens
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert checkpoint.state["conversation_history_count"] == 7
    assert store.list_messages(conversation.id, limit=100) == [*old, current]


@pytest.mark.asyncio
async def test_final_budget_view_reduces_oldest_messages_before_model_call(tmp_path, monkeypatch):
    adapter = SequenceAdapter(ModelResponse(content="已回答当前问题"))
    store, loop = _loop(tmp_path, adapter)
    conversation = store.create_conversation("最终预算兜底", user_id="test-user")
    history = []
    for index in range(10):
        item = Message(conversation_id=conversation.id, role="user" if index % 2 == 0 else "assistant",
                       content=f"历史消息 {index}：" + "研究条件" * 500)
        store.save_message(item)
        history.append(item)
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="回答当前问题")
    current = Message(conversation_id=conversation.id, role="user", content=request.user_input)
    store.save_message(current)
    prepared = await loop.prepare_request(request)
    services = loop._execution_services(request, prepared.run)
    discovery = loop._discovery_context(request, services, prepared.run)
    definitions, cards, _ = loop._tool_context(discovery, [], set(), count_tokens=adapter.count_tokens)
    initial = prepare_model_messages(loop.context.build(request, run=prepared.run), definitions, cards)
    first = narrow_model_input(initial, definitions, input_budget_tokens=1_000_000, recent_messages=8, recent_results=16)
    dialogue = [index for index, item in enumerate(first) if item["role"] in {"user", "assistant"}]
    loop.settings.model_input_tokens = model_input_tokens(
        [item for index, item in enumerate(first) if index not in dialogue[:2]], definitions, adapter.count_tokens
    )

    async def no_summary(*_args, **_kwargs):
        return False

    monkeypatch.setattr(loop.context.conversation_memory, "compact_history_before", no_summary)
    result = await loop.run(request, prepared=prepared)

    assert result.status is AgentResultStatus.SUCCESS
    sent = [item["content"] for item in adapter.requests[0].messages if item["role"] in {"user", "assistant"}]
    assert sent == [item.content for item in [*history[-5:], current]]
    assert model_input_tokens(adapter.requests[0].messages, adapter.requests[0].tools, adapter.count_tokens) <= loop.settings.model_input_tokens
    assert store.list_messages(conversation.id, limit=100) == [*history, current]


@pytest.mark.asyncio
async def test_runtime_passes_configured_output_limit(tmp_path):
    adapter = SequenceAdapter(ModelResponse(content="已完成"))
    store, loop = _loop(tmp_path, adapter)
    loop.settings.max_tokens = 12800
    _, result = await _run(loop, store, "简单请求")
    assert result.status is AgentResultStatus.SUCCESS
    assert adapter.requests[0].max_tokens == 12800


@pytest.mark.asyncio
async def test_total_input_compaction_survives_checkpoint_resume(tmp_path, monkeypatch):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"large_report"}'}}]),
        ModelResponse(tool_calls=[{"id": "old", "function": {"name": "test.large_report", "arguments": "{}"}}]),
        ModelResponse(tool_calls=[{"id": "latest", "function": {"name": "test.large_report", "arguments": "{}"}}]),
        ModelResponse(tool_calls=[{"id": "pause", "function": {"name": "agent.ask_user", "arguments": '{"question":"继续吗？"}'}}]),
        ModelResponse(content="继续完成。"),
    )
    store, loop = _loop(tmp_path, adapter)
    original_complete = adapter.complete

    async def complete(model_request):
        response = await original_complete(model_request)
        if len(adapter.requests) == 3:
            loop.settings.model_input_tokens = model_input_tokens(model_request.messages, model_request.tools, adapter.count_tokens) + 500
        return response

    monkeypatch.setattr(adapter, "complete", complete)
    executed = []
    loop.registry.register(
        ToolMetadata(name="test.large_report", description="large_report", input_schema={"type": "object"}),
        lambda _args, context: executed.append(context.call_id) or {"output": {"body": "x" * 30000}},
        deferred=True,
    )
    request, waiting = await _run(loop, store, "读取两次结果")
    assert waiting.error == "WAITING_USER"
    fourth = {item["tool_call_id"]: json.loads(item["content"]) for item in adapter.requests[3].messages if item["role"] == "tool"}
    assert fourth["old"]["context_compacted"] is True
    assert fourth["latest"]["output"] == {"body": "x" * 30000}
    assert model_input_tokens(adapter.requests[3].messages, adapter.requests[3].tools, adapter.count_tokens) <= loop.settings.model_input_tokens
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    assert "old" in checkpoint.state["compacted_tool_call_ids"]
    raw = {item["tool_call_id"]: json.loads(item["content"]) for item in checkpoint.state["protocol_messages"] if item["role"] == "tool"}
    assert raw["old"]["output"] == {"body": "x" * 30000}
    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(waiting.trace_id)),
                            resume_from=checkpoint, continuation={"type": "user_input", "content": "继续"})
    assert result.status is AgentResultStatus.SUCCESS
    assert executed == [f"{waiting.trace_id}:old", f"{waiting.trace_id}:latest"]
    resumed = {item["tool_call_id"]: json.loads(item["content"]) for item in adapter.requests[4].messages if item["role"] == "tool"}
    assert resumed["old"]["context_compacted"] is True


def test_large_observation_and_checkpoint_restore_keep_complete_json(tmp_path):
    _, loop = _loop(tmp_path, None)
    original = [{"role": "system", "content": "固定规则"}, {"role": "user", "content": "读取结果"}]
    for index in range(60):
        original.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{index}", "type": "function", "function": {"name": "test.report", "arguments": "{}"},
        }]})
        original.append({"role": "tool", "tool_call_id": f"call_{index}", "content": json.dumps({
            "status": "SUCCESS", "output": {"body": "原始结果" * 5000},
        }, ensure_ascii=False)})
    request = AgentRequest(conversation_id="test", user_id="test-user", user_input="恢复任务")
    built = loop.context.build(request, protocol_messages=original[1:], append_request=False)
    assert built[1:] == original[1:]
    assert len(built) > 48
    assert len(original[-1]["content"]) > 16000
    assert json.loads(built[-1]["content"])["output"] == {"body": "原始结果" * 5000}


@pytest.mark.asyncio
@pytest.mark.parametrize("model_failure", [False, True])
async def test_checkpoint_restore_keeps_observations_and_never_reexecutes_tools(tmp_path, monkeypatch, model_failure):
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"large_report"}'}}]),
        ModelResponse(tool_calls=[{"id": f"execute_{index}", "function": {"name": "test.large_report", "arguments": "{}"}}
                                  for index in range(4)]),
        ModelResponse(tool_calls=[{"id": "pause", "function": {"name": "agent.ask_user", "arguments": '{"question":"继续吗？"}'}}]),
        ModelResponse(content="恢复完成，无需重复执行。"),
    )
    original_complete = adapter.complete

    async def complete(request):
        response = await original_complete(request)
        if model_failure and len(adapter.requests) == 3:
            raise TimeoutError("模拟工具执行后的模型超时")
        return response

    monkeypatch.setattr(adapter, "complete", complete)
    store, loop = _loop(tmp_path, adapter)
    executed = []
    loop.registry.register(ToolMetadata(name="test.large_report", description="large_report", input_schema={"type": "object"}),
                           lambda _args, context: executed.append(context.call_id) or {"output": {"body": "x" * 4000}}, deferred=True)
    request, waiting = await _run(loop, store, "读取四项完整结果")
    assert waiting.error == ("MODEL_UNAVAILABLE" if model_failure else "WAITING_USER")
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    assert checkpoint.state["compacted_tool_call_ids"] == []
    raw = {item["tool_call_id"]: json.loads(item["content"]) for item in checkpoint.state["protocol_messages"] if item["role"] == "tool"}
    assert all(raw[f"execute_{index}"]["output"] == {"body": "x" * 4000} for index in range(4))
    assert all("context_compacted" not in payload for payload in raw.values())
    before_resume = list(executed)
    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(waiting.trace_id)),
                            resume_from=checkpoint, continuation=(
                                {"type": "technical_resume"} if model_failure else {"type": "user_input", "content": "继续"}
                            ))
    assert result.status is AgentResultStatus.SUCCESS
    assert executed == before_resume == [f"{waiting.trace_id}:execute_{index}" for index in range(4)]
    assert len(adapter.requests) == 4
    assert store.get_run(result.trace_id).token_usage.model_calls == (3 if model_failure else 4)
    assert store.get_run(result.trace_id).tool_call_count == (5 if model_failure else 6)
    for model_request in adapter.requests[2:]:
        observations = {item["tool_call_id"]: json.loads(item["content"]) for item in model_request.messages if item["role"] == "tool"}
        assert {f"execute_{index}" for index in range(4)} <= observations.keys()
        assert all(observations[identifier]["output"] == {"body": "x" * 4000} for identifier in raw if identifier.startswith("execute_"))
        declared = {call["id"] for item in model_request.messages for call in item.get("tool_calls", [])}
        assert observations.keys() == declared
    final = store.latest_checkpoint(result.trace_id)
    assert final.state["compacted_tool_call_ids"] == []
    assert store.get_tool_call(f"{result.trace_id}:execute_0")[1].output == {"body": "x" * 4000}
