from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.context import compact_tool_results
from app.agent.loop import TOOL_VISIBILITY_PREFIX, AgentLoop, _tool_observation
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
    ToolResult,
    ToolStatus,
)
from app.core.tokens import estimate_tokens
from app.execution.tools import ToolExecutor, ToolRegistry
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
                               protocol_history_tokens=25600, tool_result_compaction_ratio=0.2,
                               tool_context_tokens=3200, tool_context_max_cards=8)
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
    assert len(card_names) + sum(loop.registry.is_deferred(name) for name in visibility["callable"]) <= 8
    tokens = estimate_tokens(json.dumps(request.tools, ensure_ascii=False, separators=(",", ":")))
    tokens += estimate_tokens(messages[0]["content"])
    assert tokens == loop._tool_context_tokens(request.tools, visibility["cached"])
    assert tokens <= loop.settings.tool_context_tokens
    return visibility


async def _run(loop: AgentLoop, store: StateStore, text: str, on_model_delta=None):
    conversation = store.create_conversation("测试")
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
                yield ModelStreamChunk(content="先查询")
                assert fragments == ["先查询"]
                assert store.get_run(run.id).tool_call_count == 0
                yield ModelStreamChunk(content="数据。")
                assert store.get_run(run.id).tool_call_count == 0
                yield ModelStreamChunk(done=True, finish_reason="tool_calls", input_tokens=100, output_tokens=10,
                                       tool_calls=[{"id": "list", "function": {"name": "dataset.list", "arguments": "{}"}}])
            else:
                assert store.get_run(run.id).tool_call_count == 1
                yield ModelStreamChunk(content="没有")
                yield ModelStreamChunk(content="数据。")
                yield ModelStreamChunk(done=True, finish_reason="stop", input_tokens=200, output_tokens=20)

    adapter = StreamingAdapter()
    store, loop = _loop(tmp_path, adapter)

    async def capture(content):
        fragments.append(content)

    _, result = await _run(loop, store, "查看数据", capture)
    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "没有数据。"
    assert fragments == ["先查询", "数据。", "没有", "数据。"]
    run = store.get_run(result.trace_id)
    assert run.token_usage.reported_input_tokens == 300
    assert run.token_usage.reported_output_tokens == 30
    assert run.token_usage.model_calls == 2
    assert sum(event.event_type == "ModelResponseStarted" for event in store.list_events(run.id)) == 2


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
    names = [f"test.candidate_{index}" for index in range(3)]

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
        loop.registry.register(ToolMetadata(name=name, description="primary_marker" if index < 2 else "secondary_marker",
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
    names = [f"test.batch_{index}" for index in range(3)]
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"batch_primary","english_query":"batch_secondary"}'}}]),
        ModelResponse(tool_calls=[{"id": f"execute_{index}", "function": {"name": name, "arguments": "{}"}} for index, name in enumerate(names[:2])]),
        ModelResponse(content="已恢复剩余调用，未用候选已降为卡片。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []
    for index, name in enumerate(names):
        loop.registry.register(ToolMetadata(name=name, description="batch_primary" if index < 2 else "batch_secondary", input_schema={"type": "object"}),
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
    assert {item["name"] for item in visibility["cached"]} == {names[2]}


@pytest.mark.asyncio
async def test_unused_card_restores_from_cache_without_research_or_same_batch_execution(tmp_path):
    names = [f"test.restore_{index}" for index in range(3)]
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"restore_primary","english_query":"restore_secondary"}'}}]),
        ModelResponse(tool_calls=[{"id": "first", "function": {"name": names[0], "arguments": "{}"}}]),
        ModelResponse(tool_calls=[
            {"id": "restore", "function": {"name": "tool.search", "arguments": json.dumps({"query": names[2]})}},
            {"id": "premature", "function": {"name": names[2], "arguments": "{}"}},
        ]),
        ModelResponse(tool_calls=[{"id": "second", "function": {"name": names[2], "arguments": "{}"}}]),
        ModelResponse(content="只恢复卡片，不重复语义检索。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []
    for index, name in enumerate(names):
        loop.registry.register(ToolMetadata(name=name, description="restore_primary" if index < 2 else "restore_secondary", input_schema={"type": "object"}),
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
    assert set(checkpoint.state["used_tool_names"]) == {names[0], names[2]}
    assert set(checkpoint.state["activated_tool_names"]) == {names[0], names[2]}
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
    english_names = {"test.en_a", "test.en_b"}
    chinese_names = {"test.zh_a", "test.zh_b"}
    if mode == "overlap":
        chinese_names = {"test.en_a", "test.zh_a"}
    if mode == "empty":
        chinese_names = set()
    expected = english_names | chinese_names
    chinese_query = "唯一中文能力" if mode != "empty" else "虚空玄冥"
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[
            {"id": "bilingual", "function": {"name": "tool.search", "arguments": json.dumps({"query": chinese_query, "english_query": "quantum_marker"})}},
            {"id": "premature", "function": {"name": "test.en_a", "arguments": "{}"}},
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
        messages = loop._tool_context_messages([{"role": "system", "content": "original"}], definitions, cards,
                                              run_id="test-run", compacted_ids=set())
        assert "精确查询工具名称" in messages[1]["content"]
    original = [{"role": "system", "content": "original"}, {"role": "user", "content": "需要这个工具"}]
    messages = loop._tool_context_messages(original, definitions, cards, run_id="test-run", compacted_ids=set())
    visibility = _assert_tool_visibility(loop, ModelRequest(messages=messages, tools=definitions))
    assert visibility["cached"] == cards
    assert original == [{"role": "system", "content": "original"}, {"role": "user", "content": "需要这个工具"}]


def test_cards_and_schemas_share_one_budget_and_permission_filter(tmp_path):
    _, loop = _loop(tmp_path, None)
    names = [f"test.detailed_{index}" for index in range(10)]
    for name in names:
        loop.registry.register(ToolMetadata(name=name, description="Detailed operation", input_schema={
            "type": "object", "properties": {"mode": {"type": "string", "enum": [f"choice_{index}" for index in range(160)]}},
        }), lambda *_: {}, deferred=True)
    loop.registry.register(ToolMetadata(name="test.forbidden", description="Unavailable", required_scopes=["system.admin"]), lambda *_: {}, deferred=True)
    context = loop._discovery_context(AgentRequest(conversation_id="test", user_id="test-user", user_input="test"), loop.services_factory("test-user"))
    definitions, cards, active = loop._tool_context(context, [*names, "test.forbidden"], set(names) | {"test.forbidden"})
    assert cards and active
    visible = active | {item["name"] for item in cards}
    assert len(visible) <= 8
    assert not (active & {item["name"] for item in cards})
    assert "test.forbidden" not in visible
    assert loop._tool_context_tokens(definitions, cards) <= loop.settings.tool_context_tokens
    assert visible <= set(names[-8:])
    messages = loop._tool_context_messages([{"role": "system", "content": "original"}], definitions, cards,
                                          run_id="test-run", compacted_ids=set())
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
    conversation = store.create_conversation("历史检索")
    store.save_message(Message(conversation_id=conversation.id, role="user", content="道路缓冲距离采用 500 米。"))
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="之前的距离是多少？")
    store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))
    prepared = await loop.prepare_request(request)

    result = await loop.run(request, prepared=prepared)

    assert result.status is AgentResultStatus.SUCCESS
    assert "500 米" in adapter.requests[1].messages[-1]["content"]
    assert store.get_run(result.trace_id).tool_call_count == 1


def _history_tokens(messages):
    history = [item for item in messages if item["role"] != "system"]
    return estimate_tokens(json.dumps(history, ensure_ascii=False, separators=(",", ":")))


def _compaction_history(count, *, body="结果" * 2000):
    calls = [{"id": f"call_{index}", "type": "function", "function": {
        "name": "test.report", "arguments": json.dumps({"index": index, "description": "完整参数" * 100})
    }} for index in range(count)]
    history = [{"role": "system", "content": "固定规则" * 10000},
               {"role": "user", "content": "当前目标不应被精简"},
               {"role": "assistant", "content": "读取分析结果", "tool_calls": calls}]
    for index, call in enumerate(calls):
        result = ToolResult(call_id=call["id"], status=ToolStatus.SUCCESS, output={"body": body},
                            datasets=[f"ds_{index}"], artifacts=[f"art_{index}"], warnings=["采样结果"])
        history.append({"role": "tool", "tool_call_id": call["id"], "content": _tool_observation(result)})
    return history


@pytest.mark.parametrize("count,already_processed,ratio,expected_new", [
    (10, 0, 0.2, 2), (6, 0, 0.2, 2), (50, 40, 0.2, 2), (1, 0, 0.2, 1), (10, 0, 0.4, 4),
])
def test_compaction_uses_oldest_unprocessed_fraction_and_keeps_protocol(count, already_processed, ratio, expected_new):
    original = _compaction_history(count)
    original_json = json.dumps(original, ensure_ascii=False)
    previous_ids = {f"call_{index}" for index in range(already_processed)}
    expected_ids = {f"call_{index}" for index in range(already_processed + expected_new)}
    projected = compact_tool_results(original, token_budget=10**9, ratio=ratio,
                                     run_id="run-test", compacted_ids=set(expected_ids))
    budget = _history_tokens(projected)
    marked = set(previous_ids)
    view = compact_tool_results(original, token_budget=budget, ratio=ratio,
                                run_id="run-test", compacted_ids=marked)

    assert marked == expected_ids
    assert _history_tokens(view) <= budget
    assert view[:3] == original[:3]
    assert len(view) == len(original)
    assert json.dumps(original, ensure_ascii=False) == original_json
    assert [item["tool_call_id"] for item in view[3:]] == [f"call_{index}" for index in range(count)]
    for index, item in enumerate(view[3:]):
        payload = json.loads(item["content"])
        raw = json.loads(original[index + 3]["content"])
        if item["tool_call_id"] in marked:
            assert payload["context_compacted"] is True
            assert payload["output"] is None
            assert payload["result_reference"] == {
                "run_id": "run-test", "tool_call_id": item["tool_call_id"], "source": "checkpoint.protocol_messages",
            }
            assert {key: payload[key] for key in ("status", "datasets", "artifacts", "warnings", "error")} == {
                key: raw[key] for key in ("status", "datasets", "artifacts", "warnings", "error")
            }
        else:
            assert item == original[index + 3]
    assert compact_tool_results(original, token_budget=budget, ratio=ratio,
                                run_id="run-test", compacted_ids=marked) == view
    assert marked == expected_ids
    other_run_ids = set()
    assert compact_tool_results(original, token_budget=_history_tokens(original), ratio=ratio,
                                run_id="another-run", compacted_ids=other_run_ids) == original
    assert other_run_ids == set()


def test_compaction_reestimates_between_groups_and_is_noop_at_budget():
    original = _compaction_history(10)
    marked = set()
    assert compact_tool_results(original, token_budget=_history_tokens(original), ratio=0.2,
                                run_id="run-test", compacted_ids=marked) == original
    assert marked == set()
    target_ids = {f"call_{index}" for index in range(4)}
    expected = compact_tool_results(original, token_budget=10**9, ratio=0.2,
                                   run_id="run-test", compacted_ids=set(target_ids))
    actual = compact_tool_results(original, token_budget=_history_tokens(expected), ratio=0.2,
                                 run_id="run-test", compacted_ids=marked)
    assert marked == target_ids
    assert actual == expected


def test_compaction_keeps_errors_and_does_not_drop_oversized_arguments():
    original = _compaction_history(1)
    payload = json.loads(original[-1]["content"])
    payload.update(status="FAILED", error={"code": "CRS_MISMATCH", "message": "坐标单位不匹配", "details": {"crs": "EPSG:4326"}})
    original[-1]["content"] = json.dumps(payload, ensure_ascii=False)
    original[2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"text": "完整参数" * 10000}, ensure_ascii=False)
    marked = set()
    view = compact_tool_results(original, token_budget=100, ratio=0.2, run_id="run-test", compacted_ids=marked)
    assert marked == {"call_0"}
    assert json.loads(view[-1]["content"])["error"] == payload["error"]
    assert json.loads(view[-1]["content"])["status"] == "FAILED"
    assert view[:3] == original[:3]
    assert _history_tokens(view) > 100  # 只精简结果正文，不能擅自删除调用参数或用户目标。
    assert compact_tool_results(original[:3], token_budget=100, ratio=0.2,
                                run_id="run-test", compacted_ids=set()) == original[:3]


def test_large_observation_and_checkpoint_restore_keep_complete_json(tmp_path):
    _, loop = _loop(tmp_path, None)
    original = _compaction_history(60, body="原始结果" * 5000)
    request = AgentRequest(conversation_id="test", user_id="test-user", user_input="恢复任务")
    built = loop.context.build(request, protocol_messages=original[1:], append_request=False)
    assert built[1:] == original[1:]
    assert len(built) > 48
    assert len(original[-1]["content"]) > 16000
    assert json.loads(built[-1]["content"])["output"] == {"body": "原始结果" * 5000}


@pytest.mark.asyncio
@pytest.mark.parametrize("model_failure", [False, True])
async def test_compaction_checkpoints_restore_originals_and_never_reexecute_tools(tmp_path, monkeypatch, model_failure):
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
            raise TimeoutError("模拟精简后的模型超时")
        return response

    monkeypatch.setattr(adapter, "complete", complete)
    store, loop = _loop(tmp_path, adapter)
    loop.settings.protocol_history_tokens = 1800
    executed = []
    loop.registry.register(ToolMetadata(name="test.large_report", description="large_report", input_schema={"type": "object"}),
                           lambda _args, context: executed.append(context.call_id) or {"output": {"body": "x" * 4000}}, deferred=True)
    request, waiting = await _run(loop, store, "读取四项完整结果")
    assert waiting.error == ("MODEL_UNAVAILABLE" if model_failure else "WAITING_USER")
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    marked = set(checkpoint.state["compacted_tool_call_ids"])
    assert "execute_0" in marked
    assert "execute_3" not in marked
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
        assert all(observations[identifier]["context_compacted"] is True for identifier in marked)
        assert observations["execute_3"]["output"] == {"body": "x" * 4000}
        assert _history_tokens(model_request.messages) <= loop.settings.protocol_history_tokens
        declared = {call["id"] for item in model_request.messages for call in item.get("tool_calls", [])}
        assert observations.keys() == declared
    final = store.latest_checkpoint(result.trace_id)
    assert set(final.state["compacted_tool_call_ids"]) == marked
    assert store.get_tool_call(f"{result.trace_id}:execute_0")[1].output == {"body": "x" * 4000}
