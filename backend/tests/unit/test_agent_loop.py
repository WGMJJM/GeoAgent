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
from app.agent.skills import SKILL_CONTENT_PREFIX, SkillCatalog, skill_messages
from app.core.models import (
    AgentRequest,
    AgentResultStatus,
    Dataset,
    DatasetKind,
    Message,
    RunStatus,
    ToolMetadata,
)
from app.core.tokens import estimate_tokens
from app.execution.tools import ToolExecutor, ToolRegistry
from app.memory import ConversationMemoryService
from app.models import ModelAdapter, ModelRequest, ModelResponse, ModelStreamChunk
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


def _loop(tmp_path, adapter: ModelAdapter | None, datasets: list[Dataset] | None = None, *, skills=None):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    registry = ToolRegistry()
    register_gis_tools(registry)
    trace = TraceRecorder(store)
    executor = ToolExecutor(registry, store, trace)
    settings = SimpleNamespace(max_agent_turns=6, max_tool_calls=8, max_tokens=256,
                               model_input_tokens=128000, tool_result_recent_full=16,
                               conversation_tool_index_limit=8,
                               tool_result_emergency_fraction=0.5,
                               tool_context_tokens=25600, tool_context_max_cards=8,
                               tool_search_regex_results=2, tool_search_chinese_results=1,
                               tool_search_english_results=3,
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
        context_services={"conversation_memory": ConversationMemoryService(store), "skills": skills},
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


def _skills(tmp_path):
    directory = tmp_path / "skills" / "report"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: report\ndescription: 需要时组织结果证据\n---\n区分实际结果与推断", encoding="utf-8")
    (directory / "reference.md").write_text("参考内容", encoding="utf-8")
    return SkillCatalog(directory.parent)


@pytest.mark.asyncio
async def test_skill_reads_are_hidden_internal_actions_and_reuse_context(tmp_path):
    class StreamingSkills(SequenceAdapter):
        async def stream(self, request):
            response = await self.complete(request)
            for character in response.content:
                yield ModelStreamChunk(content=character)
            yield ModelStreamChunk(done=True, input_tokens=100, output_tokens=20)

    adapter = StreamingSkills(
        ModelResponse(content='{"action":"read_skill","name":"report"}'),
        ModelResponse(content='{"action":"read_skill","name":"report","path":"reference.md"}'),
        ModelResponse(content='{"action":"read_skill","name":"report"}'),
        ModelResponse(content='{"answer":"已整理结果"}'),
    )
    catalog = _skills(tmp_path)
    store, loop = _loop(tmp_path, adapter, skills=catalog)
    fragments, usages = [], []

    async def capture(content, usage):
        fragments.append(content)
        usages.append(usage)

    request, result = await _run(loop, store, "需要技能指导时再读取", capture)
    assert result.status is AgentResultStatus.SUCCESS
    assert "".join(fragments) == result.summary == '{"answer":"已整理结果"}'
    assert usages[-1].model_calls == 4
    assert usages[-1].reported_output_tokens == 80
    assert not skill_messages(adapter.requests[0].messages)
    assert "区分实际结果与推断" not in str(adapter.requests[0].messages)
    assert len(skill_messages(adapter.requests[1].messages)) == 1
    assert len(skill_messages(adapter.requests[3].messages)) == 2
    assert all(not any(item["function"]["name"].startswith("skill") for item in sent.tools) for sent in adapter.requests)
    assert not any(name.startswith("skill") for name in loop.registry.names())
    assert store.get_run(result.trace_id).tool_call_count == 0
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert len(checkpoint.state["skill_messages"]) == 2
    assert all("read_skill" not in item.get("content", "") for item in checkpoint.state["protocol_messages"])
    assert store.get_conversation_tool_state(request.conversation_id, user_id=request.user_id) == ([], [])


@pytest.mark.asyncio
async def test_optional_skills_do_not_load_for_direct_tool_use(tmp_path, monkeypatch):
    catalog = _skills(tmp_path)

    def unexpected_read(*_args):
        raise AssertionError("有技能目录不代表需要读取")

    monkeypatch.setattr(catalog, "read", unexpected_read)
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "list", "function": {"name": "dataset.list", "arguments": "{}"}}]),
        ModelResponse(content="没有数据。"),
    )
    store, loop = _loop(tmp_path, adapter, skills=catalog)
    _, result = await _run(loop, store, "查看数据列表")
    assert result.status is AgentResultStatus.SUCCESS
    assert store.get_run(result.trace_id).tool_call_count == 1
    assert all(not skill_messages(sent.messages) for sent in adapter.requests)


@pytest.mark.asyncio
async def test_skill_snapshot_survives_resume_but_does_not_auto_load_in_next_run(tmp_path, monkeypatch):
    catalog = _skills(tmp_path)
    reads = []
    original = catalog.read

    def counted_read(name, path):
        reads.append((name, path))
        return original(name, path)

    monkeypatch.setattr(catalog, "read", counted_read)
    adapter = SequenceAdapter(
        ModelResponse(content='{"action":"read_skill","name":"report"}'),
        ModelResponse(tool_calls=[{"id": "pause", "function": {"name": "agent.ask_user", "arguments": '{"question":"继续吗？"}'}}]),
        ModelResponse(content="按照已读指导完成。"),
        ModelResponse(content="你好。"),
    )
    store, loop = _loop(tmp_path, adapter, skills=catalog)
    request, waiting = await _run(loop, store, "整理报告")
    assert waiting.error == "WAITING_USER"
    snapshot = store.latest_checkpoint(waiting.trace_id)
    assert len(snapshot.state["skill_messages"]) == 1
    catalog.entries["report"].location.write_text("文件修改后不能替换恢复快照", encoding="utf-8")
    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(waiting.trace_id)),
                            resume_from=snapshot, continuation={"type": "user_input", "content": "继续"})
    assert result.status is AgentResultStatus.SUCCESS
    assert reads == [("report", "SKILL.md")]
    assert "区分实际结果与推断" in skill_messages(adapter.requests[2].messages)[0]["content"]
    next_request = AgentRequest(conversation_id=request.conversation_id, user_id=request.user_id, user_input="你好")
    store.save_message(Message(conversation_id=request.conversation_id, role="user", content="你好"))
    await loop.run(next_request, prepared=await loop.prepare_request(next_request))
    assert not skill_messages(adapter.requests[3].messages)
    assert reads == [("report", "SKILL.md")]


@pytest.mark.asyncio
@pytest.mark.parametrize("content,tools,protocol_error", [
    ('{"action":"read_skill","name":"unknown"}', [], False),
    ('{"action":"read_skill","name":"report","path":"../../private.txt"}', [], False),
    ('{"action":"read_skill","name":true}', [], True),
    ('{"action":"read_skill","name":"report"', [], True),
    ('{"action":"read_skill","name":"report"}', [{"id":"mixed", "function":{"name":"dataset.list", "arguments":"{}"}}], True),
])
async def test_invalid_skill_requests_never_execute_tools_or_leak_control_json(tmp_path, content, tools, protocol_error):
    adapter = SequenceAdapter(ModelResponse(content=content, tool_calls=tools), ModelResponse(content="读取不可用，没有执行业务操作。"))
    store, loop = _loop(tmp_path, adapter, skills=_skills(tmp_path))
    fragments = []

    async def capture(text, _usage):
        fragments.append(text)

    _, result = await _run(loop, store, "按需读取", capture)
    assert store.get_run(result.trace_id).tool_call_count == 0
    assert "read_skill" not in "".join(fragments)
    if protocol_error:
        assert result.error == "MODEL_PROTOCOL_ERROR"
        assert len(adapter.requests) == 1
    else:
        assert result.status is AgentResultStatus.SUCCESS
        returned = json.loads(skill_messages(adapter.requests[1].messages)[0]["content"].removeprefix(SKILL_CONTENT_PREFIX))
        assert "error" in returned and "content" not in returned


@pytest.mark.asyncio
async def test_skill_body_counts_toward_input_budget_and_is_not_a_tool_result(tmp_path, monkeypatch):
    adapter = SequenceAdapter(ModelResponse(content='{"action":"read_skill","name":"report"}'))
    store, loop = _loop(tmp_path, adapter, skills=_skills(tmp_path))
    original = adapter.complete

    async def complete(request):
        response = await original(request)
        loop.settings.model_input_tokens = model_input_tokens(request.messages, request.tools, adapter.count_tokens) + 1
        return response

    monkeypatch.setattr(adapter, "complete", complete)
    _, result = await _run(loop, store, "当前请求不能丢失")
    assert result.error == "BUDGET_EXCEEDED"
    assert len(adapter.requests) == 1
    assert store.get_run(result.trace_id).tool_call_count == 0
    snapshot = store.latest_checkpoint(result.trace_id)
    assert "区分实际结果与推断" in snapshot.state["skill_messages"][0]["content"]
    assert snapshot.state["request"]["user_input"] == "当前请求不能丢失"


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
    loop.settings.max_tokens = 12800

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
    assert all(request.max_tokens == loop.settings.max_tokens for request in adapter.requests)
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
async def test_fresh_run_reuses_conversation_tool_capability_without_research(tmp_path):
    tool_name = "test.conversation_capability"
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[{"id": "find", "function": {"name": "tool.search", "arguments": '{"query":"conversation capability"}'}}]),
        ModelResponse(tool_calls=[{"id": "execute", "function": {"name": tool_name, "arguments": "{}"}}]),
        ModelResponse(content="第一次运行完成。"),
        ModelResponse(content="新运行直接看到了此前使用的工具。"),
    )
    store, loop = _loop(tmp_path, adapter)
    loop.registry.register(
        ToolMetadata(name=tool_name, description="conversation capability", input_schema={"type": "object"}),
        lambda _arguments, _context: {"output": {"value": 42}},
        deferred=True,
    )
    conversation = store.create_conversation("跨运行工具", user_id="test-user")
    first_request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="查找并调用工具")
    first = await loop.run(first_request, prepared=await loop.prepare_request(first_request))

    second_request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="继续处理")
    second = await loop.run(second_request, prepared=await loop.prepare_request(second_request))

    assert first.status is AgentResultStatus.SUCCESS
    assert second.status is AgentResultStatus.SUCCESS
    discovered, used = store.get_conversation_tool_state(conversation.id, user_id="test-user")
    assert discovered == [tool_name]
    assert used == [tool_name]
    assert len(adapter.requests) == 4
    assert tool_name in {item["function"]["name"] for item in adapter.requests[3].tools}


@pytest.mark.asyncio
async def test_fresh_run_sees_execution_index_and_reads_one_original_result(tmp_path):
    dataset = Dataset(id="ds_roads", name="roads", kind=DatasetKind.VECTOR, path="roads.gpkg", format="GPKG")
    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[
            {"id": f"list_{index}", "function": {"name": "dataset.list", "arguments": "{}"}}
            for index in range(10)
        ]),
        ModelResponse(content="已经读取数据列表。"),
    )
    store, loop = _loop(tmp_path, adapter, [dataset])
    loop.settings.max_tool_calls = 10
    conversation = store.create_conversation("跨运行结果", user_id="test-user")
    first_request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="列出数据")
    first = await loop.run(first_request, prepared=await loop.prepare_request(first_request))
    source_call_id = f"{first.trace_id}:list_9"
    adapter.responses.extend(
        [
            ModelResponse(
                tool_calls=[{
                    "id": "read_previous",
                    "function": {
                        "name": "conversation.read_tool_result",
                        "arguments": json.dumps({"run_id": first.trace_id, "tool_call_id": source_call_id}),
                    },
                }]
            ),
            ModelResponse(content="已读取上一运行的真实工具结果。"),
        ]
    )

    second_request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input="上一轮实际看到了什么？")
    second = await loop.run(second_request, prepared=await loop.prepare_request(second_request))

    assert second.status is AgentResultStatus.SUCCESS
    execution_context = next(
        item["content"]
        for item in adapter.requests[2].messages
        if item["role"] == "system" and "recent_tool_executions" in item["content"]
    )
    assert first.trace_id in execution_context
    assert source_call_id in execution_context
    assert '"tool_name":"dataset.list"' in execution_context
    assert '"result_body_available":true' in execution_context
    assert "roads" not in execution_context
    execution_index = json.loads(execution_context.split("\n", 1)[1])["recent_tool_executions"]
    assert [item["tool_call_id"] for item in execution_index] == [
        f"{first.trace_id}:list_{index}" for index in range(2, 10)
    ]
    assert store.get_tool_call_record(f"{first.trace_id}:list_0") is not None
    observation = next(
        json.loads(item["content"])
        for item in adapter.requests[3].messages
        if item.get("tool_call_id") == "read_previous"
    )
    assert observation["status"] == "SUCCESS"
    assert observation["output"]["tool_result"]["output"]["datasets"][0]["name"] == "roads"


@pytest.mark.asyncio
async def test_unused_candidates_become_cards_and_used_schemas_survive_resume(tmp_path):
    names = [f"test.candidate_{index}" for index in range(2)]

    def call(name, arguments, call_id):
        return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}

    adapter = SequenceAdapter(
        ModelResponse(tool_calls=[call("tool.search", {"query": "primary_marker", "english_query": "secondary_marker"}, "find")]),
        ModelResponse(tool_calls=[call(names[0], {"value": 1}, "selected_0")]),
        ModelResponse(tool_calls=[call("agent.ask_user", {"question": "继续使用已选择工具吗？"}, "pause")]),
        ModelResponse(tool_calls=[call(names[0], {"value": 2}, "reuse")]),
        ModelResponse(content="已直接复用完整 Schema，未调用候选保留卡片。"),
    )
    store, loop = _loop(tmp_path, adapter)
    executed = []

    def execute(arguments, context):
        executed.append(context.call_id)
        return {"output": {"value": arguments["value"]}}

    for index, name in enumerate(names):
        loop.registry.register(ToolMetadata(name=name, description="primary_marker" if index == 0 else "secondary_marker",
                                           input_schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}),
                               execute, deferred=True)
    request, waiting = await _run(loop, store, "查找候选后只使用选中的工具")
    assert waiting.error == "WAITING_USER"
    assert set(names) <= set(_assert_tool_visibility(loop, adapter.requests[1])["callable"])
    selected = {names[0]}
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    assert set(checkpoint.state["used_tool_names"]) == selected
    assert set(checkpoint.state["activated_tool_names"]) == selected
    visibility = _assert_tool_visibility(loop, adapter.requests[2])
    assert selected <= set(visibility["callable"])
    assert set(names) - selected == {item["name"] for item in visibility["cached"]}
    observations = [json.loads(item["content"]) for item in adapter.requests[2].messages
                    if str(item.get("tool_call_id", "")).startswith("selected_")]
    assert {item["status"] for item in observations} == {"SUCCESS"}

    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(waiting.trace_id)),
                            resume_from=checkpoint, continuation={"type": "user_input", "content": "继续"})
    assert result.status is AgentResultStatus.SUCCESS
    assert executed == [f"{result.trace_id}:selected_0", f"{result.trace_id}:reuse"]
    assert len(adapter.requests) == 5
    assert store.get_run(result.trace_id).tool_call_count == 4
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
    assert observations["restore"]["output"]["source"] == "conversation_cache"
    assert observations["restore"]["output"]["already_callable"] is False
    assert observations["premature"]["error"]["code"] == "DEFERRED_TOOL_NOT_ACTIVE"
    assert executed == [f"{result.trace_id}:first", f"{result.trace_id}:second"]
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert set(checkpoint.state["used_tool_names"]) == set(names)
    assert set(checkpoint.state["activated_tool_names"]) == set(names)
    for model_request in adapter.requests:
        _assert_tool_visibility(loop, model_request)


@pytest.mark.asyncio
async def test_one_bilingual_search_preserves_union_and_executes_each_tool_once(tmp_path):
    english_names = {"test.en_a"}
    chinese_names = {"test.zh_a"}
    expected = english_names | chinese_names
    chinese_query = "唯一中文能力"
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


def test_cards_and_schemas_share_one_budget_and_permission_filter(tmp_path):
    _, loop = _loop(tmp_path, None)
    names = [f"test.detailed_{index}" for index in range(18)]
    for name in names:
        loop.registry.register(ToolMetadata(name=name, description="Detailed operation", input_schema={
            "type": "object", "properties": {"mode": {"type": "string", "enum": [f"choice_{index}" for index in range(160)]}},
        }), lambda *_: {}, deferred=True)
    loop.registry.register(ToolMetadata(name="test.forbidden", description="Unavailable", required_scopes=["system.admin"]), lambda *_: {}, deferred=True)
    context = loop._discovery_context(AgentRequest(conversation_id="test", user_id="test-user", user_input="test"), loop.services_factory("test-user"))
    visible_limit = loop.settings.tool_context_max_cards
    full_definitions = loop._tool_definitions(context, set(names[-visible_limit:]))
    loop.settings.tool_context_tokens = loop._tool_context_tokens(full_definitions, []) // 2
    definitions, cards, active = loop._tool_context(context, [*names, "test.forbidden"], set(names) | {"test.forbidden"})
    assert cards and active
    visible = active | {item["name"] for item in cards}
    assert len(visible) <= visible_limit
    assert not (active & {item["name"] for item in cards})
    assert "test.forbidden" not in visible
    assert loop._tool_context_tokens(definitions, cards) <= loop.settings.tool_context_tokens
    assert visible <= set(names[-visible_limit:])
    messages = prepare_model_messages([{"role": "system", "content": "original"}], definitions, cards)
    visibility = _assert_tool_visibility(loop, ModelRequest(messages=messages, tools=definitions))
    assert "test.forbidden" not in visibility["callable"]
    assert "test.forbidden" not in {item["name"] for item in visibility["cached"]}


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
    [(20, 20)],
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


@pytest.mark.asyncio
async def test_checkpoint_restore_keeps_observations_and_never_reexecutes_tools(tmp_path, monkeypatch):
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
        if len(adapter.requests) == 3:
            raise TimeoutError("模拟工具执行后的模型超时")
        return response

    monkeypatch.setattr(adapter, "complete", complete)
    store, loop = _loop(tmp_path, adapter)
    executed = []
    loop.registry.register(ToolMetadata(name="test.large_report", description="large_report", input_schema={"type": "object"}),
                           lambda _args, context: executed.append(context.call_id) or {"output": {"body": "x" * 4000}}, deferred=True)
    request, waiting = await _run(loop, store, "读取四项完整结果")
    assert waiting.error == "MODEL_UNAVAILABLE"
    checkpoint = store.latest_checkpoint(waiting.trace_id)
    assert checkpoint.state["compacted_tool_call_ids"] == []
    raw = {item["tool_call_id"]: json.loads(item["content"]) for item in checkpoint.state["protocol_messages"] if item["role"] == "tool"}
    assert all(raw[f"execute_{index}"]["output"] == {"body": "x" * 4000} for index in range(4))
    assert all("context_compacted" not in payload for payload in raw.values())
    before_resume = list(executed)
    result = await loop.run(request, prepared=loop.prepare_resume(request, store.get_run(waiting.trace_id)),
                            resume_from=checkpoint, continuation={"type": "technical_resume"})
    assert result.status is AgentResultStatus.SUCCESS
    assert executed == before_resume == [f"{waiting.trace_id}:execute_{index}" for index in range(4)]
    assert len(adapter.requests) == 4
    assert store.get_run(result.trace_id).token_usage.model_calls == 3
    assert store.get_run(result.trace_id).tool_call_count == 5
    for model_request in adapter.requests[2:]:
        observations = {item["tool_call_id"]: json.loads(item["content"]) for item in model_request.messages if item["role"] == "tool"}
        assert {f"execute_{index}" for index in range(4)} <= observations.keys()
        assert all(observations[identifier]["output"] == {"body": "x" * 4000} for identifier in raw if identifier.startswith("execute_"))
        declared = {call["id"] for item in model_request.messages for call in item.get("tool_calls", [])}
        assert observations.keys() == declared
    final = store.latest_checkpoint(result.trace_id)
    assert final.state["compacted_tool_call_ids"] == []
    assert store.get_tool_call(f"{result.trace_id}:execute_0")[1].output == {"body": "x" * 4000}
