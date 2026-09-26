from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

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
from app.execution.tools import ToolExecutor, ToolRegistry
from app.models import ModelAdapter, ModelRequest, ModelResponse
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
    settings = SimpleNamespace(max_agent_turns=6, max_tool_calls=8, max_tokens=256, tool_context_tokens=2500, tool_context_max_cards=8)
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


async def _run(loop: AgentLoop, store: StateStore, text: str):
    conversation = store.create_conversation("测试")
    request = AgentRequest(conversation_id=conversation.id, user_id="test-user", user_input=text)
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
    assert {tool["function"]["name"] for tool in adapter.requests[0].tools} == {
        "tool.search",
        "dataset.list",
        "dataset.inspect",
        "agent.ask_user",
        "conversation.search_history",
    }


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


@pytest.mark.asyncio
async def test_later_search_preserves_prior_tools_and_empty_search_does_not_clear_them(tmp_path):
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
    assert "vector.buffer" in {tool["function"]["name"] for tool in empty_adapter.requests[2].tools}
    assert empty_store.latest_checkpoint(empty_result.trace_id).state["activated_tool_names"] == ["vector.buffer"]


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
    assert observations["evicted"]["error"]["code"] == "DEFERRED_TOOL_NOT_ACTIVE"
    assert observations["premature"]["error"]["code"] == "DEFERRED_TOOL_NOT_ACTIVE"
    for request in adapter.requests:
        assert sum(loop.registry.is_deferred(item["function"]["name"]) for item in request.tools) <= 8
        assert loop._tool_context_tokens(request.tools, []) <= 2500
    checkpoint = store.latest_checkpoint(result.trace_id)
    assert set(checkpoint.state["discovered_tool_names"]) == set(names)
    assert len(checkpoint.state["activated_tool_names"]) == 8
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
    definitions, cards, active = loop._tool_context(context, [name])
    assert loop._tool_context_tokens(definitions, cards) <= loop.settings.tool_context_tokens
    if budget_delta == 0:
        assert active == {name}
        assert cards == []
        assert next(item["function"]["parameters"] for item in definitions if item["function"]["name"] == name) == schema
    else:
        assert active == set()
        assert [item["name"] for item in cards] == [name]
        messages = loop._tool_context_messages([{"role": "system", "content": "original"}], cards)
        assert "精确查询工具名称" in messages[1]["content"]


def test_cards_and_schemas_share_one_budget_and_permission_filter(tmp_path):
    _, loop = _loop(tmp_path, None)
    names = [f"test.detailed_{index}" for index in range(10)]
    for name in names:
        loop.registry.register(ToolMetadata(name=name, description="Detailed operation", input_schema={
            "type": "object", "properties": {"mode": {"type": "string", "enum": [f"choice_{index}" for index in range(160)]}},
        }), lambda *_: {}, deferred=True)
    loop.registry.register(ToolMetadata(name="test.forbidden", description="Unavailable", required_scopes=["system.admin"]), lambda *_: {}, deferred=True)
    context = loop._discovery_context(AgentRequest(conversation_id="test", user_id="test-user", user_input="test"), loop.services_factory("test-user"))
    definitions, cards, active = loop._tool_context(context, [*names, "test.forbidden"])
    assert cards and active
    visible = active | {item["name"] for item in cards}
    assert len(visible) <= 8
    assert not (active & {item["name"] for item in cards})
    assert "test.forbidden" not in visible
    assert loop._tool_context_tokens(definitions, cards) <= 2500
    assert visible <= set(names[-8:])


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
    assert checkpoint.state["activated_tool_names"] == ["vector.buffer"]
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
