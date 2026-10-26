from __future__ import annotations

import asyncio
import json
import math

import pytest

from app.agent.context import (
    ANSWER_PREFIX,
    _completed_tool_batches,
    compact_model_input,
    model_input_tokens,
    narrow_model_input,
    prepare_model_messages,
    protected_tool_result_ids,
)
from app.agent.loop import AgentLoop, _tool_observation
from app.config import Settings
from app.core.models import (
    AgentRequest,
    AgentResultStatus,
    Checkpoint,
    Dataset,
    DatasetKind,
    Message,
    Run,
    RunStatus,
    Task,
    ToolCall,
    ToolError,
    ToolResult,
    ToolStatus,
)
from app.execution.tools import ToolExecutor, ToolRegistry
from app.memory import ConversationMemoryService, ConversationSummarizer
from app.models import ModelAdapter, ModelRequest, ModelResponse, effective_input_budget
from app.models.config import ModelProfile
from app.observability import TraceRecorder
from app.run.checkpoints import RunCheckpointCodec
from app.run.recovery import RecoveryController
from app.state import StateStore


class MockModel(ModelAdapter):
    supports_json_object = True

    def __init__(self, responses=(), *, fail_once=False):
        self.responses = list(responses)
        self.requests = []
        self.summary_requests = []
        self.memory_requests = []
        self.fail_once = fail_once

    def count_tokens(self, value):
        return len(value)

    async def complete(self, request: ModelRequest):
        try:
            payload = json.loads(request.messages[-1]["content"])
        except (ValueError, KeyError):
            payload = {}
        if isinstance(payload, dict) and "old_summary" in payload:
            self.summary_requests.append(request.model_copy(deep=True))
            source_id = payload["messages"][0]["message_id"]
            return ModelResponse(content=json.dumps({
                "summary": "Previous analysis goal and progress retained.",
                "key_facts": [{"content": "Keep analysis goal", "source_message_id": source_id}],
                "decisions": [{"content": "Use metres", "source_message_id": source_id}],
                "unresolved_topics": [{"content": "Check accuracy", "source_message_id": source_id}],
                "references": [],
            }))
        if isinstance(payload, dict) and "memory_update_mode" in payload:
            self.memory_requests.append(request.model_copy(deep=True))
            return ModelResponse(content='{"upserts":[],"removals":[]}')
        self.requests.append(request.model_copy(deep=True))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("simulated transport failure")
        return self.responses.pop(0) if self.responses else ModelResponse(content=ANSWER_PREFIX + "done")


@pytest.fixture
def store(tmp_path):
    value = StateStore(tmp_path / "state.sqlite3")
    value.initialize()
    return value


def add_batch(store, run, messages, number, *, sizes=(2000,), name="gis.inspect", outputs=None):
    calls, results = [], []
    for ordinal, size in enumerate(sizes):
        provider_id = f"batch_{number}_{ordinal}"
        call = ToolCall(id=f"{run.id}:{provider_id}", run_id=run.id, name=name)
        output = outputs[ordinal] if outputs is not None else "x" * size
        result = ToolResult(call_id=call.id, status=ToolStatus.SUCCESS, output=output)
        store.save_tool_call(call, result)
        calls.append({"id": provider_id, "type": "function", "function": {"name": name, "arguments": "{}"}})
        results.append({"role": "tool", "tool_call_id": provider_id, "content": _tool_observation(result)})
    messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    messages.extend(results)
    return [item["id"] for item in calls]


def project(store, run, messages, compacted=None, **options):
    return compact_model_input(
        messages, run_id=run.id, compacted_ids=compacted if compacted is not None else set(),
        summarized_ids=options.pop("summarized_ids", set()), preview_tokens=options.pop("preview_tokens", 12800),
        stored_result_ids=store.list_tool_result_ids(run.id), count_tokens=len, **options,
    )


def observations(messages):
    return {item["tool_call_id"]: json.loads(item["content"]) for item in messages if item["role"] == "tool"}


async def make_loop(store, tmp_path, model=None, *, old_interactions=0, recent_tokens=6400):
    model = model or MockModel()
    conversation = store.create_conversation(user_id="user_test")
    for number in range(old_interactions):
        old_user = Message(conversation_id=conversation.id, role="user", content=f"old goal {number} " + "u" * 6000)
        store.save_message(old_user)
        old_run = Run(agent_id="test", conversation_id=conversation.id, status=RunStatus.COMPLETED,
                      metadata={"original_request_message_id": old_user.id})
        store.save_run(old_run)
        store.save_message(Message(conversation_id=conversation.id, role="assistant", content="a" * 6000, run_id=old_run.id))
    request = AgentRequest(conversation_id=conversation.id, user_id="user_test", user_input="current goal")
    store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))
    settings = Settings(_env_file=None, root=tmp_path, completion_review_enabled=False,
                        model_input_tokens=1000000, tool_context_tokens=100000,
                        summary_recent_tokens=recent_tokens)
    registry = ToolRegistry()
    trace = TraceRecorder(store)
    executor = ToolExecutor(registry, store, trace)
    memory = ConversationMemoryService(store, summarizer=ConversationSummarizer(store, summary_recent_tokens=recent_tokens))
    loop = AgentLoop(store, registry, executor, trace, settings, lambda _: model, lambda _: {},
                     context_services={"conversation_memory": memory})
    prepared = await loop.prepare_request(request)
    protocol = [item for item in loop.context.build(request, run=prepared.run) if item["role"] in {"user", "assistant", "tool"}]
    run = prepared.run.model_copy(update={"metadata": {**prepared.run.metadata, "conversation_history_count": len(protocol) - 1}})
    store.save_run(run)
    return loop, request, prepared, protocol, model


def checkpoint(loop, request, run, protocol):
    loop._save_checkpoint(request, run, protocol, run.metadata["original_request_message_id"], "tool_observation", set(), [])
    return loop.store.latest_checkpoint(run.id)


def measured_input(loop, request, run, protocol, model):
    messages = loop.context.build(request, run=run, protocol_messages=protocol, append_request=False)
    context = loop.policy.discovery_context(authenticated_user=bool(request.user_id), services={})
    tools, cards, _ = loop._tool_context(context, [], set(), count_tokens=model.count_tokens)
    return model_input_tokens(prepare_model_messages(messages, tools, cards), tools, model.count_tokens)


@pytest.mark.asyncio
async def test_below_trigger_keeps_normal_results_and_skips_history(store, tmp_path):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, old_interactions=3)
    for number in range(8):
        add_batch(store, prepared.run, protocol, number)
    run = store.get_run(prepared.run.id)
    raw_tokens = measured_input(loop, request, run, protocol, model)
    loop.settings.model_input_tokens = raw_tokens * 2
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert result.status is AgentResultStatus.SUCCESS
    assert not model.summary_requests
    assert all(value["output"] == "x" * 2000 for value in observations(model.requests[0].messages).values())
    assert not store.latest_checkpoint(run.id).state["compacted_tool_call_ids"]
    assert model_input_tokens(model.requests[0].messages, model.requests[0].tools, model.count_tokens) == store.get_run(run.id).token_usage.local_input_tokens


@pytest.mark.asyncio
async def test_history_is_summarized_before_any_tool_compaction(store, tmp_path):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, old_interactions=10, recent_tokens=1)
    add_batch(store, prepared.run, protocol, 0)
    run = store.get_run(prepared.run.id)
    original_messages = store.list_messages_after(request.conversation_id, None)
    loop.settings.model_input_tokens = math.ceil(measured_input(loop, request, run, protocol, model) / 0.90)
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert result.status is AgentResultStatus.SUCCESS
    assert len(model.summary_requests) > 1
    assert len(model.memory_requests) == 1
    assert not store.latest_checkpoint(run.id).state["compacted_tool_call_ids"]
    actual = model_input_tokens(model.requests[0].messages, model.requests[0].tools, model.count_tokens)
    assert actual <= loop.settings.model_input_tokens * 0.20
    memory = store.get_conversation_memory_for_user(request.conversation_id, request.user_id)
    assert memory.summary_version == len(model.summary_requests)
    assert memory.summarized_through_message_id != run.metadata["original_request_message_id"]
    assert memory.key_facts and memory.decisions and memory.unresolved_topics
    assert store.list_messages_after(request.conversation_id, None) == original_messages
    assert observations(model.requests[0].messages)["batch_0_0"]["output"] == "x" * 2000


@pytest.mark.asyncio
async def test_tool_stage_continues_after_summary_drops_below_trigger(store, tmp_path):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, old_interactions=4, recent_tokens=1)
    for number in range(15):
        add_batch(store, prepared.run, protocol, number)
    run = store.get_run(prepared.run.id)
    loop.settings.model_input_tokens = math.ceil(measured_input(loop, request, run, protocol, model) / 0.90)
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert result.status is AgentResultStatus.SUCCESS
    assert model.summary_requests
    assert len(store.latest_checkpoint(run.id).state["compacted_tool_call_ids"]) == 2
    actual = model_input_tokens(model.requests[0].messages, model.requests[0].tools, model.count_tokens)
    assert loop.settings.model_input_tokens * 0.20 < actual < loop.settings.model_input_tokens * 0.80


@pytest.mark.asyncio
@pytest.mark.parametrize("hard_block", [False, True])
async def test_one_batch_only_even_if_still_above_trigger(store, tmp_path, hard_block):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path)
    for number in range(15):
        add_batch(store, prepared.run, protocol, number, sizes=(500 if number < 10 else 8000,))
    run = store.get_run(prepared.run.id)
    raw_tokens = measured_input(loop, request, run, protocol, model)
    loop.settings.model_input_tokens = int(raw_tokens * 0.95) if hard_block else math.ceil(raw_tokens / 0.90)
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    saved = store.latest_checkpoint(run.id)
    assert saved.state["compacted_tool_call_ids"] == ["batch_0_0", "batch_1_0"]
    assert len(observations(saved.state["protocol_messages"])) == 15
    assert all(value["output"] is not None for value in observations(saved.state["protocol_messages"]).values())
    if hard_block:
        assert result.error == "BUDGET_EXCEEDED"
        assert not model.requests
    else:
        assert result.status is AgentResultStatus.SUCCESS
        actual = model_input_tokens(model.requests[0].messages, model.requests[0].tools, model.count_tokens)
        assert loop.settings.model_input_tokens * 0.80 < actual <= loop.settings.model_input_tokens
        view = observations(model.requests[0].messages)
        assert view["batch_2_0"]["output"] == "x" * 500
        assert all(view[f"batch_{number}_0"]["output"] == "x" * 8000 for number in range(10, 15))


@pytest.mark.parametrize("candidate_count", [0, 1, 4, 5, 6, 10, 11])
def test_frozen_candidate_count_and_ceiling(store, candidate_count):
    run = Run(agent_id="test")
    store.save_run(run)
    messages, selected = [], set()
    for number in range(candidate_count + 5):
        add_batch(store, run, messages, number)
    view = project(store, run, messages, selected, compact=True)
    assert len(selected) == math.ceil(candidate_count * 0.20)
    assert {item["tool_call_id"] for item in view if item["role"] == "tool"} == set(observations(messages))
    assert len(_completed_tool_batches(view)) == candidate_count + 5


def test_stop_early_when_target_reached_and_use_largest_in_oldest_batch(store):
    run = Run(agent_id="test")
    store.save_run(run)
    messages, selected = [], set()
    add_batch(store, run, messages, 0, sizes=(1000, 7000, 2000, 2000, 2000, 2000))
    for number in range(1, 6):
        add_batch(store, run, messages, number, sizes=(200, 200))
    target = model_input_tokens(messages, [], len) - 3000
    project(store, run, messages, selected, compact=True, target_tokens=target)
    assert selected == {"batch_0_1"}


def test_five_complete_parallel_batches_and_incomplete_protocol_protected(store):
    run = Run(agent_id="test")
    store.save_run(run)
    messages = []
    for number in range(7):
        add_batch(store, run, messages, number, sizes=(1000, 1000))
    add_batch(store, run, messages, 7, sizes=(1000, 1000))
    messages.pop()
    add_batch(store, run, messages, 8)
    payload = observations(messages)["batch_8_0"]
    payload["status"] = "BLOCKED"
    payload["error"] = {"code": "APPROVAL_REQUIRED"}
    messages[-1]["content"] = json.dumps(payload)
    protected = protected_tool_result_ids(messages)
    assert all(f"batch_{number}_{ordinal}" in protected for number in range(2, 7) for ordinal in range(2))
    assert {"batch_7_0", "batch_7_1", "batch_8_0"} <= protected
    assert "batch_1_0" not in protected
    selected = set()
    view = project(store, run, messages, selected, compact=True)
    assert len(selected) == 1
    assert len(view) == len(messages)


def test_missing_persistence_and_zero_savings_are_not_candidates(store):
    run = Run(agent_id="test")
    store.save_run(run)
    messages, selected = [], set()
    add_batch(store, run, messages, 0, sizes=(0,))
    add_batch(store, run, messages, 1)
    messages[-1]["content"] = json.dumps({"call_id": "provider_id_only", "status": "SUCCESS", "output": "raw" * 1000})
    for number in range(2, 7):
        add_batch(store, run, messages, number)
    view = project(store, run, messages, selected, compact=True)
    assert not selected
    assert observations(view)["batch_1_0"]["output"] == "raw" * 1000


def test_active_dependencies_and_metadata_references_preserved(store):
    run = Run(agent_id="test")
    store.save_run(run)
    messages, selected = [], set()
    add_batch(store, run, messages, 0, outputs=[{"count": 42, "crs": "EPSG:3857", "coordinates": [1] * 2000}])
    original = observations(messages)["batch_0_0"]
    original.update(datasets=["ds_result"], artifacts=["art_map"], warnings=["check units"],
                    status="PARTIAL_SUCCESS", error=ToolError(code="PARTIAL", message="some rows invalid").model_dump(mode="json"))
    messages[-1]["content"] = json.dumps(original)
    for number in range(1, 6):
        add_batch(store, run, messages, number)
    pending = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "pending", "function": {"name": "gis.use", "arguments": json.dumps({"dataset_id": "ds_result"})},
    }]}
    protected_view = project(store, run, messages + [pending], selected, compact=True)
    assert not selected
    assert observations(protected_view)["batch_0_0"]["output"] == original["output"]
    view = project(store, run, messages, selected, compact=True)
    compacted = observations(view)["batch_0_0"]
    assert compacted["datasets"] == ["ds_result"]
    assert compacted["artifacts"] == ["art_map"]
    assert compacted["error"] == original["error"] and compacted["warnings"] == original["warnings"]
    assert compacted["status"] == "PARTIAL_SUCCESS"
    assert compacted["output"] == {"count": 42, "crs": "EPSG:3857"}
    assert compacted["result_reference"] == {"run_id": run.id, "tool_call_id": f"{run.id}:batch_0_0"}


def test_fresh_read_bypasses_preview_until_consumed_and_restores_legacy_protocol(store):
    run = Run(agent_id="test")
    store.save_run(run)
    messages, selected = [], set()
    body = {"source_run_id": "source", "tool_call": {"id": "full_original_id"},
            "tool_result": {"output": "r" * 20000}}
    add_batch(store, run, messages, 0, name="conversation.read_tool_result", outputs=[body])
    for number in range(1, 6):
        add_batch(store, run, messages, number)
    view = project(store, run, messages, selected, compact=True, summarized_ids={"batch_0_0"})
    assert selected == {"batch_0_0"}
    assert observations(view)["batch_0_0"]["output"] == body
    view = project(store, run, messages, selected, compact=True, consumed_read_ids={"batch_0_0"})
    assert selected == {"batch_0_0"}
    assert observations(view)["batch_0_0"]["result_reference"] == {"run_id": "source", "tool_call_id": "full_original_id"}
    assert store.get_tool_call_record(f"{run.id}:batch_0_0")[1].output == body


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_read", [False, True])
async def test_hard_budget_never_deletes_user_or_fresh_read(store, tmp_path, oversized_read):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path)
    if oversized_read:
        add_batch(store, prepared.run, protocol, 0, name="conversation.read_tool_result", outputs=[{
            "source_run_id": "source", "tool_call": {"id": "original"}, "tool_result": {"output": "z" * 30000},
        }])
        loop.settings.tool_result_preview_tokens = 50
    run = store.get_run(prepared.run.id)
    loop.settings.model_input_tokens = measured_input(loop, request, run, protocol, model) - 1
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert result.error == "BUDGET_EXCEEDED"
    assert not model.requests
    saved = store.latest_checkpoint(run.id).state["protocol_messages"]
    assert saved == protocol
    assert narrow_model_input(saved, [], input_budget_tokens=1, recent_messages=1, recent_results=1, count_tokens=len) == saved


def test_single_huge_output_preview_preserves_original_and_resources(store):
    run = Run(agent_id="test")
    store.save_run(run)
    messages = []
    add_batch(store, run, messages, 0, outputs=[{"count": 3, "rows": "x" * 50000}])
    before = store.get_tool_call_record(f"{run.id}:batch_0_0")[1].model_dump(mode="json")
    selected = set()
    view = project(store, run, messages, selected, preview_tokens=200)
    payload = observations(view)["batch_0_0"]
    assert payload["output_truncated"]
    assert payload["output_metadata"] == {"count": 3}
    assert len(json.dumps(payload["output"], ensure_ascii=False)) <= 200
    assert payload["result_reference"]["tool_call_id"] == before["call_id"]
    assert not selected
    assert store.get_tool_call_record(before["call_id"])[1].model_dump(mode="json") == before


@pytest.mark.asyncio
async def test_resume_does_not_take_another_batch_without_real_progress(store, tmp_path):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, MockModel(fail_once=True))
    for number in range(15):
        add_batch(store, prepared.run, protocol, number, sizes=(500 if number < 10 else 8000,))
    run = store.get_run(prepared.run.id)
    loop.settings.model_input_tokens = math.ceil(measured_input(loop, request, run, protocol, model) / 0.90)
    first = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert first.error == "MODEL_UNAVAILABLE"
    saved = store.latest_checkpoint(run.id)
    selected = saved.state["compacted_tool_call_ids"]
    assert len(selected) == 2
    second = await loop.run(request, prepared=prepared, resume_from=saved)
    assert second.status is AgentResultStatus.SUCCESS
    assert store.latest_checkpoint(run.id).state["compacted_tool_call_ids"] == selected
    assert observations(model.requests[0].messages) == observations(model.requests[1].messages)
    saved = store.latest_checkpoint(run.id)
    changed = list(saved.state["protocol_messages"])
    add_batch(store, run, changed, 15, sizes=(500,))
    store.save_checkpoint(saved.model_copy(update={"state": {**saved.state, "protocol_messages": changed}}))
    third = await loop.run(request, prepared=prepared, resume_from=store.latest_checkpoint(run.id))
    assert third.status is AgentResultStatus.SUCCESS
    assert len(store.latest_checkpoint(run.id).state["compacted_tool_call_ids"]) > len(selected)


@pytest.mark.asyncio
async def test_transport_retry_reuses_same_compaction_batch(store, tmp_path):
    class RetryModel(MockModel):
        async def complete(self, request):
            if "memory_update_mode" in request.messages[-1]["content"]:
                return await super().complete(request)
            if not self.requests:
                self.requests.append(request.model_copy(deep=True))
                raise ConnectionError("temporary")
            return await super().complete(request)

    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, RetryModel())
    loop.settings.retry_delay_seconds = 0
    loop.recovery = RecoveryController(store, loop.settings, loop.trace)
    for number in range(15):
        add_batch(store, prepared.run, protocol, number, sizes=(500 if number < 10 else 8000,))
    run = store.get_run(prepared.run.id)
    loop.settings.model_input_tokens = math.ceil(measured_input(loop, request, run, protocol, model) / 0.90)
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert result.status is AgentResultStatus.SUCCESS
    assert len(model.requests) == 2
    assert model.requests[0].messages == model.requests[1].messages
    assert len(store.latest_checkpoint(run.id).state["compacted_tool_call_ids"]) == 2


@pytest.mark.asyncio
async def test_interrupted_history_summary_resumes_unfinished_single_batch(store, tmp_path, monkeypatch):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, old_interactions=4, recent_tokens=1)
    for number in range(15):
        add_batch(store, prepared.run, protocol, number)
    run = store.get_run(prepared.run.id)
    loop.settings.model_input_tokens = math.ceil(measured_input(loop, request, run, protocol, model) / 0.90)
    memory = loop.context.conversation_memory
    summarize = memory.compact_history_before

    async def interrupted(*args, **kwargs):
        assert await summarize(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(memory, "compact_history_before", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    saved = store.latest_checkpoint(run.id)
    assert saved.state["context_compaction"]["tool_batch_applied"] is False
    assert not saved.state["compacted_tool_call_ids"]
    turn_count = store.get_run(run.id).turn_count
    monkeypatch.setattr(memory, "compact_history_before", summarize)
    result = await loop.run(request, prepared=prepared, resume_from=saved)
    assert result.status is AgentResultStatus.SUCCESS
    saved = store.latest_checkpoint(run.id)
    assert saved.state["context_compaction"]["tool_batch_applied"] is True
    assert len(saved.state["compacted_tool_call_ids"]) == 2
    assert store.get_run(run.id).turn_count == turn_count


@pytest.mark.asyncio
@pytest.mark.parametrize("source_scope", ["same", "other_conversation", "other_user"])
async def test_explicit_read_is_persisted_consumed_and_scoped(store, tmp_path, source_scope):
    model = MockModel()
    loop, request, prepared, protocol, _ = await make_loop(store, tmp_path, model)
    source_conversation = request.conversation_id
    if source_scope != "same":
        source_conversation = store.create_conversation(user_id="other" if source_scope == "other_user" else request.user_id).id
    source = Run(agent_id="source", conversation_id=source_conversation, status=RunStatus.COMPLETED)
    store.save_run(source)
    original_call = ToolCall(id="full_persisted_original_id", name="gis.inspect", run_id=source.id)
    original = ToolResult(call_id=original_call.id, status=ToolStatus.SUCCESS, output="original" * 3000)
    store.save_tool_call(original_call, original)
    model.responses = [ModelResponse(tool_calls=[{
        "id": "read", "function": {"name": "conversation.read_tool_result", "arguments": json.dumps({
            "run_id": source.id, "tool_call_id": original_call.id,
        })},
    }])]
    loop.settings.tool_result_preview_tokens = 30
    result = await loop.run(request, prepared=prepared)
    assert result.status is AgentResultStatus.SUCCESS
    read_call, read_result = store.get_tool_call_record(f"{prepared.run.id}:read")
    assert read_call.run_id == prepared.run.id
    saved = store.latest_checkpoint(prepared.run.id)
    if source_scope == "same":
        assert read_result.output["tool_result"]["output"] == original.output
        assert observations(model.requests[1].messages)["read"]["output"] == read_result.output
        assert "read" in saved.state["context_compaction"]["consumed_read_ids"]
    else:
        assert read_result.error.code == "TOOL_RESULT_NOT_FOUND"
        assert "read" not in saved.state["context_compaction"].get("consumed_read_ids", [])


@pytest.mark.asyncio
async def test_fresh_read_survives_failed_model_call_and_checkpoint_restart(store, tmp_path):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, MockModel(fail_once=True))
    body = {"source_run_id": "source", "tool_call": {"id": "original"}, "tool_result": {"output": "full" * 6000}}
    add_batch(store, prepared.run, protocol, 0, name="conversation.read_tool_result", outputs=[body])
    loop.settings.tool_result_preview_tokens = 20
    run = store.get_run(prepared.run.id)
    first = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert first.error == "MODEL_UNAVAILABLE"
    saved = store.latest_checkpoint(run.id)
    assert not saved.state["context_compaction"].get("consumed_read_ids")
    second = await loop.run(request, prepared=prepared, resume_from=saved)
    assert second.status is AgentResultStatus.SUCCESS
    assert all(observations(item.messages)["batch_0_0"]["output"] == body for item in model.requests)
    assert store.latest_checkpoint(run.id).state["context_compaction"]["consumed_read_ids"] == ["batch_0_0"]


def test_read_deduplicates_only_identical_original_body_in_model_view(store):
    run = Run(agent_id="test")
    store.save_run(run)
    protocol = []
    add_batch(store, run, protocol, 0, sizes=(2000,))
    original_call, original_result = store.get_tool_call_record(f"{run.id}:batch_0_0")
    body = {"source_run_id": run.id, "tool_call": original_call.model_dump(mode="json"),
            "tool_result": original_result.model_dump(mode="json")}
    add_batch(store, run, protocol, 1, name="conversation.read_tool_result", outputs=[body])
    selected = set()
    view = observations(project(store, run, protocol, selected))
    assert view["batch_0_0"]["context_compacted"]
    assert view["batch_1_0"]["output"] == body
    assert not selected
    assert observations(protocol)["batch_0_0"]["output"] == "x" * 2000


@pytest.mark.asyncio
async def test_results_are_not_removed_when_read_channel_unavailable(store, tmp_path):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path)
    for number in range(10):
        add_batch(store, prepared.run, protocol, number)
    run = store.get_run(prepared.run.id).model_copy(update={"tool_call_count": loop.settings.max_tool_calls})
    store.save_run(run)
    loop.settings.model_input_tokens = 40000
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert result.status is AgentResultStatus.SUCCESS
    assert not store.latest_checkpoint(run.id).state["compacted_tool_call_ids"]
    assert all(value["output"] == "x" * 2000 for value in observations(model.requests[0].messages).values())


@pytest.mark.asyncio
@pytest.mark.parametrize("protected_status", [RunStatus.WAITING_APPROVAL, RunStatus.INTERRUPTED, RunStatus.WAITING_USER])
async def test_summary_cannot_cross_unfinished_history(store, tmp_path, protected_status):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, old_interactions=5, recent_tokens=1)
    old = next(run for run in store.list_runs_for_conversation(request.conversation_id) if run.id != prepared.run.id
               and run.metadata["original_request_message_id"] == store.list_messages_after(request.conversation_id, None)[0].id)
    store.save_run(old.model_copy(update={"status": protected_status}))
    changed = await loop.context.conversation_memory.compact_history_before(
        request, model, prepared.run.metadata["original_request_message_id"], recent_tokens=1,
    )
    assert not changed and not model.summary_requests


@pytest.mark.asyncio
async def test_current_task_history_and_dataset_lineage_remain_protected(store, tmp_path):
    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, old_interactions=4, recent_tokens=1)
    task = Task(goal="continuing analysis", conversation_id=request.conversation_id)
    store.save_task(task)
    current = store.get_run(prepared.run.id).model_copy(update={"task_id": task.id})
    store.save_run(current)
    first_message = store.list_messages_after(request.conversation_id, None)[0]
    for old in store.list_runs_for_conversation(request.conversation_id):
        if old.metadata.get("original_request_message_id") == first_message.id:
            store.save_run(old.model_copy(update={"task_id": task.id}))
    assert not await loop.context.conversation_memory.compact_history_before(
        request, model, current.metadata["original_request_message_id"], recent_tokens=1,
    )
    store.save_dataset(Dataset(id="ds_child", name="child", kind=DatasetKind.VECTOR, format="geojson",
                               path="D:/unchanged/child.geojson", owner_user_id=request.user_id,
                               source_dataset_ids=["ds_parent"]))
    pending = [{"role": "assistant", "tool_calls": [{"id": "pending", "function": {
        "name": "gis.use", "arguments": json.dumps({"dataset_id": "ds_child"}),
    }}]}]
    refs = loop._active_tool_references(pending, [], request)
    assert {"ds_child", "ds_parent", "D:/unchanged/child.geojson"} <= refs


def test_input_budget_reserves_output_only_for_known_window():
    model = MockModel()
    assert effective_input_budget(256000, model, 25600) == 256000
    profile = ModelProfile(id="small", label="small", model="test", context_window_tokens=32000)
    model.config = profile.as_config()
    assert effective_input_budget(256000, model, 4000) == 28000
    assert effective_input_budget(10000, model, 4000) == 10000
    assert effective_input_budget(256000, model, 33000) == 0
    legacy = Settings(_env_file=None, tool_result_emergency_fraction=0.5, emergency_recent_messages=8)
    assert legacy.tool_compaction_batch_ratio == 0.20
    assert RunCheckpointCodec.normalize_state({})["schema_version"] == 0


def test_summary_cas_rejects_stale_update_and_checkpoint_preserves_state(store):
    conversation = store.create_conversation(user_id="user_test")
    memory = ConversationMemoryService(store).get_or_create(conversation.id, "user_test")
    message = Message(conversation_id=conversation.id, role="user", content="goal")
    store.save_message(message)
    values = dict(conversation_id=conversation.id, user_id="user_test", expected_version=memory.summary_version,
                  expected_through_message_id=None, through_message_id=message.id, summary="summary",
                  key_facts=[], decisions=[], important_references=[], unresolved_topics=[])
    assert store.commit_conversation_summary(**values)
    assert not store.commit_conversation_summary(**values)
    run = Run(agent_id="test", conversation_id=conversation.id)
    store.save_run(run)
    state = {"last_attempt_fingerprint": "stable", "consumed_read_ids": ["read"]}
    store.save_checkpoint(Checkpoint(run_id=run.id, phase="model_pending", state={"context_compaction": state}))
    assert store.latest_checkpoint(run.id).state["context_compaction"] == state
