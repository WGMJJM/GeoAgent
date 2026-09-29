import asyncio
import json

import pytest

from app.agent.context import SYSTEM_PROMPT, USER_MEMORY_PREFIX, prepare_model_messages
from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Artifact,
    ArtifactKind,
    ConversationMemory,
    ConversationMemoryEntry,
    Dataset,
    DatasetKind,
    Message,
    Run,
    RunStatus,
    Task,
    ToolResult,
    ToolStatus,
    WorkingMemory,
)
from app.models import ModelAdapter, ModelRequest, ModelResponse


class _SummaryModel(ModelAdapter):
    supports_json_object = True

    def __init__(self, *, responder=None, error: Exception | None = None) -> None:
        self.requests: list[ModelRequest] = []
        self.responder = responder
        self.error = error

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        payload = json.loads(request.messages[1]["content"])
        response = self.responder(payload, len(self.requests)) if self.responder else _summary_response(payload, len(self.requests))
        return ModelResponse(content=json.dumps(response, ensure_ascii=False))


def _summary_response(payload: dict, revision: int = 1) -> dict:
    source = next((item["message_id"] for item in payload["messages"] if item["role"] == "user"), payload["messages"][0]["message_id"])
    return {
        "summary": f"滚动摘要版本 {revision}：会话持续围绕研究区域和分析目标。",
        "key_facts": [{"content": f"研究偏好 {revision}", "source_message_id": source}],
        "decisions": [{"content": f"采用方案 {revision}", "source_message_id": source}],
        "unresolved_topics": [],
        "references": [],
    }


def _seed_messages(
    application,
    conversation_id: str,
    *,
    start: int,
    exchanges: int,
    dataset_ids: list[str] | None = None,
    reference_text: str = "",
) -> list[Message]:
    messages: list[Message] = []
    for index in range(start, start + exchanges):
        user_message = Message(
            id=f"msg-user-{index}",
            conversation_id=conversation_id,
            role="user",
            content=f"第 {index} 轮讨论上海区域的分析背景和决定。 {reference_text if index == start else ''}",
            dataset_ids=dataset_ids or [],
        )
        assistant_message = Message(
            id=f"msg-assistant-{index}",
            conversation_id=conversation_id,
            role="assistant",
            content=f"第 {index} 轮回答：已记录研究背景。",
        )
        application.store.save_message(user_message)
        application.store.save_message(assistant_message)
        messages.extend((user_message, assistant_message))
    return messages


async def _save_messages(application, conversation_id: str, user_id: str, user_text: str, assistant_text: str) -> None:
    await application.conversation_memory.save_message(
        Message(conversation_id=conversation_id, role="user", content=user_text), user_id=user_id
    )
    await application.conversation_memory.save_message(
        Message(conversation_id=conversation_id, role="assistant", content=assistant_text), user_id=user_id
    )


def test_message_persistence_precedes_summary_and_failure_does_not_trigger_it(application, monkeypatch):
    conversation_id, user_id = "memory-save-order", "memory-save-owner"
    application.conversations.ensure(conversation_id, "统一保存入口", user_id=user_id)
    model = _SummaryModel()
    application.model_adapter = model
    user = Message(conversation_id=conversation_id, role="user", content="分析这个数据")
    assistant = Message(conversation_id=conversation_id, role="assistant", content="已读取检查结果")
    summarized = []

    async def summarize(identifier, owner, adapter):
        assert (identifier, owner, adapter) == (conversation_id, user_id, model)
        assert application.conversation_memory.list_messages(identifier, user_id=owner) == [user, assistant]
        summarized.append(identifier)
        return False

    monkeypatch.setattr(application.conversation_memory.summarizer, "summarize_pending", summarize)
    asyncio.run(application.conversation_memory.save_message(user, user_id=user_id))
    assert summarized == []

    def failed_save(_message):
        raise OSError("模拟持久化失败")

    with monkeypatch.context() as failed_persistence:
        failed_persistence.setattr(application.store, "save_message", failed_save)
        with pytest.raises(OSError, match="持久化失败"):
            asyncio.run(application.conversation_memory.save_message(assistant, user_id=user_id))
    assert summarized == []
    assert application.conversation_memory.list_messages(conversation_id, user_id=user_id) == [user]

    asyncio.run(application.conversation_memory.save_message(assistant, user_id=user_id))
    assert summarized == [conversation_id]
    assert model.requests == []


def test_incremental_summary_advances_by_message_range_and_preserves_raw_history(application):
    conversation_id, user_id = "summary-incremental", "summary-user"
    application.conversations.ensure(conversation_id, "增量摘要", user_id=user_id)
    model = _SummaryModel()
    application.model_adapter = model

    async def run_rounds(start: int, count: int) -> None:
        for index in range(start, start + count):
            await _save_messages(
                application, conversation_id, user_id,
                f"第 {index} 轮：上海区域分析偏好。", f"已记录第 {index} 轮讨论。",
            )

    asyncio.run(run_rounds(0, 9))
    assert model.requests == []
    asyncio.run(run_rounds(9, 1))
    first = application.conversation_memory.get(conversation_id, user_id)
    assert first is not None
    assert first.summary_version == 1
    assert first.summarized_through_message_id is not None
    assert first.summary_updated_at is not None
    assert len(model.requests) == 1
    first_payload = json.loads(model.requests[0].messages[1]["content"])
    first_ids = {item["message_id"] for item in first_payload["messages"]}
    assert len(first_ids) == 12
    request = AgentRequest(user_id=user_id, conversation_id=conversation_id, user_input="继续分析")
    first_context = application.agent_loop.context.build(request)
    first_tail = application.store.list_messages_after(conversation_id, first.summarized_through_message_id)
    assert len(first_tail) == 8
    assert first_ids.isdisjoint(item.id for item in first_tail)
    assert first_context[2:-1] == [{"role": item.role, "content": item.content} for item in first_tail]
    assert first_context[-1] == {"role": "user", "content": request.user_input}

    asyncio.run(run_rounds(10, 1))
    between_context = application.agent_loop.context.build(request)
    between_tail = application.store.list_messages_after(conversation_id, first.summarized_through_message_id)
    assert len(between_tail) == 10
    assert between_context[2:-1] == [{"role": item.role, "content": item.content} for item in between_tail]
    assert len(model.requests) == 1

    asyncio.run(run_rounds(11, 4))
    assert len(model.requests) == 1  # 仅 10 条较早消息，还未达到 12 条门槛。
    asyncio.run(run_rounds(15, 1))
    second = application.conversation_memory.get(conversation_id, user_id)
    assert second is not None
    assert second.summary_version == 2
    assert len(model.requests) == 2
    second_payload = json.loads(model.requests[1].messages[1]["content"])
    assert second_payload["old_summary"] == first.summary
    assert not first_ids.intersection(item["message_id"] for item in second_payload["messages"])
    assert len(second_payload["messages"]) == 12
    assert len(application.store.list_messages(conversation_id, limit=100)) == 32
    second_tail = application.store.list_messages_after(conversation_id, second.summarized_through_message_id)
    second_context = application.agent_loop.context.build(request)
    assert len(second_tail) == 8
    assert second_context[2:-1] == [{"role": item.role, "content": item.content} for item in second_tail]
    context_memory = json.loads(second_context[1]["content"].split("\n", 1)[1])["conversation_memory"]
    assert context_memory["summary"] == second.summary
    assert context_memory["summary_version"] == second.summary_version
    assert context_memory["summarized_through_message_id"] == second.summarized_through_message_id
    assert len(model.requests) == 2

    stale = application.conversation_memory.get(conversation_id, user_id)
    assert stale is not None
    application.store.save_conversation_memory(stale.model_copy(update={"summary": "过期结构化更新不应覆盖摘要"}))
    persisted = application.conversation_memory.get(conversation_id, user_id)
    assert persisted is not None
    assert persisted.summary == second.summary
    assert persisted.summary_version == second.summary_version


def test_emergency_summary_bypasses_trigger_but_keeps_latest_eight_complete(application):
    conversation_id, user_id = "summary-emergency", "summary-emergency-owner"
    application.conversations.ensure(conversation_id, "超限摘要", user_id=user_id)
    long_message = Message(id="msg-emergency-long", conversation_id=conversation_id, role="user", content="完整长消息" * 1500)
    old_answer = Message(id="msg-emergency-answer", conversation_id=conversation_id, role="assistant", content="已记录。")
    application.store.save_message(long_message)
    application.store.save_message(old_answer)
    raw = [long_message, old_answer, *_seed_messages(application, conversation_id, start=0, exchanges=4)]
    model = _SummaryModel()

    assert asyncio.run(application.conversation_memory.summarizer.summarize_all_pending(
        conversation_id, user_id, model, protected_message_id=raw[-2].id
    ))
    memory = application.conversation_memory.get(conversation_id, user_id)
    assert memory is not None
    assert memory.summarized_through_message_id == raw[1].id
    assert len(model.requests) >= 1
    payload = json.loads(model.requests[0].messages[1]["content"])
    assert payload["messages"][0]["content"] == long_message.content
    _, tail = application.conversation_memory.load_context(conversation_id, user_id=user_id, recent_message_limit=24)
    assert tail == raw[-8:]
    assert application.store.list_messages(conversation_id, limit=100) == raw


@pytest.mark.parametrize("older_tokens", [6399, 6400])
def test_summary_token_boundary_excludes_recent_messages_and_preserves_uncovered_batch(application, monkeypatch, older_tokens):
    conversation_id, user_id = "summary-token-boundary", "summary-token-owner"
    application.conversations.ensure(conversation_id, "摘要 token 边界", user_id=user_id)
    raw = _seed_messages(application, conversation_id, start=0, exchanges=5)
    model = _SummaryModel()
    counts = {raw[0].content: 3200, raw[1].content: older_tokens - 3200}
    # 最近 8 条即使很长，也不应计入较早消息的触发阈值。
    monkeypatch.setattr(model, "count_tokens", lambda value: counts.get(value, 10000))
    summarizer = application.conversation_memory.summarizer
    assert (summarizer.recent_messages, summarizer.trigger_messages, summarizer.trigger_tokens) == (8, 12, 6400)
    committed = asyncio.run(summarizer.summarize_pending(conversation_id, user_id, model))
    assert committed is (older_tokens == 6400)
    memory = application.conversation_memory.get(conversation_id, user_id)
    assert memory is not None
    if not committed:
        assert model.requests == []
        assert memory.summarized_through_message_id is None
    else:
        payload = json.loads(model.requests[0].messages[1]["content"])
        # 沿用单批约 6k 的限制，只推进实际处理部分，不丢弃剩余候选。
        assert [item["message_id"] for item in payload["messages"]] == [raw[0].id]
        assert memory.summarized_through_message_id == raw[0].id
        _, history = application.conversation_memory.load_context(conversation_id, user_id=user_id, recent_message_limit=24)
        assert history == raw[1:]
        assert history[-8:] == raw[-8:]
    assert application.store.list_messages(conversation_id, limit=100) == raw


@pytest.mark.parametrize("memory_state", ["missing", "summary_only", "cursor_only"])
def test_context_without_complete_summary_boundary_keeps_recent_history(application, memory_state):
    conversation_id, user_id = "context-without-coverage", "context-owner"
    application.conversations.ensure(conversation_id, "没有有效覆盖边界", user_id=user_id)
    raw = _seed_messages(application, conversation_id, start=0, exchanges=13)
    if memory_state != "missing":
        application.store.save_conversation_memory(
            ConversationMemory(
                conversation_id=conversation_id,
                user_id=user_id,
                summary="旧摘要没有覆盖游标" if memory_state == "summary_only" else "",
                summary_version=1,
                summarized_through_message_id=raw[17].id if memory_state == "cursor_only" else None,
            )
        )
    context = application.agent_loop.context.build(
        AgentRequest(user_id=user_id, conversation_id=conversation_id, user_input="继续")
    )
    history = [item for item in context[:-1] if item["role"] != "system"]
    assert history == [{"role": item.role, "content": item.content} for item in raw[-24:]]


def test_context_keeps_uncovered_backlog_after_summary_failure(application):
    conversation_id, user_id = "context-summary-backlog", "context-backlog-owner"
    application.conversations.ensure(conversation_id, "摘要失败积压", user_id=user_id)
    raw = _seed_messages(application, conversation_id, start=0, exchanges=10)
    summarizer = application.conversation_memory.summarizer
    assert asyncio.run(summarizer.summarize_pending(conversation_id, user_id, _SummaryModel()))
    old = application.conversation_memory.get(conversation_id, user_id)
    assert old is not None
    raw.extend(_seed_messages(application, conversation_id, start=10, exchanges=14))
    failing_model = _SummaryModel(error=RuntimeError("model unavailable"))
    application.model_adapter = failing_model
    pending_assistant = Message(conversation_id=conversation_id, role="assistant", content="这轮继续记录讨论。")
    asyncio.run(application.conversation_memory.save_message(pending_assistant, user_id=user_id))
    raw.append(pending_assistant)

    current_user = Message(conversation_id=conversation_id, role="user", content="继续当前分析")
    application.store.save_message(current_user)
    raw.append(current_user)
    context = application.agent_loop.context.build(
        AgentRequest(user_id=user_id, conversation_id=conversation_id, user_input=current_user.content)
    )
    assert len(failing_model.requests) == 1
    memory = json.loads(context[1]["content"].split("\n", 1)[1])["conversation_memory"]
    assert memory["summary"] == old.summary
    assert memory["summary_version"] == old.summary_version
    assert memory["summarized_through_message_id"] == old.summarized_through_message_id
    uncovered = raw[12:]
    assert len(uncovered) == 38
    assert context[2:] == [{"role": item.role, "content": item.content} for item in uncovered]
    assert application.store.list_messages(conversation_id, limit=100) == raw


def test_context_uses_one_summary_snapshot_during_concurrent_coverage_advance(application, monkeypatch):
    conversation_id, user_id = "context-summary-snapshot", "context-snapshot-owner"
    application.conversations.ensure(conversation_id, "摘要快照", user_id=user_id)
    raw = _seed_messages(application, conversation_id, start=0, exchanges=10)
    assert asyncio.run(application.conversation_memory.summarizer.summarize_pending(conversation_id, user_id, _SummaryModel()))
    old = application.conversation_memory.get(conversation_id, user_id)
    assert old is not None
    raw.extend(_seed_messages(application, conversation_id, start=10, exchanges=4))
    original_list_after = application.store.list_messages_after
    calls = []

    def advance_before_history_read(identifier, through_message_id):
        calls.append(through_message_id)
        assert application.store.commit_conversation_summary(
            conversation_id=conversation_id,
            user_id=user_id,
            expected_version=old.summary_version,
            expected_through_message_id=old.summarized_through_message_id,
            through_message_id=raw[15].id,
            summary="并发生成的新摘要",
            key_facts=[],
            decisions=[],
            unresolved_topics=[],
            important_references=[],
        )
        return original_list_after(identifier, through_message_id)

    monkeypatch.setattr(application.store, "list_messages_after", advance_before_history_read)
    context = application.agent_loop.context.build(
        AgentRequest(user_id=user_id, conversation_id=conversation_id, user_input="继续")
    )
    memory = json.loads(context[1]["content"].split("\n", 1)[1])["conversation_memory"]
    assert calls == [old.summarized_through_message_id]
    assert memory["summary"] == old.summary
    assert memory["summary_version"] == old.summary_version
    assert context[2:-1] == [{"role": item.role, "content": item.content} for item in raw[12:]]
    latest = application.conversation_memory.get(conversation_id, user_id)
    assert latest is not None
    assert latest.summary_version == old.summary_version + 1
    assert latest.summarized_through_message_id == raw[15].id


def test_summary_boundary_does_not_filter_resumed_tool_protocol(application, monkeypatch):
    conversation_id, user_id = "context-resumed-protocol", "context-protocol-owner"
    application.conversations.ensure(conversation_id, "恢复工具协议", user_id=user_id)
    raw = _seed_messages(application, conversation_id, start=0, exchanges=10)
    assert asyncio.run(application.conversation_memory.summarizer.summarize_pending(conversation_id, user_id, _SummaryModel()))
    def unexpected_history_read(*_args):
        raise AssertionError("恢复时应保留 Checkpoint 协议，不再读取原始会话历史。")

    monkeypatch.setattr(application.store, "list_messages_after", unexpected_history_read)
    protocol = [
        {"role": "user", "content": raw[0].content},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "inspect-call", "type": "function", "function": {"name": "dataset.inspect", "arguments": '{"dataset_id":"ds-example"}'}}
        ]},
        {"role": "tool", "content": '{"dataset_id":"ds-example","name":"DEM"}', "tool_call_id": "inspect-call"},
    ]
    run = Run(conversation_id=conversation_id, agent_id="main", status=RunStatus.RUNNING)
    application.store.save_run(run)
    context = application.agent_loop.context.build(
        AgentRequest(user_id=user_id, conversation_id=conversation_id, user_input="继续"),
        run=run,
        protocol_messages=protocol,
        append_request=False,
    )
    assert context[2:] == protocol


def test_summary_trigger_runs_after_assistant_messages_are_persisted(application):
    conversation_id, user_id = "summary-entry-modes", "summary-owner"
    application.conversations.ensure(conversation_id, "入口摘要", user_id=user_id)
    model = _SummaryModel()
    application.model_adapter = model
    _seed_messages(application, conversation_id, start=0, exchanges=9)
    asyncio.run(
        _save_messages(application, conversation_id, user_id, "补充一个研究决定", "已记录。")
    )
    after_first = application.conversation_memory.get(conversation_id, user_id)
    assert after_first is not None and after_first.summary_version == 1

    _seed_messages(application, conversation_id, start=9, exchanges=7)
    asyncio.run(
        _save_messages(application, conversation_id, user_id, "再补充一个分析决定", "已记录第二项。")
    )
    after_second = application.conversation_memory.get(conversation_id, user_id)
    assert after_second is not None and after_second.summary_version == 2


@pytest.mark.parametrize("failure", ["exception", "invalid_json", "timeout"])
def test_summary_failure_keeps_old_summary_and_does_not_break_exchange(application, failure):
    conversation_id, user_id = f"summary-failure-{failure}", "summary-failure-owner"
    application.conversations.ensure(conversation_id, "摘要失败", user_id=user_id)
    _seed_messages(application, conversation_id, start=0, exchanges=10)
    old = ConversationMemory(conversation_id=conversation_id, user_id=user_id, summary="既有摘要", summary_version=3)
    application.store.save_conversation_memory(old)

    class FailingModel(ModelAdapter):
        supports_json_object = True

        async def complete(self, request: ModelRequest) -> ModelResponse:
            if failure == "exception":
                raise RuntimeError("model unavailable")
            if failure == "timeout":
                await asyncio.sleep(0.05)
            return ModelResponse(content="not-json" if failure == "invalid_json" else json.dumps(_summary_response(json.loads(request.messages[1]["content"]))))

    if failure == "timeout":
        application.conversation_memory.summarizer.timeout_seconds = 0.005
    application.model_adapter = FailingModel()
    asyncio.run(_save_messages(application, conversation_id, user_id, "继续补充研究背景", "已记录。"))

    persisted = application.conversation_memory.get(conversation_id, user_id)
    assert persisted is not None
    assert persisted.summary == "既有摘要"
    assert persisted.summary_version == 3
    assert persisted.summarized_through_message_id is None
    assert len(application.store.list_messages(conversation_id, limit=100)) == 22


def test_concurrent_summary_commit_is_idempotent_and_merges_with_normal_updates(application):
    conversation_id, user_id = "summary-concurrent", "summary-concurrent-owner"
    application.conversations.ensure(conversation_id, "并发摘要", user_id=user_id)
    _seed_messages(application, conversation_id, start=0, exchanges=10)
    stale = application.conversation_memory.get_or_create(conversation_id, user_id)

    class BarrierModel(ModelAdapter):
        supports_json_object = True

        def __init__(self):
            self.calls = 0
            self.ready = asyncio.Event()

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls += 1
            if self.calls >= 2:
                self.ready.set()
            await self.ready.wait()
            payload = json.loads(request.messages[1]["content"])
            return ModelResponse(content=json.dumps(_summary_response(payload, 1), ensure_ascii=False))

    model = BarrierModel()
    async def update_concurrently():
        return await asyncio.gather(
            application.conversation_memory.summarizer.summarize_pending(conversation_id, user_id, model),
            application.conversation_memory.summarizer.summarize_pending(conversation_id, user_id, model),
        )

    outcomes = asyncio.run(update_concurrently())

    assert sorted(outcomes) == [False, True]
    assert model.calls == 2
    application.store.save_conversation_memory(stale.model_copy(update={"summary": "stale", "key_facts": []}))
    current = application.conversation_memory.get(conversation_id, user_id)
    assert current is not None
    assert current.summary.startswith("滚动摘要版本")
    assert current.summary_version == 1
    assert current.key_facts
    assert asyncio.run(application.conversation_memory.summarizer.summarize_pending(conversation_id, user_id, model)) is False


def test_summary_resource_references_and_run_status_are_database_verified(application):
    conversation_id, user_id = "summary-verified-resources", "summary-resource-owner"
    application.conversations.ensure(conversation_id, "校验引用", user_id=user_id)
    dataset = Dataset(
        id="ds_verified",
        name="verified-roads.geojson",
        kind=DatasetKind.VECTOR,
        path="private/verified-roads.geojson",
        format="GeoJSON",
        owner_user_id=user_id,
    )
    run = Run(
        id="run_verified",
        conversation_id=conversation_id,
        agent_id="main",
        status=RunStatus.FAILED,
        metadata={"result": {"summary": "实际运行失败"}},
    )
    artifact = Artifact(id="art_verified", name="checked-output.txt", kind=ArtifactKind.REPORT, run_id=run.id, owner_user_id=user_id)
    application.store.save_dataset(dataset)
    application.store.save_run(run)
    application.store.save_artifact(artifact)
    _seed_messages(
        application,
        conversation_id,
        start=0,
        exchanges=10,
        dataset_ids=[dataset.id],
        reference_text=f"核查 {dataset.id}、{artifact.id} 和 {run.id}，也排除 ds_ghost。",
    )

    def response(payload, _revision):
        source = payload["messages"][0]["message_id"]
        return {
            "summary": f"数据库记录显示 {run.id} 状态为 FAILED。",
            "key_facts": [{"content": "本轮有引用需要核实", "source_message_id": source}],
            "decisions": [],
            "unresolved_topics": [],
            "references": [
                {"type": "dataset", "id": dataset.id, "source_message_id": source},
                {"type": "artifact", "id": artifact.id, "source_message_id": source},
                {"type": "run", "id": run.id, "source_message_id": source},
                {"type": "dataset", "id": "ds_ghost", "source_message_id": source},
                {"type": "run", "id": "run_ghost", "source_message_id": source},
            ],
        }

    model = _SummaryModel(responder=response)
    committed = asyncio.run(application.conversation_memory.summarizer.summarize_pending(conversation_id, user_id, model))
    assert committed is True
    memory = application.conversation_memory.get(conversation_id, user_id)
    assert memory is not None
    refs = {(item.reference_type, item.reference_id): item.content for item in memory.important_references}
    assert set(refs) == {("dataset", dataset.id), ("artifact", artifact.id), ("run", run.id)}
    assert refs[("run", run.id)] == "运行状态：FAILED"
    assert "ds_ghost" not in memory.summary


def test_history_search_is_conversation_scoped_and_enters_context(application):
    first_id, second_id, user_id = "history-first", "history-second", "history-owner"
    application.conversations.ensure(first_id, "第一段历史", user_id=user_id)
    application.conversations.ensure(second_id, "另一段历史", user_id=user_id)
    old = Message(id="history-message", conversation_id=first_id, role="user", content="研究范围定位在上海浦东并使用米制投影。")
    other = Message(id="other-history-message", conversation_id=second_id, role="user", content="研究范围定位在上海浦东，但属于另一个对话。")
    application.store.save_message(old)
    application.store.save_message(other)

    matches = application.conversation_memory.search_history(first_id, user_id, "上海浦东", limit=4)
    assert [item.id for item in matches] == [old.id]
    assert application.conversation_memory.search_history(first_id, "other-user", "上海浦东") == []
    assert application.conversation_memory.list_messages(first_id, user_id="other-user") == []
    assert application.conversation_memory.load_context(first_id, user_id="other-user", recent_message_limit=24) == (None, [])
    with pytest.raises(PermissionError):
        asyncio.run(application.conversation_memory.save_message(
            Message(conversation_id=first_id, role="user", content="不应写入别人的会话"), user_id="other-user",
        ))
    assert application.conversation_memory.list_messages(first_id, user_id=user_id) == [old]
    task = Task(goal="当前任务", conversation_id=first_id)
    application.store.save_task(task)
    run = Run(conversation_id=first_id, task_id=task.id, agent_id="agent-loop", status=RunStatus.RUNNING, metadata={"original_request": "分析浦东投影"})
    application.store.save_run(run)
    dataset = Dataset(id="ds-context-history", name="浦东边界", kind=DatasetKind.VECTOR, path="pudong.geojson", format="GeoJSON", owner_user_id=user_id)
    application.store.save_dataset(dataset)
    application.store.save_working_memory(WorkingMemory(task_id=task.id, conversation_id=first_id, active_dataset_ids=[dataset.id]))
    application.profile.update(user_id, {"response_style": "concise"})
    application.store.save_conversation_memory(
        ConversationMemory(
            conversation_id=first_id,
            user_id=user_id,
            summary="摘要事实",
            summary_version=1,
            key_facts=[ConversationMemoryEntry(content="研究范围=上海浦东", source_message_id=old.id)],
        )
    )
    request = AgentRequest(user_id=user_id, conversation_id=first_id, user_input="继续分析上海浦东", dataset_ids=[dataset.id])
    context = application.agent_loop.context.build(request, run=run)
    user_memory = context[1]
    assert user_memory["content"].startswith(USER_MEMORY_PREFIX)
    assert json.loads(user_memory["content"].removeprefix(USER_MEMORY_PREFIX)) == {
        "user_profile": {"language": "zh-CN", "response_style": "concise", "measurement_system": "metric", "preferred_output_format": None},
    }
    payload = json.loads(context[2]["content"].split("\n", 1)[1])

    assert payload["conversation_memory"]["summary"] == "摘要事实"
    assert "user_profile" not in payload
    assert "project_memory" not in payload
    assert payload["selected_datasets"][0]["id"] == dataset.id
    assert payload["current_task"]["goal"] == "当前任务"
    assert payload["current_task"]["working_memory"]["active_dataset_ids"] == [dataset.id]
    assert context[-2]["content"] == old.content
    assert any(item["function"]["name"] == "conversation.search_history" for item in application.agent_loop._tool_definitions())

    # 工具状态准备不会改写固定系统规则、用户记忆或原始调用/结果。
    invocation = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "history-inspect", "type": "function", "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": dataset.id})}},
    ]}
    result = ToolResult(call_id="history-inspect", status=ToolStatus.SUCCESS, output={"body": "x" * 4000})
    observation = {"role": "tool", "tool_call_id": result.call_id, "content": result.model_dump_json()}
    original = [*context, invocation, observation]
    prepared = prepare_model_messages(original, application.agent_loop._tool_definitions(), [])
    assert prepared[0] == context[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert prepared[2] == user_memory
    assert prepared[3] == context[2]
    assert prepared[-2] == invocation
    assert json.loads(prepared[-1]["content"])["output"] == result.output
    assert json.loads(original[-1]["content"])["output"] == result.output


def test_resumed_run_assistant_persistence_updates_memory_after_recovery(application):
    conversation_id, user_id = "summary-resumed-run", "summary-resume-owner"
    application.conversations.ensure(conversation_id, "恢复后摘要", user_id=user_id)
    _seed_messages(application, conversation_id, start=0, exchanges=10)
    run = Run(id="run-resumed-summary", conversation_id=conversation_id, agent_id="main", status=RunStatus.COMPLETED)
    application.store.save_run(run)
    result = AgentResult(agent_id="main", status=AgentResultStatus.SUCCESS, summary="恢复后任务已完成", trace_id=run.id)

    async def fake_wait(_run_id: str) -> AgentResult:
        return result

    original_wait = application.conversations.run_manager.wait
    application.conversations.run_manager.wait = fake_wait
    application.model_adapter = _SummaryModel()
    try:
        returned = asyncio.run(application.conversations.wait(run.id, force_assistant=True))
    finally:
        application.conversations.run_manager.wait = original_wait

    assert returned is result
    saved = application.conversation_memory.get(conversation_id, user_id)
    assert saved is not None
    assert saved.summary_version == 1
    assert saved.summarized_through_message_id is not None
    assistant_messages = [item for item in application.store.list_messages(conversation_id, limit=100) if item.role == "assistant" and item.run_id == run.id]
    assert len(assistant_messages) == 1
