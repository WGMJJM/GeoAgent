from __future__ import annotations

import asyncio
import json
import math
from datetime import timedelta

import pytest

from app.agent.context import ANSWER_PREFIX, ContextBuilder
from app.core.models import (
    AgentRequest,
    AgentResultStatus,
    ConversationMemory,
    Message,
    Run,
    ToolCall,
    ToolResult,
    ToolStatus,
    UserMemory,
    UserMemoryEntry,
    UserProfile,
    utc_now,
)
from app.memory import ConversationMemoryService, UserMemoryService
from app.memory.user_memory import LONG_TERM_MEMORY_PREFIX
from app.models import ModelResponse
from app.state import StateStore
from tests.test_context_compaction import (
    MockModel,
    add_batch,
    checkpoint,
    make_loop,
    measured_input,
)


@pytest.fixture
def store(tmp_path):
    value = StateStore(tmp_path / "memory.sqlite3")
    value.initialize()
    return value


def context(store, text="请记住：我的长期项目是城市洪涝分析", user_id="alice"):
    conversation = store.create_conversation(user_id=user_id)
    request = AgentRequest(conversation_id=conversation.id, user_id=user_id, user_input=text)
    message = Message(conversation_id=conversation.id, role="user", content=text)
    store.save_message(message)
    run = Run(agent_id="test", conversation_id=conversation.id, metadata={"original_request_message_id": message.id})
    store.save_run(run)
    return request, run


def seed(store, request, entries):
    with store.transaction() as db:
        store._save_user_memory(db, UserMemory(user_id=request.user_id, entries=entries))


def entry(request, number=0, *, expired=False, length=20):
    now = utc_now() - timedelta(days=5)
    return UserMemoryEntry(
        content=f"{number}:" + "记" * length, scope="用户全局",
        source_conversation_id=request.conversation_id, source_ids=["message:old"],
        created_at=now, updated_at=now + timedelta(seconds=number),
        expires_at=now + timedelta(days=1 if expired else 30),
    )


class MemoryModel(MockModel):
    def __init__(self, delta=None, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    async def complete(self, request):
        if "memory_update_mode" not in request.messages[-1]["content"]:
            return await super().complete(request)
        self.memory_requests.append(request)
        payload = json.loads(request.messages[-1]["content"])
        result = self.delta(payload) if self.delta else {
            "upserts": [{"content": "用户长期项目是城市洪涝分析", "scope": "用户全局",
                         "source_ids": [payload["current_user_message_id"]]}], "removals": [],
        }
        return ModelResponse(content=json.dumps(result, ensure_ascii=False))


@pytest.mark.asyncio
async def test_remember_tool_resumes_committed_event_without_another_model_call(store, tmp_path, monkeypatch):
    model = MemoryModel()
    model.responses = [ModelResponse(tool_calls=[{
        "id": "remember_1", "type": "function", "function": {"name": "memory.remember", "arguments": "{}"},
    }]), ModelResponse(content=ANSWER_PREFIX + "已记住。")]
    loop, request, prepared, _, _ = await make_loop(store, tmp_path, model)
    store.save_message(Message(conversation_id=request.conversation_id, role="user", content="请记住我的长期项目是城市洪涝分析"))
    loop.settings.completion_review_enabled = True
    loop.settings.max_tool_calls = 1
    update = loop.user_memory.update

    async def interrupt_after_commit(*args, **kwargs):
        await update(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(loop.user_memory, "update", interrupt_after_commit)
    with pytest.raises(asyncio.CancelledError):
        await loop.run(request, prepared=prepared)
    saved = store.latest_checkpoint(prepared.run.id)
    assert saved.phase == "tool_pending"
    monkeypatch.setattr(loop.user_memory, "update", update)
    result = await loop.run(request, prepared=prepared, resume_from=saved)
    assert result.status is AgentResultStatus.SUCCESS
    assert len(model.memory_requests) == 1
    assert store.get_user_memory(request.user_id).entries[0].content == "用户长期项目是城市洪涝分析"
    assert not loop._requires_completion_review(store.get_run(prepared.run.id))
    assert not any(message["content"].startswith(LONG_TERM_MEMORY_PREFIX) for message in model.requests[-1].messages)


@pytest.mark.asyncio
async def test_automatic_no_change_preserves_expiry_but_explicit_repeat_renews(store):
    request, run = context(store)
    original = entry(request)
    seed(store, request, [original])
    service = UserMemoryService(store)
    model = MemoryModel(lambda payload: {"upserts": [{
        "id": original.id, "content": original.content, "scope": original.scope,
        "source_ids": [payload["current_user_message_id"]],
    }], "removals": []})
    first = await service.update(request, run, model, event_id="auto", mode="compaction")
    assert first["status"] == "unchanged"
    assert store.get_user_memory(request.user_id).entries == [original]
    assert await service.update(request, run, model, event_id="auto", mode="compaction") == first
    assert len(model.memory_requests) == 1
    renewed = await service.update(request, run, model, event_id="explicit", mode="explicit")
    saved = store.get_user_memory(request.user_id).entries[0]
    assert renewed["status"] == "updated"
    assert saved.created_at == original.created_at
    assert saved.expires_at == saved.updated_at + timedelta(days=30)
    assert saved.expires_at > original.expires_at
    forget = MemoryModel(lambda payload: {"upserts": [], "removals": [{
        "id": saved.id, "source_ids": [payload["current_user_message_id"]],
    }]})
    await service.update(request, run, forget, event_id="forget", mode="explicit")
    assert not store.get_user_memory(request.user_id).entries


@pytest.mark.asyncio
async def test_snapshot_is_persistent_and_manual_settings_take_priority(store):
    request, run = context(store)
    live, expired = entry(request), entry(request, 1, expired=True)
    seed(store, request, [expired, live])
    store.save_user_profile(UserProfile(user_id=request.user_id, preferred_output_format="GeoPackage"))
    memory = UserMemoryService(store)
    builder = ContextBuilder(store, conversation_memory=ConversationMemoryService(store), user_memory=memory)
    first = builder.build(request, run=run)
    snapshot = next(item for item in first if item["content"].startswith(LONG_TERM_MEMORY_PREFIX))
    assert json.loads(snapshot["content"].split("\n", 1)[1]) == [{"content": live.content, "scope": live.scope}]
    assert store.get_user_memory(request.user_id).entries == [live]
    assert "手动设置优先" in snapshot["content"]
    assert any('"preferred_output_format":"GeoPackage"' in item["content"] for item in first)
    await memory.update(request, run, MemoryModel(), event_id="new", mode="explicit")
    # 普通写入和摘要提交都必须保留冻结快照。
    store.save_conversation_memory(ConversationMemory(conversation_id=request.conversation_id, user_id=request.user_id))
    assert store.commit_conversation_summary(
        conversation_id=request.conversation_id, user_id=request.user_id, expected_version=0,
        expected_through_message_id=None, through_message_id=run.metadata["original_request_message_id"],
        summary="summary", key_facts=[], decisions=[], important_references=[], unresolved_topics=[],
    )
    rebuilt = UserMemoryService(store).snapshot_message(request)
    assert rebuilt == snapshot
    fresh_request, _ = context(store)
    assert "城市洪涝分析" in memory.snapshot_message(fresh_request)["content"]
    stranger, _ = context(store, user_id="bob")
    assert memory.snapshot_message(stranger) is None


@pytest.mark.asyncio
async def test_entry_count_and_serialized_token_limits_evict_oldest(store):
    request, run = context(store)
    originals = [entry(request, number, length=145) for number in range(30)]
    seed(store, request, originals)
    model = MemoryModel()
    model.count_tokens = lambda text: 1
    service = UserMemoryService(store)
    await service.update(request, run, model, event_id="count", mode="explicit")
    saved = store.get_user_memory(request.user_id)
    assert len(saved.entries) == 30
    assert originals[0].id not in {item.id for item in saved.entries}
    assert originals[1].id in {item.id for item in saved.entries}
    model.count_tokens = len
    model.delta = lambda payload: {"upserts": [], "removals": []}
    await service.update(request, run, model, event_id="tokens", mode="compaction")
    saved = store.get_user_memory(request.user_id)
    message = service.snapshot_message(request, len)
    assert len(json.dumps(message, ensure_ascii=False, separators=(",", ":"))) <= 3000
    assert len(saved.entries) < 30
    original_expiries = {item.id: item.expires_at for item in originals}
    assert all(item.expires_at == original_expiries[item.id] for item in saved.entries if item.id in original_expiries)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", ["too_long", "unverified_source"])
async def test_invalid_model_change_keeps_old_memory(store, bad_value):
    request, run = context(store)
    original = entry(request)
    seed(store, request, [original])
    model = MemoryModel(lambda payload: {"upserts": [{
        "content": "x" * 151 if bad_value == "too_long" else "fact", "scope": "用户全局",
        "source_ids": [payload["current_user_message_id"], "tool:invented"] if bad_value == "unverified_source" else [payload["current_user_message_id"]],
    }], "removals": []})
    result = await UserMemoryService(store).update(request, run, model, event_id="bad", mode="explicit")
    assert result["status"] == "failed"
    assert store.get_user_memory(request.user_id).entries == [original]


@pytest.mark.asyncio
async def test_concurrent_chats_use_cas_and_cannot_write_another_user(store):
    request, run = context(store)
    other_request, other_run = context(store)
    started, release = asyncio.Event(), asyncio.Event()

    class SlowModel(MemoryModel):
        async def complete(self, request):
            started.set()
            await release.wait()
            return await super().complete(request)

    first = asyncio.create_task(UserMemoryService(store).update(request, run, SlowModel(), event_id="first", mode="explicit"))
    await started.wait()
    second = await UserMemoryService(store).update(other_request, other_run, MemoryModel(), event_id="second", mode="explicit")
    release.set()
    assert (await first)["status"] == "conflict"
    assert second["status"] == "updated"
    saved = store.get_user_memory(request.user_id)
    assert saved.entries[0].source_conversation_id == other_request.conversation_id
    intruder = request.model_copy(update={"user_id": "bob"})
    blocked_model = MemoryModel()
    rejected = await UserMemoryService(store).update(intruder, run, blocked_model, event_id="intruder", mode="explicit")
    assert rejected["status"] == "failed"
    assert not blocked_model.memory_requests
    assert not store.get_user_memory("bob").entries


@pytest.mark.asyncio
async def test_compaction_memory_failure_does_not_fail_main_task(store, tmp_path):
    class FailingMemoryModel(MockModel):
        async def complete(self, request):
            if "memory_update_mode" in request.messages[-1]["content"]:
                self.memory_requests.append(request)
                raise ConnectionError("memory unavailable")
            return await super().complete(request)

    loop, request, prepared, protocol, model = await make_loop(store, tmp_path, FailingMemoryModel(), old_interactions=4, recent_tokens=1)
    for number in range(8):
        add_batch(store, prepared.run, protocol, number)
    # 失败的工具结果不能进入长期记忆证据。
    failed = ToolCall(id="failed", name="gis.inspect", run_id=prepared.run.id)
    store.save_tool_call(failed, ToolResult(call_id=failed.id, status=ToolStatus.FAILED, output="unverified"))
    run = store.get_run(prepared.run.id)
    loop.settings.model_input_tokens = math.ceil(measured_input(loop, request, run, protocol, model) / 0.90)
    result = await loop.run(request, prepared=prepared, resume_from=checkpoint(loop, request, run, protocol))
    assert result.status is AgentResultStatus.SUCCESS
    assert len(model.summary_requests) > 1
    assert len(model.memory_requests) == 1
    payload = json.loads(model.memory_requests[0].messages[-1]["content"])
    assert all(item["id"] != "tool:failed" for item in payload["evidence"])
    assert any(item["id"].startswith("tool:") for item in payload["evidence"])
    assert not store.get_user_memory(request.user_id).entries
