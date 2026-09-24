import asyncio

from app.auth.approval import ApprovalService
from app.core.models import (
    RiskLevel,
    Run,
    RunStatus,
    ToolCall,
    ToolExecutionStatus,
    ToolMetadata,
    ToolStatus,
)
from app.execution.tools.executor import ToolExecutor
from app.execution.tools.registry import ToolRegistry
from app.observability import EventBus, TraceRecorder
from app.state import StateStore


def test_completed_tool_call_reuses_result_without_repeating_side_effect(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    run = Run(id="run_tool_idempotent", agent_id="main", status=RunStatus.RUNNING)
    store.save_run(run)

    side_effects: list[dict] = []

    def handler(arguments, _context):
        side_effects.append(arguments)
        return {"output": {"value": arguments["value"]}}

    registry = ToolRegistry()
    registry.register(ToolMetadata(name="test.side_effect", description="测试一次性副作用"), handler)
    executor = ToolExecutor(registry, store, TraceRecorder(store, EventBus()))
    call = ToolCall(id="call_idempotent", run_id=run.id, name="test.side_effect", arguments={"value": 7})

    first = asyncio.run(executor.execute(call, agent_id="main", services={}, internal=True))
    second = asyncio.run(executor.execute(call, agent_id="main", services={}, internal=True))

    assert first.status is ToolStatus.SUCCESS
    assert second == first
    assert side_effects == [{"value": 7}]
    persisted = store.get_tool_call(call.id)
    assert persisted is not None
    assert persisted[0] is ToolExecutionStatus.COMPLETED
    assert persisted[1] == first


def test_direct_executor_call_requires_explicit_trusted_internal_mode(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    effects = []
    registry = ToolRegistry()
    registry.register(ToolMetadata(name="test.direct", description="trusted call test"), lambda _args, _ctx: effects.append(True))
    executor = ToolExecutor(registry, store, TraceRecorder(store, EventBus()))
    call = ToolCall(id="untrusted_direct_call", name="test.direct")

    rejected = asyncio.run(executor.execute(call, agent_id="main", services={}))
    assert rejected.status is ToolStatus.BLOCKED
    assert rejected.error.code == "TRUSTED_EXECUTION_CONTEXT_REQUIRED"
    assert effects == []
    assert store.get_tool_call(call.id) is None

    accepted = asyncio.run(executor.execute(call, agent_id="main", services={}, internal=True))
    assert accepted.status is ToolStatus.SUCCESS
    assert effects == [True]


def test_approval_cannot_be_reused_with_changed_arguments(tmp_path):
    store = StateStore(tmp_path / "state.sqlite3")
    store.initialize()
    user_id = "approval-user"
    conversation = store.create_conversation("审批测试", user_id=user_id)
    run = Run(id="run_approval_exact", conversation_id=conversation.id, agent_id="main", status=RunStatus.RUNNING)
    store.save_run(run)
    effects = []
    registry = ToolRegistry()
    registry.register(
        ToolMetadata(
            name="test.destructive",
            description="精确参数审批测试",
            input_schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False},
            risk_level=RiskLevel.DESTRUCTIVE,
        ),
        lambda arguments, _context: effects.append(arguments) or {"output": "done"},
    )
    approvals = ApprovalService(store)
    executor = ToolExecutor(registry, store, TraceRecorder(store, EventBus()), approval_service=approvals)
    call = ToolCall(id="call_approval_exact", run_id=run.id, name="test.destructive", arguments={"value": 7})
    requested = asyncio.run(executor.execute(call, agent_id="main", services={}, internal=True))
    assert requested.error is not None
    approval_id = requested.error.details["approval_id"]
    approval = approvals.approve(approval_id, user_id=user_id)

    changed = call.model_copy(update={"arguments": {"value": 8}})
    rejected = asyncio.run(
        executor.execute(changed, agent_id="main", services={}, approval_id=approval.id, internal=True)
    )

    assert rejected.status is ToolStatus.BLOCKED
    assert rejected.error is not None and rejected.error.code == "APPROVAL_MISMATCH"
    assert effects == []
    assert store.get_tool_call(call.id) is None
