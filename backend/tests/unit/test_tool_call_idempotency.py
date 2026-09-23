import asyncio

from app.core.models import Run, RunStatus, ToolCall, ToolExecutionStatus, ToolMetadata, ToolStatus
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

    first = asyncio.run(executor.execute(call, agent_id="main", services={}))
    second = asyncio.run(executor.execute(call, agent_id="main", services={}))

    assert first.status is ToolStatus.SUCCESS
    assert second == first
    assert side_effects == [{"value": 7}]
    persisted = store.get_tool_call(call.id)
    assert persisted is not None
    assert persisted[0] is ToolExecutionStatus.COMPLETED
    assert persisted[1] == first
