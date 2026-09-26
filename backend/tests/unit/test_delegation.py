from __future__ import annotations

import copy

import pytest

from app.core.models import AgentRequest, Dataset, DatasetKind, ToolCall


def valid_plan():
    return {
        "subtasks": [
            {
                "id": "a",
                "goal": "检查输入",
                "dataset_ids": ["ds_input"],
                "allowed_tools": ["dataset.inspect"],
            }
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "too_many",
        "duplicate",
        "unknown_dep",
        "self_dep",
        "repeated_dep",
        "cycle",
        "unknown_tool",
        "environment",
        "other_user",
        "missing_dataset",
        "binding_without_dep",
        "unknown_role",
        "empty_goal",
        "forged_state",
    ],
)
async def test_plan_is_rejected_before_task_or_child_side_effects(application, case):
    app = application
    conversation = app.store.create_conversation("校验", user_id="owner")
    app.store.save_dataset(
        Dataset(
            id="ds_input",
            name="输入",
            kind=DatasetKind.TABLE,
            path="input.csv",
            format="CSV",
            owner_user_id="owner",
        )
    )
    app.store.save_dataset(
        Dataset(
            id="ds_other",
            name="他人数据",
            kind=DatasetKind.TABLE,
            path="other.csv",
            format="CSV",
            owner_user_id="other",
        )
    )
    request = AgentRequest(conversation_id=conversation.id, user_id="owner", user_input="检查输入")
    parent = (await app.agent_loop.prepare_request(request)).run
    plan = valid_plan()
    subtask = plan["subtasks"][0]
    if case == "empty":
        plan["subtasks"] = []
    elif case == "too_many":
        plan["subtasks"] = [{**copy.deepcopy(subtask), "id": f"s{i}"} for i in range(6)]
    elif case == "duplicate":
        plan["subtasks"].append(copy.deepcopy(subtask))
    elif case == "unknown_dep":
        subtask["dependencies"] = ["absent"]
    elif case == "self_dep":
        subtask["dependencies"] = ["a"]
    elif case == "repeated_dep":
        subtask["dependencies"] = ["b", "b"]
    elif case == "cycle":
        subtask["dependencies"] = ["b"]
        plan["subtasks"].append({**copy.deepcopy(subtask), "id": "b", "dependencies": ["a"]})
    elif case == "unknown_tool":
        subtask["allowed_tools"] = ["unknown.tool"]
    elif case == "environment":
        subtask["allowed_tools"] = ["python.execute"]
    elif case == "other_user":
        subtask["dataset_ids"] = ["ds_other"]
    elif case == "missing_dataset":
        subtask["dataset_ids"] = ["made_up"]
    elif case == "binding_without_dep":
        subtask["upstream_dataset_bindings"] = [{"from_subtask": "b", "input_name": "buffer"}]
    elif case == "unknown_role":
        plan["subtasks"].append(
            {
                "id": "b",
                "goal": "下游",
                "allowed_tools": ["dataset.inspect"],
                "dependencies": ["a"],
                "upstream_dataset_bindings": [
                    {"from_subtask": "a", "output_role": "invented", "input_name": "buffer"}
                ],
            }
        )
    elif case == "empty_goal":
        subtask["goal"] = "  "
    else:
        subtask["assigned_agent_id"] = "forged"
    result = await app.delegation.execute(
        plan, request=request, parent=parent, call_id=f"{parent.id}:delegate"
    )
    assert result.error.code == "INVALID_DELEGATION_PLAN"
    assert app.store.get_run(parent.id).task_id is None
    assert app.store.list_child_runs(parent.id) == []
    assert app.store.list_tasks(conversation.id) == []
    assert app.store.get_delegation(f"{parent.id}:delegate") is None


@pytest.mark.asyncio
async def test_executor_rechecks_persisted_child_white_list_and_parent_scope(application):
    app = application
    conversation = app.store.create_conversation("隔离", user_id="owner")
    request = AgentRequest(conversation_id=conversation.id, user_id="owner", user_input="检查")
    parent = (await app.agent_loop.prepare_request(request)).run
    plan = {"subtasks": [{"id": "inspect", "goal": "检查", "allowed_tools": ["dataset.inspect"]}]}
    validated = app.delegation.validate(plan, request, parent)
    state = app.delegation._prepare(validated, request, parent, f"{parent.id}:delegate")
    child = app.store.get_run(state["run_ids"]["inspect"])
    services = app.agent_loop._execution_services(request, child)
    forged_context = app.agent_loop._discovery_context(request, services)
    denied = await app.tool_executor.execute(
        ToolCall(name="dataset.list", run_id=child.id),
        agent_id=child.agent_id,
        services=services,
        active_tool_names=frozenset(app.tool_registry.names()),
        discovery_context=forged_context,
    )
    assert denied.error.code == "SUBAGENT_TOOL_NOT_ALLOWED"
    parent = app.store.get_run(parent.id)
    app.store.save_run(
        parent.model_copy(update={"metadata": {**parent.metadata, "allowed_tool_names": []}})
    )
    tightened = app.agent_loop._discovery_context(request, services, child)
    assert app.agent_loop._available_tool_names(tightened, set()) == frozenset()
    assert app.agent_loop.catalog.search("buffer", tightened) == []


def test_child_workspace_rejects_shared_absolute_output(application):
    from app.execution.tools import ToolContext
    from app.tools.gis.common import output_path

    workspace = application.workspace.for_user("owner").for_run("child-a")
    context = ToolContext(run_id="child-a", agent_id="child-a", services={"workspace": workspace})
    with pytest.raises(PermissionError, match="独立目录"):
        output_path(context, str(workspace.root / "shared.gpkg"), "buffer")
    first = output_path(context, "same.gpkg", "buffer")
    other = application.workspace.for_user("owner").for_run("child-b")
    second = output_path(
        ToolContext(run_id="child-b", agent_id="child-b", services={"workspace": other}),
        "same.gpkg",
        "buffer",
    )
    assert first != second


@pytest.mark.asyncio
async def test_restored_delegation_observation_keeps_valid_json_and_critical_ids(application):
    import json

    app = application
    conversation = app.store.create_conversation("观察恢复", user_id="owner")
    request = AgentRequest(conversation_id=conversation.id, user_id="owner", user_input="恢复")
    parent = (await app.agent_loop.prepare_request(request)).run
    payload = {
        "output": {
            "delegation_id": "del_test",
            "subtasks": [
                {
                    "subtask_id": str(index),
                    "datasets": [{"dataset_id": f"ds_{index}_{output}"} for output in range(32)],
                    "status": "SUCCESS",
                }
                for index in range(20)
            ],
        }
    }
    protocol = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "delegate", "function": {"name": "agent.delegate", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "delegate", "content": json.dumps(payload)},
        {"role": "user", "content": "继续"},
    ]
    built = app.agent_loop.context.build(
        request, run=parent, protocol_messages=protocol, append_request=False
    )
    observed = next(item for item in built if item.get("role") == "tool")
    assert json.loads(observed["content"]) == payload
    assert any(item.get("tool_calls") for item in built)


@pytest.mark.asyncio
@pytest.mark.parametrize("settles", [False, True])
async def test_sync_timeout_preserves_late_evidence_or_uncertain_claim(application, settles):
    import asyncio
    from threading import Event

    from app.core.models import ToolExecutionStatus, ToolMetadata, ToolStatus

    app = application
    app.tool_executor.timeout_seconds = 0.05
    release, stopped = Event(), Event()
    executed = []

    def blocking(_arguments, context):
        executed.append(context.call_id)
        try:
            if settles:
                context.cancel_event.wait(1)
            else:
                release.wait(2)
            return {"output": {"count": 3}}
        finally:
            stopped.set()

    app.tool_registry.register(ToolMetadata(name="test.slow", description="超时边界"), blocking)
    tool_call = ToolCall(name="test.slow")
    try:
        result = await app.tool_executor.execute(
            tool_call, agent_id="main", services=app.tool_executor.services, internal=True
        )
        status, saved = app.store.get_tool_call(tool_call.id)
        if settles:
            assert result.status is ToolStatus.PARTIAL_SUCCESS
            assert saved.output == {"count": 3}
            assert result.error.code == "EXECUTION_TIMEOUT" and not result.error.retryable
        else:
            assert result.error.code == "SIDE_EFFECT_UNCERTAIN"
            assert status is ToolExecutionStatus.RUNNING and saved is None
        repeated = await app.tool_executor.execute(
            tool_call, agent_id="main", services=app.tool_executor.services, internal=True
        )
        assert repeated.call_id == tool_call.id
        assert len(executed) == 1
    finally:
        release.set()
        assert await asyncio.to_thread(stopped.wait, 2)
