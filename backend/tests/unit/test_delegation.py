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
        "cycle",
        "other_user",
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
    if case == "cycle":
        subtask["dependencies"] = ["b"]
        plan["subtasks"].append({**copy.deepcopy(subtask), "id": "b", "dependencies": ["a"]})
    else:
        subtask["dataset_ids"] = ["ds_other"]
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
    assert app.agent_loop.catalog.tool_search({"query": "buffer"}, tightened)["tools"] == []


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
