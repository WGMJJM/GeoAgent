from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import geopandas as gpd
import pytest
from shapely.geometry import Point

from app.application import Application
from app.core.models import (
    AgentRequest,
    AgentResultStatus,
    DatasetOutputPolicy,
    Message,
    RiskLevel,
    RunStatus,
    TaskStatus,
    ToolCall,
    ToolError,
    ToolMetadata,
    ToolResult,
    ToolStatus,
)
from app.models import ModelAdapter, ModelRequest, ModelResponse


def call(name, arguments, identifier="call"):
    return ModelResponse(
        tool_calls=[
            {"id": identifier, "function": {"name": name, "arguments": json.dumps(arguments)}}
        ]
    )


class DelegationModel(ModelAdapter):
    supports_tools = True

    def __init__(self, plan, scripts):
        self.plan, self.scripts = plan, {key: list(value) for key, value in scripts.items()}
        self.requests = defaultdict(list)
        self.contexts = {}
        self.observations = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        child = None
        for message in request.messages:
            if message["role"] == "system" and message["content"].startswith('{"subtask_id"'):
                child = json.loads(message["content"])
                break
        if child:
            name = child["subtask_id"]
            self.requests[name].append(request)
            self.contexts[name] = child
            response = self.scripts[name][0]
            if callable(response):
                response = response(child, request)
                if asyncio.iscoroutine(response):
                    response = await response
            self.scripts[name].pop(0)
            return response
        self.requests["main"].append(request)
        observations = [
            json.loads(message["content"])
            for message in request.messages
            if message["role"] == "tool"
        ]
        if not observations:
            return call("agent.delegate", self.plan, "delegate")
        self.observations = observations
        return ModelResponse(content="已读取结构化委派状态并汇总。")


def buffer_subtask(name, dataset_id, **extra):
    return {
        "id": name,
        "goal": f"生成 {name} 缓冲区",
        "dataset_ids": [dataset_id],
        "allowed_tools": ["dataset.inspect", "vector.buffer"],
        "output_roles": {"buffer": "vector.buffer"},
        **extra,
    }


def buffer_script(dataset_id):
    return [
        call("tool.search", {"query": "buffer"}, "search"),
        call(
            "vector.buffer",
            {"dataset_id": dataset_id, "distance": 100, "output_path": "buffer.gpkg"},
            "buffer",
        ),
        ModelResponse(content="完成，文字中不提供任何 Dataset ID。"),
    ]


def dataset(app, user_id="owner"):
    workspace = app.workspace.for_user(user_id)
    path = workspace.resolve("input/roads.gpkg")
    gpd.GeoDataFrame({"road_id": [1]}, geometry=[Point(1000, 1000)], crs="EPSG:3857").to_file(
        path, driver="GPKG"
    )
    return app.execution_services(user_id)["registry"].register_path(path, name="roads")


async def submit(app, model, user_id="owner", **extra):
    app.agent_loop.model_provider = lambda _profile: model
    conversation = app.store.create_conversation("委派测试", user_id=user_id)
    app.store.save_message(
        Message(
            conversation_id=conversation.id, role="user", content="父会话秘密，不应出现在子上下文。"
        )
    )
    request = AgentRequest(
        conversation_id=conversation.id, user_id=user_id, user_input="执行子任务", **extra
    )
    parent = await app.run_manager.submit(request)
    return request, parent, await app.run_manager.wait(parent.id)


@pytest.mark.asyncio
async def test_parallel_real_gis_and_dependency_uses_verified_dataset_id(application):
    app = application
    source = dataset(app)
    plan = {
        "subtasks": [
            buffer_subtask("a", source.id),
            buffer_subtask("b", source.id),
            {
                "id": "c",
                "goal": "检查上游缓冲",
                "allowed_tools": ["dataset.inspect"],
                "dependencies": ["a"],
                "upstream_dataset_bindings": [
                    {"from_subtask": "a", "output_role": "buffer", "input_name": "buffer_dataset"}
                ],
            },
        ]
    }
    barrier = asyncio.Event()
    entered = []
    registered = app.tool_registry.get("vector.buffer")

    async def parallel_buffer(arguments, context):
        entered.append(context.run_id)
        if len(entered) == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), 5)
        return await asyncio.to_thread(registered.handler, arguments, context)

    app.tool_registry.unregister("vector.buffer")
    app.tool_registry.register(registered.metadata, parallel_buffer, deferred=True)

    def inspect_upstream(context, _request):
        upstream = context["upstream_inputs"]["buffer_dataset"]["dataset_id"]
        assert context["selected_datasets"][0]["id"] == upstream
        return call("dataset.inspect", {"dataset_id": upstream}, "inspect")

    model = DelegationModel(
        plan,
        {
            "a": buffer_script(source.id),
            "b": buffer_script(source.id),
            "c": [inspect_upstream, ModelResponse(content="上游已验证。")],
        },
    )
    _, parent, result = await submit(app, model)
    assert result.status is AgentResultStatus.SUCCESS
    assert len(set(entered)) == 2
    observation = model.observations[-1]["output"]
    upstream_id = observation["subtasks"][0]["datasets"][0]["dataset_id"]
    assert model.contexts["c"]["upstream_inputs"]["buffer_dataset"]["dataset_id"] == upstream_id
    children = app.store.list_child_runs(parent.id)
    assert len({child.agent_id for child in children}) == 3
    paths = [
        app.store.get_dataset_for_user(identifier, "owner").path for identifier in result.datasets
    ]
    assert len(set(paths)) == 2
    assert all("runs" in path for path in paths)
    for name in ("a", "b", "c"):
        names = {tool["function"]["name"] for tool in model.requests[name][0].tools}
        assert "agent.delegate" not in names and "conversation.search_history" not in names
        assert "dataset.list" not in names
    memory = app.store.get_working_memory(app.store.get_run(parent.id).task_id)
    assert set(memory.active_dataset_ids) == set(result.datasets)
    assert sorted(item.source_run_id for item in memory.intermediate_results) == [
        item.source_run_id for item in memory.intermediate_results
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("optional", [False, True])
async def test_failure_propagates_and_optional_failure_keeps_outputs(application, optional):
    app = application
    source = dataset(app)
    plan = {
        "subtasks": [
            buffer_subtask("good", source.id),
            {
                "id": "bad",
                "goal": "失败任务",
                "allowed_tools": ["test.fail"],
                "required": not optional,
            },
            {
                "id": "downstream",
                "goal": "不能启动",
                "allowed_tools": ["dataset.inspect"],
                "dependencies": ["bad"],
                "required": not optional,
            },
        ]
    }
    app.tool_registry.register(
        ToolMetadata(name="test.fail", description="确定失败", input_schema={"type": "object"}),
        lambda _arguments, _context: ToolResult(
            call_id="placeholder",
            status=ToolStatus.FAILED,
            error=ToolError(code="MISSING_FIELD", message="必要字段缺失", retryable=False),
        ),
    )
    model = DelegationModel(
        plan,
        {
            "good": buffer_script(source.id),
            "bad": [call("test.fail", {}), ModelResponse(content="我自称成功")],
            "downstream": [],
        },
    )
    _, parent, result = await submit(app, model)
    assert result.status is (AgentResultStatus.PARTIAL if optional else AgentResultStatus.FAILED)
    observation = model.observations[-1]["output"]
    assert observation["failed_subtask_ids"] == ["bad"]
    assert observation["blocked_subtask_ids"] == ["downstream"]
    assert observation["subtasks"][1]["error"]["code"] == "MISSING_FIELD"
    assert not model.requests["downstream"]
    assert observation["added_dataset_ids"]
    assert all(
        child.status not in {RunStatus.CREATED, RunStatus.RUNNING}
        for child in app.store.list_child_runs(parent.id)
    )


@pytest.mark.asyncio
async def test_fake_outputs_and_recursive_delegation_cannot_become_success(application):
    app = application
    app.tool_registry.register(
        ToolMetadata(
            name="test.fabricate",
            description="假产出",
            input_schema={"type": "object"},
            dataset_output_policy=DatasetOutputPolicy.REQUIRED,
        ),
        lambda _arguments, _context: {"datasets": ["invented_dataset"], "output": {"count": 9}},
    )
    plan = {"subtasks": [{"id": "a", "goal": "生成数据", "allowed_tools": ["test.fabricate"]}]}
    model = DelegationModel(
        plan,
        {
            "a": [
                call("agent.delegate", plan, "recursive"),
                call("test.fabricate", {}, "fabricate"),
                ModelResponse(content="完成"),
            ]
        },
    )
    _, _, result = await submit(app, model)
    assert result.status is AgentResultStatus.FAILED
    assert "SUBAGENT_DELEGATION_FORBIDDEN" in str(model.requests["a"][1].messages)
    child = model.observations[-1]["output"]["subtasks"][0]
    assert child["datasets"] == []
    assert child["error"]["code"] in {
        "UNVERIFIED_DATASET_OUTPUT",
        "REQUIRED_DATASET_OUTPUT_MISSING",
    }


@pytest.mark.asyncio
async def test_parent_cancel_cascades_to_all_inflight_children(application):
    app = application
    entered = asyncio.Event()
    started = []
    cancelled = []

    async def hold(_args, context):
        started.append(context.run_id)
        if len(started) == 2:
            entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(context.run_id)

    app.tool_registry.register(
        ToolMetadata(name="test.hold", description="等待", input_schema={"type": "object"}), hold
    )
    plan = {
        "subtasks": [
            {"id": name, "goal": "等待", "allowed_tools": ["test.hold"]} for name in ("a", "b")
        ]
    }
    model = DelegationModel(plan, {name: [call("test.hold", {})] for name in ("a", "b")})
    app.agent_loop.model_provider = lambda _profile: model
    conversation = app.store.create_conversation("取消", user_id="owner")
    request = AgentRequest(conversation_id=conversation.id, user_id="owner", user_input="执行")
    parent = await app.run_manager.submit(request)
    await asyncio.wait_for(entered.wait(), 5)
    assert await app.run_manager.cancel(parent.id)
    assert set(cancelled) == set(started)
    assert app.store.get_run(parent.id).status is RunStatus.CANCELLED
    assert all(
        child.status is RunStatus.CANCELLED for child in app.store.list_child_runs(parent.id)
    )
    assert not app.run_manager._active
    state = app.store.get_delegation(f"{parent.id}:delegate")
    assert all(item["status"] == "CANCELLED" for item in state["result"]["subtasks"])
    assert app.store.get_task(app.store.get_run(parent.id).task_id).status is TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_restart_reuses_finished_child_and_stable_ids(application):
    app = application
    source = dataset(app)
    plan = {"subtasks": [buffer_subtask("a", source.id), buffer_subtask("b", source.id)]}
    a_persisted, b_entered = asyncio.Event(), asyncio.Event()

    async def hold_b(_context, _request):
        b_entered.set()
        await asyncio.Event().wait()

    async def observe(event):
        if event.event_type == "SubAgentCompleted" and event.payload.get("subtask_id") == "a":
            a_persisted.set()

    app.bus.subscribe(observe)
    model = DelegationModel(plan, {"a": buffer_script(source.id), "b": [hold_b]})
    app.agent_loop.model_provider = lambda _profile: model
    conversation = app.store.create_conversation("中断", user_id="owner")
    request = AgentRequest(conversation_id=conversation.id, user_id="owner", user_input="执行")
    parent = await app.run_manager.submit(request)
    await asyncio.wait_for(asyncio.gather(a_persisted.wait(), b_entered.wait()), 8)
    children_before = {
        child.metadata["subtask_id"]: child for child in app.store.list_child_runs(parent.id)
    }
    output_a = app.store.get_run(children_before["a"].id).metadata["subagent_result"]["datasets"][
        0
    ]["dataset_id"]
    state_before_shutdown = app.store.get_delegation(f"{parent.id}:delegate")
    # 模拟进程突然消失后磁盘上遗留的 in-flight 状态；新 Application 正常对账。
    await app.run_manager.cancel(parent.id)
    app.store.save_delegation(state_before_shutdown)
    for run_id in (parent.id, children_before["b"].id):
        saved = app.store.get_run(run_id)
        app.store.save_run(
            saved.model_copy(
                update={
                    "status": RunStatus.WAITING_SUBAGENT
                    if run_id == parent.id
                    else RunStatus.RUNNING
                }
            )
        )
    restarted = Application(app.settings)
    restarted.start()
    resumed_model = DelegationModel(plan, {"a": [], "b": buffer_script(source.id)})
    restarted.agent_loop.model_provider = lambda _profile: resumed_model
    try:
        assert restarted.store.get_run(parent.id).status is RunStatus.INTERRUPTED
        await restarted.run_manager.continue_run(parent.id, user_id="owner", technical=True)
        result = await restarted.run_manager.wait(parent.id)
        assert result.status is AgentResultStatus.SUCCESS
        after = {
            child.metadata["subtask_id"]: child.id
            for child in restarted.store.list_child_runs(parent.id)
        }
        assert after == {name: child.id for name, child in children_before.items()}
        assert not resumed_model.requests["a"]
        assert output_a in result.datasets
        assert len(restarted.store.list_tool_calls(children_before["a"].id)) == 1
        assert (
            len(
                [
                    item
                    for item in resumed_model.requests["main"][-1].messages
                    if item.get("role") == "tool" and item.get("tool_call_id") == "delegate"
                ]
            )
            == 1
        )
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_uncertain_inflight_tool_is_not_replayed(application):
    app = application
    source = dataset(app)
    plan = {"subtasks": [buffer_subtask("a", source.id)]}
    conversation = app.store.create_conversation("未知副作用", user_id="owner")
    request = AgentRequest(conversation_id=conversation.id, user_id="owner", user_input="执行")
    parent = (await app.agent_loop.prepare_request(request)).run
    state = app.delegation._prepare(
        app.delegation.validate(plan, request, parent), request, parent, f"{parent.id}:delegate"
    )
    child_id = state["run_ids"]["a"]
    app.store.mark_tool_call_running(
        ToolCall(
            id=f"{child_id}:uncertain",
            run_id=child_id,
            name="vector.buffer",
            arguments={"dataset_id": source.id, "distance": 100},
        )
    )
    result = await app.delegation.execute(
        plan, request=request, parent=parent, call_id=f"{parent.id}:delegate"
    )
    assert result.output["subtasks"][0]["error"]["code"] == "SIDE_EFFECT_UNCERTAIN"
    assert app.store.list_datasets_for_user("owner") == [source]


def test_user_reply_resumes_original_child_through_message_api(application, authenticated_client):
    app = application
    plan = {"subtasks": [{"id": "a", "goal": "确认再查询", "allowed_tools": ["dataset.list"]}]}
    model = DelegationModel(
        plan,
        {
            "a": [
                call("agent.ask_user", {"question": "是否列出当前数据？"}, "question"),
                call("dataset.list", {}, "list"),
                ModelResponse(content="查询已完成。"),
            ]
        },
    )
    app.agent_loop.model_provider = lambda _profile: model
    with authenticated_client as client:
        response = client.post("/api/v1/messages", json={"message": "确认并查询"})
        assert response.status_code == 200, response.text
        waiting = response.json()
        parent_id = waiting["run"]["id"]
        child = app.store.list_child_runs(parent_id)[0]
        assert app.store.get_run(parent_id).status is RunStatus.WAITING_USER
        assert child.status is RunStatus.WAITING_USER
        resumed = client.post(
            "/api/v1/messages",
            json={
                "message": "是",
                "conversation_id": waiting["run"]["conversation_id"],
                "reply_to_run_id": parent_id,
            },
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["result"]["status"] == "SUCCESS"
        assert app.store.list_child_runs(parent_id)[0].id == child.id
        assert "是" in str(model.requests["a"][1].messages)
        assistants = [
            message
            for message in app.store.list_messages(child.conversation_id)
            if message.role == "assistant"
        ]
        assert all(message.run_id == parent_id for message in assistants)


@pytest.mark.parametrize("approve", [True, False])
def test_one_shot_child_approval_resumes_parent_through_existing_api(
    application, authenticated_client, approve
):
    app = application
    writes = []
    app.tool_registry.register(
        ToolMetadata(
            name="test.sensitive",
            description="审批写入",
            risk_level=RiskLevel.DESTRUCTIVE,
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        lambda arguments, _context: writes.append(arguments) or {"output": {"written": True}},
    )
    plan = {
        "subtasks": [{"id": "a", "goal": "执行受审批写入", "allowed_tools": ["test.sensitive"]}]
    }
    model = DelegationModel(
        plan,
        {"a": [call("test.sensitive", {"value": 7}, "write"), ModelResponse(content="写入完成。")]},
    )
    app.agent_loop.model_provider = lambda _profile: model
    with authenticated_client as client:
        waiting = client.post("/api/v1/messages", json={"message": "执行受审批写入"})
        assert waiting.status_code == 200, waiting.text
        parent_id = waiting.json()["run"]["id"]
        approval_id = waiting.json()["result"]["needs_input"]["approval_id"]
        child = app.store.list_child_runs(parent_id)[0]
        assert app.store.get_run(parent_id).status is RunStatus.WAITING_APPROVAL
        assert writes == []
        action = "approve" if approve else "deny"
        approved = client.post(f"/api/v1/approvals/{approval_id}/{action}")
        assert approved.status_code == 200, approved.text
        assert approved.json()["run"]["id"] == parent_id
        assert approved.json()["result"]["status"] == ("SUCCESS" if approve else "FAILED"), {
            "child_error": app.store.get_run(child.id).error,
            "observed_error": model.observations[-1]["output"]["subtasks"][0]["error"],
            "writes": writes,
        }
        assert writes == ([{"value": 7}] if approve else [])
        assert app.store.get_approval(approval_id).source_run_id == child.id
        assert app.store.get_approval(approval_id).status.value == (
            "CONSUMED" if approve else "DENIED"
        )
        assert client.post(f"/api/v1/approvals/{approval_id}/{action}").status_code == 409
        assert writes == ([{"value": 7}] if approve else [])
