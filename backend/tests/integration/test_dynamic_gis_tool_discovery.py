from __future__ import annotations

import asyncio
import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from app.agent.loop import TOOL_VISIBILITY_PREFIX
from app.core.models import AgentRequest, AgentResultStatus, Message
from app.core.tokens import estimate_tokens
from app.models import ModelAdapter, ModelRequest, ModelResponse


class BufferAdapter(ModelAdapter):
    supports_tools = True

    def __init__(self, dataset_id: str) -> None:
        self.responses = [
            ModelResponse(
                tool_calls=[
                    {"id": "find_buffer", "function": {"name": "tool.search", "arguments": '{"query":"buffer"}'}}
                ]
            ),
            ModelResponse(
                tool_calls=[
                    {
                        "id": "create_buffer",
                        "function": {
                            "name": "vector.buffer",
                            "arguments": '{"dataset_id":"' + dataset_id + '","distance":100}',
                        },
                    }
                ]
            ),
            ModelResponse(content="缓冲区已生成。"),
        ]
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        return response


def test_search_activates_and_runs_real_vector_buffer(application):
    user_id = "gis-buffer-user"
    workspace = application.workspace.for_user(user_id)
    input_path = workspace.resolve("input/roads.gpkg")
    gpd.GeoDataFrame(
        {"road_id": [1]},
        geometry=[Point(1000, 1000)],
        crs="EPSG:3857",
    ).to_file(input_path, driver="GPKG")
    dataset = application.execution_services(user_id)["registry"].register_path(input_path, name="roads")

    adapter = BufferAdapter(dataset.id)
    application.agent_loop.model_provider = lambda _profile: adapter
    conversation = application.store.create_conversation("动态工具集成", user_id=user_id)
    request = AgentRequest(conversation_id=conversation.id, user_id=user_id, user_input="生成道路缓冲区")
    application.store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))

    async def execute():
        prepared = await application.agent_loop.prepare_request(request)
        return await application.agent_loop.run(request, prepared=prepared)

    result = asyncio.run(execute())

    first_names = {item["function"]["name"] for item in adapter.requests[0].tools}
    second_names = {item["function"]["name"] for item in adapter.requests[1].tools}
    assert "vector.buffer" not in first_names
    assert "vector.buffer" in second_names
    assert result.summary == "缓冲区已生成。"
    persisted = application.store.get_tool_call(f"{result.trace_id}:create_buffer")
    assert persisted is not None and persisted[1] is not None
    assert persisted[1].status.value == "SUCCESS"
    assert len(persisted[1].datasets) == 1
    output = application.store.get_dataset_for_user(persisted[1].datasets[0], user_id)
    assert output is not None
    assert output.source_dataset_ids == [dataset.id]
    assert workspace.resolve(output.path, allow_missing=False).exists()


@pytest.mark.parametrize("needs_statistics", [False, True])
def test_inspection_can_answer_directly_or_discover_only_missing_statistics(application, needs_statistics):
    """用可预测模型验证两条执行路径与提示契约，不声称验证真实 LLM 的选择。"""
    user_id = "gis-inspection-user"
    path = application.workspace.for_user(user_id).resolve("input/dem.tif")
    values = np.arange(64, dtype="float32").reshape(8, 8)
    values[0, 0] = -9999
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(114, 23, 0.002, 0.002),
                       nodata=-9999) as raster:
        raster.write(values, 1)
    dataset = application.execution_services(user_id)["registry"].register_path(path, name="dem")

    def call(name, arguments, call_id):
        return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}

    responses = [ModelResponse(tool_calls=[call("dataset.inspect", {"dataset_id": dataset.id}, "inspect")])]
    if needs_statistics:
        responses.extend([
            ModelResponse(content="用户要求平均高程，当前检查结果不足以给出像元统计，需要发现栅格统计能力。", tool_calls=[
                call("tool.search", {"query": "栅格统计 最小值 最大值", "english_query": "raster statistics min max"}, "bilingual"),
            ]),
            ModelResponse(tool_calls=[call("raster.inspect", {"dataset_id": dataset.id}, "statistics")]),
        ])
    responses.append(ModelResponse(content="已取得所需统计摘要。" if needs_statistics else "这是一份 GeoTIFF 栅格数据。"))

    class InspectionAdapter(ModelAdapter):
        supports_tools = True

        def __init__(self):
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            return responses.pop(0)

    adapter = InspectionAdapter()
    application.agent_loop.model_provider = lambda _profile: adapter
    conversation = application.store.create_conversation("按目标检查数据", user_id=user_id)
    request = AgentRequest(conversation_id=conversation.id, user_id=user_id, dataset_ids=[dataset.id],
                           user_input="查看这个 DEM 的平均高程" if needs_statistics else "查看这个是什么数据")
    application.store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))

    async def execute():
        prepared = await application.agent_loop.prepare_request(request)
        return await application.agent_loop.run(request, prepared=prepared)

    result = asyncio.run(execute())
    assert result.status is AgentResultStatus.SUCCESS
    assert application.store.get_run(result.trace_id).tool_call_count == (3 if needs_statistics else 1)
    calls = application.store.list_tool_calls(result.trace_id)
    assert [item[0].name for item in calls] == (["dataset.inspect", "raster.inspect"] if needs_statistics else ["dataset.inspect"])
    assert all(item[2].status.value == "SUCCESS" for item in calls)
    prompt = adapter.requests[0].messages[0]["content"]
    assert "当前工具能够满足目标且参数齐全时直接调用" in prompt
    assert "不要在尚未看到检查结果时" in prompt
    assert "缺少必须由用户提供的参数时使用 agent.ask_user" in prompt
    assert "证据足够时立即回答" in prompt
    assert "权限不足或临时执行失败不等于缺少能力" in prompt
    assert "未检索到或未开放某项能力不代表项目中不存在" in prompt
    assert "callable 中的工具已经提供完整 Schema" in prompt
    assert "历史搜索只表示曾经发现" in prompt
    assert "未调用候选降为卡片" in prompt
    assert "不需要额外发起一次选择加载" in prompt
    search_description = next(item["function"]["description"] for item in adapter.requests[0].tools if item["function"]["name"] == "tool.search")
    assert "Use callable tools directly" in search_description
    if needs_statistics:
        tool_definitions = adapter.requests[2].tools
        assert sum(item["function"]["name"] == "raster.inspect" for item in tool_definitions) == 1
        assert "采样统计" in next(item["function"]["description"] for item in tool_definitions if item["function"]["name"] == "raster.inspect")
        statistics = calls[-1][2].output["statistics"]
        assert statistics["valid_cell_count"] == 63
        assert statistics["min"] == 1
        assert statistics["max"] == 63
        assert statistics["mean"] == 32
    assert len(adapter.requests) == (4 if needs_statistics else 2)
    for model_request in adapter.requests:
        status = next(item for item in model_request.messages if item["role"] == "system" and item["content"].startswith(TOOL_VISIBILITY_PREFIX))
        visibility = json.loads(status["content"].removeprefix(TOOL_VISIBILITY_PREFIX))
        assert visibility["callable"] == [item["function"]["name"] for item in model_request.tools]
        assert application.agent_loop._tool_context_tokens(model_request.tools, visibility["cached"]) <= application.settings.tool_context_tokens


@pytest.mark.parametrize("attempt_slope_first", [False, True])
def test_reprojection_reuses_called_slope_or_restores_unused_card(application, attempt_slope_first):
    user_id = "gis-slope-cache-user"
    path = application.workspace.for_user(user_id).resolve("input/dem.tif")
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(114, 23, 0.002, 0.002), nodata=-9999) as raster:
        raster.write(np.arange(64, dtype="float32").reshape(8, 8), 1)
    source = application.execution_services(user_id)["registry"].register_path(path, name="dem")

    def call(name, arguments, call_id):
        return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}

    class SlopeAdapter(BufferAdapter):
        async def complete(self, request):
            self.requests.append(request)
            turn = len(self.requests)
            if turn == 1:
                return ModelResponse(tool_calls=[call("tool.search", {"query": "坡度", "english_query": "slope"}, "find_slope")])
            if turn == 2:
                if attempt_slope_first:
                    return ModelResponse(tool_calls=[call("raster.slope", {"dataset_id": source.id}, "initial_slope")])
                return ModelResponse(tool_calls=[call("tool.search", {"query": "栅格重投影", "english_query": "raster reproject"}, "find_projection")])
            if turn == 3:
                if attempt_slope_first:
                    observation = json.loads(next(item["content"] for item in request.messages if item.get("tool_call_id") == "initial_slope"))
                    assert observation["error"]["code"] == "CRS_UNIT_MISMATCH"
                    return ModelResponse(tool_calls=[call("tool.search", {"query": "栅格重投影", "english_query": "raster reproject"}, "find_projection")])
                return ModelResponse(tool_calls=[call("raster.reproject", {"dataset_id": source.id, "target_crs": "EPSG:32650"}, "project")])
            if turn == 4:
                status = next(item for item in request.messages if item["role"] == "system" and item["content"].startswith(TOOL_VISIBILITY_PREFIX))
                visibility = json.loads(status["content"].removeprefix(TOOL_VISIBILITY_PREFIX))
                if attempt_slope_first:
                    assert "raster.slope" in visibility["callable"]
                    return ModelResponse(tool_calls=[call("raster.reproject", {"dataset_id": source.id, "target_crs": "EPSG:32650"}, "project")])
                assert "raster.slope" not in visibility["callable"]
                assert "raster.slope" in {item["name"] for item in visibility["cached"]}
                return ModelResponse(tool_calls=[call("tool.search", {"query": "raster.slope"}, "restore_slope")])
            if turn == 5:
                status = next(item for item in request.messages if item["role"] == "system" and item["content"].startswith(TOOL_VISIBILITY_PREFIX))
                visibility = json.loads(status["content"].removeprefix(TOOL_VISIBILITY_PREFIX))
                assert "raster.slope" in visibility["callable"]
                assert "raster.slope" not in {item["name"] for item in visibility["cached"]}
                if not attempt_slope_first:
                    restored = json.loads(next(item["content"] for item in request.messages if item.get("tool_call_id") == "restore_slope"))
                    assert restored["output"]["source"] == "run_cache"
                observation = json.loads(next(item["content"] for item in request.messages if item.get("tool_call_id") == "project"))
                return ModelResponse(tool_calls=[call("raster.slope", {"dataset_id": observation["output"]["id"]}, "slope")])
            return ModelResponse(content="已复用或从卡片恢复坡度工具，完成真实坡度计算。")

    adapter = SlopeAdapter(source.id)
    application.agent_loop.model_provider = lambda _profile: adapter
    conversation = application.store.create_conversation("坡度连续工具", user_id=user_id)
    request = AgentRequest(conversation_id=conversation.id, user_id=user_id, dataset_ids=[source.id], user_input="计算坡度")
    application.store.save_message(Message(conversation_id=conversation.id, role="user", content=request.user_input))

    async def execute():
        prepared = await application.agent_loop.prepare_request(request)
        return await application.agent_loop.run(request, prepared=prepared)

    result = asyncio.run(execute())
    assert result.status is AgentResultStatus.SUCCESS
    assert application.store.get_run(result.trace_id).tool_call_count == 5
    assert len(adapter.requests) == 6
    calls = application.store.list_tool_calls(result.trace_id)
    assert [item[0].name for item in calls] == (["raster.slope"] if attempt_slope_first else []) + ["raster.reproject", "raster.slope"]
    assert all(item[2].status.value == "SUCCESS" for item in calls[-2:])
    checkpoint = application.store.latest_checkpoint(result.trace_id)
    assert set(checkpoint.state["used_tool_names"]) == {"raster.reproject", "raster.slope"}
    assert set(checkpoint.state["activated_tool_names"]) == {"raster.reproject", "raster.slope"}
    output = application.store.get_dataset_for_user(calls[-1][2].datasets[0], user_id)
    assert output.crs.authority == "EPSG:32650"
    assert application.workspace.for_user(user_id).resolve(output.path, allow_missing=False).exists()
    for model_request in adapter.requests:
        statuses = [item for item in model_request.messages if item["role"] == "system" and item["content"].startswith(TOOL_VISIBILITY_PREFIX)]
        assert len(statuses) == 1
        visibility = json.loads(statuses[0]["content"].removeprefix(TOOL_VISIBILITY_PREFIX))
        assert visibility["callable"] == [item["function"]["name"] for item in model_request.tools]
        tokens = estimate_tokens(json.dumps(model_request.tools, ensure_ascii=False, separators=(",", ":"))) + estimate_tokens(statuses[0]["content"])
        assert tokens == application.agent_loop._tool_context_tokens(model_request.tools, visibility["cached"])
        assert tokens <= application.settings.tool_context_tokens
