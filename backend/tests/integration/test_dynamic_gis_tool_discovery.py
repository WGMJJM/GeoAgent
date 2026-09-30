from __future__ import annotations

import asyncio
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.agent.context import TOOL_VISIBILITY_PREFIX
from app.core.models import AgentRequest, AgentResultStatus, Message
from app.core.tokens import estimate_tokens
from app.models import ModelAdapter, ModelRequest, ModelResponse


def test_reprojection_reuses_called_slope_after_crs_error(application):
    user_id = "gis-slope-cache-user"
    path = application.workspace.for_user(user_id).resolve("input/dem.tif")
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(114, 23, 0.002, 0.002), nodata=-9999) as raster:
        raster.write(np.arange(64, dtype="float32").reshape(8, 8), 1)
    source = application.execution_services(user_id)["registry"].register_path(path, name="dem")

    def call(name, arguments, call_id):
        return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}

    class SlopeAdapter(ModelAdapter):
        supports_tools = True

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def complete(self, request):
            self.requests.append(request)
            turn = len(self.requests)
            if turn == 1:
                return ModelResponse(tool_calls=[call("tool.search", {"query": "坡度", "english_query": "slope"}, "find_slope")])
            if turn == 2:
                return ModelResponse(tool_calls=[call("raster.slope", {"dataset_id": source.id}, "initial_slope")])
            if turn == 3:
                observation = json.loads(next(item["content"] for item in request.messages if item.get("tool_call_id") == "initial_slope"))
                assert observation["error"]["code"] == "CRS_UNIT_MISMATCH"
                return ModelResponse(tool_calls=[call("tool.search", {"query": "栅格重投影", "english_query": "raster reproject"}, "find_projection")])
            if turn == 4:
                status = next(item for item in request.messages if item["role"] == "system" and item["content"].startswith(TOOL_VISIBILITY_PREFIX))
                visibility = json.loads(status["content"].removeprefix(TOOL_VISIBILITY_PREFIX))
                assert "raster.slope" in visibility["callable"]
                return ModelResponse(tool_calls=[call("raster.reproject", {"dataset_id": source.id, "target_crs": "EPSG:32650"}, "project")])
            if turn == 5:
                status = next(item for item in request.messages if item["role"] == "system" and item["content"].startswith(TOOL_VISIBILITY_PREFIX))
                visibility = json.loads(status["content"].removeprefix(TOOL_VISIBILITY_PREFIX))
                assert "raster.slope" in visibility["callable"]
                assert "raster.slope" not in {item["name"] for item in visibility["cached"]}
                observation = json.loads(next(item["content"] for item in request.messages if item.get("tool_call_id") == "project"))
                return ModelResponse(tool_calls=[call("raster.slope", {"dataset_id": observation["output"]["id"]}, "slope")])
            return ModelResponse(content="已复用坡度工具，完成真实坡度计算。")

    adapter = SlopeAdapter()
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
    assert [item[0].name for item in calls] == ["raster.slope", "raster.reproject", "raster.slope"]
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
