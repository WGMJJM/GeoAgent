from __future__ import annotations

import asyncio

import geopandas as gpd
from shapely.geometry import Point

from app.core.models import AgentRequest, Message
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
