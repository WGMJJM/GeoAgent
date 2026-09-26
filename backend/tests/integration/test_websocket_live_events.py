import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.core.models import AgentResult, AgentResultStatus, Run, TokenUsage
from app.models import ModelAdapter, ModelStreamChunk
from app.observability.events import EventType


def _register(client: TestClient, username: str) -> None:
    response = client.post("/api/v1/auth/register", json={"username": username, "password": "password123", "display_name": username})
    assert response.status_code == 200


def test_websocket_sends_run_before_early_events_and_filters_other_runs(application, monkeypatch):
    async def finish_immediately(request, *, prepared, **_kwargs):
        run = prepared.run
        await application.trace.emit(run.id, EventType.DECISION_MADE, "收到请求", agent_id=run.agent_id)
        measured = application.store.add_run_token_usage(run.id, TokenUsage(model_calls=1, reported_calls=1,
                                                                           reported_input_tokens=100, reported_output_tokens=20))[0]
        await application.trace.emit(run.id, EventType.TOKEN_USAGE_UPDATED, payload={"token_usage": measured.token_usage.model_dump()})

        foreign = Run(id="run-other-conversation", conversation_id="conversation-other", agent_id="agent-loop")
        application.store.save_run(foreign)
        await application.trace.emit(foreign.id, EventType.RUN_CREATED, "不应进入当前连接", agent_id=foreign.agent_id)
        await application.trace.emit(foreign.id, EventType.TOKEN_USAGE_UPDATED, payload={"token_usage": {"model_calls": 999}})

        result = AgentResult(agent_id=run.agent_id, status=AgentResultStatus.SUCCESS, summary="处理完成", trace_id=run.id)
        return await application.agent_loop._finish(run, result, request=request)

    monkeypatch.setattr(application.agent_loop, "run", finish_immediately)

    with TestClient(create_app(application)) as client:
        _register(client, "socket-live-events")
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "ask", "message": "你好", "conversation_id": "conversation-live-events"})
            messages = []
            while True:
                payload = websocket.receive_json()
                messages.append(payload)
                if payload["type"] == "response":
                    break

    types = [item["type"] for item in messages]
    events = [item["data"] for item in messages if item["type"] == "event"]

    assert types.index("run") < types.index("event")
    assert {item["event_type"] for item in events} >= {EventType.RUN_CREATED, EventType.DECISION_MADE, EventType.RUN_COMPLETED}
    assert all(item["run_id"] != "run-other-conversation" for item in events)
    usage_events = [item for item in events if item["event_type"] == EventType.TOKEN_USAGE_UPDATED]
    assert len(usage_events) == 1
    assert usage_events[0]["payload"]["token_usage"]["reported_input_tokens"] == 100
    assert types.index("event") < types.index("response")
    assert messages[-1]["data"]["run"]["token_usage"] == usage_events[0]["payload"]["token_usage"]


@pytest.mark.parametrize("cancel", [False, True])
def test_websocket_delivers_text_before_model_finishes_and_closes_on_cancel(application, monkeypatch, cancel):
    class StreamingModel(ModelAdapter):
        def __init__(self):
            self.release = asyncio.Event()
            self.closed = False
            self.finished = False

        async def complete(self, request):
            raise AssertionError("只允许原来的流式请求")

        async def stream(self, request):
            try:
                yield ModelStreamChunk(content="第一段")
                await asyncio.wait_for(self.release.wait(), timeout=3)
                yield ModelStreamChunk(content="第二段")
                self.finished = True
                yield ModelStreamChunk(done=True, finish_reason="stop", input_tokens=100, output_tokens=20)
            finally:
                self.closed = True

    model = StreamingModel()
    monkeypatch.setattr(application.agent_loop, "model_provider", lambda _profile: model)
    conversation_id = "conversation-stream-cancel" if cancel else "conversation-stream-text"
    with TestClient(create_app(application)) as client:
        _register(client, "stream-cancel" if cancel else "stream-text")
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "ask", "message": "解释栅格", "conversation_id": conversation_id})
            packets = []
            while True:
                packet = websocket.receive_json()
                packets.append(packet)
                assert packet["type"] != "response", "不应等完整回答才显示首段"
                if packet["type"] == "delta":
                    break
            run_id = next(item["data"]["id"] for item in packets if item["type"] == "run")
            assert packet["content"] == "第一段"
            assert not model.finished
            run = application.store.get_run(run_id)
            assert run.status.value == "RUNNING" and run.token_usage is None
            assert any(item["type"] == "event" and item["data"]["event_type"] == "ModelResponseStarted" for item in packets)
            if cancel:
                assert client.post(f"/api/v1/runs/{run_id}/cancel").status_code == 200
            else:
                client.portal.call(model.release.set)
            while packets[-1]["type"] != "response":
                packets.append(websocket.receive_json())

    assert model.closed
    result = packets[-1]["data"]
    texts = [item["content"] for item in packets if item["type"] == "delta"]
    saved = [item.content for item in application.store.list_messages(conversation_id) if item.role == "assistant"]
    if cancel:
        assert texts == ["第一段"] and not model.finished
        assert result["run"]["status"] == "CANCELLED"
        assert "第一段" not in "".join(saved)
    else:
        assert texts == ["第一段", "第二段"]
        assert result["result"]["summary"] == "第一段第二段"
        assert result["run"]["token_usage"]["reported_output_tokens"] == 20
        assert saved == ["第一段第二段"]
