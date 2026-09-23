import asyncio

from starlette.testclient import TestClient

from app.api import create_app
from app.core.models import AgentRequest, Run, RunStatus
from app.demo import seed_demo


def test_api_health_and_dataset_listing(application, authenticated_client):
    seed_demo(application)
    with authenticated_client as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["service"] == "geoagent"
        datasets = client.get("/api/v1/datasets")
        assert datasets.status_code == 200
        assert len(datasets.json()) >= 3


def test_api_dataset_registration_is_idempotent(application, authenticated_client):
    seed_demo(application)
    with authenticated_client as client:
        user_id = application.store.get_user_by_username("test-user").id
        source = application.workspace.input_dir / "roads.geojson"
        target = application.workspace.for_user(user_id).input_dir / "roads.geojson"
        target.write_bytes(source.read_bytes())
        response = client.post("/api/v1/datasets", json={"path": "input/roads.geojson", "name": "roads"})
        second = client.post("/api/v1/datasets", json={"path": "input/roads.geojson", "name": "roads"})

    assert response.status_code == 200
    assert second.status_code == 200
    assert response.json()["id"] == second.json()["id"]


def test_api_exposes_trace_checkpoint_and_artifact(application, authenticated_client):
    ids = seed_demo(application)
    with authenticated_client as client:
        response = client.post("/api/v1/messages", json={"message": "请执行 roads 检查并生成 500 米缓冲区", "dataset_ids": [ids["roads"]]})
        assert response.status_code == 200
        result = response.json()["result"]
        checkpoint = client.get(f"/api/v1/runs/{result['trace_id']}/checkpoint")
        assert checkpoint.status_code == 200
        artifacts = client.get("/api/v1/artifacts", params={"run_id": result["trace_id"]}).json()
        assert artifacts == []
        assert result["error"] == "MODEL_NOT_CONFIGURED"


def test_waiting_for_a_run_is_idempotent_for_assistant_messages(application):
    ids = seed_demo(application)

    async def run_case():
        request = AgentRequest(user_input="请执行 roads 检查", conversation_id="conversation-idempotent", dataset_ids=[ids["roads"]])
        run = await application.conversations.submit(request)
        first = await application.conversations.wait(run.id)
        second = await application.conversations.wait(run.id)
        return first, second

    first, second = asyncio.run(run_case())
    messages = application.store.list_messages("conversation-idempotent")

    assert first.trace_id == second.trace_id
    assert sum(message.role == "assistant" and message.run_id == first.trace_id for message in messages) == 1


def test_two_conversations_can_submit_and_wait_concurrently(application):
    async def run_cases():
        requests = [
            AgentRequest(user_input="请执行会话 A 检查", conversation_id="conversation-concurrent-a"),
            AgentRequest(user_input="请执行会话 B 检查", conversation_id="conversation-concurrent-b"),
        ]
        runs = await asyncio.gather(*(application.conversations.submit(request) for request in requests))
        results = await asyncio.gather(*(application.conversations.wait(run.id) for run in runs))
        return runs, results

    runs, results = asyncio.run(run_cases())

    assert len({run.id for run in runs}) == 2
    assert {run.conversation_id for run in runs} == {"conversation-concurrent-a", "conversation-concurrent-b"}
    assert {result.trace_id for result in results} == {run.id for run in runs}


def test_run_record_can_be_deleted(application, authenticated_client):
    with authenticated_client as client:
        user_id = application.store.get_user_by_username("test-user").id
        conversation = application.store.create_conversation(user_id=user_id)
        run = Run(task_id="task-delete", conversation_id=conversation.id, agent_id="main")
        application.store.save_run(run)
        response = client.delete(f"/api/v1/runs/{run.id}")
        assert response.status_code == 200
        assert response.json() == {"deleted": True}
        assert client.get(f"/api/v1/runs/{run.id}").status_code == 404


def test_run_records_can_be_deleted_in_bulk(application, authenticated_client):
    with authenticated_client as client:
        user_id = application.store.get_user_by_username("test-user").id
        conversation = application.store.create_conversation(user_id=user_id)
        runs = [Run(task_id=f"task-delete-{index}", conversation_id=conversation.id, agent_id="main") for index in range(2)]
        for run in runs:
            application.store.save_run(run)
        response = client.request("DELETE", "/api/v1/runs", json={"run_ids": [run.id for run in runs]})
        assert response.status_code == 200
        assert set(response.json()["deleted"]) == {run.id for run in runs}


def test_conversation_delete_cancels_inflight_run_and_returns_success(application, authenticated_client):
    with authenticated_client as client:
        user_id = application.store.get_user_by_username("test-user").id
        conversation = application.store.create_conversation("执行中的对话", user_id=user_id)
        run = Run(conversation_id=conversation.id, task_id="task-conversation-guard", agent_id="main", status=RunStatus.RUNNING)
        application.store.save_run(run)

        response = client.delete(f"/api/v1/conversations/{conversation.id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert application.store.get_conversation(conversation.id) is None
    assert application.store.get_run(run.id) is None


def test_conversation_delete_is_scoped_to_authenticated_user(application):
    with TestClient(create_app(application)) as owner_client, TestClient(create_app(application)) as other_client:
        owner_client.post("/api/v1/auth/register", json={"username": "owner-delete", "password": "password123", "display_name": "Owner"})
        other_client.post("/api/v1/auth/register", json={"username": "other-delete", "password": "password123", "display_name": "Other"})
        conversation = owner_client.post("/api/v1/conversations", json={"title": "仅所有者可见"}).json()

        response = other_client.delete(f"/api/v1/conversations/{conversation['id']}")

    assert response.status_code == 404
