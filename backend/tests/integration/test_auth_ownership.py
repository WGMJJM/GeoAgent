import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import create_app
from app.auth.password import hash_session_token
from app.core.models import Artifact, ArtifactKind, Run, RunStatus, Task


def _register(client: TestClient, username: str):
    response = client.post("/api/v1/auth/register", json={"username": username, "password": "password123", "display_name": username.title()})
    assert response.status_code == 200
    return response.json()


def test_auth_register_login_me_logout_and_password_is_hashed(application):
    with TestClient(create_app(application)) as client:
        assert client.get("/api/v1/users/me").status_code == 401
        created = _register(client, "alice")
        assert created["username"] == "alice"
        assert "password_hash" not in created
        stored = application.store.get_user(created["id"])
        assert stored is not None
        assert stored.password_hash != "password123"
        assert client.get("/api/v1/users/me").json()["id"] == created["id"]
        client.post("/api/v1/auth/logout")
        assert client.get("/api/v1/users/me").status_code == 401
        assert client.post("/api/v1/auth/login", json={"identifier": "alice", "password": "wrong-password"}).status_code == 401
        assert client.post("/api/v1/auth/login", json={"identifier": "alice", "password": "password123"}).status_code == 200


def test_conversation_dataset_memory_and_artifact_are_user_scoped(application):
    with TestClient(create_app(application)) as client_a, TestClient(create_app(application)) as client_b:
        user_a = _register(client_a, "alice")
        client_a.post("/api/v1/conversations", json={"title": "Alice 对话"})
        upload = client_a.post("/api/v1/attachments", files={"file": ("alice.csv", b"x,y\n1,2\n", "text/csv")})
        assert upload.status_code == 200
        dataset_id = upload.json()["dataset"]["id"]
        assert upload.json()["dataset"]["owner_user_id"] == user_a["id"]
        client_a.post("/api/v1/memories", json={"key": "默认 CRS", "value": "EPSG:3857"})
        run = Run(conversation_id=application.store.list_conversations(user_id=user_a["id"])[0].id, agent_id="main", status=RunStatus.COMPLETED)
        application.store.save_run(run)
        artifact = Artifact(name="alice.txt", kind=ArtifactKind.OTHER, path=None, run_id=run.id, owner_user_id=user_a["id"])
        application.store.save_artifact(artifact)

        _register(client_b, "bob")
        assert client_b.get("/api/v1/conversations").json() == []
        assert client_b.get("/api/v1/datasets").json() == []
        assert client_b.get("/api/v1/memories").json() == []
        assert client_b.get(f"/api/v1/runs/{run.id}").status_code == 404
        assert client_b.get(f"/api/v1/artifacts/{artifact.id}/content").status_code == 404
        assert client_b.post("/api/v1/messages", json={"message": "检查数据", "dataset_ids": [dataset_id]}).status_code == 403


def test_workspace_paths_are_separated(application):
    with TestClient(create_app(application)) as client_a, TestClient(create_app(application)) as client_b:
        _register(client_a, "alice")
        upload_a = client_a.post("/api/v1/attachments", files={"file": ("a.csv", b"x,y\n1,2\n", "text/csv")})
        _register(client_b, "bob")
        upload_b = client_b.post("/api/v1/attachments", files={"file": ("b.csv", b"x,y\n3,4\n", "text/csv")})
        path_a = upload_a.json()["dataset"]["path"]
        path_b = upload_b.json()["dataset"]["path"]
        assert path_a != path_b
        assert "/users/" in path_a.replace("\\", "/")
        assert "/users/" in path_b.replace("\\", "/")


def test_websocket_requires_session(application):
    with TestClient(create_app(application)) as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/ws"):
                pass
        assert error.value.code == 1008


def test_run_derived_dataset_and_artifact_keep_user_owner(application):
    with TestClient(create_app(application)) as client_a, TestClient(create_app(application)) as client_b:
        user_a = _register(client_a, "owner-a")
        user_b = _register(client_b, "owner-b")
        conversation = application.conversations.create("归属测试", user_id=user_a["id"])
        task = Task(goal="归属测试", conversation_id=conversation.id)
        application.store.save_task(task)
        main_run = Run(conversation_id=conversation.id, task_id=task.id, agent_id="main", status=RunStatus.RUNNING)
        application.store.save_run(main_run)
        services = application.execution_services(user_a["id"])
        dataset_path = services["workspace"].input_dir / "owned.csv"
        dataset_path.write_text("x,y\n1,2\n", encoding="utf-8")
        dataset = services["registry"].register_path(dataset_path, run_id=main_run.id)
        artifact_path = services["workspace"].output_dir / "owned.txt"
        artifact_path.write_text("owned", encoding="utf-8")
        artifact = Artifact(
            name=artifact_path.name,
            kind=ArtifactKind.OTHER,
            path=str(artifact_path),
            run_id=main_run.id,
            owner_user_id=user_a["id"],
        )
        application.store.save_artifact(artifact)
        sub_run = Run(conversation_id=conversation.id, task_id=task.id, parent_run_id=main_run.id, agent_id="sub", status=RunStatus.RUNNING)
        application.store.save_run(sub_run)
        sub_path = services["workspace"].intermediate_dir / "sub.csv"
        sub_path.write_text("x,y\n3,4\n", encoding="utf-8")
        sub_dataset = services["registry"].register_path(sub_path, run_id=sub_run.id)
        assert dataset.owner_user_id == user_a["id"]
        assert artifact.owner_user_id == user_a["id"]
        assert sub_dataset.owner_user_id == user_a["id"]
        assert application.store.get_dataset_for_user(dataset.id, user_a["id"]) is not None
        assert application.store.get_dataset_for_user(dataset.id, user_b["id"]) is None
        assert application.store.get_artifact_for_user(artifact.id, user_b["id"]) is None
        assert client_b.get("/api/v1/datasets").json() == []


def test_authenticated_websocket_resources_are_owned_by_user(application):
    with TestClient(create_app(application)) as client_a:
        user_a = _register(client_a, "socket-owner")
        run_id = None
        with client_a.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "ask", "message": "执行一个栅格分析任务", "conversation_id": "socket-conversation"})
            while True:
                payload = websocket.receive_json()
                if payload["type"] == "run":
                    run_id = payload["data"]["id"]
                if payload["type"] == "response":
                    break
        assert run_id is not None
        assert application.store.run_belongs_to_user(run_id, user_a["id"])
    with TestClient(create_app(application)) as client_b:
        _register(client_b, "socket-other")
        assert client_b.get(f"/api/v1/runs/{run_id}").status_code == 404
        assert client_b.get("/api/v1/conversations").json() == []


def test_expired_and_inactive_sessions_are_rejected(application):
    with TestClient(create_app(application)) as client:
        created = _register(client, "session-owner")
        token = client.cookies.get(application.settings.auth_cookie_name)
        assert token is not None
        session = application.store.get_session(hash_session_token(token))
        assert session is not None
        application.store.save_session(session.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(minutes=1)}))
        assert client.get("/api/v1/users/me").status_code == 401
        client.post("/api/v1/auth/login", json={"identifier": "session-owner", "password": "password123"})
        user = application.store.get_user(created["id"])
        assert user is not None
        application.store.save_user(user.model_copy(update={"is_active": False}))
        assert client.get("/api/v1/users/me").status_code == 401


def test_python_tool_is_disabled_without_explicit_configuration(application):
    from app.core.models import ToolCall

    result = asyncio.run(application.tool_executor.execute(ToolCall(name="python.execute", arguments={"code": "print(1)"}), agent_id="main", services=application.execution_services(), internal=True))
    assert result.status.value == "BLOCKED"
    assert result.error is not None
    assert result.error.code == "TOOL_ENVIRONMENT_UNAVAILABLE"
