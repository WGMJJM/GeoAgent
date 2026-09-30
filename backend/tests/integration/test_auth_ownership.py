from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.api import create_app
from app.auth.password import hash_session_token
from app.core.models import Artifact, ArtifactKind, Run, RunStatus


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


def test_conversation_dataset_and_artifact_are_user_scoped(application):
    with TestClient(create_app(application)) as client_a, TestClient(create_app(application)) as client_b:
        user_a = _register(client_a, "alice")
        client_a.post("/api/v1/conversations", json={"title": "Alice 对话"})
        upload = client_a.post("/api/v1/attachments", files={"file": ("alice.csv", b"x,y\n1,2\n", "text/csv")})
        assert upload.status_code == 200
        dataset_id = upload.json()["dataset"]["id"]
        assert upload.json()["dataset"]["owner_user_id"] == user_a["id"]
        run = Run(conversation_id=application.store.list_conversations(user_id=user_a["id"])[0].id, agent_id="main", status=RunStatus.COMPLETED)
        application.store.save_run(run)
        artifact = Artifact(name="alice.txt", kind=ArtifactKind.OTHER, path=None, run_id=run.id, owner_user_id=user_a["id"])
        application.store.save_artifact(artifact)

        _register(client_b, "bob")
        assert client_b.get("/api/v1/conversations").json() == []
        assert client_b.get("/api/v1/datasets").json() == []
        assert client_b.get(f"/api/v1/runs/{run.id}").status_code == 404
        assert client_b.get(f"/api/v1/artifacts/{artifact.id}/content").status_code == 404
        assert client_b.post("/api/v1/messages", json={"message": "检查数据", "dataset_ids": [dataset_id]}).status_code == 403


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
