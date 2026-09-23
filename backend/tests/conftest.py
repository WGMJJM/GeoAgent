from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.application import Application
from app.config import Settings


@pytest.fixture
def application(tmp_path) -> Iterator[Application]:
    settings = Settings(root=tmp_path, database=tmp_path / "state.sqlite3", workspace=tmp_path / "workspace", model_profiles="")
    app = Application(settings)
    app.start()
    try:
        yield app
    finally:
        # 无模型客户端时 close 也可以同步结束；测试不启动网络服务。
        import asyncio

        asyncio.run(app.close())


class AuthenticatedClient:
    def __init__(self, application: Application) -> None:
        self.application = application
        self.client: TestClient | None = None

    def __enter__(self) -> TestClient:
        self.client = TestClient(create_app(self.application))
        self.client.__enter__()
        response = self.client.post("/api/v1/auth/register", json={"username": "test-user", "password": "password123", "display_name": "测试用户"})
        if response.status_code == 400 and "用户名已存在" in response.text:
            response = self.client.post("/api/v1/auth/login", json={"identifier": "test-user", "password": "password123"})
        assert response.status_code == 200, response.text
        return self.client

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.client is not None:
            self.client.__exit__(exc_type, exc, traceback)


@pytest.fixture
def authenticated_client(application) -> AuthenticatedClient:
    return AuthenticatedClient(application)
