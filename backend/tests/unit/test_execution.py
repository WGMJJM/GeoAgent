import asyncio
import subprocess
from threading import Event
from types import SimpleNamespace

import pytest

from app.core.models import ToolCall, ToolMetadata
from app.demo import seed_demo
from app.execution.process import ProcessCancelled, run_process
from app.execution.python import PythonExecutor
from app.execution.sandbox import WorkspaceManager
from app.execution.shell import ShellExecutor


def test_shell_runs_argument_list_without_shell(monkeypatch, tmp_path):
    workspace = WorkspaceManager(tmp_path / "workspace")
    captured = {}

    def fake_run(arguments, **kwargs):
        captured["arguments"] = arguments
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("app.execution.shell.executor.run_process", fake_run)

    result = ShellExecutor(workspace).execute('ogrinfo "input/roads.geojson"')

    assert result.returncode == 0
    assert captured["arguments"] == ["ogrinfo", "input/roads.geojson"]
    assert captured["timeout_seconds"] == 120


def test_process_starts_without_a_shell(monkeypatch, tmp_path):
    captured = {}

    class FinishedProcess:
        returncode = 0

        def poll(self):
            return self.returncode

        def communicate(self, **_kwargs):
            return "", ""

    def fake_popen(arguments, **kwargs):
        captured["arguments"] = arguments
        captured.update(kwargs)
        return FinishedProcess()

    monkeypatch.setattr("app.execution.process.subprocess.Popen", fake_popen)

    result = run_process(["ogrinfo", "input/roads.geojson"], cwd=tmp_path, timeout_seconds=1)

    assert result.returncode == 0
    assert captured["shell"] is False


def test_shell_rejects_command_chaining_and_malformed_quotes(tmp_path):
    executor = ShellExecutor(WorkspaceManager(tmp_path / "workspace"))

    with pytest.raises(PermissionError):
        executor.execute("ogrinfo input/roads.geojson && whoami")
    with pytest.raises(PermissionError):
        executor.execute(r"ogrinfo C:\outside\roads.geojson")
    with pytest.raises(ValueError, match="引号不匹配"):
        executor.execute('ogrinfo "input/roads.geojson')


def test_tool_executor_normalizes_subprocess_timeout(application):
    def timed_out(_arguments, _context):
        raise subprocess.TimeoutExpired(cmd="ogrinfo", timeout=1)

    application.tool_registry.register(
        ToolMetadata(name="test.timeout", description="timeout test", supports_retry=True),
        timed_out,
    )
    call = ToolCall(name="test.timeout")

    result = asyncio.run(application.tool_executor.execute(call, agent_id="main", services=application.tool_executor.services))

    assert result.error is not None
    assert result.error.code == "EXECUTION_TIMEOUT"
    assert result.error.details == {"timeout_seconds": application.tool_executor.timeout_seconds}


def test_tool_executor_signals_handler_when_async_timeout_expires(application):
    stopped = Event()

    def blocking(_arguments, context):
        while not context.cancel_event.is_set():
            stopped.wait(0.01)
        stopped.set()

    application.tool_registry.register(ToolMetadata(name="test.blocking", description="blocking test"), blocking)
    application.tool_executor.timeout_seconds = 0.05
    result = asyncio.run(
        application.tool_executor.execute(
            ToolCall(name="test.blocking"),
            agent_id="main",
            services=application.tool_executor.services,
        )
    )

    assert result.error is not None
    assert result.error.code == "EXECUTION_TIMEOUT"
    assert stopped.wait(1)


def test_tool_executor_signals_handler_when_execution_is_cancelled(application):
    started = Event()
    stopped = Event()

    def blocking(_arguments, context):
        started.set()
        while not context.cancel_event.is_set():
            stopped.wait(0.01)
        stopped.set()

    application.tool_registry.register(ToolMetadata(name="test.cancelled", description="cancel test"), blocking)

    async def run_case():
        task = asyncio.create_task(
            application.tool_executor.execute(
                ToolCall(name="test.cancelled"),
                agent_id="main",
                services=application.tool_executor.services,
            )
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_case())
    assert stopped.wait(1)


def test_python_process_stops_when_cancel_event_is_set(tmp_path):
    executor = PythonExecutor(WorkspaceManager(tmp_path / "workspace"), timeout_seconds=10)
    cancel_event = Event()

    async def run_case():
        task = asyncio.create_task(asyncio.to_thread(executor.execute, "import time; time.sleep(10)", cancel_event=cancel_event))
        await asyncio.sleep(0.1)
        cancel_event.set()
        with pytest.raises(ProcessCancelled):
            await asyncio.wait_for(task, timeout=2)

    asyncio.run(run_case())


def test_demo_registration_is_idempotent(application):
    first = seed_demo(application)
    second = seed_demo(application)

    assert second == first
    assert len(application.registry.list()) == 3
