import asyncio
from threading import Event

import pytest

from app.core.models import ToolCall, ToolMetadata
from app.execution.process import ProcessCancelled
from app.execution.python import PythonExecutor
from app.execution.sandbox import WorkspaceManager
from app.execution.shell import ShellExecutor


def test_shell_rejects_command_chaining_and_malformed_quotes(tmp_path):
    executor = ShellExecutor(WorkspaceManager(tmp_path / "workspace"))

    with pytest.raises(PermissionError):
        executor.execute("ogrinfo input/roads.geojson && whoami")
    with pytest.raises(PermissionError):
        executor.execute(r"ogrinfo C:\outside\roads.geojson")
    with pytest.raises(ValueError, match="引号不匹配"):
        executor.execute('ogrinfo "input/roads.geojson')


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
                internal=True,
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
