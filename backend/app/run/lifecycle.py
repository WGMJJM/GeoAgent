"""Run 与既有用户交互状态的持久化边界。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.models import (
    AgentResultStatus,
    ApprovalRequest,
    ApprovalStatus,
    Run,
    RunStatus,
    Task,
    TaskStatus,
)
from app.run.predicates import is_cancellable_run, is_retryable_failed_run
from app.state import StateStore


def start_run(run: Run) -> Run:
    return run.model_copy(update={"status": RunStatus.RUNNING, "started_at": datetime.now(UTC)})


def finish_run(run: Run, status: RunStatus, *, error: str | None = None) -> Run:
    return run.model_copy(update={"status": status, "error": error, "finished_at": datetime.now(UTC)})


def run_status_for_result(status: AgentResultStatus, error: str | None = None) -> RunStatus:
    if status is AgentResultStatus.SUCCESS:
        return RunStatus.COMPLETED
    if status is AgentResultStatus.PARTIAL:
        return RunStatus.PARTIAL_COMPLETED
    if status is AgentResultStatus.CANCELLED:
        return RunStatus.CANCELLED
    if status is AgentResultStatus.BLOCKED:
        if error == "APPROVAL_REQUIRED":
            return RunStatus.WAITING_APPROVAL
        if error in {"WAITING_USER", "NEEDS_CLARIFICATION"}:
            return RunStatus.WAITING_USER
        if error == "BUDGET_EXCEEDED":
            return RunStatus.BUDGET_EXCEEDED
        return RunStatus.FAILED
    return RunStatus.FAILED


def task_status_for_result(status: AgentResultStatus, error: str | None = None) -> TaskStatus:
    if status is AgentResultStatus.SUCCESS:
        return TaskStatus.SUCCEEDED
    if status is AgentResultStatus.PARTIAL:
        return TaskStatus.PARTIAL
    if status is AgentResultStatus.CANCELLED:
        return TaskStatus.CANCELLED
    if status is AgentResultStatus.BLOCKED:
        if error in {"WAITING_USER", "NEEDS_CLARIFICATION", "APPROVAL_REQUIRED"}:
            return TaskStatus.WAITING
        if error == "BUDGET_EXCEEDED":
            return TaskStatus.BLOCKED
        return TaskStatus.BLOCKED
    return TaskStatus.FAILED


def persist_result(
    store: StateStore,
    run: Run,
    task: Task | None,
    result,
    *,
    run_status: RunStatus | None = None,
    task_status: TaskStatus | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Run, Task | None]:
    """唯一负责把 Runtime 结果映射并原子写入 Run/Task 的生命周期入口。"""

    resolved_run_status = run_status or run_status_for_result(result.status, result.error)
    resolved_task_status = task_status or task_status_for_result(result.status, result.error)
    return transition(
        store,
        run,
        task,
        run_status=resolved_run_status,
        task_status=resolved_task_status,
        error=result.error,
        result_text=result.summary,
        metadata={"result": result.model_dump(mode="json"), **(metadata or {})},
    )


def persist_run(store: StateStore, run: Run) -> Run:
    """Lifecycle 统一承接运行时进度类 Run 写入。"""

    store.save_run(run)
    return run


def resume(
    store: StateStore,
    run: Run,
    task: Task | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    approval: ApprovalRequest | None = None,
) -> tuple[Run, Task | None]:
    """在同一个 Run 上恢复一次等待中的 Runtime。"""

    updated_run = run.model_copy(
        update={
            "status": RunStatus.RUNNING,
            "error": None,
            "finished_at": None,
            "metadata": {**run.metadata, **(metadata or {})},
        }
    )
    updated_task = task.model_copy(update={"status": TaskStatus.RUNNING, "result": None, "updated_at": datetime.now(UTC)}) if task is not None else None
    if approval is None:
        store.save_run_and_task(updated_run, updated_task)
    else:
        store.save_approval_and_run(approval, updated_run, updated_task)
    return updated_run, updated_task


def transition(
    store: StateStore,
    run: Run,
    task: Task | None = None,
    *,
    run_status: RunStatus,
    task_status: TaskStatus | None = None,
    error: str | None = None,
    result_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Run, Task | None]:
    """原子写入一次 Run/Task 状态转换。"""

    updated_run = finish_run(run, run_status, error=error)
    if metadata:
        updated_run = updated_run.model_copy(update={"metadata": {**run.metadata, **metadata}})
    updated_task = None
    if task is not None and task_status is not None:
        updated_task = task.model_copy(update={"status": task_status, "result": result_text, "updated_at": datetime.now(UTC)})
    store.save_run_and_task(updated_run, updated_task)
    return updated_run, updated_task


def record_approval_denied(store: StateStore, approval: ApprovalRequest) -> tuple[Run, Task | None] | None:
    """把审批拒绝后的 Run/Task 状态收敛到生命周期入口。"""

    run = store.get_run(approval.source_run_id)
    if run is None:
        store.save_approval(approval)
        return None
    task = store.get_task(run.task_id) if run.task_id else None
    if run.status is not RunStatus.WAITING_APPROVAL:
        store.save_approval_and_run(approval, run, task)
        return run, task
    if task is not None and task.status in {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
        store.save_approval_and_run(approval, run, task)
        return run, task
    updated_run = finish_run(run, RunStatus.WAITING_USER, error="APPROVAL_DENIED")
    updated_run = updated_run.model_copy(update={"metadata": {**run.metadata, "approval_denied": approval.id}})
    updated_task = task.model_copy(update={"status": TaskStatus.WAITING, "result": "等待用户选择其他方案", "updated_at": datetime.now(UTC)}) if task is not None else None
    store.save_approval_and_run(approval, updated_run, updated_task)
    return updated_run, updated_task


def record_approval_decision(
    store: StateStore,
    approval: ApprovalRequest,
) -> tuple[Run, Task | None] | None:
    """原子保存审批决定，并让同一个 Run 回到 Runtime。"""

    run = store.get_run(approval.source_run_id)
    if run is None:
        store.save_approval(approval)
        return None
    task = store.get_task(run.task_id) if run.task_id else None
    if run.status is not RunStatus.WAITING_APPROVAL:
        store.save_approval_and_run(approval, run, task)
        return run, task
    return resume(
        store,
        run,
        task,
        metadata={
            "approval_decision": approval.status.value,
            "approval_id": approval.id,
        },
        approval=approval,
    ) if approval.status in {ApprovalStatus.APPROVED, ApprovalStatus.DENIED} else None


__all__ = [
    "finish_run",
    "persist_result",
    "persist_run",
    "record_approval_decision",
    "record_approval_denied",
    "resume",
    "run_status_for_result",
    "start_run",
    "transition",
    "task_status_for_result",
]
