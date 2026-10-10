"""Task 目标状态、Run 执行状态与恢复快照的持久化边界。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.models import (
    AgentResultStatus,
    ApprovalRequest,
    ApprovalStatus,
    Checkpoint,
    Run,
    RunStatus,
    Task,
    TaskStatus,
)
from app.state import StateStore


def finish_run(run: Run, status: RunStatus, *, error: str | None = None) -> Run:
    from .predicates import EXECUTION_INFLIGHT_STATUSES, HUMAN_WAITING_STATUSES

    stopped = status not in EXECUTION_INFLIGHT_STATUSES | HUMAN_WAITING_STATUSES | {RunStatus.CREATED}
    return run.model_copy(update={"status": status, "error": error, "finished_at": datetime.now(UTC) if stopped else None})


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
        return TaskStatus.BLOCKED
    return TaskStatus.FAILED


def task_status_for_run(run: Run) -> TaskStatus:
    """导入历史任务或处理进程中断时，保留执行状态的真实含义。"""
    from .predicates import is_execution_inflight, is_waiting_for_human

    if is_execution_inflight(run):
        return TaskStatus.RUNNING
    if is_waiting_for_human(run):
        return TaskStatus.WAITING
    return {
        RunStatus.CREATED: TaskStatus.PENDING,
        RunStatus.COMPLETED: TaskStatus.SUCCEEDED,
        RunStatus.PARTIAL_COMPLETED: TaskStatus.PARTIAL,
        RunStatus.CANCELLED: TaskStatus.CANCELLED,
        RunStatus.INTERRUPTED: TaskStatus.BLOCKED,
        RunStatus.BUDGET_EXCEEDED: TaskStatus.BLOCKED,
        RunStatus.FAILED: TaskStatus.FAILED,
    }[run.status]


def persist_result(
    store: StateStore,
    run: Run,
    task: Task | None,
    result,
    *,
    run_status: RunStatus | None = None,
    task_status: TaskStatus | None = None,
    metadata: dict[str, Any] | None = None,
    checkpoint: Checkpoint | None = None,
) -> tuple[Run, Task | None]:
    """唯一负责把 Runtime 结果映射并原子写入 Run/Task 的生命周期入口。"""

    resolved_run_status = run_status or run_status_for_result(result.status, result.error)
    resolved_task_status = task_status or task_status_for_result(result.status, result.error)
    reason = result.error
    if task is not None and task_status is None:
        previous = store.get_run(task.progress_run_id) if task.progress_run_id else None
        review = run.metadata.get("completion_review")
        prior_review = previous.metadata.get("completion_review") if previous else None
        current_delivery = any(item["status"] == "satisfied" and item.get("evidence_refs") for item in (review or {}).get("items", []))
        has_delivery = current_delivery or any(item["status"] == "satisfied" and item.get("evidence_refs") for item in (prior_review or {}).get("items", []))
        if result.error in {"SIDE_EFFECT_UNCERTAIN", "COMPLETION_REVIEW_UNAVAILABLE"}:
            resolved_task_status = TaskStatus.BLOCKED
        elif result.status is AgentResultStatus.FAILED and has_delivery:
            resolved_task_status = TaskStatus.PARTIAL
        elif (result.status is AgentResultStatus.SUCCESS and run.metadata.get("task_relation") == "continue" and review is None
              and (prior_review is not None or any(item.tool_call_count for item in store.list_runs_for_task(task.id)))):
            # 纯回答仍免审核；有执行交付历史时，不能用一次纯答复抹掉尚未核验的交付要求。
            resolved_task_status = TaskStatus.PARTIAL if has_delivery else TaskStatus.BLOCKED
            reason = "COMPLETION_UNVERIFIED"
        if review and (current_delivery or review["decision"] == "accept"):
            task = task.model_copy(update={"progress_run_id": run.id})
    return transition(
        store,
        run,
        task,
        run_status=resolved_run_status,
        task_status=resolved_task_status,
        task_reason=reason,
        error=result.error,
        result_text=result.summary,
        metadata={"result": result.model_dump(mode="json"), **(metadata or {})},
        checkpoint=checkpoint,
    )


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
    updated_task = task.model_copy(update={"status": TaskStatus.RUNNING, "status_reason": None, "updated_at": datetime.now(UTC)}) if task is not None else None
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
    task_reason: str | None = None,
    error: str | None = None,
    result_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    checkpoint: Checkpoint | None = None,
) -> tuple[Run, Task | None]:
    """原子写入一次 Run/Task 状态转换。"""

    latest = store.get_run(run.id)
    if latest is not None and latest.status is RunStatus.CANCELLED and run_status is not RunStatus.CANCELLED:
        return latest, store.get_task(latest.task_id) if latest.task_id else None
    updated_run = finish_run(run, run_status, error=error)
    if metadata:
        updated_run = updated_run.model_copy(update={"metadata": {**run.metadata, **metadata}})
    updated_task = None
    if task is not None and task_status is not None:
        updated_task = task.model_copy(update={"status": task_status, "status_reason": task_reason or error,
                                               "result": result_text if result_text is not None else task.result, "updated_at": datetime.now(UTC)})
    store.save_run_and_task(updated_run, updated_task, checkpoint=checkpoint)
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
    "record_approval_decision",
    "resume",
    "run_status_for_result",
    "transition",
    "task_status_for_result",
    "task_status_for_run",
]
