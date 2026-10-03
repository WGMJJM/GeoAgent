"""Run 状态领域谓词，区分进程执行、人工等待、终态和技术恢复。"""

from app.core.models import Run, RunStatus

EXECUTION_INFLIGHT_STATUSES = frozenset(
    {
        RunStatus.RUNNING,
        RunStatus.WAITING_TOOL,
        RunStatus.WAITING_SUBAGENT,
        RunStatus.RETRYING,
    }
)
HUMAN_WAITING_STATUSES = frozenset({RunStatus.WAITING_USER, RunStatus.WAITING_APPROVAL})
TECHNICAL_RESUMABLE_STATUSES = frozenset({RunStatus.INTERRUPTED, RunStatus.CANCELLED, RunStatus.BUDGET_EXCEEDED})


def is_execution_inflight(run: Run) -> bool:
    return run.status in EXECUTION_INFLIGHT_STATUSES


def is_waiting_for_human(run: Run) -> bool:
    return run.status in HUMAN_WAITING_STATUSES


def is_cancellable_run(run: Run) -> bool:
    return is_execution_inflight(run) or is_waiting_for_human(run)


def is_resumable_run(run: Run, *, has_checkpoint: bool) -> bool:
    return has_checkpoint and run.status in TECHNICAL_RESUMABLE_STATUSES


def is_retryable_failed_run(run: Run) -> bool:
    """人工重试只针对失败的主 Run；副作用未确认不能重新发起。"""

    return run.status is RunStatus.FAILED and run.parent_run_id is None and run.error != "SIDE_EFFECT_UNCERTAIN"


__all__ = [
    "EXECUTION_INFLIGHT_STATUSES",
    "HUMAN_WAITING_STATUSES",
    "TECHNICAL_RESUMABLE_STATUSES",
    "is_cancellable_run",
    "is_execution_inflight",
    "is_resumable_run",
    "is_retryable_failed_run",
    "is_waiting_for_human",
]
