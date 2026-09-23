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
# CREATED 仍表示已登记但尚未进入执行；规划生命周期不属于 Run，
# 因此旧的 PLANNING / REPLANNING / VALIDATING 不再被视为活跃运行。
ACTIVE_RUN_STATUSES = frozenset({RunStatus.CREATED, *EXECUTION_INFLIGHT_STATUSES, *HUMAN_WAITING_STATUSES})


def is_active_run(run: Run) -> bool:
    return run.status in ACTIVE_RUN_STATUSES


def is_execution_inflight(run: Run) -> bool:
    return run.status in EXECUTION_INFLIGHT_STATUSES


def is_waiting_for_human(run: Run) -> bool:
    return run.status in HUMAN_WAITING_STATUSES


def is_cancellable_run(run: Run) -> bool:
    return is_execution_inflight(run) or is_waiting_for_human(run)


def is_retryable_failed_run(run: Run) -> bool:
    return run.status in {RunStatus.FAILED, RunStatus.INTERRUPTED, RunStatus.BUDGET_EXCEEDED}


def find_retryable_failed_run(runs) -> Run | None:
    """从按最近优先排列的运行中选择第一个可重试失败项。"""

    return next((run for run in runs if is_retryable_failed_run(run)), None)


def is_resumable_run(run: Run, *, has_checkpoint: bool) -> bool:
    return has_checkpoint and run.status in TECHNICAL_RESUMABLE_STATUSES


__all__ = [
    "ACTIVE_RUN_STATUSES",
    "EXECUTION_INFLIGHT_STATUSES",
    "HUMAN_WAITING_STATUSES",
    "TECHNICAL_RESUMABLE_STATUSES",
    "is_active_run",
    "is_cancellable_run",
    "is_execution_inflight",
    "is_resumable_run",
    "is_retryable_failed_run",
    "is_waiting_for_human",
    "find_retryable_failed_run",
]
