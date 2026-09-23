"""Evaluation case 与单次执行结果模型。"""

from __future__ import annotations

from pydantic import Field

from app.core.models import AgentResultStatus, StrictModel


class EvaluationCase(StrictModel):
    """一条可重复执行的离线评测定义。"""

    id: str
    name: str
    prompt: str
    dataset_keys: list[str] = Field(default_factory=list)
    dataset_ids: list[str] = Field(default_factory=list)
    expected_status: AgentResultStatus | None = None
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expect_delegation: bool = False
    expected_subtask_count: int | None = None
    expected_recovery: list[str] = Field(default_factory=list)
    expected_directive: str | None = None
    max_tool_calls: int | None = None
    enabled: bool = True
    notes: str = ""


class EvaluationCaseResult(StrictModel):
    """一条评测的可审计结果，不保存完整 Prompt 或 Tool 输出。"""

    case_id: str
    passed: bool
    actual_status: AgentResultStatus | None = None
    expected_status: AgentResultStatus | None = None
    selected_tools: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    delegation_count: int = 0
    subtask_count: int = 0
    subagent_statuses: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    runtime_transitions: int = 0
    execution_time_ms: float = 0.0
    budget_violation: bool = False
    failures: list[str] = Field(default_factory=list)


__all__ = ["EvaluationCase", "EvaluationCaseResult"]
