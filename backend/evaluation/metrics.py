from pydantic import BaseModel, Field, computed_field

from .models import EvaluationCaseResult


class EvaluationSummary(BaseModel):
    total: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    succeeded: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    recovery_cases_passed: int = Field(default=0, ge=0)
    recovery_cases: int = Field(default=0, ge=0)
    tool_selection_cases: int = Field(default=0, ge=0)
    tool_selection_passed: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    execution_time_ms: float = Field(default=0.0, ge=0.0)
    cases: list[EvaluationCaseResult] = Field(default_factory=list)
    delegation_cases: int = Field(default=0, ge=0)
    delegation_passed: int = Field(default=0, ge=0)
    dependency_handling_cases: int = Field(default=0, ge=0)
    dependency_handling_passed: int = Field(default=0, ge=0)
    budget_violation_count: int = Field(default=0, ge=0)
    unexpected_tool_call_count: int = Field(default=0, ge=0)

    @computed_field
    @property
    def pass_rate(self) -> float:
        """评测用例通过率，不等同于 Agent 返回 SUCCESS 的业务比例。"""

        return self.passed / self.total if self.total else 0.0

    @computed_field
    @property
    def agent_success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0

    @property
    def success_rate(self) -> float:
        """兼容旧调用方；新报告请使用 pass_rate 或 agent_success_rate。"""

        return self.agent_success_rate

    @property
    def recovery_success_rate(self) -> float:
        return self.recovery_cases_passed / self.recovery_cases if self.recovery_cases else 0.0

    @property
    def delegation_success_rate(self) -> float:
        return self.delegation_passed / self.delegation_cases if self.delegation_cases else 0.0

    @property
    def dependency_handling_rate(self) -> float:
        return self.dependency_handling_passed / self.dependency_handling_cases if self.dependency_handling_cases else 0.0

    @property
    def expected_status_accuracy(self) -> float:
        checked = [item for item in self.cases if item.expected_status is not None]
        return sum(item.actual_status is item.expected_status for item in checked) / len(checked) if checked else 0.0

    @property
    def tool_selection_accuracy(self) -> float:
        return self.tool_selection_passed / self.tool_selection_cases if self.tool_selection_cases else 0.0

    @property
    def average_tool_calls(self) -> float:
        return sum(item.tool_calls for item in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def average_runtime_transitions(self) -> float:
        return sum(item.runtime_transitions for item in self.cases) / len(self.cases) if self.cases else 0.0

    @property
    def average_execution_time_ms(self) -> float:
        return self.execution_time_ms / self.total if self.total else 0.0
