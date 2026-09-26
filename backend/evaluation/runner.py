"""可重复的离线执行评测 Runner。"""

from __future__ import annotations

import time
from collections.abc import Iterable

from app.core.models import AgentRequest, AgentResultStatus
from app.demo import seed_demo

from .cases import default_cases
from .metrics import EvaluationSummary
from .models import EvaluationCase, EvaluationCaseResult


class EvaluationRunner:
    def __init__(self, application, cases: Iterable[EvaluationCase] | None = None, *, user_id: str | None = None) -> None:
        self.application = application
        self.user_id = user_id
        self.cases = list(cases) if cases is not None else default_cases()

    async def run(self) -> EvaluationSummary:
        ids = seed_demo(self.application)
        cases = [item for item in self.cases if item.enabled]
        summary = EvaluationSummary(total=len(cases))
        started = time.perf_counter()
        for case in cases:
            result = await self._run_case(case, ids)
            summary.cases.append(result)
            self._accumulate(summary, case, result)
        summary.execution_time_ms = round((time.perf_counter() - started) * 1000, 2)
        return summary

    async def _run_case(self, case: EvaluationCase, demo_ids: dict[str, str]) -> EvaluationCaseResult:
        dataset_ids = list(case.dataset_ids) or [demo_ids[key] for key in case.dataset_keys if key in demo_ids]
        started = time.perf_counter()
        result = await self.application.ask(AgentRequest(user_input=case.prompt, dataset_ids=dataset_ids, user_id=self.user_id))
        events = self.application.store.list_events(result.trace_id)
        run = self.application.store.get_run(result.trace_id)
        selected_tools = [str(event.payload.get("tool")) for event in events if event.event_type == "ToolStarted" and event.payload.get("tool")]
        recovery_actions = _recovery_actions(events)
        delegation_events = [event for event in events if event.event_type == "DelegationCompleted"]
        subtask_events = [event for event in events if event.event_type == "SubTaskCreated"]
        subagent_statuses = [str(event.payload.get("status")) for event in events if event.event_type == "SubAgentCompleted" and event.payload.get("status")]
        tool_calls = run.tool_call_count if run else len(selected_tools)
        budget_violation = bool(run and str(run.status) == "BUDGET_EXCEEDED") or "BUDGET_EXCEEDED" in str(result.error or "")
        runtime_transitions = sum(event.event_type in {"DecisionMade", "ToolStarted", "DelegationCompleted"} for event in events)
        failures: list[str] = []
        if case.expected_status is not None and result.status is not case.expected_status:
            failures.append(f"status: 期望 {case.expected_status.value}，实际 {result.status.value}")
        missing_tools = [name for name in case.expected_tools if name not in selected_tools]
        if missing_tools:
            failures.append(f"缺少工具调用：{', '.join(missing_tools)}")
        unexpected = [name for name in selected_tools if name in case.forbidden_tools]
        if unexpected:
            failures.append(f"出现禁止工具调用：{', '.join(unexpected)}")
        if case.expect_delegation and not delegation_events:
            failures.append("未发生委派")
        if case.expected_subtask_count is not None and len(subtask_events) != case.expected_subtask_count:
            failures.append(f"SubTask 数量不符：期望 {case.expected_subtask_count}，实际 {len(subtask_events)}")
        missing_recovery = [action for action in case.expected_recovery if action not in recovery_actions]
        if missing_recovery:
            failures.append(f"缺少恢复动作：{', '.join(missing_recovery)}")
        if case.expected_directive is not None:
            directives = _event_directives(events)
            if case.expected_directive not in directives:
                failures.append(f"缺少运行时 directive：{case.expected_directive}")
        if case.max_tool_calls is not None and tool_calls > case.max_tool_calls:
            failures.append(f"工具调用超预算：{tool_calls} > {case.max_tool_calls}")
        return EvaluationCaseResult(
            case_id=case.id,
            passed=not failures,
            actual_status=result.status,
            expected_status=case.expected_status,
            selected_tools=list(dict.fromkeys(selected_tools)),
            recovery_actions=recovery_actions,
            delegation_count=len(delegation_events),
            subtask_count=len(subtask_events),
            subagent_statuses=subagent_statuses,
            tool_calls=tool_calls,
            runtime_transitions=runtime_transitions,
            execution_time_ms=round((time.perf_counter() - started) * 1000, 2),
            budget_violation=budget_violation,
            failures=failures,
        )

    @staticmethod
    def _accumulate(summary: EvaluationSummary, case: EvaluationCase, result: EvaluationCaseResult) -> None:
        if result.passed:
            summary.passed += 1
        if result.actual_status is AgentResultStatus.SUCCESS:
            summary.succeeded += 1
        elif result.actual_status is AgentResultStatus.PARTIAL:
            summary.partial += 1
        else:
            summary.failed += 1
        summary.tool_calls += result.tool_calls
        if case.expected_tools:
            summary.tool_selection_cases += 1
            if all(tool in result.selected_tools for tool in case.expected_tools):
                summary.tool_selection_passed += 1
        if case.expected_recovery:
            summary.recovery_cases += 1
            if all(action in result.recovery_actions for action in case.expected_recovery):
                summary.recovery_cases_passed += 1
        if case.expect_delegation:
            summary.delegation_cases += 1
            if result.delegation_count > 0 and not result.failures:
                summary.delegation_passed += 1
        if "dependency" in case.id:
            summary.dependency_handling_cases += 1
            if result.passed:
                summary.dependency_handling_passed += 1
        if result.budget_violation:
            summary.budget_violation_count += 1
        summary.unexpected_tool_call_count += sum(tool in case.forbidden_tools for tool in result.selected_tools)


def _recovery_actions(events) -> list[str]:
    mapping = {
        "RetryStarted": "RETRY",
        "RepairSelected": "REPAIR",
        "ReplanStarted": "REPLAN",
    }
    return list(dict.fromkeys(mapping[event.event_type] for event in events if event.event_type in mapping))


def _event_directives(events) -> list[str]:
    values = [str(event.payload.get("directive")) for event in events if event.payload.get("directive")]
    return list(dict.fromkeys(values))


__all__ = ["EvaluationRunner"]
