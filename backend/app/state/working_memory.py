"""Task 级 WorkingMemory 的加载、合并和工具结果更新。"""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.models import (
    PendingQuestion,
    ToolResult,
    ToolStatus,
    WorkingMemory,
    WorkingMemoryDelta,
    WorkingMemoryItem,
)

from .store import StateStore


class WorkingMemoryUpdater:
    """集中处理工作状态，避免把字段更新分散到各个 GIS Tool。"""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def load_or_create(self, task_id: str, conversation_id: str | None = None) -> WorkingMemory:
        memory = self.store.get_working_memory(task_id)
        if memory is not None:
            return memory
        memory = WorkingMemory(task_id=task_id, conversation_id=conversation_id)
        self.store.save_working_memory(memory)
        return memory

    def update_from_tool_result(self, task_id: str | None, result: ToolResult, *, run_id: str | None) -> WorkingMemory | None:
        if not task_id:
            return None
        memory = self.store.get_working_memory(task_id)
        if memory is None:
            return None
        delta = self.build_delta_from_tool_result(result, run_id=run_id)
        updated = self.apply_delta(memory, delta)
        self.store.save_working_memory(updated)
        return updated

    def build_delta_from_tool_result(self, result: ToolResult, *, run_id: str | None) -> WorkingMemoryDelta:
        intermediate: list[WorkingMemoryItem] = []
        if result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS} or result.error is not None:
            intermediate.append(
                WorkingMemoryItem(
                    kind="tool_result",
                    reference_id=result.call_id,
                    summary=_result_summary(result),
                    source_run_id=run_id,
                )
            )

        return WorkingMemoryDelta(
            added_dataset_ids=list(dict.fromkeys(result.datasets)),
            added_artifact_ids=list(dict.fromkeys(result.artifacts)),
            intermediate_results=intermediate,
            source_run_id=run_id,
        )

    def build_unresolved_question_delta(self, questions: list[str], *, run_id: str | None) -> WorkingMemoryDelta:
        return WorkingMemoryDelta(
            unresolved_questions=list(dict.fromkeys(item.strip() for item in questions if item and item.strip())),
            source_run_id=run_id,
        )

    def add_unresolved_questions(self, task_id: str | None, questions: list[str], *, run_id: str | None) -> WorkingMemory | None:
        if not task_id:
            return None
        memory = self.store.get_working_memory(task_id)
        if memory is None:
            return None
        updated = self.apply_delta(memory, self.build_unresolved_question_delta(questions, run_id=run_id))
        pending = list(updated.pending_questions)
        intermediates = list(updated.intermediate_results)
        for question in questions:
            normalized = question.strip()
            if normalized and not any(item.content.casefold() == normalized.casefold() and item.source_run_id == run_id for item in pending):
                pending.append(PendingQuestion(content=normalized, source_run_id=run_id))
            if normalized:
                _append_intermediate_unique(intermediates, WorkingMemoryItem(kind="question", reference_id=run_id, summary=f"待补充：{normalized}", source_run_id=run_id))
        updated = updated.model_copy(update={"pending_questions": pending, "intermediate_results": intermediates, "updated_at": _now()})
        self.store.save_working_memory(updated)
        return updated

    def resolve_questions(self, task_id: str | None, *, source_run_id: str | None) -> WorkingMemory | None:
        """只清理指定 waiting Run 产生的问题，不影响其他待确认事项。"""

        if not task_id:
            return None
        memory = self.store.get_working_memory(task_id)
        if memory is None:
            return None
        if source_run_id is None:
            return memory
        pending = [item for item in memory.pending_questions if item.source_run_id != source_run_id]
        question_texts = {item.content.casefold() for item in pending}
        # 老版本只有字符串列表时，按对应 source_run 的中间结果删除；
        # 新版本的 PendingQuestion 会成为精确来源。
        if memory.pending_questions:
            remaining_text = [question for question in memory.unresolved_questions if question.casefold() in question_texts]
        else:
            remaining_text = list(memory.unresolved_questions)
        intermediates = [
            item for item in memory.intermediate_results
            if not (item.kind == "question" and item.source_run_id == source_run_id)
        ]
        updated = memory.model_copy(update={
            "pending_questions": pending,
            "unresolved_questions": remaining_text,
            "intermediate_results": intermediates,
            "updated_at": _now(),
        })
        self.store.save_working_memory(updated)
        return updated

    def merge_delta(self, current: WorkingMemoryDelta, incoming: WorkingMemoryDelta) -> WorkingMemoryDelta:
        """合并同一 SubAgent 的多个工具结果，不写入 StateStore。"""

        datasets = list(current.added_dataset_ids)
        artifacts = list(current.added_artifact_ids)
        unresolved = list(current.unresolved_questions)
        for dataset_id in incoming.added_dataset_ids:
            _append_unique(datasets, dataset_id)
        for artifact_id in incoming.added_artifact_ids:
            _append_unique(artifacts, artifact_id)
        for question in incoming.unresolved_questions:
            _append_text_unique(unresolved, question)

        intermediate = list(current.intermediate_results)
        for item in incoming.intermediate_results:
            _append_intermediate_unique(intermediate, item)
        return WorkingMemoryDelta(
            added_dataset_ids=datasets,
            added_artifact_ids=artifacts,
            intermediate_results=intermediate,
            unresolved_questions=unresolved,
            source_run_id=current.source_run_id or incoming.source_run_id,
        )

    def apply_delta(self, memory: WorkingMemory, delta: WorkingMemoryDelta) -> WorkingMemory:
        dataset_ids = list(memory.active_dataset_ids)
        artifact_ids = list(memory.active_artifact_ids)
        for dataset_id in delta.added_dataset_ids:
            _append_unique(dataset_ids, dataset_id)
        for artifact_id in delta.added_artifact_ids:
            _append_unique(artifact_ids, artifact_id)

        intermediate = list(memory.intermediate_results)
        for item in delta.intermediate_results:
            _append_intermediate_unique(intermediate, item)
        unresolved = list(memory.unresolved_questions)
        for question in delta.unresolved_questions:
            _append_text_unique(unresolved, question)
        return memory.model_copy(
            update={
                "active_dataset_ids": dataset_ids,
                "active_artifact_ids": artifact_ids,
                "intermediate_results": intermediate,
                "unresolved_questions": unresolved,
                "updated_at": _now(),
            }
        )

    def merge_deltas(self, memory: WorkingMemory, deltas: list[WorkingMemoryDelta]) -> WorkingMemory:
        """确定性合并多个并行 SubAgent delta，不在此处写库。"""

        merged = memory
        for delta in sorted(deltas, key=_delta_sort_key):
            merged = self.apply_delta(merged, delta)
        return merged


def _result_summary(result: ToolResult) -> str:
    if result.datasets:
        return f"工具完成，涉及数据集：{', '.join(result.datasets)}。"
    if result.artifacts:
        return f"工具完成，生成产物：{', '.join(result.artifacts)}。"
    if result.error is not None:
        return f"工具未完成：{result.error.code}。"
    if isinstance(result.output, dict):
        keys = list(result.output)[:6]
        return f"工具完成，返回结构化结果字段：{', '.join(str(key) for key in keys)}。"
    return "工具执行完成。"


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _append_text_unique(values: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized.casefold() not in {item.casefold() for item in values}:
        values.append(normalized)


def _append_intermediate_unique(values: list[WorkingMemoryItem], item: WorkingMemoryItem) -> None:
    key = (item.reference_id, item.source_run_id, item.summary)
    if not any((current.reference_id, current.source_run_id, current.summary) == key for current in values):
        values.append(item)


def _delta_sort_key(delta: WorkingMemoryDelta) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return (delta.source_run_id or "", tuple(delta.added_dataset_ids), tuple(delta.added_artifact_ids))


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = ["WorkingMemoryUpdater"]
