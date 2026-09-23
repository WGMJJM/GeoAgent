"""会话级结构化记忆及增量摘要入口。"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from app.core.models import (
    AgentRequest,
    AgentResult,
    ConversationMemory,
    ConversationMemoryEntry,
    Message,
    Run,
    RunStatus,
)
from app.models import ModelAdapter
from app.state import StateStore

from .conversation_summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class ConversationMemoryService:
    MAX_ENTRIES = 20
    MAX_CONTENT_LENGTH = 500

    def __init__(
        self,
        store: StateStore,
        *,
        summarizer: ConversationSummarizer | None = None,
        model_provider: Callable[[str | None], ModelAdapter | None] | None = None,
    ) -> None:
        self.store = store
        self.summarizer = summarizer or ConversationSummarizer(store)
        self.model_provider = model_provider

    def get(self, conversation_id: str, user_id: str) -> ConversationMemory | None:
        return self.store.get_conversation_memory_for_user(conversation_id, user_id)

    def get_or_create(self, conversation_id: str, user_id: str) -> ConversationMemory:
        memory = self.get(conversation_id, user_id)
        if memory is not None:
            return memory
        if self.store.get_conversation_for_user(conversation_id, user_id) is None:
            raise PermissionError("当前用户无权访问该会话")
        memory = ConversationMemory(conversation_id=conversation_id, user_id=user_id)
        self.store.save_conversation_memory(memory)
        return self.get(conversation_id, user_id) or memory

    def apply_result(self, request: AgentRequest, run: Run, result: AgentResult) -> ConversationMemory | None:
        if not request.user_id:
            return None
        memory = self.get_or_create(request.conversation_id, request.user_id)
        stored_run = self.store.get_run(run.id)
        if (
            stored_run is None
            or stored_run.conversation_id != request.conversation_id
            or not self.store.run_belongs_to_user(stored_run.id, request.user_id)
        ):
            return memory

        references: list[ConversationMemoryEntry] = []
        for dataset_id in result.datasets:
            dataset = self.store.get_dataset_for_user(dataset_id, request.user_id)
            if dataset is None:
                continue
            references.append(
                ConversationMemoryEntry(
                    content=f"数据集：{dataset.name}（{dataset.kind.value}）",
                    source_task_id=stored_run.task_id,
                    source_run_id=stored_run.id,
                    reference_type="dataset",
                    reference_id=dataset.id,
                )
            )
        for artifact_id in result.artifacts:
            artifact = self.store.get_artifact_for_user(artifact_id, request.user_id)
            if artifact is None:
                continue
            references.append(
                ConversationMemoryEntry(
                    content=f"结果文件：{artifact.name}",
                    source_task_id=stored_run.task_id,
                    source_run_id=stored_run.id,
                    reference_type="artifact",
                    reference_id=artifact.id,
                )
            )

        unresolved: list[ConversationMemoryEntry] = []
        if stored_run.status in {RunStatus.WAITING_USER, RunStatus.WAITING_APPROVAL} and result.error in {
            "WAITING_USER",
            "NEEDS_CLARIFICATION",
            "APPROVAL_REQUIRED",
        }:
            unresolved.append(
                ConversationMemoryEntry(
                    content=_clip(result.summary),
                    source_task_id=stored_run.task_id,
                    source_run_id=stored_run.id,
                )
            )

        unresolved_topics = memory.unresolved_topics
        if stored_run.status in {RunStatus.COMPLETED, RunStatus.PARTIAL_COMPLETED} and stored_run.task_id:
            source_run_id = stored_run.metadata.get("continued_from") or stored_run.metadata.get("approved_from")
            if source_run_id:
                unresolved_topics = [item for item in unresolved_topics if item.source_run_id != source_run_id]
            elif len(unresolved_topics) == 1:
                # 兼容旧数据：只有唯一待确认项且同一 Task 完成时才清除。
                unresolved_topics = [item for item in unresolved_topics if item.source_task_id != stored_run.task_id]

        updated = self._add_entries(
            memory,
            important_references=references,
            unresolved_topics=unresolved,
            unresolved_topics_override=unresolved_topics,
        )
        self.store.save_conversation_memory(updated)
        return updated

    async def summarize_after_assistant_persisted(self, message: Message) -> None:
        """摘要失败仅记录日志，不改变已经持久化的用户对话结果。"""

        if message.role != "assistant" or self.model_provider is None:
            return
        conversation = self.store.get_conversation(message.conversation_id)
        if conversation is None or conversation.user_id is None:
            return
        try:
            adapter = self.model_provider(None)
            await self.summarizer.summarize_pending(message.conversation_id, conversation.user_id, adapter)
        except Exception:
            logger.warning(
                "conversation_summary_update_failed conversation_id=%s message_id=%s",
                message.conversation_id,
                message.id,
                exc_info=True,
            )

    def search_history(
        self,
        conversation_id: str,
        user_id: str,
        query: str,
        *,
        exclude_message_ids: set[str] | None = None,
        limit: int = 5,
    ) -> list[Message]:
        if self.store.get_conversation_for_user(conversation_id, user_id) is None:
            return []
        return self.store.search_messages(
            conversation_id,
            query,
            limit=limit,
            exclude_message_ids=exclude_message_ids,
        )

    def resolve_unresolved_topics(
        self,
        conversation_id: str,
        user_id: str,
        *,
        source_run_id: str | None = None,
        source_task_id: str | None = None,
    ) -> ConversationMemory | None:
        """只清除已回答来源的问题，避免一次成功误删其他等待项。"""

        memory = self.get(conversation_id, user_id)
        if memory is None or (source_run_id is None and source_task_id is None):
            return memory
        unresolved = [
            item
            for item in memory.unresolved_topics
            if not (
                (source_run_id is not None and item.source_run_id == source_run_id)
                or (source_run_id is None and source_task_id is not None and item.source_task_id == source_task_id)
            )
        ]
        updated = self._add_entries(memory, unresolved_topics_override=unresolved)
        self.store.save_conversation_memory(updated)
        return updated

    def _add_entries(
        self,
        memory: ConversationMemory,
        *,
        key_facts: Iterable[ConversationMemoryEntry] = (),
        decisions: Iterable[ConversationMemoryEntry] = (),
        important_references: Iterable[ConversationMemoryEntry] = (),
        unresolved_topics: Iterable[ConversationMemoryEntry] = (),
        unresolved_topics_override: list[ConversationMemoryEntry] | None = None,
    ) -> ConversationMemory:
        facts = _append_entries(memory.key_facts, key_facts)
        decisions_list = _append_entries(memory.decisions, decisions)
        references = _append_entries(memory.important_references, important_references)
        unresolved_base = unresolved_topics_override if unresolved_topics_override is not None else memory.unresolved_topics
        unresolved = _append_entries(unresolved_base, unresolved_topics)
        return memory.model_copy(
            update={
                "key_facts": facts[-self.MAX_ENTRIES :],
                "decisions": decisions_list[-self.MAX_ENTRIES :],
                "important_references": references[-self.MAX_ENTRIES :],
                "unresolved_topics": unresolved[-self.MAX_ENTRIES :],
                "updated_at": datetime.now(UTC),
            }
        )


def _append_entries(
    current: Iterable[ConversationMemoryEntry],
    incoming: Iterable[ConversationMemoryEntry],
) -> list[ConversationMemoryEntry]:
    result = list(current)
    for item in incoming:
        if not item.content.strip() or any(_same_entry(existing, item) for existing in result):
            continue
        result.append(item.model_copy(update={"content": _clip(item.content)}))
    return result


def _same_entry(left: ConversationMemoryEntry, right: ConversationMemoryEntry) -> bool:
    if left.reference_type and right.reference_type and left.reference_type == right.reference_type and left.reference_id == right.reference_id:
        return True
    return (
        left.content.casefold() == right.content.casefold()
        and left.source_message_id == right.source_message_id
        and left.source_run_id == right.source_run_id
    )


def _clip(value: str, limit: int = ConversationMemoryService.MAX_CONTENT_LENGTH) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit]


__all__ = ["ConversationMemoryService"]
