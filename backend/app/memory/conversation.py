"""会话原始消息、结构化记忆及增量摘要的统一入口。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from app.core.models import (
    AgentRequest,
    ConversationMemory,
    Message,
)
from app.models import ModelAdapter
from app.state import StateStore

from .conversation_summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class ConversationMemoryService:
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
        self._summary_tasks: dict[str, asyncio.Task[None]] = {}
        self._summary_requests: dict[str, Message] = {}

    async def save_message(
        self,
        message: Message,
        *,
        user_id: str | None,
        wait_for_summary: bool = True,
    ) -> None:
        if not self._can_access(message.conversation_id, user_id):
            raise PermissionError("当前用户无权访问该会话")
        self.store.save_message(message)
        if wait_for_summary:
            await self._summarize_after_assistant_persisted(message)
        else:
            self._schedule_summary(message)

    def _schedule_summary(self, message: Message) -> None:
        if message.role != "assistant" or self.model_provider is None:
            return
        conversation_id = message.conversation_id
        self._summary_requests[conversation_id] = message
        current = self._summary_tasks.get(conversation_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(self._run_scheduled_summaries(conversation_id))
        self._summary_tasks[conversation_id] = task
        task.add_done_callback(
            lambda completed, key=conversation_id: self._forget_summary_task(key, completed)
        )

    async def _run_scheduled_summaries(self, conversation_id: str) -> None:
        while message := self._summary_requests.pop(conversation_id, None):
            await self._summarize_after_assistant_persisted(message)

    def _forget_summary_task(self, conversation_id: str, task: asyncio.Task[None]) -> None:
        if self._summary_tasks.get(conversation_id) is task:
            self._summary_tasks.pop(conversation_id, None)

    async def wait_for_pending_summaries(self) -> None:
        while tasks := tuple(self._summary_tasks.values()):
            await asyncio.gather(*tasks)

    async def close(self) -> None:
        tasks = tuple(self._summary_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._summary_requests.clear()

    def list_messages(self, conversation_id: str, *, user_id: str | None, limit: int = 100) -> list[Message]:
        if not self._can_access(conversation_id, user_id):
            return []
        return self.store.list_messages(conversation_id, limit=limit)

    def load_context(
        self,
        conversation_id: str,
        *,
        user_id: str | None,
        recent_message_limit: int,
        include_history: bool = True,
    ) -> tuple[ConversationMemory | None, list[Message]]:
        """按同一个摘要快照读取覆盖边界之后的原文；恢复协议不受摘要过滤。"""

        if not self._can_access(conversation_id, user_id):
            return None, []
        memory = self.get(conversation_id, user_id) if user_id else None
        if not include_history:
            return memory, []
        through_message_id = memory.summarized_through_message_id if memory and memory.summary else None
        history = (
            self.store.list_messages_after(conversation_id, through_message_id)
            if through_message_id is not None
            else self.store.list_messages(conversation_id, limit=recent_message_limit)
        )
        return memory, history

    def latest_user_message_id(self, conversation_id: str, *, user_id: str | None) -> str | None:
        for message in reversed(self.list_messages(conversation_id, user_id=user_id, limit=100)):
            if message.role == "user":
                return message.id
        return None

    def _can_access(self, conversation_id: str, user_id: str | None) -> bool:
        conversation = self.store.get_conversation(conversation_id)
        return conversation is not None and (user_id is None or conversation.user_id in {None, user_id})

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

    async def _summarize_after_assistant_persisted(self, message: Message) -> None:
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

    async def compact_history_before(self, request: AgentRequest, adapter: ModelAdapter, message_id: str) -> bool:
        """仅在当前会话超限时强制推进摘要；失败不改变原始消息或已有摘要。"""

        if not request.user_id or not self._can_access(request.conversation_id, request.user_id):
            return False
        before = self.get(request.conversation_id, request.user_id)
        try:
            return await self.summarizer.summarize_all_pending(
                request.conversation_id, request.user_id, adapter, protected_message_id=message_id
            )
        except Exception:
            logger.warning("conversation_emergency_summary_failed conversation_id=%s", request.conversation_id, exc_info=True)
            after = self.get(request.conversation_id, request.user_id)
            return after is not None and after.summary_version != (before.summary_version if before else 0)

    def search_history(
        self,
        conversation_id: str,
        user_id: str | None,
        query: str,
        *,
        exclude_message_ids: set[str] | None = None,
        limit: int = 5,
    ) -> list[Message]:
        if user_id:
            if self.store.get_conversation_for_user(conversation_id, user_id) is None:
                return []
        elif not self._can_access(conversation_id, user_id):
            return []
        return self.store.search_messages(
            conversation_id,
            query,
            limit=limit,
            exclude_message_ids=exclude_message_ids,
        )

__all__ = ["ConversationMemoryService"]
