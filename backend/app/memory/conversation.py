"""会话原始消息、结构化记忆及增量摘要的统一入口。"""

from __future__ import annotations

import logging

from app.core.models import AgentRequest, ConversationMemory, Message
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
    ) -> None:
        self.store = store
        self.summarizer = summarizer or ConversationSummarizer(store)

    async def save_message(
        self,
        message: Message,
        *,
        user_id: str | None,
    ) -> None:
        if not self._can_access(message.conversation_id, user_id):
            raise PermissionError("当前用户无权访问该会话")
        self.store.save_message(message)

    def list_messages(self, conversation_id: str, *, user_id: str | None, limit: int = 100) -> list[Message]:
        if not self._can_access(conversation_id, user_id):
            return []
        return self.store.list_messages(conversation_id, limit=limit)

    def load_context(
        self,
        conversation_id: str,
        *,
        user_id: str | None,
        include_history: bool = True,
    ) -> tuple[ConversationMemory | None, list[Message]]:
        """按 SQLite 顺序读取摘要游标之后的全部原文；不以固定消息窗口漏掉历史。"""

        if not self._can_access(conversation_id, user_id):
            return None, []
        memory = self.get(conversation_id, user_id) if user_id else None
        if not include_history:
            return memory, []
        through_message_id = memory.summarized_through_message_id if memory and memory.summary else None
        if through_message_id is not None and not self.store.message_exists(conversation_id, through_message_id):
            logger.warning(
                "conversation_summary_cursor_missing conversation_id=%s cursor=%s; loading all source messages",
                conversation_id,
                through_message_id,
            )
            memory = memory.model_copy(update={"summary": ""})
            through_message_id = None
        try:
            history = self.store.list_messages_after(conversation_id, through_message_id)
        except ValueError:
            logger.warning(
                "conversation_summary_cursor_invalid conversation_id=%s cursor=%s; loading all source messages",
                conversation_id,
                through_message_id,
            )
            memory = memory.model_copy(update={"summary": ""}) if memory is not None else None
            history = self.store.list_messages_after(conversation_id, None)
        return memory, history

    def estimated_history_tokens(self, conversation_id: str, *, user_id: str | None) -> int:
        if not self._can_access(conversation_id, user_id):
            return 0
        memory = self.get(conversation_id, user_id) if user_id else None
        cursor = memory.summarized_through_message_id if memory and memory.summary else None
        if cursor is not None and not self.store.message_exists(conversation_id, cursor):
            cursor = None
        return self.store.message_token_estimate_total_after(conversation_id, cursor)

    def latest_user_message_id(self, conversation_id: str, *, user_id: str | None) -> str | None:
        if not self._can_access(conversation_id, user_id):
            return None
        return self.store.latest_user_message_id(conversation_id)

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

    async def compact_history_before(
        self,
        request: AgentRequest,
        adapter: ModelAdapter,
        message_id: str,
        *,
        recent_tokens: int,
        all_pending: bool = False,
    ) -> bool:
        """按安全完整交互推进一个摘要批次；硬预算路径可连续处理全部旧前缀。"""

        if not request.user_id or not self._can_access(request.conversation_id, request.user_id):
            return False
        try:
            if all_pending:
                return await self.summarizer.summarize_all_pending(
                    request.conversation_id,
                    request.user_id,
                    adapter,
                    protected_message_id=message_id,
                    recent_tokens=recent_tokens,
                )
            return await self.summarizer.summarize_pending(
                request.conversation_id,
                request.user_id,
                adapter,
                protected_message_id=message_id,
                recent_tokens=recent_tokens,
            )
        except Exception:
            logger.warning("conversation_emergency_summary_failed conversation_id=%s", request.conversation_id, exc_info=True)
            return False

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
