"""Conversation 与 Main Agent 的业务入口。"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from app.core.models import (
    AgentRequest,
    AgentResult,
    Conversation,
    Message,
    Run,
    TokenUsage,
    new_id,
)
from app.memory import ConversationMemoryService
from app.run import RunManager
from app.run.predicates import is_cancellable_run
from app.state import StateStore

DEFAULT_CONVERSATION_TITLE = "新对话"
CONVERSATION_TITLE_CONTENT_LIMIT = 24


def derive_conversation_title(text: str, *, limit: int = CONVERSATION_TITLE_CONTENT_LIMIT) -> str:
    """从第一条用户消息生成稳定标题，不调用模型。"""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return DEFAULT_CONVERSATION_TITLE
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "…"


class ConversationService:
    def __init__(self, store: StateStore, run_manager: RunManager, *, memory: ConversationMemoryService) -> None:
        self.store = store
        self.run_manager = run_manager
        self.memory = memory

    def ensure(self, conversation_id: str, title: str, *, user_id: str | None = None) -> None:
        existing = self.store.get_conversation(conversation_id)
        if existing is not None and user_id is not None and existing.user_id not in {None, user_id}:
            raise PermissionError("当前用户无权访问该对话")
        if existing is None:
            self.store.upsert_conversation(conversation_id, title or DEFAULT_CONVERSATION_TITLE, datetime.now(UTC).isoformat(), user_id)

    def create(self, title: str = "新对话", *, user_id: str | None = None) -> Conversation:
        return self.store.create_conversation(title, user_id=user_id)

    def list(self, limit: int = 50, *, user_id: str | None = None) -> list[Conversation]:
        return self.store.list_conversations(limit, user_id=user_id)

    async def delete(self, conversation_id: str, *, user_id: str | None = None) -> bool:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None or (user_id is not None and conversation.user_id not in {None, user_id}):
            return False
        for run in self.store.list_runs_for_conversation(conversation_id):
            current = self.store.get_run(run.id)
            if current is not None and is_cancellable_run(current):
                await self.run_manager.cancel(current.id)
        return self.store.delete_conversation(conversation_id, user_id=user_id)

    async def submit(
        self,
        request: AgentRequest,
        *,
        on_run: Callable[[Run], Awaitable[None]] | None = None,
        on_model_delta: Callable[[str, TokenUsage], Awaitable[None]] | None = None,
    ) -> Run:
        await self._save_user_message(request)
        return await self.run_manager.submit(
            request,
            on_run=on_run,
            on_model_delta=on_model_delta,
        )

    async def continue_waiting_run(
        self,
        request: AgentRequest,
        run_id: str,
        *,
        on_run: Callable[[Run], Awaitable[None]] | None = None,
        on_model_delta: Callable[[str, TokenUsage], Awaitable[None]] | None = None,
    ) -> tuple[Run, AgentResult]:
        await self._save_user_message(request)
        run = await self.run_manager.continue_run(
            run_id,
            user_input=request.user_input,
            user_id=request.user_id,
            dataset_ids=request.dataset_ids,
            attachment_ids=request.attachment_ids,
            model_profile=request.model_profile,
            reasoning_effort=request.reasoning_effort,
            on_run=on_run,
            on_model_delta=on_model_delta,
        )
        return run, await self.wait(run.id, force_assistant=True)

    async def wait(self, run_id: str, *, force_assistant: bool = False) -> AgentResult:
        result = await self.run_manager.wait(run_id)
        run = self.store.get_run(run_id)
        if run and run.conversation_id and not run.parent_run_id:
            conversation = self.store.get_conversation(run.conversation_id)
            user_id = conversation.user_id if conversation else None
            messages = self.memory.list_messages(run.conversation_id, user_id=user_id, limit=1000)
            if force_assistant or not any(message.role == "assistant" and message.run_id == result.trace_id for message in messages):
                self.ensure(run.conversation_id, "GeoAgent resumed run", user_id=user_id)
                await self.memory.save_message(
                    Message(
                        id=new_id("msg"),
                        conversation_id=run.conversation_id,
                        role="assistant",
                        content=result.summary,
                        run_id=result.trace_id,
                    ),
                    user_id=user_id,
                    wait_for_summary=False,
                )
        return result

    async def _save_user_message(self, request: AgentRequest) -> None:
        self.ensure(request.conversation_id, DEFAULT_CONVERSATION_TITLE, user_id=request.user_id)
        self.set_title_if_default(request.conversation_id, request.user_input, user_id=request.user_id)
        dataset_ids = list(dict.fromkeys([*request.dataset_ids, *request.attachment_ids]))
        await self.memory.save_message(
            Message(
                id=new_id("msg"),
                conversation_id=request.conversation_id,
                role="user",
                content=request.user_input,
                dataset_ids=dataset_ids,
            ),
            user_id=request.user_id,
        )

    def set_title_if_default(self, conversation_id: str, first_user_message: str, *, user_id: str | None = None) -> None:
        """只在默认标题阶段写入首条用户消息标题，后续消息不改名。"""

        conversation = self.store.get_conversation(conversation_id)
        if conversation is not None and conversation.title == DEFAULT_CONVERSATION_TITLE:
            self.store.upsert_conversation(
                conversation_id,
                derive_conversation_title(first_user_message),
                datetime.now(UTC).isoformat(),
                user_id,
            )
