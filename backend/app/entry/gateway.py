"""把用户消息与明确的 Run 回复事件交给同一 Agent Loop。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum

from pydantic import BaseModel

from app.core.models import AgentRequest, AgentResult, AgentResultStatus, Run
from app.entry.conversation_service import ConversationService
from app.state import StateStore


class MessageRoute(StrEnum):
    EXECUTION = "execution"
    WAITING = "waiting"
    BLOCKED = "blocked"


class MessageResponse(BaseModel):
    request_id: str
    route: MessageRoute
    message: str = ""
    run: Run | None = None
    result: AgentResult | None = None


class MessageGateway:
    """普通自然语言不做语义预分类；显式 Run 回复才绑定既有运行。"""

    def __init__(self, store: StateStore, conversations: ConversationService) -> None:
        self.store = store
        self.conversations = conversations

    async def submit(
        self,
        request: AgentRequest,
        *,
        on_run: Callable[[Run], Awaitable[None]] | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> MessageResponse:
        self._validate_dataset_access(request)
        if request.reply_to_run_id:
            source = self.store.get_run(request.reply_to_run_id)
            if source is None or source.conversation_id != request.conversation_id:
                raise LookupError("指定的等待运行不存在或不属于当前对话")
            if request.user_id and not self.store.run_belongs_to_user(source.id, request.user_id):
                raise PermissionError("当前用户无权继续该运行")
            if source.status.value != "WAITING_USER":
                raise ValueError("指定运行当前不在等待用户回复状态")
            run, result = await self.conversations.continue_waiting_run(
                request,
                source.id,
                on_run=on_run,
                on_model_delta=on_model_delta,
            )
        else:
            run = await self.conversations.submit(
                request,
                on_run=on_run,
                on_model_delta=on_model_delta,
            )
            result = await self.conversations.wait(run.id)

        final_run = self.store.get_run(run.id) or run
        if result.status is AgentResultStatus.BLOCKED and result.error == "WAITING_USER":
            route = MessageRoute.WAITING
        elif result.status is AgentResultStatus.BLOCKED:
            route = MessageRoute.BLOCKED
        else:
            route = MessageRoute.EXECUTION
        return MessageResponse(
            request_id=request.request_id,
            route=route,
            message=result.summary,
            run=final_run,
            result=result,
        )

    def _validate_dataset_access(self, request: AgentRequest) -> None:
        for dataset_id in dict.fromkeys([*request.dataset_ids, *request.attachment_ids]):
            dataset = (
                self.store.get_dataset_for_user(dataset_id, request.user_id)
                if request.user_id
                else self.store.get_dataset(dataset_id)
            )
            if dataset is None:
                raise PermissionError("当前用户无权访问指定数据集")


__all__ = ["MessageGateway", "MessageResponse", "MessageRoute"]
