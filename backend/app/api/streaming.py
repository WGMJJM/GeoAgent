"""将单次 Agent 请求转换为 SSE；连接断开时清理仍在执行的任务。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress

import anyio
from fastapi.responses import StreamingResponse
from starlette.types import Send

from app.application import Application
from app.core.models import AgentRequest, Run, TokenUsage

HEARTBEAT_SECONDS = 15
logger = logging.getLogger(__name__)


class MessageStreamResponse(StreamingResponse):
    async def stream_response(self, send: Send) -> None:
        # 包括发送失败时，也必须关闭生成器，执行其取消和退订逻辑。
        async with aclosing(self.body_iterator):
            await super().stream_response(send)


def message_stream(application: Application, request: AgentRequest) -> StreamingResponse:
    return MessageStreamResponse(
        _events(application, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


async def _events(application: Application, request: AgentRequest) -> AsyncIterator[str]:
    queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
    stream_run: Run | None = None

    async def on_run(run: Run) -> None:
        nonlocal stream_run
        stream_run = run
        queue.put_nowait(("run", run.model_dump(mode="json")))

    async def on_event(event) -> None:
        if stream_run is not None and event.run_id == stream_run.id:
            queue.put_nowait(("event", event.model_dump(mode="json")))

    async def on_model_delta(content: str, token_usage: TokenUsage) -> None:
        queue.put_nowait(("delta" if content else "usage", {
            "content": content, "token_usage": token_usage.model_dump(mode="json"),
        }))

    async def submit() -> None:
        try:
            response = await application.message_entry.submit(
                request, on_run=on_run, on_model_delta=on_model_delta,
            )
            queue.put_nowait(("response", response.model_dump(mode="json")))
        except (PermissionError, LookupError, ValueError) as exc:
            queue.put_nowait(("error", {"message": str(exc)}))
        except Exception:
            logger.exception("SSE request failed: %s", request.request_id)
            queue.put_nowait(("error", {"message": "实时请求失败，请稍后重试"}))

    application.bus.subscribe(on_event)
    producer = asyncio.create_task(submit())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), HEARTBEAT_SECONDS)
            except TimeoutError:
                kind, payload = "heartbeat", {}
            yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if kind in {"response", "error"}:
                return
    finally:
        application.bus.unsubscribe(on_event)
        # Starlette 会取消流式响应所在的作用域；清理仍需完成并写回取消状态。
        with anyio.CancelScope(shield=True):
            if stream_run is not None and application.run_manager.is_active(stream_run.id):
                await application.run_manager.cancel(stream_run.id)
            if not producer.done():
                producer.cancel()
            with suppress(asyncio.CancelledError):
                await producer
