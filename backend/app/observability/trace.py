"""把运行事件同时写入 SQLite 和实时事件总线。"""

from __future__ import annotations

from typing import Any

from app.core.models import TraceEvent
from app.observability.events import EventBus, EventType
from app.state import StateStore


class TraceRecorder:
    def __init__(self, store: StateStore, bus: EventBus | None = None, metrics=None) -> None:
        self.store = store
        self.bus = bus or EventBus()
        self.metrics = metrics

    async def emit(
        self,
        run_id: str,
        event_type: EventType | str,
        message: str = "",
        *,
        payload: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=run_id,
            event_type=str(event_type),
            message=message,
            payload=payload or {},
            agent_id=agent_id,
        )
        event = self.store.record_event(event)
        if self.metrics:
            self.metrics.increment("trace_events")
            self.metrics.increment(f"trace_events.{event.event_type}")
        await self.bus.publish(event)
        return event
