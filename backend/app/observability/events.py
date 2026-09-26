"""GeoAgent 可观测事件类型与进程内事件总线。"""

from collections.abc import Awaitable, Callable
from enum import StrEnum

from app.core.models import TraceEvent

EventHandler = Callable[[TraceEvent], Awaitable[None]]


class EventType(StrEnum):
    RUN_CREATED = "RunCreated"
    INTENT_RESOLVED = "IntentResolved"
    DECISION_MADE = "DecisionMade"
    TOKEN_USAGE_UPDATED = "TokenUsageUpdated"
    MODEL_RESPONSE_STARTED = "ModelResponseStarted"
    SUBTASK_CREATED = "SubTaskCreated"
    SUBAGENT_SPAWNED = "SubAgentSpawned"
    TOOL_STARTED = "ToolStarted"
    TOOL_COMPLETED = "ToolCompleted"
    TOOL_FAILED = "ToolFailed"
    RETRY_STARTED = "RetryStarted"
    REPAIR_SELECTED = "RepairSelected"
    REPLAN_STARTED = "ReplanStarted"
    DATASET_CREATED = "DatasetCreated"
    ARTIFACT_CREATED = "ArtifactCreated"
    VERIFICATION_STARTED = "VerificationStarted"
    VERIFICATION_FAILED = "VerificationFailed"
    SUBAGENT_COMPLETED = "SubAgentCompleted"
    DELEGATION_COMPLETED = "DelegationCompleted"
    CHECKPOINT_SAVED = "CheckpointSaved"
    RESUME_STARTED = "ResumeStarted"
    RUN_COMPLETED = "RunCompleted"
    RUN_FAILED = "RunFailed"
    RUN_CANCELLED = "RunCancelled"
    RUN_WAITING_USER = "RunWaitingUser"
    RUN_WAITING_APPROVAL = "RunWaitingApproval"
    APPROVAL_REQUESTED = "ApprovalRequested"
    APPROVAL_GRANTED = "ApprovalGranted"
    APPROVAL_DENIED = "ApprovalDenied"
    APPROVAL_CONSUMED = "ApprovalConsumed"


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        self._handlers.remove(handler)

    async def publish(self, event: TraceEvent) -> None:
        for handler in tuple(self._handlers):
            await handler(event)


__all__ = ["EventBus", "EventType"]
