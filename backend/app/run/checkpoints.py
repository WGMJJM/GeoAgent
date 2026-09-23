"""Run 的轻量检查点读写与版本边界。"""

from __future__ import annotations

from typing import Any

from app.core.models import AgentRequest, AgentResult, Checkpoint
from app.state import StateStore


class CheckpointStore:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        self.store.save_checkpoint(checkpoint)
        return checkpoint

    def latest(self, run_id: str) -> Checkpoint | None:
        return self.store.latest_checkpoint(run_id)


class RunCheckpointCodec:
    CURRENT_VERSION = 1

    @classmethod
    def normalize_state(cls, state: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(state or {})
        version = payload.get("schema_version", 0)
        if not isinstance(version, int) or isinstance(version, bool) or version < 0 or version > cls.CURRENT_VERSION:
            raise ValueError("不支持的 Run Checkpoint 版本。")
        payload["schema_version"] = version
        return payload

    @classmethod
    def request(cls, state: dict[str, Any] | None) -> AgentRequest | None:
        payload = cls.normalize_state(state).get("request")
        return AgentRequest.model_validate(payload) if isinstance(payload, dict) else None

    @classmethod
    def result(cls, state: dict[str, Any] | None) -> AgentResult | None:
        payload = cls.normalize_state(state).get("result")
        return AgentResult.model_validate(payload) if isinstance(payload, dict) else None


__all__ = ["CheckpointStore", "RunCheckpointCodec"]
