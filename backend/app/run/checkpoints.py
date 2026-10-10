"""Run 检查点的版本校验与请求、结果解码。"""

from __future__ import annotations

from typing import Any

from app.core.models import AgentRequest, AgentResult


class RunCheckpointCodec:
    RECOVERY_VERSION = 2
    CURRENT_VERSION = RECOVERY_VERSION

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


__all__ = ["RunCheckpointCodec"]
