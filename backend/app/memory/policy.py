"""ProjectMemory 的确定性写入边界。"""

from __future__ import annotations

from .models import MemoryCandidate


class MemoryWritePolicy:
    """只允许稳定、明确且具有项目级持久价值的候选进入长期记忆。"""

    _FORBIDDEN_CATEGORIES = {
        "run_state",
        "runtime",
        "execution",
        "temporary",
        "tool_output",
        "artifact",
        "dataset",
    }
    _FORBIDDEN_KEYS = {
        "last_run_id",
        "last_result_summary",
        "turn_count",
        "tool_call_count",
        "retry_count",
        "run_status",
        "last_error",
    }

    def accepts(self, candidate: MemoryCandidate) -> bool:
        key = candidate.key.strip().casefold()
        category = candidate.category.strip().casefold()
        durability = candidate.durability.strip().casefold()
        if not key or not candidate.value.strip():
            return False
        if key in self._FORBIDDEN_KEYS or category in self._FORBIDDEN_CATEGORIES:
            return False
        if durability not in {"project", "durable", "long_term"}:
            return False
        if candidate.confidence < 0.7 or candidate.importance < 0.3:
            return False
        return True


__all__ = ["MemoryWritePolicy"]
