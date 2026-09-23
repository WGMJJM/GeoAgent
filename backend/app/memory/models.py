"""ProjectMemory 的候选模型。"""

from typing import Any

from pydantic import Field

from app.core.models import MemoryItem, StrictModel


class MemoryCandidate(StrictModel):
    """待判断的长期记忆候选，不代表已经写入 ProjectMemory。"""

    key: str
    value: str
    owner_user_id: str | None = None
    category: str = "project_fact"
    source_task_id: str | None = None
    source_run_id: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    importance: float = Field(default=0.5, ge=0, le=1)
    durability: str = "project"
    metadata: dict[str, Any] = Field(default_factory=dict)

__all__ = ["MemoryCandidate", "MemoryItem"]
