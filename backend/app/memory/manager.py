"""Working / Project Memory 的轻量持久化门面。

Memory 与当前 Run 的 state 分开保存。离线模式不做未经确认的自动写入，
但会按当前问题对已有记忆做有限召回，避免把所有历史 key/value 无差别放进
模型上下文。
"""

from __future__ import annotations

import re

from app.core.models import MemoryItem
from app.state import StateStore

from .models import MemoryCandidate
from .policy import MemoryWritePolicy


class MemoryManager:
    def __init__(self, store: StateStore, policy: MemoryWritePolicy | None = None) -> None:
        self.store = store
        self.policy = policy or MemoryWritePolicy()

    def set(self, key: str, value: str, *, scope: str = "project", metadata: dict | None = None, user_id: str | None = None) -> MemoryItem:
        item = MemoryItem(owner_user_id=user_id, scope=scope, key=key, value=value, metadata=metadata or {})
        self.store.save_memory(item)
        return item

    def write_candidate(self, candidate: MemoryCandidate) -> MemoryItem | None:
        """经过 ProjectMemory 策略后写入；显式 API 仍使用 set。"""

        if not self.policy.accepts(candidate):
            return None
        existing = self.get(candidate.key, scope="project", user_id=candidate.owner_user_id)
        metadata = {
            **candidate.metadata,
            "category": candidate.category,
            "confidence": candidate.confidence,
            "importance": candidate.importance,
            "durability": candidate.durability,
            "source_task_id": candidate.source_task_id,
            "source_run_id": candidate.source_run_id,
        }
        if existing is not None and existing.value == candidate.value and existing.metadata == metadata:
            return existing
        item = MemoryItem(
            id=existing.id if existing is not None else MemoryItem(key=candidate.key, value=candidate.value).id,
            owner_user_id=candidate.owner_user_id,
            scope="project",
            key=candidate.key,
            value=candidate.value,
            metadata=metadata,
        )
        self.store.save_memory(item)
        return item

    def write_candidates(self, candidates: list[MemoryCandidate]) -> list[MemoryItem]:
        written: list[MemoryItem] = []
        for candidate in candidates:
            item = self.write_candidate(candidate)
            if item is not None:
                written.append(item)
        return written

    def get(self, key: str, *, scope: str = "project", user_id: str | None = None) -> MemoryItem | None:
        return next((item for item in self.store.list_memories(scope, owner_user_id=user_id) if item.key == key), None)

    def list(self, scope: str = "project", *, user_id: str | None = None) -> list[MemoryItem]:
        return self.store.list_memories(scope, owner_user_id=user_id)

    def recall(self, query: str, *, scope: str = "project", user_id: str | None = None, limit: int = 5) -> list[MemoryItem]:
        """按简单词项重合召回相关记忆；没有命中时不返回全部记忆。"""

        if limit < 1:
            return []
        query_terms = _terms(query)
        scored: list[tuple[int, MemoryItem]] = []
        for item in self.list(scope, user_id=user_id):
            terms = _terms(f"{item.key} {item.value} {item.metadata}")
            score = len(query_terms.intersection(terms))
            if item.key.casefold() in query.casefold():
                score += 3
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].updated_at.timestamp()))
        return [item for _, item in scored[:limit]]


def _terms(value: str) -> set[str]:
    normalized = value.casefold()
    tokens = set(re.findall(r"[a-z0-9_:.+-]+", normalized))
    tokens.update(part for token in tuple(tokens) for part in token.split("_") if part)
    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
        tokens.update(sequence)
    return {token for token in tokens if token}


__all__ = ["MemoryManager"]
