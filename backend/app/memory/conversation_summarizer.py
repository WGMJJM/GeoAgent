"""基于原始消息增量更新会话摘要。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.core.models import ConversationMemory, ConversationMemoryEntry, Message
from app.core.tokens import estimate_tokens
from app.models import ModelAdapter, ModelRequest
from app.state import StateStore

logger = logging.getLogger(__name__)

SUMMARY_TRIGGER_MESSAGES = 8
SUMMARY_TRIGGER_TOKENS = 3000
SUMMARY_RECENT_MESSAGES = 8
SUMMARY_MAX_BATCH_MESSAGES = 32
SUMMARY_MAX_BATCH_TOKENS = 6000
SUMMARY_TIMEOUT_SECONDS = 20
SUMMARY_MAX_LENGTH = 1800
SUMMARY_MAX_ENTRY_LENGTH = 500

_SYSTEM_PROMPT = """
你负责把会话历史压缩为可继续使用的滚动摘要，并提取关键事实、用户决定、待解决问题及重要资源引用。
输入同时包含旧摘要、既有结构化记忆、新消息及经过数据库核验的资源和运行状态。

规则：
- 只根据旧摘要和新消息总结；新摘要应吸收旧摘要并更新，而不是简单拼接。
- 每条事实、决定、待解决问题和资源引用必须填写其来源 message_id，且只能使用输入中提供的 ID。
- 不得编造 Dataset、Artifact、Run ID。资源引用只能取自 verified_resources。
- Run 状态只能引用 verified_runs 中的数据库状态；没有核验结果时，不得声称任务已成功、失败或产生结果。
- assistant 消息若关联 run_id，应以 verified_runs 中的状态和结果为准；不要把消息文本中的未经核验表述当作事实。
- 忽略密码、密钥、令牌等敏感凭据；保留有后续价值的用户偏好、决定和未解决问题。
- 只输出 JSON：summary、key_facts、decisions、unresolved_topics、references。
- 列表元素格式：事实类为 {"content": string, "source_message_id": string}；引用类为 {"type": "dataset|artifact|run", "id": string, "source_message_id": string}。
""".strip()


class ConversationSummarizer:
    """对摘要游标之后、且不属于最近原始消息窗口的内容执行一次增量摘要。"""

    def __init__(
        self,
        store: StateStore,
        *,
        trigger_messages: int = SUMMARY_TRIGGER_MESSAGES,
        trigger_tokens: int = SUMMARY_TRIGGER_TOKENS,
        recent_messages: int = SUMMARY_RECENT_MESSAGES,
        timeout_seconds: float = SUMMARY_TIMEOUT_SECONDS,
    ) -> None:
        self.store = store
        self.trigger_messages = max(1, trigger_messages)
        self.trigger_tokens = max(1, trigger_tokens)
        self.recent_messages = max(1, recent_messages)
        self.timeout_seconds = max(0.1, timeout_seconds)

    async def summarize_pending(self, conversation_id: str, user_id: str, adapter: ModelAdapter | None) -> bool:
        if adapter is None:
            return False
        memory = self.store.get_conversation_memory_for_user(conversation_id, user_id)
        if memory is None:
            if self.store.get_conversation_for_user(conversation_id, user_id) is None:
                return False
            self.store.save_conversation_memory(ConversationMemory(conversation_id=conversation_id, user_id=user_id))
            memory = self.store.get_conversation_memory_for_user(conversation_id, user_id)
        if memory is None:
            return False

        unprocessed = self.store.list_messages_after(conversation_id, memory.summarized_through_message_id)
        if len(unprocessed) <= self.recent_messages:
            return False
        eligible = unprocessed[:-self.recent_messages]
        if not eligible:
            return False
        pending_tokens = sum(estimate_tokens(item.content) for item in eligible)
        if len(eligible) < self.trigger_messages and pending_tokens < self.trigger_tokens:
            return False

        batch = _bounded_batch(eligible)
        if not batch:
            return False
        verified_runs, verified_resources = self._verified_context(batch, conversation_id, user_id)
        payload = {
            "old_summary": memory.summary,
            "existing_memory": {
                "key_facts": [_entry_view(item) for item in memory.key_facts[-20:]],
                "decisions": [_entry_view(item) for item in memory.decisions[-20:]],
                "unresolved_topics": [_entry_view(item) for item in memory.unresolved_topics[-20:]],
                "important_references": [_entry_view(item) for item in memory.important_references[-20:]],
            },
            "messages": [
                {
                    "message_id": item.id,
                    "role": item.role,
                    "content": item.content[:5000],
                    "dataset_ids": item.dataset_ids,
                    "run_id": item.run_id,
                }
                for item in batch
            ],
            "verified_runs": verified_runs,
            "verified_resources": verified_resources,
        }
        response_format = {"type": "json_object"} if _supports_json_object(adapter) else None
        response = await asyncio.wait_for(
            adapter.complete(
                ModelRequest(
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                    ],
                    temperature=0,
                    max_tokens=1800,
                    response_format=response_format,
                )
            ),
            timeout=self.timeout_seconds,
        )
        parsed = _parse_output(response.content)
        message_ids = {item.id for item in batch}
        verified_ids = {item["id"] for item in verified_runs}
        verified_ids.update(item["id"] for item in verified_resources)
        message_by_id = {item.id: item for item in batch}
        run_by_id = {item["id"]: item for item in verified_runs}
        key_facts = _parse_entries(parsed.get("key_facts"), message_by_id, verified_ids, run_by_id, category="fact")
        decisions = _parse_entries(parsed.get("decisions"), message_by_id, verified_ids, run_by_id, category="decision")
        unresolved = _parse_entries(parsed.get("unresolved_topics"), message_by_id, verified_ids, run_by_id, category="unresolved")
        references = self._parse_references(
            parsed.get("references"),
            message_ids,
            verified_resources,
            user_id,
            conversation_id,
        )
        summary = _clean_summary(parsed.get("summary"), verified_runs, verified_resources)
        if not summary:
            raise ValueError("ConversationSummarizer 输出缺少有效 summary")

        return self.store.commit_conversation_summary(
            conversation_id=conversation_id,
            user_id=user_id,
            expected_version=memory.summary_version,
            expected_through_message_id=memory.summarized_through_message_id,
            through_message_id=batch[-1].id,
            summary=summary,
            key_facts=key_facts,
            decisions=decisions,
            unresolved_topics=unresolved,
            important_references=references,
        )

    def _verified_context(self, messages: list[Message], conversation_id: str, user_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        message_text = "\n".join(item.content for item in messages)
        source_ids = {item.id for item in messages}
        runs = self.store.list_runs_for_conversation(conversation_id, limit=1000)
        source_runs = {
            run.id: run
            for run in runs
            if run.id in {item.run_id for item in messages if item.run_id}
            or run.id in message_text
        }
        verified_runs = [
            {
                "id": run.id,
                "status": run.status.value,
                "error": run.error,
                "result_summary": _stored_result_summary(run),
                "source_message_ids": [item.id for item in messages if item.run_id == run.id],
            }
            for run in source_runs.values()
            if self.store.run_belongs_to_user(run.id, user_id)
        ]

        datasets = self.store.list_datasets_for_user(user_id)
        artifacts = self.store.list_artifacts_for_user(user_id)
        candidate_resources: dict[tuple[str, str], set[str]] = {}
        for message in messages:
            named_ids = set(message.dataset_ids)
            for dataset in datasets:
                if dataset.id in named_ids or dataset.id in message.content or dataset.name in message.content:
                    candidate_resources.setdefault(("dataset", dataset.id), set()).add(message.id)
            for artifact in artifacts:
                if artifact.id in message.content or artifact.name in message.content:
                    candidate_resources.setdefault(("artifact", artifact.id), set()).add(message.id)
            for run in source_runs.values():
                if run.id in message.content or message.run_id == run.id:
                    candidate_resources.setdefault(("run", run.id), set()).add(message.id)
                    result = run.metadata.get("result") if isinstance(run.metadata, dict) else None
                    if isinstance(result, dict):
                        for dataset_id in result.get("datasets", []):
                            if isinstance(dataset_id, str) and any(item.id == dataset_id for item in datasets):
                                candidate_resources.setdefault(("dataset", dataset_id), set()).add(message.id)
                        for artifact_id in result.get("artifacts", []):
                            if isinstance(artifact_id, str) and any(item.id == artifact_id for item in artifacts):
                                candidate_resources.setdefault(("artifact", artifact_id), set()).add(message.id)

        verified_resources = [
            {"type": kind, "id": identifier, "source_message_ids": sorted(owners)}
            for (kind, identifier), owners in candidate_resources.items()
            if owners <= source_ids
        ]
        return verified_runs, verified_resources

    def _parse_references(
        self,
        values: Any,
        message_ids: set[str],
        verified_resources: list[dict[str, Any]],
        user_id: str,
        conversation_id: str,
    ) -> list[ConversationMemoryEntry]:
        if not isinstance(values, list):
            return []
        candidates = {
            (item["type"], item["id"]): set(item["source_message_ids"])
            for item in verified_resources
        }
        result: list[ConversationMemoryEntry] = []
        for item in values[:40]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", "")).casefold()
            identifier = str(item.get("id", ""))
            source_message_id = str(item.get("source_message_id", ""))
            if source_message_id not in message_ids or source_message_id not in candidates.get((kind, identifier), set()):
                continue
            if kind == "dataset":
                resource = self.store.get_dataset_for_user(identifier, user_id)
                if resource is None:
                    continue
                content = f"数据集：{resource.name}（{resource.kind.value}）"
                source_run_id = self._verified_source_run(resource.created_by_run_id, user_id, conversation_id)
            elif kind == "artifact":
                resource = self.store.get_artifact_for_user(identifier, user_id)
                if resource is None:
                    continue
                content = f"结果文件：{resource.name}"
                source_run_id = self._verified_source_run(resource.run_id, user_id, conversation_id)
            elif kind == "run":
                resource = self.store.get_run(identifier)
                if resource is None or resource.conversation_id != conversation_id or not self.store.run_belongs_to_user(identifier, user_id):
                    continue
                content = f"运行状态：{resource.status.value}"
                source_run_id = resource.id
            else:
                continue
            result.append(
                ConversationMemoryEntry(
                    content=content,
                    source_message_id=source_message_id,
                    source_run_id=source_run_id,
                    reference_type=kind,
                    reference_id=identifier,
                )
            )
        return result

    def _verified_source_run(self, run_id: str | None, user_id: str, conversation_id: str) -> str | None:
        if not run_id:
            return None
        run = self.store.get_run(run_id)
        if run is None or run.conversation_id != conversation_id or not self.store.run_belongs_to_user(run_id, user_id):
            return None
        return run.id


def _parse_output(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE | re.DOTALL)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("ConversationSummarizer 输出必须为 JSON 对象")
    return parsed


def _parse_entries(
    values: Any,
    messages: dict[str, Message],
    verified_ids: set[str],
    verified_runs: dict[str, dict[str, Any]],
    *,
    category: str,
) -> list[ConversationMemoryEntry]:
    if not isinstance(values, list):
        return []
    result = []
    for item in values[:40]:
        if not isinstance(item, dict):
            continue
        content = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()
        source_message_id = str(item.get("source_message_id", ""))
        source_message = messages.get(source_message_id)
        if not content or source_message is None or not _contains_only_verified_ids(content, verified_ids):
            continue
        source_run_id = source_message.run_id
        if source_run_id:
            verified_run = verified_runs.get(source_run_id)
            if category != "unresolved" or verified_run is None or verified_run["status"] not in {"WAITING_USER", "WAITING_APPROVAL"}:
                continue
        result.append(
            ConversationMemoryEntry(
                content=content[:SUMMARY_MAX_ENTRY_LENGTH],
                source_message_id=source_message_id,
                source_run_id=source_run_id,
            )
        )
    return result


def _contains_only_verified_ids(content: str, verified_ids: set[str]) -> bool:
    identifiers = re.findall(r"\b(?:ds|dataset|art|artifact|run)[_-][A-Za-z0-9_-]+\b", content, flags=re.IGNORECASE)
    return all(identifier in verified_ids for identifier in identifiers)


def _clean_summary(value: Any, verified_runs: list[dict[str, Any]], verified_resources: list[dict[str, Any]]) -> str:
    summary = re.sub(r"\s+", " ", str(value or "")).strip()
    allowed_ids = {item["id"] for item in verified_runs}
    allowed_ids.update(item["id"] for item in verified_resources)
    for identifier in re.findall(r"\b(?:ds|dataset|art|artifact|run)[_-][A-Za-z0-9_-]+\b", summary, flags=re.IGNORECASE):
        if identifier not in allowed_ids:
            return ""
    return summary[:SUMMARY_MAX_LENGTH]


def _stored_result_summary(run) -> str | None:
    value = run.metadata.get("result") if isinstance(run.metadata, dict) else None
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    return str(summary)[:500] if isinstance(summary, str) and summary else None


def _entry_view(entry: ConversationMemoryEntry) -> dict[str, str | None]:
    return {
        "content": entry.content,
        "source_message_id": entry.source_message_id,
        "source_run_id": entry.source_run_id,
        "reference_type": entry.reference_type,
        "reference_id": entry.reference_id,
    }


def _bounded_batch(messages: list[Message]) -> list[Message]:
    batch: list[Message] = []
    tokens = 0
    for message in messages[:SUMMARY_MAX_BATCH_MESSAGES]:
        estimate = estimate_tokens(message.content[:5000])
        if batch and tokens + estimate > SUMMARY_MAX_BATCH_TOKENS:
            break
        batch.append(message)
        tokens += estimate
    return batch


def _supports_json_object(adapter: ModelAdapter) -> bool:
    return bool(
        getattr(adapter, "supports_json_object", False)
        or getattr(adapter, "supports_json_schema", False)
        or getattr(adapter, "supports_structured_output", False)
    )


__all__ = ["ConversationSummarizer"]
