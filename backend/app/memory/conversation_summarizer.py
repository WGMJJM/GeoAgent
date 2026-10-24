"""基于原始消息增量更新会话摘要。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from time import perf_counter
from typing import Any

from app.core.models import ConversationMemory, ConversationMemoryEntry, Message, Run, RunStatus
from app.models import ModelAdapter, ModelRequest, effective_input_budget
from app.state import StateStore

logger = logging.getLogger(__name__)

SUMMARY_MAX_INPUT_TOKENS = 28672
SUMMARY_TARGET_BATCH_TOKENS = 16384
SUMMARY_MAX_BATCH_TOKENS = 20480
SUMMARY_MAX_OUTPUT_TOKENS = 3200
SUMMARY_TIMEOUT_SECONDS = 120
SUMMARY_MAX_LENGTH = 1800
SUMMARY_MAX_ENTRY_LENGTH = 500

_PROTECTED_RUN_STATUSES = {
    RunStatus.CREATED,
    RunStatus.RUNNING,
    RunStatus.WAITING_TOOL,
    RunStatus.WAITING_SUBAGENT,
    RunStatus.WAITING_USER,
    RunStatus.WAITING_APPROVAL,
    RunStatus.RETRYING,
    RunStatus.INTERRUPTED,
    RunStatus.BUDGET_EXCEEDED,
}

_SYSTEM_PROMPT = """
你负责把会话历史压缩为可继续使用的滚动摘要，并提取关键事实、用户决定、待解决问题及重要资源引用。
输入同时包含旧摘要、既有结构化记忆、新消息及经过数据库核验的资源和运行状态。

规则：
- 只根据旧摘要、既有结构化信息和本批新消息总结；新摘要应吸收并更新旧摘要，而不是简单拼接。
- 四类结构化列表是当前仍有效的完整视图：保留仍成立的旧条目，删除已被用户后续决定替代或已经解决的条目。
- 每条新事实、决定、待解决问题和资源引用必须填写其来源 message_id，且只能使用输入中提供的 ID；原样保留旧条目时使用其原来源 ID。
- 不得编造 Dataset、Artifact、Run ID。资源引用只能取自 verified_resources。
- Run 状态只能引用 verified_runs 中的数据库状态；没有核验结果时，不得声称任务已成功、失败或产生结果。
- assistant 消息若关联 run_id，应以 verified_runs 中的状态和结果为准；不要把消息文本中的未经核验表述当作事实。
- 忽略密码、密钥、令牌等敏感凭据；保留有后续价值的用户偏好、决定和未解决问题。
- 只输出 JSON：summary、key_facts、decisions、unresolved_topics、references。
- 列表元素格式：事实类为 {"content": string, "source_message_id": string}；引用类为 {"type": "dataset|artifact|run", "id": string, "source_message_id": string}。
""".strip()


class ConversationSummarizer:
    """只摘要安全切点之前的完整历史交互；摘要游标始终连续前进。"""

    def __init__(
        self,
        store: StateStore,
        *,
        summary_recent_tokens: int = 6400,
        model_input_tokens: int = 256000,
        timeout_seconds: float = SUMMARY_TIMEOUT_SECONDS,
    ) -> None:
        self.store = store
        self.summary_recent_tokens = max(1, summary_recent_tokens)
        self.model_input_tokens = max(1, model_input_tokens)
        self.timeout_seconds = max(0.1, timeout_seconds)

    async def summarize_all_pending(
        self,
        conversation_id: str,
        user_id: str,
        adapter: ModelAdapter | None,
        *,
        protected_message_id: str,
        recent_tokens: int,
    ) -> bool:
        """逐批摘要安全旧前缀；每批成功提交后才继续下一批。"""

        changed = False
        while await self.summarize_pending(
            conversation_id,
            user_id,
            adapter,
            protected_message_id=protected_message_id,
            recent_tokens=recent_tokens,
        ):
            changed = True
        return changed

    async def summarize_pending(
        self,
        conversation_id: str,
        user_id: str,
        adapter: ModelAdapter | None,
        *,
        protected_message_id: str | None = None,
        recent_tokens: int | None = None,
    ) -> bool:
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
        if memory.summarized_through_message_id is not None and not self.store.message_exists(
            conversation_id, memory.summarized_through_message_id
        ):
            logger.warning(
                "conversation_summary_stopped_missing_cursor conversation_id=%s cursor=%s",
                conversation_id,
                memory.summarized_through_message_id,
            )
            return False
        unprocessed = self.store.list_messages_after(conversation_id, memory.summarized_through_message_id)
        if not unprocessed or protected_message_id is None:
            return False
        positions = {item.id: index for index, item in enumerate(unprocessed)}
        protected_index = positions.get(protected_message_id)
        if protected_index is None:
            return False
        runs = self.store.list_runs_for_conversation(conversation_id)
        runs_by_id = {run.id: run for run in runs}
        active_start = _earliest_protected_run_start(unprocessed, positions, runs)
        if active_start is None:
            return False
        protected_index = min(protected_index, active_start)
        history = unprocessed[:protected_index]
        segments = _completed_interactions(history, runs_by_id)
        if not segments:
            return False

        keep_tokens = max(1, recent_tokens or self.summary_recent_tokens)
        recent_start = len(segments)
        recent_count = 0
        for index in range(len(segments) - 1, -1, -1):
            recent_count += _messages_tokens(segments[index], adapter.count_tokens)
            recent_start = index
            if recent_count >= keep_tokens:
                break
        if recent_count < keep_tokens or recent_start == 0:
            return False

        batch_segments = _summary_batch_segments(segments[:recent_start], adapter.count_tokens)
        if not batch_segments:
            return False
        batch: list[Message] = []
        payload: dict[str, Any] | None = None
        request_messages: list[dict[str, str]] | None = None
        request_input_tokens = 0
        input_budget = min(
            SUMMARY_MAX_INPUT_TOKENS,
            effective_input_budget(self.model_input_tokens, adapter, SUMMARY_MAX_OUTPUT_TOKENS),
        )
        while batch_segments and input_budget > 0:
            batch = [message for segment in batch_segments for message in segment]
            verified_runs, verified_resources = self._verified_context(batch, conversation_id, user_id)
            payload = {
                "old_summary": memory.summary,
                "existing_memory": {
                    "key_facts": [_entry_view(item) for item in memory.key_facts],
                    "decisions": [_entry_view(item) for item in memory.decisions],
                    "unresolved_topics": [_entry_view(item) for item in memory.unresolved_topics],
                    "important_references": [_entry_view(item) for item in memory.important_references],
                },
                "messages": [
                    {
                        "message_id": item.id,
                        "role": item.role,
                        "content": item.content,
                        "dataset_ids": item.dataset_ids,
                        "run_id": item.run_id,
                    }
                    for item in batch
                ],
                "verified_runs": verified_runs,
                "verified_resources": verified_resources,
            }
            request_messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ]
            request_input_tokens = adapter.count_tokens(
                json.dumps({"messages": request_messages, "tools": []}, ensure_ascii=False, separators=(",", ":"))
            )
            if request_input_tokens <= input_budget:
                break
            batch_segments.pop()
        if not batch_segments or payload is None or request_messages is None:
            logger.info(
                "conversation_summary_skipped conversation_id=%s cursor=%s reason=summary_input_budget",
                conversation_id,
                memory.summarized_through_message_id,
            )
            return False

        response_format = {"type": "json_object"} if _supports_json_object(adapter) else None
        started = perf_counter()
        response = await asyncio.wait_for(
            adapter.complete(
                ModelRequest(
                    messages=request_messages,
                    temperature=0,
                    max_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
                    response_format=response_format,
                    reasoning_effort=adapter.minimum_reasoning_effort,
                    timeout_seconds=self.timeout_seconds,
                )
            ),
            timeout=self.timeout_seconds,
        )
        if adapter.count_tokens(response.content) > SUMMARY_MAX_OUTPUT_TOKENS:
            raise ValueError("ConversationSummarizer 输出超过 Token 预算")
        parsed = _parse_output(response.content)
        message_ids = {item.id for item in batch}
        verified_ids = {item["id"] for item in verified_runs}
        verified_ids.update(item["id"] for item in verified_resources)
        verified_ids.update(_known_ids(memory.summary))
        verified_ids.update(
            item.reference_id for item in memory.important_references if item.reference_id is not None
        )
        message_by_id = {item.id: item for item in batch}
        run_by_id = {item["id"]: item for item in verified_runs}
        existing_entries = {
            "key_facts": memory.key_facts,
            "decisions": memory.decisions,
            "unresolved_topics": memory.unresolved_topics,
        }
        key_facts = _parse_entries(
            parsed.get("key_facts"), message_by_id, verified_ids, run_by_id,
            category="fact", existing=existing_entries["key_facts"],
        )
        decisions = _parse_entries(
            parsed.get("decisions"), message_by_id, verified_ids, run_by_id,
            category="decision", existing=existing_entries["decisions"],
        )
        unresolved = _parse_entries(
            parsed.get("unresolved_topics"), message_by_id, verified_ids, run_by_id,
            category="unresolved", existing=existing_entries["unresolved_topics"],
        )
        references = self._parse_references(
            parsed.get("references"),
            message_ids,
            verified_resources,
            memory.important_references,
            user_id,
            conversation_id,
        )
        summary = _clean_summary(parsed.get("summary"), verified_runs, verified_resources, memory.summary)
        if not summary:
            raise ValueError("ConversationSummarizer 输出缺少有效 summary")

        committed = self.store.commit_conversation_summary(
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
        if not committed:
            current = self.store.get_conversation_memory_for_user(conversation_id, user_id)
            return current is not None and (
                current.summary_version != memory.summary_version
                or current.summarized_through_message_id != memory.summarized_through_message_id
            )
        logger.info(
            "conversation_summary_committed conversation_id=%s cursor_before=%s cursor_after=%s "
            "history_messages=%s summary_input_tokens=%s elapsed_seconds=%.2f",
            conversation_id,
            memory.summarized_through_message_id,
            batch[-1].id,
            len(batch),
            request_input_tokens,
            perf_counter() - started,
        )
        return True

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
        existing_entries: list[ConversationMemoryEntry],
        user_id: str,
        conversation_id: str,
    ) -> list[ConversationMemoryEntry]:
        if not isinstance(values, list):
            return []
        candidates = {
            (item["type"], item["id"]): set(item["source_message_ids"])
            for item in verified_resources
        }
        existing = {
            (item.reference_type, item.reference_id, item.source_message_id): item
            for item in existing_entries
            if item.reference_type and item.reference_id and item.source_message_id
        }
        result: list[ConversationMemoryEntry] = []
        for item in values[:40]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", "")).casefold()
            identifier = str(item.get("id", ""))
            source_message_id = str(item.get("source_message_id", ""))
            from_current_batch = (
                source_message_id in message_ids
                and source_message_id in candidates.get((kind, identifier), set())
            )
            from_existing_memory = (kind, identifier, source_message_id) in existing
            if not from_current_batch and not from_existing_memory:
                continue
            if from_existing_memory and not self.store.message_exists(conversation_id, source_message_id):
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
            result.append(ConversationMemoryEntry(
                content=content,
                source_message_id=source_message_id,
                source_run_id=source_run_id,
                reference_type=kind,
                reference_id=identifier,
            ))
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
    if not isinstance(parsed.get("summary"), str) or any(
        not isinstance(parsed.get(key), list)
        for key in ("key_facts", "decisions", "unresolved_topics", "references")
    ):
        raise ValueError("ConversationSummarizer 输出缺少完整的结构化字段")
    return parsed


def _parse_entries(
    values: Any,
    messages: dict[str, Message],
    verified_ids: set[str],
    verified_runs: dict[str, dict[str, Any]],
    *,
    category: str,
    existing: list[ConversationMemoryEntry],
) -> list[ConversationMemoryEntry]:
    if not isinstance(values, list):
        return []
    existing_by_source = {
        (item.content, item.source_message_id): item
        for item in existing
        if item.source_message_id is not None
    }
    result: list[ConversationMemoryEntry] = []
    for item in values[:40]:
        if not isinstance(item, dict):
            continue
        content = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()
        source_message_id = str(item.get("source_message_id", ""))
        previous = existing_by_source.get((content, source_message_id))
        if previous is not None:
            result.append(previous)
            continue
        source_message = messages.get(source_message_id)
        if not content or source_message is None or not _contains_only_verified_ids(content, verified_ids):
            continue
        source_run_id = source_message.run_id
        if source_run_id:
            verified_run = verified_runs.get(source_run_id)
            if category != "unresolved" or verified_run is None or verified_run["status"] not in {"WAITING_USER", "WAITING_APPROVAL"}:
                continue
        if len(content) > SUMMARY_MAX_ENTRY_LENGTH:
            continue
        result.append(
            ConversationMemoryEntry(
                content=content,
                source_message_id=source_message_id,
                source_run_id=source_run_id,
            )
        )
    return result[-20:]


def _contains_only_verified_ids(content: str, verified_ids: set[str]) -> bool:
    identifiers = re.findall(r"\b(?:ds|dataset|art|artifact|run)[_-][A-Za-z0-9_-]+\b", content, flags=re.IGNORECASE)
    return all(identifier in verified_ids for identifier in identifiers)


def _clean_summary(
    value: Any,
    verified_runs: list[dict[str, Any]],
    verified_resources: list[dict[str, Any]],
    old_summary: str,
) -> str:
    summary = re.sub(r"\s+", " ", str(value or "")).strip()
    allowed_ids = {item["id"] for item in verified_runs}
    allowed_ids.update(item["id"] for item in verified_resources)
    allowed_ids.update(_known_ids(old_summary))
    for identifier in re.findall(r"\b(?:ds|dataset|art|artifact|run)[_-][A-Za-z0-9_-]+\b", summary, flags=re.IGNORECASE):
        if identifier not in allowed_ids:
            return ""
    return summary if len(summary) <= SUMMARY_MAX_LENGTH else ""


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


def _completed_interactions(
    messages: list[Message], runs_by_id: dict[str, Run]
) -> list[list[Message]]:
    """只把连续的 user 起点到下一个 user 起点作为可切分单元。"""

    if not messages or messages[0].role != "user":
        return []
    segments: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.role == "user":
            if current:
                if not _interaction_completed(current, runs_by_id):
                    return segments
                segments.append(current)
            current = [message]
        elif message.role == "assistant" and current:
            current.append(message)
        else:
            return segments
    if current:
        if _interaction_completed(current, runs_by_id):
            segments.append(current)
    return segments


def _interaction_completed(messages: list[Message], runs_by_id: dict[str, Run]) -> bool:
    if not any(item.role == "assistant" for item in messages):
        return False
    for message in messages:
        if message.role != "assistant" or message.run_id is None:
            continue
        run = runs_by_id.get(message.run_id)
        if run is None or run.status in _PROTECTED_RUN_STATUSES:
            return False
    return True


def _earliest_protected_run_start(
    messages: list[Message], positions: dict[str, int], runs: list[Run]
) -> int | None:
    protected = [
        run for run in runs
        if run.parent_run_id is None and run.status in _PROTECTED_RUN_STATUSES
    ]
    active_tasks = {run.task_id for run in protected if run.task_id is not None}
    protected.extend(run for run in runs if run.parent_run_id is None and run.task_id in active_tasks and run not in protected)
    if not protected:
        return len(messages)
    starts: list[int] = []
    for run in protected:
        message_id = run.metadata.get("original_request_message_id")
        if not isinstance(message_id, str) or message_id not in positions:
            return None
        starts.append(positions[message_id])
    return min(starts)


def _messages_tokens(messages: list[Message], count_tokens) -> int:
    return sum(count_tokens(message.content) for message in messages)


def _summary_batch_segments(segments: list[list[Message]], count_tokens) -> list[list[Message]]:
    selected: list[list[Message]] = []
    total = 0
    for segment in segments:
        segment_tokens = _messages_tokens(segment, count_tokens)
        if segment_tokens > SUMMARY_MAX_BATCH_TOKENS:
            if not selected:
                return []
            break
        if total + segment_tokens > SUMMARY_MAX_BATCH_TOKENS:
            break
        selected.append(segment)
        total += segment_tokens
        if total >= SUMMARY_TARGET_BATCH_TOKENS:
            break
    return selected


def _known_ids(value: str) -> set[str]:
    return set(re.findall(r"\b(?:ds|dataset|art|artifact|run)[_-][A-Za-z0-9_-]+\b", value, flags=re.IGNORECASE))


def _supports_json_object(adapter: ModelAdapter) -> bool:
    return bool(
        getattr(adapter, "supports_json_object", False)
        or getattr(adapter, "supports_json_schema", False)
        or getattr(adapter, "supports_structured_output", False)
    )


__all__ = ["ConversationSummarizer"]
