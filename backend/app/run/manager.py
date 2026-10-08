"""Run 的异步调度、恢复、截止时间与取消边界。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Checkpoint,
    ReasoningEffort,
    Run,
    RunStatus,
    TaskStatus,
    TokenUsage,
    ToolExecutionStatus,
    new_id,
)
from app.observability import EventType
from app.run.checkpoints import RunCheckpointCodec
from app.run.lifecycle import persist_result, transition
from app.run.lifecycle import resume as resume_lifecycle
from app.run.predicates import (
    is_cancellable_run,
    is_execution_inflight,
    is_resumable_run,
    is_retryable_failed_run,
)


class RunManager:
    def __init__(self, agent_loop, store, metrics=None, *, execution_timeout_seconds: float | None = None) -> None:
        self.agent_loop = agent_loop
        self.store = store
        self.metrics = metrics
        self.execution_timeout_seconds = execution_timeout_seconds
        self._active: dict[str, asyncio.Task[AgentResult]] = {}
        self._finished: dict[str, AgentResult] = {}
        self._retry_locks: dict[str, asyncio.Lock] = {}
        self._resume_locks: dict[str, asyncio.Lock] = {}

    async def submit(
        self,
        request: AgentRequest,
        *,
        metadata: dict[str, object] | None = None,
        on_run: Callable[[Run], Awaitable[None]] | None = None,
        on_model_delta: Callable[[str, TokenUsage], Awaitable[None]] | None = None,
    ) -> Run:
        prepared = await self.agent_loop.prepare_request(request, metadata=metadata)
        if on_run is not None:
            await on_run(prepared.run)
        await self.agent_loop.trace.emit(
            prepared.run.id,
            EventType.RUN_CREATED,
            "已建立 Agent Loop 运行",
            payload={"request_id": request.request_id},
            agent_id=prepared.run.agent_id,
        )
        execution = asyncio.create_task(self._execute(prepared, on_model_delta=on_model_delta))
        self._active[prepared.run.id] = execution
        if self.metrics:
            self.metrics.increment("runs.submitted")
        return prepared.run

    async def retry_run(self, run_id: str, *, user_id: str) -> Run:
        """新快照按失败位置接续；旧快照保留明确的重新发起兼容入口。"""

        source = self.store.get_run(run_id)
        if source is None:
            raise KeyError(f"运行不存在：{run_id}")
        if not user_id or not self.store.run_belongs_to_user(run_id, user_id):
            raise PermissionError("当前用户无权重试该运行")
        async with self._retry_locks.setdefault(run_id, asyncio.Lock()):
            source = self.store.get_run(run_id)
            if self.is_active(run_id) and source.metadata.get("last_continuation") == "technical_resume":
                return source
            if not is_retryable_failed_run(source) or self.is_active(run_id):
                raise RuntimeError("仅可重试已失败的主运行；副作用未确认的运行不能重试")
            related = [source, *self.store.list_child_runs(run_id)]
            if any(
                self.is_active(item.id) or is_execution_inflight(item)
                or any(status is ToolExecutionStatus.RUNNING for _, status, _ in self.store.list_tool_calls(item.id))
                for item in related
            ):
                raise RuntimeError("原运行或子运行仍有未结束的执行记录，请先确认执行结果")
            # 重复点击共享仍在运行/等待的新尝试；不启动第二份相同请求。
            for attempt in self.store.list_runs_for_conversation(source.conversation_id):
                if attempt.metadata.get("retry_of") == run_id and is_cancellable_run(attempt):
                    return attempt
            checkpoint = self.store.latest_checkpoint(run_id)
            saved_request = RunCheckpointCodec.request(checkpoint.state) if checkpoint is not None else None
            if saved_request is None or saved_request.conversation_id != source.conversation_id:
                raise RuntimeError("失败运行没有有效的原始请求，不能重试")
            for dataset_id in dict.fromkeys([*saved_request.dataset_ids, *saved_request.attachment_ids]):
                if self.store.get_dataset_for_user(dataset_id, user_id) is None:
                    raise PermissionError("原请求中的数据集不存在或当前用户已无权访问")
            if checkpoint.state.get("schema_version", 0) >= RunCheckpointCodec.RECOVERY_VERSION:
                if not self.agent_loop.recovery.can_resume(source):
                    raise RuntimeError("该失败位置无法安全接续，或自动重试额度已用尽；不会从头重复执行")
                return await self.continue_run(run_id, user_id=user_id, technical=True)
            request = saved_request.model_copy(update={
                "request_id": new_id("req"),
                "user_id": user_id,
                "reply_to_run_id": None,
                "related_task_id": None,
                "task_relation": None,
                "referenced_run_ids": list(dict.fromkeys([*saved_request.referenced_run_ids, run_id])),
            })
            return await self.submit(request, metadata={
                "retry_of": run_id,
                "retry_mode": "restart_request",
                "original_request_message_id": source.metadata.get("original_request_message_id"),
            })

    async def continue_run(
        self, run_id: str, **options,
    ) -> Run:
        async with self._resume_locks.setdefault(run_id, asyncio.Lock()):
            return await self._continue_run(run_id, **options)

    async def _continue_run(
        self,
        run_id: str,
        *,
        user_input: str | None = None,
        user_id: str | None = None,
        dataset_ids: list[str] | None = None,
        attachment_ids: list[str] | None = None,
        model_profile: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        approval_id: str | None = None,
        approved: bool | None = None,
        technical: bool = False,
        on_run: Callable[[Run], Awaitable[None]] | None = None,
        on_model_delta: Callable[[str, TokenUsage], Awaitable[None]] | None = None,
        on_result: Callable[[str, AgentResult], Awaitable[None]] | None = None,
    ) -> Run:
        current = self.store.get_run(run_id)
        if current is None:
            raise KeyError(f"运行不存在：{run_id}")
        if user_id and not self.store.run_belongs_to_user(run_id, user_id):
            raise PermissionError("当前用户无权继续该运行")
        if self.is_active(run_id):
            raise RuntimeError("运行当前仍在执行，不能重复恢复")
        recovery = self.agent_loop.recovery
        if technical and (user_input is not None or approval_id is not None or approved is not None):
            raise ValueError("技术恢复不能附带用户输入或审批结果")
        if technical and any(status is ToolExecutionStatus.RUNNING for item in [current, *self.store.list_child_runs(run_id)]
                             for _, status, _ in self.store.list_tool_calls(item.id)):
            raise RuntimeError("原调用尚未结束或副作用无法确认，不能重复执行")
        if technical and recovery is not None and recovery.current(run_id).get("published_text"):
            raise RuntimeError("已发布正文的请求不能重放，请在对话中提出新的请求")
        if user_input is not None and current.status is not RunStatus.WAITING_USER:
            raise RuntimeError("运行当前不在等待用户输入状态")
        if approved is not None and current.status not in {RunStatus.WAITING_APPROVAL, RunStatus.RUNNING}:
            raise RuntimeError("运行当前不在等待审批状态")
        failed_recovery = recovery is not None and current.status is RunStatus.FAILED and recovery.can_resume(current)
        if technical and not (failed_recovery or is_resumable_run(current, has_checkpoint=self.store.latest_checkpoint(run_id) is not None)):
            raise RuntimeError("运行当前不支持技术恢复")
        if user_input is None and approved is None and not technical:
            raise ValueError("连续恢复必须提供用户输入或审批结果")
        if (approval_id is None) != (approved is None):
            raise ValueError("审批恢复必须同时提供 approval_id 和 approved")

        checkpoint = self.store.latest_checkpoint(run_id)
        if checkpoint is None:
            raise RuntimeError("运行没有可恢复 Checkpoint")
        saved_request = RunCheckpointCodec.request(checkpoint.state)
        if saved_request is None:
            raise RuntimeError("Checkpoint 缺少原始请求")
        if user_input is not None and not user_input.strip():
            raise ValueError("用户补充内容不能为空")

        if user_input is not None:
            continuation = {"type": "user_input", "content": user_input.strip()}
        elif technical:
            continuation = {"type": "technical_resume"}
        else:
            continuation = {"type": "approval_result", "approval_id": approval_id, "approved": approved}

        request = saved_request.model_copy(
            update={
                "user_id": user_id or saved_request.user_id,
                "model_profile": model_profile or saved_request.model_profile,
                "reasoning_effort": reasoning_effort or saved_request.reasoning_effort,
                "dataset_ids": list(dict.fromkeys([*saved_request.dataset_ids, *(dataset_ids or [])])),
                "attachment_ids": list(dict.fromkeys([*saved_request.attachment_ids, *(attachment_ids or [])])),
            }
        )
        domain_task = self.store.get_task(current.task_id) if current.task_id and not current.parent_run_id else None
        resumed, _ = resume_lifecycle(self.store, current, domain_task, metadata={"last_continuation": continuation["type"]})
        if technical and recovery is not None:
            state = recovery.current(run_id)
            if state.get("scope") == "model" and state.get("status") == "running":
                recovery.store.update_recovery(run_id, {**state, "status": "failed", "retryable": True,
                                                        "error_code": "PROCESS_RESTARTED"})
        prepared = self.agent_loop.prepare_resume(request, resumed)
        if on_run is not None:
            await on_run(resumed)
        execution = asyncio.create_task(
            self._execute(prepared, checkpoint, continuation=continuation, on_model_delta=on_model_delta, on_result=on_result)
        )
        self._active[run_id] = execution
        self._finished.pop(run_id, None)
        if self.metrics:
            self.metrics.increment("runs.continued")
        return resumed

    async def submit_child(self, request: AgentRequest, child: Run) -> Run:
        """只执行调度器已经持久化的子身份，不接受外部请求指定父子关系。"""

        current = self.store.get_run(child.id)
        if current is None or current.status is not RunStatus.CREATED or not current.parent_run_id:
            raise RuntimeError("子 Run 未处于已准备状态")
        if not request.user_id or not self.store.run_belongs_to_user(current.id, request.user_id):
            raise PermissionError("子 Run 身份无效")
        prepared = self.agent_loop.prepare_child_request(request, current)
        self._active[current.id] = asyncio.create_task(self._execute(prepared))
        return prepared.run

    async def cancel_children(self, parent_run_id: str) -> None:
        for child in self.store.list_child_runs(parent_run_id):
            if child.status is RunStatus.CREATED:
                result = AgentResult(agent_id=child.agent_id, status=AgentResultStatus.CANCELLED,
                                     summary="父运行已停止，子任务没有启动。", error="CANCELLED", trace_id=child.id)
                persist_result(self.store, child, None, result, run_status=RunStatus.CANCELLED)
            elif is_cancellable_run(child):
                await self.cancel(child.id)

    async def _execute(
        self,
        prepared,
        checkpoint: Checkpoint | None = None,
        *,
        continuation: dict[str, object] | None = None,
        on_model_delta: Callable[[str, TokenUsage], Awaitable[None]] | None = None,
        on_result: Callable[[str, AgentResult], Awaitable[None]] | None = None,
    ) -> AgentResult:
        run = prepared.run
        started = time.perf_counter()
        budget_used = float(run.metadata.get("execution_elapsed_seconds", 0))
        try:
            execute = self.agent_loop.run(
                prepared.request,
                prepared=prepared,
                resume_from=checkpoint,
                continuation=continuation,
                on_model_delta=on_model_delta,
            )
            if self.execution_timeout_seconds is None:
                result = await execute
            else:
                remaining = max(0, self.execution_timeout_seconds - budget_used)
                deadline = run.metadata.get("execution_deadline_at")
                if deadline:
                    remaining = min(remaining, max(0, (datetime.fromisoformat(deadline) - datetime.now(UTC)).total_seconds()))
                if run.parent_run_id:
                    parent = self.store.get_run(run.parent_run_id)
                    if parent and parent.metadata.get("execution_deadline_at"):
                        remaining = min(remaining, max(0, (datetime.fromisoformat(parent.metadata["execution_deadline_at"]) - datetime.now(UTC)).total_seconds()))
                budget_used = max(budget_used, self.execution_timeout_seconds - remaining)
                latest = self.store.get_run(run.id) or run
                self.store.save_run(latest.model_copy(update={"metadata": {**latest.metadata,
                    "execution_deadline_at": (datetime.now(UTC) + timedelta(seconds=remaining)).isoformat()}}))
                if remaining <= 0:
                    execute.close()
                    raise TimeoutError
                result = await asyncio.wait_for(execute, timeout=max(0.001, remaining))
            self._finished[run.id] = result
            if self.metrics:
                self.metrics.increment(f"runs.finished.{result.status.value.casefold()}")
            return result
        except TimeoutError:
            self.agent_loop.executor.cancel_run(run.id)
            await self.cancel_children(run.id)
            if self.agent_loop.delegation is not None:
                await self.agent_loop.delegation.record_stopped(run.id)
            current = self.store.get_run(run.id) or run
            result = AgentResult(
                agent_id=current.agent_id,
                status=AgentResultStatus.BLOCKED,
                summary="运行超过总执行时间限制，已停止。",
                error="BUDGET_EXCEEDED",
                trace_id=current.id,
            )
            task = self.store.get_task(current.task_id) if current.task_id and not current.parent_run_id else None
            persist_result(self.store, current, task, result, run_status=RunStatus.BUDGET_EXCEEDED)
            await self.agent_loop.trace.emit(current.id, "RunDeadlineExceeded", result.summary, payload={"timeout_seconds": self.execution_timeout_seconds}, agent_id=current.agent_id)
            self._finished[run.id] = result
            return result
        except asyncio.CancelledError:
            self.agent_loop.executor.cancel_run(run.id)
            await self.cancel_children(run.id)
            if self.agent_loop.delegation is not None:
                await self.agent_loop.delegation.record_stopped(run.id)
            current = self.store.get_run(run.id) or run
            result = AgentResult(
                agent_id=current.agent_id,
                status=AgentResultStatus.CANCELLED,
                summary="运行已取消。",
                error="CANCELLED",
                trace_id=current.id,
            )
            task = self.store.get_task(current.task_id) if current.task_id and not current.parent_run_id else None
            persist_result(self.store, current, task, result, run_status=RunStatus.CANCELLED)
            await self.agent_loop.trace.emit(current.id, EventType.RUN_CANCELLED, result.summary, agent_id=current.agent_id)
            self._finished[run.id] = result
            return result
        except Exception as exc:
            await self.cancel_children(run.id)
            if self.agent_loop.delegation is not None:
                await self.agent_loop.delegation.record_stopped(run.id)
            current = self.store.get_run(run.id) or run
            result = AgentResult(agent_id=current.agent_id, status=AgentResultStatus.FAILED,
                                 summary="执行过程中发生异常，运行已停止。", error="RUNTIME_ERROR", trace_id=current.id)
            task = self.store.get_task(current.task_id) if current.task_id and not current.parent_run_id else None
            persist_result(self.store, current, task, result)
            await self.agent_loop.trace.emit(current.id, EventType.RUN_FAILED, result.summary,
                                            payload={"error_type": type(exc).__name__}, agent_id=current.agent_id)
            self._finished[run.id] = result
            return result
        finally:
            latest = self.store.get_run(run.id)
            if latest is not None:
                metadata = {**latest.metadata, "execution_elapsed_seconds": budget_used + time.perf_counter() - started}
                metadata.pop("execution_deadline_at", None)
                self.store.save_run(latest.model_copy(update={"metadata": metadata}))
            if self.metrics:
                self.metrics.observe("total_run_ms", (time.perf_counter() - started) * 1000)
            self._active.pop(run.id, None)
            if on_result is not None and run.id in self._finished:
                await on_result(run.id, self._finished[run.id])

    async def wait(self, run_id: str) -> AgentResult:
        task = self._active.get(run_id)
        if task is not None:
            return await task
        if run_id in self._finished:
            return self._finished[run_id]
        run = self.store.get_run(run_id)
        if run and isinstance(run.metadata.get("result"), dict):
            result = AgentResult.model_validate(run.metadata["result"])
            self._finished[run_id] = result
            return result
        raise KeyError(f"运行不存在或尚未提交：{run_id}")

    def is_active(self, run_id: str) -> bool:
        task = self._active.get(run_id)
        return task is not None and not task.done()

    def forget(self, run_id: str) -> None:
        self._finished.pop(run_id, None)

    def reconcile_orphaned_runs(self) -> list[str]:
        interrupted = []
        for run in self.store.list_runs(limit=10000):
            if is_execution_inflight(run) and not self.is_active(run.id):
                task = self.store.get_task(run.task_id) if run.task_id and not run.parent_run_id else None
                transition(self.store, run, task, run_status=RunStatus.INTERRUPTED,
                           task_status=TaskStatus.BLOCKED, error="PROCESS_RESTARTED")
                interrupted.append(run.id)
        return interrupted

    async def recover_interrupted_runs(self, *, on_result=None) -> list[str]:
        """连接工具服务后恢复有安全游标的主运行；子运行由原委派图接续。"""
        recovered = []
        recovery = self.agent_loop.recovery
        if recovery is None:
            return recovered
        for run in self.store.list_runs(limit=10000):
            if run.status is not RunStatus.INTERRUPTED or run.parent_run_id or self.is_active(run.id):
                continue
            related = [run, *self.store.list_child_runs(run.id)]
            if not recovery.can_resume(run) or any(
                status is ToolExecutionStatus.RUNNING
                for item in related for _, status, _ in self.store.list_tool_calls(item.id)
            ):
                continue
            try:
                await self.continue_run(run.id, user_id=self.store.user_id_for_run(run.id), technical=True, on_result=on_result)
            except (KeyError, ValueError, RuntimeError, PermissionError) as exc:
                await self.agent_loop.trace.emit(run.id, "RecoveryBlocked", "中断运行无法自动接续",
                                                payload={"error_type": type(exc).__name__})
            else:
                recovered.append(run.id)
        return recovered

    async def cancel(self, run_id: str) -> bool:
        run = self.store.get_run(run_id)
        if run is None or not is_cancellable_run(run):
            return False
        task = self._active.get(run_id)
        if task and not task.done():
            self.agent_loop.executor.cancel_run(run_id)
            task.cancel()
            await task
            if self.metrics:
                self.metrics.increment("runs.cancelled")
            return True
        await self.cancel_children(run_id)
        if self.agent_loop.delegation is not None:
            await self.agent_loop.delegation.record_stopped(run_id)
        result = AgentResult(
            agent_id=run.agent_id,
            status=AgentResultStatus.CANCELLED,
            summary="运行已取消。",
            error="CANCELLED",
            trace_id=run_id,
        )
        domain_task = self.store.get_task(run.task_id) if run.task_id and not run.parent_run_id else None
        persist_result(self.store, run, domain_task, result, run_status=RunStatus.CANCELLED)
        await self.agent_loop.trace.emit(run.id, EventType.RUN_CANCELLED, result.summary, agent_id=run.agent_id)
        self._finished[run_id] = result
        if self.metrics:
            self.metrics.increment("runs.cancelled")
        return True

    async def close(self) -> None:
        for run_id in list(self._active):
            if self.is_active(run_id):
                await self.cancel(run_id)


__all__ = ["RunManager"]
