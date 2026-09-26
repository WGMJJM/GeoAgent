"""单服务进程内的受控 DAG 委派；身份、结果和恢复游标持久化到 SQLite。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from typing import Any

from pydantic import ValidationError

from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Checkpoint,
    DatasetOutputPolicy,
    DatasetOutputRef,
    DelegationPlan,
    DelegationResult,
    DelegationSubtask,
    ErrorCategory,
    Run,
    RunStatus,
    SubAgentResult,
    SubTask,
    Task,
    TaskStatus,
    ToolError,
    ToolExecutionStatus,
    ToolResult,
    ToolStatus,
    WorkingMemory,
    WorkingMemoryDelta,
    new_id,
)
from app.observability import EventType
from app.run.lifecycle import persist_result
from app.state.working_memory import WorkingMemoryUpdater

DELEGATE_TOOL = {
    "type": "function",
    "function": {
        "name": "agent.delegate",
        "description": "将可拆分任务交给独立子 Agent。提交结构化依赖、输入数据和最小工具白名单；output_roles 将输出角色映射到真实产出工具名。子 Agent 不能再次委派。",
        "parameters": DelegationPlan.model_json_schema(),
    },
}


class DelegationCoordinator:
    def __init__(self, store, loop, manager, settings) -> None:
        self.store, self.loop, self.manager, self.settings = store, loop, manager, settings
        self.updater = WorkingMemoryUpdater(store)
        self._locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(settings.max_parallel_agents)
        # 共享写资源只在单进程内互斥；当前未开放的 Python/Shell 将来也须串行。
        self._exclusive_lock = asyncio.Lock()

    def validate(
        self, arguments: dict[str, Any], request: AgentRequest, parent: Run
    ) -> DelegationPlan:
        plan = DelegationPlan.model_validate(arguments)
        if parent.parent_run_id:
            raise ValueError("SUBAGENT_DELEGATION_FORBIDDEN: 委派深度上限为 1")
        if not request.user_id or not self.store.run_belongs_to_user(parent.id, request.user_id):
            raise ValueError("DELEGATION_PERMISSION_DENIED: 委派需要有效的父运行认证身份")
        if len(plan.subtasks) > self.settings.max_subagents:
            raise ValueError("子任务数量超过 max_subagents")
        by_id = {item.id: item for item in plan.subtasks}
        if len(by_id) != len(plan.subtasks):
            raise ValueError("子任务 ID 不能重复")
        context = self.loop._discovery_context(
            request, self.loop.services_factory(request.user_id), parent
        )
        for item in plan.subtasks:
            if item.id in item.dependencies or any(name not in by_id for name in item.dependencies):
                raise ValueError("依赖必须引用计划内其他子任务")
            for dataset_id in item.dataset_ids:
                if self.store.get_dataset_for_user(dataset_id, request.user_id) is None:
                    raise ValueError("DATASET_NOT_VISIBLE: 子任务输入不存在或不可访问")
            for name in item.allowed_tools:
                try:
                    metadata = self.loop.registry.get(name).metadata
                except KeyError as exc:
                    raise ValueError(f"UNKNOWN_TOOL: {name}") from exc
                if not self.loop.policy.is_discoverable(metadata, context):
                    raise ValueError(f"TOOL_NOT_AVAILABLE: {name}")
            if any(
                not role.strip()
                or len(role) > 64
                or name not in item.allowed_tools
                or self.loop.registry.get(name).metadata.dataset_output_policy
                is DatasetOutputPolicy.NONE
                for role, name in item.output_roles.items()
            ):
                raise ValueError("输出角色必须指向允许的 Dataset 产出工具")
            names = [binding.input_name for binding in item.upstream_dataset_bindings]
            if len(set(names)) != len(names):
                raise ValueError("上游输入名称不能重复")
            for binding in item.upstream_dataset_bindings:
                if binding.from_subtask not in item.dependencies:
                    raise ValueError("上游绑定必须引用已声明依赖")
                if (
                    binding.output_role is not None
                    and binding.output_role not in by_id[binding.from_subtask].output_roles
                ):
                    raise ValueError("上游输出角色未声明")
        remaining = set(by_id)
        while remaining:
            ready = {
                name for name in remaining if not remaining.intersection(by_id[name].dependencies)
            }
            if not ready:
                raise ValueError("依赖图存在有向环")
            remaining -= ready
        return plan

    async def execute(
        self,
        arguments: dict[str, Any],
        *,
        request: AgentRequest,
        parent: Run,
        call_id: str,
        continuation: dict[str, object] | None = None,
    ) -> ToolResult:
        lock = self._locks.setdefault(call_id, asyncio.Lock())
        async with lock:
            try:
                state = self.store.get_delegation(call_id)
                if state is None:
                    plan = self.validate(arguments, request, parent)
                    state = self._prepare(plan, request, parent, call_id)
                else:
                    plan = DelegationPlan.model_validate(arguments)
                    if (
                        state["fingerprint"] != _fingerprint(plan)
                        or state["parent_run_id"] != parent.id
                    ):
                        raise ValueError("DELEGATION_CALL_MISMATCH: 稳定调用 ID 的计划不可修改")
                    if (
                        not request.user_id
                        or not self.store.run_belongs_to_user(parent.id, request.user_id)
                        or parent.parent_run_id
                    ):
                        raise ValueError("DELEGATION_PERMISSION_DENIED")
                results = {
                    name: SubAgentResult.model_validate(payload)
                    for name, payload in state["results"].items()
                }
                self._validate_saved_references(results, request.user_id)
                await self._emit_created(state, plan)
                if state.get("completed"):
                    return self._observation(
                        call_id, DelegationResult.model_validate(state["result"])
                    )
                await self._resume_waiting(state, plan, results, request, continuation)
                await self._schedule(state, plan, results, request)
                outcome = self._aggregate(state, plan, results)
                waiting = next(
                    (
                        item
                        for item in outcome.subtasks
                        if item.execution_status is TaskStatus.WAITING
                    ),
                    None,
                )
                if waiting is None:
                    memory = self.updater.load_or_create(state["task_id"], request.conversation_id)
                    merged = self.updater.merge_deltas(
                        memory, [item.memory_delta for item in outcome.subtasks]
                    )
                    state.update(
                        completed=True, merged=True, result=outcome.model_dump(mode="json")
                    )
                    self.store.save_delegation(state, memory=merged)
                    await self.loop.trace.emit(
                        parent.id,
                        EventType.DELEGATION_COMPLETED,
                        "结构化委派结果已保存并合并",
                        payload={"delegation_id": state["id"], "status": outcome.status.value},
                        agent_id=parent.agent_id,
                    )
                else:
                    state["result"] = outcome.model_dump(mode="json")
                    self.store.save_delegation(state)
                return self._observation(call_id, outcome)
            except (ValueError, ValidationError) as exc:
                return ToolResult(
                    call_id=call_id,
                    status=ToolStatus.BLOCKED,
                    error=ToolError(
                        code="INVALID_DELEGATION_PLAN",
                        category=ErrorCategory.INPUT,
                        message=str(exc)[:1500],
                    ),
                )

    def _prepare(
        self, plan: DelegationPlan, request: AgentRequest, parent: Run, call_id: str
    ) -> dict[str, Any]:
        parent = self.store.get_run(parent.id) or parent
        task = self.store.get_task(parent.task_id) if parent.task_id else None
        task = task or Task(
            goal=str(parent.metadata.get("original_request") or request.user_input),
            conversation_id=request.conversation_id,
            status=TaskStatus.RUNNING,
        )
        delegation_id = new_id("del")
        context = self.loop._discovery_context(
            request, self.loop.services_factory(request.user_id), parent
        )
        subtasks, children = [], []
        for item in plan.subtasks:
            subtask_id = f"{delegation_id}:{item.id}"
            child = Run(
                parent_run_id=parent.id,
                task_id=task.id,
                conversation_id=request.conversation_id,
                agent_id=f"subagent:{subtask_id}",
                metadata={
                    "delegation_id": delegation_id,
                    "subtask_id": item.id,
                    "original_request": item.goal,
                    "allowed_tool_names": item.allowed_tools,
                    "parent_granted_scopes": sorted(context.granted_scopes),
                    "parent_available_envs": sorted(context.available_envs),
                    "output_roles": item.output_roles,
                    "protocol_version": 1,
                },
            )
            children.append(child)
            subtasks.append(
                SubTask(
                    id=subtask_id,
                    goal=item.goal,
                    description=item.description,
                    dataset_ids=item.dataset_ids,
                    dependencies=[f"{delegation_id}:{name}" for name in item.dependencies],
                    allowed_tools=item.allowed_tools,
                    parallelizable=item.parallelizable,
                    required=item.required,
                    assigned_agent_id=child.agent_id,
                )
            )
        task = task.model_copy(
            update={"subtasks": [*task.subtasks, *(item.id for item in subtasks)]}
        )
        parent = parent.model_copy(
            update={"task_id": task.id, "status": RunStatus.WAITING_SUBAGENT}
        )
        state = {
            "id": delegation_id,
            "call_id": call_id,
            "parent_run_id": parent.id,
            "task_id": task.id,
            "fingerprint": _fingerprint(plan),
            "plan": plan.model_dump(mode="json"),
            "run_ids": {
                item.id: child.id for item, child in zip(plan.subtasks, children, strict=True)
            },
            "results": {},
            "completed": False,
            "merged": False,
        }
        memory = self.store.get_working_memory(task.id) or WorkingMemory(
            task_id=task.id, conversation_id=request.conversation_id
        )
        if not self.store.create_delegation(state, parent, task, subtasks, children, memory):
            return self.store.get_delegation(call_id)
        return state

    async def _emit_created(self, state, plan) -> None:
        created = {
            event.payload.get("run_id")
            for event in self.store.list_events(state["parent_run_id"])
            if event.event_type == EventType.SUBTASK_CREATED
        }
        for item in plan.subtasks:
            run_id = state["run_ids"][item.id]
            if run_id not in created:
                await self.loop.trace.emit(
                    state["parent_run_id"],
                    EventType.SUBTASK_CREATED,
                    item.goal,
                    payload={"subtask_id": item.id, "run_id": run_id},
                )

    async def _resume_waiting(self, state, plan, results, request, continuation) -> None:
        if not continuation:
            return
        for item in plan.subtasks:
            saved = results.get(item.id)
            if saved is None or saved.execution_status is not TaskStatus.WAITING:
                continue
            needs = saved.needs_input or {}
            async with self._semaphore:
                if continuation.get("type") == "user_input" and needs.get("question"):
                    await self.manager.continue_run(
                        saved.run_id,
                        user_input=str(continuation["content"]),
                        user_id=request.user_id,
                        dataset_ids=request.dataset_ids,
                    )
                elif continuation.get("type") == "approval_result" and needs.get(
                    "approval_id"
                ) == continuation.get("approval_id"):
                    await self.manager.continue_run(
                        saved.run_id,
                        user_id=request.user_id,
                        approval_id=str(continuation["approval_id"]),
                        approved=continuation.get("approved"),
                    )
                else:
                    continue
                await self.manager.wait(saved.run_id)
            result = self._collect_result(item, self.store.get_run(saved.run_id), request.user_id)
            await self._persist_result(state, result)
            results[item.id] = result
            break  # 一次用户交互只恢复其绑定的原子子任务。

    async def _schedule(self, state, plan, results, request) -> None:
        active: dict[asyncio.Task, DelegationSubtask] = {}
        try:
            while True:
                running = {item.id for item in active.values()}
                pending = [
                    item
                    for item in plan.subtasks
                    if item.id not in results and item.id not in running
                ]
                for item in pending:
                    if any(
                        dep in results
                        and results[dep].execution_status
                        not in {TaskStatus.SUCCEEDED, TaskStatus.WAITING, TaskStatus.PENDING}
                        for dep in item.dependencies
                    ):
                        result = self._blocked(
                            item, state, "DEPENDENCY_FAILED", "上游没有成功，后继任务未启动。"
                        )
                        await self._persist_result(state, result)
                        results[item.id] = result
                ready = [
                    item
                    for item in pending
                    if item.id not in results
                    and all(
                        dep in results and results[dep].execution_status is TaskStatus.SUCCEEDED
                        for dep in item.dependencies
                    )
                ]
                if not any(not item.parallelizable for item in active.values()):
                    for item in ready:
                        if len(active) >= self.settings.max_parallel_agents:
                            break
                        if not item.parallelizable and active:
                            break
                        execution = asyncio.create_task(
                            self._run_subtask(item, state, results, request)
                        )
                        active[execution] = item
                        if not item.parallelizable:
                            break
                if not active:
                    break
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                # 完成顺序不用于 WorkingMemory 顺序；只持久化各子结果。
                for execution in sorted(done, key=lambda value: active[value].id):
                    item = active.pop(execution)
                    result = execution.result()
                    await self._persist_result(state, result)
                    results[item.id] = result
        finally:
            for execution in active:
                execution.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            if active:
                await self.manager.cancel_children(state["parent_run_id"])

    async def _run_subtask(self, item, state, results, request) -> SubAgentResult:
        child = self.store.get_run(state["run_ids"][item.id])
        inputs = list(item.dataset_ids)
        bindings = {}
        for binding in item.upstream_dataset_bindings:
            candidates = [
                ref
                for ref in results[binding.from_subtask].datasets
                if binding.output_role is None or ref.role == binding.output_role
            ]
            unique = {ref.dataset_id for ref in candidates}
            if len(unique) != 1:
                return self._blocked(
                    item, state, "UPSTREAM_OUTPUT_AMBIGUOUS", "声明的上游角色缺失或包含多个输出。"
                )
            dataset_id = unique.pop()
            dataset = self.store.get_dataset_for_user(dataset_id, request.user_id)
            if dataset is None or dataset.created_by_run_id != results[binding.from_subtask].run_id:
                return self._blocked(
                    item, state, "UPSTREAM_OUTPUT_NOT_VISIBLE", "上游输出不存在或来源不匹配。"
                )
            inputs.append(dataset_id)
            bindings[binding.input_name] = {
                "dataset_id": dataset_id,
                "from_subtask": binding.from_subtask,
                "role": binding.output_role,
            }
        if any(
            self.store.get_dataset_for_user(identifier, request.user_id) is None
            for identifier in inputs
        ):
            return self._blocked(item, state, "DATASET_NOT_VISIBLE", "子任务输入权限已变更。")
        calls = self.store.list_tool_calls(child.id)
        if any(status is ToolExecutionStatus.RUNNING for _, status, _ in calls):
            return self._blocked(
                item,
                state,
                "SIDE_EFFECT_UNCERTAIN",
                "中断工具的副作用无法确认，需要人工处理；没有重跑。",
            )
        if child.status in {RunStatus.COMPLETED, RunStatus.PARTIAL_COMPLETED, RunStatus.FAILED}:
            return self._collect_result(item, child, request.user_id)
        child = child.model_copy(
            update={"metadata": {**child.metadata, "upstream_inputs": bindings}}
        )
        self.store.save_run(child)
        child_request = AgentRequest(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            user_input=item.goal,
            dataset_ids=list(dict.fromkeys(inputs)),
            model_profile=request.model_profile,
        )
        async with self._semaphore:
            # 代码/命令执行或覆盖源资源需要共享资源互斥，现有默认权限仍会拒绝其发现。
            shared_write = bool(
                {"python.execute", "shell.execute"}.intersection(item.allowed_tools)
            )
            if shared_write:
                async with self._exclusive_lock:
                    await self._execute_child(child, child_request, item)
            else:
                await self._execute_child(child, child_request, item)
        return self._collect_result(item, self.store.get_run(child.id), request.user_id)

    async def _execute_child(self, child, request, item) -> None:
        subtask = self.store.get_subtask(
            child.task_id, f"{child.metadata['delegation_id']}:{item.id}"
        )
        self.store.save_subtask(
            child.task_id, subtask.model_copy(update={"status": TaskStatus.RUNNING})
        )
        if self.store.latest_checkpoint(child.id) is None:
            await self.manager.submit_child(request, child)
        else:
            await self.manager.continue_run(child.id, user_id=request.user_id, technical=True)
        await self.loop.trace.emit(
            child.parent_run_id,
            EventType.SUBAGENT_SPAWNED,
            "子 Run 已受控启动",
            payload={"subtask_id": item.id, "run_id": child.id},
            agent_id=child.agent_id,
        )
        await self.manager.wait(child.id)

    def _collect_result(self, item, child: Run, user_id: str) -> SubAgentResult:
        raw = child.metadata.get("result") or {}
        result = AgentResult.model_validate(raw) if raw else None
        status = {
            RunStatus.COMPLETED: AgentResultStatus.SUCCESS,
            RunStatus.PARTIAL_COMPLETED: AgentResultStatus.PARTIAL,
            RunStatus.CANCELLED: AgentResultStatus.CANCELLED,
            RunStatus.WAITING_USER: AgentResultStatus.BLOCKED,
            RunStatus.WAITING_APPROVAL: AgentResultStatus.BLOCKED,
        }.get(child.status, AgentResultStatus.FAILED)
        execution_status = {
            AgentResultStatus.SUCCESS: TaskStatus.SUCCEEDED,
            AgentResultStatus.PARTIAL: TaskStatus.PARTIAL,
            AgentResultStatus.FAILED: TaskStatus.FAILED,
            AgentResultStatus.CANCELLED: TaskStatus.CANCELLED,
            AgentResultStatus.BLOCKED: TaskStatus.WAITING,
        }[status]
        error = ToolError(code=child.error, message=child.error) if child.error else None
        if any(
            state is ToolExecutionStatus.RUNNING
            for _, state, _ in self.store.list_tool_calls(child.id)
        ):
            error = ToolError(
                code="SIDE_EFFECT_UNCERTAIN",
                category=ErrorCategory.EXECUTION,
                message="工具执行或清理中断，副作用无法确认，需要人工处理。",
            )
        outputs, artifacts, metrics, sources, warnings = {}, [], {}, {}, []
        by_tool: dict[str, set[str]] = {}
        delta = WorkingMemoryDelta(source_run_id=child.id)
        successful = 0
        partial = False
        for call, _, value in self.store.list_tool_calls(child.id):
            if value is None:
                continue
            if value.error:
                error = value.error
            if value.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
                clean = value.model_copy(update={"datasets": [], "artifacts": []})
                delta = self.updater.merge_delta(
                    delta, self.updater.build_delta_from_tool_result(clean, run_id=child.id)
                )
                continue
            successful += 1
            partial = partial or value.status is ToolStatus.PARTIAL_SUCCESS
            metadata = (
                self.loop.registry.get(call.name).metadata
                if call.name in self.loop.registry.names()
                else None
            )
            verified_ids, verified_artifacts = [], []
            for identifier in value.datasets:
                dataset = self.store.get_dataset_for_user(identifier, user_id)
                if dataset is None:
                    error = ToolError(
                        code="UNVERIFIED_DATASET_OUTPUT", message="工具引用的数据集不存在或不可见。"
                    )
                    status, execution_status = AgentResultStatus.FAILED, TaskStatus.FAILED
                elif dataset.created_by_run_id == child.id:
                    verified_ids.append(identifier)
                    outputs[identifier] = DatasetOutputRef(
                        dataset_id=identifier, source_dataset_ids=dataset.source_dataset_ids
                    )
                    by_tool.setdefault(call.name, set()).add(identifier)
            for identifier in value.artifacts:
                artifact = self.store.get_artifact_for_user(identifier, user_id)
                if artifact is None or artifact.run_id != child.id:
                    error = ToolError(
                        code="UNVERIFIED_ARTIFACT_OUTPUT", message="工具产物不存在或来源不匹配。"
                    )
                    status, execution_status = AgentResultStatus.FAILED, TaskStatus.FAILED
                else:
                    verified_artifacts.append(identifier)
                    if identifier not in artifacts:
                        artifacts.append(identifier)
            if (
                metadata
                and metadata.dataset_output_policy is DatasetOutputPolicy.REQUIRED
                and not verified_ids
            ):
                error = ToolError(
                    code="REQUIRED_DATASET_OUTPUT_MISSING",
                    message="必要 Dataset 输出没有登记到当前子 Run。",
                )
                status, execution_status = AgentResultStatus.FAILED, TaskStatus.FAILED
            clean = value.model_copy(
                update={"datasets": verified_ids, "artifacts": verified_artifacts}
            )
            delta = self.updater.merge_delta(
                delta, self.updater.build_delta_from_tool_result(clean, run_id=child.id)
            )
            for key, scalar in _metrics(value.output).items():
                metric_key = f"{call.id}.{key}"
                metrics[metric_key] = scalar
                sources[metric_key] = call.id
            warnings.extend(value.warnings)
        refs = []
        if status is AgentResultStatus.SUCCESS:
            # 工具名单声明能力，不声明必须执行的工作；必要产出由 output_roles 指定。
            if not successful:
                error = error or ToolError(
                    code="SUBAGENT_NO_VERIFIED_RESULT",
                    message="模型回复缺少必要的成功工具和已验证产出。",
                )
                status, execution_status = AgentResultStatus.FAILED, TaskStatus.FAILED
            elif not partial:
                error = None  # 已正常修复的历史错误不会覆盖实际完成状态。
        for role, name in item.output_roles.items():
            candidates = by_tool.get(name, set())
            if len(candidates) != 1 and status is AgentResultStatus.SUCCESS:
                error = ToolError(
                    code="OUTPUT_ROLE_AMBIGUOUS", message=f"输出角色 {role} 缺失或存在多个候选。"
                )
                status, execution_status = AgentResultStatus.FAILED, TaskStatus.FAILED
            elif len(candidates) == 1:
                refs.append(outputs[next(iter(candidates))].model_copy(update={"role": role}))
        used = {ref.dataset_id for ref in refs}
        refs.extend(outputs[name] for name in sorted(outputs) if name not in used)
        if status is AgentResultStatus.SUCCESS and partial:
            status, execution_status = AgentResultStatus.PARTIAL, TaskStatus.PARTIAL
        if (
            execution_status is TaskStatus.WAITING
            and result
            and result.needs_input
            and result.needs_input.get("question")
        ):
            delta = self.updater.merge_delta(
                delta,
                self.updater.build_unresolved_question_delta(
                    [result.needs_input["question"]], run_id=child.id
                ),
            )
        return SubAgentResult(
            subtask_id=item.id,
            run_id=child.id,
            status=status,
            execution_status=execution_status,
            datasets=refs,
            artifact_ids=artifacts,
            metrics=metrics,
            metric_sources=sources,
            warnings=list(dict.fromkeys(warnings)),
            error=error,
            needs_input=result.needs_input if result else None,
            memory_delta=delta,
            summary=result.summary if result else None,
        )

    def _blocked(self, item, state, code, message) -> SubAgentResult:
        return SubAgentResult(
            subtask_id=item.id,
            run_id=state["run_ids"][item.id],
            status=AgentResultStatus.BLOCKED,
            execution_status=TaskStatus.BLOCKED,
            error=ToolError(code=code, category=ErrorCategory.EXECUTION, message=message),
        )

    async def _persist_result(self, state, result: SubAgentResult) -> None:
        child = self.store.get_run(result.run_id)
        raw = child.metadata.get("result") or {}
        agent_result = AgentResult(
            agent_id=child.agent_id,
            task_id=child.task_id,
            status=result.status,
            summary=result.summary or (result.error.message if result.error else "子任务已结束。"),
            datasets=[ref.dataset_id for ref in result.datasets],
            artifacts=result.artifact_ids,
            error=result.error.code if result.error else None,
            needs_input=result.needs_input,
            trace_id=child.id,
        )
        # 保留等待状态的确切错误码，防止历史工具错误将等待转换为失败。
        if result.execution_status is TaskStatus.WAITING:
            agent_result = agent_result.model_copy(update={"error": raw.get("error")})
        persist_result(
            self.store,
            child,
            None,
            agent_result,
            metadata={"subagent_result": result.model_dump(mode="json")},
        )
        checkpoint = self.store.latest_checkpoint(child.id)
        if checkpoint is not None:
            self.store.save_checkpoint(
                Checkpoint(
                    run_id=child.id,
                    phase=checkpoint.phase,
                    state={**checkpoint.state, "result": agent_result.model_dump(mode="json")},
                )
            )
        subtask = self.store.get_subtask(state["task_id"], f"{state['id']}:{result.subtask_id}")
        self.store.save_subtask(
            state["task_id"], subtask.model_copy(update={"status": result.execution_status})
        )
        state["results"][result.subtask_id] = result.model_dump(mode="json")
        self.store.save_delegation(state)
        await self.loop.trace.emit(
            state["parent_run_id"],
            EventType.SUBAGENT_COMPLETED,
            "子任务结构化状态已持久化",
            payload={
                "subtask_id": result.subtask_id,
                "run_id": result.run_id,
                "status": result.status.value,
                "execution_status": result.execution_status.value,
                "error": result.error.code if result.error else None,
            },
            agent_id=child.agent_id,
        )

    def _aggregate(self, state, plan, results) -> DelegationResult:
        ordered = []
        for item in plan.subtasks:
            result = results.get(item.id)
            if result is None:
                result = self._blocked(item, state, "DEPENDENCY_WAITING", "等待上游子任务恢复。")
                result = result.model_copy(update={"execution_status": TaskStatus.PENDING})
            ordered.append(result)
        failures = [
            item.subtask_id
            for item in ordered
            if item.execution_status in {TaskStatus.FAILED, TaskStatus.CANCELLED}
        ]
        blocked = [
            item.subtask_id
            for item in ordered
            if item.execution_status in {TaskStatus.BLOCKED, TaskStatus.PENDING}
        ]
        required = {item.id for item in plan.subtasks if item.required}
        if any(item.execution_status is TaskStatus.WAITING for item in ordered):
            status = AgentResultStatus.BLOCKED
        elif required.intersection(failures + blocked):
            status = AgentResultStatus.FAILED
        elif (
            failures or blocked or any(item.status is AgentResultStatus.PARTIAL for item in ordered)
        ):
            status = AgentResultStatus.PARTIAL
        else:
            status = AgentResultStatus.SUCCESS
        return DelegationResult(
            delegation_id=state["id"],
            parent_run_id=state["parent_run_id"],
            status=status,
            subtasks=ordered,
            added_dataset_ids=list(
                dict.fromkeys(ref.dataset_id for item in ordered for ref in item.datasets)
            ),
            added_artifact_ids=list(
                dict.fromkeys(ref for item in ordered for ref in item.artifact_ids)
            ),
            failed_subtask_ids=failures,
            blocked_subtask_ids=blocked,
        )

    def _validate_saved_references(self, results, user_id) -> None:
        for item in results.values():
            for ref in item.datasets:
                dataset = self.store.get_dataset_for_user(ref.dataset_id, user_id)
                if dataset is None or dataset.created_by_run_id != item.run_id:
                    raise ValueError("已保存子输出不存在、权限变更或来源不匹配")
            for identifier in item.artifact_ids:
                artifact = self.store.get_artifact_for_user(identifier, user_id)
                if artifact is None or artifact.run_id != item.run_id:
                    raise ValueError("已保存产物不存在、权限变更或来源不匹配")

    async def record_stopped(self, parent_run_id: str) -> None:
        """执行清理结束后保存所有子任务终态；已取消委派保持终态，不技术重跑。"""

        for state in self.store.list_delegations(parent_run_id):
            if state.get("completed"):
                continue
            plan = DelegationPlan.model_validate(state["plan"])
            user_id = self.store.user_id_for_run(parent_run_id)
            results = {}
            for item in plan.subtasks:
                result = self._collect_result(
                    item, self.store.get_run(state["run_ids"][item.id]), user_id
                )
                await self._persist_result(state, result)
                results[item.id] = result
            outcome = self._aggregate(state, plan, results).model_copy(
                update={"status": AgentResultStatus.CANCELLED}
            )
            memory = self.updater.load_or_create(state["task_id"])
            merged = self.updater.merge_deltas(
                memory, [item.memory_delta for item in results.values()]
            )
            state.update(completed=True, merged=True, result=outcome.model_dump(mode="json"))
            self.store.save_delegation(state, memory=merged)

    @staticmethod
    def _observation(call_id, result: DelegationResult) -> ToolResult:
        payload = result.model_dump(mode="json")
        for item in payload["subtasks"]:
            item.pop("memory_delta")
            item.pop("metric_sources")
            item["metrics"] = dict(list(item["metrics"].items())[:8])
            item["warnings"] = [text[:200] for text in item["warnings"][:4]]
            item["summary"] = (item["summary"] or "")[:300]
            for reference in item["datasets"]:
                reference.pop(
                    "source_dataset_ids"
                )  # lineage 保存在完整结果，模型仅需要 ID 和角色。
            if item["error"]:
                item["error"]["details"] = {}
                item["error"]["message"] = item["error"]["message"][:300]
        if len(json.dumps(payload, ensure_ascii=False)) > 7000:
            for item in payload["subtasks"]:
                item["metrics"] = {}
                item["summary"] = ""
                item["warnings"] = []
        waiting = next(
            (item for item in result.subtasks if item.execution_status is TaskStatus.WAITING), None
        )
        error = None
        if waiting:
            needs = waiting.needs_input or {}
            code = "APPROVAL_REQUIRED" if needs.get("approval_id") else "WAITING_USER"
            error = ToolError(
                code=code,
                message=(needs.get("question") or "子任务正在等待审批。"),
                details={
                    **needs,
                    "child_run_id": waiting.run_id,
                    "delegation_id": result.delegation_id,
                },
            )
        tool_status = {
            AgentResultStatus.SUCCESS: ToolStatus.SUCCESS,
            AgentResultStatus.PARTIAL: ToolStatus.PARTIAL_SUCCESS,
            AgentResultStatus.BLOCKED: ToolStatus.BLOCKED,
            AgentResultStatus.FAILED: ToolStatus.FAILED,
            AgentResultStatus.CANCELLED: ToolStatus.CANCELLED,
        }[result.status]
        return ToolResult(
            call_id=call_id,
            status=tool_status,
            output=payload,
            error=error,
            datasets=result.added_dataset_ids,
            artifacts=result.added_artifact_ids,
        )


def _fingerprint(plan: DelegationPlan) -> str:
    return hashlib.sha256(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _metrics(output: Any) -> dict[str, Any]:
    # 数组、几何、完整 Dataset/记录均不进入 metrics；每项附带真实 call.id。
    if not isinstance(output, dict):
        return {}
    return {
        str(key)[:80]: value
        for key, value in list(output.items())[:32]
        if isinstance(value, (bool, int, float))
        and (not isinstance(value, float) or math.isfinite(value))
    }


__all__ = ["DELEGATE_TOOL", "DelegationCoordinator"]
