"""FastAPI 接口：Chat、Dataset、Run、Trace 和 Artifact。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from app.application import Application
from app.core.models import (
    AgentRequest,
    ApprovalStatus,
    RunStatus,
    User,
    UserView,
    new_id,
)
from app.gis.preview import DatasetPreview
from app.run.lifecycle import record_approval_decision
from app.run.predicates import is_cancellable_run, is_resumable_run
from app.run.checkpoints import RunCheckpointCodec


class AskBody(BaseModel):
    message: str
    conversation_id: str | None = None
    model_profile: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)
    referenced_run_ids: list[str] = Field(default_factory=list)
    reply_to_run_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ConversationBody(BaseModel):
    title: str = "新对话"


class DatasetBody(BaseModel):
    path: str
    name: str | None = None


class MemoryBody(BaseModel):
    key: str
    value: str
    scope: str = "project"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunDeleteBody(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


class ApprovalDecisionBody(BaseModel):
    note: str | None = None


class RegisterBody(BaseModel):
    username: str
    password: str
    email: str | None = None
    display_name: str | None = None


class LoginBody(BaseModel):
    identifier: str | None = None
    username: str | None = None
    email: str | None = None
    password: str


class UserUpdateBody(BaseModel):
    display_name: str | None = None
    email: str | None = None


class UserProfileUpdateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    response_style: str | None = None
    measurement_system: str | None = None
    preferred_output_format: str | None = None


def get_current_user(request: Request) -> User:
    geoagent = request.app.state.geoagent
    user = geoagent.auth.authenticate_token(request.cookies.get(geoagent.settings.auth_cookie_name))
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def _set_session_cookie(response: Response, geoagent: Application, token: str) -> None:
    response.set_cookie(
        geoagent.settings.auth_cookie_name,
        token,
        httponly=True,
        secure=geoagent.settings.auth_cookie_secure,
        samesite="lax",
        max_age=geoagent.settings.auth_session_ttl_hours * 3600,
        path="/",
    )


def create_app(application: Application | None = None) -> FastAPI:
    geoagent = application or Application()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        geoagent.start()
        yield
        await geoagent.close()

    api = FastAPI(title="GeoAgent API", version="0.1.0", lifespan=lifespan)
    api.state.geoagent = geoagent

    @api.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "geoagent", "tools": len(geoagent.tool_registry.names()), "model_configured": bool(geoagent.model_adapters)}

    @api.post("/api/v1/auth/register")
    async def register(body: RegisterBody, response: Response) -> dict[str, Any]:
        try:
            user = geoagent.auth.register(body.username, body.password, email=body.email, display_name=body.display_name)
            _, token = geoagent.auth.login(body.username, body.password)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _set_session_cookie(response, geoagent, token)
        return UserView.from_user(user).model_dump(mode="json")

    @api.post("/api/v1/auth/login")
    async def login(body: LoginBody, response: Response) -> dict[str, Any]:
        try:
            identifier = body.identifier or body.username or body.email or ""
            user, token = geoagent.auth.login(identifier, body.password)
        except Exception as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        _set_session_cookie(response, geoagent, token)
        return UserView.from_user(user).model_dump(mode="json")

    @api.post("/api/v1/auth/logout")
    async def logout(request: Request, response: Response) -> dict[str, bool]:
        geoagent.auth.logout(request.cookies.get(geoagent.settings.auth_cookie_name))
        response.delete_cookie(geoagent.settings.auth_cookie_name, path="/")
        return {"logged_out": True}

    @api.get("/api/v1/users/me")
    async def current_user(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        return UserView.from_user(current_user).model_dump(mode="json")

    @api.patch("/api/v1/users/me")
    async def update_current_user(body: UserUpdateBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        try:
            updated = geoagent.auth.update_user(current_user, display_name=body.display_name or current_user.display_name, email=body.email)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UserView.from_user(updated).model_dump(mode="json")

    @api.get("/api/v1/users/me/profile")
    async def current_profile(current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        return geoagent.profile.get_or_create(current_user.id).model_dump(mode="json")

    @api.patch("/api/v1/users/me/profile")
    async def update_current_profile(body: UserProfileUpdateBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        try:
            profile = geoagent.profile.update(current_user.id, body.model_dump(exclude_unset=True))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return profile.model_dump(mode="json")

    @api.get("/api/v1/models")
    async def model_status(_: User = Depends(get_current_user)) -> dict[str, object]:
        return geoagent.model_status()

    @api.get("/api/v1/metrics")
    async def metrics(_: User = Depends(get_current_user)) -> dict[str, int | float]:
        return geoagent.metrics.snapshot()

    @api.get("/api/v1/tools")
    async def tools(_: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.tool_registry.definitions()]

    @api.get("/api/v1/datasets")
    async def datasets(current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.registry.list(user_id=current_user.id)]

    @api.post("/api/v1/datasets")
    async def register_dataset(body: DatasetBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        try:
            dataset = geoagent.register_dataset(body.path, name=body.name, user_id=current_user.id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return dataset.model_dump(mode="json")

    @api.post("/api/v1/attachments")
    async def upload_attachment(file: UploadFile = File(...), current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        """接收用户文件，保存到当前用户 workspace/input 并立即登记为 Dataset。"""
        filename = file.filename or ""
        try:
            content = await file.read()
            path = geoagent.attachments.accept(filename, content, user_id=current_user.id)
            dataset = geoagent.registry.for_user(current_user.id).register_path(path, name=Path(filename).stem or path.stem)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "attachment_id": dataset.id,
            "dataset": dataset.model_dump(mode="json"),
        }

    @api.get("/api/v1/datasets/{dataset_id}/lineage")
    async def dataset_lineage(dataset_id: str, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if geoagent.registry.get(dataset_id, user_id=current_user.id) is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        return geoagent.store.list_lineage(dataset_id)

    @api.get("/api/v1/datasets/{dataset_id}/preview", response_model=DatasetPreview)
    async def dataset_preview(dataset_id: str, current_user: User = Depends(get_current_user)) -> DatasetPreview:
        dataset = geoagent.registry.get(dataset_id, user_id=current_user.id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        try:
            return geoagent.dataset_preview.preview(
                dataset,
                geoagent.workspace.for_user(current_user.id),
                max_features=geoagent.settings.max_preview_features,
                max_fields=geoagent.settings.max_preview_fields,
                max_property_length=geoagent.settings.max_preview_property_length,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="dataset file not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail="数据预览失败，文件格式或内容不可读取") from exc

    @api.post("/api/v1/messages")
    async def message(body: AskBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        _ensure_body_conversation_access(geoagent, body, current_user)
        try:
            response = await geoagent.message_entry.submit(_request_from_body(body, user_id=current_user.id))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.model_dump(mode="json")

    @api.get("/api/v1/conversations")
    async def conversations(limit: int = 50, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.conversations.list(limit, user_id=current_user.id)]

    @api.post("/api/v1/conversations")
    async def create_conversation(body: ConversationBody | None = None, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        title = body.title.strip() if body and body.title.strip() else "新对话"
        return geoagent.conversations.create(title, user_id=current_user.id).model_dump(mode="json")

    @api.delete("/api/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str, current_user: User = Depends(get_current_user)) -> dict[str, bool]:
        deleted = await geoagent.conversations.delete(conversation_id, user_id=current_user.id)
        if not deleted:
            raise HTTPException(status_code=404, detail="conversation not found")
        return {"deleted": True}

    @api.get("/api/v1/conversations/{conversation_id}/messages")
    async def conversation_messages(conversation_id: str, limit: int = 100, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if geoagent.store.get_conversation_for_user(conversation_id, current_user.id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return [item.model_dump(mode="json") for item in geoagent.store.list_messages(conversation_id, limit)]

    @api.get("/api/v1/runs")
    async def runs(limit: int = 50, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.store.list_runs(limit, user_id=current_user.id)]

    @api.get("/api/v1/runs/{run_id}")
    async def run(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        item = geoagent.store.get_run(run_id) if geoagent.store.run_belongs_to_user(run_id, current_user.id) else None
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        return item.model_dump(mode="json")

    @api.delete("/api/v1/runs")
    async def delete_runs(body: RunDeleteBody, current_user: User = Depends(get_current_user)) -> dict[str, list[str]]:
        requested = list(dict.fromkeys(run_id.strip() for run_id in body.run_ids if run_id.strip()))
        if any(not geoagent.store.run_belongs_to_user(run_id, current_user.id) for run_id in requested):
            raise HTTPException(status_code=404, detail="run not found")
        active = [run_id for run_id in requested if (item := geoagent.store.get_run(run_id)) is not None and (geoagent.run_manager.is_active(run_id) or is_cancellable_run(item))]
        deleted = geoagent.store.delete_runs([run_id for run_id in requested if run_id not in active])
        for run_id in deleted:
            geoagent.run_manager.forget(run_id)
        return {"deleted": deleted, "skipped_active": active}

    @api.delete("/api/v1/runs/{run_id}")
    async def delete_run(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, bool]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        item = geoagent.store.get_run(run_id)
        if item is not None and (geoagent.run_manager.is_active(run_id) or is_cancellable_run(item)):
            raise HTTPException(status_code=409, detail="运行中的记录不能删除，请先取消运行")
        if not geoagent.store.delete_run(run_id):
            raise HTTPException(status_code=404, detail="run not found")
        geoagent.run_manager.forget(run_id)
        return {"deleted": True}

    @api.post("/api/v1/runs/{run_id}/cancel")
    async def cancel_run(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        if not await geoagent.run_manager.cancel(run_id):
            raise HTTPException(status_code=409, detail="run is not active")
        item = geoagent.store.get_run(run_id)
        return item.model_dump(mode="json") if item else {"id": run_id, "status": "CANCELLED"}

    @api.get("/api/v1/runs/{run_id}/events")
    async def events(run_id: str, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        return [item.model_dump(mode="json") for item in geoagent.store.list_events(run_id)]

    @api.get("/api/v1/artifacts")
    async def artifacts(run_id: str | None = None, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        if run_id is not None and not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        return [item.model_dump(mode="json") for item in geoagent.store.list_artifacts_for_user(current_user.id, run_id)]

    @api.get("/api/v1/artifacts/{artifact_id}/content")
    async def artifact_content(artifact_id: str, current_user: User = Depends(get_current_user)):
        artifact = geoagent.store.get_artifact_for_user(artifact_id, current_user.id)
        if artifact is None or not artifact.path:
            raise HTTPException(status_code=404, detail="artifact not found")
        try:
            workspace = geoagent.workspace.for_user(artifact.owner_user_id)
            path = workspace.resolve(artifact.path, allow_missing=False)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="artifact file not found") from exc
        return FileResponse(path, media_type=artifact.media_type, filename=artifact.name)

    @api.get("/api/v1/memories")
    async def memories(scope: str = "project", current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.memory.list(scope, user_id=current_user.id)]

    @api.post("/api/v1/memories")
    async def save_memory(body: MemoryBody, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        if not body.key.strip() or not body.scope.strip():
            raise HTTPException(status_code=400, detail="memory key and scope cannot be empty")
        item = geoagent.memory.set(body.key.strip(), body.value, scope=body.scope.strip(), metadata=body.metadata, user_id=current_user.id)
        return item.model_dump(mode="json")

    @api.get("/api/v1/runs/{run_id}/checkpoint")
    async def checkpoint(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        if not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        item = geoagent.checkpoints.latest(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="checkpoint not found")
        return item.model_dump(mode="json")

    @api.get("/api/v1/approvals")
    async def approvals(status: ApprovalStatus | None = None, limit: int = 50, current_user: User = Depends(get_current_user)) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in geoagent.approvals.list(current_user.id, status=status, limit=limit)]

    @api.get("/api/v1/approvals/{approval_id}")
    async def approval(approval_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        item = geoagent.approvals.get(approval_id, user_id=current_user.id)
        if item is None:
            raise HTTPException(status_code=404, detail="approval not found")
        return item.model_dump(mode="json")

    @api.post("/api/v1/approvals/{approval_id}/approve")
    async def approve_approval(approval_id: str, body: ApprovalDecisionBody | None = None, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        item = geoagent.approvals.get(approval_id, user_id=current_user.id)
        if item is None:
            raise HTTPException(status_code=404, detail="approval not found")
        if item.status is not ApprovalStatus.PENDING:
            raise HTTPException(status_code=409, detail="审批已经结束")
        source = geoagent.store.get_run(item.source_run_id)
        if source is None or geoagent.store.user_id_for_run(source.id) not in {None, current_user.id} or source.status is not RunStatus.WAITING_APPROVAL:
            raise HTTPException(status_code=409, detail="原运行当前不在等待审批状态")
        checkpoint = geoagent.checkpoints.latest(source.id)
        if checkpoint is None:
            raise HTTPException(status_code=409, detail="审批来源运行没有可恢复上下文")
        if RunCheckpointCodec.request(checkpoint.state) is None:
            raise HTTPException(status_code=409, detail="审批来源缺少原始请求")
        approved = geoagent.approvals.approve(approval_id, user_id=current_user.id, note=body.note if body else None, persist=False)
        if approved is None:
            raise HTTPException(status_code=404, detail="approval not found")
        if record_approval_decision(geoagent.store, approved) is None:
            raise HTTPException(status_code=409, detail="审批来源运行不存在")
        run = await geoagent.run_manager.continue_run(source.id, user_id=current_user.id, approval_id=approval_id, approved=True)
        await geoagent.trace.emit(source.id, "ApprovalGranted", "用户已批准工具执行", payload={"approval_id": approval_id, "tool": item.tool_name, "status": approved.status.value}, agent_id="main")
        result = await geoagent.conversations.wait(run.id, force_assistant=True)
        final_approval = geoagent.approvals.get(approval_id, user_id=current_user.id) or approved
        if final_approval.status is ApprovalStatus.APPROVED:
            final_approval = geoagent.approvals.expire_if_unconsumed(approval_id, user_id=current_user.id) or final_approval
        return {"approval": final_approval.model_dump(mode="json"), "run": run.model_dump(mode="json"), "result": result.model_dump(mode="json")}

    @api.post("/api/v1/approvals/{approval_id}/deny")
    async def deny_approval(approval_id: str, body: ApprovalDecisionBody | None = None, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        current = geoagent.approvals.get(approval_id, user_id=current_user.id)
        if current is None:
            raise HTTPException(status_code=404, detail="approval not found")
        if current.status is not ApprovalStatus.PENDING:
            raise HTTPException(status_code=409, detail="审批已经结束")
        item = geoagent.approvals.deny(approval_id, user_id=current_user.id, note=body.note if body else None, persist=False)
        if item is None:
            raise HTTPException(status_code=404, detail="approval not found")
        if record_approval_decision(geoagent.store, item) is None:
            raise HTTPException(status_code=409, detail="审批来源运行不存在")
        run = await geoagent.run_manager.continue_run(item.source_run_id, user_id=current_user.id, approval_id=approval_id, approved=False)
        await geoagent.trace.emit(item.source_run_id, "ApprovalDenied", "用户拒绝了工具执行", payload={"approval_id": item.id, "tool": item.tool_name, "status": item.status.value}, agent_id="main")
        result = await geoagent.conversations.wait(run.id, force_assistant=True)
        return {"approval": item.model_dump(mode="json"), "run": run.model_dump(mode="json"), "result": result.model_dump(mode="json")}

    @api.post("/api/v1/runs/{run_id}/resume")
    async def resume(run_id: str, current_user: User = Depends(get_current_user)) -> dict[str, Any]:
        previous = geoagent.store.get_run(run_id)
        if previous is None or not geoagent.store.run_belongs_to_user(run_id, current_user.id):
            raise HTTPException(status_code=404, detail="run not found")
        checkpoint = geoagent.checkpoints.latest(run_id)
        if checkpoint is None:
            raise HTTPException(status_code=409, detail="run has no checkpoint")
        checkpoint_result = RunCheckpointCodec.result(checkpoint.state)
        if previous.status in {RunStatus.COMPLETED, RunStatus.PARTIAL_COMPLETED} and checkpoint.phase == "run_completed" and checkpoint_result is not None:
            result = checkpoint_result
            return {"resumed_from": run_id, "run_id": run_id, "checkpoint": checkpoint.id, "result": result.model_dump(mode="json")}
        if previous.status is RunStatus.WAITING_USER:
            raise HTTPException(status_code=409, detail="该运行正在等待用户补充信息，请发送新的消息继续任务")
        if previous.status is RunStatus.WAITING_APPROVAL:
            raise HTTPException(status_code=409, detail="该运行正在等待审批，请先处理审批请求")
        if previous.status is RunStatus.FAILED:
            raise HTTPException(status_code=409, detail="业务失败应通过重试继续，不能直接技术恢复")
        if not is_resumable_run(previous, has_checkpoint=True):
            raise HTTPException(status_code=409, detail=f"运行状态 {previous.status.value} 不支持技术恢复")
        if RunCheckpointCodec.request(checkpoint.state) is None:
            raise HTTPException(status_code=409, detail="checkpoint 缺少原始请求")
        run = await geoagent.run_manager.continue_run(run_id, user_id=current_user.id, technical=True)
        result = await geoagent.conversations.wait(run.id, force_assistant=True)
        return {"resumed_from": run_id, "run_id": run.id, "checkpoint": checkpoint.id, "result": result.model_dump(mode="json")}

    @api.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        current_user = geoagent.auth.authenticate_token(websocket.cookies.get(geoagent.settings.auth_cookie_name))
        if current_user is None:
            await websocket.close(code=1008, reason="请先登录")
            return
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_json()
                if payload.get("type") != "ask":
                    await websocket.send_json({"type": "error", "message": "只支持 type=ask"})
                    continue
                request = AgentRequest(
                    user_input=payload.get("message", ""),
                    conversation_id=payload.get("conversation_id") or new_id("conv"),
                    user_id=current_user.id,
                    model_profile=payload.get("model_profile"),
                    dataset_ids=payload.get("dataset_ids", []),
                    attachment_ids=payload.get("attachment_ids", []),
                    referenced_run_ids=payload.get("referenced_run_ids", []),
                    reply_to_run_id=payload.get("reply_to_run_id"),
                    context=payload.get("context", {}),
                )
                event_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
                stream_run = None

                async def on_event(event) -> None:
                    if stream_run is not None and event.run_id == stream_run.id:
                        event_queue.put_nowait(("event", event))

                async def on_model_delta(content: str) -> None:
                    event_queue.put_nowait(("delta", content))

                async def on_run(item) -> None:
                    nonlocal stream_run
                    stream_run = item
                    event_queue.put_nowait(("run", item))

                geoagent.bus.subscribe(on_event)
                response_task = asyncio.create_task(
                    geoagent.message_entry.submit(
                        request,
                        on_run=on_run,
                        on_model_delta=on_model_delta,
                    )
                )

                async def emit_heartbeat() -> None:
                    while True:
                        await asyncio.sleep(15)
                        event_queue.put_nowait(("heartbeat", None))

                heartbeat_task = asyncio.create_task(emit_heartbeat())

                waiter = response_task
                event_waiter = asyncio.create_task(event_queue.get())
                try:
                    while True:
                        done, _ = await asyncio.wait((waiter, event_waiter), return_when=asyncio.FIRST_COMPLETED)
                        if event_waiter in done:
                            kind, item = event_waiter.result()
                            if kind == "run":
                                await websocket.send_json({"type": "run", "data": item.model_dump(mode="json")})
                            elif kind == "event":
                                await websocket.send_json({"type": "event", "data": item.model_dump(mode="json")})
                            elif kind == "heartbeat":
                                await websocket.send_json({"type": "heartbeat"})
                            else:
                                await websocket.send_json({"type": "delta", "content": item})
                            event_waiter = asyncio.create_task(event_queue.get())
                            continue
                        response = waiter.result()
                        while not event_queue.empty():
                            kind, item = event_queue.get_nowait()
                            if kind == "run":
                                await websocket.send_json({"type": "run", "data": item.model_dump(mode="json")})
                            elif kind == "event":
                                await websocket.send_json({"type": "event", "data": item.model_dump(mode="json")})
                            elif kind == "heartbeat":
                                await websocket.send_json({"type": "heartbeat"})
                            else:
                                await websocket.send_json({"type": "delta", "content": item})
                        await websocket.send_json({"type": "response", "data": response.model_dump(mode="json")})
                        break
                finally:
                    geoagent.bus.unsubscribe(on_event)
                    if not event_waiter.done():
                        event_waiter.cancel()
                    heartbeat_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat_task
                    if not waiter.done():
                        if stream_run is not None:
                            await geoagent.run_manager.cancel(stream_run.id)
                        if not response_task.done():
                            response_task.cancel()
        except WebSocketDisconnect:
            return

    return api


def _request_from_body(body: AskBody, *, conversation_id: str | None = None, user_id: str | None = None) -> AgentRequest:
    return AgentRequest(
        user_input=body.message,
        conversation_id=conversation_id or body.conversation_id or new_id("conv"),
        user_id=user_id,
        model_profile=body.model_profile,
        dataset_ids=body.dataset_ids,
        attachment_ids=body.attachment_ids,
        referenced_run_ids=body.referenced_run_ids,
        reply_to_run_id=body.reply_to_run_id,
        context=body.context,
    )


def _ensure_body_conversation_access(geoagent: Application, body: AskBody, user: User) -> None:
    if body.conversation_id and geoagent.store.get_conversation(body.conversation_id) is not None and geoagent.store.get_conversation_for_user(body.conversation_id, user.id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
