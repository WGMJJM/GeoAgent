"""GeoAgent composition root。

这里仅装配领域服务、状态、工具和 Agent，不把 GIS 业务逻辑藏在 FastAPI 路由中。
CLI、HTTP 和测试都通过同一个 Application 入口运行。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.loop import AgentLoop
from app.auth import ApprovalService, AuthService, PermissionPolicy
from app.config import Settings
from app.core.models import AgentRequest, AgentResult, AgentResultStatus, new_id
from app.entry import AttachmentService, ConversationService, MessageGateway
from app.execution.python import PythonExecutor
from app.execution.sandbox import WorkspaceManager
from app.execution.shell import ShellExecutor
from app.execution.tools import ToolExecutor, ToolRegistry
from app.gis.crs.service import CRSService
from app.gis.dataset import DatasetInspector, DatasetRegistry
from app.gis.preview import DatasetPreviewService
from app.gis.raster import RasterService
from app.gis.vector import VectorService
from app.gis.visualization import MapRenderer
from app.memory import (
    ConversationMemoryService,
    MemoryManager,
    UserProfileService,
)
from app.models import ModelAdapter
from app.models.config import ModelProfile
from app.models.providers import OpenAICompatibleAdapter
from app.observability import EventBus, Metrics, TraceRecorder
from app.run import RunManager
from app.run.checkpoints import CheckpointStore
from app.state import StateStore
from app.tools.gis import register_gis_tools
from app.tools.runtime import register_runtime_tools


class Application:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.store = StateStore(self.settings.database_path)
        self.auth = AuthService(self.store, self.settings)
        self.approvals = ApprovalService(self.store)
        self.bus = EventBus()
        self.metrics = Metrics()
        self.trace = TraceRecorder(self.store, self.bus, self.metrics)
        self.workspace = WorkspaceManager(self.settings.workspace_path)
        self.attachments = AttachmentService(self.workspace)
        self.inspector = DatasetInspector()
        self.registry = DatasetRegistry(self.store, self.inspector, system_owned=True)
        self.vectors = VectorService()
        self.rasters = RasterService()
        self.crs = CRSService(self.vectors, default_crs=self.settings.default_crs)
        self.renderer = MapRenderer()
        self.dataset_preview = DatasetPreviewService()
        self.python_executor = PythonExecutor(self.workspace, timeout_seconds=self.settings.tool_timeout_seconds)
        self.shell_executor = ShellExecutor(self.workspace, timeout_seconds=self.settings.tool_timeout_seconds)
        self.checkpoints = CheckpointStore(self.store)
        self.memory = MemoryManager(self.store)
        self.profile = UserProfileService(self.store)
        self.conversation_memory = ConversationMemoryService(self.store, model_provider=self.get_model_adapter)
        self.model_profiles: dict[str, ModelProfile] = {}
        self.model_adapters: dict[str, ModelAdapter] = {}
        self.default_model_profile: str | None = None
        self.model_adapter: ModelAdapter | None = None
        self._model_config_source = "环境变量"
        self._load_model_profiles()
        self.tool_registry = ToolRegistry()
        register_gis_tools(self.tool_registry)
        register_runtime_tools(self.tool_registry)
        self.tool_executor = ToolExecutor(self.tool_registry, self.store, self.trace, PermissionPolicy(), timeout_seconds=self.settings.tool_timeout_seconds, metrics=self.metrics, approval_service=self.approvals)
        self.tool_executor.services = {
            "settings": self.settings,
            "store": self.store,
            "workspace": self.workspace,
            "registry": self.registry,
            "inspector": self.inspector,
            "vectors": self.vectors,
            "rasters": self.rasters,
            "crs": self.crs,
            "renderer": self.renderer,
            "python": self.python_executor,
            "shell": self.shell_executor,
            "allow_unsafe_python": self.settings.enable_unsafe_python,
            "system_owned": True,
        }
        self.agent_loop = AgentLoop(
            self.store,
            self.tool_registry,
            self.tool_executor,
            self.trace,
            self.settings,
            self.get_model_adapter,
            self.execution_services,
            context_services={
                "profile": self.profile,
                "project_memory": self.memory,
                "conversation_memory": self.conversation_memory,
            },
            metrics=self.metrics,
        )
        self.run_manager = RunManager(
            self.agent_loop,
            self.store,
            self.metrics,
            execution_timeout_seconds=self.settings.max_execution_seconds,
        )
        self.conversations = ConversationService(
            self.store,
            self.run_manager,
            on_assistant_message_persisted=self.conversation_memory.summarize_after_assistant_persisted,
        )
        self.message_entry = MessageGateway(self.store, self.conversations)

    def start(self) -> None:
        self.store.initialize()
        self.auth.bootstrap_if_configured()
        self.run_manager.reconcile_orphaned_runs()

    def execution_services(self, user_id: str | None = None) -> dict[str, object]:
        """为一次执行构造同一用户的 Registry、Workspace 和执行器。"""

        workspace = self.workspace.for_user(user_id)
        services = dict(self.tool_executor.services)
        services.update(
            {
                "workspace": workspace,
                "registry": self.registry.for_user(user_id, system_owned=user_id is None),
                "python": PythonExecutor(workspace, timeout_seconds=self.settings.tool_timeout_seconds),
                "shell": ShellExecutor(workspace, timeout_seconds=self.settings.tool_timeout_seconds),
                "user_id": user_id,
                "allow_unsafe_python": self.settings.enable_unsafe_python,
                "system_owned": user_id is None,
            }
        )
        return services

    def _load_model_profiles(self) -> None:
        profiles: list[ModelProfile] = []
        if self.settings.model_profiles:
            try:
                raw_profiles = json.loads(self.settings.model_profiles)
            except json.JSONDecodeError as exc:
                raise ValueError("GEOAGENT_MODEL_PROFILES 必须是有效的 JSON 数组。") from exc
            if not isinstance(raw_profiles, list):
                raise ValueError("GEOAGENT_MODEL_PROFILES 必须是 JSON 数组。")
            profiles = [ModelProfile.model_validate(item) for item in raw_profiles]
        for profile in profiles:
            if profile.id in self.model_profiles:
                raise ValueError(f"模型配置的编号重复：{profile.id}")
            self.model_profiles[profile.id] = profile
            adapter = OpenAICompatibleAdapter(profile.as_config())
            self.model_adapters[profile.id] = adapter

        if profiles:
            selected = next((item for item in profiles if item.default), profiles[0])
            self.default_model_profile = selected.id
            self.model_adapter = self.model_adapters[selected.id]

    async def close(self) -> None:
        adapters = list(self.model_adapters.values())
        if self.model_adapter is not None and all(id(self.model_adapter) != id(item) for item in adapters):
            adapters.append(self.model_adapter)
        closed: set[int] = set()
        for adapter in adapters:
            if id(adapter) not in closed:
                await adapter.close()
                closed.add(id(adapter))

    def model_status(self) -> dict[str, object]:
        """返回脱敏后的模型连接状态，绝不把 API Key 返回给前端。"""

        profiles = [
            {
                "id": profile.id,
                "label": profile.label,
                "provider": profile.provider,
                "base_url": profile.base_url,
                "model": profile.model,
                "timeout_seconds": profile.timeout_seconds,
                "temperature": profile.temperature,
                "supports_stream": profile.supports_stream,
                "supports_tools": profile.supports_tools,
                "supports_json_object": profile.supports_json_object,
                "supports_json_schema": profile.supports_json_schema,
                "has_api_key": bool(profile.api_key),
                "default": profile.id == self.default_model_profile,
            }
            for profile in self.model_profiles.values()
        ]
        return {
            "configured": bool(self.model_adapters),
            "source": self._model_config_source if self.model_profiles else "未配置",
            "default_profile": self.default_model_profile,
            "profiles": profiles,
        }

    def get_model_adapter(self, profile_id: str | None = None) -> ModelAdapter | None:
        """按请求选择规划或对话模型；不返回任何未配置的模型。"""

        if profile_id:
            return self.model_adapters.get(profile_id)
        if self.model_adapter is not None:
            return self.model_adapter
        # 保持运行时注入模型与统一入口使用同一模型；正式配置仍优先使用
        # Application 的 profile adapter。
        return self.agent_loop.model_adapter if hasattr(self, "agent_loop") else None

    async def ask(self, user_input: str | AgentRequest, *, conversation_id: str | None = None, dataset_ids: list[str] | None = None, model_profile: str | None = None):
        request = user_input if isinstance(user_input, AgentRequest) else AgentRequest(
            user_input=user_input,
            conversation_id=conversation_id or new_id("conv"),
            dataset_ids=[item for item in dataset_ids or () if item],
            model_profile=model_profile,
        )
        response = await self.message_entry.submit(request)
        if response.result is not None:
            return response.result
        return AgentResult(
            agent_id="entry",
            status=AgentResultStatus.SUCCESS,
            summary=response.message,
            trace_id=f"direct_{request.request_id}",
        )

    def register_dataset(self, path: str | Path, *, name: str | None = None, user_id: str | None = None):
        workspace = self.workspace.for_user(user_id)
        target = workspace.resolve(path, allow_missing=False)
        return self.registry.for_user(user_id, system_owned=user_id is None).register_path(target, name=name, system_owned=user_id is None)
