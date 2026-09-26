"""GeoAgent 的领域模型。

这些模型描述 GIS Agent 的事实边界：请求、任务、运行、工具结果、数据集、
产物和追踪事件。执行器只接受/返回这些结构化对象，避免把异常字符串直接
当成成功结果交给上层 Agent。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def new_id(prefix: str) -> str:
    """生成可读且不会依赖数据库自增的领域 ID。"""

    return f"{prefix}_{uuid4().hex[:12]}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=(), populate_by_name=True, serialize_by_alias=True)


class DatasetKind(StrEnum):
    VECTOR = "VECTOR"
    RASTER = "RASTER"
    TABLE = "TABLE"
    POINT_CLOUD = "POINT_CLOUD"
    TRAJECTORY = "TRAJECTORY"
    NETWORK = "NETWORK"
    SERVICE = "SERVICE"


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class RunStatus(StrEnum):
    """运行状态；规划会话使用 PlanningStatus，不混入 Run 生命周期。"""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_TOOL = "WAITING_TOOL"
    WAITING_SUBAGENT = "WAITING_SUBAGENT"
    WAITING_USER = "WAITING_USER"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    PARTIAL_COMPLETED = "PARTIAL_COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class ResponseStyle(StrEnum):
    CONCISE = "concise"
    BALANCED = "balanced"
    DETAILED = "detailed"


class MeasurementSystem(StrEnum):
    METRIC = "metric"
    IMPERIAL = "imperial"


class ToolStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class ToolExecutionStatus(StrEnum):
    """ToolCall 的持久化执行状态，与单次返回结果状态分离。"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class DatasetOutputPolicy(StrEnum):
    """Tool 对 Dataset 输出的契约强度。"""

    NONE = "NONE"
    OPTIONAL = "OPTIONAL"
    REQUIRED = "REQUIRED"


class ErrorCategory(StrEnum):
    INPUT = "INPUT"
    CRS = "CRS"
    GEOMETRY = "GEOMETRY"
    DATA = "DATA"
    RASTER = "RASTER"
    RESOURCE = "RESOURCE"
    PERMISSION = "PERMISSION"
    EXECUTION = "EXECUTION"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


class RiskLevel(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"
    EXTERNAL = "EXTERNAL"


class ApprovalStatus(StrEnum):
    """一次性工具审批的生命周期。"""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    CONSUMED = "CONSUMED"
    EXPIRED = "EXPIRED"


class AgentResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class ArtifactKind(StrEnum):
    DATASET = "DATASET"
    MAP = "MAP"
    REPORT = "REPORT"
    TABLE = "TABLE"
    LOG = "LOG"
    OTHER = "OTHER"


class CRSInfo(StrictModel):
    authority: str | None = None
    name: str | None = None
    is_geographic: bool = False
    linear_unit: str | None = None


class BoundingBox(StrictModel):
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @field_validator("max_x")
    @classmethod
    def max_x_not_before_min_x(cls, value: float, info: Any) -> float:
        if "min_x" in info.data and value < info.data["min_x"]:
            raise ValueError("max_x must be >= min_x")
        return value

    @field_validator("max_y")
    @classmethod
    def max_y_not_before_min_y(cls, value: float, info: Any) -> float:
        if "min_y" in info.data and value < info.data["min_y"]:
            raise ValueError("max_y must be >= min_y")
        return value


class DatasetSchema(StrictModel):
    fields: dict[str, str] = Field(default_factory=dict)
    geometry_type: str | None = None
    feature_count: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    bands: int | None = Field(default=None, ge=0)
    resolution: tuple[float, float] | None = None
    nodata: float | int | None = None
    invalid_geometry_count: int | None = Field(default=None, ge=0)


class Dataset(StrictModel):
    id: str = Field(default_factory=lambda: new_id("ds"))
    name: str
    kind: DatasetKind
    path: str
    format: str
    crs: CRSInfo | None = None
    extent: BoundingBox | None = None
    schema_: DatasetSchema | None = Field(default=None, alias="schema")
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_dataset_ids: list[str] = Field(default_factory=list)
    created_by_run_id: str | None = None
    owner_user_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("name", "path", "format")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("dataset text fields cannot be empty")
        return value

    @property
    def schema(self) -> DatasetSchema | None:
        return self.schema_

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(*args, **kwargs)


class Artifact(StrictModel):
    id: str = Field(default_factory=lambda: new_id("art"))
    name: str
    kind: ArtifactKind
    path: str | None = None
    media_type: str | None = None
    dataset_id: str | None = None
    run_id: str | None = None
    owner_user_id: str | None = None
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AgentRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: new_id("req"))
    conversation_id: str = Field(default_factory=lambda: new_id("conv"))
    user_id: str | None = None
    user_input: str
    dataset_ids: list[str] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)
    referenced_run_ids: list[str] = Field(default_factory=list)
    model_profile: str | None = None
    reply_to_run_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("user_input")
    @classmethod
    def non_empty_input(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_input cannot be empty")
        return value


class Task(StrictModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    subtasks: list[str] = Field(default_factory=list)
    result: str | None = None
    conversation_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SubTask(StrictModel):
    id: str = Field(default_factory=lambda: new_id("sub"))
    goal: str
    description: str
    operation: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    parallelizable: bool = True
    required: bool = True
    assigned_agent_id: str | None = None
    status: TaskStatus = TaskStatus.PENDING


class ToolMetadata(StrictModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    required_scopes: list[str] = Field(default_factory=list)
    required_envs: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.READ
    supports_retry: bool = False
    dataset_output_policy: DatasetOutputPolicy = DatasetOutputPolicy.NONE
    produces_artifact: bool = False
    tags: list[str] = Field(default_factory=list)

class ToolCall(StrictModel):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    agent_id: str | None = None
    attempt: int = Field(default=1, ge=1)


class ToolError(StrictModel):
    code: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationIssue(StrictModel):
    code: str
    message: str
    recoverable: bool = False


class ToolResult(StrictModel):
    call_id: str
    status: ToolStatus
    output: Any = None
    error: ToolError | None = None
    warnings: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    retryable: bool = False
    duration_ms: float = Field(default=0.0, ge=0.0)


class AgentResult(StrictModel):
    agent_id: str
    task_id: str | None = None
    status: AgentResultStatus
    summary: str
    findings: list[Any] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    needs_input: dict[str, Any] | None = None
    trace_id: str


class Run(StrictModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    parent_run_id: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None
    agent_id: str
    status: RunStatus = RunStatus.CREATED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    turn_count: int = 0
    tool_call_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkingMemoryItem(StrictModel):
    """工作记忆中的轻量引用，不保存完整工具输出。"""

    kind: str
    reference_id: str | None = None
    summary: str
    source_run_id: str | None = None


class PendingQuestion(StrictModel):
    """与具体等待 Run 关联的待回答问题。"""

    id: str = Field(default_factory=lambda: new_id("question"))
    content: str
    source_run_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class WorkingMemory(StrictModel):
    """以 Task 为作用域的结构化工作状态。"""

    task_id: str
    conversation_id: str | None = None
    active_dataset_ids: list[str] = Field(default_factory=list)
    active_artifact_ids: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    intermediate_results: list[WorkingMemoryItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    pending_questions: list[PendingQuestion] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class ApprovalRequest(StrictModel):
    """绑定到单个用户、Run 和精确 Tool 参数的一次性审批。"""

    id: str = Field(default_factory=lambda: new_id("approval"))
    user_id: str
    conversation_id: str | None = None
    task_id: str | None = None
    source_run_id: str
    tool_call_id: str
    tool_name: str
    argument_fingerprint: str
    risk_level: RiskLevel
    argument_preview: dict[str, Any] = Field(default_factory=dict)
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None
    consumed_at: datetime | None = None
    continuation_run_id: str | None = None
    decision_note: str | None = None


class WorkingMemoryDelta(StrictModel):
    """SubAgent 本轮执行产生的局部工作状态变化，不直接持久化。"""

    added_dataset_ids: list[str] = Field(default_factory=list)
    added_artifact_ids: list[str] = Field(default_factory=list)
    intermediate_results: list[WorkingMemoryItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    source_run_id: str | None = None


class UpstreamDatasetBinding(StrictModel):
    from_subtask: str = Field(min_length=1, max_length=64)
    output_role: str | None = Field(default=None, max_length=64)
    input_name: str = Field(min_length=1, max_length=64)


class DelegationSubtask(StrictModel):
    """模型只能请求执行计划，不能设置子 Run 身份、状态或权限凭证。"""

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    goal: str = Field(min_length=1, max_length=2000)
    description: str = Field(default="", max_length=2000)
    dataset_ids: list[str] = Field(default_factory=list, max_length=32)
    dependencies: list[str] = Field(default_factory=list, max_length=20)
    allowed_tools: list[str] = Field(min_length=1, max_length=32)
    upstream_dataset_bindings: list[UpstreamDatasetBinding] = Field(default_factory=list, max_length=16)
    # 角色 -> 产出该角色的真实工具名；该角色必须恰好有一个经验证的输出。
    output_roles: dict[str, str] = Field(default_factory=dict, max_length=8)
    parallelizable: bool = True
    required: bool = True

    @field_validator("goal")
    @classmethod
    def non_empty_goal(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("子任务目标不能为空")
        return value.strip()

    @field_validator("dataset_ids", "dependencies", "allowed_tools")
    @classmethod
    def bounded_identifiers(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 160 for value in values):
            raise ValueError("引用名称不能为空或超过 160 字符")
        if len(set(values)) != len(values):
            raise ValueError("引用列表不能重复")
        return values


class DelegationPlan(StrictModel):
    subtasks: list[DelegationSubtask] = Field(min_length=1, max_length=20)


class DatasetOutputRef(StrictModel):
    dataset_id: str
    role: str | None = None
    source_dataset_ids: list[str] = Field(default_factory=list)


class SubAgentResult(StrictModel):
    subtask_id: str
    run_id: str
    status: AgentResultStatus
    execution_status: TaskStatus
    datasets: list[DatasetOutputRef] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metric_sources: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    needs_input: dict[str, Any] | None = None
    memory_delta: WorkingMemoryDelta = Field(default_factory=WorkingMemoryDelta)
    summary: str | None = None


class DelegationResult(StrictModel):
    delegation_id: str
    parent_run_id: str
    status: AgentResultStatus
    subtasks: list[SubAgentResult]
    added_dataset_ids: list[str] = Field(default_factory=list)
    added_artifact_ids: list[str] = Field(default_factory=list)
    failed_subtask_ids: list[str] = Field(default_factory=list)
    blocked_subtask_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Checkpoint(StrictModel):
    id: str = Field(default_factory=lambda: new_id("cp"))
    run_id: str
    phase: str
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class TraceEvent(StrictModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    event_type: str
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence: int = 0
    timestamp: datetime = Field(default_factory=utc_now)
    agent_id: str | None = None


class Conversation(StrictModel):
    id: str = Field(default_factory=lambda: new_id("conv"))
    title: str = "新对话"
    user_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Message(StrictModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    conversation_id: str
    role: str
    content: str
    run_id: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationMemoryEntry(StrictModel):
    """会话级派生事实，始终保留其来源引用。"""

    id: str = Field(default_factory=lambda: new_id("cmem"))
    content: str
    source_message_id: str | None = None
    source_task_id: str | None = None
    source_run_id: str | None = None
    reference_type: str | None = None
    reference_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ConversationMemory(StrictModel):
    """Conversation 作用域的结构化派生状态，不替代原始 Messages。"""

    conversation_id: str
    user_id: str
    summary: str = ""
    key_facts: list[ConversationMemoryEntry] = Field(default_factory=list)
    decisions: list[ConversationMemoryEntry] = Field(default_factory=list)
    important_references: list[ConversationMemoryEntry] = Field(default_factory=list)
    unresolved_topics: list[ConversationMemoryEntry] = Field(default_factory=list)
    summary_version: int = Field(default=0, ge=0)
    summarized_through_message_id: str | None = None
    summary_updated_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryItem(StrictModel):
    id: str = Field(default_factory=lambda: new_id("mem"))
    owner_user_id: str | None = None
    scope: str = "project"
    key: str
    value: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class User(StrictModel):
    """认证和资源归属使用的最小用户身份。"""

    id: str = Field(default_factory=lambda: new_id("user"))
    username: str
    email: str | None = None
    password_hash: str
    display_name: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class UserProfile(StrictModel):
    """跨会话的明确工作交互偏好，不保存用户画像或敏感信息。"""

    user_id: str
    language: str = "zh-CN"
    response_style: ResponseStyle = ResponseStyle.BALANCED
    measurement_system: MeasurementSystem = MeasurementSystem.METRIC
    preferred_output_format: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class UserView(StrictModel):
    """可返回给前端的安全用户视图，不包含密码哈希。"""

    id: str
    username: str
    email: str | None = None
    display_name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_user(cls, user: User) -> UserView:
        return cls.model_validate(user.model_dump(exclude={"password_hash"}))


class UserSession(StrictModel):
    """服务端 Session 记录；token 只以哈希形式持久化。"""

    id: str = Field(default_factory=lambda: new_id("session"))
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime | None = None


__all__ = [name for name in globals() if not name.startswith("_")]
