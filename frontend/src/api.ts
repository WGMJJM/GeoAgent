export type RunStatus = "CREATED" | "RUNNING" | "WAITING_TOOL" | "WAITING_SUBAGENT" | "WAITING_USER" | "WAITING_APPROVAL" | "RETRYING" | "COMPLETED" | "PARTIAL_COMPLETED" | "FAILED" | "INTERRUPTED" | "CANCELLED" | "BUDGET_EXCEEDED";
export type AgentResultStatus = "SUCCESS" | "PARTIAL" | "FAILED" | "BLOCKED" | "CANCELLED";
export type MessageRoute = "execution" | "waiting" | "blocked";
export type ApprovalStatus = "PENDING" | "APPROVED" | "DENIED" | "CONSUMED" | "EXPIRED";
export type RiskLevel = "READ" | "WRITE" | "DESTRUCTIVE" | "EXTERNAL" | string;

export type Dataset = { id: string; name: string; kind: string; path: string; format: string; crs?: { authority?: string; name?: string; is_geographic?: boolean; linear_unit?: string } | null; extent?: { min_x: number; min_y: number; max_x: number; max_y: number } | null; schema?: { feature_count?: number; fields?: Record<string, string>; geometry_type?: string; invalid_geometry_count?: number; width?: number; height?: number; bands?: number; resolution?: [number, number]; nodata?: number | null } | null; metadata?: Record<string, unknown>; source_dataset_ids?: string[]; created_by_run_id?: string | null; created_at?: string | null };
export type Result = { agent_id: string; task_id?: string | null; status: AgentResultStatus; summary: string; findings: unknown[]; datasets: string[]; artifacts: string[]; evidence: unknown[]; warnings: string[]; error?: string | null; trace_id: string };
export type RunMetadata = { request_id?: string; original_request?: string; original_request_message_id?: string | null; protocol_version?: number; last_continuation?: string; result?: Result; [key: string]: unknown };
export type Run = { id: string; parent_run_id?: string | null; conversation_id?: string | null; task_id?: string | null; agent_id: string; status: RunStatus; started_at?: string | null; finished_at?: string | null; error?: string | null; turn_count: number; tool_call_count: number; metadata: RunMetadata };
export type Event = { id: string; run_id: string; event_type: string; message: string; sequence: number; timestamp: string; payload: Record<string, unknown>; agent_id?: string | null };
export type Artifact = { id: string; name: string; kind: string; path?: string | null; media_type?: string | null; dataset_id?: string | null; run_id?: string | null; description: string; metadata: Record<string, unknown>; created_at?: string | null };
export type ResumeResponse = { resumed_from: string; run_id: string; checkpoint: string; result: Result };
export type Conversation = { id: string; title: string; created_at: string; updated_at: string };
export type User = { id: string; username: string; email?: string | null; display_name: string; is_active: boolean; created_at: string; updated_at: string };
export type ResponseStyle = "concise" | "balanced" | "detailed";
export type MeasurementSystem = "metric" | "imperial";
export type UserProfile = { user_id: string; language: string; response_style: ResponseStyle; measurement_system: MeasurementSystem; preferred_output_format?: string | null; updated_at: string };
export type ConversationMessage = { id: string; conversation_id: string; role: string; content: string; run_id?: string | null; created_at?: string };
export type ModelProfile = { id: string; label: string; provider: string; base_url?: string | null; model: string; timeout_seconds: number; temperature: number; supports_stream?: boolean; supports_tools?: boolean; supports_json_object?: boolean; supports_json_schema?: boolean; has_api_key: boolean; default: boolean };
export type ModelStatus = { configured: boolean; source: string; default_profile?: string | null; profiles: ModelProfile[] };
export type RunDetails = { run: Run; result: Result | null; events: Event[]; artifacts: Artifact[] };
export type MessageResponse = { request_id: string; route: MessageRoute; message: string; run?: Run | null; result?: Result | null };
export type ApprovalRequest = { id: string; user_id: string; conversation_id?: string | null; task_id?: string | null; source_run_id: string; tool_call_id: string; tool_name: string; argument_fingerprint: string; risk_level: RiskLevel; argument_preview: Record<string, unknown>; reason: string; status: ApprovalStatus; created_at: string; decided_at?: string | null; consumed_at?: string | null; continuation_run_id?: string | null; decision_note?: string | null };
export type ApprovalActionResponse = { approval: ApprovalRequest; run?: Run | null; result?: Result | null };
export type DatasetPreview = { dataset_id: string; kind: string; crs?: string | null; source_crs?: string | null; bbox?: number[] | null; feature_count?: number | null; truncated: boolean; geojson?: { type: string; features?: unknown[] } | null; width?: number | null; height?: number | null; bands?: number | null; resolution?: number[] | null; columns: string[]; rows: Record<string, unknown>[] };
export type DatasetLineage = { id: string; run_id?: string | null; operation: string; input_dataset_ids: string[]; output_dataset_id: string; tool_call_id?: string | null; parameters: Record<string, unknown>; created_at: string };

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const HTTP_TIMEOUT_MS = 15_000;
const UPLOAD_TIMEOUT_MS = 120_000;
const STREAM_INACTIVITY_TIMEOUT_MS = 60_000;

async function request<T>(url: string, init?: RequestInit, timeoutMs = HTTP_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { credentials: "include", headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) }, ...init, signal: init?.signal ?? controller.signal });
    const body = await response.text();
    if (!response.ok) {
      let message = body;
      try {
        const payload = JSON.parse(body) as { detail?: string };
        message = payload.detail ?? body;
      } catch { /* 非 JSON 错误直接使用响应文本。 */ }
      throw new ApiError(message || `请求失败（${response.status}）`, response.status);
    }
    return (body ? JSON.parse(body) : undefined) as T;
  } finally {
    window.clearTimeout(timeout);
  }
}

function websocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

async function uploadAttachment(file: File): Promise<Dataset> {
  const form = new FormData();
  form.append("file", file);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);
  try {
    const response = await fetch("/api/v1/attachments", { method: "POST", credentials: "include", body: form, signal: controller.signal });
    const body = await response.text();
    if (!response.ok) {
      let message = body;
      try {
        const payload = JSON.parse(body) as { detail?: string };
        message = payload.detail ?? body;
      } catch { /* 非 JSON 错误直接使用响应文本。 */ }
      throw new ApiError(message || `文件上传失败（${response.status}）`, response.status);
    }
    const payload = JSON.parse(body) as { dataset: Dataset };
    return payload.dataset;
  } finally {
    window.clearTimeout(timeout);
  }
}

function streamMessage(message: string, datasetIds: string[], attachmentIds: string[], onRun: (run: Run) => void, onEvent: (event: Event) => void, onDelta: (content: string) => void, conversationId?: string, modelProfile?: string, onProgress?: () => void, replyToRunId?: string) {
  let cancelRequest = () => undefined;
  const promise = new Promise<MessageResponse>((resolve, reject) => {
    let socket: WebSocket | null = null;
    let settled = false;
    let watchdog: number | undefined;
    const clearWatchdog = () => {
      if (watchdog !== undefined) window.clearTimeout(watchdog);
      watchdog = undefined;
    };
    const touchConnection = () => {
      if (settled) return;
      clearWatchdog();
      watchdog = window.setTimeout(() => fail(new Error("GeoAgent 实时通道长时间无响应")), STREAM_INACTIVITY_TIMEOUT_MS);
    };
    const markExecutionProgress = () => {
      if (!settled) onProgress?.();
    };
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      clearWatchdog();
      socket?.close();
      reject(error);
    };
    cancelRequest = () => {
      if (settled) return;
      settled = true;
      clearWatchdog();
      socket?.close();
      reject(new Error("请求已取消"));
    };
    socket = new WebSocket(websocketUrl());
    touchConnection();
    socket.onopen = () => {
      touchConnection();
      try {
        socket.send(JSON.stringify({ type: "ask", message, conversation_id: conversationId, model_profile: modelProfile || undefined, dataset_ids: datasetIds, attachment_ids: attachmentIds, reply_to_run_id: replyToRunId || undefined }));
      } catch (error) {
        fail(error instanceof Error ? error : new Error("无法发送 GeoAgent 请求"));
      }
    };
    socket.onmessage = (raw) => {
      try {
        // heartbeat 只刷新连接 watchdog，不进入 execution progress。
        touchConnection();
        const payload = JSON.parse(raw.data as string) as { type: string; data?: Run | Event | MessageResponse; content?: string; message?: string };
        if (payload.type === "run") { markExecutionProgress(); onRun(payload.data as Run); }
        else if (payload.type === "event") { markExecutionProgress(); onEvent(payload.data as Event); }
        else if (payload.type === "delta") { markExecutionProgress(); onDelta(payload.content ?? ""); }
        else if (payload.type === "heartbeat") return;
        else if (payload.type === "response") {
          markExecutionProgress();
          settled = true;
          clearWatchdog();
          resolve(payload.data as MessageResponse);
          socket?.close();
        } else if (payload.type === "error") fail(new Error(payload.message ?? "实时请求失败"));
      } catch (error) {
        fail(error instanceof Error ? error : new Error("GeoAgent 返回了无效的实时消息"));
      }
    };
    socket.onerror = () => fail(new Error("无法连接 GeoAgent 实时通道"));
    socket.onclose = () => fail(new Error("GeoAgent 实时通道已断开"));
  });
  return Object.assign(promise, { cancel: () => cancelRequest() });
}

export const api = {
  me: () => request<User>("/api/v1/users/me"),
  register: (username: string, password: string, displayName: string, email?: string) => request<User>("/api/v1/auth/register", { method: "POST", body: JSON.stringify({ username, password, display_name: displayName, email: email || undefined }) }),
  login: (identifier: string, password: string) => request<User>("/api/v1/auth/login", { method: "POST", body: JSON.stringify({ identifier, password }) }),
  logout: () => request<{ logged_out: boolean }>("/api/v1/auth/logout", { method: "POST" }),
  updateMe: (displayName: string, email?: string) => request<User>("/api/v1/users/me", { method: "PATCH", body: JSON.stringify({ display_name: displayName, email: email || null }) }),
  profile: () => request<UserProfile>("/api/v1/users/me/profile"),
  updateProfile: (changes: Partial<Omit<UserProfile, "user_id" | "updated_at">>) => request<UserProfile>("/api/v1/users/me/profile", { method: "PATCH", body: JSON.stringify(changes) }),
  datasets: () => request<Dataset[]>("/api/v1/datasets"),
  uploadAttachment,
  conversations: (limit = 50) => request<Conversation[]>(`/api/v1/conversations?limit=${limit}`),
  createConversation: (title = "新对话") => request<Conversation>("/api/v1/conversations", { method: "POST", body: JSON.stringify({ title }) }),
  deleteConversation: (conversationId: string) => request<{ deleted: boolean }>(`/api/v1/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" }),
  messages: (conversationId: string) => request<ConversationMessage[]>(`/api/v1/conversations/${encodeURIComponent(conversationId)}/messages`),
  registerDataset: (path: string, name?: string) => request<Dataset>("/api/v1/datasets", { method: "POST", body: JSON.stringify({ path, name: name || undefined }) }),
  runs: () => request<Run[]>("/api/v1/runs"),
  run: (runId: string) => request<Run>(`/api/v1/runs/${runId}`),
  deleteRun: (runId: string) => request<{ deleted: boolean }>(`/api/v1/runs/${encodeURIComponent(runId)}`, { method: "DELETE" }),
  deleteRuns: (runIds: string[]) => request<{ deleted: string[]; skipped_active: string[] }>("/api/v1/runs", { method: "DELETE", body: JSON.stringify({ run_ids: runIds }) }),
  events: (runId: string) => request<Event[]>(`/api/v1/runs/${runId}/events`),
  artifacts: (runId?: string) => request<Artifact[]>(runId ? `/api/v1/artifacts?run_id=${encodeURIComponent(runId)}` : "/api/v1/artifacts"),
  artifactUrl: (artifactId: string) => `/api/v1/artifacts/${encodeURIComponent(artifactId)}/content`,
  datasetPreview: (datasetId: string) => request<DatasetPreview>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/preview`),
  datasetLineage: (datasetId: string) => request<DatasetLineage[]>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/lineage`),
  streamMessage,
  cancelRun: (runId: string) => request<Run>(`/api/v1/runs/${runId}/cancel`, { method: "POST" }),
  resumeRun: (runId: string) => request<ResumeResponse>(`/api/v1/runs/${runId}/resume`, { method: "POST" }),
  modelStatus: () => request<ModelStatus>("/api/v1/models"),
  approvals: (status?: ApprovalStatus, limit = 50) => request<ApprovalRequest[]>(`/api/v1/approvals?limit=${limit}${status ? `&status=${encodeURIComponent(status)}` : ""}`),
  approval: (approvalId: string) => request<ApprovalRequest>(`/api/v1/approvals/${encodeURIComponent(approvalId)}`),
  approve: (approvalId: string, note?: string) => request<ApprovalActionResponse>(`/api/v1/approvals/${encodeURIComponent(approvalId)}/approve`, { method: "POST", body: JSON.stringify(note ? { note } : {}) }),
  deny: (approvalId: string, note?: string) => request<ApprovalRequest>(`/api/v1/approvals/${encodeURIComponent(approvalId)}/deny`, { method: "POST", body: JSON.stringify(note ? { note } : {}) }),
};

function resultFromRun(run: Run): Result | null {
  const saved = run.metadata.result;
  return saved && typeof saved === "object" ? saved : null;
}

export async function fetchRunView(runId: string, knownRuns: Run[] = []): Promise<RunDetails> {
  const run = await api.run(runId);
  const allRuns = knownRuns.length > 0 ? knownRuns : await api.runs();
  const children = run.parent_run_id === null || run.parent_run_id === undefined
    ? allRuns.filter((item) => item.parent_run_id === run.id)
    : [];
  const traceRunIds = [run.id, ...children.map((item) => item.id)];
  const eventGroups = await Promise.all(traceRunIds.map((id) => api.events(id)));
  const events = eventGroups.flat().sort((left, right) => {
    const timestamp = left.timestamp.localeCompare(right.timestamp);
    if (timestamp !== 0) return timestamp;
    const runOrder = left.run_id.localeCompare(right.run_id);
    return runOrder !== 0 ? runOrder : left.sequence - right.sequence;
  });
  const relatedResults = [run, ...children].map(resultFromRun).filter((item): item is Result => item !== null);
  const result = resultFromRun(run) ?? relatedResults[0] ?? null;
  const artifactIds = new Set(relatedResults.flatMap((item) => item.artifacts));
  const artifactGroups = artifactIds.size > 0 ? await Promise.all(traceRunIds.map((id) => api.artifacts(id))) : [];
  const artifacts = [...new Map(artifactGroups.flat().filter((item) => artifactIds.has(item.id)).map((item) => [item.id, item])).values()];
  return { run, result, events, artifacts };
}
