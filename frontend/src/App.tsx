import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, ApprovalRequest, Artifact, Conversation, Dataset, Event, fetchRunView, MeasurementSystem, ModelStatus, ResponseStyle, Result, Run, TokenUsage, User, UserProfile } from "./api";
import { childRunsOf, isExecutionInflight, isMainRun, runDurationMs, runTitle, runsForConversation } from "./domain";
import { ApprovalCard } from "./components/ApprovalCard";
import { Icon, IconName } from "./components/Icon";
import { LineagePanel } from "./components/LineagePanel";
import { MapViewer } from "./components/MapViewer";
import { RunPanel } from "./components/RunPanel";
import { agentLabel, displayEventMessage, eventLabel, findingText, formatLabel, kindLabel, statusLabel } from "./labels";

type View = "chat" | "datasets" | "agents" | "runs" | "settings";
type ChatMessage = { id: string; role: "user" | "assistant"; content: string; kind?: "text" | "execution"; runId?: string };
type ExecutionPhase = "connecting" | "running";
type ConversationExecutionState = {
  requestId: string;
  runId: string | null;
  phase: ExecutionPhase;
  startedAt: number;
  progressAt: number;
  streamingReply: string;
  events: Event[];
};
type ConversationDraft = {
  message: string;
  selectedDatasetIds: string[];
  uploadedFiles: Dataset[];
};

const emptyDraft = (): ConversationDraft => ({ message: "", selectedDatasetIds: [], uploadedFiles: [] });

function assistantText(value: string): string {
  return value
    .replace(/\*\*(.*?)\*\*/gs, "$1")
    .replace(/__(.*?)__/gs, "$1")
    .replace(/^\s*[*-]\s+/gm, "• ")
    .replace(/\*\*/g, "");
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.name === "AbortError") return "请求超时，请稍后重试。";
  if (error instanceof ApiError) {
    if (error.status === 403) return "没有权限执行这项操作。";
    if (error.status === 404) return "请求的记录不存在，可能已被删除。";
    if (error.status === 409) return error.message || "当前状态不允许执行这项操作。";
  }
  return error instanceof Error ? error.message : String(error);
}

export function formatDuration(durationMs: number, { live = false }: { live?: boolean } = {}): string {
  const normalized = Math.max(0, durationMs);
  if (!live && normalized < 1000) return "用时 <1秒";
  const totalSeconds = Math.floor(normalized / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes === 0) return `用时 ${seconds}秒`;
  if (seconds === 0) return `用时 ${minutes}分钟`;
  return `用时 ${minutes}分钟 ${seconds}秒`;
}

export function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [view, setView] = useState<View>("chat");
  const [accountOpen, setAccountOpen] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messagesByConversation, setMessagesByConversation] = useState<Record<string, ChatMessage[]>>({});
  const [draftsByConversation, setDraftsByConversation] = useState<Record<string, ConversationDraft>>({});
  const [executionsByConversation, setExecutionsByConversation] = useState<Record<string, ConversationExecutionState>>({});
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [uploadingByConversation, setUploadingByConversation] = useState<Record<string, boolean>>({});
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [conversationErrors, setConversationErrors] = useState<Record<string, string>>({});
  const [globalError, setGlobalError] = useState("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [selectedModelProfile, setSelectedModelProfile] = useState("");
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([]);
  const [approvalBusyId, setApprovalBusyId] = useState<string | null>(null);
  const [now, setNow] = useState(() => performance.now());
  const [resumingRunId, setResumingRunId] = useState<string | null>(null);
  const [replyToRunId, setReplyToRunId] = useState<string | null>(null);
  const activeConversationRef = useRef("");
  const streamCancels = useRef<Record<string, () => void>>({});
  const cancelledRequests = useRef<Set<string>>(new Set());
  const conversationLoadSequence = useRef(0);
  const runLoadSequence = useRef(0);

  const currentDraft = draftsByConversation[conversationId] ?? emptyDraft();
  const messages = messagesByConversation[conversationId] ?? [];
  const activeExecution = executionsByConversation[conversationId];
  const uploading = Boolean(uploadingByConversation[conversationId]);
  const error = globalError || conversationErrors[conversationId] || "";
  const conversationRunning = Boolean(activeExecution);
  const message = currentDraft.message;
  const selectedDatasetIds = currentDraft.selectedDatasetIds;
  const uploadedFiles = currentDraft.uploadedFiles;
  const requestDatasetIds = selectedDatasetIds;
  const requestAttachmentIds = useMemo(() => uploadedFiles.map((item) => item.id), [uploadedFiles]);
  const conversationRuns = useMemo(() => runsForConversation(conversationId, runs), [conversationId, runs]);
  const agentCount = useMemo(() => {
    return conversationRuns.filter(isExecutionInflight).length;
  }, [conversationRuns]);
  const elapsedMs = activeExecution ? Math.max(0, now - activeExecution.startedAt) : 0;
  const streamingReply = activeExecution?.streamingReply ?? "";
  const activeRunId = activeExecution?.runId ?? null;
  const liveEvents = activeExecution?.events ?? events;

  useEffect(() => {
    if (Object.keys(executionsByConversation).length === 0) return;
    const timer = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(timer);
  }, [executionsByConversation]);

  const updateDraft = (id: string, changes: Partial<ConversationDraft>) => {
    setDraftsByConversation((current) => ({ ...current, [id]: { ...(current[id] ?? emptyDraft()), ...changes } }));
  };
  const setMessage = (value: string) => updateDraft(conversationId, { message: value });
  const setSelectedDatasetIds = (update: string[] | ((current: string[]) => string[])) => {
    setDraftsByConversation((current) => {
      const draft = current[conversationId] ?? emptyDraft();
      return { ...current, [conversationId]: { ...draft, selectedDatasetIds: typeof update === "function" ? update(draft.selectedDatasetIds) : update } };
    });
  };
  const setUploadedFiles = (update: Dataset[] | ((current: Dataset[]) => Dataset[])) => {
    setDraftsByConversation((current) => {
      const draft = current[conversationId] ?? emptyDraft();
      return { ...current, [conversationId]: { ...draft, uploadedFiles: typeof update === "function" ? update(draft.uploadedFiles) : update } };
    });
  };
  const setMessagesForConversation = (id: string, update: ChatMessage[] | ((current: ChatMessage[]) => ChatMessage[])) => {
    setMessagesByConversation((current) => ({ ...current, [id]: typeof update === "function" ? update(current[id] ?? []) : update }));
  };
  const setExecution = (id: string, update: ConversationExecutionState | undefined | ((current: ConversationExecutionState | undefined) => ConversationExecutionState | undefined)) => {
    setExecutionsByConversation((current) => {
      const next = typeof update === "function" ? update(current[id]) : update;
      const copy = { ...current };
      if (next) copy[id] = next;
      else delete copy[id];
      return copy;
    });
  };
  const setConversationError = (id: string, message: string) => {
    if (!id) {
      setGlobalError(message);
      return;
    }
    setGlobalError("");
    setConversationErrors((current) => ({ ...current, [id]: message }));
  };
  const setError = (message: string) => setConversationError(conversationId, message);

  const refreshDatasets = async () => {
    const next = await api.datasets();
    setDatasets(next);
  };
  const refreshRuns = async () => {
    const next = await api.runs();
    setRuns(next);
  };
  const refreshApprovals = async () => {
    const next = await api.approvals("PENDING");
    setApprovals(next);
  };
  const refreshConversations = async () => {
    const next = await api.conversations();
    setConversations(next);
    return next;
  };
  const refreshModelStatus = async () => {
    const next = await api.modelStatus();
    setModelStatus(next);
    setSelectedModelProfile((current) => {
      const profileIds = next.profiles.map((profile) => profile.id);
      if (current && profileIds.includes(current)) return current;
      return next.default_profile ?? profileIds[0] ?? "";
    });
  };
  const refreshAll = async () => {
    try {
      await Promise.all([refreshDatasets(), refreshRuns(), refreshModelStatus(), refreshApprovals(), refreshConversations()]);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const background = (work: Promise<unknown>, targetId = conversationId) => {
    void work.catch((err) => {
      if (targetId === conversationId) setError(errorMessage(err));
    });
  };

  const loadConversation = async (id: string) => {
    const sequence = ++conversationLoadSequence.current;
    const isCurrentTarget = () => sequence === conversationLoadSequence.current && activeConversationRef.current === id;
    try {
      const history = await api.messages(id);
      if (!isCurrentTarget()) return;
      setSelectedRunId(null);
      // 历史消息只恢复正文和 run_id；事件、产物和子 Run 在用户明确查看
      // 运行详情时再通过 fetchRunView 加载，避免切换 Conversation 产生 N+1。
      const restored = history.map((item): ChatMessage => ({
        id: item.id,
        role: item.role === "user" ? "user" : "assistant",
        content: item.content,
        kind: item.role === "user" ? "text" : item.run_id ? "execution" : "text",
        runId: item.run_id ?? undefined,
      }));
      if (!isCurrentTarget()) return;
      setMessagesForConversation(id, restored);
      setResult(null);
      setEvents([]);
      setArtifacts([]);
    } catch (err) {
      if (isCurrentTarget()) setConversationError(id, errorMessage(err));
    }
  };

  const loadConversations = async () => {
    const next = await api.conversations();
    setConversations(next);
    return next;
  };

  const initializeConversations = async () => {
    try {
      let next = await loadConversations();
      if (next.length === 0) {
        next = [await api.createConversation()];
        setConversations(next);
      }
      const savedId = window.sessionStorage.getItem("geoagent.conversation_id");
      const selected = next.find((item) => item.id === savedId) ?? next[0];
      setConversationId(selected.id);
      activeConversationRef.current = selected.id;
      window.sessionStorage.setItem("geoagent.conversation_id", selected.id);
      await loadConversation(selected.id);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  useEffect(() => {
    let disposed = false;
    void api.me().then((user) => {
      if (!disposed) setCurrentUser(user);
    }).catch(() => {
      if (!disposed) setCurrentUser(null);
    }).finally(() => {
      if (!disposed) setAuthLoading(false);
    });
    return () => { disposed = true; };
  }, []);

  useEffect(() => {
    if (!currentUser) return;
    void api.profile().then(setUserProfile).catch((err) => setError(errorMessage(err)));
    void (async () => {
      await refreshAll();
      await initializeConversations();
    })();
  }, [currentUser?.id]);

  const clearSessionState = () => {
    Object.values(streamCancels.current).forEach((cancel) => cancel());
    streamCancels.current = {};
    cancelledRequests.current.clear();
    setUserProfile(null);
    setConversationId("");
    activeConversationRef.current = "";
    setConversations([]);
    setMessagesByConversation({});
    setDraftsByConversation({});
    setExecutionsByConversation({});
    setUploadingByConversation({});
    setDatasets([]);
    setRuns([]);
    setEvents([]);
    setArtifacts([]);
    setResult(null);
    setSelectedRunId(null);
    setReplyToRunId(null);
    setConversationErrors({});
    setGlobalError("");
    setApprovals([]);
    setModelStatus(null);
    setSelectedModelProfile("");
    window.sessionStorage.removeItem("geoagent.conversation_id");
  };

  const logout = async () => {
    try {
      await api.logout();
    } catch {
      // 即使服务端 Session 已失效，也必须清理当前浏览器状态。
    }
    clearSessionState();
    setCurrentUser(null);
    setAccountOpen(false);
    setView("chat");
  };

  const selectConversation = async (id: string) => {
    setPendingDeleteId(null);
    setView("chat");
    if (id === conversationId) return;
    setReplyToRunId(null);
    setConversationId(id);
    activeConversationRef.current = id;
    window.sessionStorage.setItem("geoagent.conversation_id", id);
    setConversationError(id, "");
    setSelectedRunId(null);
    setResult(null);
    setEvents([]);
    setArtifacts([]);
    await loadConversation(id);
  };

  const createNewConversation = async () => {
    try {
      const created = await api.createConversation();
      setReplyToRunId(null);
      setConversations((current) => [created, ...current]);
      setConversationId(created.id);
      activeConversationRef.current = created.id;
      window.sessionStorage.setItem("geoagent.conversation_id", created.id);
      setMessagesForConversation(created.id, []);
      setEvents([]);
      setArtifacts([]);
      setResult(null);
      updateDraft(created.id, emptyDraft());
      setSelectedRunId(null);
      setView("chat");
      setConversationError(created.id, "");
      setPendingDeleteId(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const deleteConversation = async (conversation: Conversation) => {
    try {
      const execution = executionsByConversation[conversation.id];
      if (execution) {
        cancelledRequests.current.add(execution.requestId);
        streamCancels.current[conversation.id]?.();
        delete streamCancels.current[conversation.id];
        setExecutionsByConversation((current) => {
          const next = { ...current };
          delete next[conversation.id];
          return next;
        });
      }
      await api.deleteConversation(conversation.id);
      const remaining = conversations.filter((item) => item.id !== conversation.id);
      setMessagesByConversation((current) => {
        const next = { ...current };
        delete next[conversation.id];
        return next;
      });
      setDraftsByConversation((current) => {
        const next = { ...current };
        delete next[conversation.id];
        return next;
      });
      setConversationErrors((current) => {
        const next = { ...current };
        delete next[conversation.id];
        return next;
      });
      if (conversation.id !== conversationId) {
        setConversations(remaining);
        setPendingDeleteId(null);
        return;
      }
      if (remaining.length === 0) {
        const created = await api.createConversation();
        setConversations([created]);
        setConversationId(created.id);
        window.sessionStorage.setItem("geoagent.conversation_id", created.id);
        setMessagesForConversation(created.id, []);
        setResult(null);
        setEvents([]);
        setArtifacts([]);
        updateDraft(created.id, emptyDraft());
        setSelectedRunId(null);
        setPendingDeleteId(null);
        return;
      }
      const next = remaining[0];
      setConversations(remaining);
      setConversationId(next.id);
      activeConversationRef.current = next.id;
      window.sessionStorage.setItem("geoagent.conversation_id", next.id);
      await loadConversation(next.id);
      setPendingDeleteId(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const loadRun = async (runId: string) => {
    const targetConversationId = conversationId;
    const sequence = ++runLoadSequence.current;
    setSelectedRunId(runId);
    setResult(null);
    setArtifacts([]);
    setError("");
    try {
      const details = await fetchRunView(runId, runs);
      if (sequence !== runLoadSequence.current || activeConversationRef.current !== targetConversationId) return;
      setEvents(details.events);
      setResult(details.result);
      setArtifacts(details.artifacts);
    } catch (err) {
      if (sequence === runLoadSequence.current && activeConversationRef.current === targetConversationId) setError(errorMessage(err));
    }
  };

  const send = async () => {
    const targetConversationId = conversationId;
    const targetDraft = draftsByConversation[targetConversationId] ?? emptyDraft();
    const prompt = targetDraft.message.trim();
    if (!prompt || !targetConversationId || executionsByConversation[targetConversationId]) return;
    const targetDatasetIds = [...targetDraft.selectedDatasetIds];
    const targetAttachmentIds = targetDraft.uploadedFiles.map((item) => item.id);
    const targetReplyToRunId = replyToRunId;
    const requestId = `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const startedAt = performance.now();
    setError("");
    setMessagesForConversation(targetConversationId, (current) => [...current, { id: `local-user-${requestId}`, role: "user", content: prompt, kind: "text" }]);
    updateDraft(targetConversationId, { message: "", selectedDatasetIds: [], uploadedFiles: [] });
    setExecution(targetConversationId, { requestId, runId: null, phase: "connecting", startedAt, progressAt: startedAt, streamingReply: "", events: [] });
    let receivedEvents: Event[] = [];
    const stream = api.streamMessage(
      prompt,
      targetDatasetIds,
      targetAttachmentIds,
      (run) => {
        setExecution(targetConversationId, (current) => current ? { ...current, runId: run.id, phase: "running" } : current);
        if (activeConversationRef.current === targetConversationId) setSelectedRunId(run.id);
      },
      (event) => {
        receivedEvents = [...receivedEvents, event];
        setExecution(targetConversationId, (current) => current ? { ...current, events: [...current.events, event], streamingReply: event.event_type === "ModelResponseStarted" ? "" : current.streamingReply } : current);
      },
      (content) => setExecution(targetConversationId, (current) => current ? { ...current, streamingReply: current.streamingReply + content } : current),
      targetConversationId,
      selectedModelProfile,
      () => setExecution(targetConversationId, (current) => current ? { ...current, progressAt: performance.now() } : current),
      targetReplyToRunId ?? undefined,
    );
    streamCancels.current[targetConversationId] = stream.cancel;
    try {
      const response = await stream;
      if (streamCancels.current[targetConversationId] === stream.cancel) delete streamCancels.current[targetConversationId];
      if (targetReplyToRunId) setReplyToRunId((current) => current === targetReplyToRunId ? null : current);
      if (response.result && response.run) {
        const messageId = `local-assistant-${response.result.trace_id}`;
        setRuns((current) => [response.run!, ...current.filter((item) => item.id !== response.run!.id)]);
        setMessagesForConversation(targetConversationId, (current) => [...current, { id: messageId, role: "assistant", content: response.result!.summary, kind: "execution", runId: response.run!.id }]);
        if (activeConversationRef.current === targetConversationId) {
          setResult(response.result);
          setEvents(receivedEvents);
        }
        background((async () => {
          // 当前消息已经拥有 result 和本次流式 events；完整 events/artifacts/child runs
          // 只在用户明确点击“查看运行详情”时通过 loadRun 懒加载。
          const refreshes: Promise<unknown>[] = [refreshRuns(), refreshConversations()];
          if (response.result!.datasets.length > 0) refreshes.push(refreshDatasets());
          await Promise.allSettled(refreshes);
        })(), targetConversationId);
      } else {
        setMessagesForConversation(targetConversationId, (current) => [...current, { id: `local-assistant-${response.request_id}`, role: "assistant", content: response.message, kind: "text" }]);
        background(refreshConversations(), targetConversationId);
      }
      // 先让完成消息能从 runs 缓存派生摘要，再移除 live state；React 会在同一轮提交中
      // 从实时进度平滑切换到 CompletedRunSummary。
      setExecution(targetConversationId, undefined);
    } catch (err) {
      setExecution(targetConversationId, undefined);
      if (streamCancels.current[targetConversationId] === stream.cancel) delete streamCancels.current[targetConversationId];
      if (!cancelledRequests.current.delete(requestId) && activeConversationRef.current === targetConversationId) setError(errorMessage(err));
    }
  };

  const approve = async (approval: ApprovalRequest) => {
    const targetConversationId = conversationId;
    setApprovalBusyId(approval.id);
    setError("");
    try {
      const response = await api.approve(approval.id);
      setApprovals((current) => current.filter((item) => item.id !== approval.id));
      if (response.result) {
        setResult(response.result);
        setSelectedRunId(response.run?.id ?? null);
      }
      await Promise.all([refreshRuns(), refreshApprovals(), refreshConversations(), refreshDatasets()]);
      await loadConversation(targetConversationId);
    } catch (err) {
      if (activeConversationRef.current === targetConversationId) setError(errorMessage(err));
    } finally {
      setApprovalBusyId(null);
    }
  };

  const deny = async (approval: ApprovalRequest) => {
    setApprovalBusyId(approval.id);
    setError("");
    try {
      await api.deny(approval.id);
      setApprovals((current) => current.filter((item) => item.id !== approval.id));
      await refreshApprovals();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setApprovalBusyId(null);
    }
  };

  const cancelRun = async (runId: string) => {
    setError("");
    try {
      await api.cancelRun(runId);
      background(refreshRuns());
      if (selectedRunId === runId) {
        background((async () => {
          const details = await fetchRunView(runId, runs);
          setEvents(details.events);
          setResult(details.result);
          setArtifacts(details.artifacts);
        })());
      }
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const cancelCurrentExecution = async () => {
    const execution = executionsByConversation[conversationId];
    if (!execution) return;
    if (execution.runId) {
      await cancelRun(execution.runId);
      setExecution(conversationId, (current) => current?.requestId === execution.requestId ? undefined : current);
      return;
    }
    cancelledRequests.current.add(execution.requestId);
    streamCancels.current[conversationId]?.();
  };

  const deleteRun = async (runId: string) => {
    setError("");
    try {
      await api.deleteRun(runId);
      setRuns((current) => current.filter((item) => item.id !== runId));
      if (selectedRunId === runId) {
        setSelectedRunId(null);
        setEvents([]);
        setArtifacts([]);
        setResult(null);
      }
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const deleteRunRecords = async (runIds: string[]) => {
    if (runIds.length === 0) return;
    setError("");
    try {
      const response = await api.deleteRuns(runIds);
      const deleted = new Set(response.deleted);
      setRuns((current) => current.filter((item) => !deleted.has(item.id)));
      if (selectedRunId && deleted.has(selectedRunId)) {
        setSelectedRunId(null);
        setEvents([]);
        setArtifacts([]);
        setResult(null);
      }
      if (response.skipped_active.length > 0) setError(`${response.skipped_active.length} 条正在运行的记录未删除，请先取消运行。`);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const resumeRun = async (runId: string) => {
    setResumingRunId(runId);
    setError("");
    try {
      const resumed = await api.resumeRun(runId);
      setSelectedRunId(resumed.run_id);
      setView("runs");
      await refreshRuns();
      const details = await fetchRunView(resumed.run_id, runs);
      setResult(details.result ?? resumed.result);
      setEvents(details.events);
      setArtifacts(details.artifacts);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setResumingRunId(null);
    }
  };

  const registerDataset = async (path: string, name: string) => {
    setError("");
    try {
      await api.registerDataset(path, name);
      await refreshDatasets();
    } catch (err) {
      setError(errorMessage(err));
      throw err;
    }
  };

  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    const targetConversationId = conversationId;
    setUploadingByConversation((current) => ({ ...current, [targetConversationId]: true }));
    setError("");
    try {
      for (const file of Array.from(files)) {
        const dataset = await api.uploadAttachment(file);
        setDraftsByConversation((current) => ({ ...current, [targetConversationId]: { ...(current[targetConversationId] ?? emptyDraft()), uploadedFiles: [...(current[targetConversationId]?.uploadedFiles ?? []), dataset] } }));
      }
      await refreshDatasets();
    } catch (err) {
      if (activeConversationRef.current === targetConversationId) setError(errorMessage(err));
    } finally {
      setUploadingByConversation((current) => ({ ...current, [targetConversationId]: false }));
    }
  };

  if (authLoading) return <div className="auth-shell"><div className="auth-card"><span className="eyebrow">GeoAgent</span><h1>正在检查登录状态</h1><p>请稍候…</p></div></div>;
  if (!currentUser) return <AuthPage onAuthenticated={setCurrentUser} />;

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Icon name="compass" size={21} /></span><div><b>GeoAgent</b><small>空间智能</small></div></div>
       <section className="sidebar-block workspace-block"><div className="sidebar-block-title">工作区</div><Nav label="数据集" icon="layers" count={datasets.length} active={view === "datasets"} onClick={() => setView("datasets")} /><Nav label="智能体" icon="agent" count={agentCount} active={view === "agents"} onClick={() => setView("agents")} /><Nav label="运行与追踪" icon="activity" count={conversationRuns.length} active={view === "runs"} onClick={() => setView("runs")} /></section>
      <section className="sidebar-block conversation-block"><div className="sidebar-block-head"><span>对话</span><button type="button" className="new-conversation" aria-label="新建对话" title="新建对话" onClick={() => void createNewConversation()}><Icon name="plus" size={18} /></button></div><div className="conversation-list">{conversations.length === 0 ? <span className="conversation-empty">正在加载对话…</span> : conversations.map((item) => <div className="conversation-row" key={item.id}><button type="button" className={`conversation-select ${item.id === conversationId ? "active" : ""}`} onClick={() => void selectConversation(item.id)}><span className="conversation-dot" /><span className="conversation-title">{item.title || "新对话"}</span></button><button type="button" className="conversation-delete" aria-label={`删除对话 ${item.title}`} title="删除对话" onClick={() => setPendingDeleteId((current) => current === item.id ? null : item.id)}><Icon name="trash" size={13} /></button>{pendingDeleteId === item.id && <div className="conversation-confirm" role="dialog" aria-label={`确认删除对话 ${item.title}`}><span>删除这个对话？</span><div><button type="button" className="conversation-confirm-delete" onClick={() => void deleteConversation(item)}>删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setPendingDeleteId(null)}>取消</button></div></div>}</div>)}</div></section>
      <div className="account-area">{accountOpen && <div className="account-menu"><button type="button" onClick={() => { setView("settings"); setAccountOpen(false); }}>设置</button><button type="button" onClick={() => void logout()}>退出登录</button><div className="account-menu-note">当前账号：{currentUser.username}</div></div>}<button type="button" className="account-button" aria-expanded={accountOpen} onClick={() => setAccountOpen((current) => !current)}><span className="account-avatar">{(currentUser.display_name || currentUser.username).slice(0, 1).toUpperCase()}</span><span className="account-copy"><b>{currentUser.display_name || currentUser.username}</b><small>@{currentUser.username}</small></span></button></div>
    </aside>
    <main className="main">
       {view !== "chat" && <header className="topbar"><div><h1>{view === "datasets" ? "数据集登记" : view === "agents" ? "智能体活动" : view === "runs" ? "运行与追踪" : "设置"}</h1></div><div className="topbar-actions"><button className="ghost" onClick={() => setView("chat")}>返回对话</button><button className="close-view" type="button" aria-label="关闭当前页面" title="关闭" onClick={() => setView("chat")}><Icon name="close" size={18} /></button><button className="ghost" onClick={() => void refreshAll()}><Icon name="refresh" size={13} /> 刷新</button></div></header>}
      {error && <div className="error">{error}</div>}
       {view === "chat" && <ProductChat message={message} setMessage={setMessage} busy={conversationRunning} conversationReady={Boolean(conversationId)} streamingReply={streamingReply} elapsedMs={elapsedMs} activeRunId={activeRunId} events={liveEvents} messages={messages} runs={conversationRuns} send={send} cancel={cancelCurrentExecution} onShowRun={(runId) => { void loadRun(runId); setView("runs"); }} replyToRunId={replyToRunId} onReplyToRun={setReplyToRunId} datasets={datasets} selectedDatasetIds={selectedDatasetIds} onRemoveDataset={(id) => setSelectedDatasetIds((current) => current.filter((item) => item !== id))} uploadedFiles={uploadedFiles} uploading={uploading} onUpload={uploadFiles} onRemoveFile={(id) => setUploadedFiles((current) => current.filter((item) => item.id !== id))} modelStatus={modelStatus} selectedModelProfile={selectedModelProfile} onModelChange={setSelectedModelProfile} approvals={approvals.filter((item) => item.conversation_id === conversationId)} approvalBusyId={approvalBusyId} onApprove={approve} onDeny={deny} />}
      {view === "datasets" && <DatasetPanel datasets={datasets} selectedDatasetIds={selectedDatasetIds} onToggleRequestDataset={(id) => setSelectedDatasetIds((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])} onRegister={registerDataset} busy={false} />}
      {view === "agents" && <AgentPanel runs={conversationRuns} />}
       {view === "runs" && <RunPanel runs={conversationRuns} selectedRunId={selectedRunId} events={events} result={result} datasets={datasets} artifacts={artifacts} onSelect={loadRun} onCancel={cancelRun} onResume={resumeRun} onDelete={deleteRun} onDeleteMany={deleteRunRecords} busy={resumingRunId !== null} />}
      {view === "settings" && <SettingsPanel currentUser={currentUser} onSaved={setCurrentUser} profile={userProfile} onProfileSaved={setUserProfile} modelStatus={modelStatus} />}
    </main>
  </div>;
}

function Nav({ label, icon, count, active, onClick }: { label: string; icon: IconName; count?: number; active: boolean; onClick: () => void }) { return <button className={`nav ${active ? "active" : ""}`} onClick={onClick}><span aria-hidden="true"><Icon name={icon} /></span>{label}{count !== undefined && <em>{count}</em>}</button>; }

type ProductChatProps = {
  message: string;
  setMessage: (value: string) => void;
  busy: boolean;
  conversationReady: boolean;
  streamingReply: string;
  elapsedMs: number;
  send: () => Promise<void>;
  cancel: () => Promise<void>;
  activeRunId: string | null;
  events: Event[];
  messages: ChatMessage[];
  runs?: Run[];
  onShowRun: (runId: string) => void;
  replyToRunId: string | null;
  onReplyToRun: (runId: string | null) => void;
  datasets: Dataset[];
  selectedDatasetIds: string[];
  onRemoveDataset: (id: string) => void;
  uploadedFiles: Dataset[];
  uploading: boolean;
  onUpload: (files: FileList | null) => Promise<void>;
  onRemoveFile: (id: string) => void;
  modelStatus: ModelStatus | null;
  selectedModelProfile: string;
  onModelChange: (value: string) => void;
  approvals: ApprovalRequest[];
  approvalBusyId: string | null;
  onApprove: (approval: ApprovalRequest) => Promise<void>;
  onDeny: (approval: ApprovalRequest) => Promise<void>;
};

export function ProductChat({
  message, setMessage, busy, conversationReady, streamingReply, elapsedMs, send, cancel, activeRunId, events, messages, runs = [], onShowRun, replyToRunId, onReplyToRun, datasets, selectedDatasetIds, onRemoveDataset, uploadedFiles, uploading, onUpload, onRemoveFile, modelStatus, selectedModelProfile, onModelChange, approvals, approvalBusyId, onApprove, onDeny,
}: ProductChatProps) {
  const sending = busy;
  return <section className="chat-layout product-chat">
    {approvals.length > 0 && <div className="approval-stack">{approvals.map((approval) => <ApprovalCard key={approval.id} approval={approval} busy={approvalBusyId === approval.id} onApprove={onApprove} onDeny={onDeny} />)}</div>}
    {(messages.length > 0 || busy) && <div className="chat-history" aria-live="polite">
      {messages.map((item) => <ChatBubble item={item} runs={runs} onShowRun={onShowRun} onReplyToRun={onReplyToRun} key={item.id} />)}
      {busy && <>
        <div className="live-execution-row"><LiveExecutionStatus events={events} durationMs={elapsedMs} phase={activeRunId ? "running" : "connecting"} tokenUsage={runs.find((run) => run.id === activeRunId)?.token_usage} /></div>
        {streamingReply && <div className="chat-message assistant streaming-message"><div className="chat-bubble">{assistantText(streamingReply)}<span className="typing-cursor" aria-hidden="true" /></div></div>}
      </>}
    </div>}
    <div className="composer">
      {replyToRunId && <div className="reply-target-banner" role="status">正在补充运行 {replyToRunId}<button type="button" onClick={() => onReplyToRun(null)}>取消</button></div>}
      <textarea className="composer-input" value={message} onChange={(event) => setMessage(event.target.value)} rows={2} aria-label="输入消息" placeholder="输入消息" disabled={!conversationReady || sending} />
      <div className="composer-foot"><div className="composer-left"><label className="file-button" title="添加文件" aria-label="添加文件"><span aria-hidden="true"><Icon name="plus" size={22} /></span><input type="file" multiple accept=".geojson,.json,.gpkg,.shp,.zip,.kml,.gml,.tif,.tiff,.img,.vrt,.asc,.csv,.tsv,.parquet,.jsonl" disabled={sending || uploading} onChange={(event) => { void onUpload(event.currentTarget.files); event.currentTarget.value = ""; }} /></label>{selectedDatasetIds.length > 0 && <div className="file-chips request-dataset-chips"><span className="resource-chip-label">数据：</span>{selectedDatasetIds.map((id) => { const dataset = datasets.find((item) => item.id === id); return <span className="file-chip" key={id}><span className="file-chip-name"><Icon name="layers" size={11} /> {dataset?.name ?? id}</span><button type="button" className="file-remove" title={`移除数据集 ${dataset?.name ?? id}`} aria-label={`移除数据集 ${dataset?.name ?? id}`} onClick={() => onRemoveDataset(id)}><Icon name="close" size={11} /></button></span>; })}</div>}{uploadedFiles.length > 0 && <div className="file-chips request-attachment-chips">{uploadedFiles.map((file) => <span className="file-chip" key={file.id}><span className="file-chip-name"><Icon name="attachment" size={11} /> {file.name}</span><button type="button" className="file-remove" title={`移除 ${file.name}`} aria-label={`移除 ${file.name}`} onClick={() => onRemoveFile(file.id)}><Icon name="close" size={11} /></button></span>)}</div>}</div><div className="composer-right">{uploading && <span className="uploading">正在上传…</span>}{modelStatus && (modelStatus.profiles.length > 0 ? <div className="model-picker"><span>模型</span><select value={selectedModelProfile} onChange={(event) => onModelChange(event.target.value)} disabled={sending} aria-label="选择模型">{modelStatus.profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.label}</option>)}</select></div> : <span className="model-picker-offline">未配置模型</span>)}{busy ? <button className="cancel" aria-label="取消运行" onClick={() => void cancel()}>取消运行</button> : <button className="primary send-button" aria-label="发送" title="发送" disabled={!message.trim() || sending || uploading || !conversationReady} onClick={() => void send()}><Icon name="send" size={19} /></button>}</div></div>
    </div>
  </section>;
}

export function ChatBubble({ item, runs = [], onShowRun, onReplyToRun }: { item: ChatMessage; runs?: Run[]; onShowRun: (runId: string) => void; onReplyToRun: (runId: string | null) => void }) {
  const run = item.kind === "execution" && item.runId ? runs.find((candidate) => candidate.id === item.runId) : undefined;
  const executionMessage = item.role === "assistant" && item.kind === "execution";
  return <div className={`chat-message ${item.role} ${executionMessage ? "execution-message" : ""}`}>{run && <CompletedRunSummary run={run} />}<div className={`chat-bubble ${executionMessage ? "execution-answer" : ""}`}>{item.role === "assistant" ? assistantText(item.content) : item.content}</div>{run?.status === "WAITING_USER" && <button type="button" className="chat-result-link" onClick={() => onReplyToRun(run.id)}>继续此运行</button>}{item.runId && <button className="chat-result-link" onClick={() => onShowRun(item.runId!)}>查看运行详情</button>}</div>;
}

export function CompletedRunSummary({ run }: { run: Run }) {
  const failed = ["FAILED", "CANCELLED", "INTERRUPTED", "BUDGET_EXCEEDED"].includes(run.status);
  const waiting = ["WAITING_USER", "WAITING_APPROVAL"].includes(run.status);
  const detail = failed ? "运行未完成" : waiting ? "等待补充信息" : run.status === "PARTIAL_COMPLETED" ? "运行部分完成" : "运行完成";
  return <div className="run-summary"><div className="run-summary-head"><span className="run-summary-time">{formatDuration(runDurationMs(run))}</span><span className={`run-summary-status ${failed ? "failed" : waiting ? "waiting" : run.status.toLowerCase()}`}>{statusLabel(run.status)}</span></div><div className="run-summary-current"><span className={`run-summary-marker ${failed ? "failed" : waiting ? "waiting" : ""}`}><Icon name={failed ? "close" : waiting ? "clock" : "check"} size={11} /></span><span>{detail} · {run.tool_call_count} 次工具调用</span><TokenUsageSummary usage={run.token_usage} /></div></div>;
}

export function TokenUsageSummary({ usage }: { usage?: TokenUsage | null }) {
  if (!usage || usage.model_calls === 0) return null;
  const reported = usage.reported_calls === usage.model_calls;
  const input = reported ? usage.reported_input_tokens : usage.local_input_tokens;
  const output = reported ? usage.reported_output_tokens : usage.local_output_tokens;
  const format = (value: number) => `${(value / 1000).toFixed(2)}k`;
  const source = reported ? "模型返回的实际用量" : "本地分词估算，不等同于账单用量";
  return <span className="run-token-usage" title={`${source}；输入 ${input.toLocaleString("zh-CN")} tokens，输出 ${output.toLocaleString("zh-CN")} tokens；累计 ${usage.model_calls} 次模型调用（包含子运行），每轮输入重复计入。每次模型响应结束后更新。`}>输入 {format(input)} · 输出 {format(output)}{reported ? "" : "（本地估算）"}</span>;
}

export function LiveExecutionStatus({ events, durationMs, phase, tokenUsage }: { events: Event[]; durationMs: number; phase: ExecutionPhase; tokenUsage?: TokenUsage | null }) {
  return <RunProgress events={events} durationMs={durationMs} live phase={phase} tokenUsage={tokenUsage} />;
}

export function RunProgress({ events, durationMs, status, live = false, phase = "running", tokenUsage }: { events: Event[]; durationMs: number; status?: string; live?: boolean; phase?: ExecutionPhase; tokenUsage?: TokenUsage | null }) {
  const recentEvents = [...events].reverse();
  const currentEvent = recentEvents.find((event) => event.event_type !== "TokenUsageUpdated");
  const usage = events.filter((event) => event.event_type === "TokenUsageUpdated").reduce<TokenUsage | null | undefined>((current, event) => {
    const snapshot = event.payload.token_usage as TokenUsage;
    return !current || snapshot.model_calls > current.model_calls ? snapshot : current;
  }, tokenUsage);
  const connecting = live && phase === "connecting";
  const displayStatus = connecting ? "正在处理" : live ? "正在运行" : statusLabel(status ?? "COMPLETED");
  return <div className="run-progress"><div className="run-progress-head"><span className="run-progress-time">{formatDuration(durationMs, { live })}</span><span className={`run-progress-status ${connecting ? "connecting" : live ? "running" : (status ?? "COMPLETED").toLowerCase()}`}>{displayStatus}</span></div><div className="run-progress-current">{connecting ? <><span className="run-progress-marker waiting"><Icon name="clock" size={11} /></span><span className="run-progress-text">正在接入 Agent Loop…</span></> : currentEvent ? <><span className="run-progress-marker"><Icon name="check" size={11} /></span><span className="run-progress-text">{eventLabel(currentEvent.event_type)}：{displayEventMessage(currentEvent.message)}</span></> : <><span className="run-progress-spinner" /><span className="run-progress-text">等待智能体事件…</span></>}<TokenUsageSummary usage={usage} /></div></div>;
}

function DatasetPanel({ datasets, selectedDatasetIds, onToggleRequestDataset, onRegister, busy }: { datasets: Dataset[]; selectedDatasetIds: string[]; onToggleRequestDataset: (id: string) => void; onRegister: (path: string, name: string) => Promise<void>; busy: boolean }) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [registering, setRegistering] = useState(false);
  const [propertyDataset, setPropertyDataset] = useState<Dataset | null>(null);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!path.trim()) return;
    setRegistering(true);
    try {
      await onRegister(path.trim(), name.trim());
      setPath("");
      setName("");
    } catch {
      // The parent renders the request error.
    } finally {
      setRegistering(false);
    }
  };
  return <section className="panel"><div className="panel-head"><div><span className="eyebrow">已登记数据源</span><h2>数据集登记</h2></div><span className="count-badge">{datasets.length} 个数据集</span></div><form className="dataset-register" onSubmit={(event) => void submit(event)}><input value={path} onChange={(event) => setPath(event.target.value)} placeholder="请输入工作区内的数据文件路径" aria-label="数据文件路径" /><input value={name} onChange={(event) => setName(event.target.value)} placeholder="显示名称（可选）" aria-label="显示名称" /><button className="primary" disabled={busy || registering || !path.trim()}>{registering ? "正在登记…" : "登记数据集"}</button></form>{datasets.length === 0 ? <Empty text="还没有数据集。请输入工作区内的文件路径进行登记。" /> : <div className="dataset-grid">{datasets.map((item) => { const selected = selectedDatasetIds.includes(item.id); return <div className={`dataset-card ${selected ? "request-selected" : ""}`} key={item.id}><b className="dataset-name">{item.name}</b><div className="dataset-card-actions"><button type="button" className={`small-action ${selected ? "selected-action" : ""}`} onClick={() => onToggleRequestDataset(item.id)}>{selected ? "移出本轮" : "用于下一条消息"}</button><button type="button" className="small-action" onClick={() => setPropertyDataset(item)}>属性</button></div></div>; })}</div>}{propertyDataset && <div className="dataset-modal" role="dialog" aria-modal="true" aria-label="数据集属性" onClick={() => setPropertyDataset(null)}><div className="dataset-modal-card" onClick={(event) => event.stopPropagation()}><div className="dataset-modal-head"><div><span className="eyebrow">数据集属性</span><h2>{propertyDataset.name}</h2></div><button type="button" className="modal-close" onClick={() => setPropertyDataset(null)}>关闭</button></div><div className="dataset-preview-layout"><div><div className="property-grid"><Property label="数据集编号" value={propertyDataset.id} /><Property label="数据类型" value={kindLabel(propertyDataset.kind)} /><Property label="文件格式" value={formatLabel(propertyDataset.format)} /><Property label="坐标系" value={propertyDataset.crs?.authority ?? "未提供"} /><Property label="坐标系名称" value={propertyDataset.crs?.name ?? "未提供"} /><Property label="要素数量" value={propertyDataset.schema?.feature_count !== undefined ? String(propertyDataset.schema.feature_count) : "不适用"} /><Property label="几何类型" value={propertyDataset.schema?.geometry_type ?? "未提供"} /><Property label="栅格尺寸" value={propertyDataset.schema?.width !== undefined ? `${propertyDataset.schema.width} × ${propertyDataset.schema.height ?? "?"}，${propertyDataset.schema.bands ?? "?"} 个波段` : "不适用"} /><Property label="空间范围" value={propertyDataset.extent ? `${propertyDataset.extent.min_x}, ${propertyDataset.extent.min_y} 至 ${propertyDataset.extent.max_x}, ${propertyDataset.extent.max_y}` : "未提供"} /><Property label="文件路径" value={propertyDataset.path} /><Property label="创建时间" value={propertyDataset.created_at ?? "未提供"} /><Property label="创建运行" value={propertyDataset.created_by_run_id ?? "手动登记或上传"} /></div><h3>字段</h3><pre>{propertyDataset.schema?.fields && Object.keys(propertyDataset.schema.fields).length > 0 ? JSON.stringify(propertyDataset.schema.fields, null, 2) : "未提供字段信息"}</pre><h3>附加信息</h3><pre>{JSON.stringify(propertyDataset.metadata ?? {}, null, 2)}</pre></div><div><MapViewer datasetId={propertyDataset.id} title={propertyDataset.name} /><LineagePanel datasetId={propertyDataset.id} datasetNames={Object.fromEntries(datasets.map((item) => [item.id, item.name]))} /></div></div></div></div>}</section>;
}

function Property({ label, value }: { label: string; value: string }) { return <div className="property-item"><span>{label}</span><b>{value}</b></div>; }

function AgentPanel({ runs }: { runs: Run[] }) {
  const mainRuns = runs.filter(isMainRun);
  return <section className="panel"><div className="panel-head"><div><span className="eyebrow">执行树</span><h2>智能体活动</h2></div><span className="count-badge">{runs.length} 个执行节点</span></div>{mainRuns.length === 0 ? <Empty text="运行任务后，这里会显示主运行和子智能体执行树。" /> : <div className="agent-grid">{mainRuns.map((main) => { const children = childRunsOf(main.id, runs); return <div className="agent-card execution-tree-card" key={main.id}><div className="agent-icon"><Icon name="agent" size={20} /></div><div className="execution-tree-copy"><b>{runTitle(main)}</b><small>主运行 · {statusLabel(main.status)} · {main.id}</small><span>{children.length ? `包含 ${children.length} 个子智能体执行` : "尚未委派子智能体"}</span>{children.length > 0 && <div className="execution-children">{children.map((child) => <div className="execution-child" key={child.id}><i><Icon name="branch" size={12} /></i><div><b>{runTitle(child)}</b><small>子智能体 · {statusLabel(child.status)}</small></div></div>)}</div>}</div></div>; })}</div>}</section>;
}

function SettingsPanel({ currentUser, onSaved, profile, onProfileSaved, modelStatus }: { currentUser: User; onSaved: (user: User) => void; profile: UserProfile | null; onProfileSaved: (profile: UserProfile) => void; modelStatus: ModelStatus | null }) {
  const [displayName, setDisplayName] = useState(currentUser.display_name);
  const [email, setEmail] = useState(currentUser.email ?? "");
  const [language, setLanguage] = useState(profile?.language ?? "zh-CN");
  const [responseStyle, setResponseStyle] = useState<ResponseStyle>(profile?.response_style ?? "balanced");
  const [measurementSystem, setMeasurementSystem] = useState<MeasurementSystem>(profile?.measurement_system ?? "metric");
  const [preferredOutputFormat, setPreferredOutputFormat] = useState(profile?.preferred_output_format ?? "");
  const [saving, setSaving] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [saved, setSaved] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    setLanguage(profile?.language ?? "zh-CN");
    setResponseStyle(profile?.response_style ?? "balanced");
    setMeasurementSystem(profile?.measurement_system ?? "metric");
    setPreferredOutputFormat(profile?.preferred_output_format ?? "");
  }, [profile]);
  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setSaved(false);
    setError("");
    try {
      const user = await api.updateMe(displayName, email);
      onSaved(user);
      setSaved(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSaving(false);
    }
  };
  const saveProfile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSavingProfile(true);
    setProfileSaved(false);
    setError("");
    try {
      const updated = await api.updateProfile({ language, response_style: responseStyle, measurement_system: measurementSystem, preferred_output_format: preferredOutputFormat || null });
      onProfileSaved(updated);
      setProfileSaved(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSavingProfile(false);
    }
  };
  return <section className="panel settings-panel"><div className="panel-head"><div><span className="eyebrow">账号与运行配置</span><h2>设置</h2></div></div><form className="account-settings-form" onSubmit={(event) => void save(event)}><h3>账号信息</h3><label>显示名称<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label><label>用户名<input value={currentUser.username} readOnly /></label><label>邮箱<input value={email} onChange={(event) => setEmail(event.target.value)} type="email" /></label><button className="primary" disabled={saving || !displayName.trim()}>{saving ? "正在保存…" : "保存账号信息"}</button>{saved && <span className="settings-success">已保存</span>}</form><form className="account-settings-form" onSubmit={(event) => void saveProfile(event)}><h3>用户偏好</h3><label>语言<select value={language} onChange={(event) => setLanguage(event.target.value)}><option value="zh-CN">中文</option><option value="en-US">English</option></select></label><label>回答风格<select value={responseStyle} onChange={(event) => setResponseStyle(event.target.value as ResponseStyle)}><option value="concise">简洁</option><option value="balanced">平衡</option><option value="detailed">详细</option></select></label><label>单位制<select value={measurementSystem} onChange={(event) => setMeasurementSystem(event.target.value as MeasurementSystem)}><option value="metric">公制</option><option value="imperial">英制</option></select></label><label>默认输出格式<select value={preferredOutputFormat} onChange={(event) => setPreferredOutputFormat(event.target.value)}><option value="">自动</option><option value="GeoPackage">GeoPackage</option><option value="GeoJSON">GeoJSON</option><option value="GeoTIFF">GeoTIFF</option><option value="CSV">CSV</option></select></label><button className="primary" disabled={savingProfile}>{savingProfile ? "正在保存…" : "保存用户偏好"}</button>{profileSaved && <span className="settings-success">偏好已保存</span>}{error && <span className="settings-inline-error">{error}</span>}<p className="settings-note">这些偏好只作为默认交互方式；当前请求的明确要求优先，格式偏好仅在任务能力允许时使用。</p></form><div className="settings-grid"><div className="setting-item"><span>登录状态</span><b>已登录</b></div><div className="setting-item"><span>默认模型</span><b>{modelStatus?.profiles.find((profile) => profile.id === modelStatus.default_profile)?.label ?? "未配置"}</b></div><div className="setting-item"><span>可用模型</span><b>{modelStatus?.profiles.length ?? 0} 个</b></div></div><p className="settings-note">模型接口从后端环境配置中读取，发送消息时可在对话框右下角切换。</p></section>;
}

function AuthPage({ onAuthenticated }: { onAuthenticated: (user: User) => void }) {
  const [registering, setRegistering] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const user = registering
        ? await api.register(username, password, displayName || username, email)
        : await api.login(username, password);
      onAuthenticated(user);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };
  return <div className="auth-shell"><div className="auth-card"><div className="brand auth-brand"><span className="brand-mark"><Icon name="compass" size={21} /></span><div><b>GeoAgent</b><small>空间智能</small></div></div><span className="eyebrow">{registering ? "创建账号" : "欢迎回来"}</span><h1>{registering ? "创建你的 GeoAgent 账号" : "登录 GeoAgent"}</h1><p>{registering ? "账号创建后，你的数据、对话和运行记录将独立保存。" : "登录后继续访问你的对话和空间数据。"}</p><form className="auth-form" onSubmit={(event) => void submit(event)}><label>用户名或邮箱<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" required /></label><label>密码<input value={password} onChange={(event) => setPassword(event.target.value)} type="password" autoComplete={registering ? "new-password" : "current-password"} required /></label>{registering && <><label>显示名称<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="可选，默认使用用户名" /></label><label>邮箱<input value={email} onChange={(event) => setEmail(event.target.value)} type="email" placeholder="可选" /></label></>}<button className="primary auth-submit" disabled={busy}>{busy ? "处理中…" : registering ? "注册并登录" : "登录"}</button>{error && <div className="error auth-error">{error}</div>}</form><button type="button" className="auth-switch" onClick={() => { setRegistering((value) => !value); setError(""); }}>{registering ? "已有账号？返回登录" : "还没有账号？注册"}</button></div></div>;
}

function Empty({ text }: { text: string }) { return <div className="empty">{text}</div>; }
