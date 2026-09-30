import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApprovalCard } from "./components/ApprovalCard";
import { api, ApprovalRequest, Event, MessageResponse, ReasoningEffort, Run, TokenUsage } from "./api";
import { MapViewer } from "./components/MapViewer";
import { RunPanel } from "./components/RunPanel";
import { App, CompletedRunSummary, LiveExecutionStatus, ProductChat } from "./App";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const approval: ApprovalRequest = { id: "approval-1", user_id: "user-1", conversation_id: "conv-1", task_id: "task-1", source_run_id: "run-1", tool_call_id: "call-1", tool_name: "vector.buffer", argument_fingerprint: "fingerprint", risk_level: "WRITE", argument_preview: { distance: "500m", path: "<已隐藏>" }, reason: "该操作将生成新的结果数据。", status: "PENDING", created_at: "2026-01-01T00:00:00Z" };

describe("运行累计 Token 用量", () => {
  const usage: TokenUsage = { local_input_tokens: 900, local_output_tokens: 100, reported_input_tokens: 1200, reported_output_tokens: 250, model_calls: 2, reported_calls: 2 };

  it("完成摘要在工具调用数后以k显示输入和输出，不再显示总量", () => {
    const { container } = render(<CompletedRunSummary run={{ ...runRecord("COMPLETED"), tool_call_count: 9, token_usage: usage }} />);
    expect(screen.getByText("运行完成 · 9 次工具调用")).toBeTruthy();
    expect(screen.getByText("输入 1.20k · 输出 0.25k")).toBeTruthy();
    expect(screen.queryByText(/Token 1,450/)).toBeNull();
    expect(screen.queryByText(/本地估算/)).toBeNull();
    expect(container.querySelector(".run-summary-current .run-token-usage")?.getAttribute("title")).toContain("输入 1,200 tokens");
  });

  it("实时更新使用累计快照，重复事件不重复累加，也不覆盖当前执行状态", () => {
    const progress: Event = { id: "progress", run_id: "run-1", event_type: "ToolStarted", sequence: 1, timestamp: "2026-01-01T10:00:00Z", message: "检查栅格", payload: {} };
    const measured: Event = { ...progress, id: "usage", event_type: "TokenUsageUpdated", sequence: 2, payload: { token_usage: usage } };
    const { rerender } = render(<LiveExecutionStatus phase="running" events={[progress]} durationMs={1000} tokenUsage={{ ...usage, reported_calls: 1 }} />);
    expect(screen.getByText("输入 0.90k · 输出 0.10k")).toBeTruthy();
    rerender(<LiveExecutionStatus phase="running" events={[progress, measured, measured]} durationMs={2000} />);
    expect(screen.getByText("输入 1.20k · 输出 0.25k")).toBeTruthy();
    expect(screen.getByText("正在执行：检查栅格")).toBeTruthy();
    expect(screen.queryByText(/模型累计用量已更新/)).toBeNull();
    const older: Event = { ...measured, id: "older", payload: { token_usage: { ...usage, model_calls: 1, reported_calls: 1, reported_input_tokens: 100 } } };
    rerender(<LiveExecutionStatus phase="running" events={[progress, measured, older]} durationMs={2000} />);
    expect(screen.getByText("输入 1.20k · 输出 0.25k")).toBeTruthy(); // 并行子运行事件晚到不能使计数倒退。
    rerender(<LiveExecutionStatus phase="running" events={[progress]} durationMs={0} />);
    expect(screen.queryByText(/输入 .*k · 输出/)).toBeNull();
  });

  it("实时 Token 连续追赶新目标，临时缺少快照时不消失", () => {
    vi.useFakeTimers();
    const progress: Event = { id: "progress", run_id: "run-1", event_type: "ToolStarted", sequence: 1, timestamp: "2026-01-01T10:00:00Z", message: "检查数据", payload: {} };
    const first = { ...usage, reported_calls: 1, local_input_tokens: 8_000, local_output_tokens: 1_000 };
    const { container, rerender } = render(<LiveExecutionStatus phase="running" events={[progress]} durationMs={1000} tokenUsage={first} smoothTokenUsage />);
    expect(container.querySelector(".run-token-usage")).toBeTruthy();
    act(() => vi.advanceTimersByTime(200));
    const before = container.querySelector(".run-token-usage")?.textContent;
    rerender(<LiveExecutionStatus phase="running" events={[progress]} durationMs={1200} tokenUsage={{ ...first, local_input_tokens: 10_000, local_output_tokens: 2_000 }} smoothTokenUsage />);
    act(() => vi.advanceTimersByTime(200));
    const after = container.querySelector(".run-token-usage")?.textContent;
    expect(after).not.toBe(before);
    rerender(<LiveExecutionStatus phase="running" events={[progress]} durationMs={1400} smoothTokenUsage />);
    expect(container.querySelector(".run-token-usage")).toBeTruthy();
  });
});

describe("统一聊天输入区", () => {
  const productChatProps = {
    message: "",
    setMessage: vi.fn(),
    busy: false,
    conversationReady: true,
    streamingReply: "",
    elapsedMs: 0,
    send: vi.fn(async () => undefined),
    cancel: vi.fn(async () => undefined),
    activeRunId: null,
    events: [],
    messages: [{ id: "message-1", role: "user" as const, content: "历史消息" }],
    onShowResult: vi.fn(),
    onShowRun: vi.fn(),
    replyToRunId: null,
    onReplyToRun: vi.fn(),
    datasets: [],
    selectedDatasetIds: [],
    onRemoveDataset: vi.fn(),
    uploadedFiles: [],
    uploading: false,
    onUpload: vi.fn(async () => undefined),
    onRemoveFile: vi.fn(),
    modelStatus: { configured: true, source: "test", default_profile: "qwen", profiles: [{ id: "qwen", label: "通义千问", provider: "openai-compatible", model: "qwen", timeout_seconds: 60, temperature: 0.2, reasoning_efforts: [], has_api_key: true, default: true }] },
    selectedModelProfile: "qwen",
    onModelChange: vi.fn(),
    selectedReasoningEffort: "" as const,
    onReasoningChange: vi.fn(),
    approvals: [],
    approvalBusyId: null,
    onApprove: vi.fn(async () => undefined),
    onDeny: vi.fn(async () => undefined),
  };

  it("只为声明支持的模型显示中文思考程度", () => {
    const onReasoningChange = vi.fn();
    const profile = { ...productChatProps.modelStatus.profiles[0], id: "deepseek-flash", label: "DeepSeek Flash", reasoning_efforts: ["low", "medium", "high", "xhigh", "max"] as ReasoningEffort[], default_reasoning_effort: "medium" as const };
    const { container } = render(<ProductChat {...productChatProps} modelStatus={{ ...productChatProps.modelStatus, default_profile: profile.id, profiles: [profile] }} selectedModelProfile={profile.id} selectedReasoningEffort="xhigh" onReasoningChange={onReasoningChange} />);
    const trigger = screen.getByRole("button", { name: "选择模型和思考程度：DeepSeek Flash 极高" });
    expect(screen.queryByRole("slider", { name: "选择思考程度" })).toBeNull();
    fireEvent.click(trigger);
    expect(container.querySelector('[data-icon="bolt"]')).toBeNull();
    const slider = screen.getByRole("slider", { name: "选择思考程度" });
    expect((slider as HTMLInputElement).value).toBe("3");
    expect(screen.getAllByText("极高")).toHaveLength(2);
    fireEvent.change(slider, { target: { value: "4" } });
    expect(onReasoningChange).toHaveBeenCalledWith("max");
  });

  it("Enter 发送，Shift+Enter 保留换行", () => {
    const send = vi.fn(async () => undefined);
    render(<ProductChat {...productChatProps} message="检查数据" send={send} />);
    const input = screen.getByRole("textbox", { name: "输入消息" });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(send).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: "Enter" });
    expect(send).toHaveBeenCalledOnce();
  });

  it("用户消息在正文上方显示本轮文件名", () => {
    const dataset = { id: "file-dem", name: "dem", kind: "RASTER", path: "D:\\workspace\\input\\dem.tif", format: "GeoTIFF" };
    const { container } = render(<ProductChat {...productChatProps} datasets={[dataset]} messages={[{ id: "message-file", role: "user", content: "检查这个数据", datasetIds: [dataset.id] }]} />);
    expect(screen.getByText("dem.tif")).toBeTruthy();
    const message = container.querySelector(".chat-message.user")!;
    expect(message.querySelector(".message-resources")?.nextElementSibling?.textContent).toBe("检查这个数据");
  });

  it("新消息和流式增量自动跟随到底部，用户上翻后暂停跟随", () => {
    vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockReturnValue(1_000);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(300);
    const { container, rerender } = render(<ProductChat {...productChatProps} busy streamingReply="第一段" />);
    const history = container.querySelector(".chat-history") as HTMLDivElement;
    expect(history.scrollTop).toBe(1_000);

    history.scrollTop = 100;
    fireEvent.scroll(history);
    rerender(<ProductChat {...productChatProps} busy streamingReply="第一段和第二段" />);
    expect(history.scrollTop).toBe(100);

    history.scrollTop = 700;
    fireEvent.scroll(history);
    rerender(<ProductChat {...productChatProps} busy streamingReply="第一段、第二段和第三段" />);
    expect(history.scrollTop).toBe(1_000);
  });

});

describe("ApprovalCard", () => {
  it("只展示安全参数摘要并支持批准/拒绝", () => {
    const onApprove = vi.fn(async () => undefined);
    const onDeny = vi.fn(async () => undefined);
    render(<ApprovalCard approval={approval} busy={false} onApprove={onApprove} onDeny={onDeny} />);
    expect(screen.getByText("该操作将生成新的结果数据。")).toBeTruthy();
    expect(screen.getByText(/<已隐藏>/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "批准并继续" }));
    fireEvent.click(screen.getByRole("button", { name: "拒绝" }));
    expect(onApprove).toHaveBeenCalledOnce();
    expect(onDeny).toHaveBeenCalledOnce();
  });

});

describe("MapViewer", () => {
  it("展示矢量预览和数量限制提示", async () => {
    vi.spyOn(api, "datasetPreview").mockResolvedValue({
      dataset_id: "dataset-1", kind: "VECTOR", crs: "EPSG:4326", source_crs: "EPSG:4326", bbox: [0, 0, 1, 1], feature_count: 20, truncated: true,
      geojson: { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "Point", coordinates: [0, 0] }, properties: {} }] },
      columns: [], rows: [],
    });
    render(<MapViewer datasetId="dataset-1" title="道路" />);
    expect(await screen.findByRole("img", { name: "道路地图预览" })).toBeTruthy();
    expect(screen.getByText("仅显示前 1 个要素")).toBeTruthy();
  });

});

const runRecord = (status: Run["status"]): Run => ({
  id: `run-${status.toLowerCase()}`, parent_run_id: null, conversation_id: "conversation-1", task_id: "task-1", agent_id: "main", status,
  turn_count: 0, tool_call_count: 0, metadata: { goal: "检查道路" },
});

describe("RunPanel", () => {
  const renderRunPanel = (status: Run["status"], overrides: Partial<ComponentProps<typeof RunPanel>> = {}) => {
    const props = { runs: [runRecord(status)], selectedRunId: null, events: [], onSelect: vi.fn(async () => undefined), onCancel: vi.fn(async () => undefined), onResume: vi.fn(async () => undefined), onDelete: vi.fn(async () => undefined), onDeleteMany: vi.fn(async () => undefined), busy: false, ...overrides };
    render(<RunPanel {...props} />);
    return props;
  };

  it("中断运行显示恢复，失败运行不显示恢复", () => {
    const props = renderRunPanel("INTERRUPTED");
    fireEvent.click(screen.getByRole("button", { name: "从检查点恢复" }));
    expect(props.onResume).toHaveBeenCalledWith("run-interrupted");
    expect(screen.queryByRole("button", { name: "取消" })).toBeNull();
  });

});

describe("流式回答", () => {
  it("跨工具轮次连续追加正文并只更新过程状态", async () => {
    const user = { id: "user-stream", username: "streamer", display_name: "测试", is_active: true, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" };
    const conversation = { id: "conversation-stream", title: "流式测试", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" };
    vi.spyOn(api, "me").mockResolvedValue(user);
    vi.spyOn(api, "profile").mockResolvedValue({ user_id: user.id, language: "zh-CN", response_style: "balanced", measurement_system: "metric", updated_at: user.updated_at });
    vi.spyOn(api, "datasets").mockResolvedValue([]);
    vi.spyOn(api, "runs").mockResolvedValue([]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "conversations").mockResolvedValue([conversation]);
    vi.spyOn(api, "messages").mockResolvedValue([]);
    vi.spyOn(api, "modelStatus").mockResolvedValue({ configured: true, source: "test", profiles: [] });

    let callbacks!: Parameters<typeof api.streamMessage>;
    let finish!: (response: MessageResponse) => void;
    const pending = Object.assign(new Promise<MessageResponse>((resolve) => { finish = resolve; }), { cancel: vi.fn() });
    vi.spyOn(api, "streamMessage").mockImplementation((...args) => { callbacks = args; return pending; });
    const { container } = render(<App />);
    await screen.findByRole("button", { name: "删除对话 流式测试" });
    const input = screen.getByRole("textbox", { name: "输入消息" });
    fireEvent.change(input, { target: { value: "查看数据" } });
    await waitFor(() => expect((screen.getByRole("button", { name: "发送" }) as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "发送" }));
    const run = { ...runRecord("RUNNING"), conversation_id: conversation.id };
    const started: Event = { id: "start-1", run_id: run.id, event_type: "ModelResponseStarted", message: "正在思考", sequence: 1, timestamp: user.created_at, payload: { turn: 1 } };
    act(() => { callbacks[3](run); callbacks[4](started); });
    expect(screen.getByText("正在思考")).toBeTruthy();
    expect(container.querySelector(".streaming-message")).toBeNull();
    act(() => callbacks[5]("我先查看数据。"));
    expect(container.querySelector(".streaming-message")?.textContent).toBe("我先查看数据。");
    act(() => callbacks[4]({ id: "prepare", run_id: run.id, event_type: "ToolPreparing", message: "正在准备 1 个工具调用", sequence: 2, timestamp: user.created_at, payload: { tools: ["dataset.inspect"], tool_count: 1 } }));
    expect(screen.getByText("正在准备：检查数据集")).toBeTruthy();
    expect(container.querySelector(".streaming-message")?.textContent).toBe("我先查看数据。");
    act(() => callbacks[4]({ ...started, id: "start-2", sequence: 3, payload: { turn: 2 } }));
    act(() => { callbacks[5]("最终"); callbacks[5]("答案。"); });
    expect(container.querySelector(".streaming-message")?.textContent).toBe("我先查看数据。\n\n最终答案。");
    await act(async () => finish({ request_id: "request-stream", route: "execution", message: "最终答案。", run: { ...run, status: "COMPLETED" },
      result: { agent_id: "main", status: "SUCCESS", summary: "最终答案。", findings: [], datasets: [], artifacts: [], evidence: [], warnings: [], trace_id: run.id } }));
    expect(container.querySelector(".streaming-message")).toBeNull();
    expect(container.querySelector(".execution-answer")?.textContent).toBe("我先查看数据。\n\n最终答案。");
    expect(api.streamMessage).toHaveBeenCalledOnce();
  });
});
