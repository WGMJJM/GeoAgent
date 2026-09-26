import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApprovalCard } from "./components/ApprovalCard";
import { api, ApprovalRequest, Event, Run, TokenUsage } from "./api";
import { MapViewer } from "./components/MapViewer";
import { RunPanel } from "./components/RunPanel";
import { App, CompletedRunSummary, LiveExecutionStatus, ProductChat, TokenUsageSummary } from "./App";

afterEach(() => vi.restoreAllMocks());

const approval: ApprovalRequest = { id: "approval-1", user_id: "user-1", conversation_id: "conv-1", task_id: "task-1", source_run_id: "run-1", tool_call_id: "call-1", tool_name: "vector.buffer", argument_fingerprint: "fingerprint", risk_level: "WRITE", argument_preview: { distance: "500m", path: "<已隐藏>" }, reason: "该操作将生成新的结果数据。", status: "PENDING", created_at: "2026-01-01T00:00:00Z" };

describe("运行累计 Token 用量", () => {
  const usage: TokenUsage = { local_input_tokens: 900, local_output_tokens: 100, reported_input_tokens: 1200, reported_output_tokens: 250, model_calls: 2, reported_calls: 2 };

  it("完成摘要在工具调用数后显示供应商实际累计量", () => {
    const { container } = render(<CompletedRunSummary run={{ ...runRecord("COMPLETED"), tool_call_count: 9, token_usage: usage }} />);
    expect(screen.getByText("运行完成 · 9 次工具调用")).toBeTruthy();
    expect(screen.getByText("Token 1,450")).toBeTruthy();
    expect(screen.getByText("（输入 1,200 / 输出 250）")).toBeTruthy();
    expect(container.querySelector(".run-summary-current .run-token-usage")?.getAttribute("title")).toContain("模型返回的实际用量");
  });

  it("任一轮缺少 usage 时不混合实际量和估算量", () => {
    render(<TokenUsageSummary usage={{ ...usage, reported_calls: 1 }} />);
    expect(screen.getByText("Token（本地估算） 1,000")).toBeTruthy();
    expect(screen.getByText("（输入 900 / 输出 100）")).toBeTruthy();
  });

  it("没有用量的旧 Run 不显示虚假的零消耗，真实零用量可以显示", () => {
    const { container, rerender } = render(<TokenUsageSummary />);
    expect(container.querySelector(".run-token-usage")).toBeNull();
    rerender(<TokenUsageSummary usage={{ ...usage, reported_input_tokens: 0, reported_output_tokens: 0 }} />);
    expect(screen.getByText("Token 0")).toBeTruthy();
  });

  it("实时更新使用累计快照，重复事件不重复累加，也不覆盖当前执行状态", () => {
    const progress: Event = { id: "progress", run_id: "run-1", event_type: "ToolStarted", sequence: 1, timestamp: "2026-01-01T10:00:00Z", message: "检查栅格", payload: {} };
    const measured: Event = { ...progress, id: "usage", event_type: "TokenUsageUpdated", sequence: 2, payload: { token_usage: usage } };
    const { rerender } = render(<LiveExecutionStatus phase="running" events={[progress]} durationMs={1000} tokenUsage={{ ...usage, reported_calls: 1 }} />);
    expect(screen.getByText("Token（本地估算） 1,000")).toBeTruthy();
    rerender(<LiveExecutionStatus phase="running" events={[progress, measured, measured]} durationMs={2000} />);
    expect(screen.getByText("Token 1,450")).toBeTruthy();
    expect(screen.getByText("工具开始执行：检查栅格")).toBeTruthy();
    expect(screen.queryByText(/模型累计用量已更新/)).toBeNull();
    const older: Event = { ...measured, id: "older", payload: { token_usage: { ...usage, model_calls: 1, reported_calls: 1, reported_input_tokens: 100 } } };
    rerender(<LiveExecutionStatus phase="running" events={[progress, measured, older]} durationMs={2000} />);
    expect(screen.getByText("Token 1,450")).toBeTruthy(); // 并行子运行事件晚到不能使计数倒退。
    rerender(<LiveExecutionStatus phase="running" events={[progress]} durationMs={0} />);
    expect(screen.queryByText(/Token/)).toBeNull();
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
    modelStatus: { configured: true, source: "test", default_profile: "qwen", profiles: [{ id: "qwen", label: "通义千问", provider: "openai-compatible", model: "qwen", timeout_seconds: 60, temperature: 0.2, has_api_key: true, default: true }] },
    selectedModelProfile: "qwen",
    onModelChange: vi.fn(),
    approvals: [],
    approvalBusyId: null,
    onApprove: vi.fn(async () => undefined),
    onDeny: vi.fn(async () => undefined),
  };

  it("使用同一条消息时间线及输入能力，不显示独立规划入口", () => {
    render(<ProductChat {...productChatProps} />);
    expect(screen.getByText("历史消息")).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "输入消息" })).toBeTruthy();
    expect(screen.getByLabelText("添加文件")).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "选择模型" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "规划" })).toBeNull();
    expect(screen.getByText("历史消息")).toBeTruthy();
    expect(screen.getAllByRole("textbox")).toHaveLength(1);
  });

  it("连接阶段从发送开始显示统一状态和计时", () => {
    render(<ProductChat {...productChatProps} busy activeRunId={null} elapsedMs={2_400} />);
    expect(screen.getByText("用时 2秒")).toBeTruthy();
    expect(screen.getByText("正在处理")).toBeTruthy();
    expect(screen.getByText("正在接入 Agent Loop…")).toBeTruthy();
    expect(screen.queryByText("等待智能体事件…")).toBeNull();
  });

  it("真实运行显示最新请求理解事件", () => {
    const decision: Event = { id: "event-decision", run_id: "run-1", event_type: "DecisionMade", message: "准备调用 dataset.list", sequence: 2, timestamp: "2026-01-01T10:00:05.000Z", payload: {}, agent_id: "agent-loop" };
    render(<ProductChat {...productChatProps} busy activeRunId="run-1" elapsedMs={5_100} events={[decision]} />);
    expect(screen.getByText("用时 5秒")).toBeTruthy();
    expect(screen.getByText("正在运行")).toBeTruthy();
    expect(screen.getByText("已确定下一步动作：准备调用 列出数据集")).toBeTruthy();
  });

  it("真实运行已建立但暂无事件时显示等待提示", () => {
    render(<ProductChat {...productChatProps} busy activeRunId="run-1" elapsedMs={5_100} />);
    expect(screen.getByText("等待智能体事件…")).toBeTruthy();
  });

  it("真实运行消息只提供运行详情入口", () => {
    render(<ProductChat {...productChatProps} messages={[{ id: "run-message", role: "assistant", content: "运行完成", kind: "execution", runId: "run-1" }]} />);
    expect(screen.getByRole("button", { name: "查看运行详情" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "查看详细结果" })).toBeNull();
  });

  it("完成运行从 runs 缓存派生摘要和真实耗时", () => {
    const completed: Run = { ...runRecord("COMPLETED"), started_at: "2026-01-01T10:00:00.000Z", finished_at: "2026-01-01T10:00:18.000Z", tool_call_count: 2 };
    const { container } = render(<ProductChat {...productChatProps} runs={[completed]} messages={[{ id: "completed-message", role: "assistant", content: "道路处理完成", kind: "execution", runId: completed.id }]} />);
    expect(screen.getByText("用时 18秒")).toBeTruthy();
    expect(screen.getByText("运行完成 · 2 次工具调用")).toBeTruthy();
    expect(screen.getByText("已完成")).toBeTruthy();
    expect(screen.queryByText("等待智能体事件…")).toBeNull();
    expect(container.querySelector(".execution-message")).toBeTruthy();
  });

  it("失败运行仍显示摘要和运行详情入口", () => {
    const failed: Run = { ...runRecord("FAILED"), started_at: "2026-01-01T10:00:00.000Z", finished_at: "2026-01-01T10:00:36.000Z", tool_call_count: 2 };
    render(<ProductChat {...productChatProps} runs={[failed]} messages={[{ id: "failed-message", role: "assistant", content: "运行失败", kind: "execution", runId: failed.id }]} />);
    expect(screen.getByText("用时 36秒")).toBeTruthy();
    expect(screen.getByText("运行未完成 · 2 次工具调用")).toBeTruthy();
    expect(screen.getByText("失败")).toBeTruthy();
    expect(screen.getByRole("button", { name: "查看运行详情" })).toBeTruthy();
  });

  it("快速等待用户的运行显示小于一秒而不是零秒", () => {
    const waiting: Run = { ...runRecord("WAITING_USER"), started_at: "2026-01-01T10:00:00.000Z", finished_at: "2026-01-01T10:00:00.420Z" };
    const onReplyToRun = vi.fn();
    render(<ProductChat {...productChatProps} onReplyToRun={onReplyToRun} runs={[waiting]} messages={[{ id: "waiting-message", role: "assistant", content: "当前请求需要补充信息后才能继续。", kind: "execution", runId: waiting.id }]} />);
    expect(screen.getByText("用时 <1秒")).toBeTruthy();
    expect(screen.getByText("等待补充信息 · 0 次工具调用")).toBeTruthy();
    expect(screen.getByText("当前请求需要补充信息后才能继续。")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "继续此运行" }));
    expect(onReplyToRun).toHaveBeenCalledWith(waiting.id);
  });

  it("实时进度会直接切换为完成摘要", () => {
    const completed: Run = { ...runRecord("COMPLETED"), id: "run-transition", started_at: "2026-01-01T10:00:00.000Z", finished_at: "2026-01-01T10:00:03.000Z" };
    const { rerender } = render(<ProductChat {...productChatProps} busy activeRunId={completed.id} elapsedMs={3_000} />);
    expect(screen.getByText("等待智能体事件…")).toBeTruthy();
    rerender(<ProductChat {...productChatProps} runs={[completed]} messages={[{ id: "transition-message", role: "assistant", content: "处理完成", kind: "execution", runId: completed.id }]} />);
    expect(screen.queryByText("等待智能体事件…")).toBeNull();
    expect(screen.getByText("运行完成 · 0 次工具调用")).toBeTruthy();
  });

  it("普通对话和直接查询回复不保留运行摘要", () => {
    const { container } = render(<ProductChat {...productChatProps} messages={[{ id: "direct-message", role: "assistant", content: "当前登记了 1 个数据集。", kind: "text" }]} />);
    expect(container.querySelector(".execution-message")).toBeNull();
    expect(screen.queryByText(/运行完成 ·/)).toBeNull();
    expect(screen.queryByRole("button", { name: "查看运行详情" })).toBeNull();
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

  it("已拒绝后不显示执行按钮", () => {
    render(<ApprovalCard approval={{ ...approval, status: "DENIED" }} busy={false} onApprove={vi.fn(async () => undefined)} onDeny={vi.fn(async () => undefined)} />);
    expect(screen.queryByRole("button", { name: "批准并继续" })).toBeNull();
    expect(screen.getByText("已拒绝该操作")).toBeTruthy();
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

  it("局部展示预览错误而不影响页面", async () => {
    vi.spyOn(api, "datasetPreview").mockRejectedValue(new Error("数据不存在"));
    render(<MapViewer datasetId="missing" />);
    expect(await screen.findByText("预览失败：数据不存在")).toBeTruthy();
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

  it("运行中允许取消但不显示恢复", () => {
    const props = renderRunPanel("RUNNING");
    fireEvent.click(screen.getByRole("button", { name: "取消" }));
    expect(props.onCancel).toHaveBeenCalledWith("run-running");
    expect(screen.queryByRole("button", { name: "从检查点恢复" })).toBeNull();
  });

  it("中断运行显示恢复，失败运行不显示恢复", () => {
    const props = renderRunPanel("INTERRUPTED");
    fireEvent.click(screen.getByRole("button", { name: "从检查点恢复" }));
    expect(props.onResume).toHaveBeenCalledWith("run-interrupted");
    expect(screen.queryByRole("button", { name: "取消" })).toBeNull();
  });

  it("等待用户时显示取消任务而不是技术恢复", () => {
    renderRunPanel("WAITING_USER");
    expect(screen.getByRole("button", { name: "取消任务" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "从检查点恢复" })).toBeNull();
  });
});

describe("对话删除", () => {
  it("运行或等待中的对话仍可发起删除", async () => {
    const user = { id: "user-1", username: "tester", display_name: "测试用户", is_active: true, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" };
    const conversation = { id: "conversation-running", title: "正在处理", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" };
    vi.spyOn(api, "me").mockResolvedValue(user);
    vi.spyOn(api, "profile").mockResolvedValue({ user_id: user.id, language: "zh-CN", response_style: "balanced", measurement_system: "metric", updated_at: "2026-01-01T00:00:00Z" });
    vi.spyOn(api, "datasets").mockResolvedValue([]);
    vi.spyOn(api, "runs").mockResolvedValue([{ ...runRecord("WAITING_USER"), conversation_id: conversation.id }]);
    vi.spyOn(api, "approvals").mockResolvedValue([]);
    vi.spyOn(api, "conversations").mockResolvedValue([conversation]);
    vi.spyOn(api, "messages").mockResolvedValue([]);
    vi.spyOn(api, "modelStatus").mockResolvedValue({ configured: false, source: "test", profiles: [] });

    render(<App />);

    const remove = await screen.findByRole("button", { name: "删除对话 正在处理" });
    await waitFor(() => expect((remove as HTMLButtonElement).disabled).toBe(false));
    expect(remove.getAttribute("title")).toBe("删除对话");
    fireEvent.click(remove);
    expect(screen.getByRole("button", { name: "删除" })).toBeTruthy();
  });
});
