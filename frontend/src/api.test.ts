import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("前端 API 契约", () => {
  it("实时消息直接进入统一对话循环并提交本轮资源", async () => {
    class FakeWebSocket {
      static OPEN = 1;
      readyState = FakeWebSocket.OPEN;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      sent: string[] = [];
      close = vi.fn();
      send = (payload: string) => {
        this.sent.push(payload);
        this.onmessage?.({ data: JSON.stringify({ type: "response", data: { request_id: "request-1", route: "execution", message: "已处理" } }) } as MessageEvent);
      };
      constructor() {
        queueMicrotask(() => this.onopen?.());
      }
    }
    const socket = new FakeWebSocket();
    vi.stubGlobal("WebSocket", vi.fn(() => socket));
    await api.streamMessage("规划道路缓冲区", ["dataset-1"], ["dataset-2"], vi.fn(), vi.fn(), vi.fn(), "conversation-1", "qwen", undefined, "run-waiting");
    expect(JSON.parse(socket.sent[0])).toMatchObject({ message: "规划道路缓冲区", conversation_id: "conversation-1", dataset_ids: ["dataset-1"], attachment_ids: ["dataset-2"], model_profile: "qwen", reply_to_run_id: "run-waiting" });
    expect(JSON.parse(socket.sent[0])).not.toHaveProperty("execution_mode");
    expect(JSON.parse(socket.sent[0])).not.toHaveProperty("planning_session_id");
  });

  it("审批接口保持 approve/deny 路径", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ approval: { id: "approval-1", status: "APPROVED" }, run: null, result: null }), { status: 200, headers: { "Content-Type": "application/json" } }));
    await api.approve("approval-1");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/approvals/approval-1/approve");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST");
  });

  it("保留 HTTP 状态供界面解释 409", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ detail: "运行状态不支持恢复" }), { status: 409 }));
    await expect(api.resumeRun("run-1")).rejects.toMatchObject({ status: 409, message: "运行状态不支持恢复" });
  });

  it("不同请求使用独立 WebSocket，流式回调不会串线", async () => {
    class FakeWebSocket {
      static OPEN = 1;
      static instances: FakeWebSocket[] = [];
      readyState = FakeWebSocket.OPEN;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      sent: string[] = [];
      close = vi.fn();
      send = (payload: string) => this.sent.push(payload);
      constructor() {
        FakeWebSocket.instances.push(this);
        queueMicrotask(() => this.onopen?.());
      }
    }
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const first = vi.fn();
    const second = vi.fn();
    const firstRequest = api.streamMessage("任务 A", [], [], vi.fn(), vi.fn(), first, "conversation-a");
    const secondRequest = api.streamMessage("任务 B", [], [], vi.fn(), vi.fn(), second, "conversation-b");
    await Promise.resolve();
    expect(FakeWebSocket.instances).toHaveLength(2);
    const [socketA, socketB] = FakeWebSocket.instances;
    socketA.onmessage?.({ data: JSON.stringify({ type: "delta", content: "AAA" }) } as MessageEvent);
    socketB.onmessage?.({ data: JSON.stringify({ type: "delta", content: "BBB" }) } as MessageEvent);
    socketA.onmessage?.({ data: JSON.stringify({ type: "response", data: { request_id: "a", route: "direct", message: "完成 A" } }) } as MessageEvent);
    socketB.onmessage?.({ data: JSON.stringify({ type: "response", data: { request_id: "b", route: "direct", message: "完成 B" } }) } as MessageEvent);
    await expect(firstRequest).resolves.toMatchObject({ request_id: "a" });
    await expect(secondRequest).resolves.toMatchObject({ request_id: "b" });
    expect(first).toHaveBeenCalledWith("AAA");
    expect(second).toHaveBeenCalledWith("BBB");
  });

  it("HTTP 请求在超时后终止而不是无限等待", async () => {
    vi.useFakeTimers();
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("请求超时", "AbortError")));
    }));
    const pending = api.run("run-timeout");
    const rejection = expect(pending).rejects.toMatchObject({ name: "AbortError" });
    await vi.advanceTimersByTimeAsync(15_001);
    await rejection;
  });

  it("WebSocket 长时间无消息时按请求独立触发 watchdog", async () => {
    vi.useFakeTimers();
    class SilentWebSocket {
      static OPEN = 1;
      readyState = SilentWebSocket.OPEN;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      close = vi.fn();
      send = vi.fn();
      constructor() { queueMicrotask(() => this.onopen?.()); }
    }
    vi.stubGlobal("WebSocket", SilentWebSocket);
    const pending = api.streamMessage("长任务", [], [], vi.fn(), vi.fn(), vi.fn(), "conversation-timeout");
    const rejection = expect(pending).rejects.toThrow("长时间无响应");
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(60_001);
    await rejection;
  });

  it("WebSocket heartbeat 会刷新 watchdog 活跃时间", async () => {
    vi.useFakeTimers();
    class HeartbeatWebSocket {
      static OPEN = 1;
      static instance: HeartbeatWebSocket;
      readyState = HeartbeatWebSocket.OPEN;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      close = vi.fn();
      send = vi.fn();
      constructor() {
        HeartbeatWebSocket.instance = this;
        queueMicrotask(() => this.onopen?.());
      }
    }
    vi.stubGlobal("WebSocket", HeartbeatWebSocket);
    const pending = api.streamMessage("心跳任务", [], [], vi.fn(), vi.fn(), vi.fn(), "conversation-heartbeat");
    const rejection = expect(pending).rejects.toThrow("请求已取消");
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(50_000);
    HeartbeatWebSocket.instance.onmessage?.({ data: JSON.stringify({ type: "heartbeat" }) } as MessageEvent);
    await vi.advanceTimersByTimeAsync(50_000);
    expect(HeartbeatWebSocket.instance.close).not.toHaveBeenCalled();
    pending.cancel();
    await rejection;
  });

  it("heartbeat 不伪装成执行进度，真实事件才更新进度回调", async () => {
    class ProgressWebSocket {
      static OPEN = 1;
      static instance: ProgressWebSocket;
      readyState = ProgressWebSocket.OPEN;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      close = vi.fn();
      send = vi.fn();
      constructor() {
        ProgressWebSocket.instance = this;
        queueMicrotask(() => this.onopen?.());
      }
    }
    vi.stubGlobal("WebSocket", ProgressWebSocket);
    const onProgress = vi.fn();
    const pending = api.streamMessage("进度任务", [], [], vi.fn(), vi.fn(), vi.fn(), "conversation-progress", undefined, onProgress);
    await Promise.resolve();
    ProgressWebSocket.instance.onmessage?.({ data: JSON.stringify({ type: "heartbeat" }) } as MessageEvent);
    expect(onProgress).not.toHaveBeenCalled();
    ProgressWebSocket.instance.onmessage?.({ data: JSON.stringify({ type: "event", data: { id: "event-1" } }) } as MessageEvent);
    ProgressWebSocket.instance.onmessage?.({ data: JSON.stringify({ type: "delta", content: "处理中" }) } as MessageEvent);
    expect(onProgress).toHaveBeenCalledTimes(2);
    pending.cancel();
    await expect(pending).rejects.toThrow("请求已取消");
  });
});
