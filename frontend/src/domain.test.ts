import { describe, expect, it } from "vitest";
import { lineageKind, lineageLabel, lineageSource, isCancellable, isExecutionInflight, isHumanWaiting, isResumable, isTerminal, runDurationMs } from "./domain";
import { Run } from "./api";

const run = (status: Run["status"], metadata: Run["metadata"] = {}): Run => ({
  id: "run-1",
  parent_run_id: null,
  conversation_id: "conversation-1",
  task_id: "task-1",
  agent_id: "main",
  status,
  turn_count: 0,
  tool_call_count: 0,
  metadata,
});

describe("运行状态谓词", () => {
  it("区分执行中、人工等待、可恢复和终态", () => {
    expect(isExecutionInflight(run("RUNNING"))).toBe(true);
    expect(isHumanWaiting(run("WAITING_USER"))).toBe(true);
    expect(isCancellable(run("WAITING_APPROVAL"))).toBe(true);
    expect(isResumable(run("INTERRUPTED"))).toBe(true);
    expect(isResumable(run("FAILED"))).toBe(false);
    expect(isTerminal(run("COMPLETED"))).toBe(true);
  });
});

describe("运行 lineage 展示", () => {
  it.each([
    ["continued_from", "继续自"],
    ["retry_of", "重试自"],
    ["resumed_from", "恢复自"],
    ["approved_from", "审批继续自"],
  ] as const)("识别 %s", (key, label) => {
    const value = run("COMPLETED", { [key]: "run-old" });
    expect(lineageKind(value)).toBe(key);
    expect(lineageLabel(lineageKind(value))).toBe(label);
    expect(lineageSource(value)).toBe("run-old");
  });
});

describe("运行摘要耗时", () => {
  it("使用 Run 的开始和结束时间计算耗时", () => {
    const value = run("COMPLETED");
    value.started_at = "2026-01-01T10:00:00.000Z";
    value.finished_at = "2026-01-01T10:00:18.000Z";
    expect(runDurationMs(value)).toBe(18_000);
  });
});
