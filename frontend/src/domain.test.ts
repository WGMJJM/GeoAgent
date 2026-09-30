import { describe, expect, it } from "vitest";
import { isCancellable, isExecutionInflight, isHumanWaiting, isResumable, isTerminal } from "./domain";
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
