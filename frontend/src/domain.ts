import { Event, Run, RunStatus } from "./api";

export const EXECUTION_INFLIGHT_STATUSES: ReadonlySet<RunStatus> = new Set([
  "RUNNING",
  "WAITING_TOOL",
  "WAITING_SUBAGENT",
  "RETRYING",
]);

export const HUMAN_WAITING_STATUSES: ReadonlySet<RunStatus> = new Set(["WAITING_USER", "WAITING_APPROVAL"]);

export const ACTIVE_RUN_STATUSES: ReadonlySet<RunStatus> = new Set(["CREATED", ...EXECUTION_INFLIGHT_STATUSES, ...HUMAN_WAITING_STATUSES]);

export const TERMINAL_RUN_STATUSES: ReadonlySet<RunStatus> = new Set([
  "COMPLETED",
  "PARTIAL_COMPLETED",
  "FAILED",
  "INTERRUPTED",
  "CANCELLED",
  "BUDGET_EXCEEDED",
]);

export const RESUMABLE_RUN_STATUSES: ReadonlySet<RunStatus> = new Set(["CANCELLED", "INTERRUPTED", "BUDGET_EXCEEDED"]);

export function isActiveRun(run: Run): boolean {
  return ACTIVE_RUN_STATUSES.has(run.status);
}

export function isExecutionInflight(run: Run): boolean {
  return EXECUTION_INFLIGHT_STATUSES.has(run.status);
}

export function isHumanWaiting(run: Run): boolean {
  return HUMAN_WAITING_STATUSES.has(run.status);
}

export function isCancellable(run: Run): boolean {
  return isExecutionInflight(run) || isHumanWaiting(run);
}

export function isTerminal(run: Run): boolean {
  return ["COMPLETED", "PARTIAL_COMPLETED", "FAILED", "CANCELLED", "BUDGET_EXCEEDED"].includes(run.status);
}

export function isResumable(run: Run): boolean {
  return RESUMABLE_RUN_STATUSES.has(run.status);
}

export function runDurationMs(run: Run, now = Date.now()): number {
  if (!run.started_at) return 0;
  const started = Date.parse(run.started_at);
  if (!Number.isFinite(started)) return 0;
  const finished = run.finished_at ? Date.parse(run.finished_at) : now;
  return Math.max(0, (Number.isFinite(finished) ? finished : now) - started);
}

export function isMainRun(run: Run): boolean {
  return run.parent_run_id === null || run.parent_run_id === undefined;
}

export function childRunsOf(runId: string, runs: Run[]): Run[] {
  return runs.filter((run) => run.parent_run_id === runId);
}

export function runsForConversation(conversationId: string, runs: Run[]): Run[] {
  return conversationId ? runs.filter((run) => run.conversation_id === conversationId) : [];
}

export type LineageKind = "continued_from" | "retry_of" | "resumed_from" | "approved_from";

export type RunLineage = { kind: LineageKind; sourceRunId: string };

export function runLineage(run: Run): RunLineage | null {
  const sources: Array<[LineageKind, unknown]> = [
    ["continued_from", run.metadata.continued_from],
    ["retry_of", run.metadata.retry_of],
    ["resumed_from", run.metadata.resumed_from],
    ["approved_from", run.metadata.approved_from],
  ];
  const source = sources.find(([, value]) => typeof value === "string" && value.length > 0);
  return source ? { kind: source[0], sourceRunId: source[1] as string } : null;
}

export function lineageSource(run: Run): string | null {
  return runLineage(run)?.sourceRunId ?? null;
}

export function lineageKind(run: Run): LineageKind | null {
  return runLineage(run)?.kind ?? null;
}

export function lineageLabel(kind: LineageKind | null): string {
  return kind === "continued_from" ? "继续自" : kind === "retry_of" ? "重试自" : kind === "resumed_from" ? "恢复自" : kind === "approved_from" ? "审批继续自" : "";
}

export function runTitle(run: Run): string {
  return String(run.metadata.subtask_goal ?? run.metadata.goal ?? run.agent_id);
}

export function eventRuns(event: Event, runs: Run[]): Run | undefined {
  return runs.find((run) => run.id === event.run_id);
}

export function groupRunsByTask(runs: Run[]): Map<string, Run[]> {
  const groups = new Map<string, Run[]>();
  for (const run of runs) {
    const key = run.task_id ?? "__command__";
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }
  return groups;
}
