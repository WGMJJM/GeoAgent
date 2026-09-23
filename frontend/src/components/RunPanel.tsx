import { useEffect, useState } from "react";
import { Artifact, Dataset, Event, Result, Run, api } from "../api";
import { childRunsOf, eventRuns, groupRunsByTask, isCancellable, isHumanWaiting, isMainRun, isResumable, lineageKind, lineageLabel, lineageSource, runTitle } from "../domain";
import { agentLabel, displayEventMessage, eventLabel, findingText, kindLabel, statusLabel } from "../labels";
import { LineagePanel } from "./LineagePanel";
import { MapViewer } from "./MapViewer";

type RunPanelProps = {
  runs: Run[];
  selectedRunId: string | null;
  events: Event[];
  result?: Result | null;
  datasets?: Dataset[];
  artifacts?: Artifact[];
  onSelect: (id: string) => Promise<void>;
  onCancel: (id: string) => Promise<void>;
  onResume: (id: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onDeleteMany: (ids: string[]) => Promise<void>;
  busy: boolean;
};

export function RunPanel({ runs, selectedRunId, events, result = null, datasets = [], artifacts = [], onSelect, onCancel, onResume, onDelete, onDeleteMany, busy }: RunPanelProps) {
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [bulkDeletePending, setBulkDeletePending] = useState(false);
  const deletableIds = runs.filter((run) => !isCancellable(run)).map((run) => run.id);
  const allSelected = deletableIds.length > 0 && deletableIds.every((id) => selectedIds.includes(id));
  const groups = [...groupRunsByTask(runs).entries()];

  useEffect(() => {
    const currentIds = new Set(runs.map((run) => run.id));
    setSelectedIds((current) => current.filter((id) => currentIds.has(id)));
  }, [runs]);

  const toggleRun = (runId: string) => {
    setSelectedIds((current) => current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId]);
  };

  const toggleAll = () => {
    setSelectedIds(allSelected ? [] : deletableIds);
  };

  const confirmBulkDelete = async () => {
    const ids = [...selectedIds];
    setBulkDeletePending(false);
    setSelectedIds([]);
    await onDeleteMany(ids);
  };

  const renderRun = (run: Run, nested = false) => {
    const humanWaiting = isHumanWaiting(run);
    const cancellable = isCancellable(run);
    const source = lineageSource(run);
    return <div className={`run-row ${nested ? "nested-run" : ""} ${selectedRunId === run.id ? "selected" : ""}`} key={run.id}>
      <label className="run-check" title={cancellable ? "运行中的记录不能删除" : "选择运行记录"}><input type="checkbox" checked={selectedIds.includes(run.id)} disabled={busy || cancellable} onChange={() => toggleRun(run.id)} aria-label={`选择运行记录 ${run.id}`} /></label>
      <button className="run-select" onClick={() => void onSelect(run.id)}><span className={`run-state ${run.status.toLowerCase()}`} /><div><b>{runTitle(run)}</b><small>{run.id} · {agentLabel(run.agent_id)} · {run.tool_call_count} 次工具调用</small>{source && <small className="lineage-note">{lineageLabel(lineageKind(run))}：{source}</small>}{run.parent_run_id && <small className="lineage-note">父运行：{run.parent_run_id}</small>}</div><em>{statusLabel(run.status)}</em></button>
      <div className="run-actions">{cancellable && <button className="small-action danger" onClick={() => void onCancel(run.id)}>{humanWaiting ? "取消任务" : "取消"}</button>}{isResumable(run) && <button className="small-action" disabled={busy} onClick={() => void onResume(run.id)}>从检查点恢复</button>}{!cancellable && <button className="small-action danger" disabled={busy} onClick={() => setPendingDeleteId((current) => current === run.id ? null : run.id)}>删除</button>}</div>
      {pendingDeleteId === run.id && <div className="run-delete-confirm" role="dialog" aria-label={`确认删除运行 ${run.id}`}><span>删除这条运行记录？</span><div><button type="button" className="conversation-confirm-delete" onClick={() => { setPendingDeleteId(null); void onDelete(run.id); }}>删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setPendingDeleteId(null)}>取消</button></div></div>}
    </div>;
  };

  return <section className="panel two-col"><div><div className="panel-head run-panel-head"><div><span className="eyebrow">当前对话运行</span><h2>运行记录</h2></div><div className="run-bulk-actions"><label className="run-select-all"><input type="checkbox" checked={allSelected} disabled={busy || deletableIds.length === 0} onChange={toggleAll} />全选可删除记录</label>{selectedIds.length > 0 && <><span className="run-selected-count">已选 {selectedIds.length} 条</span>{bulkDeletePending ? <div className="run-bulk-confirm"><span>删除已选记录？</span><button type="button" className="conversation-confirm-delete" onClick={() => void confirmBulkDelete()}>确认删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setBulkDeletePending(false)}>取消</button></div> : <button type="button" className="small-action danger" disabled={busy} onClick={() => setBulkDeletePending(true)}>删除已选</button>}</>}</div></div>{runs.length === 0 ? <Empty text="当前对话还没有运行记录。" /> : <div className="run-list">{groups.map(([taskId, taskRuns]) => { const commandGroup = taskId === "__command__"; const mains = taskRuns.filter(isMainRun); const linkedIds = new Set(mains.flatMap((main) => [main.id, ...childRunsOf(main.id, taskRuns).map((child) => child.id)])); const orphanRuns = taskRuns.filter((run) => !linkedIds.has(run.id)); return <section className="run-task-group" key={taskId}><div className="run-task-heading"><b>{commandGroup ? "查询 / 命令运行" : `任务 ${taskId}`}</b>{!commandGroup && <small>{runTitle(mains[0] ?? taskRuns[0])}</small>}</div>{mains.map((main) => <div key={main.id}>{renderRun(main)}{childRunsOf(main.id, taskRuns).map((child) => renderRun(child, true))}</div>)}{orphanRuns.map((run) => renderRun(run, !isMainRun(run)))}</section>; })}</div>}</div><div className="run-detail-stack"><div className="trace-box"><div className="eyebrow">运行追踪 · {selectedRunId ?? "未选择"} · {events.length} 个事件</div>{events.length === 0 ? <Empty text="选择一个运行记录查看事件。" /> : events.map((event) => { const eventRun = eventRuns(event, runs); const eventAgent = eventRun ? (isMainRun(eventRun) ? "主智能体" : `子智能体 · ${runTitle(eventRun)}`) : (event.agent_id === "main" ? "主智能体" : "子智能体"); return <div className="event" key={event.id}><span>{String(event.sequence).padStart(2, "0")}</span><div><b>{eventAgent} · {eventLabel(event.event_type)}</b><small>{displayEventMessage(event.message)}</small></div></div>; })}</div><RunResultDetails result={result} datasets={datasets} artifacts={artifacts} events={events} /></div></section>;
}

function RunResultDetails({ result, datasets, artifacts, events }: { result: Result | null; datasets: Dataset[]; artifacts: Artifact[]; events: Event[] }) {
  const output = result?.datasets.map((id) => datasets.find((dataset) => dataset.id === id)).find((dataset) => dataset?.kind === "VECTOR" || dataset?.kind === "RASTER");
  const names = Object.fromEntries(datasets.map((dataset) => [dataset.id, dataset.name]));
  return <section className="run-result-details">{!result ? <Empty text="选择一次运行后查看结果、证据和输出数据。" /> : <><div className="result-head"><div><span className={`pill ${result.status.toLowerCase()}`}>{statusLabel(result.status)}</span><h2>{result.summary}</h2></div><code>{result.trace_id}</code></div>{result.error && <div className="result-error"><b>错误</b><span>{result.error}</span></div>}<div className="result-columns"><div><h3>分析发现</h3>{result.findings.length === 0 ? <Empty text="没有结构化发现。" /> : result.findings.map((finding, index) => <pre key={index}>{findingText(finding)}</pre>)}<h3>关联数据集</h3>{result.datasets.length === 0 ? <p className="muted-text">本次运行没有关联数据集。</p> : <div className="dataset-result-list">{result.datasets.map((datasetId) => { const dataset = datasets.find((item) => item.id === datasetId); return <div className="dataset-result-item" key={datasetId}><b>{dataset?.name ?? datasetId}</b><small>{dataset?.kind ? kindLabel(dataset.kind) : "数据集"} · {datasetId}</small></div>; })}</div>}<h3>结果文件</h3>{result.artifacts.length === 0 ? <p className="muted-text">本次运行没有产物。</p> : <div className="artifact-list">{result.artifacts.map((artifactId) => { const artifact = artifacts.find((item) => item.id === artifactId); return <a className="artifact-link" href={api.artifactUrl(artifactId)} target="_blank" rel="noreferrer" key={artifactId}>{artifact?.name ?? artifactId} <span>↗</span></a>; })}</div>}{result.evidence.length > 0 && <><h3>证据</h3>{result.evidence.map((evidence, index) => <pre key={index}>{findingText(evidence)}</pre>)}</>}</div><div><h3>执行情况</h3><div className="metric"><b>{events.length}</b><span>追踪事件</span></div><div className="metric"><b>{result.datasets.length}</b><span>涉及数据集</span></div><div className="metric"><b>{result.artifacts.length}</b><span>结果文件</span></div>{result.warnings.length > 0 && <><h3>警告</h3>{result.warnings.map((warning) => <p className="warning" key={warning}>{warning}</p>)}</>}</div></div>{output && <div className="result-map-panel"><div className="panel-head"><div><span className="eyebrow">GIS 结果</span><h3>地图与来源</h3></div><span className="count-badge">{output.name}</span></div><MapViewer datasetId={output.id} title={output.name} /><LineagePanel datasetId={output.id} datasetNames={names} /></div>}</>}</section>;
}

function Empty({ text }: { text: string }) {
  return <div className="empty">{text}</div>;
}
