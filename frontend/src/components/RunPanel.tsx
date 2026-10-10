import { useEffect, useState } from "react";
import { Artifact, Dataset, Event, Result, Run, api } from "../api";
import { childRunsOf, eventRuns, groupRunsByTask, isCancellable, isHumanWaiting, isMainRun, isResumable, isRetryable, lineageKind, lineageLabel, lineageSource, runTitle } from "../domain";
import { agentLabel, displayEventMessage, eventLabel, findingText, kindLabel, statusLabel } from "../labels";
import { LineagePanel } from "./LineagePanel";
import { DatasetPreviewButton, MapViewer } from "./MapViewer";
import { Icon } from "./Icon";
import { MarkdownContent } from "./MarkdownContent";
import { DatasetActions } from "./DatasetActions";
import "./RunPanel.css";

type RunPanelProps = {
  selectedDatasetIds?: string[];
  onUseDataset?: (id: string) => void;
  runs: Run[];
  selectedRunId: string | null;
  events: Event[];
  result?: Result | null;
  datasets?: Dataset[];
  artifacts?: Artifact[];
  onSelect: (id: string) => Promise<void>;
  onCancel: (id: string) => Promise<void>;
  onResume: (id: string) => Promise<void>;
  onRetry: (id: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onDeleteMany: (ids: string[]) => Promise<void>;
  busy: boolean;
};

export function RunPanel({ selectedDatasetIds = [], onUseDataset, runs, selectedRunId, events, result = null, datasets = [], artifacts = [], onSelect, onCancel, onResume, onRetry, onDelete, onDeleteMany, busy }: RunPanelProps) {
  const visibleEvents = events.filter((event) => !["completion_review", "model_protocol", "task_association"].includes(String(event.payload.scope)));
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [pendingRetryId, setPendingRetryId] = useState<string | null>(null);
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
      <div className="run-actions">{cancellable && <button className="small-action danger" onClick={() => void onCancel(run.id)}>{humanWaiting ? "取消任务" : "取消"}</button>}{isResumable(run) && <button className="small-action" disabled={busy} onClick={() => void onResume(run.id)}>从检查点恢复</button>}{isRetryable(run) && <button className="small-action" disabled={busy} onClick={() => setPendingRetryId((current) => current === run.id ? null : run.id)}>重试原请求</button>}{!cancellable && <button className="small-action danger" disabled={busy} onClick={() => setPendingDeleteId((current) => current === run.id ? null : run.id)}>删除</button>}</div>
      {pendingRetryId === run.id && <div className="run-delete-confirm" role="dialog" aria-label={`确认重试运行 ${run.id}`}><span>将重新发起原请求，可能再次执行已完成的操作。旧运行和结果会保留。</span><div><button type="button" className="small-action" disabled={busy} onClick={() => { setPendingRetryId(null); void onRetry(run.id); }}>确认重试</button><button type="button" className="conversation-confirm-cancel" onClick={() => setPendingRetryId(null)}>取消</button></div></div>}
      {pendingDeleteId === run.id && <div className="run-delete-confirm" role="dialog" aria-label={`确认删除运行 ${run.id}`}><span>删除这条运行记录？</span><div><button type="button" className="conversation-confirm-delete" onClick={() => { setPendingDeleteId(null); void onDelete(run.id); }}>删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setPendingDeleteId(null)}>取消</button></div></div>}
    </div>;
  };

  return <section className="panel two-col"><div><div className="panel-head run-panel-head"><div><span className="eyebrow">当前对话运行</span><h2>运行记录</h2></div><div className="run-bulk-actions"><label className="run-select-all"><input type="checkbox" checked={allSelected} disabled={busy || deletableIds.length === 0} onChange={toggleAll} />全选可删除记录</label>{selectedIds.length > 0 && <><span className="run-selected-count">已选 {selectedIds.length} 条</span>{bulkDeletePending ? <div className="run-bulk-confirm"><span>删除已选记录？</span><button type="button" className="conversation-confirm-delete" onClick={() => void confirmBulkDelete()}>确认删除</button><button type="button" className="conversation-confirm-cancel" onClick={() => setBulkDeletePending(false)}>取消</button></div> : <button type="button" className="small-action danger" disabled={busy} onClick={() => setBulkDeletePending(true)}>删除已选</button>}</>}</div></div>{runs.length === 0 ? <Empty text="当前对话还没有运行记录。" /> : <div className="run-list">{groups.map(([taskId, taskRuns]) => { const commandGroup = taskId === "__command__"; const mains = taskRuns.filter(isMainRun); const linkedIds = new Set(mains.flatMap((main) => [main.id, ...childRunsOf(main.id, taskRuns).map((child) => child.id)])); const orphanRuns = taskRuns.filter((run) => !linkedIds.has(run.id)); return <section className="run-task-group" key={taskId}><div className="run-task-heading"><b>{commandGroup ? "查询 / 命令运行" : `任务 ${taskId}`}</b>{!commandGroup && <small>{runTitle(mains[0] ?? taskRuns[0])}</small>}</div>{mains.map((main) => <div key={main.id}>{renderRun(main)}{childRunsOf(main.id, taskRuns).map((child) => renderRun(child, true))}</div>)}{orphanRuns.map((run) => renderRun(run, !isMainRun(run)))}</section>; })}</div>}</div><div className="run-detail-stack"><RunResultDetails result={result} datasets={datasets} artifacts={artifacts} selectedDatasetIds={selectedDatasetIds} onUseDataset={onUseDataset} /><details className="technical-details run-trace"><summary>执行记录 · {visibleEvents.length} 个事件</summary><div className="trace-box"><div className="eyebrow">运行追踪 · {selectedRunId ?? "未选择"} · {visibleEvents.length} 个事件</div>{visibleEvents.length === 0 ? <Empty text="选择一个运行记录查看事件。" /> : visibleEvents.map((event) => { const eventRun = eventRuns(event, runs); const eventAgent = eventRun ? (isMainRun(eventRun) ? "主智能体" : `子智能体 · ${runTitle(eventRun)}`) : (event.agent_id === "main" ? "主智能体" : "子智能体"); return <div className="event" key={event.id}><span>{String(event.sequence).padStart(2, "0")}</span><div><b>{eventAgent} · {eventLabel(event.event_type)}</b><small>{displayEventMessage(event.message)}</small></div></div>; })}</div></details></div></section>;
}

function RunResultDetails({ result, datasets, artifacts, selectedDatasetIds, onUseDataset }: {
  result: Result | null; datasets: Dataset[]; artifacts: Artifact[];
  selectedDatasetIds: string[]; onUseDataset?: (id: string) => void;
}) {
  const [previewSelection, setPreviewSelection] = useState<{ traceId: string; datasetId: string } | null>(null);
  const previewDatasets = (result?.datasets ?? []).map((id) => datasets.find((dataset) => dataset.id === id))
    .filter((dataset): dataset is Dataset => Boolean(dataset && ["VECTOR", "RASTER", "IMAGE", "TABLE", "DOCUMENT"].includes(dataset.kind)));
  const output = previewDatasets.find((dataset) => previewSelection?.traceId === result?.trace_id && dataset.id === previewSelection?.datasetId) ?? previewDatasets[0];
  const names = Object.fromEntries(datasets.map((dataset) => [dataset.id, dataset.name]));
  if (!result) return <section className="run-result-details"><Empty text="选择一次运行后查看结果、证据和输出数据。" /></section>;
  return <section className="run-result-details">
    <div className="result-head"><h2>运行结果</h2><span className={`pill ${result.status.toLowerCase()}`}>{statusLabel(result.status)}</span></div>
    <MarkdownContent content={result.summary} />
    {result.error && <div className="result-error"><b>未完成原因</b><span>{result.error}</span></div>}
    {result.warnings.length > 0 && <section className="result-warnings" aria-label="需要注意">{result.warnings.map((warning, index) => <p className="warning" key={index}>{warning}</p>)}</section>}
    {result.findings.length > 0 && <section><h3>分析发现</h3>{result.findings.map((finding, index) => <MarkdownContent key={index} content={findingText(finding)} />)}</section>}
    {result.datasets.length > 0 && <section><h3>关联数据</h3><div className="dataset-result-list">{result.datasets.map((datasetId) => {
      const dataset = datasets.find((item) => item.id === datasetId);
      return <div className="dataset-result-item" key={datasetId}>
        {dataset ? <><b><DatasetPreviewButton datasetId={dataset.id} title={dataset.name} className="dataset-preview-name">{dataset.name}</DatasetPreviewButton></b><small>{kindLabel(dataset.kind)}</small><DatasetActions dataset={dataset} selected={selectedDatasetIds.includes(datasetId)} onUse={onUseDataset} /></> : <span>该关联数据当前不可用</span>}
      </div>;
    })}</div></section>}
    {result.artifacts.length > 0 && <section><h3>结果文件</h3><div className="artifact-list">{result.artifacts.map((artifactId) => {
      const artifact = artifacts.find((item) => item.id === artifactId);
      return <a className="artifact-link" href={api.artifactUrl(artifactId)} download target="_blank" rel="noreferrer" key={artifactId}>{artifact?.name ?? "结果文件"} <Icon name="external" size={12} /></a>;
    })}</div></section>}
    {output && <div className="result-map-panel"><div className="panel-head"><h3>关联数据预览</h3></div>
      {previewDatasets.length > 1 && <label className="result-preview-select">选择预览数据
        <select value={output.id} onChange={(event) => setPreviewSelection({ traceId: result.trace_id, datasetId: event.target.value })}>
          {previewDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {kindLabel(dataset.kind)}</option>)}
        </select>
      </label>}
      <MapViewer datasetId={output.id} title={output.name} />
    </div>}
    <details className="technical-details"><summary>技术信息与证据</summary>
      <dl className="result-identifiers"><dt>追踪编号</dt><dd>{result.trace_id}</dd><dt>数据集编号</dt><dd>{result.datasets.join("、") || "无"}</dd><dt>结果文件编号</dt><dd>{result.artifacts.join("、") || "无"}</dd></dl>
      {result.evidence.map((evidence, index) => <pre key={index}>{findingText(evidence)}</pre>)}
      {output && <LineagePanel datasetId={output.id} datasetNames={names} />}
    </details>
  </section>;
}

function Empty({ text }: { text: string }) {
  return <div className="empty">{text}</div>;
}
