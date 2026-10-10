import { useState } from "react";
import type { FormEvent } from "react";
import { Dataset } from "../api";
import { formatLabel, kindLabel } from "../labels";
import { DatasetActions } from "./DatasetActions";
import { DatasetPreviewButton } from "./MapViewer";

export function DatasetPanel({ datasets, conversationDatasetIds, selectedDatasetIds, onUse, onRemove, onRegister, busy }: {
  datasets: Dataset[]; conversationDatasetIds: Set<string>; selectedDatasetIds: string[];
  onUse: (id: string) => void; onRemove: (id: string) => void;
  onRegister: (path: string, name: string) => Promise<void>; busy: boolean;
}) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [registering, setRegistering] = useState(false);
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState("");
  const [scope, setScope] = useState<"all" | "conversation">("all");
  const kinds = [...new Set(datasets.map((dataset) => dataset.kind))].sort();
  const visible = datasets.filter((dataset) => dataset.name.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())
    && (!kind || dataset.kind === kind) && (scope === "all" || conversationDatasetIds.has(dataset.id)));
  const conversationCount = datasets.filter((dataset) => conversationDatasetIds.has(dataset.id)).length;
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!path.trim()) return;
    setRegistering(true);
    try { await onRegister(path.trim(), name.trim()); setPath(""); setName(""); }
    catch { /* 父页面显示登记错误。 */ }
    finally { setRegistering(false); }
  };
  return <section className="panel">
    <div className="panel-head"><div><span className="eyebrow">已登记数据源</span><h2>数据集</h2></div><span className="count-badge">{visible.length} / {datasets.length} 个数据集</span></div>
    <form className="dataset-register" onSubmit={(event) => void submit(event)}><input value={path} onChange={(event) => setPath(event.target.value)} placeholder="请输入工作区内的数据文件路径" aria-label="数据文件路径" /><input value={name} onChange={(event) => setName(event.target.value)} placeholder="显示名称（可选）" aria-label="显示名称" /><button className="primary" disabled={busy || registering || !path.trim()}>{registering ? "正在登记…" : "登记数据集"}</button></form>
    <div className="dataset-toolbar">
      <div className="dataset-scope" role="group" aria-label="数据范围"><button type="button" aria-pressed={scope === "all"} onClick={() => setScope("all")}>全部数据 · {datasets.length}</button><button type="button" aria-pressed={scope === "conversation"} onClick={() => setScope("conversation")}>当前会话 · {conversationCount}</button></div>
      <input type="search" placeholder="搜索数据集名称" aria-label="搜索数据集名称" value={query} onChange={(event) => setQuery(event.target.value)} />
      <select aria-label="筛选数据类型" value={kind} onChange={(event) => setKind(event.target.value)}><option value="">全部类型</option>{kinds.map((value) => <option key={value} value={value}>{kindLabel(value)}</option>)}</select>
    </div>
    {visible.length === 0 ? <div className="dataset-filter-empty">{datasets.length ? "没有符合当前筛选条件的数据集。" : "还没有数据集，可上传文件或通过工作区路径登记。"}</div> : <div className="dataset-grid">{visible.map((dataset) => <div className={`dataset-card ${selectedDatasetIds.includes(dataset.id) ? "request-selected" : ""}`} key={dataset.id}>
      <b className="dataset-name"><DatasetPreviewButton datasetId={dataset.id} title={dataset.name} className="dataset-preview-name">{dataset.name}</DatasetPreviewButton></b>
      <div className="dataset-tags"><span>{kindLabel(dataset.kind)}</span><span>{dataset.created_by_run_id ? "生成数据" : "上传／登记"}</span><small>{formatLabel(dataset.format)}{dataset.crs?.authority && ` · ${dataset.crs.authority}`}</small></div>
      <DatasetActions dataset={dataset} selected={selectedDatasetIds.includes(dataset.id)} onUse={onUse} onRemove={onRemove} />
    </div>)}</div>}
  </section>;
}
