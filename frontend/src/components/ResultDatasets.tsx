import { useState } from "react";
import type { Dataset } from "../api";
import { DatasetActions } from "./DatasetActions";
import { MapViewer } from "./MapViewer";

// 仅限制聊天里的默认展开数量，不限制结果数量，也不判断哪份是最终成果。
const INLINE_PREVIEW_COUNT = 2;

export function ResultDatasets({ datasets, runId, selectedIds, onUse }: {
  datasets: Dataset[]; runId?: string; selectedIds: string[]; onUse?: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? datasets : datasets.slice(0, INLINE_PREVIEW_COUNT);
  return <section className="chat-result-datasets" aria-label="回复关联数据">
    <div className="chat-result-heading">关联数据 · {datasets.length} 份</div>
    <div className="chat-preview-grid">{visible.map((dataset) => <div className="chat-preview-item" key={dataset.id}>
      <small>{runId && dataset.created_by_run_id === runId ? "本次生成" : "关联数据"}</small>
      <MapViewer datasetId={dataset.id} title={dataset.name} compact />
      <DatasetActions dataset={dataset} selected={selectedIds.includes(dataset.id)} onUse={onUse} />
    </div>)}</div>
    {datasets.length > INLINE_PREVIEW_COUNT && <button type="button" className="chat-result-link result-expand" aria-expanded={expanded} onClick={() => setExpanded((current) => !current)}>
      {expanded ? "收起更多结果" : `查看其余 ${datasets.length - INLINE_PREVIEW_COUNT} 份数据`}
    </button>}
  </section>;
}
