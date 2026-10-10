import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, Dataset } from "../api";
import { formatLabel, kindLabel } from "../labels";
import { LineagePanel } from "./LineagePanel";
import { MapViewer } from "./MapViewer";
import "./DatasetActions.css";

export function DatasetActions({ dataset, selected = false, onUse, onRemove }: { dataset: Dataset; selected?: boolean; onUse?: (id: string) => void; onRemove?: (id: string) => void }) {
  const [properties, setProperties] = useState(false);
  return <div className="dataset-card-actions">
    {onUse && <button type="button" className={`small-action ${selected ? "selected-action" : ""}`} disabled={selected && !onRemove} onClick={() => selected && onRemove ? onRemove(dataset.id) : onUse(dataset.id)}>{selected ? onRemove ? "移出本轮" : "已加入下一条" : "用于下一条消息"}</button>}
    <a className="small-action dataset-download" href={api.datasetUrl(dataset.id)} download target="_blank" rel="noreferrer" title={/\.shp$/i.test(dataset.path) ? "下载 SHP 及同名伴随文件（ZIP）" : "下载原始数据文件"}>下载</a>
    <button type="button" className="small-action" onClick={() => setProperties(true)}>属性</button>
    {properties && <DatasetProperties dataset={dataset} onClose={() => setProperties(false)} />}
  </div>;
}

function DatasetProperties({ dataset, onClose }: { dataset: Dataset; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current!;
    const trigger = document.activeElement as HTMLElement | null;
    element.showModal();
    return () => { element.close(); trigger?.focus(); };
  }, []);
  return createPortal(<dialog className="dataset-properties-dialog dataset-modal-card" ref={dialog} aria-label="数据集属性" onCancel={onClose} onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div>
      <div className="dataset-modal-head"><div><span className="eyebrow">数据集属性</span><h2>{dataset.name}</h2></div><button type="button" className="modal-close" onClick={onClose}>关闭属性</button></div>
      <div className="dataset-preview-layout"><div>
        <div className="property-grid">
          <Property label="数据集编号" value={dataset.id} />
          <Property label="数据类型" value={kindLabel(dataset.kind)} />
          <Property label="文件格式" value={formatLabel(dataset.format)} />
          <Property label="坐标系" value={dataset.crs?.authority ?? "未提供"} />
          <Property label="坐标系名称" value={dataset.crs?.name ?? "未提供"} />
          <Property label="要素数量" value={dataset.schema?.feature_count != null ? String(dataset.schema.feature_count) : "不适用"} />
          <Property label="几何类型" value={dataset.schema?.geometry_type ?? "未提供"} />
          <Property label="栅格尺寸" value={dataset.schema?.width != null ? `${dataset.schema.width} × ${dataset.schema.height ?? "?"}，${dataset.schema.bands ?? "?"} 个波段` : "不适用"} />
          <Property label="空间范围" value={dataset.extent ? `${dataset.extent.min_x}, ${dataset.extent.min_y} 至 ${dataset.extent.max_x}, ${dataset.extent.max_y}` : "未提供"} />
          <Property label="文件路径" value={dataset.path} />
          <Property label="创建时间" value={dataset.created_at ?? "未提供"} />
          <Property label="创建运行" value={dataset.created_by_run_id ?? "手动登记或上传"} />
        </div>
        <h3>字段</h3><pre>{dataset.schema?.fields && Object.keys(dataset.schema.fields).length ? JSON.stringify(dataset.schema.fields, null, 2) : "未提供字段信息"}</pre>
        <details className="technical-details"><summary>附加信息</summary><pre>{JSON.stringify(dataset.metadata ?? {}, null, 2)}</pre></details>
      </div><div><MapViewer datasetId={dataset.id} title={dataset.name} /><details className="technical-details"><summary>数据来源与处理参数</summary><LineagePanel datasetId={dataset.id} datasetNames={{ [dataset.id]: dataset.name }} /></details></div></div>
    </div>
  </dialog>, document.body);
}

function Property({ label, value }: { label: string; value: string }) {
  return <div className="property-item"><span>{label}</span><b>{value}</b></div>;
}
