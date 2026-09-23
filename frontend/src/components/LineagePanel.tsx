import { useEffect, useState } from "react";
import { api, DatasetLineage } from "../api";

export function LineagePanel({ datasetId, datasetNames }: { datasetId: string; datasetNames?: Record<string, string> }) {
  const [items, setItems] = useState<DatasetLineage[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    void api.datasetLineage(datasetId).then((value) => { if (active) setItems(value); }).catch(() => { if (active) setError("来源链路暂时无法加载"); });
    return () => { active = false; };
  }, [datasetId]);
  return <div className="lineage-panel"><h3>数据来源链路</h3>{error ? <p className="muted-text">{error}</p> : items.length === 0 ? <p className="muted-text">这是根数据，暂无上游操作。</p> : items.map((item) => <div className="lineage-card" key={item.id}><div className="lineage-flow"><span>{item.input_dataset_ids.map((id) => datasetNames?.[id] ?? id).join("、") || "无输入"}</span><b>↓ {item.operation}</b><span>{datasetNames?.[item.output_dataset_id] ?? item.output_dataset_id}</span></div><small>{item.run_id ? `运行：${item.run_id}` : "未绑定运行"} · {new Date(item.created_at).toLocaleString("zh-CN")}</small>{item.tool_call_id && <small>工具调用：{item.tool_call_id}</small>}</div>)}</div>;
}
