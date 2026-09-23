import { ApprovalRequest } from "../api";

const RISK_LABELS: Record<string, string> = { READ: "读取", WRITE: "写入", DESTRUCTIVE: "高风险修改", EXTERNAL: "外部访问" };

export function ApprovalCard({ approval, busy, onApprove, onDeny }: { approval: ApprovalRequest; busy: boolean; onApprove: (approval: ApprovalRequest) => Promise<void>; onDeny: (approval: ApprovalRequest) => Promise<void> }) {
  const pending = approval.status === "PENDING";
  return <article className={`approval-card ${pending ? "pending" : "closed"}`}><div className="approval-card-head"><div><span className="eyebrow">需要你的确认</span><h3>{approval.tool_name}</h3></div><span className={`approval-risk risk-${approval.risk_level.toLowerCase()}`}>{RISK_LABELS[approval.risk_level] ?? approval.risk_level}</span></div><p>{approval.reason || "该操作需要你确认后才能继续。"}</p><div className="approval-preview"><span>安全参数摘要</span><code>{JSON.stringify(approval.argument_preview, null, 2)}</code></div>{pending ? <div className="approval-actions"><button type="button" className="primary" disabled={busy} onClick={() => void onApprove(approval)}>{busy ? "正在继续…" : "批准并继续"}</button><button type="button" className="small-action danger" disabled={busy} onClick={() => void onDeny(approval)}>拒绝</button></div> : <div className="approval-closed">{approval.status === "APPROVED" ? "已批准，正在继续" : approval.status === "DENIED" ? "已拒绝该操作" : `状态：${approval.status}`}</div>}</article>;
}
