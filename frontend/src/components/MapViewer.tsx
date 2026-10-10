import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { api, DatasetPreview } from "../api";
import { Icon } from "./Icon";
import "./MapViewer.css";

type Position = [number, number];
type GeoShape = { type?: string; coordinates?: unknown; geometries?: GeoShape[] };
type Feature = { geometry?: GeoShape | null; properties?: Record<string, unknown> | null };
type ViewerProps = { datasetId: string; title?: string };
const COLLAPSED_VALUE_LENGTH = 120;

function usePreview(datasetId: string, enabled = true, initial?: DatasetPreview, encoding = "") {
  const [preview, setPreview] = useState<DatasetPreview | undefined>(initial);
  const [error, setError] = useState("");
  const [supportsEncoding, setSupportsEncoding] = useState(initial?.supports_encoding ?? false);
  useEffect(() => {
    if (!enabled) return;
    setError("");
    if (initial && !encoding) { setPreview(initial); return; }
    setPreview(undefined);
    let active = true;
    void api.datasetPreview(datasetId, encoding || undefined).then((value) => {
      if (active) { setPreview(value); setSupportsEncoding(value.supports_encoding ?? false); }
    })
      .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "预览失败"); });
    return () => { active = false; };
  }, [datasetId, enabled, initial, encoding]);
  return { preview, error, supportsEncoding };
}

// 聊天缩略卡片进入视口才读取；放大复用已有数据，不生成磁盘缓存。
export function MapViewer(props: ViewerProps & { compact?: boolean }) {
  return <PreviewCard key={props.datasetId} {...props} />;
}

function PreviewCard({ datasetId, title = "数据集", compact = false }: ViewerProps & { compact?: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(!compact);
  const [open, setOpen] = useState(false);
  const { preview, error } = usePreview(datasetId, visible);
  useEffect(() => {
    if (visible || !host.current) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setVisible(true); observer.disconnect(); }
    });
    observer.observe(host.current);
    return () => observer.disconnect();
  }, [visible]);
  return <div ref={host} className={`dataset-viewer ${compact ? "preview-card" : "preview-inline"}`}>
    <button className="preview-open" type="button" aria-label={`放大查看 ${title}`} onClick={() => setOpen(true)}>
      <div className="preview-card-head"><b>{title}</b><Icon name="expand" size={14} /></div>
      {preview ? <PreviewContent preview={preview} title={title} compact={compact} /> : <p className="preview-message">{error ? `预览失败：${error}` : "正在加载预览…"}</p>}
    </button>
    {open && <PreviewDialog datasetId={datasetId} title={title} preview={preview} onClose={() => setOpen(false)} />}
  </div>;
}

export function DatasetPreviewButton({ datasetId, title, children, className }: ViewerProps & { children?: ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  return <><button type="button" className={className ?? "small-action"} onClick={() => setOpen(true)} aria-label={`查看 ${title ?? "数据集"}`}>
    {children ?? "查看"}
  </button>{open && <PreviewDialog key={datasetId} datasetId={datasetId} title={title} onClose={() => setOpen(false)} />}</>;
}

function PreviewDialog({ datasetId, title = "数据集", preview: initial, onClose }: ViewerProps & { preview?: DatasetPreview; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [encoding, setEncoding] = useState("");
  const { preview, error, supportsEncoding } = usePreview(datasetId, true, initial, encoding);
  useEffect(() => {
    const element = dialog.current!;
    const trigger = document.activeElement as HTMLElement | null;
    element.showModal();
    return () => { element.close(); trigger?.focus(); };
  }, []);
  return createPortal(<dialog className={`preview-dialog ${preview?.kind === "VECTOR" || supportsEncoding ? "preview-dialog-vector" : ""}`} ref={dialog} aria-label={`${title}预览`} onCancel={onClose} onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="preview-dialog-content">
      <header><div><b>{title}</b><small>只读数据预览</small></div><button type="button" className="preview-control" aria-label="关闭预览" onClick={onClose}><Icon name="close" /></button></header>
      {supportsEncoding && <div className="preview-encoding">
        <label>文字编码<select value={encoding} onChange={(event) => setEncoding(event.currentTarget.value)}>
          <option value="">自动（按文件声明）</option><option value="utf-8">UTF-8</option><option value="gb18030">GB18030（兼容 GBK）</option>
        </select></label>
        <small>中文乱码时可切换，仅影响本次预览。</small>
      </div>}
      {preview ? <PreviewContent preview={preview} title={title} interactive /> : <p className="preview-message" role={error ? "alert" : "status"}>{error ? `预览失败：${error}${supportsEncoding ? "。请切换文字编码重试。" : ""}` : "正在加载预览…"}</p>}
    </div>
  </dialog>, document.body);
}

function PreviewContent({ preview, title, interactive = false, compact = false }: { preview: DatasetPreview; title: string; interactive?: boolean; compact?: boolean }) {
  const [selected, setSelected] = useState<number | null>(null);
  const features = (preview.geojson?.features ?? []) as Feature[];
  const visual = preview.kind === "VECTOR" || Boolean(preview.image_data_url);
  return <div className={`preview-content ${interactive && preview.kind === "VECTOR" ? "preview-vector-layout" : ""}`}>
    <div className="preview-main">
    {visual ? <PreviewCanvas preview={preview} title={title} interactive={interactive} selected={selected} onSelect={setSelected} /> : preview.kind === "DOCUMENT" ?
      <pre className="document-preview">{preview.text || "该文件暂时无法提取正文。"}</pre> : preview.kind === "TABLE" ?
        <PreviewTable rows={preview.rows} columns={preview.columns} interactive={interactive} /> : <p className="preview-message">该数据暂不支持图像预览。</p>}
    <div className="preview-caption">
      {preview.kind === "VECTOR" ? <span>{preview.truncated ? `仅显示前 ${features.length} 个要素` : `${preview.feature_count ?? features.length} 个要素`}{interactive ? " · 点击要素查看属性" : ""}</span> : preview.width && <span>{preview.width} × {preview.height} · {preview.bands} 波段／通道</span>}
      {preview.source_crs && <span>源坐标系 {preview.source_crs}</span>}
      {!compact && preview.render_note && <span>{preview.render_note}</span>}
      {preview.truncated && preview.kind !== "VECTOR" && <span>仅显示部分内容</span>}
    </div>
    </div>
    {interactive && preview.kind === "VECTOR" && <FeatureAttributes key={selected} feature={selected === null ? undefined : features[selected]} index={selected} columns={preview.columns} />}
  </div>;
}

function FeatureAttributes({ feature, index, columns }: { feature?: Feature; index: number | null; columns: string[] }) {
  const [query, setQuery] = useState("");
  const properties = feature?.properties ?? {};
  const fields = columns.length ? columns : Object.keys(properties);
  const search = query.trim().toLocaleLowerCase();
  const matching = fields.filter((field) => `${field}\n${previewValue(properties[field])}`.toLocaleLowerCase().includes(search));
  return <section className="preview-attributes" aria-label="要素属性">
    <div className="preview-attributes-head"><h3>要素属性</h3><span>{index === null ? "尚未选择" : `要素 ${index + 1} · ${fields.length} 个预览字段`}</span></div>
    {feature ? <>
      <div className="preview-field-search"><input type="search" aria-label="搜索字段或值" placeholder="搜索字段或值" value={query} onChange={(event) => setQuery(event.currentTarget.value)} />
        {query && <button type="button" className="preview-control" aria-label="清除字段搜索" onClick={() => setQuery("")}><Icon name="close" size={14} /></button>}
      </div>
      <div className="preview-attribute-scroll" role="region" aria-label="要素字段和值，可滚动" tabIndex={0}>
        {matching.length ? <table className="preview-attribute-table"><thead><tr><th scope="col">字段名</th><th scope="col">值</th></tr></thead><tbody>
          {matching.map((field) => <tr key={field}><th scope="row">{field}</th><td><PreviewValue key={field} value={properties[field]} expandable /></td></tr>)}
        </tbody></table> : <p className="preview-attribute-empty">{fields.length ? "没有匹配的字段或值" : "该要素没有可显示的属性"}</p>}
      </div>
      <small className="preview-attribute-note">{search ? `匹配 ${matching.length} / ${fields.length} 个字段 · ` : ""}保持原始字段名和数值，不修改数据。</small>
    </> : <div className="preview-attribute-empty"><Icon name="layers" size={24} /><p>点击地图中的要素</p><small>在这里查看对应的字段和值。<br />几何预览无底图，范围以实际数据为准。</small></div>}
  </section>;
}

function previewValue(value: unknown): string {
  return value == null ? "—" : typeof value === "object" ? JSON.stringify(value) : String(value);
}

function PreviewValue({ value, expandable = false }: { value: unknown; expandable?: boolean }) {
  const text = previewValue(value);
  if (expandable && text.length > COLLAPSED_VALUE_LENGTH) return <details className="preview-long-value"><summary>展开预览内容 · {text.length} 字符</summary><div>{text}</div></details>;
  return <span className={`preview-value ${typeof value === "number" ? "preview-number" : ""} ${value == null || value === "" ? "preview-empty-value" : ""}`} title={text}>{value === "" ? "空文本" : text}</span>;
}

function PreviewTable({ rows, columns, interactive = false }: { rows: Record<string, unknown>[]; columns: string[]; interactive?: boolean }) {
  return <div className="preview-table" role={interactive ? "region" : undefined} aria-label={interactive ? "数据表，可横向滚动" : undefined} tabIndex={interactive ? 0 : undefined}><table><thead><tr>{columns.map((key) => <th key={key} scope="col">{key}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{columns.map((key) => <td key={key}><PreviewValue value={row[key]} expandable={interactive} /></td>)}</tr>)}</tbody></table></div>;
}

function PreviewCanvas({ preview, title, interactive, selected, onSelect }: { preview: DatasetPreview; title: string; interactive: boolean; selected: number | null; onSelect: (index: number) => void }) {
  const canvas = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const features = useMemo(() => (preview.geojson?.features ?? []) as Feature[], [preview.geojson]);
  const positions = useMemo(() => features.flatMap((feature) => geometryPositions(feature.geometry)), [features]);
  const zoom = (factor: number) => setView((old) => ({ ...old, zoom: Math.max(1, Math.min(16, old.zoom * factor)) }));
  useEffect(() => {
    if (!interactive) return;
    const element = canvas.current!;
    const wheel = (event: WheelEvent) => { event.preventDefault(); zoom(event.deltaY < 0 ? 1.2 : 1 / 1.2); };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, [interactive]);
  const bounds = positions.reduce((b, [x, y]) => [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)], [Infinity, Infinity, -Infinity, -Infinity]);
  const scale = Math.min(472 / (bounds[2] - bounds[0] || 1), 252 / (bounds[3] - bounds[1] || 1));
  const project = ([x, y]: Position): Position => [260 + (x - (bounds[0] + bounds[2]) / 2) * scale, 150 - (y - (bounds[1] + bounds[3]) / 2) * scale];
  return <div className={`preview-canvas-wrap ${interactive ? "interactive" : ""}`}>
    {interactive && <div className="preview-toolbar"><button className="preview-control" type="button" aria-label="放大" onClick={() => zoom(1.25)}><Icon name="plus" /></button><button className="preview-control" type="button" aria-label="缩小" onClick={() => zoom(0.8)}><Icon name="minus" /></button><button className="preview-control" type="button" onClick={() => setView({ zoom: 1, x: 0, y: 0 })}>复位</button><span>{Math.round(view.zoom * 100)}%</span></div>}
    <svg ref={canvas} className="preview-canvas" viewBox={`${view.x + 260 - 260 / view.zoom} ${view.y + 150 - 150 / view.zoom} ${520 / view.zoom} ${300 / view.zoom}`} role="img" aria-label={`${title}地图预览`}
      onPointerDown={interactive ? (event) => { if (event.button === 0) drag.current = { x: event.clientX, y: event.clientY, moved: false }; } : undefined}
      onPointerMove={interactive ? (event) => {
        if (!drag.current || event.buttons !== 1) return;
        const dx = event.clientX - drag.current.x, dy = event.clientY - drag.current.y;
        if (Math.abs(dx) + Math.abs(dy) < 2) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        const rect = event.currentTarget.getBoundingClientRect();
        const ratio = Math.max(520 / rect.width, 300 / rect.height) / view.zoom;
        setView((old) => ({ ...old, x: old.x - dx * ratio, y: old.y - dy * ratio }));
        drag.current = { x: event.clientX, y: event.clientY, moved: true };
      } : undefined}
      onPointerUp={interactive ? (event) => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); } : undefined}
      onPointerCancel={() => { drag.current = null; }}
      onClickCapture={(event) => { if (drag.current?.moved) event.stopPropagation(); drag.current = null; }}>
      {preview.image_data_url ? <image href={preview.image_data_url} x="0" y="0" width="520" height="300" preserveAspectRatio="xMidYMid meet" /> : positions.length ? features.map((feature, index) => <g key={index} className={`preview-feature ${selected === index ? "selected" : ""}`} role={interactive ? "button" : undefined} tabIndex={interactive ? 0 : undefined} aria-label={interactive ? `要素 ${index + 1}` : undefined} onClick={interactive ? () => onSelect(index) : undefined} onKeyDown={interactive ? (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(index); } } : undefined}><Geometry geometry={feature.geometry} project={project} /></g>) : <text x="260" y="150" textAnchor="middle" fill="currentColor">没有可显示的几何要素</text>}
    </svg>
  </div>;
}

function Geometry({ geometry, project }: { geometry?: GeoShape | null; project: (p: Position) => Position }) {
  if (!geometry) return null;
  if (geometry.type === "GeometryCollection") return <>{geometry.geometries?.map((item, index) => <Geometry key={index} geometry={item} project={project} />)}</>;
  if (geometry.type === "Point" || geometry.type === "MultiPoint") return <>{collectPositions(geometry.coordinates).map((point, index) => { const [x, y] = project(point); return <circle key={index} cx={x} cy={y} r="3.5" />; })}</>;
  if (geometry.type === "MultiPolygon") return <>{(geometry.coordinates as unknown[]).map((coordinates, index) => <Geometry key={index} geometry={{ type: "Polygon", coordinates }} project={project} />)}</>;
  const lines = collectLines(geometry.coordinates);
  if (geometry.type === "Polygon") return <path d={lines.map((line) => `M${line.map((p) => project(p).join(",")).join("L")}Z`).join(" ")} fillRule="evenodd" />;
  return <>{lines.map((line, index) => <polyline key={index} points={line.map((p) => project(p).join(",")).join(" ")} fill="none" />)}</>;
}

function geometryPositions(geometry?: GeoShape | null): Position[] {
  return geometry?.type === "GeometryCollection" ? (geometry.geometries ?? []).flatMap(geometryPositions) : collectPositions(geometry?.coordinates);
}

function collectPositions(value: unknown): Position[] {
  if (!Array.isArray(value)) return [];
  if (value.length >= 2 && typeof value[0] === "number" && typeof value[1] === "number") return Number.isFinite(value[0]) && Number.isFinite(value[1]) ? [[value[0], value[1]]] : [];
  return value.flatMap(collectPositions);
}

function collectLines(value: unknown): Position[][] {
  if (!Array.isArray(value)) return [];
  if (value.length && Array.isArray(value[0]) && typeof value[0][0] === "number") return [collectPositions(value)];
  return value.flatMap(collectLines);
}
