import { useEffect, useMemo, useState } from "react";
import { api, DatasetPreview } from "../api";

type Position = [number, number];
type Feature = { geometry?: { type?: string; coordinates?: unknown } | null; properties?: Record<string, unknown> | null };

export function MapViewer({ datasetId, title }: { datasetId: string; title?: string }) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    void api.datasetPreview(datasetId).then((value) => { if (active) setPreview(value); }).catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "预览失败"); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [datasetId]);

  if (loading) return <div className="map-viewer map-message">正在加载地图预览…</div>;
  if (error) return <div className="map-viewer map-message map-error">预览失败：{error}</div>;
  if (!preview) return <div className="map-viewer map-message">暂无预览数据。</div>;
  if (preview.kind !== "VECTOR" || !preview.geojson) return <RasterSummary preview={preview} title={title} />;
  return <VectorMap preview={preview} title={title} />;
}

function RasterSummary({ preview, title }: { preview: DatasetPreview; title?: string }) {
  return <div className="map-viewer map-summary"><div className="map-viewer-head"><b>{title ?? "栅格数据概览"}</b><span>只读预览</span></div><div className="map-summary-grid"><span>尺寸<b>{preview.width ?? "-"} × {preview.height ?? "-"}</b></span><span>波段<b>{preview.bands ?? "-"}</b></span><span>分辨率<b>{preview.resolution?.join(" × ") ?? "-"}</b></span><span>坐标系<b>{preview.crs ?? "未提供"}</b></span></div></div>;
}

function VectorMap({ preview, title }: { preview: DatasetPreview; title?: string }) {
  const features = (preview.geojson?.features ?? []) as Feature[];
  const positions = useMemo(() => features.flatMap((feature) => collectPositions(feature.geometry?.coordinates)), [features]);
  if (features.length === 0 || positions.length === 0) return <div className="map-viewer map-message">该数据集没有可显示的要素。</div>;
  const bounds = boundsOf(positions);
  const project = (position: Position): [number, number] => [24 + ((position[0] - bounds.minX) / Math.max(bounds.maxX - bounds.minX, 1e-9)) * 472, 276 - ((position[1] - bounds.minY) / Math.max(bounds.maxY - bounds.minY, 1e-9)) * 252];
  return <div className="map-viewer"><div className="map-viewer-head"><b>{title ?? "地图预览"}</b><span>{preview.truncated ? `仅显示前 ${features.length} 个要素` : `${preview.feature_count ?? features.length} 个要素`}</span></div><svg className="map-canvas" viewBox="0 0 520 300" role="img" aria-label={`${title ?? "数据集"}地图预览`}><rect x="0" y="0" width="520" height="300" rx="9" fill="var(--surface-muted)" />{features.map((feature, index) => <Geometry key={index} geometry={feature.geometry} project={project} />)}</svg>{preview.truncated && <small className="map-hint">数据量较大，地图仅展示有限要素。</small>}</div>;
}

function Geometry({ geometry, project }: { geometry?: Feature["geometry"]; project: (position: Position) => [number, number] }) {
  if (!geometry?.coordinates) return null;
  const paths = geometry.type === "Point" ? [] : collectLines(geometry.coordinates).map((line) => line.map((position) => project(position).join(",")).join(" "));
  if (geometry.type === "Point") {
    const point = collectPositions(geometry.coordinates)[0];
    if (!point) return null;
    const [x, y] = project(point);
    return <circle cx={x} cy={y} r="3.5" fill="var(--accent)" stroke="var(--surface)" strokeWidth="1.5" />;
  }
  return <>{paths.map((points, index) => geometry.type?.includes("Polygon") ? <polygon key={index} points={points} fill="var(--accent-border)" fillOpacity=".35" stroke="var(--accent)" strokeWidth="1.2" /> : <polyline key={index} points={points} fill="none" stroke="var(--accent)" strokeWidth="1.5" />)}</>;
}

function collectPositions(value: unknown): Position[] {
  if (!Array.isArray(value)) return [];
  if (value.length >= 2 && typeof value[0] === "number" && typeof value[1] === "number") return [[value[0], value[1]]];
  return value.flatMap(collectPositions);
}

function collectLines(value: unknown): Position[][] {
  if (!Array.isArray(value)) return [];
  if (value.length >= 2 && Array.isArray(value[0]) && typeof value[0][0] === "number") return [value as Position[]];
  return value.flatMap(collectLines);
}

function boundsOf(positions: Position[]) {
  return positions.reduce((bounds, [x, y]) => ({ minX: Math.min(bounds.minX, x), maxX: Math.max(bounds.maxX, x), minY: Math.min(bounds.minY, y), maxY: Math.max(bounds.maxY, y) }), { minX: positions[0][0], maxX: positions[0][0], minY: positions[0][1], maxY: positions[0][1] });
}
