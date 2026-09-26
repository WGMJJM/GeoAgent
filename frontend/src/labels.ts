export const RUN_STATUS_LABELS: Record<string, string> = {
  CREATED: "已创建",
  RUNNING: "运行中",
  WAITING_TOOL: "等待工具",
  WAITING_SUBAGENT: "等待子智能体",
  WAITING_USER: "等待补充信息",
  WAITING_APPROVAL: "等待确认",
  RETRYING: "重试中",
  COMPLETED: "已完成",
  PARTIAL_COMPLETED: "部分完成",
  CANCELLED: "已取消",
  INTERRUPTED: "已中断",
  BUDGET_EXCEEDED: "超出预算",
};

export const RESULT_STATUS_LABELS: Record<string, string> = {
  SUCCESS: "成功",
  PARTIAL: "部分完成",
  FAILED: "失败",
  BLOCKED: "已阻塞",
  CANCELLED: "已取消",
};

export const KIND_LABELS: Record<string, string> = {
  VECTOR: "矢量",
  RASTER: "栅格",
  TABLE: "表格",
  POINT_CLOUD: "点云",
  TRAJECTORY: "轨迹",
  NETWORK: "网络",
  SERVICE: "服务",
};

export const FORMAT_LABELS: Record<string, string> = {
  geojson: "矢量文件",
  gpkg: "空间数据库",
  shp: "矢量文件",
  tif: "栅格文件",
  tiff: "栅格文件",
  csv: "表格文件",
  tsv: "表格文件",
  parquet: "表格文件",
};

export const EVENT_LABELS: Record<string, string> = {
  RunCreated: "开始处理请求",
  IntentResolved: "请求理解完成",
  DecisionMade: "已确定下一步动作",
  TokenUsageUpdated: "Token 用量更新",
  SubTaskCreated: "已创建子任务",
  SubAgentSpawned: "已启动子智能体",
  ToolStarted: "工具开始执行",
  ToolCompleted: "工具执行完成",
  ToolFailed: "工具执行失败",
  RetryStarted: "开始重试",
  RepairSelected: "已选择修复方案",
  ReplanStarted: "正在重新评估执行策略",
  DatasetCreated: "已创建数据集",
  ArtifactCreated: "已生成结果文件",
  VerificationStarted: "开始验证结果",
  VerificationFailed: "结果验证失败",
  SubAgentCompleted: "子智能体已完成",
  DelegationCompleted: "子任务委派完成",
  CheckpointSaved: "已保存检查点",
  ResumeStarted: "开始恢复运行",
  RunCompleted: "运行完成",
  RunFailed: "运行失败",
  RunCancelled: "运行已取消",
  RunWaitingUser: "等待补充信息",
  RunWaitingApproval: "等待用户确认",
  ApprovalRequested: "已请求操作确认",
  ApprovalGranted: "已批准操作",
  ApprovalDenied: "已拒绝操作",
  ApprovalConsumed: "已消费操作确认",
};

export const TOOL_LABELS: Record<string, string> = {
  "dataset.list": "列出数据集",
  "dataset.inspect": "检查数据集",
  "dataset.register": "登记数据集",
  "crs.inspect": "检查坐标系",
  "crs.reproject": "重投影",
  "vector.validate": "验证矢量数据",
  "vector.repair": "修复矢量几何",
  "vector.buffer": "生成矢量缓冲区",
  "vector.clip": "裁剪矢量数据",
  "vector.intersection": "计算矢量相交",
  "vector.dissolve": "融合矢量数据",
  "vector.spatial_join": "执行空间连接",
  "raster.inspect": "检查栅格数据",
  "raster.clip": "裁剪栅格数据",
  "raster.reproject": "重投影栅格",
  "raster.slope": "计算坡度",
  "analysis.distance": "距离分析",
  "analysis.zonal_statistics": "分区统计",
  "map.render": "生成地图",
  "python.execute": "执行分析代码",
  "shell.execute": "执行空间命令",
};

export const FIELD_LABELS: Record<string, string> = {
  model: "模型",
  content: "内容",
  tool: "工具",
  status: "状态",
  output: "输出",
  error: "错误",
  source: "来源",
  dataset: "数据集",
  datasets: "数据集",
  distance_m: "距离（米）",
  distance: "距离",
  threshold: "距离阈值",
  within_threshold: "阈值内数量",
  min_distance: "最小距离",
  max_distance: "最大距离",
  mean_distance: "平均距离",
  repair_applied: "是否已修复坐标系",
  verification: "验证结果",
  inspection: "检查结果",
  road_quality: "道路质量",
  terrain: "地形分析",
  population_fields: "人口字段",
  feature_count: "要素数量",
  agent_id: "智能体",
  summary: "摘要",
  findings: "分析发现",
  scope: "范围",
  goal: "目标",
  context_dataset_count: "上下文数据集数量",
  run_id: "运行编号",
  result: "结果",
  path: "路径",
  name: "名称",
  kind: "类型",
  format: "格式",
  crs: "坐标系",
  extent: "范围",
  schema: "结构",
  metadata: "元数据",
};

export function statusLabel(value: string): string {
  return RUN_STATUS_LABELS[value] ?? RESULT_STATUS_LABELS[value] ?? "未知状态";
}

export function kindLabel(value: string): string {
  return KIND_LABELS[value] ?? "其他数据";
}

export function formatLabel(value: string): string {
  return FORMAT_LABELS[value.toLowerCase()] ?? "空间数据文件";
}

export function eventLabel(value: string): string {
  return EVENT_LABELS[value] ?? value;
}

export function agentLabel(value: string): string {
  if (value === "agent-loop") return "GeoAgent";
  return value === "main" ? "主智能体" : "子智能体";
}

function replaceText(value: string, search: string, replacement: string): string {
  return value.split(search).join(replacement);
}

export function displayEventMessage(message: string): string {
  let displayed = message;
  for (const [name, label] of Object.entries(TOOL_LABELS)) displayed = replaceText(displayed, name, label);
  for (const [name, label] of [["Main Agent", "主智能体"], ["SubAgent", "子智能体"], ["Dataset", "数据集"], ["Artifact", "结果文件"], ["Checkpoint", "检查点"], ["Tool", "工具"], ["CRS", "坐标系"], ["road", "道路"], ["population", "人口"], ["terrain", "地形"], ["SUCCESS", "成功"], ["PARTIAL_SUCCESS", "部分完成"], ["FAILED", "失败"], ["CANCELLED", "已取消"]]) {
    displayed = replaceText(displayed, name, label);
  }
  return displayed;
}

export function findingText(value: unknown): string {
  const translated = translateFinding(value);
  return typeof translated === "string" ? translated : JSON.stringify(translated, null, 2) ?? String(translated);
}

function translateFinding(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(translateFinding);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([key, item]) => [FIELD_LABELS[key] ?? key, translateFinding(item)]));
  }
  if (typeof value === "string") return RUN_STATUS_LABELS[value] ?? RESULT_STATUS_LABELS[value] ?? TOOL_LABELS[value] ?? value;
  return value;
}
