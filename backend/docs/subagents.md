# 动态 SubAgent 执行与恢复

本次沿用现有 Registry、Catalog、Executor、AgentLoop、RunManager、Checkpoint 和 SQLite。普通聊天仍只创建 Run；第一次真实委派时才创建 Task。未增加第二套 Agent Loop，也未调整界面架构。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `app/core/models.py` | `DelegationPlan`、`DelegationSubtask`、`UpstreamDatasetBinding`、`DatasetOutputRef`、`SubAgentResult`、`DelegationResult` |
| `app/config.py` | `max_subagents=5`、`max_parallel_agents=3`，支持原有环境变量前缀 |
| `app/agent/delegation.py`（新增） | 校验计划、准备身份、DAG 调度、解析上游输入、验证结果、统一合并 WorkingMemory |
| `app/agent/loop.py` | 主 Agent 常驻 `agent.delegate`，复用子任务循环，保存待执行批次与稳定调用 ID |
| `app/agent/context.py` | 子任务最小上下文；委派观察保留完整 JSON 和关键引用 |
| `app/auth/policy.py` | 用户权限、环境、父 Run 和子任务工具名单的交集 |
| `app/execution/tools/executor.py` | 再次检查身份/权限，稳定调用幂等，取消清理与副作用不确定性处理 |
| `app/execution/tools/model.py`、`app/tools/gis/common.py` | 将真实 ToolCall ID 写入数据 lineage，校验子任务输出路径 |
| `app/execution/sandbox/manager.py`、`app/gis/dataset/registry.py` | 每个子 Run 的输出目录、输入数据和自产数据可见性隔离 |
| `app/state/store.py` | 最小委派表、唯一调用键、事务提交、真实 ToolResult 查询；Task/Run 改用 UPSERT，避免 REPLACE 触发级联删除 |
| `app/run/manager.py` | 执行已准备的子 Run，父子取消/截止时间协调，显式技术恢复 |
| `app/application.py` | 装配 Coordinator，关闭服务时先清理运行 |
| `app/entry/conversation_service.py`、`app/api/app.py` | 用户补充和审批恢复父委派及原子子任务，不把子回复写入主会话 |
| `evaluation/cases.py`、`evaluation/runner.py` | 选择性启用两个已验证案例；Runner 支持显式传入认证用户 |
| `tests/unit/test_delegation.py`（新增） | 非法计划、权限、路径、观察协议和同步超时边界 |
| `tests/integration/test_subagent_runtime.py`（新增） | 真实 GIS 闭环、并发/依赖、等待恢复、失败、取消、提交窗口与重启幂等 |

## 实际调用链

```text
AgentLoop.run（主 Run，处理 agent.delegate）
  -> DelegationCoordinator.execute
     -> validate -> _prepare -> StateStore.create_delegation（事务）
     -> _schedule（DAG ready 集合 + 有界并发）
        -> _run_subtask（绑定经验证上游 Dataset）
           -> _execute_child
              -> RunManager.submit_child / continue_run
                 -> AgentLoop.run（原有循环，独立子上下文）
                    -> tool.search -> ToolExecutor.execute -> 实际 GIS handler
        -> _collect_result -> _persist_result（SubAgentResult）
     -> _aggregate
     -> WorkingMemoryUpdater.merge_deltas
     -> StateStore.save_delegation（结果与记忆同一事务）
     -> _observation（精简 DelegationResult）
  -> AgentLoop 将工具观察交回主模型，继续决策/回复
```

并发测试以 barrier 证明 A/B 同时进入真实缓冲工具，不通过耗时猜测并发。C 等待 A 成功，通过 `upstream_dataset_bindings` 获得 A 的真实输出 ID，并用 `dataset.inspect` 读取该数据。

## 请求与结果协议

`DelegationSubtask` 有目标、输入 Dataset ID、依赖、允许的工具、输出角色、是否可并行和是否必需。模型不能指定服务端身份。拒绝空目标、重复 ID/依赖、自依赖、未知依赖、有向环、越权 Dataset、未知工具和不可用环境；拒绝委派与其他工具混合批次，禁止子任务递归委派。

`output_roles` 是“角色 -> 真实 Dataset 产出工具名”，例如 `{"buffer": "vector.buffer"}`。下游绑定必须引用已声明的依赖及其角色；服务端只接受恰好一个经验证输出，不从摘要猜测、不默认选第一个。上游部分成功不满足依赖。

工具名单只声明可用能力，不表示必须生成数据。允许使用缓冲工具的元数据检查任务可仅执行检查并成功；明确声明的输出角色缺失则不能成功。第一版成功子任务仍要求至少一项真实成功业务工具证据，不支持仅凭模型文字完成的纯推理子任务。

完整 `SubAgentResult` 保存：子任务/Run ID、结果状态、执行状态、Dataset 引用、Artifact ID、小体积指标、指标来源 ToolCall ID、警告、结构化错误、待输入信息、WorkingMemoryDelta、可选摘要。资源来自持久化 ToolResult，并重新检查数据库、归属和创建 Run；模型文字不是执行证据。

模型态 `DelegationResult` 保留所有子任务身份、状态、关键资源引用和错误码；去除局部记忆与指标来源，限量筛选指标/警告/摘要。大结果优先删除可选字段，不能从字符串中间切断 JSON；极端情况下关键引用可超过软字符预算。完整结果保存在委派表以及子 Run metadata 的 `subagent_result` 中。

聚合规则：

- 全部成功：`SUCCESS`。
- 必需任务失败或被依赖阻断：`FAILED`；独立任务继续，已验证输出保留。
- 仅可选任务失败或发生部分成功：`PARTIAL`。
- 有子任务等待用户/审批：`BLOCKED`，父 Run 实际进入 `WAITING_USER` / `WAITING_APPROVAL`。
- 未启动后继以独立 `execution_status=BLOCKED/PENDING` 表达，不修改原有枚举。

## 权限、上下文与写入隔离

最终权限 = 当前认证用户/真实环境权限 ∩ 持久化父 Run 限制 ∩ 子任务工具名单。发现、Schema 注入、Checkpoint 恢复及执行前都重新取交集。`tool.search` 每路查询最多返回 2 项，同一批次不能提前使用刚搜索激活的工具。

中英文发现改为一次模型工具调用：`tool.search({"query":"坡度计算","english_query":"raster slope","limit":2})`。服务端内部复用原检索器，各路独立排名、各取最多 2 项，再按工具名称去重取并集，一次返回最多 4 项；`limit` 是每路上限，不是并集上限。不在不同语言之间比较分数，不自动翻译或补充英文查询。模型调用计数增加 1，内部两路检索不另计模型工具调用；下一轮按预算统一提供 Schema，不自动执行候选。任何一路空结果都不覆盖另一路有效结果。保留原 `query` 单查询协议，已有 Checkpoint 中的独立搜索调用仍按原稳定 ID 恢复；组合查询中断后可重做无副作用的内部检索，但同一模型调用不重复计数。已发现的工具名称跨批次保留在本 Run 的 Checkpoint 中，不再由新搜索整批替换；没有跨 Run 或跨用户共享发现缓存。

`GEOAGENT_TOOL_CONTEXT_TOKENS=3200` 和 `GEOAGENT_TOOL_CONTEXT_MAX_CARDS=8` 集中配置本轮工具上下文预算。最多展示 8 个已发现延迟工具条目（完整 Schema 或卡片，同名不重复），按最近发现/使用顺序优先；常驻工具定义也计入 Schema token 预算，但不属于 8 个延迟工具条目。完整 Schema 与本轮工具状态说明（含卡片）合计计入约 3200 tokens，Schema 放不下时提供精简卡片，卡片也放不下则本轮不展示。预算使用当前模型适配器配置的本地 tokenizer，不计用户消息、记忆或业务工具输出，也不截断单个参数 Schema。常驻定义与状态说明本身超过配置预算时明确报错，不自动扩大预算。

候选 Schema 的生命周期按真实模型选择收口：首次检索或精确名称缓存恢复后，下一轮按现有预算提供候选完整 Schema，模型可以直接填参数调用，不增加一次独立的选择加载交互。该轮完整工具批次结束后，未调用候选转为卡片，调用过的工具继续按预算提供 Schema。同一批次可保留多个实际调用工具；尚未完成的批次不提前降级，恢复继续沿用原批次调用 ID 和激活快照。常驻工具不参与降级。

Checkpoint 增加 `used_tool_names`，仅记录本 Run 中模型对已激活、可用延迟工具提出的调用尝试；参数需修正、执行失败或等待审批仍保留其 Schema 资格，不把这些情况当作未选择。未知、未激活或不可用的伪造调用不进入该集合。这不是任务成功记录，更不是授权；恢复、注入和执行继续按当前权限与环境过滤。未调用卡片后续需要时，按精确名称恢复 Schema，不重新语义检索，不自动执行。同名不可用时的近似候选匹配规则保持不变。

这项调整不保证所有任务减少模型交互：首次发现没有新增加载步骤，但先搜索后一直未调用的工具会降为卡片，后来需要它时仍须一次精确名称恢复。已经调用过的工具可在预算允许时直接复用；超过 token 或 8 项展示上限仍可能退出本轮视图，发现记录保留。本次未增加新常驻工具、工具加载框架、业务关键词规则或每个 GIS 工具的前置条件。

每次模型请求只提供一份实时工具状态：`{"callable":["工具名称"],"cached":[{"name":"工具名称","description":"能力描述","parameter_names":["参数名称"]}]}`。`callable` 直接来自本轮实际发送的完整 Schema（包含常驻工具），模型直接填参数调用；`cached` 只含本轮展示的卡片，需要时按精确名称恢复 Schema。二者不重复，不维护额外工具白名单或数据库字段。工具调用次数耗尽时，实际 Schema 与状态的两类列表均为空，仍可根据既有观察回答。

状态只加入本轮模型视图，不累积写入历史或 Checkpoint；恢复后按最新 Registry、权限、环境和预算重新构建。历史命中仅表示曾经发现，不能替代当前状态。对当前已提供 Schema 的精确名称缓存查询返回 `source=run_cache`、`already_callable=true` 和无需再次搜索的说明，不重新排名，也不自动执行；查询仍如实计入工具调用。现有 `DecisionMade` 事件额外保存本批次工具状态与检索/缓存命中信息，未增加新的事件系统。权限、审批、检索评分、中英文并集规则和 Run 缓存范围保持不变。

只见卡片或历史名称、未见完整 Schema 时，可以只提供 `query` 精确名称请求复用本 Run 发现记录，不携带 `english_query`，经过当前权限/环境校验后优先注入下一轮；这个激活请求仍计入工具调用，不需要重新做双语能力发现。携带 `english_query` 时必须执行两路发现，不因为主查询命中缓存而跳过英文查询。同批次不能抢先执行。原始搜索结果保存于 Checkpoint，模型视图只保留历史命中名称，避免旧卡片描述和本轮 Schema 重复；新 Run 从空发现记录开始。尚未调整原关键词匹配规则：目标不可用时普通检索仍可能返回近似候选，不自动执行替代工具。

主/子 Agent 共用工具选择提示：先对照用户目标、当前工具描述/Schema 和已有观察；已有工具够用就直接调用，结果足够就回答，不能为补充背景扩大分析或列出整个工作区。必要参数缺失时询问用户，权限/参数/临时执行错误不等于缺少能力。确需发现额外能力时先简要说明目标相关的缺口，中英文同一能力检索只发起一次组合调用，不自动执行全部候选。未搜索到不代表不存在。这些是模型决策约束，不新增语义 Gate、匹配分数阈值或服务端硬性禁止搜索；真实模型仍可能偏离提示。

`raster.inspect` 的工具描述补全首波段统计能力及边界：最多 512×512 采样的最小值、最大值、均值和有效像元数，无直方图。没有改变计算实现；大栅格的采样统计不是全图精确统计。

子上下文只有自身目标、经验证输入元数据、上游绑定、预期输出、允许工具及自己的协议消息；不复制父会话历史、长期记忆、主 WorkingMemory，也不允许调用历史检索工具。

子目录为 `workspace/users/<用户>/runs/<child_run_id>/{intermediate,output,temp}`。输出不能逃逸该写入根目录；相同文件名在不同子 Run 中不会冲突。每个子 Run 有独立 agent_id、Checkpoint、工具激活集与预算。共享 WorkingMemory 只由 Coordinator 调用原有合并方法，按稳定 `source_run_id` 顺序合并，不取决于并发完成顺序；完成状态和合并结果同事务提交。

## 幂等与恢复

委派表使用唯一 `call_id` 和计划 SHA-256 指纹。在执行任何副作用前原子保存计划、Task、SubTask、子 Run 身份、父子关联。再次调用相同调用 ID 返回原结果，修改其计划则拒绝。

Checkpoint 增加待执行批次、稳定 ToolCall ID、已计预算标记和批次激活集合。重启仍沿用已有 `reconcile_orphaned_runs()` 标记中断，不自动续跑；通过现有技术恢复入口继续原 Run。已完成子结果、已提交 ToolResult 和合法输出复用，未完成部分继续；工具结果已提交但观察尚未保存的窗口也不会重跑缓冲区。

如果持久化工具占用仍为 RUNNING、又无法确认外部副作用，返回 `SIDE_EFFECT_UNCERTAIN`，不重执行。正常取消的委派为终态；技术恢复不会把取消当作崩溃续跑。

用户补充继续走现有会话消息入口；ConversationService 选择等待中的父 Run，Coordinator 只恢复对应原子子 Run。审批继续使用 `POST /api/v1/approvals/{approval_id}/approve` 和 `/deny`：审批仍绑定真实子 Run、用户、工具、参数和原调用 ID；接口继续父 Run，由内部路由恢复该子任务。一次性消费与拒绝路径均有端到端测试。

父取消或整体时限到达时，设置原有 cancel_event，取消未结束子任务，等待有界清理并持久化终态；已完成产物不删除。同步 GIS 线程不能被 asyncio 强制终止：若清理期间已确认完成，保存真实结果；若仍未退出，保留工具执行占用并报告不确定性，不能声称线程已被强制杀死。

## 真实测试结果摘录

以下 ID 来自 2026-09-26 使用集成测试辅助模型、临时 SQLite、真实 GeoPackage 和真实 `vector.buffer` 的一次执行，并非写死的协议示例。该临时环境运行结束后已清理，ID 不是当前用户库中的资源。省略可选字段：

```json
{
  "delegation_id": "del_ec18548e3da7",
  "parent_run_id": "run_7bb144e39670",
  "status": "SUCCESS",
  "subtasks": [{
    "subtask_id": "a",
    "run_id": "run_f017abe9babe",
    "status": "SUCCESS",
    "execution_status": "SUCCEEDED",
    "datasets": [{"dataset_id": "ds_79cb1ef3aec4", "role": "buffer"}],
    "metrics": {"run_f017abe9babe:buffer.distance": 100.0},
    "error": null
  }],
  "added_dataset_ids": ["ds_79cb1ef3aec4"],
  "failed_subtask_ids": [],
  "blocked_subtask_ids": []
}
```

## Run 历史预算与结果精简

`GEOAGENT_PROTOCOL_HISTORY_TOKENS=25600` 配置模型协议历史的估算目标，包含非 system 的用户/助手消息、完整工具调用参数与 ID、工具结果；不包含系统提示、可信状态或上述工具定义预算。`GEOAGENT_TOOL_RESULT_COMPACTION_RATIO=0.2` 集中配置每组处理比例。两类预算不是整个模型输入的总限额，也不改变会话摘要保留最近 12 条原始消息的设置。

每次模型请求前重新估算。超预算时，从尚未处理的工具结果中取最旧的 20%（向上取整），只将模型视图中的 `output` 正文置空，保留状态、Dataset/Artifact 引用、警告和错误，并添加 `context_compacted=true` 与 `result_reference`。一组完成后重新估算；未达目标时继续处理当前剩余未处理结果的最旧 20%。不按历史总数计算，不设组内条数上限，不增加模型调用。

助手的完整工具调用列表、名称、参数和 ID，以及所有对应结果消息始终保留，不按字符或消息条数拆散批次。`compacted_tool_call_ids` 是本 Run 的 Checkpoint 状态，模型请求前持久化；暂停、模型失败和技术恢复均沿用标记，已处理正文不会重新参加比例计算。后续 Checkpoint 仍保存原始协议，数据库中的 ToolResult 也不改写。

`result_reference` 指向 `checkpoint.protocol_messages` 中的真实 Run/调用 ID，只是原文位置，不是事实正文，也没有新增读取原始结果的工具。模型不能从引用猜测已清除正文的数值或为恢复证据重新执行有副作用的工具。本次不做语义摘要、自动回读或额外模型压缩；原有 ConversationMemory 摘要机制不变。

这是正文精简目标，不是硬性总上下文上限：如果全部结果正文都已移除，用户消息、调用参数或保留的错误/引用仍超过预算，历史仍可能超过 25.6k。不擅自删除批次、裁剪参数或添加预算不足提示。可信上下文原有字符限制、旧 Checkpoint 已被截断内容的修复、跨 Run 缓存及工具检索规则不在本次范围内。

## 本地分词与运行用量

`tokenizers==0.23.2` 是正式依赖。默认使用随源码提供的 `resources/tokenizers/qwen3.json`，固定上游版本、哈希和 Apache-2.0 许可证见同目录 `SOURCE.md`。不在运行时下载词表，也不额外请求模型计数接口。词表按绝对路径缓存，禁用截断/填充；替换旧的中英文字符估算逻辑。

全局 `GEOAGENT_TOKENIZER_FILE` 可指定本地 `tokenizer.json` 文件；`GEOAGENT_MODEL_PROFILES` 每项的 `tokenizer_file` 可覆盖全局配置，建议使用绝对路径。模型适配器启动时校验词表，文件不存在或内容不合法时明确失败，不静默使用另一种算法。历史 25.6k、工具 3.2k 和 8 个延迟条目的预算保持不变；历史精简、工具选择预算、会话摘要触发及批次预算都使用当前适配器的本地分词方法。

本地 token 数仅是对应词表的编码数，不模拟云端聊天模板、隐藏推理和协议开销。默认公开 Qwen3 词表不能证明与 `qwen3.8-max` 一致；需要精确预算应提供对应模型的词表。供应商响应的 `usage` 用于实际运行消耗，不参与发送前的历史/工具分项预算。

`Run.token_usage` 分别保留本地输入/输出、完整 API usage 输入/输出及模型调用次数。每次 AgentLoop 正常收到模型响应后计入一次（工具调用名称和参数也计入本地输出），并发布 `TokenUsageUpdated` 累计快照。不增加模型请求，也不把工具调用次数当作模型次数。SQL 事务同时累加子 Run 和父 Run，普通状态保存保留最新计数；并行、暂停恢复、重复委派不重置或重复累加已完成调用。

聊天界面在工具调用次数后显示 `Token 总量（输入 / 输出）`，运行中每轮响应完成后更新；子 Run 用量包含在主 Run。所有已完成调用均返回完整 usage 时优先展示实际量；任一轮缺少完整 usage 则展示全程本地估算并明确标注，不能将部分实际量与估算相加冒充精确量。累计快照按模型调用次数选择，重复或并行晚到事件不重复累加、不倒退，也不替换工具执行进度。刷新后从 Run 持久化字段恢复，旧 Run 无用量不显示虚假的零消耗。

限制：现有决策仍是 `complete` 调用，因此不是逐 token 流式计量；正在进行的请求结束前无法得到最终 usage。失败/超时或进程在响应后、用量入库前中断的调用无法核实实际消耗，不计入本指标。会话摘要等 Run 外模型请求不包含在运行用量中；本指标不是账户账单、费用或整个模型输入总限额。恢复只累加新的已完成响应，待执行工具批次恢复本身不新增模型计数。

## 验证与边界

2026-09-26，在 `backend/` 使用 `D:\Python3.12\venvs\GeoAgent\Scripts\python.exe` 运行：

```powershell
& 'D:\Python3.12\venvs\GeoAgent\Scripts\python.exe' -m pytest -q
```

SubAgent 实现阶段完整回归：**158 passed，0 failed，9 项已有依赖弃用警告，89.66 秒**。其中原有 114 项测试保持通过，新增 20 项单测、24 项集成测试，只新增两个测试文件。

中英文工具检索合并调整后完整回归：**170 passed，0 failed，9 项已有依赖弃用警告，100.14 秒**。本次复用已有测试文件，覆盖每次最多两项、同批次去重取并集、空结果不覆盖、下一轮统一提供工具，以及两次检索之间中断后的 Checkpoint 恢复；权限与原有执行流程测试保持通过。

现有工具优先与必要能力检索调整后完整回归：**174 passed，0 failed，11 项依赖弃用警告，79.82 秒**。仅在已有测试文件增加中英文统计发现、真实栅格检查后直接回答，以及缺少统计能力后检索并真实执行的案例；验证提示传递、下一轮工具注入、统计值及调用次数。测试使用可预测模型，验证执行路径与提示契约，不证明真实外部 LLM 一定遵守选择规则；没有在线 LLM 验收。

工具预算与 Run 发现缓存调整后完整回归：**179 passed，0 failed，13 项依赖弃用警告，92.05 秒**。复用已有测试文件，覆盖 Schema 与卡片合计预算、8 项展示上限、Schema 完整性、跨批次保留、超过展示上限后的精确名称缓存恢复、暂停续跑及恢复后的权限复查。真实 GeoTIFF 重投影后执行坡度，验证先前发现的坡度 Schema 不会被后续发现替换；该案例使用可预测模型，未进行在线 LLM 验收。精确名称不可用时的近似候选匹配规则按用户要求暂不修改。

中英文单次组合调用调整后完整回归：**188 passed，0 failed，13 项依赖弃用警告，86.64 秒**。复用已有测试文件，覆盖每路 1/2 项限制、去重取并集、任意一路或两路为空、权限过滤、组合参数校验、单次计数、旧单查询恢复、内部两路检索之间中断后的恢复及缓存不能跳过英文查询。可预测模型驱动真实 GIS 测试：统计路径由 4 次降为 3 次调用，发现坡度及重投影后真实执行的路径由 6 次降为 4 次调用。没有在线 LLM 验收；没有调整模型误判 Schema 状态或坡度 CRS 前置条件描述，也没有修改约 2500 tokens / 8 项预算或匹配规则，不保证模型永远不额外发起搜索。

工具可见状态同步调整后完整回归：**189 passed，0 failed，13 项依赖弃用警告，89.56 秒**。仅复用现有测试文件，新增一个已可调用工具重复缓存查询案例，并补充实际 Schema 与状态逐轮一致、历史状态不累积、工具次数耗尽、8 项淘汰/恢复、权限环境复查、完整 Schema/卡片预算边界和追踪数据断言。真实 GIS 检查、重投影和坡度执行仍通过；状态说明与完整 Schema/卡片合计不超过配置预算。没有在线 LLM 验收，不保证真实模型绝不重复搜索；没有修改匹配规则、前置条件、执行权限或跨 Run 缓存范围。

未用候选卡片化调整后完整回归：**198 passed，0 failed，15 项依赖弃用警告，113.47 秒**。没有新增测试文件，在现有文件覆盖首次检索直接提供候选完整 Schema、单个/多个调用工具的保留、未调用候选降级、执行失败及参数修正后的复用、暂停续跑、批次中断后不提前降级/不重复执行、精确名称卡片恢复及禁止同批提前调用。真实 GeoTIFF 测试分别验证已调用且因 CRS 失败的坡度工具直接复用，以及未调用坡度候选降级后从缓存恢复再执行；子任务环境收紧测试保留恢复前后激活范围与无产物的严格断言。首次发现不新增选择加载轮次，未用工具后续恢复仍可能增加交互。本次修改 Python 文件的 Ruff 检查通过，未修复下述既有全仓库问题；测试仍使用可预测模型，没有在线 LLM 验收。

Run 历史预算与工具结果精简调整后完整回归：**214 passed，0 failed，15 项已有依赖弃用警告，108.87 秒**。在现有测试文件增加 10 个案例（含参数化），验证预算等值边界、最旧未处理结果的 20% 与向上取整、重新估算后停止、重复触发、跨 Run 标记隔离、完整调用批次/参数/JSON、资源引用与错误保留、暂停及模型超时后的 Checkpoint 恢复、原文不改写与不重复执行。没有新增测试文件，没有额外摘要模型调用，也没有在线 LLM 验收。本次修改 Python 文件的 Ruff 检查与 `git diff --check` 均通过；以下既有全仓库问题仍未改动。

对本次全部修改的 Python 文件执行 `ruff check`：通过；`git diff --check`：通过。全仓库 `ruff check app evaluation tests --output-format concise` 仍报 5 项修改前已存在的问题：`app/run/__init__.py` 的导入排序、`app/run/lifecycle.py` 的两个未使用导入、`tests/integration/test_conversation_memory.py` 的两个未使用局部变量。未为追求全绿修改无关文件。

本地分词与运行用量接入完整回归：**225 passed，0 failed，15 项已有依赖弃用警告，119.14 秒**；界面测试 **36 passed**，TypeScript 检查及生产构建通过，依赖 `pip check` 通过。在已有测试文件补充词表一致性、缓存、配置覆盖、缺失/损坏文件、禁止 padding/truncation、实际零 usage 与缺失 usage、多轮累计及工具参数输出、子任务汇总、并发及旧快照覆盖、暂停/超时恢复、WebSocket 事件隔离、界面实际/估算切换、重复/晚到事件等案例，未新增永久测试文件。浏览器本地模拟验证桌面、390px 窄屏和累计量更新，临时预览代码已删除。

使用真实 GIS 库/文件/Executor 和可预测假模型验证，没有使用真实外部 LLM/API Key 进行在线验收。

当前明确限制：

- 调度与锁仅支持一个服务进程，不提供跨进程/分布式执行锁；建议单 worker 部署。
- 同步原生 GIS 计算须协作取消；不确定调用需人工核查后处理，不能自动重跑。
- Python/Shell 的隔离环境与执行权限默认仍未开放，委派不提升权限。
- 指标仅保留有来源的小体积扁平数值/布尔值；不传数组、完整栅格或矢量记录。
- 未增加自动业务重试/自治重规划平台、跨用户资源共享、前端多 Agent 专用展示。
- 只启用 `parallel_two_subtasks`、`optional_failure` 评测；其余预留案例不声称已由通用评测 Runner 支持。
