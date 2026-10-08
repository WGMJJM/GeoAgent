"""主 Agent 的任务关联控制协议，不创建额外的路由模型或业务工具。"""

from __future__ import annotations

import json
import re
from typing import Literal

from app.core.models import StrictModel

TASK_FEEDBACK_PREFIX = "任务关联结果（内部状态，不是用户答复）：\n"
TASK_PROMPT = """任务关联规则：Task 表示用户目标，Run 表示本次执行，Checkpoint 属于各自 Run。
先根据当前请求、task_candidates/legacy_task_candidates 和历史判断目标归属，不用关键词或最近顺序代替理解。
独立新目标可直接调用工具或回答，程序自动创建轻量新任务；普通问候不增加分类或审核轮次。
延续同一未完成目标、补充约束或换方法完成原交付时，在业务操作之前单独输出内部控制 JSON：
{"action":"associate_task","relation":"continue","task_id":"真实任务 ID"}。
只查询旧结果、解释旧失败、不要求继续完成原目标时使用 relation="reference"，不会重开或改写旧任务。
旧执行还没有 Task 时，可用 source_run_id 替代 task_id，程序核验后关联；两个标识只能提供一个。
候选列表有限；目录不足时可用 conversation.search_history 查找原消息及关联标识，不能猜 ID。
归属不明确时用 agent.ask_user 提问，不默认选择第一个或最近的任务。不要将控制 JSON 与正文或工具调用混合。
关联成功后先读取重新提供的原目标、已核验成果引用和剩余事项，再继续；这不自动恢复旧 Run，也不复制旧 Checkpoint。
已确定归属后不切换 Task。关联被拒绝时解释或澄清原因，不能新建任务绕过未完成执行、审批或未知副作用。
已完成目标的新要求创建新任务；仅引用旧成果不改变旧任务状态。当前回答结束不等于原任务全部完成。
关联动作不计业务工具次数，不输出给用户；最终正文仍使用原有回答协议。"""


class TaskAssociation(StrictModel):
    action: Literal["associate_task"]
    relation: Literal["continue", "reference"]
    task_id: str | None = None
    source_run_id: str | None = None


def parse_task_association(content: str) -> TaskAssociation | None:
    """只读取保留的控制动作，不对普通自然语言或普通 JSON 做语义分类。"""
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        if re.match(r'^\s*\{\s*"action"\s*:\s*"associate_task"', content):
            raise ValueError("任务关联请求必须是完整 JSON") from None
        return None
    if not isinstance(value, dict) or value.get("action") != "associate_task":
        return None
    request = TaskAssociation.model_validate(value)
    if bool(request.task_id) == bool(request.source_run_id):
        raise ValueError("请提供 task_id 或 source_run_id 中的一个真实标识")
    return request
