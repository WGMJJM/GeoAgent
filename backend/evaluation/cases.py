"""默认评测清单。

清单只描述可观察契约；具体模型响应仍由 Application 的离线/模型配置决定。
测试可以传入自己的 EvaluationCase 列表和 fake model，以保持确定性。
"""

from .models import EvaluationCase


def default_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            id="chat_basic",
            name="普通对话不调用工具",
            prompt="你好",
            expected_status="SUCCESS",
            forbidden_tools=["agent.delegate"],
            max_tool_calls=0,
        ),
        EvaluationCase(
            id="inspect_dataset",
            name="检查单个 Dataset",
            prompt="检查 roads",
            dataset_keys=["roads"],
            expected_tools=["dataset.inspect"],
            notes="需要模型配置或专用 fake model 才会进入工具执行路径。",
        ),
        EvaluationCase(
            id="react_buffer",
            name="多回合 ReAct 缓冲区分析",
            prompt="检查 roads 并生成 500 米缓冲区",
            dataset_keys=["roads"],
            expected_tools=["dataset.inspect", "vector.buffer"],
            expected_recovery=["REPAIR"],
        ),
        EvaluationCase(
            id="delegation_three_topics",
            name="道路、人口和地形委派",
            prompt="综合道路、人口和 DEM，从三个方面评价当前区域",
            dataset_keys=["roads", "population", "dem"],
            expect_delegation=True,
            expected_subtask_count=3,
        ),
        EvaluationCase(
            id="knowledge_query",
            name="GIS 知识查询",
            prompt="什么是 EPSG:4326，坐标单位是什么？",
            expected_status="SUCCESS",
            max_tool_calls=0,
        ),
        EvaluationCase(
            id="missing_dataset_reference",
            name="缺失数据引用需要阻断",
            prompt="用刚才不存在的数据继续",
            expected_status="BLOCKED",
            max_tool_calls=0,
        ),
    ]


def execution_cases() -> list[EvaluationCase]:
    """需要 fake model 或已配置模型的执行层契约清单。"""

    return [
        EvaluationCase(id="invalid_tool_arguments", name="非法工具参数", prompt="用错误参数检查 roads", dataset_keys=["roads"], expected_status="BLOCKED", notes="由 fake model 产生非法 JSON 工具参数。", enabled=False),
        EvaluationCase(id="retry_recovery", name="可重试失败恢复", prompt="执行一次可重试工具", expected_recovery=["RETRY"], enabled=False),
        EvaluationCase(id="crs_repair", name="CRS 修复恢复", prompt="用经纬度数据计算距离", dataset_keys=["roads"], expected_recovery=["REPAIR"], enabled=False),
        EvaluationCase(id="verification_failure", name="输出验证失败", prompt="生成一个不可读结果", expected_directive="ABORT", enabled=False),
        EvaluationCase(id="ask_user_signal", name="执行层请求用户补充", prompt="执行缺少输入的操作", expected_directive="ASK_USER", enabled=False),
        EvaluationCase(id="replan_signal", name="执行层请求重新决策", prompt="执行需要重新规划的操作", expected_directive="REPLAN", enabled=False),
        EvaluationCase(id="tool_batch_mixed", name="混合工具批次", prompt="同时执行多个相互独立的操作", enabled=False),
        EvaluationCase(id="runtime_budget", name="运行时预算保护", prompt="执行一个超过回合预算的操作", expected_status="BLOCKED", enabled=False),
    ]


def multi_agent_cases() -> list[EvaluationCase]:
    """多智能体调度与依赖传播清单。"""

    return [
        EvaluationCase(id="parallel_two_subtasks", name="两个独立子任务并行", prompt="分别检查 roads 和 population", dataset_keys=["roads", "population"], expect_delegation=True, expected_subtask_count=2, notes="已由真实 GIS 并发集成测试覆盖；Runner 必须传入 user_id 和模型。"),
        EvaluationCase(id="parallel_three_subtasks", name="三个独立子任务并行", prompt="综合道路、人口和 DEM", dataset_keys=["roads", "population", "dem"], expect_delegation=True, expected_subtask_count=3, enabled=False),
        EvaluationCase(id="dependency_failure_blocks", name="依赖失败阻断后继", prompt="先处理输入再处理依赖结果", expect_delegation=True, expected_directive="ABORT", enabled=False),
        EvaluationCase(id="required_failure", name="必需子任务失败", prompt="执行必需子任务", expect_delegation=True, expected_directive="ABORT", enabled=False),
        EvaluationCase(id="optional_failure", name="可选子任务失败", prompt="执行可选质量检查", expect_delegation=True, expected_status="PARTIAL", notes="已由必需/可选失败集成测试覆盖；Runner 必须传入 user_id 和模型。"),
        EvaluationCase(id="subagent_timeout", name="子智能体超时", prompt="执行一个超时子任务", expect_delegation=True, expected_status="BLOCKED", enabled=False),
        EvaluationCase(id="subagent_ask_user", name="子智能体请求补充", prompt="子任务需要用户输入", expect_delegation=True, expected_directive="ASK_USER", enabled=False),
        EvaluationCase(id="subagent_replan", name="子智能体请求重规划", prompt="子任务需要重新规划", expect_delegation=True, expected_directive="REPLAN", enabled=False),
        EvaluationCase(id="duplicate_delegation", name="重复委派无进展保护", prompt="重复执行相同委派", expect_delegation=True, enabled=False),
    ]


def lifecycle_cases() -> list[EvaluationCase]:
    """需要预置 Run/Approval/Checkpoint 的生命周期契约清单。

    这些案例由集成测试或带状态工厂的 Runner 驱动，默认不混入普通离线清单。
    """

    return [
        EvaluationCase(id="ask_user_waiting", name="ASK_USER 进入人工等待", prompt="请补充一个必要字段", expected_status="BLOCKED", expected_directive="ASK_USER", enabled=False),
        EvaluationCase(id="answer_continue_same_task", name="回答后继续同一任务", prompt="继续并使用刚才的回答", enabled=False),
        EvaluationCase(id="answer_creates_new_run", name="回答创建新运行", prompt="回答后应创建新的 Run", enabled=False),
        EvaluationCase(id="unresolved_question_resolved", name="待回答问题按来源清理", prompt="回答上一个问题", enabled=False),
        EvaluationCase(id="repeated_clarification", name="重复澄清不产生重复问题", prompt="再次请求同一澄清", enabled=False),
        EvaluationCase(id="risky_tool_requests_approval", name="高风险工具请求审批", prompt="执行需要审批的工具", expected_status="BLOCKED", enabled=False),
        EvaluationCase(id="approve_exact_call", name="批准精确工具调用", prompt="批准同一参数的工具调用", enabled=False),
        EvaluationCase(id="approval_is_one_shot", name="审批只能消费一次", prompt="重复使用同一审批", enabled=False),
        EvaluationCase(id="changed_arguments_require_new_approval", name="参数变化需要新审批", prompt="修改已审批工具参数", enabled=False),
        EvaluationCase(id="deny_blocks_execution", name="拒绝审批阻止执行", prompt="拒绝危险工具", enabled=False),
        EvaluationCase(id="completed_resume_is_idempotent", name="完成运行恢复幂等", prompt="恢复已完成运行", enabled=False),
        EvaluationCase(id="waiting_user_cannot_resume", name="等待用户不能技术恢复", prompt="恢复等待用户运行", enabled=False),
        EvaluationCase(id="waiting_approval_cannot_resume_without_grant", name="等待审批不能绕过恢复", prompt="恢复等待审批运行", enabled=False),
        EvaluationCase(id="failed_run_uses_retry_not_resume", name="失败运行使用重试", prompt="恢复业务失败运行", enabled=False),
        EvaluationCase(id="restart_reconciliation", name="重启对账中断孤儿运行", prompt="服务重启后检查运行状态", enabled=False),
        EvaluationCase(id="cancel_cascades_children", name="取消父运行级联子运行", prompt="取消带子运行的任务", enabled=False),
    ]


__all__ = ["default_cases", "execution_cases", "lifecycle_cases", "multi_agent_cases"]
