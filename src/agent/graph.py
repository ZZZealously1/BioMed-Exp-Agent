"""
LangGraph 工作流图定义
实现感知-规划-执行-反思的完整工作流
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from .state import ExperimentState
from .nodes.perceive import perceive_node
from .nodes.plan import plan_node
from .nodes.execute import execute_node
from .nodes.reflect import reflect_node


def create_experiment_graph() -> StateGraph:
    """
    创建实验分析工作流图

    工作流结构:
    START -> perceive -> plan -> execute -> reflect -> END
                              ^              |
                              |              v
                              +---(retry)----+

    Returns:
        StateGraph: 配置好的工作流图
    """
    # 创建状态图
    workflow = StateGraph(ExperimentState)

    # 添加节点
    workflow.add_node("perceive", perceive_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("reflect", reflect_node)

    # 设置入口点
    workflow.set_entry_point("perceive")

    # 添加边
    workflow.add_edge("perceive", "plan")
    workflow.add_edge("plan", "execute")

    # 条件边：执行后根据结果决定下一步
    workflow.add_conditional_edges(
        "execute",
        decide_next_step,
        {
            "reflect": "reflect",
            "plan": "plan",  # 需要重新规划
            "end": END
        }
    )

    # 条件边：反思后决定是否重试
    workflow.add_conditional_edges(
        "reflect",
        decide_retry,
        {
            "execute": "execute",  # 重试执行
            "plan": "plan",  # 重新规划
            "end": END
        }
    )

    return workflow.compile()


def decide_next_step(state: ExperimentState) -> Literal["reflect", "plan", "end"]:
    """
    决定执行后的下一步

    Args:
        state: 当前实验状态

    Returns:
        下一个节点名称
    """
    # 如果执行失败且没有更多步骤
    if state.error_message and state.current_step >= len(state.current_plan):
        return "end"

    # 如果还有计划中的步骤
    if state.current_step < len(state.current_plan):
        return "reflect"

    # 所有步骤完成
    return "end"


def decide_retry(state: ExperimentState) -> Literal["execute", "plan", "end"]:
    """
    决定是否需要重试或继续下一步

    Args:
        state: 当前实验状态

    Returns:
        下一个节点名称
    """
    # 如果还有更多步骤，继续执行
    if state.current_step < len(state.current_plan):
        return "execute"

    # 所有步骤已完成
    if state.is_complete:
        return "end"

    # 检查质量是否可接受
    if state.quality_metrics and state.quality_metrics.is_acceptable():
        return "end"

    # 检查是否需要人工审核
    if state.needs_human_review:
        return "end"

    # 检查是否可以重试
    if state.should_retry():
        # 根据失败模式决定重试策略
        last_result = state.get_last_tool_result()
        if last_result and last_result.suggested_fix:
            # 有建议的修复方案，重新执行
            return "execute"
        else:
            # 没有明确的修复方案，重新规划
            return "plan"

    # 超过最大重试次数
    return "end"


# 预编译的工作流实例
experiment_graph = create_experiment_graph()
