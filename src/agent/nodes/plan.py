"""
规划节点
负责生成和调整执行计划
"""

from ..state import ExperimentState
from typing import Any
import json
import re


def plan_node(state: ExperimentState) -> dict[str, Any]:
    """
    规划节点：根据当前状态生成或调整执行计划

    Args:
        state: 当前实验状态

    Returns:
        状态更新字典
    """
    updates = {}

    # 如果是初始规划
    if not state.current_plan:
        # 1. 检索历史经验
        similar_experiences = _retrieve_memories(state)

        # 2. 判断是否可以复用高质量历史策略
        best = _find_best_reusable_strategy(similar_experiences)
        if best:
            import json
            plan = json.loads(best.plan_steps)
            updates["current_plan"] = plan
            updates["current_step"] = 0
            updates["memory_hint"] = (
                f"复用历史策略 (quality={best.quality_score:.2f}, "
                f"detection={best.detection_rate:.1%}, "
                f"track={best.track_continuity:.1%})"
            )
            print(f"[plan] {updates['memory_hint']}")
            return updates

        # 3. 生成新计划（将历史经验注入作为参考）
        plan = _generate_initial_plan_with_llm(state, similar_experiences=similar_experiences)
        updates["current_plan"] = plan
        updates["current_step"] = 0
    else:
        # 根据反思结果调整计划
        plan = _adjust_plan(state)
        updates["current_plan"] = plan

    return updates


def _retrieve_memories(state: ExperimentState):
    """从长期记忆中检索相似实验的经验"""
    try:
        from ..memory.store import MemoryStore
        store = MemoryStore()
        similar = store.retrieve_similar(
            experiment_type=state.experiment_type or "unknown",
            species=state.species or "unknown",
            constraints=state.constraints,
            top_k=3,
        )
        if similar:
            print(f"[plan] 发现 {len(similar)} 条相似历史经验")
        return similar
    except Exception as e:
        print(f"[plan] 记忆检索失败: {e}")
        return []


def _find_best_reusable_strategy(similar_experiences) -> "ExperimentExperience | None":
    """判断历史经验中是否有可直接复用的高质量策略

    复用标准：
    - quality_score >= 0.92（优秀）
    - success == True（一次通过，无修复）
    - 至少修复 0 次
    """
    if not similar_experiences:
        return None
    best = similar_experiences[0]
    if best.quality_score >= 0.92 and best.success and best.repair_attempts == 0:
        return best
    return None


def _format_memories_for_prompt(experiences) -> str:
    """将历史经验格式化为 LLM prompt 文本"""
    if not experiences:
        return ""

    lines = ["\n\n历史经验参考（同类型实验的成功案例）："]
    for i, exp in enumerate(experiences[:3], 1):
        import json
        steps = json.loads(exp.plan_steps)
        lines.append(f"\n案例 {i} (质量分: {exp.quality_score:.2f}, 检测率: {exp.detection_rate:.1%}, 跟踪连续性: {exp.track_continuity:.1%}):")
        lines.append(f"  - 策略: {' -> '.join(steps)}")
        if exp.used_enhance_video:
            lines.append("  - 使用了视频增强")
        if exp.failure_mode:
            lines.append(f"  - 曾遇到问题: {exp.failure_mode}")
        else:
            lines.append("  - 一次通过，无修复")
    return "\n".join(lines)


def _generate_initial_plan_with_llm(state: ExperimentState, similar_experiences=None) -> list[str]:
    """
    使用 LLM 生成初始执行计划

    Args:
        state: 当前实验状态

    Returns:
        执行步骤列表
    """
    # 尝试使用 LLM
    try:
        from ..prompts import PLAN_SYSTEM_PROMPT, PLAN_USER_PROMPT
        from src.llm import get_llm_client

        llm = get_llm_client()

        # 准备约束条件描述
        constraints = []
        if state.constraints.get("low_light"):
            constraints.append("- 低光环境，需要视频增强")
        if state.constraints.get("target_behavior"):
            constraints.append(f"- 关注行为: {state.constraints['target_behavior']}")

        constraints_str = "\n".join(constraints) if constraints else "无特殊约束"

        # 识别质量问题
        quality_issues = []
        if state.video_metadata and state.video_metadata.brightness:
            if state.video_metadata.brightness < 50:
                quality_issues.append("低亮度")
        quality_issues_str = ", ".join(quality_issues) if quality_issues else "正常"

        user_prompt = PLAN_USER_PROMPT.format(
            user_request=state.user_request,
            experiment_type=state.experiment_type or "未指定",
            species=state.species or "未指定",
            duration=state.video_metadata.duration if state.video_metadata else 0,
            quality_issues=quality_issues_str,
            constraints=constraints_str
        )

        # 注入历史经验到 prompt
        memory_context = _format_memories_for_prompt(similar_experiences)
        if memory_context:
            user_prompt += memory_context
            user_prompt += "\n\n请优先参考上述历史成功案例的策略，如果当前条件相似，建议采用相同的分析流程。"

        # 使用同步方法调用 LLM
        response = llm.chat_sync([
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], timeout=30)

        # 解析响应
        parsed = _extract_json_from_response(response)

        if parsed and "plan" in parsed:
            # 验证并过滤计划步骤
            valid_steps = _validate_plan_steps(parsed["plan"])
            if valid_steps:
                # 线虫实验必须包含 segment（在 track 之前）
                if (state.species == "worm" or state.experiment_type == "worm_assay") and "segment" not in valid_steps:
                    if "track" in valid_steps:
                        valid_steps.insert(valid_steps.index("track"), "segment")
                    else:
                        valid_steps.append("segment")

                # 确保可视化步骤始终包含在计划中
                if "generate_trajectory_plot" not in valid_steps:
                    valid_steps.append("generate_trajectory_plot")
                if "generate_heatmap" not in valid_steps:
                    valid_steps.append("generate_heatmap")
                # generate_report 由用户在分析完成后手动触发，不加入自动计划
                return valid_steps

    except Exception as e:
        print(f"[plan] LLM 规划失败，使用规则引擎: {e}")

    # 回退到规则引擎
    return _generate_initial_plan_rules(state)


def _validate_plan_steps(steps: list[str]) -> list[str]:
    """
    验证并过滤计划步骤

    Args:
        steps: 原始步骤列表

    Returns:
        有效步骤列表
    """
    valid_tools = {
        "detect", "track", "segment", "enhance_video",
        "calculate_open_field_metrics", "calculate_water_maze_metrics",
        "calculate_epm_metrics", "calculate_worm_metrics",
        "calculate_zebrafish_metrics",
        "generate_trajectory_plot", "generate_heatmap",
        "generate_report",
    }

    valid_steps = []
    for step in steps:
        # 规范化步骤名称
        step_lower = step.lower().strip()
        step_normalized = step_lower.replace(" ", "_").replace("-", "_")

        # 匹配有效工具
        for tool in valid_tools:
            if tool in step_normalized or step_normalized in tool:
                if tool not in valid_steps:  # 避免重复
                    valid_steps.append(tool)
                break

    return valid_steps if valid_steps else None


def _extract_json_from_response(response: str) -> dict | None:
    """
    从 LLM 响应中提取 JSON

    Args:
        response: LLM 响应文本

    Returns:
        解析后的字典，失败返回 None
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块（非贪婪匹配，避免跨块）
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*?\}'
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def _generate_initial_plan_rules(state: ExperimentState) -> list[str]:
    """
    使用规则引擎生成初始计划 (回退方案)

    Args:
        state: 当前实验状态

    Returns:
        执行步骤列表
    """
    plan = []

    # 基础流程
    plan.append("detect")  # 目标检测

    # 线虫实验需要 SAM 分割
    if state.species == "worm" or state.experiment_type == "worm_assay":
        plan.append("segment")

    plan.append("track")   # 目标跟踪

    # 根据实验类型添加分析步骤
    if state.experiment_type == "open_field":
        plan.append("calculate_open_field_metrics")
    elif state.experiment_type == "morris_water_maze":
        plan.append("calculate_water_maze_metrics")
    elif state.experiment_type == "epm":
        plan.append("calculate_epm_metrics")
    elif state.experiment_type == "worm_assay" or state.species == "worm":
        plan.append("calculate_worm_metrics")
    elif state.experiment_type == "zebrafish_plate" or state.species == "zebrafish":
        plan.append("calculate_zebrafish_metrics")
    else:
        # 默认使用旷场
        plan.append("calculate_open_field_metrics")

    # 检查是否需要预处理
    if state.constraints.get("low_light"):
        # 在检测前添加增强步骤
        plan.insert(0, "enhance_video")

    # 可视化步骤
    plan.append("generate_trajectory_plot")
    plan.append("generate_heatmap")
    plan.append("generate_report")

    return plan


def _adjust_plan(state: ExperimentState) -> list[str]:
    """
    根据反思结果调整计划

    Args:
        state: 当前实验状态

    Returns:
        调整后的执行步骤列表
    """
    current_plan = state.current_plan.copy()
    last_result = state.get_last_tool_result()

    if not last_result:
        return current_plan

    # 根据失败模式调整计划
    if last_result.failure_mode == "low_detection_rate":
        # 检测率低，尝试增强
        if "enhance_video" not in current_plan:
            current_plan.insert(0, "enhance_video")

    elif last_result.failure_mode == "track_discontinuity":
        # 跟踪不连续，调整跟踪参数
        current_plan.append("retrack_with_adjusted_params")

    elif last_result.failure_mode == "segmentation_error":
        # 分割错误，尝试不同的分割方法
        current_plan.append("segment_alternative")

    return current_plan


def _get_experiment_specific_steps(experiment_type: str) -> list[str]:
    """
    获取实验特定的分析步骤

    Args:
        experiment_type: 实验类型

    Returns:
        步骤列表
    """
    steps_map = {
        "open_field": [
            "calculate_center_time",
            "calculate_total_distance",
            "calculate_speed_profile",
            "generate_heatmap"
        ],
        "morris_water_maze": [
            "calculate_escape_latency",
            "calculate_path_efficiency",
            "calculate_quadrant_time",
            "generate_trajectory_plot"
        ]
    }

    return steps_map.get(experiment_type, [])


# 保留旧函数名作为别名
_generate_initial_plan = _generate_initial_plan_rules
