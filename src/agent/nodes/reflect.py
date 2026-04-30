"""
反思节点
负责质量评估和修复策略生成
"""

from ..state import ExperimentState, QualityMetrics
from typing import Any
import json
import re


def reflect_node(state: ExperimentState) -> dict[str, Any]:
    """
    反思节点：评估执行结果并决定是否需要修复

    Args:
        state: 当前实验状态

    Returns:
        状态更新字典
    """
    updates = {}

    # 获取最近的执行结果
    last_result = state.get_last_tool_result()

    if not last_result:
        updates["is_complete"] = True
        return updates

    print(f"[reflect] last_result: tool={last_result.tool_name}, success={last_result.success}, has_output={last_result.output is not None}")

    # 评估质量
    if last_result.quality:
        quality = last_result.quality
        updates["quality_metrics"] = quality

        # 检查是否需要修复
        if not quality.is_acceptable():
            updates["repair_attempts"] = state.repair_attempts + 1

            # 生成修复建议
            failure_mode, suggested_fix = _diagnose_failure(state, last_result)
            if failure_mode:
                updates["error_message"] = f"质量不达标: {failure_mode}"

            # 检查是否需要人工审核
            if updates["repair_attempts"] >= state.max_repair_attempts:
                updates["needs_human_review"] = True

    # 使用 LLM 生成解释（应写入指标计算工具的结果，而非最后一个可视化工具）
    metric_result = _find_metric_result(state)
    if metric_result and metric_result.success and metric_result.output:
        interpretation = _generate_interpretation_with_llm(state, metric_result)
        if interpretation:
            metric_result.output["interpretation"] = interpretation

    # 检查是否完成所有步骤
    if state.current_step >= len(state.current_plan):
        if last_result.success and (not last_result.quality or last_result.quality.is_acceptable()):
            updates["is_complete"] = True

    # 实验完成（无论成功还是需人工审核），写入长期记忆
    is_complete = updates.get("is_complete") or state.is_complete
    if is_complete or updates.get("needs_human_review"):
        try:
            from ..memory.store import MemoryStore
            store = MemoryStore()
            exp = store.add_experience(state)
            print(
                f"[reflect] 经验已存入记忆: id={exp.id}, "
                f"type={exp.experiment_type}/{exp.species}, "
                f"quality={exp.quality_score:.2f}, success={exp.success}"
            )
        except Exception as e:
            print(f"[reflect] 记忆存储失败: {e}")

    return updates


def _get_interpretation_schema(experiment_type: str | None) -> str:
    """根据实验类型返回 interpretation 字段的 schema 描述

    确保 LLM 返回的字段名与规则引擎和前端展示一致。
    """
    schemas = {
        "open_field": """
- anxiety_level: 焦虑水平（高/中/低）
- anxiety_description: 焦虑水平描述
- activity_level: 活动水平（高/中/低）
- activity_description: 活动水平描述""",
        "morris_water_maze": """
- learning_level: 学习水平（优秀/良好/中等/受损）
- learning_description: 学习水平描述
- path_efficiency: 路径效率（路径效率优秀/良好/中等/低）
- path_efficiency_description: 路径效率描述
- search_strategy: 搜索策略（空间搜索/边缘搜索/随机搜索）
- strategy_description: 搜索策略描述
- motor_level: 运动能力（运动能力活跃/正常/低下）
- motor_description: 运动能力描述""",
        "epm": """
- anxiety_level: 焦虑水平（高/中/低）
- anxiety_description: 焦虑水平描述
- activity_level: 活动水平（高/中/低）
- activity_description: 活动水平描述""",
        "worm_assay": """
- activity_level: 运动能力（运动能力低下/正常/活跃）
- activity_description: 运动能力描述
- bending_level: 弯曲频率（弯曲频率低/正常/高）
- bending_description: 弯曲频率描述
- omega_turn_level: Omega turn 水平（无/少量/频繁）
- omega_turn_description: Omega turn 描述""",
        "zebrafish_plate": """
- activity_level: 运动能力（运动能力低下/正常/活跃）
- activity_description: 运动能力描述
- stress_level: 应激水平（高应激/中等应激/低应激）
- stress_description: 应激水平描述
- exploration_level: 探索行为（探索行为低下/正常/活跃）
- exploration_description: 探索行为描述""",
    }
    return schemas.get(experiment_type, schemas.get("open_field", ""))


def _fill_missing_descriptions(interpretation: dict[str, Any]) -> dict[str, Any]:
    """补全 LLM 返回中缺失的 *_description 字段

    如果 LLM 只返回了 *_level 没有返回对应的 *_description，
    根据 level 值自动生成一个描述，避免网页端显示为空。
    """
    pairs = [
        ("activity_level", "activity_description", "运动活性"),
        ("anxiety_level", "anxiety_description", "焦虑水平"),
        ("stress_level", "stress_description", "应激水平"),
        ("exploration_level", "exploration_description", "探索行为"),
        ("learning_level", "learning_description", "学习能力"),
        ("bending_level", "bending_description", "身体弯曲"),
        ("omega_turn_level", "omega_turn_description", "Omega turn"),
        ("search_strategy", "strategy_description", "搜索策略"),
        ("path_efficiency", "path_efficiency_description", "路径效率"),
        ("motor_level", "motor_description", "运动能力"),
    ]

    for level_key, desc_key, label in pairs:
        if level_key in interpretation and desc_key not in interpretation:
            level = interpretation[level_key]
            interpretation[desc_key] = f"动物表现出{level}的{label}特征"

    return interpretation


def _find_metric_result(state: ExperimentState) -> "ToolResult | None":
    """找到指标计算工具的结果（用于写入 LLM 解释）"""
    metric_tools = {
        "calculate_open_field_metrics",
        "calculate_water_maze_metrics",
        "calculate_epm_metrics",
        "calculate_worm_metrics",
        "calculate_zebrafish_metrics",
    }
    for result in reversed(state.tool_results):
        if result.tool_name in metric_tools and result.success:
            return result
    return None


def _generate_interpretation_with_llm(state: ExperimentState, result) -> dict | None:
    """
    使用 LLM 生成结果解释

    Args:
        state: 当前实验状态
        result: 工具执行结果

    Returns:
        解释字典
    """
    try:
        from ..prompts import REFLECT_SYSTEM_PROMPT, REFLECT_USER_PROMPT
        from src.llm import get_llm_client

        llm = get_llm_client()

        # 提取关键指标（减少 token 数，加速 LLM 响应）
        metrics = result.output.get("metrics", {}) if result.output else {}
        key_metrics = _extract_key_metrics(metrics, state.experiment_type)
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in key_metrics.items()])

        # 根据实验类型构建不同的 interpretation 字段要求
        interpretation_schema = _get_interpretation_schema(state.experiment_type)

        user_prompt = REFLECT_USER_PROMPT.format(
            experiment_type=state.experiment_type or "未指定",
            species=state.species or "未指定",
            metrics=metrics_str,
            detection_rate=result.quality.detection_rate if result.quality else 0,
            track_continuity=result.quality.track_continuity if result.quality else 0
        )

        # 追加实验类型特定的字段要求
        user_prompt += f"\n\n请特别注意，interpretation 字段必须包含以下键（与系统约定一致）：\n{interpretation_schema}"

        # 使用同步方法调用 LLM（reflect 提示词较长，给 60 秒超时）
        response = llm.chat_sync([
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], timeout=90)

        # 解析响应
        parsed = _extract_json_from_response(response)

        if parsed:
            print(f"[reflect] LLM 解释成功，返回字段: {list(parsed.keys())}")
            # 合并 interpretation 子字典，确保字段名与规则引擎一致
            interpretation = parsed.get("interpretation", {})
            base = {
                "quality_assessment": parsed.get("quality_assessment", "良好"),
                "quality_issues": parsed.get("quality_issues", []),
                "summary": parsed.get("summary", ""),
                "detailed_analysis": parsed.get("detailed_analysis", ""),
                "recommendations": parsed.get("recommendations", []),
            }
            base.update(interpretation)
            # 补全缺失的 *_description 字段
            base = _fill_missing_descriptions(base)
            return base
        else:
            # LLM 调用成功但 JSON 解析失败，尝试用原始响应作为 summary
            print(f"[reflect] LLM 返回了解释但 JSON 解析失败，原始响应前200字: {response[:200]}...")
            if response and len(response) > 10:
                return {
                    "quality_assessment": "良好",
                    "quality_issues": [],
                    "summary": response.strip()[:500],
                    "detailed_analysis": response.strip(),
                    "recommendations": []
                }

    except Exception as e:
        print(f"[reflect] LLM 解释失败，使用规则引擎: {e}")

    # 回退到规则引擎
    return _generate_interpretation_rules(state, result)


def _extract_key_metrics(metrics: dict[str, Any], experiment_type: str | None) -> dict[str, Any]:
    """根据实验类型提取关键指标，减少 LLM prompt 的 token 数

    只保留最具解释价值的 4-6 个核心指标，过滤掉中间计算值和次要指标。
    """
    if not metrics:
        return {}

    key_map: dict[str, list[str]] = {
        "open_field": [
            "center_time_percent", "edge_time_percent", "total_distance",
            "avg_speed", "center_entries", "immobile_time_percent",
        ],
        "morris_water_maze": [
            "escape_latency_seconds", "target_quadrant_time_percent",
            "platform_crossings", "thigmotaxis_percent", "avg_swim_speed",
        ],
        "epm": [
            "open_arm_time_percent", "open_arm_entries", "closed_arm_entries",
            "total_distance", "avg_speed",
        ],
        "worm_assay": [
            "avg_speed_mm_s", "body_bend_frequency", "omega_turn_count",
            "immobile_time_percent", "avg_body_length_mm",
        ],
        "zebrafish_plate": [
            "total_distance", "avg_speed", "immobile_time_percent",
            "edge_time_percent", "center_time_percent", "crossing_count",
        ],
    }

    keys = key_map.get(experiment_type, list(metrics.keys())[:6])
    return {k: metrics[k] for k in keys if k in metrics}


def _generate_interpretation_rules(state: ExperimentState, result) -> dict:
    """
    使用规则引擎生成解释 (回退方案)

    Args:
        state: 当前实验状态
        result: 工具执行结果

    Returns:
        解释字典
    """
    # 如果工具结果已有解释，优先复用
    existing = result.output.get("interpretation", {}) if result.output else {}
    if existing and isinstance(existing, dict) and "summary" in existing:
        return existing

    metrics = result.output.get("metrics", {}) if result.output else {}

    if state.experiment_type == "open_field":
        return _interpret_open_field(metrics)
    elif state.experiment_type == "morris_water_maze":
        return _interpret_water_maze(metrics)
    elif state.experiment_type == "worm_assay":
        return _interpret_worm(metrics)
    elif state.experiment_type == "zebrafish_plate":
        from ...tools.calculate import _interpret_zebrafish
        return _interpret_zebrafish(metrics)

    return {"summary": "分析完成"}


def _interpret_open_field(metrics: dict[str, float]) -> dict[str, str]:
    """解释旷场实验结果"""
    center_time = metrics.get("center_time_percent", 0)
    immobile_time = metrics.get("immobile_time_percent", 0)
    total_distance = metrics.get("total_distance", 0)

    # 焦虑水平评估
    if center_time < 20:
        anxiety_level = "高焦虑"
        anxiety_desc = "动物在中心区域停留时间较短，表现出较高的焦虑水平"
    elif center_time < 40:
        anxiety_level = "中等焦虑"
        anxiety_desc = "动物表现出中等程度的焦虑行为"
    else:
        anxiety_level = "低焦虑"
        anxiety_desc = "动物在中心区域探索较多，表现出较低的焦虑水平"

    # 活动水平评估
    if total_distance < 1000:
        activity_level = "低活动性"
        activity_desc = "动物移动较少，可能存在运动障碍或抑郁样行为"
    elif total_distance < 3000:
        activity_level = "正常活动"
        activity_desc = "动物表现出正常的探索行为"
    else:
        activity_level = "高活动性"
        activity_desc = "动物表现出活跃的探索行为或过度兴奋"

    return {
        "anxiety_level": anxiety_level,
        "anxiety_description": anxiety_desc,
        "activity_level": activity_level,
        "activity_description": activity_desc,
        "summary": f"{anxiety_level}，{activity_level}"
    }


def _interpret_water_maze(metrics: dict[str, float]) -> dict[str, str]:
    """解释水迷宫实验结果"""
    latency = metrics.get("escape_latency_seconds", 0)
    target_time = metrics.get("target_quadrant_time_percent", 0)
    thigmotaxis = metrics.get("thigmotaxis_percent", 0)

    # 学习水平评估
    if latency < 15:
        learning_level = "学习优秀"
        learning_desc = "动物能快速定位平台，空间学习能力优秀"
    elif latency < 30:
        learning_level = "学习良好"
        learning_desc = "动物表现出良好的空间学习能力"
    elif latency < 45:
        learning_level = "学习中等"
        learning_desc = "动物学习过程正常，但仍有改进空间"
    else:
        learning_level = "学习受损"
        learning_desc = "动物难以找到平台，可能存在空间认知障碍"

    # 搜索策略评估
    if thigmotaxis > 50:
        strategy = "边缘搜索"
        strategy_desc = "动物主要沿池壁游泳，表明焦虑或缺乏空间策略"
    elif target_time > 35:
        strategy = "空间搜索"
        strategy_desc = "动物在目标象限停留较多，表明良好的空间记忆"
    else:
        strategy = "随机搜索"
        strategy_desc = "动物搜索策略不明显"

    return {
        "learning_level": learning_level,
        "learning_description": learning_desc,
        "search_strategy": strategy,
        "strategy_description": strategy_desc,
        "summary": f"{learning_level}，{strategy}"
    }


def _interpret_worm(metrics: dict[str, float]) -> dict[str, str]:
    """解释线虫行为实验结果"""
    avg_speed = metrics.get("avg_speed_mm_s", 0)
    immobile = metrics.get("immobile_time_percent", 0)
    omega_turns = metrics.get("omega_turn_count", 0)
    bend_freq = metrics.get("body_bend_frequency", 0)

    # 运动能力评估
    if avg_speed < 0.1:
        activity_level = "运动能力低下"
        activity_desc = "线虫移动极慢，可能存在运动神经元功能障碍或药物抑制作用"
    elif avg_speed < 0.3:
        activity_level = "运动能力正常"
        activity_desc = "线虫表现出正常的爬行速度"
    else:
        activity_level = "运动能力活跃"
        activity_desc = "线虫移动迅速，表现出强烈的运动活性"

    # 身体弯曲评估
    if bend_freq < 0.3:
        bending_level = "弯曲频率低"
        bending_desc = "身体摆动较少，可能与运动协调性下降有关"
    elif bend_freq < 0.8:
        bending_level = "弯曲频率正常"
        bending_desc = "身体正弦波爬行模式正常"
    else:
        bending_level = "弯曲频率高"
        bending_desc = "身体快速摆动，可能处于应激或兴奋状态"

    # Omega turn 评估
    if omega_turns == 0:
        omega_level = "无 Omega turn"
        omega_desc = "未观察到明显的方向反转行为"
    elif omega_turns <= 3:
        omega_level = "少量 Omega turn"
        omega_desc = "线虫表现出正常的趋避反应"
    else:
        omega_level = "频繁 Omega turn"
        omega_desc = "线虫频繁改变运动方向，可能与化学趋化或应激反应有关"

    return {
        "activity_level": activity_level,
        "activity_description": activity_desc,
        "bending_level": bending_level,
        "bending_description": bending_desc,
        "omega_turn_level": omega_level,
        "omega_turn_description": omega_desc,
        "summary": f"{activity_level}，{bending_level}，{omega_level}"
    }


def _diagnose_failure(state: ExperimentState, result) -> tuple[str | None, str | None]:
    """
    诊断失败原因并生成修复建议

    Args:
        state: 当前实验状态
        result: 工具执行结果

    Returns:
        (failure_mode, suggested_fix) 元组
    """
    if not result.quality:
        return None, None

    quality = result.quality

    # 检测率低
    if quality.detection_rate < 0.9:
        if state.constraints.get("low_light"):
            return (
                "low_detection_rate",
                "启用 CLAHE 图像增强后重新检测"
            )
        else:
            return (
                "low_detection_rate",
                "降低置信度阈值或调整检测参数"
            )

    # 跟踪不连续
    if quality.track_continuity < 0.85:
        return (
            "track_discontinuity",
            "调整跟踪器的 max_age 和 min_hits 参数"
        )

    # 分割质量差
    if quality.segmentation_iou and quality.segmentation_iou < 0.8:
        return (
            "segmentation_error",
            "尝试不同的 SAM 模型或调整点提示"
        )

    return None, None


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


def _generate_repair_strategy(state: ExperimentState, failure_mode: str) -> list[str]:
    """
    生成修复策略

    Args:
        state: 当前实验状态
        failure_mode: 失败模式

    Returns:
        修复步骤列表
    """
    strategies = {
        "low_detection_rate": [
            "enhance_video",
            "detect_with_lower_threshold",
            "validate_detection"
        ],
        "track_discontinuity": [
            "adjust_tracker_params",
            "retrack",
            "validate_tracking"
        ],
        "segmentation_error": [
            "refine_sam_prompts",
            "resegment",
            "validate_segmentation"
        ]
    }

    return strategies.get(failure_mode, [])
