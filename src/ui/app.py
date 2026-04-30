"""
Gradio Web UI 界面
用于生物医学实验分析的交互式界面
"""

import gradio as gr
import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.state import ExperimentState, VideoMetadata
from src.agent.nodes.perceive import perceive_node
from src.agent.nodes.plan import plan_node
from src.agent.nodes.execute import execute_node
from src.agent.nodes.reflect import reflect_node


def analyze_video(
    video_path: str,
    user_request: str,
    progress=gr.Progress()
) -> tuple[dict, str, str, str, str | None, str | None, dict | None, list | None, object]:
    """
    分析视频主函数

    Args:
        video_path: 视频文件路径
        user_request: 用户请求 (LLM 自动推断实验类型和物种)
        progress: Gradio 进度条

    Returns:
        (metrics_dict, interpretation_str, quality_str, detected_info_str,
         trajectory_plot_path, heatmap_path, audit_data, trajectory_df, analysis_state)
    """
    import cv2

    if not video_path:
        return {}, "请先上传视频文件", "", "", None, None, None, None, None, ""

    progress(0.05, desc="正在初始化...")

    # 初始化审计日志
    from src.scientific.audit import AuditLogger, AuditEventType
    audit_logger = AuditLogger()
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audit_logger.start_experiment(
        experiment_id=experiment_id,
        user_request=user_request or "请分析这个实验视频中的动物行为",
        video_path=video_path,
        experiment_type=None  # 将在 perceive 后更新
    )

    try:
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 创建初始状态 (不指定实验类型和物种，由 perceive_node 推断)
        progress(0.1, desc="创建实验状态...")

        default_request = user_request or "请分析这个实验视频中的动物行为"

        state = ExperimentState(
            user_request=default_request,
            video_path=video_path,
            video_metadata=VideoMetadata(
                path=video_path,
                duration=frame_count / fps if fps > 0 else 0,
                fps=fps,
                width=width,
                height=height,
                brightness=65.0
            ),
            current_plan=[],
            current_step=0,
            tool_results=[]
        )

        # 感知：LLM 自动推断实验类型和物种
        progress(0.15, desc="正在分析实验类型...")
        audit_logger.log_event(AuditEventType.NODE_ENTER, node_name="perceive",
                               input_data={"request": state.user_request})
        perceive_updates = perceive_node(state)
        state = ExperimentState(**{**state.model_dump(), **perceive_updates})
        audit_logger.log_decision("perceive",
                                  decision=f"type={state.experiment_type}, species={state.species}",
                                  reason="LLM/规则解析用户请求")
        audit_logger.log_event(AuditEventType.NODE_EXIT, node_name="perceive")

        # 显示检测到的信息
        detected_info = f"""
**检测到的实验类型**: {state.experiment_type or '未知'}
**检测到的物种**: {state.species or '未知'}
**约束条件**: {state.constraints or '无'}
"""

        # 规划
        progress(0.2, desc=f"正在规划 {state.experiment_type or '实验'} 分析流程...")
        audit_logger.log_event(AuditEventType.NODE_ENTER, node_name="plan")
        plan_updates = plan_node(state)
        state = ExperimentState(**{**state.model_dump(), **plan_updates})
        audit_logger.log_decision("plan",
                                  decision=f"plan={state.current_plan}",
                                  reason="LLM/规则生成计划")
        audit_logger.log_event(AuditEventType.NODE_EXIT, node_name="plan")

        # 执行工作流
        total_steps = len(state.current_plan)

        while not state.is_complete:
            step_idx = state.current_step
            current_action = state.current_plan[step_idx] if step_idx < total_steps else "完成"

            # 计算进度
            step_progress = 0.3 + (step_idx / max(total_steps, 1)) * 0.6
            progress(step_progress, desc=f"正在执行: {current_action} ({step_idx + 1}/{total_steps})")

            # 执行一步
            audit_logger.log_tool_call(current_action, input_data={"step": step_idx})
            result = execute_node(state)
            state = ExperimentState(**{**state.model_dump(), **result})

            # 记录工具结果
            last_result = state.tool_results[-1] if state.tool_results else None
            if last_result:
                audit_logger.log_event(
                    AuditEventType.TOOL_RESULT,
                    tool_name=last_result.tool_name,
                    output_data={"success": last_result.success, "error": last_result.error}
                )

        # 反思
        progress(0.95, desc="正在评估结果...")
        audit_logger.log_event(AuditEventType.NODE_ENTER, node_name="reflect")
        reflect_updates = reflect_node(state)
        state = ExperimentState(**{**state.model_dump(), **reflect_updates})
        # 记录质量检查
        if state.quality_metrics:
            audit_logger.log_quality_check(
                quality_metrics=state.quality_metrics.model_dump(),
                is_acceptable=state.quality_metrics.is_acceptable(),
                issues=[] if state.quality_metrics.is_acceptable() else ["检测率或跟踪连续性偏低"]
            )
        audit_logger.log_event(AuditEventType.NODE_EXIT, node_name="reflect")

        progress(1.0, desc="分析完成!")

        # 提取结果 — 从 tool_results 中查找指标计算结果 (不是最后一个)
        metrics_result = None
        for result in reversed(state.tool_results):
            if result.success and result.output and "metrics" in result.output:
                metrics_result = result
                break

        if metrics_result and metrics_result.output:
            metrics = metrics_result.output.get("metrics", {})
            interpretation = metrics_result.output.get("interpretation", {})
            arena_info_from_metrics = metrics_result.output.get("arena_info", {})

            # 格式化指标
            metrics_formatted = {}

            # 旷场实验指标
            open_field_metric_names = {
                "center_time_percent": "中心区时间占比 (%)",
                "edge_time_percent": "边缘区时间占比 (%)",
                "immobile_time_percent": "不动时间占比 (%)",
                "center_entries": "进入中心区次数",
                "avg_distance_to_center": "平均到中心距离 (px)",
                "total_distance": "总移动距离 (px)",
                "total_distance_cm": "总移动距离 (cm)",
                "avg_speed": "平均速度 (px/帧)",
                "avg_speed_cm_s": "平均速度 (cm/s)",
                "max_speed": "最大速度 (px/帧)",
                "path_efficiency": "路径效率"
            }

            # EPM 高架十字迷宫指标
            epm_metric_names = {
                "open_arm_time_percent": "开臂时间占比 (%)",
                "open_arm_entry_percent": "开臂进入次数占比 (%)",
                "closed_arm_entries": "闭臂进入次数",
                "open_arm_entries": "开臂进入次数",
                "closed_arm_time_percent": "闭臂时间占比 (%)",
                "center_time_percent": "中央区时间占比 (%)",
                "total_distance": "总移动距离 (px)",
                "total_distance_cm": "总移动距离 (cm)",
                "avg_speed": "平均速度 (px/帧)",
                "avg_speed_cm_s": "平均速度 (cm/s)",
                "immobile_time_percent": "不动时间占比 (%)"
            }

            # 水迷宫指标
            water_maze_metric_names = {
                "escape_latency_frames": "逃逸潜伏期 (帧)",
                "escape_latency_seconds": "逃逸潜伏期 (秒)",
                "path_length": "路径长度 (cm)",
                "path_length_px": "路径长度 (px)",
                "target_quadrant_time_percent": "目标象限时间占比 (%)",
                "platform_crossings": "平台穿越次数",
                "avg_swim_speed": "平均游泳速度 (cm/s)",
                "avg_swim_speed_px_frame": "平均游泳速度 (px/帧)",
                "avg_distance_to_platform": "平均到平台距离 (px)",
                "thigmotaxis_percent": "边缘游泳比例 (%)"
            }

            # 线虫行为指标
            worm_metric_names = {
                "num_tracks": "检测到的线虫数量",
                "total_distance": "总移动距离 (px)",
                "total_distance_mm": "总移动距离 (mm)",
                "avg_speed": "平均速度 (px/s)",
                "avg_speed_mm_s": "平均速度 (mm/s)",
                "max_speed": "最大速度 (px/s)",
                "immobile_time_percent": "静止时间占比 (%)",
                "avg_body_length": "平均体长 (px)",
                "avg_body_length_mm": "平均体长 (mm)",
                "avg_body_width": "平均体宽 (px)",
                "avg_body_width_mm": "平均体宽 (mm)",
                "body_bend_frequency": "身体弯曲频率 (Hz)",
                "body_wavelength_mean": "平均弯曲波长 (px)",
                "omega_turn_count": "Omega turn 次数",
            }

            # 斑马鱼孔板实验指标
            zebrafish_metric_names = {
                "num_tracks": "检测到的鱼数量",
                "num_tracks_raw": "原始检测轨迹数",
                "total_distance": "总移动距离 (px)",
                "total_distance_mm": "总移动距离 (mm)",
                "avg_speed": "平均速度 (px/s)",
                "avg_speed_mm_s": "平均速度 (mm/s)",
                "max_speed": "最大速度 (px/s)",
                "immobile_time_percent": "静止时间占比 (%)",
                "edge_time_percent": "边缘区时间占比 (%)",
                "center_time_percent": "中心区时间占比 (%)",
                "crossing_count": "穿越次数",
            }

            # 根据实验类型选择指标名称
            experiment_type = state.experiment_type or "open_field"
            if experiment_type == "epm":
                metric_names = epm_metric_names
            elif experiment_type == "morris_water_maze":
                metric_names = water_maze_metric_names
            elif experiment_type == "worm_assay":
                metric_names = worm_metric_names
            elif experiment_type == "zebrafish_plate":
                metric_names = zebrafish_metric_names
            else:
                metric_names = open_field_metric_names

            for key, value in metrics.items():
                display_name = metric_names.get(key, key)
                if isinstance(value, float):
                    metrics_formatted[display_name] = round(value, 2)
                else:
                    metrics_formatted[display_name] = value

            # 格式化解释
            if interpretation:
                # EPM 实验解释
                if experiment_type == "epm":
                    interp_str = f"""
**焦虑水平**: {interpretation.get('anxiety_level', '-')}

**活动水平**: {interpretation.get('activity_level', '-')}

**探索水平**: {interpretation.get('exploration_level', '-')}

**焦虑描述**: {interpretation.get('anxiety_description', '-')}

**活动描述**: {interpretation.get('activity_description', '-')}

**总结**: {interpretation.get('summary', '-')}
"""
                # 水迷宫实验解释
                elif experiment_type == "morris_water_maze":
                    interp_str = f"""
**学习水平**: {interpretation.get('learning_level', '-')}

**路径效率**: {interpretation.get('path_efficiency', '-')}

**搜索策略**: {interpretation.get('search_strategy', '-')}

**运动能力**: {interpretation.get('motor_level', '-')}

**学习描述**: {interpretation.get('learning_description', '-')}

**路径描述**: {interpretation.get('path_efficiency_description', '-')}

**策略描述**: {interpretation.get('strategy_description', '-')}

**运动描述**: {interpretation.get('motor_description', '-')}

**总结**: {interpretation.get('summary', '-')}
"""
                # 线虫实验解释
                elif experiment_type == "worm_assay":
                    track_details = arena_info_from_metrics.get("track_details", [])
                    track_summary_lines = []
                    if track_details:
                        track_summary_lines.append("\n**各线虫个体指标**:")
                        for td in track_details:
                            track_summary_lines.append(
                                f"- Track {td.get('track_id', '?')}: "
                                f"速度={td.get('avg_speed_mm_s', 0):.2f} mm/s, "
                                f"体长={td.get('avg_body_length_mm', 0):.2f} mm, "
                                f"弯曲={td.get('body_bend_frequency', 0):.2f} Hz, "
                                f"Omega={td.get('omega_turn_count', 0)}"
                            )
                    track_summary = "\n".join(track_summary_lines) if track_summary_lines else ""

                    interp_str = f"""
**运动能力**: {interpretation.get('activity_level', '-')}

**弯曲模式**: {interpretation.get('bending_level', '-')}

**Omega turn**: {interpretation.get('omega_turn_level', '-')}

**运动描述**: {interpretation.get('activity_description', '-')}

**弯曲描述**: {interpretation.get('bending_description', '-')}

**Omega turn 描述**: {interpretation.get('omega_turn_description', '-')}

**总结**: {interpretation.get('summary', '-')}
{track_summary}
"""
                # 斑马鱼孔板实验解释
                elif experiment_type == "zebrafish_plate":
                    track_details = arena_info_from_metrics.get("track_details", [])
                    track_summary_lines = []
                    if track_details:
                        track_summary_lines.append("\n**各鱼个体指标**:")
                        for td in track_details:
                            track_summary_lines.append(
                                f"- Track {td.get('track_id', '?')}: "
                                f"速度={td.get('avg_speed_mm_s', 0):.2f} mm/s, "
                                f"静止={td.get('immobile_time_percent', 0):.1f}%, "
                                f"边缘={td.get('edge_time_percent', 0):.1f}%, "
                                f"穿越={td.get('crossing_count', 0)}"
                            )
                    track_summary = "\n".join(track_summary_lines) if track_summary_lines else ""

                    interp_str = f"""
**运动能力**: {interpretation.get('activity_level', '-')}

**应激水平**: {interpretation.get('stress_level', '-')}

**探索行为**: {interpretation.get('exploration_level', '-')}

**运动描述**: {interpretation.get('activity_description', '-')}

**应激描述**: {interpretation.get('stress_description', '-')}

**探索描述**: {interpretation.get('exploration_description', '-')}

**总结**: {interpretation.get('summary', '-')}
{track_summary}
"""
                # 旷场实验解释
                else:
                    interp_str = f"""
**焦虑水平**: {interpretation.get('anxiety_level', '-')}

**活动水平**: {interpretation.get('activity_level', '-')}

**焦虑描述**: {interpretation.get('anxiety_description', '-')}

**活动描述**: {interpretation.get('activity_description', '-')}

**总结**: {interpretation.get('summary', '-')}
"""
            else:
                interp_str = "无解释信息"

            # 质量评分
            quality = state.quality_metrics
            if quality:
                quality_str = f"""
**检测率**: {quality.detection_rate:.1%}

**跟踪连续性**: {quality.track_continuity:.1%}

**整体评分**: {'✅ 优秀' if quality.is_acceptable() else '⚠️ 需改进'}
"""
            else:
                quality_str = "无质量信息"

            # 提取可视化路径
            trajectory_plot_path = None
            heatmap_path = None
            for result in state.tool_results:
                if result.success and result.output:
                    if result.tool_name == "generate_trajectory_plot":
                        trajectory_plot_path = result.output.get("image_path")
                    elif result.tool_name == "generate_heatmap":
                        heatmap_path = result.output.get("image_path")

            # 构建原始轨迹数据 DataFrame
            trajectory_df = _build_trajectory_df(state)

            # 结束审计
            audit_path = audit_logger.end_experiment(
                status="success",
                metrics=metrics_formatted,
            )
            # 读取审计 JSON 用于展示
            audit_data = None
            if audit_path:
                import json
                with open(audit_path, 'r', encoding='utf-8') as f:
                    audit_data = json.load(f)

            return metrics_formatted, interp_str, quality_str, detected_info, trajectory_plot_path, heatmap_path, audit_data, trajectory_df, state

        else:
            audit_logger.end_experiment(status="no_results")
            return {}, "分析完成但无结果", "", detected_info, None, None, None, None, state

    except Exception as e:
        import traceback
        try:
            audit_logger.end_experiment(status="error", metrics={"error": str(e)})
        except Exception:
            pass
        return {}, f"分析失败: {str(e)}\n\n{traceback.format_exc()}", "", "", None, None, None, None, None


def generate_report_ui(state: ExperimentState, report_sections: list[str]) -> str:
    """根据分析状态生成报告（UI 调用）"""
    from src.tools.report import generate_behavior_report, REPORT_SECTIONS

    if not state:
        return "请先完成视频分析后再生成报告。"

    # 提取视频信息
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success and result.output:
            vi = result.output.get("video_info", {})
            video_info = {
                "fps": vi.get("fps", 25.0),
                "width": vi.get("width", 640),
                "height": vi.get("height", 480),
                "duration": vi.get("duration", 0),
                "frame_count": result.output.get("frame_count", 0),
            }
            break

    # 提取指标结果
    metric_tools = {
        "calculate_open_field_metrics",
        "calculate_water_maze_metrics",
        "calculate_epm_metrics",
        "calculate_worm_metrics",
        "calculate_zebrafish_metrics",
    }
    metrics = {}
    interpretation = {}
    arena_info = {}
    for result in reversed(state.tool_results):
        if result.tool_name in metric_tools and result.success and result.output:
            metrics = result.output.get("metrics", {})
            interpretation = result.output.get("interpretation", {})
            arena_info = result.output.get("arena_info", {})
            break

    # 提取可视化路径
    visualization_paths = {"trajectory_plot": None, "heatmap": None}
    for result in state.tool_results:
        if result.tool_name == "generate_trajectory_plot" and result.success:
            visualization_paths["trajectory_plot"] = result.output.get("image_path")
        if result.tool_name == "generate_heatmap" and result.success:
            visualization_paths["heatmap"] = result.output.get("image_path")

    # 质量指标
    quality = None
    if state.quality_metrics:
        quality = {
            "detection_rate": state.quality_metrics.detection_rate,
            "track_continuity": state.quality_metrics.track_continuity,
        }

    # 过滤无效板块
    sections = [s for s in report_sections if s in REPORT_SECTIONS]
    if not sections:
        sections = list(REPORT_SECTIONS.keys())

    try:
        report_md = generate_behavior_report(
            experiment_type=state.experiment_type or "unknown",
            species=state.species,
            video_path=state.video_path,
            video_info=video_info,
            metrics=metrics,
            interpretation=interpretation,
            arena_info=arena_info,
            quality_metrics=quality,
            visualization_paths=visualization_paths,
            user_request=state.user_request,
            experiment_id=state.experiment_id or "",
            sections=sections,
        )

        # 保存报告文件
        import os
        output_dir = os.path.join(os.path.dirname(state.video_path), f"{os.path.splitext(os.path.basename(state.video_path))[0]}_output")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "behavior_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        # 在报告末尾添加文件路径
        report_md += f"\n\n---\n\n**报告已保存**: `{report_path}`"
        return report_md
    except Exception as e:
        return f"报告生成失败: {str(e)}"


def export_report_html(state: ExperimentState, report_sections: list[str]) -> str | None:
    """导出 HTML 报告（供浏览器打印为 PDF）"""
    from src.tools.report import generate_behavior_report, REPORT_SECTIONS, generate_html_report
    import os

    if not state:
        return None

    # 提取视频信息
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success and result.output:
            vi = result.output.get("video_info", {})
            video_info = {
                "fps": vi.get("fps", 25.0),
                "width": vi.get("width", 640),
                "height": vi.get("height", 480),
                "duration": vi.get("duration", 0),
                "frame_count": result.output.get("frame_count", 0),
            }
            break

    # 提取指标结果
    metric_tools = {
        "calculate_open_field_metrics",
        "calculate_water_maze_metrics",
        "calculate_epm_metrics",
        "calculate_worm_metrics",
        "calculate_zebrafish_metrics",
    }
    metrics = {}
    interpretation = {}
    arena_info = {}
    for result in reversed(state.tool_results):
        if result.tool_name in metric_tools and result.success and result.output:
            metrics = result.output.get("metrics", {})
            interpretation = result.output.get("interpretation", {})
            arena_info = result.output.get("arena_info", {})
            break

    # 提取可视化路径
    visualization_paths = {"trajectory_plot": None, "heatmap": None}
    for result in state.tool_results:
        if result.tool_name == "generate_trajectory_plot" and result.success:
            visualization_paths["trajectory_plot"] = result.output.get("image_path")
        if result.tool_name == "generate_heatmap" and result.success:
            visualization_paths["heatmap"] = result.output.get("image_path")

    # 质量指标
    quality = None
    if state.quality_metrics:
        quality = {
            "detection_rate": state.quality_metrics.detection_rate,
            "track_continuity": state.quality_metrics.track_continuity,
        }

    sections = [s for s in report_sections if s in REPORT_SECTIONS]
    if not sections:
        sections = list(REPORT_SECTIONS.keys())

    try:
        report_md = generate_behavior_report(
            experiment_type=state.experiment_type or "unknown",
            species=state.species,
            video_path=state.video_path,
            video_info=video_info,
            metrics=metrics,
            interpretation=interpretation,
            arena_info=arena_info,
            quality_metrics=quality,
            visualization_paths=visualization_paths,
            user_request=state.user_request,
            experiment_id=state.experiment_id or "",
            sections=sections,
        )

        html_content = generate_html_report(report_md, title="生物医学实验行为分析报告")

        # 保存 HTML 文件
        output_dir = os.path.join(os.path.dirname(state.video_path), f"{os.path.splitext(os.path.basename(state.video_path))[0]}_output")
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, "behavior_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path
    except Exception as e:
        print(f"[export_report] 导出失败: {e}")
        return None


def _build_followup_context(state: ExperimentState) -> str:
    """构建追问对话的 system prompt"""
    import json
    from src.agent.prompts import FOLLOWUP_SYSTEM_PROMPT

    # 提取指标
    metrics = {}
    interpretation = {}
    arena_info = {}

    for result in reversed(state.tool_results):
        if result.success and result.output:
            if "metrics" in result.output:
                metrics = result.output["metrics"]
            if "interpretation" in result.output:
                interpretation = result.output["interpretation"]
            if "arena_info" in result.output:
                arena_info = result.output["arena_info"]

    # 质量指标
    quality = state.quality_metrics
    detection_rate = quality.detection_rate if quality else 0
    track_continuity = quality.track_continuity if quality else 0

    # 视频信息
    vm = state.video_metadata
    duration = vm.duration if vm else 0
    fps = vm.fps if vm else 0
    width = vm.width if vm else 0
    height = vm.height if vm else 0

    # 构建轨迹数据 (带帧号和区域分类)
    return FOLLOWUP_SYSTEM_PROMPT.format(
        experiment_type=state.experiment_type or "未知",
        species=state.species or "未知",
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        metrics_json=json.dumps(metrics, ensure_ascii=False, indent=2),
        interpretation_json=json.dumps(interpretation, ensure_ascii=False, indent=2),
        arena_info_json=json.dumps(arena_info, ensure_ascii=False, indent=2),
        detection_rate=detection_rate,
        track_continuity=track_continuity,
    )


def _build_trajectory_df(state: ExperimentState) -> list | None:
    """从跟踪结果构建轨迹 DataFrame 数据"""
    from src.tools.calculate import tracks_to_trajectories

    # 获取 track_history
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result or not track_result.output:
        return None

    track_history = track_result.output.get("track_history", {})
    if not track_history:
        return None

    trajectories = tracks_to_trajectories(track_history)
    if not trajectories:
        return None

    fps = state.video_metadata.fps if state.video_metadata else 25.0
    experiment_type = state.experiment_type or "open_field"

    # 获取 arena_info
    arena_info = {}
    for result in reversed(state.tool_results):
        if result.success and result.output and "arena_info" in result.output:
            arena_info = result.output["arena_info"]
            break

    if experiment_type in ("worm_assay", "zebrafish_plate"):
        # 多目标实验：汇总所有轨迹，增加 track_id 列
        # 按原始 track_id 排序，并重新映射为连续的 1-based ID
        sorted_trajs = sorted(trajectories, key=lambda t: t.get("track_id", 0))
        id_map = {t.get("track_id", i): i + 1 for i, t in enumerate(sorted_trajs)}
        rows = []
        for traj in sorted_trajs:
            track_id = id_map.get(traj.get("track_id", 0), traj.get("track_id", 0))
            for p in traj.get("positions", []):
                frame = p["frame_idx"]
                time_s = frame / fps if fps > 0 else 0
                x = p["x"]
                y = p["y"]
                zone = _classify_zone(x, y, arena_info, experiment_type)
                rows.append([int(track_id), int(frame), round(time_s, 3), round(x, 1), round(y, 1), zone])
        return rows
    else:
        # 其他：取最长轨迹，Track ID 固定为 0
        main_traj = max(trajectories, key=lambda t: len(t["positions"]))
        positions = main_traj["positions"]
        rows = []
        for p in positions:
            frame = p["frame_idx"]
            time_s = frame / fps if fps > 0 else 0
            x = p["x"]
            y = p["y"]
            zone = _classify_zone(x, y, arena_info, experiment_type)
            rows.append([0, int(frame), round(time_s, 3), round(x, 1), round(y, 1), zone])
        return rows


def _classify_zone(x: float, y: float, arena_info: dict, experiment_type: str) -> str:
    """简单区域分类"""
    import math

    if not arena_info:
        return "未知"

    if experiment_type in ("open_field", "zebrafish_plate"):
        cx = arena_info.get("center_x", 0)
        cy = arena_info.get("center_y", 0)
        r = arena_info.get("center_radius", 0)
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist <= r:
            return "中心区"
        return "边缘区"

    elif experiment_type == "epm":
        cx = arena_info.get("center_x", 0)
        cy = arena_info.get("center_y", 0)
        arm_w = arena_info.get("arm_width", 0)
        arm_l = arena_info.get("arm_length", 0)
        center_sz = arena_info.get("center_size", arm_w)

        half_aw = arm_w / 2
        half_cs = center_sz / 2

        # 中央区
        if abs(x - cx) <= half_cs and abs(y - cy) <= half_cs:
            return "中央区"

        # 开臂 (水平方向)
        if abs(y - cy) <= half_aw:
            if x > cx + half_cs and x <= cx + half_cs + arm_l:
                return "开臂(右)"
            if x < cx - half_cs and x >= cx - half_cs - arm_l:
                return "开臂(左)"

        # 闭臂 (垂直方向)
        if abs(x - cx) <= half_aw:
            if y > cy + half_cs and y <= cy + half_cs + arm_l:
                return "闭臂(下)"
            if y < cy - half_cs and y >= cy - half_cs - arm_l:
                return "闭臂(上)"

        return "中央区边缘"

    elif experiment_type == "morris_water_maze":
        pc = arena_info.get("pool_center", {})
        pcx = pc.get("x", 0)
        pcy = pc.get("y", 0)
        pool_r = arena_info.get("pool_diameter", 0) / 2
        plat_c = arena_info.get("platform_center", {})
        plat_r = arena_info.get("platform_radius", 0)

        # 平台
        dist_plat = math.sqrt((x - plat_c.get("x", 0)) ** 2 + (y - plat_c.get("y", 0)) ** 2)
        if dist_plat <= plat_r:
            return "平台"

        # 象限
        dx = x - pcx
        dy = y - pcy
        dist_pool = math.sqrt(dx ** 2 + dy ** 2)
        if dist_pool > pool_r:
            return "池外"

        # 判断象限 (以平台在目标象限为例)
        plat_dx = plat_c.get("x", 0) - pcx
        plat_dy = plat_c.get("y", 0) - pcy
        if plat_dx >= 0 and plat_dy >= 0:
            target_quad = "目标象限(右下)"
        elif plat_dx < 0 and plat_dy >= 0:
            target_quad = "目标象限(左下)"
        elif plat_dx >= 0 and plat_dy < 0:
            target_quad = "目标象限(右上)"
        else:
            target_quad = "目标象限(左上)"

        # 简化: 用位置判断
        if dx * plat_dx >= 0 and dy * plat_dy >= 0:
            return target_quad

        quad = f"象限({'右' if dx >= 0 else '左'}{'下' if dy >= 0 else '上'})"
        return quad

    return "未知"


def _extract_trajectory_arrays(state: ExperimentState) -> tuple | None:
    """
    从 analysis_state 提取轨迹数据为 numpy 数组 (供 followup 计算工具使用)

    Returns:
        (positions, frame_indices, fps, arena_info, experiment_type) 或 None
    """
    import numpy as np
    from src.tools.calculate import tracks_to_trajectories

    # 获取 track_history
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result or not track_result.output:
        return None

    track_history = track_result.output.get("track_history", {})
    if not track_history:
        return None

    trajectories = tracks_to_trajectories(track_history)
    if not trajectories:
        return None

    # 取最长轨迹
    main_traj = max(trajectories, key=lambda t: len(t["positions"]))
    pos_list = main_traj["positions"]

    positions = np.array([[p["x"], p["y"]] for p in pos_list])
    frame_indices = np.array([p["frame_idx"] for p in pos_list])

    fps = state.video_metadata.fps if state.video_metadata else 25.0

    # 获取 arena_info
    arena_info = {}
    for result in reversed(state.tool_results):
        if result.success and result.output and "arena_info" in result.output:
            arena_info = result.output["arena_info"]
            break

    experiment_type = state.experiment_type or "open_field"

    return positions, frame_indices, fps, arena_info, experiment_type


def handle_followup(
    user_msg: str,
    chat_history: list[dict],
    analysis_state: ExperimentState | None,
) -> tuple[list[dict], list[dict]]:
    """
    处理用户追问 (支持 LLM 工具调用进行精确计算)

    流程:
    1. 提取轨迹数据 (positions, frame_indices 等)
    2. 将用户问题 + FOLLOWUP_TOOLS 发给 LLM
    3. 如果 LLM 返回 tool_calls → 执行 compute_followup → 将结果返回给 LLM
    4. LLM 生成最终自然语言回答

    Args:
        user_msg: 用户追问内容
        chat_history: 对话历史
        analysis_state: 分析状态 (ExperimentState)

    Returns:
        (chatbot_history, chat_history)
    """
    import json
    from src.llm.client import get_llm_client
    from src.tools.followup import compute_followup, FOLLOWUP_TOOLS

    if not analysis_state:
        chat_history.append({"role": "assistant", "content": "请先完成视频分析后再进行追问。"})
        return chat_history, chat_history

    if not user_msg or not user_msg.strip():
        return chat_history, chat_history

    try:
        llm = get_llm_client()

        # 构建 system prompt (含分析指标上下文)
        system_prompt = _build_followup_context(analysis_state)

        # 提取轨迹数据
        traj_data = _extract_trajectory_arrays(analysis_state)

        # 构建多轮消息
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_msg})

        # 第一轮: 带工具调用发送给 LLM
        tool_response = _call_llm_with_tools_sync(llm, messages, FOLLOWUP_TOOLS)

        # 检查是否有工具调用
        tool_calls = tool_response.get("tool_calls")
        if tool_calls and traj_data:
            positions, frame_indices, fps, arena_info, experiment_type = traj_data

            # 处理每个工具调用
            tool_results_msgs = []
            for tc in tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                # 执行计算
                result_str = compute_followup(
                    func_name, func_args,
                    positions, frame_indices, fps,
                    arena_info, experiment_type
                )

                tool_results_msgs.append({
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": tc.id,
                })

            # 第二轮: 将工具结果返回给 LLM 生成自然语言回答
            # 添加 assistant 的工具调用消息
            messages.append(tool_response["message"])
            messages.extend(tool_results_msgs)

            response = llm.chat_sync(messages, timeout=30)
        else:
            # 无工具调用，直接使用 LLM 回复
            response = tool_response.get("content") or "抱歉，我无法回答这个问题。"

        # 更新对话历史
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": response})

    except Exception as e:
        import traceback
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": f"抱歉，处理您的问题时出错: {str(e)}"})

    return chat_history, chat_history


def _call_llm_with_tools_sync(llm, messages: list[dict], tools: list[dict]) -> dict:
    """
    同步调用 LLM 带工具支持

    Args:
        llm: LLMClient 实例
        messages: 消息列表
        tools: 工具定义列表

    Returns:
        {"content": str, "tool_calls": list | None, "message": dict}
    """
    response = llm._client.chat.completions.create(
        model=llm.config.model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=llm.config.temperature,
        max_tokens=llm.config.max_tokens,
        timeout=30
    )

    message = response.choices[0].message

    # 构建 assistant message 用于后续多轮
    assistant_msg = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in message.tool_calls
        ]

    return {
        "content": message.content,
        "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None,
        "message": assistant_msg,
    }


# 创建 Gradio 界面
def create_ui():
    """创建 Gradio UI"""

    with gr.Blocks(title="BioMed-Exp Agent") as demo:

        gr.Markdown("""
# 🧬 BioMed-Exp Agent
**生物医学实验智能分析系统**

上传实验视频，描述你的分析需求，系统将自动识别实验类型并分析动物行为指标。
""")

        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Video(
                    label="📹 实验视频",
                    sources=["upload"],
                    format="mp4"
                )

                user_request = gr.Textbox(
                    label="📝 分析请求",
                    placeholder="例如: 分析这只小鼠在高架十字迷宫中的焦虑行为\n或者: 分析这个旷场实验中大鼠的探索活动\n或者: 分析线虫的运动行为和神经表型",
                    lines=3,
                    value=""
                )

                analyze_btn = gr.Button(
                    "🔍 开始分析",
                    variant="primary",
                    size="lg"
                )

                with gr.Accordion("📋 检测信息", open=False):
                    detected_info = gr.Markdown(label="")

                with gr.Accordion("📄 报告设置", open=False):
                    from src.tools.report import REPORT_SECTIONS
                    report_sections = gr.CheckboxGroup(
                        choices=[(v, k) for k, v in REPORT_SECTIONS.items()],
                        value=list(REPORT_SECTIONS.keys()),
                        label="选择报告中包含的板块",
                    )

            with gr.Column(scale=3):
                with gr.Tab("📊 分析指标"):
                    metrics_output = gr.JSON(
                        label="计算指标",
                        elem_classes=["metrics-table"]
                    )

                with gr.Tab("📋 结果解释"):
                    interpretation_output = gr.Markdown(
                        label="",
                        elem_classes=["interpretation-box"]
                    )

                with gr.Tab("✅ 质量评估"):
                    quality_output = gr.Markdown(
                        label="",
                    )

                with gr.Tab("🗺️ 运动轨迹"):
                    trajectory_plot_output = gr.Image(
                        label="轨迹图",
                        height=500,
                    )

                with gr.Tab("🔥 热力图"):
                    heatmap_output = gr.Image(
                        label="空间密度热力图",
                        height=500,
                    )

                with gr.Tab("💬 追问对话"):
                    chatbot = gr.Chatbot(
                        height=400,
                        label="",
                    )
                    with gr.Row():
                        followup_input = gr.Textbox(
                            placeholder="追问关于分析结果的问题，例如：小鼠处于中心区最长时间是多久？",
                            scale=4,
                            show_label=False,
                        )
                        followup_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Tab("📄 报告"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            from src.tools.report import REPORT_SECTIONS
                            report_sections_ui = gr.CheckboxGroup(
                                choices=[(v, k) for k, v in REPORT_SECTIONS.items()],
                                value=list(REPORT_SECTIONS.keys()),
                                label="选择报告中包含的板块",
                            )
                        with gr.Column(scale=1):
                            generate_report_btn = gr.Button("📄 生成报告", variant="primary")
                            export_html_btn = gr.Button("🖨️ 导出 HTML / PDF", variant="secondary")
                    report_output = gr.Markdown(
                        label="行为分析报告",
                        elem_classes=["report-box"]
                    )
                    report_file_output = gr.File(
                        label="下载报告文件",
                        visible=False,
                    )

                with gr.Tab("🔍 审计日志"):
                    audit_output = gr.JSON(
                        label="实验审计记录",
                    )

                with gr.Tab("🗂️ 原始轨迹"):
                    trajectory_data_output = gr.Dataframe(
                        label="原始轨迹数据",
                        headers=["Track ID", "帧号", "时间(s)", "X(px)", "Y(px)", "区域"],
                        interactive=False,
                    )

        # 示例和说明
        gr.Markdown("""
---
**使用说明**:
1. 上传 MP4 格式的实验视频
2. 用自然语言描述你的分析需求（系统会自动识别实验类型和物种）
3. 点击 "开始分析" 等待结果
4. 分析完成后可切换到 "追问对话" Tab 进行追问

**支持的实验类型**:
- 旷场实验 (Open Field Test) - 焦虑行为、探索活动
- 高架十字迷宫 (Elevated Plus Maze) - 焦虑行为
- 水迷宫实验 (Morris Water Maze) - 空间学习与记忆
- 线虫行为分析 (C. elegans Behavior Assay) - 运动能力、神经表型
- 斑马鱼孔板实验 (Zebrafish Plate Assay) - 药物运动影响、应激反应

**示例请求**:
- "分析这只小鼠在旷场实验中的焦虑行为"
- "评估这只大鼠在高架十字迷宫中的探索活动"
- "分析水迷宫实验中小鼠的空间学习能力"
- "分析线虫在运动平板上的爬行行为"
- "分析斑马鱼孔板实验中的药物运动影响"
""")

        # 分析状态持久化
        analysis_state = gr.State(None)

        # 绑定分析事件
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, user_request],
            outputs=[metrics_output, interpretation_output, quality_output,
                     detected_info, trajectory_plot_output, heatmap_output,
                     audit_output, trajectory_data_output, analysis_state]
        )

        # 绑定生成报告事件
        generate_report_btn.click(
            fn=generate_report_ui,
            inputs=[analysis_state, report_sections_ui],
            outputs=[report_output]
        )

        # 绑定导出 HTML/PDF 事件
        def _export_and_show(state, sections):
            path = export_report_html(state, sections)
            if path:
                return gr.File(value=path, visible=True)
            return gr.File(value=None, visible=False)

        export_html_btn.click(
            fn=_export_and_show,
            inputs=[analysis_state, report_sections_ui],
            outputs=[report_file_output]
        )

        # 绑定追问事件
        followup_btn.click(
            fn=handle_followup,
            inputs=[followup_input, chatbot, analysis_state],
            outputs=[chatbot, chatbot]
        ).then(
            fn=lambda: "",
            outputs=[followup_input]
        )

    return demo


# 便捷启动函数
def launch_ui(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7861):
    """
    启动 Gradio UI

    Args:
        share: 是否创建公开链接
        server_name: 服务器地址
        server_port: 端口号
    """
    demo = create_ui()
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )


if __name__ == "__main__":
    launch_ui()
