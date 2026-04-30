"""
执行节点
负责调用 MCP 工具执行具体操作
"""

from ..state import ExperimentState, ToolResult, QualityMetrics
from typing import Any
import numpy as np
import cv2


def execute_node(state: ExperimentState) -> dict[str, Any]:
    """
    执行节点：调用工具执行当前步骤

    Args:
        state: 当前实验状态

    Returns:
        状态更新字典
    """
    updates = {}

    # 获取当前步骤
    if state.current_step >= len(state.current_plan):
        updates["is_complete"] = True
        return updates

    current_action = state.current_plan[state.current_step]

    # 执行工具调用
    result = _execute_tool(state, current_action)

    # 更新状态
    updates["tool_results"] = state.tool_results + [result]
    updates["current_step"] = state.current_step + 1

    # 更新质量指标（使用最后一个工具的质量指标）
    if result.quality:
        updates["quality_metrics"] = result.quality

    # 检查是否完成所有步骤
    if updates["current_step"] >= len(state.current_plan):
        updates["is_complete"] = True

    if result.error:
        updates["error_message"] = result.error

    return updates


def _execute_tool(state: ExperimentState, action: str) -> ToolResult:
    """
    执行工具调用

    Args:
        state: 当前实验状态
        action: 动作名称

    Returns:
        ToolResult 对象
    """
    # 工具映射
    tool_map = {
        "detect": _execute_detect,
        "track": _execute_track,
        "segment": _execute_segment,
        "enhance_video": _execute_enhance,
        "calculate_open_field_metrics": _execute_open_field_metrics,
        "calculate_water_maze_metrics": _execute_water_maze_metrics,
        "calculate_epm_metrics": _execute_epm_metrics,
        "calculate_worm_metrics": _execute_worm_metrics,
        "calculate_zebrafish_metrics": _execute_zebrafish_metrics,
        "generate_trajectory_plot": _execute_trajectory_plot,
        "generate_heatmap": _execute_heatmap,
        "generate_report": _execute_generate_report,
    }

    executor = tool_map.get(action)
    if not executor:
        return ToolResult(
            tool_name=action,
            success=False,
            output=None,
            error=f"未知工具: {action}"
        )

    try:
        return executor(state)
    except Exception as e:
        return ToolResult(
            tool_name=action,
            success=False,
            output=None,
            error=str(e),
            failure_mode="execution_error",
            suggested_fix="检查输入参数和系统配置"
        )


def _execute_detect(state: ExperimentState) -> ToolResult:
    """执行目标检测"""
    # 水迷宫实验：使用颜色阈值法（跳过 YOLO，检测率更高）
    if state.experiment_type == "morris_water_maze":
        from ...tools.detect import detect_mwm_mouse

        det_result = detect_mwm_mouse(state.video_path)

        return ToolResult(
            tool_name="detect",
            success=True,
            output={
                "boxes": det_result.boxes,
                "frame_count": det_result.frame_count,
                "video_info": det_result.video_info,
            },
            quality=QualityMetrics(
                detection_rate=det_result.detection_rate,
                track_continuity=0.0,
            )
        )

    from ...tools.detect import load_model, parse_results, TemporalSmoother

    # 加载模型 (按物种路由)
    model = load_model(species=state.species)
    class_names = model.names if hasattr(model, 'names') else {}

    # 初始化时序平滑器 (单目标实验使用；线虫/斑马鱼多目标实验跳过)
    use_smoother = state.species not in ("worm", "zebrafish")
    smoother = TemporalSmoother(
        history_size=10,
        max_displacement=30.0,
        size_variation_threshold=2.0,
        adapt_size=True,
        expected_area_range=None,  # 不使用面积预过滤
        prefer_smaller=True        # 优先选择较小的检测框 (动物 vs 静态结构)
    ) if use_smoother else None

    # 打开视频
    cap = cv2.VideoCapture(state.video_path)
    if not cap.isOpened():
        return ToolResult(
            tool_name="detect",
            success=False,
            output=None,
            error=f"无法打开视频: {state.video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_boxes = []
    frames_with_detection = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理 (斑马鱼使用更低置信度，避免静止鱼漏检)
        conf_threshold = 0.3 if state.species == "zebrafish" else 0.5
        results = model(frame, conf=conf_threshold, verbose=False)

        # 解析结果
        boxes = parse_results(
            results,
            frame_idx,
            class_names,
            single_object=False
        )

        # 单目标时序平滑 (线虫多目标跳过)
        if boxes:
            if use_smoother and smoother:
                boxes = smoother.update(boxes, frame_idx)
            frames_with_detection += 1

        all_boxes.extend(boxes)
        frame_idx += 1

    cap.release()

    # 计算质量指标
    detection_rate = frames_with_detection / frame_count if frame_count > 0 else 0

    return ToolResult(
        tool_name="detect",
        success=True,
        output={
            "boxes": all_boxes,
            "frame_count": frame_count,
            "video_info": {
                "fps": fps,
                "width": width,
                "height": height,
                "duration": frame_count / fps if fps > 0 else 0
            }
        },
        quality=QualityMetrics(
            detection_rate=round(detection_rate, 4),
            track_continuity=0.0
        )
    )


def _execute_track(state: ExperimentState) -> ToolResult:
    """执行目标跟踪"""
    from ...tools.detect import load_model, parse_results, TemporalSmoother
    from ...tools.track import SORTTracker

    # 获取上一个检测结果
    detect_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "detect" and result.success:
            detect_result = result
            break

    if not detect_result:
        # 如果没有检测结果，先执行检测
        detect_result = _execute_detect(state)

    if not detect_result.success:
        return ToolResult(
            tool_name="track",
            success=False,
            output=None,
            error="检测失败，无法执行跟踪"
        )

    # 获取视频信息
    cap = cv2.VideoCapture(state.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    track_history: dict[int, list[dict]] = {}
    wells: list[dict] = []
    frames_with_track = 0

    if state.experiment_type == "morris_water_maze":
        # 水迷宫：复用颜色阈值检测结果，SORT 跟踪（单目标、宽松参数）
        boxes = detect_result.output.get("boxes", []) if detect_result.output else []

        # 按 frame_idx 组织检测框
        boxes_by_frame: dict[int, list[dict]] = {}
        for box in boxes:
            fi = box.get("frame_idx", 0)
            if fi not in boxes_by_frame:
                boxes_by_frame[fi] = []
            boxes_by_frame[fi].append(box)

        tracker = SORTTracker(max_age=60, min_hits=1, iou_threshold=0.1)

        for frame_idx in range(frame_count):
            frame_boxes = boxes_by_frame.get(frame_idx, [])
            if frame_boxes:
                dets = np.array([[b['x'], b['y'], b['w'], b['h']] for b in frame_boxes])
            else:
                dets = np.empty((0, 4))

            tracks = tracker.update(dets)

            for t in tracks:
                x, y, w, h, track_id = t
                track_id = int(track_id)
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append({
                    'frame': frame_idx,
                    'x': float(x),
                    'y': float(y),
                    'w': float(w),
                    'h': float(h)
                })
                frames_with_track += 1

    elif state.species == "zebrafish":
        # 斑马鱼：基于孔位置的空间分割跟踪
        from ...tools.track import well_based_track
        boxes = detect_result.output.get("boxes", []) if detect_result.output else []
        well_result = well_based_track(
            state.video_path, boxes, frame_count, fps
        )
        track_history = well_result["track_history"]
        wells = well_result.get("wells", [])
        frames_with_track = sum(len(v) for v in track_history.values())
    else:
        # 其他物种：全局 SORT 跟踪
        tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)

        cap = cv2.VideoCapture(state.video_path)
        model = load_model(species=state.species)
        class_names = model.names if hasattr(model, 'names') else {}
        use_smoother = state.species not in ("worm", "zebrafish")
        smoother = TemporalSmoother(
            history_size=10,
            max_displacement=30.0,
            size_variation_threshold=2.0,
            adapt_size=True
        ) if use_smoother else None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.5, verbose=False)
            boxes = parse_results(results, frame_idx, class_names, single_object=False)

            if boxes and smoother:
                boxes = smoother.update(boxes, frame_idx)

            if boxes:
                dets = np.array([[b['x'], b['y'], b['w'], b['h']] for b in boxes])
            else:
                dets = np.empty((0, 4))

            tracks = tracker.update(dets)

            for t in tracks:
                x, y, w, h, track_id = t
                track_id = int(track_id)
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append({
                    'frame': frame_idx,
                    'x': float(x),
                    'y': float(y),
                    'w': float(w),
                    'h': float(h)
                })
                frames_with_track += 1

            frame_idx += 1

        cap.release()

        # 线虫：全局轨迹后处理
        if state.species == "worm" and track_history:
            from ...tools.track import refine_track_history
            boxes = detect_result.output.get("boxes", []) if detect_result.output else []
            track_history = refine_track_history(
                track_history, all_detections=boxes, max_gap=30, max_distance=500.0
            )
            frames_with_track = sum(len(v) for v in track_history.values())

    # 计算质量指标
    covered_frames = set()
    for positions in track_history.values():
        for p in positions:
            covered_frames.add(p["frame"])
    track_continuity = len(covered_frames) / frame_count if frame_count > 0 else 0

    output: dict[str, Any] = {
        "track_history": track_history,
        "total_frames": frame_count,
        "num_tracks": len(track_history)
    }
    if wells:
        output["wells"] = wells

    return ToolResult(
        tool_name="track",
        success=True,
        output=output,
        quality=QualityMetrics(
            detection_rate=detect_result.quality.detection_rate if detect_result.quality else 0,
            track_continuity=round(track_continuity, 4)
        )
    )


def _execute_segment(state: ExperimentState) -> ToolResult:
    """执行 SAM 图像分割 (YOLO bbox 作为提示)"""
    from ...tools.segment import segment_video

    # 获取检测结果
    detect_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "detect" and result.success:
            detect_result = result
            break

    if not detect_result:
        return ToolResult(
            tool_name="segment",
            success=False,
            output=None,
            error="没有检测结果，无法进行 SAM 分割"
        )

    boxes = detect_result.output.get("boxes", [])
    if not boxes:
        return ToolResult(
            tool_name="segment",
            success=False,
            output=None,
            error="检测结果为空，无法进行 SAM 分割"
        )

    # 按 frame_idx 组织检测框
    boxes_by_frame: dict[int, list[dict]] = {}
    for box in boxes:
        frame_idx = box["frame_idx"]
        if frame_idx not in boxes_by_frame:
            boxes_by_frame[frame_idx] = []
        boxes_by_frame[frame_idx].append(box)

    try:
        seg_result = segment_video(state.video_path, boxes_by_frame)
        mean_iou = seg_result.get("mean_iou", 0.0)

        return ToolResult(
            tool_name="segment",
            success=True,
            output=seg_result,
            quality=QualityMetrics(
                detection_rate=detect_result.quality.detection_rate if detect_result.quality else 0,
                track_continuity=0.0,
                segmentation_iou=round(mean_iou, 4)
            )
        )
    except Exception as e:
        return ToolResult(
            tool_name="segment",
            success=False,
            output=None,
            error=f"SAM 分割失败: {str(e)}"
        )


def _execute_enhance(state: ExperimentState) -> ToolResult:
    """执行视频增强"""
    from ...tools.detect import enhance_frame

    # 打开视频
    cap = cv2.VideoCapture(state.video_path)
    if not cap.isOpened():
        return ToolResult(
            tool_name="enhance_video",
            success=False,
            output=None,
            error=f"无法打开视频: {state.video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出文件路径
    import os
    base_name = os.path.splitext(os.path.basename(state.video_path))[0]
    output_path = os.path.join(
        os.path.dirname(state.video_path),
        f"{base_name}_enhanced.mp4"
    )

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 增强帧
        enhanced = enhance_frame(frame)
        out.write(enhanced)
        frame_idx += 1

    cap.release()
    out.release()

    return ToolResult(
        tool_name="enhance_video",
        success=True,
        output={
            "enhanced_path": output_path,
            "frames_processed": frame_idx
        },
        quality=None
    )


def _execute_open_field_metrics(state: ExperimentState) -> ToolResult:
    """计算旷场实验指标"""
    from ...tools.calculate import calculate_metrics, tracks_to_trajectories

    # 获取跟踪结果
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result:
        return ToolResult(
            tool_name="calculate_open_field_metrics",
            success=False,
            output=None,
            error="没有跟踪数据，无法计算指标"
        )

    track_history = track_result.output.get("track_history", {})

    if not track_history:
        return ToolResult(
            tool_name="calculate_open_field_metrics",
            success=False,
            output=None,
            error="跟踪数据为空"
        )

    # 获取视频信息
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success:
            video_info = result.output.get("video_info", {})
            break

    # 转换轨迹格式
    trajectories = tracks_to_trajectories(track_history)

    # 场地配置
    arena_config = {
        "width": video_info.get("width", 640),
        "height": video_info.get("height", 480),
        "center_ratio": 0.3
    }

    # 计算指标
    result = calculate_metrics(
        trajectories,
        "open_field",
        arena_config=arena_config,
        fps=video_info.get("fps", 25.0)
    )

    return ToolResult(
        tool_name="calculate_open_field_metrics",
        success=True,
        output={
            "metrics": result.metrics,
            "interpretation": result.interpretation,
            "arena_info": result.arena_info
        },
        quality=QualityMetrics(
            detection_rate=track_result.quality.detection_rate if track_result.quality else 0,
            track_continuity=track_result.quality.track_continuity if track_result.quality else 0
        )
    )


def _execute_water_maze_metrics(state: ExperimentState) -> ToolResult:
    """计算水迷宫实验指标"""
    from ...tools.calculate import calculate_metrics, tracks_to_trajectories

    # 获取跟踪结果
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result:
        return ToolResult(
            tool_name="calculate_water_maze_metrics",
            success=False,
            output=None,
            error="没有跟踪数据，无法计算指标"
        )

    track_history = track_result.output.get("track_history", {})

    if not track_history:
        return ToolResult(
            tool_name="calculate_water_maze_metrics",
            success=False,
            output=None,
            error="跟踪数据为空"
        )

    # 获取视频信息
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success:
            video_info = result.output.get("video_info", {})
            break

    # 转换轨迹格式
    trajectories = tracks_to_trajectories(track_history)

    # 场地配置：水迷宫使用硬编码椭圆场地参数（基于视频标定）
    arena_config = {
        "width": 540,                       # 椭圆长轴直径 (POOL_A * 2 = 270 * 2)
        "pool_center": {"x": 280.0, "y": 280.0},
        "pool_axis_x": 270.0,
        "pool_axis_y": 280.0,
        "platform_center": {"x": 180.0, "y": 200.0},
        "platform_radius": 20.0,
        "pixel_to_cm": 120.0 / 540,         # 120cm 实际直径对应 540px
    }

    # 计算指标
    result = calculate_metrics(
        trajectories,
        "morris_water_maze",
        arena_config=arena_config,
        fps=video_info.get("fps", 25.0)
    )

    return ToolResult(
        tool_name="calculate_water_maze_metrics",
        success=True,
        output={
            "metrics": result.metrics,
            "interpretation": result.interpretation,
            "arena_info": result.arena_info
        },
        quality=QualityMetrics(
            detection_rate=track_result.quality.detection_rate if track_result.quality else 0,
            track_continuity=track_result.quality.track_continuity if track_result.quality else 0
        )
    )


def _execute_epm_metrics(state: ExperimentState) -> ToolResult:
    """计算高架十字迷宫实验指标"""
    from ...tools.calculate import calculate_metrics, tracks_to_trajectories

    # 获取跟踪结果
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result:
        return ToolResult(
            tool_name="calculate_epm_metrics",
            success=False,
            output=None,
            error="没有跟踪数据，无法计算指标"
        )

    track_history = track_result.output.get("track_history", {})

    if not track_history:
        return ToolResult(
            tool_name="calculate_epm_metrics",
            success=False,
            output=None,
            error="跟踪数据为空"
        )

    # 获取视频信息
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success:
            video_info = result.output.get("video_info", {})
            break

    # 转换轨迹格式
    trajectories = tracks_to_trajectories(track_history)

    # EPM 场地配置
    # 十字迷宫参数 (可根据实际设备调整)
    arena_width = video_info.get("width", 1280)
    arena_height = video_info.get("height", 960)

    arena_config = {
        "width": arena_width,
        "height": arena_height,
        "arm_width": arena_width * 0.15,   # 臂宽度约15%
        "arm_length": arena_width * 0.4,   # 臂长度约40%
    }

    # 计算指标
    result = calculate_metrics(
        trajectories,
        "epm",
        arena_config=arena_config,
        fps=video_info.get("fps", 25.0)
    )

    return ToolResult(
        tool_name="calculate_epm_metrics",
        success=True,
        output={
            "metrics": result.metrics,
            "interpretation": result.interpretation,
            "arena_info": result.arena_info
        },
        quality=QualityMetrics(
            detection_rate=track_result.quality.detection_rate if track_result.quality else 0,
            track_continuity=track_result.quality.track_continuity if track_result.quality else 0
        )
    )


def _execute_worm_metrics(state: ExperimentState) -> ToolResult:
    """计算线虫行为指标"""
    from ...tools.calculate import calculate_metrics, tracks_to_trajectories
    from ...tools.skeleton import extract_skeletons_from_masks

    # 获取跟踪结果
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result:
        return ToolResult(
            tool_name="calculate_worm_metrics",
            success=False,
            output=None,
            error="没有跟踪数据，无法计算指标"
        )

    track_history = track_result.output.get("track_history", {})
    if not track_history:
        return ToolResult(
            tool_name="calculate_worm_metrics",
            success=False,
            output=None,
            error="跟踪数据为空"
        )

    # 获取视频信息
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success:
            video_info = result.output.get("video_info", {})
            break

    # 转换轨迹格式
    trajectories = tracks_to_trajectories(track_history)

    # 获取骨架数据
    skeletons = []
    segment_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "segment" and result.success:
            segment_result = result
            break

    if segment_result and segment_result.output:
        masks = segment_result.output.get("masks", [])
        if masks:
            skeletons = extract_skeletons_from_masks(masks)

    # 计算指标
    result = calculate_metrics(
        trajectories,
        "worm_assay",
        arena_config={},
        fps=video_info.get("fps", 25.0),
        skeletons=skeletons
    )

    # 合并 segmentation_iou 到质量指标
    seg_iou = None
    if segment_result and segment_result.quality:
        seg_iou = segment_result.quality.segmentation_iou

    return ToolResult(
        tool_name="calculate_worm_metrics",
        success=True,
        output={
            "metrics": result.metrics,
            "interpretation": result.interpretation,
            "arena_info": result.arena_info
        },
        quality=QualityMetrics(
            detection_rate=track_result.quality.detection_rate if track_result.quality else 0,
            track_continuity=track_result.quality.track_continuity if track_result.quality else 0,
            segmentation_iou=seg_iou
        )
    )


def _execute_zebrafish_metrics(state: ExperimentState) -> ToolResult:
    """计算斑马鱼孔板实验指标"""
    from ...tools.calculate import calculate_metrics, tracks_to_trajectories

    # 获取跟踪结果
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result:
        return ToolResult(
            tool_name="calculate_zebrafish_metrics",
            success=False,
            output=None,
            error="没有跟踪数据，无法计算指标"
        )

    track_history = track_result.output.get("track_history", {})
    if not track_history:
        return ToolResult(
            tool_name="calculate_zebrafish_metrics",
            success=False,
            output=None,
            error="跟踪数据为空"
        )

    # 获取视频信息
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success:
            video_info = result.output.get("video_info", {})
            break

    # 转换轨迹格式
    trajectories = tracks_to_trajectories(track_history)

    # 场地配置（包含孔位信息用于 per-well 指标计算）
    arena_config = {
        "width": video_info.get("width", 640),
        "height": video_info.get("height", 480),
        "center_ratio": 0.3
    }
    wells = track_result.output.get("wells", [])
    if wells:
        arena_config["wells"] = wells

    # 计算指标
    result = calculate_metrics(
        trajectories,
        "zebrafish_plate",
        arena_config=arena_config,
        fps=video_info.get("fps", 25.0)
    )

    return ToolResult(
        tool_name="calculate_zebrafish_metrics",
        success=True,
        output={
            "metrics": result.metrics,
            "interpretation": result.interpretation,
            "arena_info": result.arena_info
        },
        quality=QualityMetrics(
            detection_rate=track_result.quality.detection_rate if track_result.quality else 0,
            track_continuity=track_result.quality.track_continuity if track_result.quality else 0
        )
    )


def _extract_viz_data(state: ExperimentState) -> tuple:
    """从 state.tool_results 提取可视化所需数据"""
    from ...tools.calculate import tracks_to_trajectories

    # 1. 获取 track_history
    track_result = None
    for result in reversed(state.tool_results):
        if result.tool_name == "track" and result.success:
            track_result = result
            break

    if not track_result or not track_result.output:
        return None, None, None, None, None

    track_history = track_result.output.get("track_history", {})
    if not track_history:
        return None, None, None, None, None

    # 转换为 positions 数组
    trajectories = tracks_to_trajectories(track_history)
    if not trajectories:
        return None, None, None, None, None

    experiment_type = state.experiment_type or "open_field"

    if experiment_type in ("worm_assay", "zebrafish_plate"):
        # 多目标实验：合并所有轨迹的位置点
        all_positions = []
        for traj in trajectories:
            for p in traj.get("positions", []):
                all_positions.append([p["x"], p["y"]])
        if not all_positions:
            return None, None, None, None, None
        positions = np.array(all_positions)
    else:
        # 其他：取最长轨迹
        main_traj = max(trajectories, key=lambda t: len(t["positions"]))
        pos_list = main_traj["positions"]
        positions = np.array([[p["x"], p["y"]] for p in pos_list])

    # 2. 获取 arena_info
    arena_info = {}

    metric_tool_map = {
        "open_field": "calculate_open_field_metrics",
        "morris_water_maze": "calculate_water_maze_metrics",
        "epm": "calculate_epm_metrics",
        "worm_assay": "calculate_worm_metrics",
        "zebrafish_plate": "calculate_zebrafish_metrics",
    }
    metric_tool_name = metric_tool_map.get(experiment_type)
    if metric_tool_name:
        for result in reversed(state.tool_results):
            if result.tool_name == metric_tool_name and result.success and result.output:
                arena_info = result.output.get("arena_info", {})
                break

    # 3. 获取骨架数据 (worm)
    skeletons = None
    if experiment_type == "worm_assay":
        segment_result = None
        for result in reversed(state.tool_results):
            if result.tool_name == "segment" and result.success:
                segment_result = result
                break
        if segment_result and segment_result.output:
            from ...tools.skeleton import extract_skeletons_from_masks
            skeletons = extract_skeletons_from_masks(segment_result.output.get("masks", []))

    # 4. 获取视频尺寸
    video_info = {}
    for result in state.tool_results:
        if result.tool_name == "detect" and result.success and result.output:
            video_info = result.output.get("video_info", {})
            break

    video_size = (
        video_info.get("width", state.video_metadata.width if state.video_metadata else 640),
        video_info.get("height", state.video_metadata.height if state.video_metadata else 480),
    )

    return positions, arena_info, experiment_type, video_size, skeletons


def _get_viz_output_path(state: ExperimentState, filename: str) -> str:
    """确定可视化输出路径"""
    import os
    video_dir = os.path.dirname(state.video_path)
    base_name = os.path.splitext(os.path.basename(state.video_path))[0]
    output_dir = os.path.join(video_dir, f"{base_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def _execute_trajectory_plot(state: ExperimentState) -> ToolResult:
    """生成轨迹图"""
    from ...tools.visualize import generate_trajectory_plot

    positions, arena_info, experiment_type, video_size, skeletons = _extract_viz_data(state)

    if positions is None:
        print(f"[visualize] 轨迹图: 数据不足 (track_history或arena_info为空)")
        return ToolResult(
            tool_name="generate_trajectory_plot",
            success=False, output=None,
            error="数据不足，无法生成轨迹图"
        )

    print(f"[visualize] 轨迹图: {len(positions)}个点, 场地={experiment_type}, 尺寸={video_size}, arena_info={bool(arena_info)}")

    output_path = _get_viz_output_path(state, "trajectory_plot.png")

    # 获取分轨轨迹用于多目标绘制
    trajectories_for_viz = None
    if experiment_type in ("worm_assay", "zebrafish_plate"):
        track_result = None
        for result in reversed(state.tool_results):
            if result.tool_name == "track" and result.success:
                track_result = result
                break
        if track_result and track_result.output:
            from ...tools.calculate import tracks_to_trajectories
            trajectories_for_viz = tracks_to_trajectories(track_result.output.get("track_history", {}))

    try:
        result_path = generate_trajectory_plot(
            positions=positions,
            arena_info=arena_info,
            experiment_type=experiment_type,
            video_size=video_size,
            output_path=output_path,
            skeletons=skeletons,
            tracks=trajectories_for_viz,
        )
        return ToolResult(
            tool_name="generate_trajectory_plot",
            success=True,
            output={"image_path": result_path}
        )
    except Exception as e:
        print(f"[visualize] 轨迹图生成失败: {e}")
        return ToolResult(
            tool_name="generate_trajectory_plot",
            success=False, output=None, error=str(e)
        )


def _execute_heatmap(state: ExperimentState) -> ToolResult:
    """生成热力图"""
    from ...tools.visualize import generate_heatmap

    positions, arena_info, experiment_type, video_size, _ = _extract_viz_data(state)

    if positions is None:
        print(f"[visualize] 热力图: 数据不足")
        return ToolResult(
            tool_name="generate_heatmap",
            success=False, output=None,
            error="数据不足，无法生成热力图"
        )

    print(f"[visualize] 热力图: {len(positions)}个点")

    output_path = _get_viz_output_path(state, "heatmap.png")

    try:
        result_path = generate_heatmap(
            positions=positions,
            arena_info=arena_info,
            experiment_type=experiment_type,
            video_size=video_size,
            output_path=output_path,
        )
        return ToolResult(
            tool_name="generate_heatmap",
            success=True,
            output={"image_path": result_path}
        )
    except Exception as e:
        return ToolResult(
            tool_name="generate_heatmap",
            success=False, output=None, error=str(e)
        )


def _execute_generate_report(state: ExperimentState) -> ToolResult:
    """生成行为学分析报告"""
    from ...tools.report import generate_behavior_report, REPORT_SECTIONS

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
    visualization_paths: dict[str, str | None] = {"trajectory_plot": None, "heatmap": None}
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

    # 获取用户选择的板块（从 state.constraints 中读取，默认为全部）
    sections = state.constraints.get("report_sections", list(REPORT_SECTIONS.keys()))

    # 生成报告
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
        output_dir = os.path.join(os.path.dirname(state.video_path), f"{os.path.splitext(os.path.basename(state.video_path))[0]}_output")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "behavior_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        return ToolResult(
            tool_name="generate_report",
            success=True,
            output={
                "report_path": report_path,
                "report_content": report_md,
            }
        )
    except Exception as e:
        return ToolResult(
            tool_name="generate_report",
            success=False,
            output=None,
            error=f"报告生成失败: {str(e)}"
        )
