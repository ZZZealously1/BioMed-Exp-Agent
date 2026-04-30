"""
YOLO 检测工具包装
将 YOLO 模型包装为 MCP 工具
"""

from typing import Any
from pydantic import BaseModel, Field
from mcp.types import Tool
import cv2
import numpy as np
from pathlib import Path
import torch

# 全局模型缓存
_model = None
_model_path = Path(__file__).parent.parent.parent / "weights" / "YOLO26" / "best.pt"
_SPECIES_MODEL_MAP = {
    "worm": Path(__file__).parent.parent.parent / "weights" / "YOLO26 for worm" / "best.pt",
    "zebrafish": Path(__file__).parent.parent.parent / "weights" / "YOLO26 for zebrafish" / "best.pt",
}


class DetectionResult(BaseModel):
    """检测结果"""
    boxes: list[dict[str, Any]]  # [{frame_idx, x, y, w, h, confidence, class_id, class_name}]
    frame_count: int
    detection_rate: float
    quality_score: float
    video_info: dict[str, Any]
    model_info: dict[str, Any]


class DetectInput(BaseModel):
    """检测工具输入"""
    video_path: str = Field(description="视频文件路径")
    confidence: float = Field(default=0.5, description="置信度阈值")
    iou_threshold: float = Field(default=0.45, description="IOU 阈值")
    target_classes: list[str] | None = Field(default=None, description="目标类别")
    enhance_low_light: bool = Field(default=False, description="是否增强低光照")
    batch_size: int = Field(default=1, description="批处理大小")
    skip_frames: int = Field(default=1, description="跳帧间隔 (1=不跳帧)")
    single_object: bool = Field(default=True, description="单目标模式 (只保留置信度最高的检测框)")
    temporal_smooth: bool = Field(default=True, description="是否启用时序平滑 (利用前后帧连续性)")
    max_displacement: float = Field(default=50.0, description="最大允许位移 (像素)，超过此值的检测框视为跳变")


# MCP 工具定义
detect_tool = Tool(
    name="detect",
    description="""
使用 YOLO 模型进行目标检测。

输入: 视频文件路径
输出: 检测框列表 + 质量指标

质量指标:
- detection_rate: 检测到目标的帧数占比
- quality_score: 整体检测质量评分
""",
    inputSchema={
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "视频文件路径"
            },
            "confidence": {
                "type": "number",
                "default": 0.5,
                "description": "置信度阈值 (0-1)"
            },
            "iou_threshold": {
                "type": "number",
                "default": 0.45,
                "description": "NMS IOU 阈值 (0-1)"
            },
            "target_classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "目标类别列表 (可选)"
            },
            "enhance_low_light": {
                "type": "boolean",
                "default": False,
                "description": "是否对低光照视频进行增强"
            },
            "batch_size": {
                "type": "integer",
                "default": 1,
                "description": "批处理大小"
            },
            "skip_frames": {
                "type": "integer",
                "default": 1,
                "description": "跳帧间隔 (1=不跳帧)"
            },
            "single_object": {
                "type": "boolean",
                "default": True,
                "description": "单目标模式 (只保留置信度最高的检测框)"
            }
        },
        "required": ["video_path"]
    }
)


def load_model(model_path: Path | None = None, species: str | None = None):
    """
    加载 YOLO 模型

    Args:
        model_path: 模型权重路径，默认使用内置权重
        species: 物种名称，用于自动路由到对应模型权重

    Returns:
        YOLO 模型实例
    """
    global _model

    if _model is not None and model_path is None and species is None:
        return _model

    if model_path is not None:
        path = model_path
    elif species is not None:
        path = _SPECIES_MODEL_MAP.get(species, _model_path)
    else:
        path = _model_path

    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在: {path}")

    # 使用 ultralytics 加载模型
    from ultralytics import YOLO
    # 当显式指定 model_path 或 species 时，加载特定模型而不缓存到全局
    if model_path is not None or species is not None:
        return YOLO(str(path))

    _model = YOLO(str(path))
    return _model


class TemporalSmoother:
    """
    时序平滑器

    利用前后帧的连续性来提高检测准确性：
    1. 维护最近 N 帧的检测框历史
    2. 预测当前帧的期望位置
    3. 对于多个候选检测框，综合考虑置信度和位置连续性
    4. 过滤异常尺寸的检测框 (统一优化)
    """

    def __init__(
        self,
        history_size: int = 5,
        max_displacement: float = 50.0,
        size_variation_threshold: float = 2.0,
        adapt_size: bool = True,
        expected_area_range: tuple[float, float] | None = None,
        prefer_smaller: bool = True
    ):
        """
        初始化平滑器

        Args:
            history_size: 历史帧数
            max_displacement: 最大允许位移 (像素)
            size_variation_threshold: 尺寸变化阈值 (相对于历史平均尺寸的倍数)
            adapt_size: 是否自适应学习参考尺寸
            expected_area_range: 预期面积范围 (min, max)，用于过滤异常大小
            prefer_smaller: 当有多个检测时，是否优先选择较小的 (适合动物检测，过滤静态结构)
        """
        self.history_size = history_size
        self.max_displacement = max_displacement
        self.size_variation_threshold = size_variation_threshold
        self.adapt_size = adapt_size
        self.expected_area_range = expected_area_range
        self.prefer_smaller = prefer_smaller
        self.history: list[dict] = []  # 最近 N 帧的检测框
        self.reference_area: float | None = None  # 参考面积
        self._initial_frames_processed = 0  # 用于智能初始化

    def update(self, boxes: list[dict], frame_idx: int) -> list[dict]:
        """
        更新历史并选择最佳检测框

        Args:
            boxes: 当前帧的候选检测框列表
            frame_idx: 当前帧索引

        Returns:
            选择后的检测框列表
        """
        if not boxes:
            return []

        # 预过滤：基于预期面积范围
        if self.expected_area_range:
            min_area, max_area = self.expected_area_range
            filtered = [b for b in boxes if min_area <= b["w"] * b["h"] <= max_area]
            if filtered:
                boxes = filtered

        # 初始化阶段：选择最小面积的检测框 (通常是动物，而非静态结构)
        if not self.history and self.prefer_smaller and len(boxes) > 1:
            boxes = sorted(boxes, key=lambda b: b["w"] * b["h"])
            boxes = [boxes[0]]

        # 面积过滤 (基于历史参考尺寸) - 强制过滤
        if self.reference_area and len(boxes) > 1:
            filtered_boxes = self._filter_by_size(boxes, frame_idx)
            if filtered_boxes:
                boxes = filtered_boxes
            else:
                # 过滤后为空，选择面积最接近参考值的
                boxes = [min(boxes, key=lambda b: abs(b["w"] * b["h"] - self.reference_area))]

        if len(boxes) == 1:
            self._add_to_history(boxes[0])
            return boxes

        # 多个检测框：选择最佳的
        best_box = self._select_best_box(boxes)
        self._add_to_history(best_box)
        return [best_box]

    def _filter_by_size(self, boxes: list[dict], frame_idx: int) -> list[dict]:
        """根据尺寸一致性过滤检测框"""
        if not self.reference_area:
            return boxes

        filtered = []
        for box in boxes:
            area = box["w"] * box["h"]
            ratio = area / self.reference_area if self.reference_area > 0 else 1

            if 1 / self.size_variation_threshold <= ratio <= self.size_variation_threshold:
                filtered.append(box)

        return filtered

    def _select_best_box(self, boxes: list[dict]) -> dict:
        """选择最佳检测框"""
        if not self.history:
            # 没有历史，选择置信度最高的 (或面积最小的，如果 prefer_smaller)
            if self.prefer_smaller:
                return min(boxes, key=lambda b: b["w"] * b["h"])
            return max(boxes, key=lambda b: b["confidence"])

        predicted = self._predict_position()

        scored_boxes = []
        for box in boxes:
            score = self._calculate_box_score(box, predicted)
            scored_boxes.append((box, score))

        scored_boxes.sort(key=lambda x: x[1], reverse=True)
        return scored_boxes[0][0]

    def _predict_position(self) -> dict:
        """
        基于历史预测当前位置

        使用简单的线性预测：位置 = 上一帧位置 + 平均速度
        """
        if len(self.history) < 2:
            # 历史不足，返回最后一帧的位置
            if self.history:
                last = self.history[-1]
                return {"cx": last["cx"], "cy": last["cy"], "w": last["w"], "h": last["h"]}
            return None

        # 计算平均速度
        velocities = []
        for i in range(1, len(self.history)):
            prev = self.history[i - 1]
            curr = self.history[i]
            frame_gap = curr["frame_idx"] - prev["frame_idx"]
            if frame_gap > 0:
                vx = (curr["cx"] - prev["cx"]) / frame_gap
                vy = (curr["cy"] - prev["cy"]) / frame_gap
                velocities.append((vx, vy))

        if velocities:
            avg_vx = np.mean([v[0] for v in velocities])
            avg_vy = np.mean([v[1] for v in velocities])
        else:
            avg_vx, avg_vy = 0, 0

        # 预测位置
        last = self.history[-1]
        frame_gap = 1  # 假设相邻帧

        return {
            "cx": last["cx"] + avg_vx * frame_gap,
            "cy": last["cy"] + avg_vy * frame_gap,
            "w": last["w"],
            "h": last["h"]
        }

    def _calculate_box_score(self, box: dict, predicted: dict | None) -> float:
        """
        计算检测框的综合得分

        Args:
            box: 候选检测框
            predicted: 预测位置

        Returns:
            综合得分 (0-1)
        """
        # 置信度得分 (权重 0.4)
        confidence_score = box["confidence"]

        if not predicted:
            return confidence_score

        # 计算中心点
        box_cx = box["x"] + box["w"] / 2
        box_cy = box["y"] + box["h"] / 2

        # 位置偏差得分 (权重 0.4)
        distance = np.sqrt((box_cx - predicted["cx"])**2 + (box_cy - predicted["cy"])**2)
        # 使用指数衰减：距离越小得分越高
        position_score = np.exp(-distance / self.max_displacement)

        # 尺寸一致性得分 (权重 0.2)
        size_diff = abs(box["w"] - predicted["w"]) + abs(box["h"] - predicted["h"])
        avg_size = (predicted["w"] + predicted["h"]) / 2
        size_score = max(0, 1 - size_diff / (avg_size + 1e-6))

        # 综合得分
        total_score = (
            0.4 * confidence_score +
            0.4 * position_score +
            0.2 * size_score
        )

        return total_score

    def _add_to_history(self, box: dict):
        """添加到历史记录并更新参考尺寸"""
        history_entry = {
            "frame_idx": box["frame_idx"],
            "cx": box["x"] + box["w"] / 2,
            "cy": box["y"] + box["h"] / 2,
            "w": box["w"],
            "h": box["h"]
        }
        self.history.append(history_entry)

        # 保持历史长度
        if len(self.history) > self.history_size:
            self.history.pop(0)

        # 自适应更新参考尺寸
        if self.adapt_size:
            current_area = box["w"] * box["h"]
            if self.reference_area is None:
                self.reference_area = current_area
            else:
                # 指数移动平均，平滑更新
                alpha = 0.1  # 平滑因子
                self.reference_area = (1 - alpha) * self.reference_area + alpha * current_area


async def detect_handler(arguments: dict) -> DetectionResult:
    """
    检测工具处理器

    Args:
        arguments: 工具参数

    Returns:
        DetectionResult 对象
    """
    input_data = DetectInput(**arguments)

    # 加载视频
    cap = cv2.VideoCapture(input_data.video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_data.video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 加载 YOLO 模型
    model = load_model()

    # 获取类别名称映射
    class_names = model.names if hasattr(model, 'names') else {}

    # 初始化时序平滑器
    smoother = TemporalSmoother(
        history_size=5,
        max_displacement=input_data.max_displacement
    ) if input_data.temporal_smooth else None

    all_boxes = []
    frames_with_detection = 0

    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 跳帧处理
        if frame_idx % input_data.skip_frames != 0:
            frame_idx += 1
            continue

        # 低光照增强
        if input_data.enhance_low_light:
            frame = enhance_frame(frame)

        # YOLO 推理
        results = model(
            frame,
            conf=input_data.confidence,
            iou=input_data.iou_threshold,
            verbose=False
        )

        # 解析检测结果 (不在此处处理单目标模式)
        boxes = parse_results(
            results,
            frame_idx,
            class_names,
            input_data.target_classes,
            single_object=False  # 先获取所有候选框
        )

        # 单目标 + 时序平滑处理
        if input_data.single_object and boxes:
            if smoother:
                boxes = smoother.update(boxes, frame_idx)
            elif len(boxes) > 1:
                # 仅使用置信度过滤
                boxes = [max(boxes, key=lambda b: b["confidence"])]

        all_boxes.extend(boxes)

        if boxes:
            frames_with_detection += 1

        frame_idx += 1
        processed_frames += 1

    cap.release()

    # 计算质量指标
    detection_rate = frames_with_detection / processed_frames if processed_frames > 0 else 0
    quality_score = _calculate_quality_score(detection_rate)

    return DetectionResult(
        boxes=all_boxes,
        frame_count=frame_count,
        detection_rate=round(detection_rate, 4),
        quality_score=round(quality_score, 4),
        video_info={
            "fps": fps,
            "width": width,
            "height": height,
            "duration": round(frame_count / fps, 2) if fps > 0 else 0
        },
        model_info={
            "model_path": str(_model_path),
            "model_type": "YOLO26 (custom trained)",
            "num_classes": len(class_names),
            "class_names": class_names
        }
    )


def parse_results(
    results,
    frame_idx: int,
    class_names: dict,
    target_classes: list[str] | None = None,
    single_object: bool = True
) -> list[dict]:
    """
    解析 YOLO 推理结果

    Args:
        results: YOLO 推理结果
        frame_idx: 当前帧索引
        class_names: 类别 ID 到名称的映射
        target_classes: 目标类别过滤列表
        single_object: 单目标模式，只保留置信度最高的检测框

    Returns:
        检测框列表
    """
    boxes = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            # 获取边界框坐标 (xyxy 格式)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = class_names.get(class_id, f"class_{class_id}")

            # 类别过滤
            if target_classes and class_name not in target_classes:
                continue

            boxes.append({
                "frame_idx": frame_idx,
                "x": float(x1),
                "y": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "confidence": round(confidence, 4),
                "class_id": class_id,
                "class_name": class_name
            })

    # 单目标模式：只保留置信度最高的检测框
    if single_object and len(boxes) > 1:
        boxes = [max(boxes, key=lambda b: b["confidence"])]

    return boxes


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    增强低光照帧

    Args:
        frame: 输入帧

    Returns:
        增强后的帧
    """
    # CLAHE 增强
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _calculate_quality_score(detection_rate: float) -> float:
    """计算质量评分"""
    if detection_rate >= 0.95:
        return 1.0
    elif detection_rate >= 0.9:
        return 0.95
    elif detection_rate >= 0.8:
        return 0.85
    elif detection_rate >= 0.7:
        return 0.7
    else:
        return detection_rate


# 便捷函数
def detect_video(
    video_path: str,
    confidence: float = 0.5,
    **kwargs
) -> DetectionResult:
    """
    同步检测接口

    Args:
        video_path: 视频路径
        confidence: 置信度阈值
        **kwargs: 其他参数

    Returns:
        DetectionResult
    """
    import asyncio
    return asyncio.run(detect_handler({
        "video_path": video_path,
        "confidence": confidence,
        **kwargs
    }))


def detect_mouse_by_background_subtraction(
    video_path: str,
    confidence_threshold: float = 0.3,
    min_area: int = 50,
    max_area: int = 5000,
    diff_threshold: int = 15,
    smoothing_window: int = 5,
    max_displacement: float = 80.0,
    roi: dict | None = None,
) -> DetectionResult:
    """
    使用背景减除法检测水迷宫中的小鼠（优化版）

    适用于：水池固定、小鼠为唯一运动目标的场景。
    通过中值帧建立背景模型，帧差分提取前景运动目标，并添加时间平滑。

    Args:
        video_path: 视频路径
        confidence_threshold: 前景打分阈值 (0-1)
        min_area: 最小检测面积 (像素)
        max_area: 最大检测面积 (像素)
        diff_threshold: 帧差分阈值 (0-255)
        smoothing_window: 滑动平均窗口大小
        max_displacement: 两帧间最大允许位移 (像素)，超过则视为噪声并插值

    Returns:
        DetectionResult
    """
    # 1. 计算背景（中值帧）
    cap = cv2.VideoCapture(video_path)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(
        0, min(frame_count_total - 1, 99),
        min(frame_count_total, 100), dtype=int
    )
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    if not frames:
        return DetectionResult(
            boxes=[], frame_count=0, detection_rate=0.0,
            quality_score=0.0, video_info={}, model_info={"method": "background_subtraction"}
        )

    background = np.median(np.array(frames), axis=0).astype(np.uint8)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # 2. 逐帧处理
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_boxes = []          # 原始检测结果
    all_boxes = []          # 平滑后结果
    frames_with_detection = 0
    frame_idx = 0

    # 形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 水池中心与半径
    pool_cx, pool_cy = width / 2, height / 2
    pool_radius = min(width, height) * 0.42

    # 上一帧有效位置（用于噪声过滤和插值）
    prev_cx, prev_cy = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. 背景差分（增大模糊核降噪）
        diff = cv2.absdiff(gray, bg_gray)
        blurred = cv2.GaussianBlur(diff, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, diff_threshold, 255, cv2.THRESH_BINARY)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # 4. 找连通区域
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w / 2, y + h / 2

            # ROI 过滤（如果指定了场地范围）
            if roi:
                if not (roi["x_min"] <= cx <= roi["x_max"] and roi["y_min"] <= cy <= roi["y_max"]):
                    continue

            # 距离水池中心过滤
            dist_from_center = np.sqrt((cx - pool_cx) ** 2 + (cy - pool_cy) ** 2)
            if dist_from_center > pool_radius * 1.1:
                continue

            # 长宽比过滤
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > 3.0:
                continue

            # 打分
            centrality = max(0, 1 - dist_from_center / pool_radius)
            area_score = min(area / 300, 1.0)
            score = centrality * 0.5 + area_score * 0.5

            if score > best_score:
                best_score = score
                best_box = {
                    "frame_idx": frame_idx,
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "cx": float(cx),
                    "cy": float(cy),
                    "confidence": float(score),
                    "class_id": 0,
                    "class_name": "mouse",
                }

        # 5. 噪声过滤：位移过大时丢弃并使用预测位置
        if best_box and best_score >= confidence_threshold:
            cx, cy = best_box["cx"], best_box["cy"]
            if prev_cx is not None:
                displacement = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                if displacement > max_displacement:
                    # 视为噪声，使用线性预测位置
                    predicted_cx = prev_cx + (prev_cx - prev_prev_cx) if 'prev_prev_cx' in dir() else prev_cx
                    predicted_cy = prev_cy + (prev_cy - prev_prev_cy) if 'prev_prev_cy' in dir() else prev_cy
                    # 简化为使用上一帧位置
                    best_box["cx"] = prev_cx
                    best_box["cy"] = prev_cy
                    best_box["x"] = prev_cx - best_box["w"] / 2
                    best_box["y"] = prev_cy - best_box["h"] / 2
                    cx, cy = prev_cx, prev_cy

            prev_cx, prev_cy = cx, cy
            raw_boxes.append(best_box)
        else:
            # 未检测到，使用上一帧位置（保持连续性）
            if prev_cx is not None:
                raw_boxes.append({
                    "frame_idx": frame_idx,
                    "x": prev_cx - 20,
                    "y": prev_cy - 20,
                    "w": 40.0,
                    "h": 40.0,
                    "cx": prev_cx,
                    "cy": prev_cy,
                    "confidence": 0.1,
                    "class_id": 0,
                    "class_name": "mouse",
                })

        frame_idx += 1

    cap.release()

    # 6. 时间平滑：滑动平均
    if raw_boxes:
        # 补齐缺失帧（确保 frame_idx 连续）
        filled_boxes = []
        box_map = {b["frame_idx"]: b for b in raw_boxes}
        last_valid = raw_boxes[0]
        for i in range(frame_count):
            if i in box_map:
                last_valid = box_map[i]
                filled_boxes.append(box_map[i].copy())
            else:
                filled_boxes.append(last_valid.copy())
                filled_boxes[-1]["frame_idx"] = i

        # 滑动平均平滑
        window = smoothing_window
        half = window // 2
        for i in range(len(filled_boxes)):
            start = max(0, i - half)
            end = min(len(filled_boxes), i + half + 1)
            avg_cx = sum(filled_boxes[j]["cx"] for j in range(start, end)) / (end - start)
            avg_cy = sum(filled_boxes[j]["cy"] for j in range(start, end)) / (end - start)
            w = filled_boxes[i]["w"]
            h = filled_boxes[i]["h"]
            filled_boxes[i]["x"] = avg_cx - w / 2
            filled_boxes[i]["y"] = avg_cy - h / 2
            filled_boxes[i]["cx"] = avg_cx
            filled_boxes[i]["cy"] = avg_cy

        # 移除 cx/cy 临时字段，输出标准格式
        for b in filled_boxes:
            b.pop("cx", None)
            b.pop("cy", None)
            all_boxes.append(b)

    frames_with_detection = len(all_boxes)
    detection_rate = frames_with_detection / frame_count if frame_count > 0 else 0
    quality_score = _calculate_quality_score(detection_rate)

    return DetectionResult(
        boxes=all_boxes,
        frame_count=frame_count,
        detection_rate=round(detection_rate, 4),
        quality_score=round(quality_score, 4),
        video_info={"fps": fps, "width": width, "height": height, "frame_count": frame_count},
        model_info={
            "method": "background_subtraction_v2",
            "diff_threshold": diff_threshold,
            "min_area": min_area,
            "max_area": max_area,
            "smoothing_window": smoothing_window,
            "max_displacement": max_displacement,
        },
    )


def detect_mwm_mouse(
    video_path: str,
    pool_center: tuple[float, float] = (280.0, 280.0),
    pool_axes: tuple[float, float] = (270.0, 280.0),
    platform_center: tuple[float, float] = (180.0, 200.0),
    platform_radius: float = 20.0,
    gray_threshold: int = 90,
    min_area: int = 15,
    max_area: int = 3000,
) -> DetectionResult:
    """
    水迷宫小鼠检测（颜色阈值法）

    针对 Morris Water Maze 场景优化：在椭圆场地内通过灰度阈值找黑色小鼠，
    跳过通用 YOLO 检测，直接利用小鼠与水池的颜色对比度进行定位。

    Args:
        video_path: 视频路径
        pool_center: 椭圆水池中心 (cx, cy)
        pool_axes: 椭圆半轴 (a, b)
        platform_center: 平台中心 (x, y)
        platform_radius: 平台半径
        gray_threshold: 灰度阈值，低于此值为前景（小鼠）
        min_area: 最小检测面积
        max_area: 最大检测面积

    Returns:
        DetectionResult
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pool_cx, pool_cy = pool_center
    pool_a, pool_b = pool_axes
    plat_x, plat_y = platform_center

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    all_boxes = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 椭圆掩码：只在场地内检测
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (int(pool_cx), int(pool_cy)), (int(pool_a), int(pool_b)), 0, 0, 360, 255, -1)

        # 灰度阈值：找黑色区域（小鼠）
        roi = cv2.bitwise_and(gray, gray, mask=mask)
        _, thresh = cv2.threshold(roi, gray_threshold, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w / 2, y + h / 2

            # 椭圆内检查
            if ((cx - pool_cx) / pool_a) ** 2 + ((cy - pool_cy) / pool_b) ** 2 > 1.0:
                continue

            # 打分：面积适中 + 距中心适中
            dist_from_center = np.sqrt((cx - pool_cx) ** 2 + (cy - pool_cy) ** 2)
            score = area * max(0.1, 1 - dist_from_center / 400)

            if score > best_score:
                best_score = score
                best_box = {
                    "frame_idx": frame_idx,
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "confidence": float(min(score / 500, 1.0)),
                    "class_id": 0,
                    "class_name": "mouse",
                }

        if best_box:
            all_boxes.append(best_box)

        frame_idx += 1

    cap.release()

    detection_rate = len(all_boxes) / frame_count if frame_count > 0 else 0
    quality_score = _calculate_quality_score(detection_rate)

    return DetectionResult(
        boxes=all_boxes,
        frame_count=frame_count,
        detection_rate=round(detection_rate, 4),
        quality_score=round(quality_score, 4),
        video_info={"fps": fps, "width": width, "height": height, "frame_count": frame_count},
        model_info={
            "method": "color_threshold",
            "gray_threshold": gray_threshold,
            "pool_center": pool_center,
            "pool_axes": pool_axes,
            "platform_center": platform_center,
        },
    )


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path>")
        sys.exit(1)

    result = detect_video(sys.argv[1], confidence=0.5)
    print(f"检测完成:")
    print(f"  - 总帧数: {result.frame_count}")
    print(f"  - 检测率: {result.detection_rate:.2%}")
    print(f"  - 质量分: {result.quality_score:.2f}")
    print(f"  - 检测框数: {len(result.boxes)}")
