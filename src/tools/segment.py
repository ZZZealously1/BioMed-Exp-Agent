"""
SAM 分割工具包装
将 SAM 模型包装为 MCP 工具
"""

from typing import Any
from pydantic import BaseModel, Field
from mcp.types import Tool
import numpy as np
import cv2
from pathlib import Path


class SegmentationResult(BaseModel):
    """分割结果"""
    masks: list[dict[str, Any]]  # [{frame_idx, mask_data, bbox, area}]
    frame_count: int
    mean_iou: float
    quality_score: float
    statistics: dict[str, Any]


class SegmentInput(BaseModel):
    """分割工具输入"""
    frames: list[dict[str, Any]] = Field(description="待分割的帧")
    points: list[dict[str, int]] | None = Field(default=None, description="点提示 [{x, y, label}]")
    boxes: list[dict[str, int]] | None = Field(default=None, description="框提示 [{x1, y1, x2, y2}]")
    model_type: str = Field(default="sam2.1_t", description="SAM 模型类型")


# MCP 工具定义
segment_tool = Tool(
    name="segment",
    description="""
    使用 SAM 模型进行图像分割。

    输入: 帧数据 + 提示 (点/框)
    输出: 分割掩码 + 质量指标

    支持的模型:
    - sam2.1_t: SAM 2.1 Tiny (快速，适合视频)
    """,
    inputSchema={
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_idx": {"type": "integer"},
                        "image_path": {"type": "string"}
                    }
                },
                "description": "待分割的帧列表"
            },
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "label": {"type": "integer", "description": "1=前景点, 0=背景点"}
                    }
                },
                "description": "点提示列表 (可选)"
            },
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "integer"},
                        "y1": {"type": "integer"},
                        "x2": {"type": "integer"},
                        "y2": {"type": "integer"}
                    }
                },
                "description": "边界框提示列表 (可选)"
            },
            "model_type": {
                "type": "string",
                "enum": ["sam2.1_t"],
                "default": "sam2.1_t",
                "description": "SAM 模型类型"
            }
        },
        "required": ["frames"]
    }
)


# 全局模型缓存
_sam_model = None
_sam_model_path = Path(__file__).parent.parent.parent / "weights" / "sam" / "ultralytics_sam2.1_t.pt"


def load_sam_model():
    """加载 SAM 模型 (Ultralytics)"""
    global _sam_model
    if _sam_model is not None:
        return _sam_model

    if not _sam_model_path.exists():
        raise FileNotFoundError(f"SAM 模型权重不存在: {_sam_model_path}")

    from ultralytics import SAM
    _sam_model = SAM(str(_sam_model_path))
    return _sam_model


def segment_frame(frame: np.ndarray, bboxes: list[list[float]]) -> tuple[list[np.ndarray], list[float]]:
    """
    对单帧进行 SAM 分割

    Args:
        frame: BGR 图像 (H, W, 3)
        bboxes: 检测框列表 [[x1, y1, x2, y2], ...]

    Returns:
        (masks_list, scores_list)
    """
    if not bboxes:
        return [], []

    model = load_sam_model()
    h, w = frame.shape[:2]

    # Ultralytics SAM 支持 bboxes 参数
    # bboxes 需要是 [[x1, y1, x2, y2], ...] 格式
    results = model(frame, bboxes=bboxes, verbose=False)

    masks = []
    scores = []

    if results and len(results) > 0:
        result = results[0]
        if result.masks is not None:
            for mask_tensor in result.masks.data:
                mask = mask_tensor.cpu().numpy()
                # 转为 bool 类型掩码
                mask = (mask > 0.5).astype(np.uint8)
                # SAM 可能返回 resize 后的 mask，需要还原到原图尺寸
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                masks.append(mask)
            # 使用 mask 与 bbox 的面积重叠度作为 proxy IoU
            for i, bbox in enumerate(bboxes):
                if i < len(masks):
                    mask_area = masks[i].sum()
                    bbox_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
                    iou = min(mask_area / bbox_area, 1.0)
                    scores.append(iou)

    return masks, scores


def segment_video(video_path: str, boxes_by_frame: dict[int, list[dict]], skip_frames: int = 1) -> dict[str, Any]:
    """
    对视频逐帧进行 SAM 分割

    Args:
        video_path: 视频路径
        boxes_by_frame: {frame_idx: [box_dict, ...]}
        skip_frames: 跳帧间隔 (1=不跳帧, 3=每3帧处理1帧)

    Returns:
        {
            "masks": [{frame_idx, mask_data, bbox, area}, ...],
            "frame_count": int,
            "mean_iou": float,
            "quality_score": float,
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_masks = []
    total_iou = 0.0
    iou_count = 0
    processed_frames = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = boxes_by_frame.get(frame_idx, [])
        if boxes and frame_idx % skip_frames == 0:
            bboxes = [[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]] for b in boxes]
            masks, scores = segment_frame(frame, bboxes)

            for i, (mask, score) in enumerate(zip(masks, scores)):
                all_masks.append({
                    "frame_idx": frame_idx,
                    "mask_data": mask,
                    "bbox": boxes[i],
                    "area": int(mask.sum()),
                    "iou": round(score, 4)
                })
                total_iou += score
                iou_count += 1
            processed_frames += 1

            if processed_frames % 10 == 0:
                print(f"[segment] 已处理 {processed_frames} 帧 (当前 frame_idx={frame_idx}), 累计 masks={len(all_masks)}")

        frame_idx += 1

    cap.release()
    print(f"[segment] 完成. 总帧数={frame_count}, 处理帧数={processed_frames}, 总 masks={len(all_masks)}, mean_iou={total_iou / iou_count if iou_count > 0 else 0:.3f}")

    mean_iou = total_iou / iou_count if iou_count > 0 else 0.0
    quality_score = _calculate_quality_score(mean_iou)

    return {
        "masks": all_masks,
        "frame_count": frame_count,
        "mean_iou": round(mean_iou, 4),
        "quality_score": round(quality_score, 4),
        "statistics": {
            "total_masks": len(all_masks),
            "mean_mask_area": float(np.mean([m["area"] for m in all_masks])) if all_masks else 0.0,
            "mean_iou": round(mean_iou, 4)
        }
    }


async def segment_handler(arguments: dict) -> SegmentationResult:
    """
    分割工具处理器 (MCP 接口)

    Args:
        arguments: 工具参数

    Returns:
        SegmentationResult 对象
    """
    input_data = SegmentInput(**arguments)

    all_masks = []
    total_iou = 0.0
    model = load_sam_model()

    for frame_info in input_data.frames:
        image_path = frame_info.get("image_path")
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        bboxes = []
        if input_data.boxes:
            bboxes = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in input_data.boxes]

        masks, scores = segment_frame(frame, bboxes)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            box = input_data.boxes[i] if input_data.boxes and i < len(input_data.boxes) else {}
            all_masks.append({
                "frame_idx": frame_info.get("frame_idx", 0),
                "mask_data": mask,
                "bbox": box,
                "area": int(mask.sum())
            })
            total_iou += score

    mean_iou = total_iou / len(all_masks) if all_masks else 0
    quality_score = _calculate_quality_score(mean_iou)

    statistics = {
        "total_masks": len(all_masks),
        "mean_mask_area": float(np.mean([m["area"] for m in all_masks])) if all_masks else 0.0,
        "mean_iou": round(mean_iou, 4)
    }

    return SegmentationResult(
        masks=all_masks,
        frame_count=len(input_data.frames),
        mean_iou=round(mean_iou, 4),
        quality_score=round(quality_score, 4),
        statistics=statistics
    )


def _calculate_quality_score(mean_iou: float) -> float:
    """计算质量评分"""
    if mean_iou >= 0.9:
        return 1.0
    elif mean_iou >= 0.85:
        return 0.95
    elif mean_iou >= 0.8:
        return 0.9
    elif mean_iou >= 0.7:
        return 0.8
    else:
        return mean_iou
