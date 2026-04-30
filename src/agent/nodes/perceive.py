"""
感知节点
负责视频理解和元数据提取
"""

from ..state import ExperimentState, VideoMetadata
from typing import Any
import cv2
import json
import re
import os


def perceive_node(state: ExperimentState) -> dict[str, Any]:
    """
    感知节点：提取视频元数据并理解实验上下文

    Args:
        state: 当前实验状态

    Returns:
        状态更新字典
    """
    updates = {}

    # 提取视频元数据
    video_metadata = _extract_video_metadata(state.video_path)
    updates["video_metadata"] = video_metadata

    # 使用 LLM 解析用户请求
    parsed = _parse_user_request_with_llm(
        state.user_request,
        video_metadata
    )
    updates.update(parsed)

    # 根据视频特征识别潜在问题
    if video_metadata.brightness and video_metadata.brightness < 50:
        updates.setdefault("constraints", {})["low_light"] = True

    return updates


def _extract_video_metadata(video_path: str) -> VideoMetadata:
    """
    提取视频元数据

    Args:
        video_path: 视频文件路径

    Returns:
        VideoMetadata 对象
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 基本属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    # 计算亮度和对比度（采样）
    brightness, contrast = _calculate_video_statistics(cap)

    cap.release()

    return VideoMetadata(
        path=video_path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        brightness=brightness,
        contrast=contrast
    )


def _calculate_video_statistics(cap: cv2.VideoCapture, sample_frames: int = 100) -> tuple[float, float]:
    """
    计算视频统计信息

    Args:
        cap: OpenCV VideoCapture 对象
        sample_frames: 采样帧数

    Returns:
        (brightness, contrast) 元组
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // sample_frames)

    brightness_values = []
    contrast_values = []

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(gray.mean())
        contrast_values.append(gray.std())

    avg_brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 0
    avg_contrast = sum(contrast_values) / len(contrast_values) if contrast_values else 0

    return avg_brightness, avg_contrast


def _parse_user_request_with_llm(request: str, video_metadata: VideoMetadata) -> dict[str, Any]:
    """
    使用 LLM 解析用户请求

    Args:
        request: 用户自然语言请求
        video_metadata: 视频元数据

    Returns:
        解析结果字典
    """
    # 尝试使用 LLM
    try:
        from ..prompts import PERCEIVE_SYSTEM_PROMPT, PERCEIVE_USER_PROMPT
        from src.llm import get_llm_client

        llm = get_llm_client()
        print(f"[perceive] 使用 provider: {llm.config.provider.value}, model: {llm.config.model}")

        user_prompt = PERCEIVE_USER_PROMPT.format(
            user_request=request,
            duration=video_metadata.duration,
            fps=video_metadata.fps,
            width=video_metadata.width,
            height=video_metadata.height,
            brightness=video_metadata.brightness or 0
        )

        # 使用同步方法调用 LLM
        response = llm.chat_sync([
            {"role": "system", "content": PERCEIVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], timeout=30)

        # 解析 JSON 响应
        parsed = _extract_json_from_response(response)

        if parsed:
            result = {
                "experiment_type": parsed.get("experiment_type"),
                "species": parsed.get("species"),
                "constraints": parsed.get("constraints", {})
            }
            # 确保默认值
            if not result["experiment_type"]:
                result["experiment_type"] = "open_field"
            if not result["species"]:
                result["species"] = "mouse"
            return result

    except Exception as e:
        print(f"[perceive] LLM 解析失败，使用规则引擎: {e}")

    # 回退到规则引擎
    return _parse_user_request_rules(request)


def _extract_json_from_response(response: str) -> dict | None:
    """
    从 LLM 响应中提取 JSON

    Args:
        response: LLM 响应文本

    Returns:
        解析后的字典，失败返回 None
    """
    # 尝试直接解析
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


def _parse_user_request_rules(request: str) -> dict[str, Any]:
    """
    使用规则引擎解析用户请求 (回退方案)

    Args:
        request: 用户自然语言请求

    Returns:
        解析结果字典
    """
    result = {
        "experiment_type": None,
        "species": None,
        "constraints": {}
    }

    request_lower = request.lower()

    # 检测实验类型
    if any(kw in request_lower for kw in ["旷场", "open field", "openfield"]):
        result["experiment_type"] = "open_field"
    elif any(kw in request_lower for kw in ["水迷宫", "morris", "water maze"]):
        result["experiment_type"] = "morris_water_maze"
    elif any(kw in request_lower for kw in ["高架十字", "十字迷宫", "epm", "elevated plus maze", "plus maze"]):
        result["experiment_type"] = "epm"
    elif any(kw in request_lower for kw in ["线虫", "worm", "c.elegans", "秀丽隐杆线虫", "蠕虫"]):
        result["experiment_type"] = "worm_assay"
    elif any(kw in request_lower for kw in ["斑马鱼", "zebrafish", "danio", "孔板"]):
        result["experiment_type"] = "zebrafish_plate"

    # 检测物种
    if any(kw in request_lower for kw in ["小鼠", "mouse", "mice", "c57"]):
        result["species"] = "mouse"
    elif any(kw in request_lower for kw in ["大鼠", "rat"]):
        result["species"] = "rat"
    elif any(kw in request_lower for kw in ["线虫", "worm", "c.elegans", "秀丽隐杆线虫", "蠕虫"]):
        result["species"] = "worm"
    elif any(kw in request_lower for kw in ["斑马鱼", "zebrafish", "danio"]):
        result["species"] = "zebrafish"

    # 检测约束
    if any(kw in request_lower for kw in ["暗", "低光", "low light", "dark"]):
        result["constraints"]["low_light"] = True
    if any(kw in request_lower for kw in ["焦虑", "anxiety"]):
        result["constraints"]["target_behavior"] = "anxiety"

    return result


# 保留旧函数名作为别名
_parse_user_request = _parse_user_request_rules
