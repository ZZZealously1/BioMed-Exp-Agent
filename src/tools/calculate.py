"""
指标计算工具包装
计算实验分析指标
"""

from typing import Any
from pydantic import BaseModel, Field
from mcp.types import Tool
import numpy as np
from dataclasses import dataclass


# ============== 数据结构 ==============

class MetricsResult(BaseModel):
    """指标计算结果"""
    metrics: dict[str, float]
    experiment_type: str
    quality_score: float
    interpretation: dict[str, str]
    arena_info: dict[str, Any] = {}


class CalculateInput(BaseModel):
    """计算工具输入"""
    trajectories: list[dict[str, Any]] = Field(description="轨迹数据")
    experiment_type: str = Field(description="实验类型")
    arena_config: dict[str, Any] | None = Field(default=None, description="实验场地配置")
    species: str = Field(default="mouse", description="物种")
    fps: float = Field(default=25.0, description="视频帧率")
    skeletons: list[dict[str, Any]] | None = Field(default=None, description="线虫骨架数据 (用于 worm_assay)")


@dataclass
class Position:
    """位置数据"""
    frame_idx: int
    x: float
    y: float
    w: float = 0
    h: float = 0


# MCP 工具定义
calculate_tool = Tool(
    name="calculate",
    description="""
计算实验分析指标。

支持的实验类型:
- open_field: 旷场实验 (中心时间、移动距离、速度、进入次数等)
- morris_water_maze: 水迷宫实验 (逃逸潜伏期、路径效率等)
- epm: 高架十字迷宫实验 (开臂时间、开臂进入次数、头探次数等)
- worm_assay: 线虫行为分析 (运动速度、身体弯曲频率、omega turn 次数等)
- zebrafish_plate: 斑马鱼孔板实验 (总移动距离、平均速度、静止时间占比、边缘时间占比、穿越次数等)

输入: 轨迹数据 + 实验类型
输出: 指标结果 + 解释
""",
    inputSchema={
        "type": "object",
        "properties": {
            "trajectories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "track_id": {"type": "integer"},
                        "positions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "frame_idx": {"type": "integer"},
                                    "w": {"type": "number"},
                                    "h": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                "description": "轨迹数据列表"
            },
            "experiment_type": {
                "type": "string",
                "enum": ["open_field", "morris_water_maze", "epm", "worm_assay", "zebrafish_plate"],
                "description": "实验类型"
            },
            "arena_config": {
                "type": "object",
                "properties": {
                    "width": {"type": "number", "description": "场地宽度 (像素)"},
                    "height": {"type": "number", "description": "场地高度 (像素)"},
                    "center_ratio": {"type": "number", "description": "中心区域占比"},
                    "platform_radius": {"type": "number", "description": "平台半径 (像素)"},
                    "platform_center": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "description": "平台中心坐标"
                    },
                    "arm_width": {"type": "number", "description": "EPM 臂宽度"},
                    "arm_length": {"type": "number", "description": "EPM 臂长度"}
                },
                "description": "实验场地配置"
            },
            "species": {
                "type": "string",
                "default": "mouse",
                "description": "物种"
            },
            "fps": {
                "type": "number",
                "default": 25.0,
                "description": "视频帧率"
            }
        },
        "required": ["trajectories", "experiment_type"]
    }
)


# ============== 处理器 ==============

def calculate_handler(arguments: dict) -> MetricsResult:
    """
    指标计算处理器

    Args:
        arguments: 工具参数

    Returns:
        MetricsResult 对象
    """
    input_data = CalculateInput(**arguments)

    if input_data.experiment_type == "open_field":
        metrics, arena_info = _calculate_open_field_metrics(input_data)
        interpretation = _interpret_open_field(metrics)
    elif input_data.experiment_type == "morris_water_maze":
        metrics, arena_info = _calculate_water_maze_metrics(input_data)
        interpretation = _interpret_water_maze(metrics)
    elif input_data.experiment_type == "epm":
        metrics, arena_info = _calculate_epm_metrics(input_data)
        interpretation = _interpret_epm(metrics)
    elif input_data.experiment_type == "worm_assay":
        metrics, arena_info = _calculate_worm_metrics(input_data)
        interpretation = _interpret_worm(metrics)
    elif input_data.experiment_type == "zebrafish_plate":
        metrics, arena_info = _calculate_zebrafish_metrics(input_data)
        interpretation = _interpret_zebrafish(metrics)
    else:
        metrics = {}
        arena_info = {}
        interpretation = {"error": f"未知实验类型: {input_data.experiment_type}"}

    quality_score = _calculate_quality_score(metrics)

    return MetricsResult(
        metrics=metrics,
        experiment_type=input_data.experiment_type,
        quality_score=quality_score,
        interpretation=interpretation,
        arena_info=arena_info
    )


# ============== 旷场实验指标 ==============

def _calculate_open_field_metrics(input_data: CalculateInput) -> tuple[dict, dict]:
    """
    计算旷场实验指标

    指标包括:
    - center_time_percent: 中心区时间占比
    - center_entries: 进入中心区次数
    - total_distance: 总移动距离
    - avg_speed: 平均速度
    - max_speed: 最大速度
    - immobile_time_percent: 不动时间占比
    - edge_time_percent: 边缘区时间占比
    """
    if not input_data.trajectories:
        return {}, {}

    # 获取主轨迹
    main_trajectory = input_data.trajectories[0]
    raw_positions = main_trajectory.get("positions", [])

    if not raw_positions:
        return {}, {}

    # 转换为 Position 对象
    positions = [
        Position(
            frame_idx=p.get("frame_idx", i),
            x=p["x"],
            y=p["y"],
            w=p.get("w", 0),
            h=p.get("h", 0)
        )
        for i, p in enumerate(raw_positions)
    ]

    # 提取坐标
    x_coords = np.array([p.x for p in positions])
    y_coords = np.array([p.y for p in positions])

    # 获取场地配置
    arena_config = input_data.arena_config or {}

    # 自动检测场地范围（基于轨迹范围 + 边距）
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # 估计场地尺寸（轨迹范围 + 20% 边距）
    trajectory_width = x_max - x_min
    trajectory_height = y_max - y_min
    margin = 0.1  # 10% 边距

    arena_width = arena_config.get("width", trajectory_width / (1 - 2 * margin))
    arena_height = arena_config.get("height", trajectory_height / (1 - 2 * margin))
    center_ratio = arena_config.get("center_ratio", 0.3)

    # 计算场地中心（基于轨迹中心，而非视频中心）
    arena_center_x = arena_config.get("center_x", (x_min + x_max) / 2)
    arena_center_y = arena_config.get("center_y", (y_min + y_max) / 2)

    # 计算中心区域半径
    center_radius = min(arena_width, arena_height) * center_ratio / 2

    # 边缘区域宽度
    edge_width = min(arena_width, arena_height) * 0.15

    arena_info = {
        "width": round(arena_width, 2),
        "height": round(arena_height, 2),
        "center_x": round(arena_center_x, 2),
        "center_y": round(arena_center_y, 2),
        "center_radius": round(center_radius, 2),
        "edge_width": round(edge_width, 2),
        "trajectory_bounds": {
            "x_min": round(x_min, 2),
            "x_max": round(x_max, 2),
            "y_min": round(y_min, 2),
            "y_max": round(y_max, 2)
        }
    }

    # 1. 计算到中心的距离
    distances_from_center = np.sqrt(
        (x_coords - arena_center_x) ** 2 +
        (y_coords - arena_center_y) ** 2
    )

    # 2. 中心区时间占比
    in_center = distances_from_center < center_radius
    center_time_percent = np.sum(in_center) / len(positions) * 100

    # 3. 进入中心区次数
    center_entries = _count_transitions(in_center)

    # 4. 边缘区时间占比（距离中心 > 场地半径 - 边缘宽度）
    arena_radius = min(arena_width, arena_height) / 2
    in_edge = distances_from_center > (arena_radius - edge_width)
    edge_time_percent = np.sum(in_edge) / len(positions) * 100

    # 5. 计算移动距离和速度
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    frame_distances = np.sqrt(dx ** 2 + dy ** 2)

    total_distance = np.sum(frame_distances)
    avg_speed = np.mean(frame_distances) if len(frame_distances) > 0 else 0
    max_speed = np.max(frame_distances) if len(frame_distances) > 0 else 0

    # 6. 不动时间 (速度 < 2 px/frame)
    immobile_threshold = 2.0  # 像素/帧
    immobile_frames = np.sum(frame_distances < immobile_threshold)
    immobile_time_percent = immobile_frames / len(frame_distances) * 100 if len(frame_distances) > 0 else 0

    # 7. 平均到中心距离
    avg_distance_to_center = np.mean(distances_from_center)

    # 8. 轨迹效率 (直线距离 / 实际距离)
    if len(positions) > 1:
        straight_distance = np.sqrt(
            (x_coords[-1] - x_coords[0]) ** 2 +
            (y_coords[-1] - y_coords[0]) ** 2
        )
        path_efficiency = straight_distance / total_distance if total_distance > 0 else 1.0
    else:
        path_efficiency = 1.0

    metrics = {
        # 时间指标
        "center_time_percent": round(center_time_percent, 2),
        "edge_time_percent": round(edge_time_percent, 2),
        "immobile_time_percent": round(immobile_time_percent, 2),

        # 空间指标
        "center_entries": int(center_entries),
        "avg_distance_to_center": round(avg_distance_to_center, 2),

        # 运动指标
        "total_distance": round(total_distance, 2),
        "avg_speed": round(avg_speed, 2),
        "max_speed": round(max_speed, 2),

        # 效率指标
        "path_efficiency": round(path_efficiency, 4)
    }

    # 添加真实单位 (假设 1 像素 = 0.1 cm，可根据实际调整)
    pixel_to_cm = 0.1
    metrics["total_distance_cm"] = round(total_distance * pixel_to_cm, 2)
    metrics["avg_speed_cm_s"] = round(avg_speed * pixel_to_cm * input_data.fps, 2)

    return metrics, arena_info


def _count_transitions(binary_array: np.ndarray) -> int:
    """计算状态转换次数 (从 False 到 True)"""
    if len(binary_array) < 2:
        return 0
    transitions = np.diff(binary_array.astype(int))
    return np.sum(transitions == 1)


# ============== 水迷宫实验指标 ==============

def _calculate_water_maze_metrics(input_data: CalculateInput) -> tuple[dict, dict]:
    """计算水迷宫实验指标"""
    if not input_data.trajectories:
        return {}, {}

    main_trajectory = input_data.trajectories[0]
    raw_positions = main_trajectory.get("positions", [])

    if not raw_positions:
        return {}, {}

    positions = [
        Position(
            frame_idx=p.get("frame_idx", i),
            x=p["x"],
            y=p["y"]
        )
        for i, p in enumerate(raw_positions)
    ]

    x_coords = np.array([p.x for p in positions])
    y_coords = np.array([p.y for p in positions])

    # 获取配置
    arena_config = input_data.arena_config or {}
    pool_diameter = arena_config.get("width", np.max(x_coords) - np.min(x_coords) + 100)
    pool_radius = pool_diameter / 2

    platform_center = arena_config.get("platform_center", {})
    platform_x = platform_center.get("x", pool_radius)
    platform_y = platform_center.get("y", pool_radius)
    platform_radius = arena_config.get("platform_radius", pool_diameter * 0.1)

    pool_center = arena_config.get("pool_center", {})
    pool_center_x = pool_center.get("x", pool_diameter / 2)
    pool_center_y = pool_center.get("y", pool_diameter / 2)

    # 像素到厘米转换因子（优先从配置读取，与旷场/EPM保持一致）
    pixel_to_cm = arena_config.get("pixel_to_cm", 0.1)

    arena_info = {
        "pool_diameter": pool_diameter,
        "pool_center": {"x": pool_center_x, "y": pool_center_y},
        "platform_center": {"x": platform_x, "y": platform_y},
        "platform_radius": platform_radius,
        "pool_axis_x": arena_config.get("pool_axis_x", pool_radius),
        "pool_axis_y": arena_config.get("pool_axis_y", pool_radius),
    }

    # 1. 计算到平台的距离
    distances_to_platform = np.sqrt(
        (x_coords - platform_x) ** 2 +
        (y_coords - platform_y) ** 2
    )

    # 2. 逃逸潜伏期
    on_platform = distances_to_platform < platform_radius
    platform_frames = np.where(on_platform)[0]
    escape_latency_frames = platform_frames[0] if len(platform_frames) > 0 else len(positions)
    escape_latency_seconds = escape_latency_frames / input_data.fps

    # 3. 路径长度
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    frame_distances = np.sqrt(dx ** 2 + dy ** 2)
    path_length_px = np.sum(frame_distances)
    path_length_cm = path_length_px * pixel_to_cm

    # 4. 目标象限时间（根据平台位置动态确定目标象限）
    # 以水池中心为原点，平台所在象限即为目标象限
    if platform_x >= pool_center_x and platform_y <= pool_center_y:
        # 第一象限（右上）
        in_target_quadrant = (x_coords >= pool_center_x) & (y_coords <= pool_center_y)
    elif platform_x < pool_center_x and platform_y <= pool_center_y:
        # 第二象限（左上）
        in_target_quadrant = (x_coords < pool_center_x) & (y_coords <= pool_center_y)
    elif platform_x < pool_center_x and platform_y > pool_center_y:
        # 第三象限（左下）
        in_target_quadrant = (x_coords < pool_center_x) & (y_coords > pool_center_y)
    else:
        # 第四象限（右下）
        in_target_quadrant = (x_coords >= pool_center_x) & (y_coords > pool_center_y)
    target_quadrant_time = np.sum(in_target_quadrant) / len(positions) * 100

    # 5. 平台穿越次数
    platform_crossings = _count_transitions(on_platform)

    # 6. 游泳速度 = 总游泳距离(cm) / 总游泳时间(秒)
    total_swim_time = len(positions) / input_data.fps
    swim_speed_cm_s = path_length_cm / total_swim_time if total_swim_time > 0 else 0

    # 7. 搜索策略评估
    avg_distance_to_platform = np.mean(distances_to_platform)
    thigmotaxis = _calculate_thigmotaxis(x_coords, y_coords, pool_center_x, pool_center_y, pool_radius)

    metrics = {
        "escape_latency_frames": int(escape_latency_frames),
        "escape_latency_seconds": round(escape_latency_seconds, 2),
        "path_length": round(path_length_cm, 2),
        "path_length_px": round(path_length_px, 2),
        "target_quadrant_time_percent": round(target_quadrant_time, 2),
        "platform_crossings": int(platform_crossings),
        "avg_swim_speed": round(swim_speed_cm_s, 2),
        "avg_swim_speed_px_frame": round(np.mean(frame_distances) if len(frame_distances) > 0 else 0, 2),
        "avg_distance_to_platform": round(avg_distance_to_platform, 2),
        "thigmotaxis_percent": round(thigmotaxis, 2)
    }

    return metrics, arena_info


def _calculate_thigmotaxis(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    center_x: float,
    center_y: float,
    pool_radius: float
) -> float:
    """计算边缘游泳时间占比 (thigmotaxis)"""
    distances_from_center = np.sqrt(
        (x_coords - center_x) ** 2 +
        (y_coords - center_y) ** 2
    )
    # 边缘区域定义为距离中心 > 70% 半径
    in_periphery = distances_from_center > pool_radius * 0.7
    return np.sum(in_periphery) / len(x_coords) * 100


# ============== 高架十字迷宫实验指标 ==============

def _calculate_epm_metrics(input_data: CalculateInput) -> tuple[dict, dict]:
    """
    计算高架十字迷宫 (EPM) 实验指标

    指标包括:
    - open_arm_time_percent: 开臂时间占比
    - open_arm_entry_percent: 开臂进入次数占比
    - closed_arm_entries: 闭臂进入次数
    - center_time_percent: 中央区时间占比
    - total_distance: 总移动距离
    - head_dips: 头探次数 (需要关键点数据)
    """
    if not input_data.trajectories:
        return {}, {}

    main_trajectory = input_data.trajectories[0]
    raw_positions = main_trajectory.get("positions", [])

    if not raw_positions:
        return {}, {}

    positions = [
        Position(
            frame_idx=p.get("frame_idx", i),
            x=p["x"],
            y=p["y"],
            w=p.get("w", 0),
            h=p.get("h", 0)
        )
        for i, p in enumerate(raw_positions)
    ]

    x_coords = np.array([p.x for p in positions])
    y_coords = np.array([p.y for p in positions])

    # 获取场地配置
    arena_config = input_data.arena_config or {}

    # 自动检测场地范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # EPM 场地参数
    arena_width = arena_config.get("width", x_max - x_min + 100)
    arena_height = arena_config.get("height", y_max - y_min + 100)

    # 中心点
    center_x = arena_width / 2
    center_y = arena_height / 2

    # 臂的尺寸 (假设十字形对称)
    arm_width = arena_config.get("arm_width", arena_width * 0.15)  # 臂宽度约15%
    arm_length = arena_config.get("arm_length", arena_width * 0.4)  # 臂长度约40%

    # 中央区尺寸
    center_size = arm_width  # 中央区与臂宽度相同

    arena_info = {
        "width": round(arena_width, 2),
        "height": round(arena_height, 2),
        "center_x": round(center_x, 2),
        "center_y": round(center_y, 2),
        "arm_width": round(arm_width, 2),
        "arm_length": round(arm_length, 2),
        "center_size": round(center_size, 2),
        "trajectory_bounds": {
            "x_min": round(x_min, 2),
            "x_max": round(x_max, 2),
            "y_min": round(y_min, 2),
            "y_max": round(y_max, 2)
        }
    }

    # 判断每个位置所在的区域
    regions = _classify_epm_regions(x_coords, y_coords, center_x, center_y,
                                     arm_width, arm_length, center_size)

    # 1. 开臂时间占比 (两个开臂)
    open_arm_time_percent = (np.sum(regions == "open_arm_x") +
                              np.sum(regions == "open_arm_y")) / len(positions) * 100

    # 2. 闭臂时间占比 (两个闭臂)
    closed_arm_time_percent = (np.sum(regions == "closed_arm_x") +
                                np.sum(regions == "closed_arm_y")) / len(positions) * 100

    # 3. 中央区时间占比
    center_time_percent = np.sum(regions == "center") / len(positions) * 100

    # 4. 计算进入次数
    open_arm_entries = _count_arm_entries(regions, "open_arm")
    closed_arm_entries = _count_arm_entries(regions, "closed_arm")

    # 5. 开臂进入次数占比
    total_entries = open_arm_entries + closed_arm_entries
    open_arm_entry_percent = (open_arm_entries / total_entries * 100) if total_entries > 0 else 0

    # 6. 计算移动距离
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    frame_distances = np.sqrt(dx ** 2 + dy ** 2)
    total_distance = np.sum(frame_distances)

    # 7. 平均速度
    avg_speed = np.mean(frame_distances) if len(frame_distances) > 0 else 0

    # 8. 不动时间
    immobile_threshold = 2.0
    immobile_frames = np.sum(frame_distances < immobile_threshold)
    immobile_time_percent = immobile_frames / len(frame_distances) * 100 if len(frame_distances) > 0 else 0

    metrics = {
        # 焦虑相关指标
        "open_arm_time_percent": round(open_arm_time_percent, 2),
        "open_arm_entry_percent": round(open_arm_entry_percent, 2),
        "closed_arm_entries": int(closed_arm_entries),
        "open_arm_entries": int(open_arm_entries),

        # 时间分布
        "closed_arm_time_percent": round(closed_arm_time_percent, 2),
        "center_time_percent": round(center_time_percent, 2),

        # 活动指标
        "total_distance": round(total_distance, 2),
        "avg_speed": round(avg_speed, 2),
        "immobile_time_percent": round(immobile_time_percent, 2),
    }

    # 添加真实单位
    pixel_to_cm = 0.1
    metrics["total_distance_cm"] = round(total_distance * pixel_to_cm, 2)
    metrics["avg_speed_cm_s"] = round(avg_speed * pixel_to_cm * input_data.fps, 2)

    return metrics, arena_info


def _classify_epm_regions(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    center_x: float,
    center_y: float,
    arm_width: float,
    arm_length: float,
    center_size: float
) -> np.ndarray:
    """
    判断每个位置所在的 EPM 区域

    EPM 布局 (假设):
    - 中央区: 中心正方形
    - 开臂: X 轴方向的两个臂 (左右)
    - 闭臂: Y 轴方向的两个臂 (上下)

    Returns:
        区域标签数组: "center", "open_arm_x", "open_arm_y", "closed_arm_x", "closed_arm_y", "outside"
    """
    regions = np.full(len(x_coords), "outside", dtype=object)

    half_arm_width = arm_width / 2
    half_center = center_size / 2

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        # 相对于中心的偏移
        dx = x - center_x
        dy = y - center_y

        # 中央区判断
        if abs(dx) <= half_center and abs(dy) <= half_center:
            regions[i] = "center"
        # 开臂 (X轴方向) - 假设为左右两个臂
        elif abs(dx) > half_center and abs(dy) <= half_arm_width:
            regions[i] = "open_arm_x"  # 开臂
        # 闭臂 (Y轴方向) - 假设为上下两个臂
        elif abs(dy) > half_center and abs(dx) <= half_arm_width:
            regions[i] = "closed_arm_y"  # 闭臂
        # 其他情况
        else:
            # 检查是否在臂的延伸区域
            if abs(dx) <= arm_length and abs(dy) <= arm_length:
                if abs(dx) > abs(dy):
                    regions[i] = "open_arm_x"
                else:
                    regions[i] = "closed_arm_y"

    return regions


def _count_arm_entries(regions: np.ndarray, arm_prefix: str) -> int:
    """
    计算进入特定类型臂的次数

    Args:
        regions: 区域标签数组
        arm_prefix: 臂类型前缀 ("open_arm" 或 "closed_arm")

    Returns:
        进入次数
    """
    # 创建布尔数组表示是否在该类型臂中
    in_arm = np.array([r.startswith(arm_prefix) for r in regions])

    # 从非臂区域进入臂区域计为一次进入
    transitions = np.diff(in_arm.astype(int))
    entries = np.sum(transitions == 1)

    return entries


def _interpret_epm(metrics: dict[str, float]) -> dict[str, str]:
    """解释高架十字迷宫实验结果"""
    open_arm_time = metrics.get("open_arm_time_percent", 0)
    open_arm_entry = metrics.get("open_arm_entry_percent", 0)
    total_distance = metrics.get("total_distance", 0)

    # 焦虑水平评估 (基于开臂时间)
    if open_arm_time < 15:
        anxiety_level = "高焦虑"
        anxiety_desc = "动物在开臂停留时间较短，表现出较高的焦虑水平"
    elif open_arm_time < 30:
        anxiety_level = "中等焦虑"
        anxiety_desc = "动物表现出中等程度的焦虑行为"
    else:
        anxiety_level = "低焦虑"
        anxiety_desc = "动物在开臂探索较多，表现出较低的焦虑水平"

    # 活动水平评估
    if total_distance < 1000:
        activity_level = "低活动性"
        activity_desc = "动物移动较少，需排除运动能力差异对焦虑指标的影响"
    elif total_distance < 3000:
        activity_level = "正常活动"
        activity_desc = "动物表现出正常的探索行为"
    else:
        activity_level = "高活动性"
        activity_desc = "动物表现出活跃的探索行为"

    # 探索行为评估
    if open_arm_entry < 20:
        exploration = "低探索"
        exploration_desc = "动物进入开臂次数较少，探索动机较低"
    elif open_arm_entry < 40:
        exploration = "中等探索"
        exploration_desc = "动物表现出中等的探索行为"
    else:
        exploration = "高探索"
        exploration_desc = "动物频繁探索开臂区域"

    return {
        "anxiety_level": anxiety_level,
        "anxiety_description": anxiety_desc,
        "activity_level": activity_level,
        "activity_description": activity_desc,
        "exploration_level": exploration,
        "exploration_description": exploration_desc,
        "summary": f"{anxiety_level}，{activity_level}，{exploration}"
    }


# ============== 结果解释 ==============

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
    path_length = metrics.get("path_length", 0)
    swim_speed = metrics.get("avg_swim_speed", 0)

    # 学习水平评估（基于逃逸潜伏期）
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

    # 路径效率评估
    if path_length < 200:
        path_efficiency = "路径效率优秀"
        path_desc = "游泳路径短，定位效率高，表明良好的空间记忆"
    elif path_length < 400:
        path_efficiency = "路径效率良好"
        path_desc = "游泳路径适中，定位效率正常"
    elif path_length < 600:
        path_efficiency = "路径效率中等"
        path_desc = "游泳路径较长，空间学习效率有待提高"
    else:
        path_efficiency = "路径效率低"
        path_desc = "游泳路径过长，可能存在空间记忆缺陷或非空间搜索策略"

    # 搜索策略评估
    if thigmotaxis > 50:
        strategy = "边缘搜索"
        strategy_desc = "动物主要沿池壁游泳，表明焦虑或缺乏空间策略，可能掩盖真实学习能力"
    elif target_time > 35:
        strategy = "空间搜索"
        strategy_desc = "动物在目标象限停留较多，表明良好的空间记忆"
    else:
        strategy = "随机搜索"
        strategy_desc = "动物搜索策略不明显，空间记忆可能较弱"

    # 运动能力/非认知因素评估
    if swim_speed < 5:
        motor_level = "运动能力低下"
        motor_desc = "游泳速度明显下降，提示运动能力、体力或视觉/动机因素可能干扰认知结果，不能单独解释为空间记忆缺陷"
    elif swim_speed < 15:
        motor_level = "运动能力正常"
        motor_desc = "游泳速度正常，运动能力未对空间学习评估造成明显干扰"
    else:
        motor_level = "运动能力活跃"
        motor_desc = "游泳速度较快，运动能力良好"

    return {
        "learning_level": learning_level,
        "learning_description": learning_desc,
        "path_efficiency": path_efficiency,
        "path_efficiency_description": path_desc,
        "search_strategy": strategy,
        "strategy_description": strategy_desc,
        "motor_level": motor_level,
        "motor_description": motor_desc,
        "summary": f"{learning_level}，{path_efficiency}，{strategy}"
    }


# ============== 斑马鱼孔板实验指标 ==============

def _calculate_zebrafish_metrics(input_data: CalculateInput) -> tuple[dict, dict]:
    """
    计算斑马鱼孔板实验指标（支持多目标，支持 per-well 几何）

    指标包括:
    - num_tracks: 检测到的鱼数量
    - total_distance: 总移动距离（平均值）
    - avg_speed: 平均速度
    - max_speed: 最大速度
    - immobile_time_percent: 静止时间占比
    - edge_time_percent: 边缘区时间占比（趋触性）
    - center_time_percent: 中心区时间占比
    - crossing_count: 中心/边缘区穿越次数
    """
    trajectories = input_data.trajectories or []
    fps = input_data.fps or 25.0

    if not trajectories:
        return {}, {}

    # 按原始 track_id 排序，重新映射为连续 1-based ID
    sorted_trajs = sorted(trajectories, key=lambda t: t.get("track_id", 0))
    id_map = {t.get("track_id", i): i + 1 for i, t in enumerate(sorted_trajs)}

    # 场地配置
    arena_config = input_data.arena_config or {}
    arena_width = arena_config.get("width", 640)
    arena_height = arena_config.get("height", 480)
    wells = arena_config.get("wells", [])

    # 构建 well 查找表（按 well_id）
    well_by_id: dict[int, dict] = {}
    if wells:
        for w in wells:
            wid = w.get("well_id", 0)
            well_by_id[wid] = w

    # 斑马鱼专用：过滤短轨迹和静止误检碎片
    raw_track_count = len(sorted_trajs)
    filtered_trajs = []
    for traj in sorted_trajs:
        raw_positions = traj.get("positions", [])
        if len(raw_positions) < 5:
            continue  # 少于 5 帧视为碎片
        xs = [p["x"] for p in raw_positions]
        ys = [p["y"] for p in raw_positions]
        dx = np.diff(xs)
        dy = np.diff(ys)
        total_dist = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
        if total_dist < 10.0:
            continue  # 总移动距离小于 10 像素视为静止误检
        filtered_trajs.append(traj)

    sorted_trajs = filtered_trajs

    all_track_metrics: list[dict] = []

    for traj in sorted_trajs:
        raw_track_id = traj.get("track_id", 0)
        track_id = id_map.get(raw_track_id, raw_track_id)
        raw_positions = traj.get("positions", [])
        if not raw_positions:
            continue

        positions = [
            Position(frame_idx=p.get("frame_idx", i), x=p["x"], y=p["y"])
            for i, p in enumerate(raw_positions)
        ]

        x_coords = np.array([p.x for p in positions])
        y_coords = np.array([p.y for p in positions])

        # 速度计算
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        frame_distances = np.sqrt(dx ** 2 + dy ** 2)
        speeds = frame_distances * fps  # px/s

        total_distance = float(np.sum(frame_distances))
        avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

        # 静止时间占比 (速度 < 5 px/s)
        immobile_threshold = 5.0
        immobile_frames = np.sum(speeds < immobile_threshold)
        immobile_time_percent = (immobile_frames / len(speeds) * 100) if len(speeds) > 0 else 0.0

        # 区域判定：优先使用 per-well 几何，否则回退到全局
        if wells and raw_track_id in well_by_id:
            well = well_by_id[raw_track_id]
            wx, wy, wr = well["center_x"], well["center_y"], well["radius"]
            distances_from_center = np.sqrt(
                (x_coords - wx) ** 2 + (y_coords - wy) ** 2
            )
            # 孔内边缘区：距离孔中心 > 0.7 * 孔半径（趋触性）
            in_edge = distances_from_center > wr * 0.7
            # 孔内中心区：距离孔中心 < 0.3 * 孔半径
            in_center = distances_from_center < wr * 0.3
        else:
            # 全局回退（把整个孔板当作一个场地）
            center_ratio = arena_config.get("center_ratio", 0.3)
            arena_center_x = arena_config.get("center_x", arena_width / 2)
            arena_center_y = arena_config.get("center_y", arena_height / 2)
            center_radius = min(arena_width, arena_height) * center_ratio / 2
            arena_radius = min(arena_width, arena_height) / 2
            edge_width = min(arena_width, arena_height) * 0.15

            distances_from_center = np.sqrt(
                (x_coords - arena_center_x) ** 2 +
                (y_coords - arena_center_y) ** 2
            )
            in_edge = distances_from_center > (arena_radius - edge_width)
            in_center = distances_from_center < center_radius

        edge_time_percent = float(np.sum(in_edge) / len(positions) * 100) if len(positions) > 0 else 0.0
        center_time_percent = float(np.sum(in_center) / len(positions) * 100) if len(positions) > 0 else 0.0

        # 穿越次数（中心 ↔ 边缘）
        crossing_count = _count_transitions(in_center) + _count_transitions(~in_center & in_edge)

        px_to_mm = 0.01
        track_metrics = {
            "track_id": track_id,
            "total_distance": round(total_distance, 2),
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(max_speed, 2),
            "immobile_time_percent": round(immobile_time_percent, 2),
            "edge_time_percent": round(edge_time_percent, 2),
            "center_time_percent": round(center_time_percent, 2),
            "crossing_count": int(crossing_count),
            "total_distance_mm": round(total_distance * px_to_mm, 2),
            "avg_speed_mm_s": round(avg_speed * px_to_mm, 2),
        }
        all_track_metrics.append(track_metrics)

    if not all_track_metrics:
        return {}, {}

    def _avg(key: str):
        vals = [m[key] for m in all_track_metrics if key in m]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _max(key: str):
        vals = [m[key] for m in all_track_metrics if key in m]
        return max(vals) if vals else 0.0

    metrics = {
        "num_tracks": len(all_track_metrics),
        "num_tracks_raw": raw_track_count,
        "total_distance": _avg("total_distance"),
        "avg_speed": _avg("avg_speed"),
        "max_speed": _max("max_speed"),
        "immobile_time_percent": _avg("immobile_time_percent"),
        "edge_time_percent": _avg("edge_time_percent"),
        "center_time_percent": _avg("center_time_percent"),
        "crossing_count": _avg("crossing_count"),
        "total_distance_mm": _avg("total_distance_mm"),
        "avg_speed_mm_s": _avg("avg_speed_mm_s"),
    }

    arena_info = {
        "duration_seconds": round(len(sorted_trajs[0].get("positions", [])) / fps, 2) if sorted_trajs else 0,
        "num_tracks": len(all_track_metrics),
        "num_tracks_raw": raw_track_count,
        "track_details": all_track_metrics,
        "width": arena_width,
        "height": arena_height,
    }
    if wells:
        arena_info["wells"] = wells

    return metrics, arena_info


# ============== 线虫行为指标 ==============

def _calculate_worm_metrics(input_data: CalculateInput) -> tuple[dict, dict]:
    """
    计算线虫行为指标（支持多目标）

    基于骨架数据和轨迹计算:
    - avg_speed, max_speed, total_distance
    - immobile_time_percent
    - body_bend_frequency
    - body_wavelength_mean
    - omega_turn_count
    """
    trajectories = input_data.trajectories or []
    skeletons = input_data.skeletons or []
    fps = input_data.fps or 25.0

    if not trajectories:
        return {}, {}

    # 按原始 track_id 排序，并重新映射为连续的 1-based ID
    sorted_trajs = sorted(trajectories, key=lambda t: t.get("track_id", 0))
    id_map = {t.get("track_id", i): i + 1 for i, t in enumerate(sorted_trajs)}

    # 按 frame_idx 组织 skeletons，用于快速查找
    skeletons_by_frame: dict[int, list[dict]] = {}
    for sk in skeletons:
        idx = sk.get("frame_idx", 0)
        skeletons_by_frame.setdefault(idx, []).append(sk)

    all_track_metrics: list[dict] = []

    for traj in sorted_trajs:
        track_id = id_map.get(traj.get("track_id", 0), traj.get("track_id", 0))
        raw_positions = traj.get("positions", [])
        if not raw_positions:
            continue

        # 为该轨迹匹配最近的 skeletons
        traj_skeletons: list[dict] = []
        for pos in raw_positions:
            frame_idx = pos.get("frame_idx", 0)
            candidates = skeletons_by_frame.get(frame_idx, [])
            best_sk = None
            best_dist = float("inf")
            for sk in candidates:
                pts = sk.get("skeleton_points")
                if pts is None or len(pts) == 0:
                    continue
                mid = pts[len(pts) // 2]
                dist = float(np.hypot(pos["x"] - mid[0], pos["y"] - mid[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_sk = sk
            if best_sk and best_dist < 200.0:
                traj_skeletons.append(best_sk)

        positions = [
            Position(frame_idx=p.get("frame_idx", i), x=p["x"], y=p["y"])
            for i, p in enumerate(raw_positions)
        ]

        x_coords = np.array([p.x for p in positions])
        y_coords = np.array([p.y for p in positions])

        # 速度计算
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        frame_distances = np.sqrt(dx ** 2 + dy ** 2)
        speeds = frame_distances * fps  # px/s

        total_distance = float(np.sum(frame_distances))
        avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        max_speed = float(np.max(speeds)) if len(speeds) > 0 else 0.0

        # 不动时间占比 (速度 < 5 px/s)
        immobile_threshold = 5.0
        immobile_frames = np.sum(speeds < immobile_threshold)
        immobile_time_percent = (immobile_frames / len(speeds) * 100) if len(speeds) > 0 else 0.0

        # 体长与体宽
        centerline_lengths = [sk.get("centerline_length", 0) for sk in traj_skeletons if sk.get("centerline_length", 0) > 0]
        body_widths = [sk.get("body_width_mean", 0) for sk in traj_skeletons if sk.get("body_width_mean", 0) > 0]
        avg_body_length = float(np.mean(centerline_lengths)) if centerline_lengths else 0.0
        avg_body_width = float(np.mean(body_widths)) if body_widths else 0.0

        # 弯曲频率与波长
        bend_frequency, wavelength_mean = _calculate_bending(traj_skeletons, fps)

        # Omega turn
        omega_turn_count = _calculate_omega_turns(traj_skeletons)

        px_to_mm = 0.01
        track_metrics = {
            "track_id": track_id,
            "total_distance": round(total_distance, 2),
            "avg_speed": round(avg_speed, 2),
            "max_speed": round(max_speed, 2),
            "immobile_time_percent": round(immobile_time_percent, 2),
            "avg_body_length": round(avg_body_length, 2),
            "avg_body_width": round(avg_body_width, 2),
            "body_bend_frequency": round(bend_frequency, 3),
            "body_wavelength_mean": round(wavelength_mean, 2),
            "omega_turn_count": int(omega_turn_count),
            "total_distance_mm": round(total_distance * px_to_mm, 2),
            "avg_speed_mm_s": round(avg_speed * px_to_mm, 2),
            "avg_body_length_mm": round(avg_body_length * px_to_mm, 2),
            "avg_body_width_mm": round(avg_body_width * px_to_mm, 2),
        }
        all_track_metrics.append(track_metrics)

    if not all_track_metrics:
        return {}, {}

    # 标记骨架可靠的轨迹（成年线虫体长一般 >= 1.0 mm = 100 px）
    for m in all_track_metrics:
        m["skeleton_valid"] = m.get("avg_body_length", 0) >= 100.0

    valid_tracks = [m for m in all_track_metrics if m.get("skeleton_valid", False)]

    # 汇总所有 track 的指标
    def _avg(key: str, tracks=None):
        tracks = tracks or all_track_metrics
        vals = [m[key] for m in tracks if key in m]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _total(key: str, tracks=None):
        tracks = tracks or all_track_metrics
        vals = [m[key] for m in tracks if key in m]
        return round(sum(vals), 2) if vals else 0.0

    def _max(key: str, tracks=None):
        tracks = tracks or all_track_metrics
        vals = [m[key] for m in tracks if key in m]
        return max(vals) if vals else 0.0

    metrics = {
        "num_tracks": len(all_track_metrics),
        "total_distance": _avg("total_distance"),
        "avg_speed": _avg("avg_speed"),
        "max_speed": _max("max_speed"),
        "immobile_time_percent": _avg("immobile_time_percent"),
        "avg_body_length": _avg("avg_body_length", valid_tracks),
        "avg_body_width": _avg("avg_body_width", valid_tracks),
        "body_bend_frequency": _avg("body_bend_frequency", valid_tracks),
        "body_wavelength_mean": _avg("body_wavelength_mean", valid_tracks),
        "omega_turn_count": _total("omega_turn_count", valid_tracks),
        "total_distance_mm": _avg("total_distance_mm"),
        "avg_speed_mm_s": _avg("avg_speed_mm_s"),
        "avg_body_length_mm": _avg("avg_body_length_mm", valid_tracks),
        "avg_body_width_mm": _avg("avg_body_width_mm", valid_tracks),
    }

    arena_info = {
        "duration_seconds": round(len(trajectories[0].get("positions", [])) / fps, 2) if trajectories else 0,
        "num_skeletons": len(skeletons),
        "num_tracks": len(all_track_metrics),
        "num_valid_tracks": len(valid_tracks),
        "track_details": all_track_metrics,
    }

    return metrics, arena_info


def _calculate_bending(skeletons: list[dict], fps: float) -> tuple[float, float]:
    """
    计算身体弯曲频率和平均波长（基于时间序列）

    方法：
    1. 对每帧骨架，计算中点相对于首尾连线的带符号垂直偏移量
    2. 将偏移量组成时间序列
    3. 统计时间轴上的过零次数
    4. 频率 = 过零次数 / 2 / 时长
    """
    if not skeletons:
        return 0.0, 0.0

    # 按 frame_idx 排序
    sorted_skels = sorted(skeletons, key=lambda s: s.get("frame_idx", 0))

    amplitudes = []
    wavelengths = []

    for sk in sorted_skels:
        pts = sk.get("skeleton_points")
        if pts is None or len(pts) < 5:
            amplitudes.append(0.0)
            continue

        pts = np.array(pts, dtype=np.float32)
        n = len(pts)
        head = pts[0]
        tail = pts[-1]
        q = n // 4
        head_mid = pts[q]
        tail_mid = pts[3 * q]
        mid = pts[n // 2]

        # 用前半段中点到后半段中点的向量作为身体主轴
        body_vec = tail_mid - head_mid
        body_len = np.linalg.norm(body_vec)
        if body_len < 1e-6:
            amplitudes.append(0.0)
            continue

        # 整体中点相对于身体主轴的带符号垂直偏移
        am_vec = mid - head_mid
        cross_z = body_vec[0] * am_vec[1] - body_vec[1] * am_vec[0]
        amplitude = cross_z / body_len
        amplitudes.append(amplitude)

        # 单帧波长估计：骨架长度 / (几何弯曲数/2 + 1)
        length = sk.get("centerline_length", 0)
        if length > 0:
            centered = pts - pts.mean(axis=0)
            cov = np.cov(centered.T)
            if cov.ndim >= 2:
                eigvals, eigvecs = np.linalg.eig(cov)
                principal = eigvecs[:, np.argmax(eigvals)]
                normal = np.array([-principal[1], principal[0]])
                proj_p = centered @ principal
                proj_n = centered @ normal
                order = np.argsort(proj_p)
                proj_n_sorted = proj_n[order]
                local_zeros = np.sum(np.diff(np.sign(proj_n_sorted)) != 0)
                if local_zeros > 0:
                    wavelength = length / (local_zeros / 2 + 1)
                    wavelengths.append(wavelength)

    if len(amplitudes) < 3:
        return 0.0, 0.0

    # 平滑时间序列（3点移动平均）
    amplitudes = np.array(amplitudes)
    smoothed = np.convolve(amplitudes, np.ones(3)/3, mode='same')

    # 统计时间轴上过零次数
    zero_crossings = 0
    for i in range(1, len(smoothed)):
        if smoothed[i-1] == 0:
            continue
        if smoothed[i-1] * smoothed[i] < 0:
            zero_crossings += 1

    duration = len(sorted_skels) / fps if fps > 0 else 1.0
    bend_frequency = (zero_crossings / 2) / duration if duration > 0 else 0.0
    wavelength_mean = float(np.mean(wavelengths)) if wavelengths else 0.0

    return bend_frequency, wavelength_mean


def _calculate_omega_turns(skeletons: list[dict]) -> int:
    """
    检测 Omega turn 次数

    方法：计算头-中-尾夹角，当夹角从正常(~180°)突变为锐角(<90°)再恢复时，记为一次 omega turn
    """
    if not skeletons:
        return 0

    angles = []
    for sk in skeletons:
        pts = sk.get("skeleton_points")
        if pts is None or len(pts) < 3:
            angles.append(180.0)
            continue

        pts = np.array(pts, dtype=np.float32)
        head = pts[0]
        tail = pts[-1]
        mid = pts[len(pts) // 2]

        v1 = head - mid
        v2 = tail - mid
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            angles.append(180.0)
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    angles = np.array(angles)
    if len(angles) == 0:
        return 0

    # Omega turn: 角度 < 90° 且持续 3-30 帧
    in_turn = angles < 90.0
    omega_count = 0
    i = 0
    while i < len(in_turn):
        if in_turn[i]:
            start = i
            while i < len(in_turn) and in_turn[i]:
                i += 1
            duration = i - start
            if 3 <= duration <= 60:  # 约 0.1s ~ 2s @ 30fps
                omega_count += 1
        else:
            i += 1

    return omega_count


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


def _interpret_zebrafish(metrics: dict[str, float]) -> dict[str, str]:
    """解释斑马鱼孔板实验结果"""
    avg_speed = metrics.get("avg_speed_mm_s", 0)
    immobile = metrics.get("immobile_time_percent", 0)
    edge = metrics.get("edge_time_percent", 0)
    crossing = metrics.get("crossing_count", 0)

    # 运动能力评估
    if avg_speed < 2.0:
        activity_level = "运动能力低下"
        activity_desc = "斑马鱼游动缓慢，可能存在药物抑制作用或神经系统功能障碍"
    elif avg_speed < 8.0:
        activity_level = "运动能力正常"
        activity_desc = "斑马鱼表现出正常的游动速度"
    else:
        activity_level = "运动能力活跃"
        activity_desc = "斑马鱼游动迅速，表现出强烈的运动活性"

    # 应激/焦虑评估（基于边缘偏好）
    if edge > 50:
        stress_level = "高应激/趋触性强"
        stress_desc = "斑马鱼长时间停留在边缘区域，表现出强烈的趋触性（thigmotaxis），可能与焦虑或应激反应有关"
    elif edge > 30:
        stress_level = "中等应激"
        stress_desc = "斑马鱼有一定程度的边缘偏好，应激水平中等"
    else:
        stress_level = "低应激"
        stress_desc = "斑马鱼在场地中分布均匀，没有明显的边缘焦虑"

    # 探索行为评估
    if crossing < 2:
        exploration_level = "探索行为低下"
        exploration_desc = "斑马鱼很少在中心区和边缘区之间穿梭，探索意愿低"
    elif crossing < 6:
        exploration_level = "探索行为正常"
        exploration_desc = "斑马鱼表现出正常的空间探索行为"
    else:
        exploration_level = "探索行为活跃"
        exploration_desc = "斑马鱼频繁在不同区域间穿梭，探索意愿强烈"

    return {
        "activity_level": activity_level,
        "activity_description": activity_desc,
        "stress_level": stress_level,
        "stress_description": stress_desc,
        "exploration_level": exploration_level,
        "exploration_description": exploration_desc,
        "summary": f"{activity_level}，{stress_level}，{exploration_level}"
    }


def _calculate_quality_score(metrics: dict[str, float]) -> float:
    """计算质量评分"""
    if not metrics:
        return 0.0

    # 检查关键指标是否存在
    key_metrics = ["center_time_percent", "total_distance", "avg_speed"]
    present = sum(1 for k in key_metrics if k in metrics and metrics[k] is not None)

    return present / len(key_metrics)


# ============== 便捷函数 ==============

def calculate_metrics(
    trajectories: list[dict],
    experiment_type: str,
    arena_config: dict | None = None,
    fps: float = 25.0,
    skeletons: list[dict] | None = None
) -> MetricsResult:
    """
    同步计算接口

    Args:
        trajectories: 轨迹数据
        experiment_type: 实验类型
        arena_config: 场地配置
        fps: 帧率
        skeletons: 线虫骨架数据 (可选)

    Returns:
        MetricsResult
    """
    payload = {
        "trajectories": trajectories,
        "experiment_type": experiment_type,
        "arena_config": arena_config,
        "fps": fps
    }
    if skeletons is not None:
        payload["skeletons"] = skeletons
    return calculate_handler(payload)


def tracks_to_trajectories(track_history: dict[int, list[dict]]) -> list[dict]:
    """
    将 SORT 跟踪结果转换为轨迹格式

    Args:
        track_history: {track_id: [{frame, x, y, w, h}, ...]}

    Returns:
        trajectories 格式
    """
    trajectories = []

    for track_id, positions in track_history.items():
        trajectory = {
            "track_id": track_id,
            "positions": [
                {
                    "frame_idx": p["frame"],
                    "x": p["x"] + p.get("w", 0) / 2,  # 中心点
                    "y": p["y"] + p.get("h", 0) / 2,
                    "w": p.get("w", 0),
                    "h": p.get("h", 0)
                }
                for p in positions
            ]
        }
        trajectories.append(trajectory)

    return trajectories


if __name__ == "__main__":
    # 测试代码
    test_trajectories = [
        {
            "track_id": 1,
            "positions": [
                {"frame_idx": i, "x": 100 + i * 2, "y": 100 + np.sin(i * 0.1) * 50}
                for i in range(100)
            ]
        }
    ]

    arena_config = {
        "width": 640,
        "height": 480,
        "center_ratio": 0.3
    }

    result = calculate_metrics(
        test_trajectories,
        "open_field",
        arena_config,
        fps=25.0
    )

    print("=== 旷场实验指标 ===")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")

    print(f"\\n解释: {result.interpretation['summary']}")
