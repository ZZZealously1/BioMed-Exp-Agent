"""
追问计算工具
用于追问对话中的精确数据计算
"""

import numpy as np
from typing import Any


def compute_followup(
    tool_name: str,
    arguments: dict[str, Any],
    positions: np.ndarray,
    frame_indices: np.ndarray,
    fps: float,
    arena_info: dict,
    experiment_type: str,
) -> str:
    """
    执行追问计算

    Args:
        tool_name: 工具名称
        arguments: 工具参数
        positions: (N, 2) 位置数组
        frame_indices: (N,) 帧号数组
        fps: 视频帧率
        arena_info: 场地信息
        experiment_type: 实验类型

    Returns:
        计算结果描述
    """
    tool_map = {
        "max_consecutive_in_zone": _tool_max_consecutive_in_zone,
        "zone_time_between": _tool_zone_time_between,
        "speed_stats_between": _tool_speed_stats_between,
        "zone_entries": _tool_zone_entries,
        "distance_traveled": _tool_distance_traveled,
        "time_at_position": _tool_time_at_position,
    }

    handler = tool_map.get(tool_name)
    if not handler:
        return f"未知计算工具: {tool_name}"

    try:
        return handler(
            arguments, positions, frame_indices, fps, arena_info, experiment_type
        )
    except Exception as e:
        return f"计算出错: {str(e)}"


# ========== 工具定义 (供 LLM 使用) ==========

FOLLOWUP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "max_consecutive_in_zone",
            "description": "计算动物连续处于某个区域的最长时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "enum": ["center", "edge", "open_arm", "closed_arm", "platform"],
                        "description": "区域名称"
                    }
                },
                "required": ["zone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zone_time_between",
            "description": "计算某时间段内动物在各区域的停留时间占比",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "起始时间(秒)"},
                    "end_time": {"type": "number", "description": "结束时间(秒)"},
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "speed_stats_between",
            "description": "计算某时间段内的速度统计 (平均速度、最大速度、最小速度)",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "起始时间(秒)"},
                    "end_time": {"type": "number", "description": "结束时间(秒)"},
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "zone_entries",
            "description": "计算动物进入某区域的次数和每次进入的时刻/持续时长",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "enum": ["center", "edge", "open_arm", "closed_arm", "platform"],
                        "description": "区域名称"
                    }
                },
                "required": ["zone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "distance_traveled",
            "description": "计算某时间段内的移动距离",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "起始时间(秒), 不填则从开头算"},
                    "end_time": {"type": "number", "description": "结束时间(秒), 不填则算到结尾"},
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "time_at_position",
            "description": "计算动物处于某位置范围的时间 (例如离中心小于某距离)",
            "parameters": {
                "type": "object",
                "properties": {
                    "center_x": {"type": "number", "description": "中心点X坐标(px)"},
                    "center_y": {"type": "number", "description": "中心点Y坐标(px)"},
                    "max_distance": {"type": "number", "description": "最大距离(px)"},
                },
                "required": ["center_x", "center_y", "max_distance"]
            }
        }
    },
]


# ========== 区域分类 ==========

def _classify_zones(positions: np.ndarray, arena_info: dict, experiment_type: str) -> np.ndarray:
    """
    对每帧位置分类区域

    Returns:
        (N,) 字符串数组，每帧的区域标签
    """
    import math
    zones = np.full(len(positions), "unknown", dtype=object)

    if experiment_type == "open_field":
        cx = arena_info.get("center_x", 0)
        cy = arena_info.get("center_y", 0)
        r = arena_info.get("center_radius", 0)
        dists = np.sqrt((positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2)
        zones[dists <= r] = "center"
        zones[dists > r] = "edge"

    elif experiment_type == "epm":
        cx = arena_info.get("center_x", 0)
        cy = arena_info.get("center_y", 0)
        arm_w = arena_info.get("arm_width", 0)
        arm_l = arena_info.get("arm_length", 0)
        center_sz = arena_info.get("center_size", arm_w)

        half_aw = arm_w / 2
        half_cs = center_sz / 2

        x, y = positions[:, 0], positions[:, 1]

        # 中央区
        center_mask = (np.abs(x - cx) <= half_cs) & (np.abs(y - cy) <= half_cs)
        zones[center_mask] = "center"

        # 开臂 (水平方向)
        open_right = (np.abs(y - cy) <= half_aw) & (x > cx + half_cs) & (x <= cx + half_cs + arm_l)
        open_left = (np.abs(y - cy) <= half_aw) & (x < cx - half_cs) & (x >= cx - half_cs - arm_l)
        zones[open_right | open_left] = "open_arm"

        # 闭臂 (垂直方向)
        closed_down = (np.abs(x - cx) <= half_aw) & (y > cy + half_cs) & (y <= cy + half_cs + arm_l)
        closed_up = (np.abs(x - cx) <= half_aw) & (y < cy - half_cs) & (y >= cy - half_cs - arm_l)
        zones[closed_down | closed_up] = "closed_arm"

    elif experiment_type == "morris_water_maze":
        pc = arena_info.get("pool_center", {})
        plat_c = arena_info.get("platform_center", {})
        plat_r = arena_info.get("platform_radius", 0)

        # 平台
        dist_plat = np.sqrt((positions[:, 0] - plat_c.get("x", 0)) ** 2 +
                            (positions[:, 1] - plat_c.get("y", 0)) ** 2)
        zones[dist_plat <= plat_r] = "platform"

        # 目标象限 vs 其他象限
        pcx, pcy = pc.get("x", 0), pc.get("y", 0)
        pool_r = arena_info.get("pool_diameter", 0) / 2
        dx = positions[:, 0] - pcx
        dy = positions[:, 1] - pcy
        dist_pool = np.sqrt(dx ** 2 + dy ** 2)

        # 简化: 平台所在象限为目标象限
        plat_dx = plat_c.get("x", 0) - pcx
        plat_dy = plat_c.get("y", 0) - pcy

        target_mask = (dx * plat_dx >= 0) & (dy * plat_dy >= 0) & (zones != "platform")
        other_mask = (~target_mask) & (dist_pool <= pool_r) & (zones != "platform")

        zones[target_mask] = "target_quadrant"
        zones[other_mask] = "other_quadrant"

    return zones


# ========== 工具实现 ==========

def _tool_max_consecutive_in_zone(args, positions, frame_indices, fps, arena_info, experiment_type):
    """计算连续处于某区域的最长时间"""
    zone = args["zone"]
    zones = _classify_zones(positions, arena_info, experiment_type)

    # 计算连续帧数
    in_zone = zones == zone
    if not np.any(in_zone):
        return f"动物未进入区域 '{zone}'"

    max_consec = 0
    current = 0
    for flag in in_zone:
        if flag:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    max_time = max_consec / fps if fps > 0 else 0
    return f"动物连续处于 '{zone}' 区域的最长时间为 {max_time:.2f} 秒 (连续 {max_consec} 帧)"


def _tool_zone_time_between(args, positions, frame_indices, fps, arena_info, experiment_type):
    """计算时间段内各区域停留占比"""
    start_t = args["start_time"]
    end_t = args["end_time"]

    start_frame = int(start_t * fps)
    end_frame = int(end_t * fps)

    mask = (frame_indices >= start_frame) & (frame_indices <= end_frame)
    if not np.any(mask):
        return f"时间范围 {start_t}-{end_t}s 内无数据"

    sub_positions = positions[mask]
    sub_zones = _classify_zones(sub_positions, arena_info, experiment_type)
    total = len(sub_zones)

    result_parts = [f"时间 {start_t}-{end_t}s 内 (共 {total} 帧):"]
    for zone_name in np.unique(sub_zones):
        count = np.sum(sub_zones == zone_name)
        pct = count / total * 100
        result_parts.append(f"  {zone_name}: {count} 帧 ({pct:.1f}%), 约 {count / fps:.2f}s")

    return "\n".join(result_parts)


def _tool_speed_stats_between(args, positions, frame_indices, fps, arena_info, experiment_type):
    """计算时间段内速度统计"""
    start_t = args.get("start_time", 0)
    end_t = args.get("end_time", float('inf'))

    start_frame = int(start_t * fps)
    end_frame = int(end_t * fps) if end_t != float('inf') else int(frame_indices[-1])

    mask = (frame_indices >= start_frame) & (frame_indices <= end_frame)
    sub_pos = positions[mask]

    if len(sub_pos) < 2:
        return "数据不足，无法计算速度"

    # 帧间距离
    diffs = np.diff(sub_pos, axis=0)
    distances = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
    speeds = distances * fps  # px/s

    return (
        f"时间 {start_t}-{min(end_t, frame_indices[-1] / fps):.1f}s 内速度统计:\n"
        f"  平均速度: {np.mean(speeds):.2f} px/s\n"
        f"  最大速度: {np.max(speeds):.2f} px/s\n"
        f"  最小速度: {np.min(speeds):.2f} px/s\n"
        f"  中位速度: {np.median(speeds):.2f} px/s"
    )


def _tool_zone_entries(args, positions, frame_indices, fps, arena_info, experiment_type):
    """计算进入某区域的次数和详情"""
    zone = args["zone"]
    zones = _classify_zones(positions, arena_info, experiment_type)
    in_zone = zones == zone

    # 检测进入和离开
    entries = []
    i = 0
    while i < len(in_zone):
        if in_zone[i]:
            entry_frame = frame_indices[i]
            start_time = entry_frame / fps if fps > 0 else 0
            # 找到连续区域的结束
            while i < len(in_zone) and in_zone[i]:
                i += 1
            exit_frame = frame_indices[i - 1] if i > 0 else entry_frame
            duration = (exit_frame - entry_frame) / fps if fps > 0 else 0
            entries.append({
                "entry_time": round(start_time, 2),
                "duration": round(duration, 2),
                "frames": int(exit_frame - entry_frame + 1)
            })
        else:
            i += 1

    if not entries:
        return f"动物未进入区域 '{zone}'"

    result_parts = [f"动物进入 '{zone}' 区域共 {len(entries)} 次:"]
    for idx, e in enumerate(entries):
        result_parts.append(
            f"  第{idx + 1}次: {e['entry_time']:.2f}s 进入, "
            f"持续 {e['duration']:.2f}s ({e['frames']} 帧)"
        )

    total_time = sum(e["duration"] for e in entries)
    result_parts.append(f"  总停留时间: {total_time:.2f}s")

    return "\n".join(result_parts)


def _tool_distance_traveled(args, positions, frame_indices, fps, arena_info, experiment_type):
    """计算移动距离"""
    start_t = args.get("start_time")
    end_t = args.get("end_time")

    start_frame = int(start_t * fps) if start_t is not None else 0
    end_frame = int(end_t * fps) if end_t is not None else int(frame_indices[-1])

    mask = (frame_indices >= start_frame) & (frame_indices <= end_frame)
    sub_pos = positions[mask]

    if len(sub_pos) < 2:
        return "数据不足"

    diffs = np.diff(sub_pos, axis=0)
    total_dist = np.sum(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2))

    start_str = f"{start_t}s" if start_t is not None else "开始"
    end_str = f"{end_t}s" if end_t is not None else "结束"

    return (
        f"时间 {start_str} → {end_str} 内:\n"
        f"  总移动距离: {total_dist:.1f} px\n"
        f"  帧数: {len(sub_pos)}"
    )


def _tool_time_at_position(args, positions, frame_indices, fps, arena_info, experiment_type):
    """计算处于某位置范围内的时间"""
    cx = args["center_x"]
    cy = args["center_y"]
    max_dist = args["max_distance"]

    dists = np.sqrt((positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2)
    in_range = dists <= max_dist
    count = np.sum(in_range)
    time_s = count / fps if fps > 0 else 0

    return (
        f"距离位置 ({cx:.0f}, {cy:.0f}) {max_dist:.0f}px 以内:\n"
        f"  停留帧数: {count}/{len(positions)} ({count / len(positions) * 100:.1f}%)\n"
        f"  停留时间: {time_s:.2f}s"
    )
