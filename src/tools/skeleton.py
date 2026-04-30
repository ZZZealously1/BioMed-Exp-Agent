"""
线虫骨架提取工具
从 SAM 分割掩码中提取中心骨架线
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from collections import deque


def extract_skeleton(mask: np.ndarray, num_points: int = 30) -> dict:
    """
    从二值掩码中提取线虫中心骨架线

    Args:
        mask: 二值掩码 (H, W)，0/1 或 0/255
        num_points: 骨架等距采样点数

    Returns:
        {
            "skeleton_points": np.ndarray,  # shape (num_points, 2), 从头到尾排序
            "centerline_length": float,
            "body_width_mean": float,
            "head": tuple[float, float],
            "tail": tuple[float, float],
        }
    """
    # 确保二值化
    binary = (mask > 0).astype(np.uint8)

    if binary.sum() == 0:
        return {
            "skeleton_points": np.zeros((num_points, 2), dtype=np.float32),
            "centerline_length": 0.0,
            "body_width_mean": 0.0,
            "head": (0.0, 0.0),
            "tail": (0.0, 0.0),
        }

    # 形态学闭运算：填补小空洞，连接断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 骨架细化
    skel = skeletonize(binary > 0).astype(np.uint8)

    # 提取骨架坐标
    skel_coords = np.column_stack(np.where(skel > 0))
    if len(skel_coords) == 0:
        return {
            "skeleton_points": np.zeros((num_points, 2), dtype=np.float32),
            "centerline_length": 0.0,
            "body_width_mean": 0.0,
            "head": (0.0, 0.0),
            "tail": (0.0, 0.0),
        }

    # 找出端点 (8-邻域内只有一个邻居的像素)
    endpoints = _find_endpoints(skel)
    if len(endpoints) < 2:
        # 如果没有两个端点，退化为PCA主轴采样
        return _fallback_skeleton(binary, num_points)

    # BFS 找最长路径 (从第一个端点到最远端点)
    longest_path = _trace_longest_path(skel, endpoints)
    if len(longest_path) < 2:
        return _fallback_skeleton(binary, num_points)

    # np.where 返回的是 (y, x)，需要翻转为 (x, y) 以匹配图像坐标系
    longest_path_xy = [(float(x), float(y)) for y, x in longest_path]

    # 计算距离变换得到体宽
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    body_widths = []
    for x, y in longest_path_xy:
        w = dist_transform[int(y), int(x)] * 2  # 直径
        body_widths.append(w)
    body_width_mean = float(np.mean(body_widths)) if body_widths else 0.0

    # 等距采样
    sampled = _resample_points(np.array(longest_path_xy, dtype=np.float32), num_points)

    # 头尾判定：用曲率或运动方向，这里简单取路径两端
    head = tuple(sampled[0])
    tail = tuple(sampled[-1])

    # 中心线长度
    diffs = np.diff(sampled, axis=0)
    centerline_length = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))

    return {
        "skeleton_points": sampled,
        "centerline_length": centerline_length,
        "body_width_mean": body_width_mean,
        "head": head,
        "tail": tail,
    }


def extract_skeletons_from_masks(masks_data: list[dict], num_points: int = 30) -> list[dict]:
    """
    批量从 SAM 分割结果中提取骨架

    Args:
        masks_data: segment_video 返回的 masks 列表 [{frame_idx, mask_data, bbox, area}]
        num_points: 骨架采样点数

    Returns:
        带骨架信息的列表 [{frame_idx, skeleton_points, centerline_length, body_width_mean, head, tail, bbox}, ...]
    """
    results = []
    for item in masks_data:
        mask = item.get("mask_data")
        if mask is None:
            continue
        skel = extract_skeleton(mask, num_points)
        results.append({
            "frame_idx": item.get("frame_idx", 0),
            "skeleton_points": skel["skeleton_points"],
            "centerline_length": skel["centerline_length"],
            "body_width_mean": skel["body_width_mean"],
            "head": skel["head"],
            "tail": skel["tail"],
            "bbox": item.get("bbox", {}),
            "area": item.get("area", 0),
        })
    return results


def _find_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    """找出骨架的端点"""
    coords = np.column_stack(np.where(skel > 0))
    endpoints = []
    for y, x in coords:
        # 8-邻域
        neighbors = skel[max(0, y-1):y+2, max(0, x-1):x+2].sum() - 1
        if neighbors == 1:
            endpoints.append((int(y), int(x)))
    return endpoints


def _trace_longest_path(skel: np.ndarray, endpoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """从骨架端点中找出最长路径"""
    skel_coords = set(map(tuple, np.column_stack(np.where(skel > 0))))
    longest = []

    for start in endpoints:
        # BFS
        queue = deque([(start, [start])])
        visited = {start}
        local_longest = [start]

        while queue:
            (y, x), path = queue.popleft()
            if len(path) > len(local_longest):
                local_longest = path

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in skel_coords and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append(((ny, nx), path + [(ny, nx)]))

        if len(local_longest) > len(longest):
            longest = local_longest

    return longest


def _resample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    沿点序列等距重采样

    Args:
        points: (N, 2) 点序列
        num_points: 目标点数

    Returns:
        (num_points, 2) 等距采样点
    """
    if len(points) <= 1:
        return np.zeros((num_points, 2), dtype=np.float32)

    # 计算累积弧长
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length <= 0:
        return np.zeros((num_points, 2), dtype=np.float32)

    # 目标采样位置
    target_lengths = np.linspace(0, total_length, num_points)

    # 插值
    resampled = np.zeros((num_points, 2), dtype=np.float32)
    for i, t in enumerate(target_lengths):
        idx = np.searchsorted(cum_lengths, t)
        if idx == 0:
            resampled[i] = points[0]
        elif idx >= len(points):
            resampled[i] = points[-1]
        else:
            # 线性插值
            t0, t1 = cum_lengths[idx - 1], cum_lengths[idx]
            ratio = (t - t0) / (t1 - t0) if t1 > t0 else 0
            resampled[i] = points[idx - 1] * (1 - ratio) + points[idx] * ratio

    return resampled


def _fallback_skeleton(binary: np.ndarray, num_points: int) -> dict:
    """当骨架细化失败时的回退方案：用 PCA 主轴采样"""
    coords = np.column_stack(np.where(binary > 0)).astype(np.float32)
    if len(coords) == 0:
        return {
            "skeleton_points": np.zeros((num_points, 2), dtype=np.float32),
            "centerline_length": 0.0,
            "body_width_mean": 0.0,
            "head": (0.0, 0.0),
            "tail": (0.0, 0.0),
        }

    # PCA 找主轴 (coords 是 (y, x))
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal = eigvecs[:, np.argmax(eigvals)]

    # 投影到主轴
    projections = centered @ principal
    min_idx, max_idx = np.argmin(projections), np.argmax(projections)
    p_start_yx = coords[min_idx]
    p_end_yx = coords[max_idx]

    # 翻转为 (x, y)
    p_start = np.array([p_start_yx[1], p_start_yx[0]], dtype=np.float32)
    p_end = np.array([p_end_yx[1], p_end_yx[0]], dtype=np.float32)

    sampled = np.linspace(p_start, p_end, num_points).astype(np.float32)
    centerline_length = float(np.linalg.norm(p_end - p_start))

    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    body_width_mean = float(dist_transform[coords[:, 0].astype(int), coords[:, 1].astype(int)].mean() * 2)

    return {
        "skeleton_points": sampled,
        "centerline_length": centerline_length,
        "body_width_mean": body_width_mean,
        "head": tuple(p_start),
        "tail": tuple(p_end),
    }
