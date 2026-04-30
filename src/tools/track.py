"""
SORT 跟踪工具包装
将 SORT 跟踪器包装为 MCP 工具

SORT: Simple Online and Realtime Tracking
https://arxiv.org/abs/1602.00763
"""

from typing import Any
from pydantic import BaseModel, Field
from mcp.types import Tool
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2


class Track(BaseModel):
    """单个跟踪轨迹"""
    track_id: int
    boxes: list[dict[str, float]]  # 每帧的边界框
    start_frame: int
    end_frame: int
    continuity: float = Field(description="轨迹连续性评分")


class TrackingResult(BaseModel):
    """跟踪结果"""
    tracks: list[Track]
    total_frames: int
    track_continuity: float
    quality_score: float
    statistics: dict[str, Any]


class TrackInput(BaseModel):
    """跟踪工具输入"""
    boxes: list[dict[str, Any]] = Field(description="检测结果 (每帧的边界框)")
    frame_count: int = Field(description="总帧数")
    max_age: int = Field(default=30, description="跟踪器最大丢失帧数")
    min_hits: int = Field(default=3, description="确认跟踪所需的最小命中数")
    iou_threshold: float = Field(default=0.3, description="IOU 匹配阈值")


# MCP 工具定义
track_tool = Tool(
    name="track",
    description="""
使用 SORT 算法进行多目标跟踪。

输入: 检测结果 (边界框列表)
输出: 跟踪轨迹 + 质量指标

质量指标:
- track_continuity: 轨迹连续性 (0-1)
- quality_score: 整体跟踪质量
""",
    inputSchema={
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_idx": {"type": "integer"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "w": {"type": "number"},
                        "h": {"type": "number"},
                        "confidence": {"type": "number"}
                    }
                },
                "description": "检测结果列表"
            },
            "frame_count": {
                "type": "integer",
                "description": "视频总帧数"
            },
            "max_age": {
                "type": "integer",
                "default": 30,
                "description": "跟踪器最大丢失帧数"
            },
            "min_hits": {
                "type": "integer",
                "default": 3,
                "description": "确认跟踪所需的最小命中数"
            },
            "iou_threshold": {
                "type": "number",
                "default": 0.3,
                "description": "IOU 匹配阈值"
            }
        },
        "required": ["boxes", "frame_count"]
    }
)


# ============== SORT 核心实现 ==============

def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    计算两个边界框的 IOU

    Args:
        bbox1: [x, y, w, h]
        bbox2: [x, y, w, h]

    Returns:
        IOU 值
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 计算交集
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # 计算并集
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def associate_detections_to_trackers(
    dets: np.ndarray,
    trks: np.ndarray,
    iou_threshold: float = 0.3
) -> tuple[list, list, list]:
    """
    将检测结果与跟踪器关联

    Args:
        dets: 检测框 [N, 4]
        trks: 跟踪框 [M, 4]
        iou_threshold: IOU 阈值

    Returns:
        matches: [(det_idx, trk_idx), ...]
        unmatched_dets: 未匹配的检测索引
        unmatched_trks: 未匹配的跟踪索引
    """
    if len(trks) == 0:
        return [], list(range(len(dets))), []

    if len(dets) == 0:
        return [], [], list(range(len(trks)))

    # 计算 IOU 矩阵
    iou_matrix = np.zeros((len(dets), len(trks)))
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_matrix[d, t] = iou(det, trk)

    # 匈牙利算法求解最优匹配
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_dets = list(range(len(dets)))
    unmatched_trks = list(range(len(trks)))

    for row, col in zip(row_indices, col_indices):
        if iou_matrix[row, col] >= iou_threshold:
            matches.append((row, col))
            if row in unmatched_dets:
                unmatched_dets.remove(row)
            if col in unmatched_trks:
                unmatched_trks.remove(col)

    return matches, unmatched_dets, unmatched_trks


class KalmanBoxTracker:
    """
    基于卡尔曼滤波的边界框跟踪器

    状态向量: [x, y, s, r, vx, vy, vs]
    观测向量: [x, y, s, r]
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        """初始化跟踪器"""
        # 使用 OpenCV 的卡尔曼滤波器
        self.kf = cv2.KalmanFilter(7, 4)

        # 状态转移矩阵
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # 观测矩阵
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # 初始化状态
        x, y, w, h = bbox
        s = w * h  # 面积
        r = w / (h + 1e-6)  # 宽高比
        self.kf.statePost = np.array([[x + w/2], [y + h/2], [s], [r], [0], [0], [0]], dtype=np.float32)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history: list[np.ndarray] = []

    def predict(self) -> np.ndarray:
        """预测下一帧状态"""
        # 面积不能为负
        if self.kf.statePost[2] + self.kf.statePost[6] <= 0:
            self.kf.statePost[6] = 0

        prediction = self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self._state_to_bbox(prediction)

    def update(self, bbox: np.ndarray):
        """用检测结果更新状态"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        x, y, w, h = bbox
        s = w * h
        r = w / (h + 1e-6)
        measurement = np.array([[x + w/2], [y + h/2], [s], [r]], dtype=np.float32)
        self.kf.correct(measurement)

    def get_state(self) -> np.ndarray:
        """获取当前边界框"""
        return self._state_to_bbox(self.kf.statePost)

    def _state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        """将状态转换为边界框 [x, y, w, h]"""
        cx, cy, s, r = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x = cx - w / 2
        y = cy - h / 2
        return np.array([x, y, w, h])


class SORTTracker:
    """SORT 多目标跟踪器"""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        更新跟踪器

        Args:
            dets: 当前帧的检测结果 [N, 4] (x, y, w, h)

        Returns:
            跟踪结果 [M, 5] (x, y, w, h, track_id)
        """
        self.frame_count += 1

        # 预测所有跟踪器的状态
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks.append(pos)

        # 删除无效跟踪器
        for t in reversed(to_del):
            self.trackers.pop(t)

        trks = np.array(trks) if trks else np.empty((0, 4))

        # 关联检测与跟踪
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # 更新匹配的跟踪器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 创建新的跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 收集结果并删除过期的跟踪器
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < self.max_age) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


# ============== MCP 工具处理器 ==============

async def track_handler(arguments: dict) -> TrackingResult:
    """
    跟踪工具处理器

    Args:
        arguments: 工具参数

    Returns:
        TrackingResult 对象
    """
    input_data = TrackInput(**arguments)

    # 按帧组织检测框
    detections_by_frame = _organize_detections(input_data.boxes)

    # 初始化 SORT 跟踪器
    tracker = SORTTracker(
        max_age=input_data.max_age,
        min_hits=input_data.min_hits,
        iou_threshold=input_data.iou_threshold
    )

    # 存储每帧的跟踪结果
    frame_tracks: dict[int, list[dict]] = {}

    # 逐帧处理
    for frame_idx in range(input_data.frame_count):
        # 获取当前帧的检测框
        frame_dets = detections_by_frame.get(frame_idx, [])

        if frame_dets:
            dets = np.array([[d['x'], d['y'], d['w'], d['h']] for d in frame_dets])
        else:
            dets = np.empty((0, 4))

        # 更新跟踪器
        tracks = tracker.update(dets)

        # 存储跟踪结果
        frame_tracks[frame_idx] = []
        for t in tracks:
            x, y, w, h, track_id = t
            frame_tracks[frame_idx].append({
                'track_id': int(track_id),
                'x': float(x),
                'y': float(y),
                'w': float(w),
                'h': float(h)
            })

    # 全局轨迹后处理：合并碎片、消除 ID Switch
    detections_by_frame = _organize_detections(input_data.boxes)
    target_count = max((len(v) for v in detections_by_frame.values()), default=0)
    frame_tracks = _merge_fragmented_tracks(
        frame_tracks, target_count=target_count, max_gap=15, max_distance=100.0
    )

    # 组织成轨迹
    tracks = _organize_tracks(frame_tracks)

    # 计算质量指标
    track_continuity = _calculate_continuity(tracks, input_data.frame_count)
    quality_score = _calculate_quality_score(track_continuity, len(tracks))

    statistics = {
        "total_tracks": len(tracks),
        "avg_track_length": float(np.mean([t.end_frame - t.start_frame + 1 for t in tracks])) if tracks else 0,
        "max_track_length": max([t.end_frame - t.start_frame + 1 for t in tracks]) if tracks else 0,
        "frame_coverage": sum(len(v) for v in frame_tracks.values()) / input_data.frame_count if input_data.frame_count > 0 else 0
    }

    return TrackingResult(
        tracks=tracks,
        total_frames=input_data.frame_count,
        track_continuity=round(track_continuity, 4),
        quality_score=round(quality_score, 4),
        statistics=statistics
    )


def _organize_detections(boxes: list[dict]) -> dict[int, list[dict]]:
    """按帧组织检测框"""
    result = {}
    for box in boxes:
        frame_idx = box.get("frame_idx", 0)
        if frame_idx not in result:
            result[frame_idx] = []
        result[frame_idx].append(box)
    return result


def _organize_tracks(frame_tracks: dict[int, list[dict]]) -> list[Track]:
    """
    将帧级跟踪结果组织为轨迹
    """
    # 收集每个 track_id 的所有边界框
    track_boxes: dict[int, list[tuple[int, dict]]] = {}

    for frame_idx, tracks in frame_tracks.items():
        for t in tracks:
            tid = t['track_id']
            if tid not in track_boxes:
                track_boxes[tid] = []
            track_boxes[tid].append((frame_idx, {
                'x': t['x'],
                'y': t['y'],
                'w': t['w'],
                'h': t['h']
            }))

    # 构建 Track 对象
    result = []
    for tid, boxes_with_frame in track_boxes.items():
        if not boxes_with_frame:
            continue

        # 按帧排序
        boxes_with_frame.sort(key=lambda x: x[0])
        frames = [f for f, _ in boxes_with_frame]
        boxes = [b for _, b in boxes_with_frame]

        # 计算连续性
        start_frame = frames[0]
        end_frame = frames[-1]
        expected_frames = end_frame - start_frame + 1
        actual_frames = len(frames)
        continuity = actual_frames / expected_frames if expected_frames > 0 else 0

        result.append(Track(
            track_id=tid,
            boxes=boxes,
            start_frame=start_frame,
            end_frame=end_frame,
            continuity=round(continuity, 4)
        ))

    # 按 track_id 排序
    result.sort(key=lambda t: t.track_id)
    return result


def _merge_fragmented_tracks(
    frame_tracks: dict[int, list[dict]],
    target_count: int,
    max_gap: int = 15,
    max_distance: float = 100.0
) -> dict[int, list[dict]]:
    """
    全局轨迹优化：合并碎片轨迹，消除 ID Switch。

    策略：
    1. 将 frame_tracks 按 track_id 聚类为短轨迹
    2. 计算两两轨迹之间的时序间隙和空间距离（支持重叠帧的 ID Switch 修复）
    3. 按代价从小到大贪心合并，直到轨迹数 <= target_count
    4. 合并时处理重叠帧：保留较长轨迹的数据
    """
    if target_count <= 0:
        return frame_tracks

    # 按 track_id 收集所有帧记录
    track_records: dict[int, list[tuple[int, dict]]] = {}
    for frame_idx, tracks in frame_tracks.items():
        for t in tracks:
            tid = t['track_id']
            track_records.setdefault(tid, []).append((frame_idx, {
                'x': t['x'], 'y': t['y'], 'w': t['w'], 'h': t['h']
            }))

    if len(track_records) <= target_count:
        return frame_tracks

    # 排序并计算每条轨迹的属性
    segments: list[dict] = []
    for tid, records in track_records.items():
        records.sort(key=lambda r: r[0])
        frames = [r[0] for r in records]
        boxes = [r[1] for r in records]
        start_f = frames[0]
        end_f = frames[-1]
        # 中心点
        cx = [b['x'] + b['w']/2 for b in boxes]
        cy = [b['y'] + b['h']/2 for b in boxes]
        segments.append({
            'tid': tid,
            'records': records,
            'start': start_f,
            'end': end_f,
            'cx': cx,
            'cy': cy,
            'length': len(records)
        })

    def _seg_center(seg, idx):
        """获取 segment 第 idx 个记录的中心点"""
        if idx < 0:
            idx = 0
        if idx >= len(seg['records']):
            idx = len(seg['records']) - 1
        b = seg['records'][idx][1]
        return (b['x'] + b['w']/2, b['y'] + b['h']/2)

    def _velocity_at_end(seg):
        """计算 segment 末尾的平均速度（px/帧）"""
        n = len(seg['records'])
        if n < 2:
            return (0.0, 0.0)
        # 取最后 min(3, n-1) 帧计算平均速度
        k = min(3, n - 1)
        vx = (seg['records'][-1][1]['x'] + seg['records'][-1][1]['w']/2 -
              (seg['records'][-k-1][1]['x'] + seg['records'][-k-1][1]['w']/2)) / k
        vy = (seg['records'][-1][1]['y'] + seg['records'][-1][1]['h']/2 -
              (seg['records'][-k-1][1]['y'] + seg['records'][-k-1][1]['h']/2)) / k
        return (vx, vy)

    def _predicted_distance(seg_i, seg_j):
        """用 seg_i 末尾速度预测到 seg_j 起点的距离"""
        vx, vy = _velocity_at_end(seg_i)
        gap = seg_j['start'] - seg_i['end']
        if gap < 0:
            gap = 0
        x1, y1 = _seg_center(seg_i, -1)
        pred_x = x1 + vx * gap
        pred_y = y1 + vy * gap
        x2, y2 = _seg_center(seg_j, 0)
        return np.hypot(x2 - pred_x, y2 - pred_y)

    def _overlap_cooccurrence(seg_i, seg_j):
        """返回两段时间重叠的帧数"""
        overlap_start = max(seg_i['start'], seg_j['start'])
        overlap_end = min(seg_i['end'], seg_j['end'])
        return max(0, overlap_end - overlap_start + 1)

    def _overlap_mean_distance(seg_i, seg_j):
        """计算重叠帧内两轨迹的平均中心距离"""
        frames_i = {r[0]: r[1] for r in seg_i['records']}
        frames_j = {r[0]: r[1] for r in seg_j['records']}
        overlap_frames = set(frames_i.keys()) & set(frames_j.keys())
        if not overlap_frames:
            return float('inf')
        dists = []
        for f in overlap_frames:
            cx_i = frames_i[f]['x'] + frames_i[f]['w']/2
            cy_i = frames_i[f]['y'] + frames_i[f]['h']/2
            cx_j = frames_j[f]['x'] + frames_j[f]['w']/2
            cy_j = frames_j[f]['y'] + frames_j[f]['h']/2
            dists.append(np.hypot(cx_i - cx_j, cy_i - cy_j))
        return float(np.mean(dists))

    merge_count = 0
    while len(segments) > target_count:
        best_pair = None
        best_cost = float('inf')

        for i in range(len(segments)):
            for j in range(len(segments)):
                if i == j:
                    continue
                seg_i = segments[i]
                seg_j = segments[j]

                overlap = _overlap_cooccurrence(seg_i, seg_j)
                if overlap > 0:
                    # 允许少量重叠（ID Switch 通常只有 1-3 帧重叠）
                    if overlap > 3:
                        continue
                    mean_dist = _overlap_mean_distance(seg_i, seg_j)
                    if mean_dist > max_distance:
                        continue
                    # 重叠场景的代价：平均距离 + 重叠惩罚
                    cost = mean_dist + overlap * 20.0
                else:
                    gap = seg_j['start'] - seg_i['end']
                    if gap > max_gap:
                        continue
                    dist = _predicted_distance(seg_i, seg_j)
                    if dist > max_distance:
                        continue
                    cost = dist + gap * 5.0

                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        seg_i = segments[i]
        seg_j = segments[j]

        # 合并：保留 seg_i 的所有记录，seg_j 中不冲突的记录追加
        merged_records = list(seg_i['records'])
        existing_frames = {r[0] for r in merged_records}
        for r in seg_j['records']:
            if r[0] not in existing_frames:
                merged_records.append(r)
        merged_records.sort(key=lambda r: r[0])

        new_seg = {
            'tid': seg_i['tid'],
            'records': merged_records,
            'start': merged_records[0][0],
            'end': merged_records[-1][0],
            'cx': [r[1]['x'] + r[1]['w']/2 for r in merged_records],
            'cy': [r[1]['y'] + r[1]['h']/2 for r in merged_records],
            'length': len(merged_records)
        }

        # 移除 i, j，插入新 segment
        indices = sorted([i, j], reverse=True)
        for idx in indices:
            segments.pop(idx)
        segments.append(new_seg)
        merge_count += 1

    # 若仍超 target_count，进行第二轮：强制将最短轨迹并入最近邻
    while len(segments) > target_count:
        # 找最短的轨迹
        seg_idx = min(range(len(segments)), key=lambda i: segments[i]['length'])
        seg_short = segments[seg_idx]
        best_target = None
        best_cost = float('inf')

        for j in range(len(segments)):
            if j == seg_idx:
                continue
            seg_j = segments[j]
            # 计算 seg_short 到 seg_j 的最小中心距离（不限 gap，但惩罚大 gap）
            min_dist = float('inf')
            min_gap = float('inf')
            for fi, bi in seg_short['records']:
                cx_i = bi['x'] + bi['w']/2
                cy_i = bi['y'] + bi['h']/2
                for fj, bj in seg_j['records']:
                    cx_j = bj['x'] + bj['w']/2
                    cy_j = bj['y'] + bj['h']/2
                    d = np.hypot(cx_i - cx_j, cy_i - cy_j)
                    if d < min_dist:
                        min_dist = d
                        min_gap = abs(fj - fi)

            # 大 gap 强惩罚，但允许合并
            cost = min_dist + min_gap * 10.0
            if cost < best_cost:
                best_cost = cost
                best_target = j

        if best_target is None:
            break

        seg_target = segments[best_target]
        merged_records = list(seg_target['records'])
        existing_frames = {r[0] for r in merged_records}
        for r in seg_short['records']:
            if r[0] not in existing_frames:
                merged_records.append(r)
        merged_records.sort(key=lambda r: r[0])

        new_seg = {
            'tid': seg_target['tid'],
            'records': merged_records,
            'start': merged_records[0][0],
            'end': merged_records[-1][0],
            'cx': [r[1]['x'] + r[1]['w']/2 for r in merged_records],
            'cy': [r[1]['y'] + r[1]['h']/2 for r in merged_records],
            'length': len(merged_records)
        }

        indices = sorted([seg_idx, best_target], reverse=True)
        for idx in indices:
            segments.pop(idx)
        segments.append(new_seg)
        merge_count += 1

    # 最终清理：强制将极短轨迹并入最近的长轨迹
    # 动态阈值：取 max(5, 平均长度 * 0.25)，防止中等长度碎片残留
    avg_length = sum(seg['length'] for seg in segments) / len(segments) if segments else 0
    min_track_length = max(5, int(avg_length * 0.25))
    while True:
        short_indices = [i for i, seg in enumerate(segments) if seg['length'] < min_track_length]
        if not short_indices:
            break
        seg_idx = short_indices[0]
        seg_short = segments[seg_idx]
        best_target = None
        best_cost = float('inf')

        for j in range(len(segments)):
            if j == seg_idx:
                continue
            seg_j = segments[j]
            min_dist = float('inf')
            for fi, bi in seg_short['records']:
                cx_i = bi['x'] + bi['w']/2
                cy_i = bi['y'] + bi['h']/2
                for fj, bj in seg_j['records']:
                    cx_j = bj['x'] + bj['w']/2
                    cy_j = bj['y'] + bj['h']/2
                    d = np.hypot(cx_i - cx_j, cy_i - cy_j)
                    if d < min_dist:
                        min_dist = d
            if min_dist < best_cost:
                best_cost = min_dist
                best_target = j

        if best_target is None:
            break

        seg_target = segments[best_target]
        merged_records = list(seg_target['records'])
        existing_frames = {r[0] for r in merged_records}
        for r in seg_short['records']:
            if r[0] not in existing_frames:
                merged_records.append(r)
        merged_records.sort(key=lambda r: r[0])

        new_seg = {
            'tid': seg_target['tid'],
            'records': merged_records,
            'start': merged_records[0][0],
            'end': merged_records[-1][0],
            'cx': [r[1]['x'] + r[1]['w']/2 for r in merged_records],
            'cy': [r[1]['y'] + r[1]['h']/2 for r in merged_records],
            'length': len(merged_records)
        }

        indices = sorted([seg_idx, best_target], reverse=True)
        for idx in indices:
            segments.pop(idx)
        segments.append(new_seg)

    # 重建 frame_tracks
    new_frame_tracks: dict[int, list[dict]] = {}
    for seg in segments:
        for frame_idx, box in seg['records']:
            new_frame_tracks.setdefault(frame_idx, []).append({
                'track_id': seg['tid'],
                **box
            })

    return new_frame_tracks


def refine_track_history(
    track_history: dict[int, list[dict]],
    all_detections: list[dict],
    target_count: int | None = None,
    max_gap: int = 15,
    max_distance: float = 100.0
) -> dict[int, list[dict]]:
    """
    对 SORT 输出的 track_history 进行全局后处理：合并碎片轨迹，消除 ID Switch。

    Args:
        track_history: {track_id: [{frame, x, y, w, h}, ...]}
        all_detections: 原始检测框列表，用于推断目标数量
        target_count: 目标数量上限，未指定时自动计算检测框最多的一帧
        max_gap: 允许合并的最大帧间隙
        max_distance: 允许合并的最大空间距离（像素）

    Returns:
        优化后的 track_history
    """
    # 构建 frame_tracks
    frame_tracks: dict[int, list[dict]] = {}
    for tid, positions in track_history.items():
        for p in positions:
            frame_tracks.setdefault(p["frame"], []).append({
                "track_id": tid,
                "x": p["x"],
                "y": p["y"],
                "w": p["w"],
                "h": p["h"]
            })

    if target_count is None:
        detections_by_frame: dict[int, list] = {}
        for b in all_detections:
            detections_by_frame.setdefault(b.get("frame_idx", 0), []).append(b)
        target_count = max((len(v) for v in detections_by_frame.values()), default=0)

    if target_count <= 0 or len(track_history) <= target_count:
        return track_history

    # 调用全局合并
    refined_frame_tracks = _merge_fragmented_tracks(
        frame_tracks, target_count=target_count, max_gap=max_gap, max_distance=max_distance
    )

    # 重建 track_history
    refined_history: dict[int, list[dict]] = {}
    for fidx, tracks in refined_frame_tracks.items():
        for t in tracks:
            tid = int(t["track_id"])
            if tid not in refined_history:
                refined_history[tid] = []
            refined_history[tid].append({
                "frame": fidx,
                "x": t["x"],
                "y": t["y"],
                "w": t["w"],
                "h": t["h"]
            })

    # 保持每个 track 按帧排序
    for tid in refined_history:
        refined_history[tid].sort(key=lambda p: p["frame"])

    return refined_history


def _calculate_continuity(tracks: list[Track], total_frames: int) -> float:
    """计算轨迹连续性"""
    if not tracks:
        return 0.0

    # 计算被跟踪覆盖的帧数
    covered_frames = set()
    for track in tracks:
        for frame_idx in range(track.start_frame, track.end_frame + 1):
            covered_frames.add(frame_idx)

    coverage = len(covered_frames) / total_frames if total_frames > 0 else 0

    # 考虑轨迹的平均连续性
    avg_continuity = np.mean([t.continuity for t in tracks]) if tracks else 0

    return float(coverage * 0.6 + avg_continuity * 0.4)


def _calculate_quality_score(continuity: float, track_count: int) -> float:
    """计算质量评分"""
    if continuity >= 0.95:
        return 1.0
    elif continuity >= 0.9:
        return 0.95
    elif continuity >= 0.85:
        return 0.9
    elif continuity >= 0.7:
        return 0.8
    else:
        return continuity * 0.9


# ============== 孔板孔位检测与孔内独立跟踪 ==============

def _compute_median_frame(video_path: str, max_frames: int = 100) -> np.ndarray | None:
    """计算视频中位数帧（消除移动物体）"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, min(frame_count - 1, max_frames - 1),
                                  min(frame_count, max_frames), dtype=int)
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    if not frames:
        return None
    return np.median(np.array(frames), axis=0).astype(np.uint8)


def detect_wells_from_median_frame(
    video_path: str,
    expected_rows: int = 4,
    expected_cols: int = 6
) -> list[dict]:
    """
    基于视频中位数帧的孔板孔位检测 (v5: k-means 聚类)

    方法：
    1. 计算中位数帧（消除鱼的移动）
    2. 霍夫圆变换检测孔壁（v2 高精度参数）
    3. 用 k-means 聚类圆心为行列，生成规则网格
    """
    median = _compute_median_frame(video_path)
    if median is None:
        return []

    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # v2 高精度霍夫圆参数
    min_r = int(min(h, w) / 12)
    max_r = int(min(h, w) / 6)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2,
        param1=80,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r
    )

    detected = []
    if circles is not None:
        for c in circles[0]:
            detected.append({"x": c[0], "y": c[1], "r": c[2]})

    # 网格拟合
    if len(detected) >= expected_rows * expected_cols * 0.4:
        return _fit_grid_by_clustering(detected, w, h, expected_rows, expected_cols)
    else:
        return _uniform_grid(w, h, expected_rows, expected_cols)


def _fit_grid_by_clustering(detected, w, h, expected_rows, expected_cols):
    """用 k-means 聚类检测到的圆心为行列（最稳定）"""
    from scipy.cluster.vq import kmeans
    centers = np.array([[d["x"], d["y"]] for d in detected])

    # y 方向聚类（行）
    y_centroids, _ = kmeans(centers[:, 1].reshape(-1, 1), expected_rows)
    y_centroids = sorted(y_centroids.flatten())

    # x 方向聚类（列）
    x_centroids, _ = kmeans(centers[:, 0].reshape(-1, 1), expected_cols)
    x_centroids = sorted(x_centroids.flatten())

    # 估算半径
    if len(x_centroids) >= 2 and len(y_centroids) >= 2:
        dx = np.median(np.diff(x_centroids))
        dy = np.median(np.diff(y_centroids))
        radius = min(dx, dy) * 0.42
    else:
        radius = min(w / expected_cols, h / expected_rows) * 0.4

    wells = []
    for row_idx, cy in enumerate(y_centroids):
        for col_idx, cx in enumerate(x_centroids):
            wells.append({
                "well_id": row_idx * expected_cols + col_idx + 1,
                "center_x": float(cx),
                "center_y": float(cy),
                "radius": float(radius),
                "row": row_idx,
                "col": col_idx,
            })
    return wells


def _uniform_grid(w, h, expected_rows, expected_cols):
    """均匀网格回退方案"""
    spacing_x = w / expected_cols
    spacing_y = h / expected_rows
    radius = min(spacing_x, spacing_y) * 0.4
    offset_x = spacing_x / 2
    offset_y = spacing_y / 2

    wells = []
    for row_idx in range(expected_rows):
        for col_idx in range(expected_cols):
            wells.append({
                "well_id": row_idx * expected_cols + col_idx + 1,
                "center_x": float(offset_x + col_idx * spacing_x),
                "center_y": float(offset_y + row_idx * spacing_y),
                "radius": float(radius),
                "row": row_idx,
                "col": col_idx,
            })
    return wells


def assign_boxes_to_wells(
    boxes: list[dict],
    wells: list[dict]
) -> dict[int, list[dict]]:
    """将检测框分配到最近的孔（距离在 1.5 倍半径内）"""
    if not wells or not boxes:
        return {}

    import numpy as np
    well_centers = np.array([[w["center_x"], w["center_y"]] for w in wells])
    assignments: dict[int, list[dict]] = {}

    for box in boxes:
        cx = box["x"] + box["w"] / 2
        cy = box["y"] + box["h"] / 2
        dists = np.sqrt((well_centers[:, 0] - cx) ** 2 + (well_centers[:, 1] - cy) ** 2)
        nearest_idx = int(np.argmin(dists))
        if dists[nearest_idx] <= wells[nearest_idx]["radius"] * 1.5:
            assignments.setdefault(nearest_idx, []).append(box)

    return assignments


def well_based_track(
    video_path: str,
    boxes: list[dict],
    frame_count: int,
    fps: float = 25.0
) -> dict[str, Any]:
    """
    基于孔位置的空间分割跟踪

    流程：
    1. 自动检测孔位（基于视频中位数帧）
    2. 每帧将检测框按空间位置分配到对应孔
    3. 每个孔内独立运行 SORT（单目标，因为每孔一条鱼）
    4. 输出统一格式的 track_history

    Returns:
        {
            "track_history": {track_id: [{frame, x, y, w, h}, ...]},
            "wells": [孔信息列表],
            "num_tracks": 轨迹数,
            "total_frames": frame_count,
        }
    """
    import numpy as np

    # 1. 检测孔位
    wells = detect_wells_from_median_frame(video_path, expected_rows=4, expected_cols=6)
    if not wells:
        # 回退：从检测框推断视频尺寸并生成均匀网格
        if boxes:
            max_x = max(b["x"] + b["w"] for b in boxes)
            max_y = max(b["y"] + b["h"] for b in boxes)
            wells = _uniform_grid(max_x, max_y, expected_rows=4, expected_cols=6)
        else:
            return {"track_history": {}, "wells": [], "num_tracks": 0, "total_frames": frame_count}

    # 2. 按帧组织检测框
    boxes_by_frame: dict[int, list[dict]] = {}
    for box in boxes:
        fidx = box.get("frame_idx", 0)
        boxes_by_frame.setdefault(fidx, []).append(box)

    # 3. 每个孔一个 SORT 跟踪器（单目标模式，孔内只有一条鱼）
    well_trackers = {i: SORTTracker(max_age=20, min_hits=2, iou_threshold=0.1)
                     for i in range(len(wells))}
    well_histories: dict[int, list[tuple[int, dict]]] = {i: [] for i in range(len(wells))}

    for frame_idx in range(frame_count):
        frame_boxes = boxes_by_frame.get(frame_idx, [])
        assignments = assign_boxes_to_wells(frame_boxes, wells)

        for well_idx, well_boxes in assignments.items():
            if well_idx not in well_trackers:
                continue
            tracker = well_trackers[well_idx]

            # 单孔内只应有一条鱼，取置信度最高的一个检测框
            if len(well_boxes) > 1:
                well_boxes = [max(well_boxes, key=lambda b: b.get("confidence", 0))]

            if well_boxes:
                b = well_boxes[0]
                dets = np.array([[b["x"], b["y"], b["w"], b["h"]]])
            else:
                dets = np.empty((0, 4))

            tracks = tracker.update(dets)

            for t in tracks:
                x, y, w, h, track_id = t
                # track_id 在孔内独立计数，需要映射为全局唯一 ID
                global_tid = well_idx * 1000 + int(track_id)
                well_histories[well_idx].append((frame_idx, {
                    "x": float(x), "y": float(y),
                    "w": float(w), "h": float(h),
                    "global_tid": global_tid,
                }))

    # 4. 合并各孔的历史到统一的 track_history
    track_history: dict[int, list[dict]] = {}
    for well_idx, records in well_histories.items():
        if not records:
            continue
        # 孔内可能因漏检产生多个 track_id，取最长的作为主轨迹
        tid_records: dict[int, list[tuple[int, dict]]] = {}
        for frame_idx, data in records:
            gtid = data["global_tid"]
            tid_records.setdefault(gtid, []).append((frame_idx, data))

        # 选择最长的轨迹
        if tid_records:
            main_tid = max(tid_records.keys(), key=lambda k: len(tid_records[k]))
            for frame_idx, data in sorted(tid_records[main_tid], key=lambda r: r[0]):
                # 使用 well_idx + 1 作为最终的 track_id（1-based，连续）
                final_tid = well_idx + 1
                if final_tid not in track_history:
                    track_history[final_tid] = []
                track_history[final_tid].append({
                    "frame": frame_idx,
                    "x": data["x"], "y": data["y"],
                    "w": data["w"], "h": data["h"],
                })

    return {
        "track_history": track_history,
        "wells": wells,
        "num_tracks": len(track_history),
        "total_frames": frame_count,
    }


# 便捷函数
def track_detections(
    boxes: list[dict],
    frame_count: int,
    **kwargs
) -> TrackingResult:
    """同步跟踪接口"""
    import asyncio
    return asyncio.run(track_handler({
        "boxes": boxes,
        "frame_count": frame_count,
        **kwargs
    }))


if __name__ == "__main__":
    print("SORT 跟踪器测试")

    # 模拟检测数据
    test_boxes = [
        {"frame_idx": 0, "x": 100, "y": 100, "w": 50, "h": 50, "confidence": 0.9},
        {"frame_idx": 1, "x": 105, "y": 102, "w": 48, "h": 52, "confidence": 0.9},
        {"frame_idx": 2, "x": 110, "y": 104, "w": 50, "h": 50, "confidence": 0.9},
        {"frame_idx": 3, "x": 115, "y": 106, "w": 52, "h": 48, "confidence": 0.9},
        {"frame_idx": 4, "x": 120, "y": 108, "w": 50, "h": 50, "confidence": 0.9},
    ]

    result = track_detections(test_boxes, frame_count=5)
    print(f"跟踪完成:")
    print(f"  - 轨迹数: {len(result.tracks)}")
    print(f"  - 连续性: {result.track_continuity:.2%}")
    print(f"  - 质量分: {result.quality_score:.2f}")

    for track in result.tracks:
        print(f"  - 轨迹 {track.track_id}: 帧 {track.start_frame}-{track.end_frame}, 连续性 {track.continuity:.2f}")
