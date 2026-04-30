"""
实验状态定义
定义 LangGraph 工作流中的状态结构
"""

from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import operator


class VideoMetadata(BaseModel):
    """视频元数据"""
    path: str
    duration: float  # 秒
    fps: float
    width: int
    height: int
    brightness: Optional[float] = None
    contrast: Optional[float] = None


class QualityMetrics(BaseModel):
    """质量指标"""
    detection_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    track_continuity: float = Field(default=0.0, ge=0.0, le=1.0)
    segmentation_iou: Optional[float] = None

    def is_acceptable(self, thresholds: dict[str, float] | None = None) -> bool:
        """检查质量是否可接受"""
        thresholds = thresholds or {
            "detection_rate": 0.9,
            "track_continuity": 0.85
        }
        return (
            self.detection_rate >= thresholds.get("detection_rate", 0.9) and
            self.track_continuity >= thresholds.get("track_continuity", 0.85)
        )


class ToolResult(BaseModel):
    """工具执行结果"""
    tool_name: str
    success: bool
    output: Any
    quality: Optional[QualityMetrics] = None
    error: Optional[str] = None
    failure_mode: Optional[str] = None
    suggested_fix: Optional[str] = None


class ExperimentMetrics(BaseModel):
    """实验分析指标"""
    # 旷场实验指标
    center_time: Optional[float] = None  # 中心区域时间占比
    total_distance: Optional[float] = None  # 总移动距离
    avg_speed: Optional[float] = None  # 平均速度

    # 水迷宫指标
    escape_latency: Optional[float] = None  # 逃逸潜伏期
    path_length: Optional[float] = None  # 路径长度
    quadrant_time: Optional[float] = None  # 目标象限时间


class ExperimentState(BaseModel):
    """
    实验状态 - LangGraph 工作流的核心状态对象

    使用 Annotated 实现状态累积和合并
    """

    # 输入信息
    user_request: str  # 用户的自然语言请求
    video_path: str  # 视频文件路径

    # 解析后的信息
    experiment_type: Optional[str] = None  # 实验类型 (open_field, morris_water_maze, etc.)
    species: Optional[str] = None  # 物种
    constraints: dict[str, Any] = Field(default_factory=dict)  # 约束条件

    # 视频元数据
    video_metadata: Optional[VideoMetadata] = None

    # 规划与执行
    current_plan: list[str] = Field(default_factory=list)  # 当前执行计划
    current_step: int = 0  # 当前步骤索引

    # 工具调用历史
    tool_results: Annotated[list[ToolResult], operator.add] = Field(default_factory=list)

    # 质量评估
    quality_metrics: Optional[QualityMetrics] = None
    repair_attempts: int = 0
    max_repair_attempts: int = 3

    # 最终结果
    metrics: Optional[ExperimentMetrics] = None
    report_path: Optional[str] = None

    # 状态控制
    is_complete: bool = False
    needs_human_review: bool = False
    error_message: Optional[str] = None

    # 元信息
    experiment_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # 记忆系统
    memory_hint: Optional[str] = None  # 记忆系统给出的提示（如"复用历史策略"）

    def update_timestamp(self) -> None:
        """更新时间戳"""
        self.updated_at = datetime.now()

    def should_retry(self) -> bool:
        """判断是否应该重试"""
        return self.repair_attempts < self.max_repair_attempts

    def get_last_tool_result(self) -> Optional[ToolResult]:
        """获取最后一个工具结果"""
        return self.tool_results[-1] if self.tool_results else None
