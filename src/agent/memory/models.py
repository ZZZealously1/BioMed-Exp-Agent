"""
记忆模块数据模型
使用 SQLModel 定义实验经验记录表
"""

from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime


class ExperimentExperience(SQLModel, table=True):
    """实验经验记录 - 长期记忆的核心数据结构

    每次实验完成后，reflect 节点将策略和结果写入此表。
    plan 节点在生成计划前检索相似实验的历史经验，辅助决策。
    """
    id: Optional[int] = Field(default=None, primary_key=True)

    # 实验特征（用于相似度匹配）
    experiment_type: str = Field(index=True)  # open_field / zebrafish_plate / ...
    species: str = Field(index=True)  # mouse / zebrafish / worm / ...
    video_duration: Optional[float] = None  # 视频时长（秒）
    video_brightness: Optional[float] = None  # 亮度
    has_low_light: bool = Field(default=False, index=True)  # 是否低光
    target_behavior: Optional[str] = None  # 关注的行为类型

    # 执行策略
    plan_steps: str  # JSON 序列化的计划步骤列表
    used_enhance_video: bool = False  # 是否用了视频增强
    used_segment: bool = False  # 是否用了分割
    tracker_config: Optional[str] = None  # 跟踪器参数 JSON

    # 结果质量
    detection_rate: float = Field(ge=0.0, le=1.0)
    track_continuity: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)  # 综合分数 (detection * 0.5 + continuity * 0.5)
    success: bool = Field(default=False, index=True)  # 是否一次通过（无修复）
    repair_attempts: int = 0  # 修复次数
    failure_mode: Optional[str] = None  # 失败模式（如果有）

    # 元信息
    created_at: datetime = Field(default_factory=datetime.now)
    experiment_id: Optional[str] = None  # 关联的实验 ID
