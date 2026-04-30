"""
质量评估器
评估 CV 工具输出的质量
"""

from typing import Any
from pydantic import BaseModel


class QualityAssessment(BaseModel):
    """质量评估结果"""
    is_acceptable: bool
    detection_rate: float
    track_continuity: float
    overall_score: float
    issues: list[str]
    suggestions: list[str]


class QualityAssessor:
    """
    质量评估器

    评估 CV 工具输出的质量，提供诊断和修复建议
    """

    def __init__(
        self,
        detection_threshold: float = 0.9,
        continuity_threshold: float = 0.85,
        iou_threshold: float = 0.8
    ):
        """
        初始化质量评估器

        Args:
            detection_threshold: 检测率阈值
            continuity_threshold: 连续性阈值
            iou_threshold: IOU 阈值
        """
        self.detection_threshold = detection_threshold
        self.continuity_threshold = continuity_threshold
        self.iou_threshold = iou_threshold

    def assess_detection(self, result: dict[str, Any]) -> QualityAssessment:
        """
        评估检测结果质量

        Args:
            result: 检测结果

        Returns:
            QualityAssessment 对象
        """
        detection_rate = result.get("detection_rate", 0)
        issues = []
        suggestions = []

        if detection_rate < self.detection_threshold:
            issues.append(f"检测率 ({detection_rate:.2%}) 低于阈值 ({self.detection_threshold:.2%})")

            # 诊断问题
            if result.get("video_info", {}).get("brightness", 100) < 50:
                suggestions.append("视频亮度较低，建议启用低光照增强")
            else:
                suggestions.append("尝试降低置信度阈值或调整检测参数")

        is_acceptable = detection_rate >= self.detection_threshold
        overall_score = detection_rate

        return QualityAssessment(
            is_acceptable=is_acceptable,
            detection_rate=detection_rate,
            track_continuity=0.0,
            overall_score=overall_score,
            issues=issues,
            suggestions=suggestions
        )

    def assess_tracking(self, result: dict[str, Any]) -> QualityAssessment:
        """
        评估跟踪结果质量

        Args:
            result: 跟踪结果

        Returns:
            QualityAssessment 对象
        """
        continuity = result.get("track_continuity", 0)
        issues = []
        suggestions = []

        if continuity < self.continuity_threshold:
            issues.append(f"跟踪连续性 ({continuity:.2%}) 低于阈值 ({self.continuity_threshold:.2%})")

            # 诊断问题
            track_count = result.get("statistics", {}).get("total_tracks", 0)
            if track_count > 10:
                suggestions.append("轨迹数量过多，可能存在 ID 切换问题，建议调整 min_hits 参数")
            else:
                suggestions.append("尝试增加 max_age 参数以保持更长时间的跟踪")

        is_acceptable = continuity >= self.continuity_threshold
        overall_score = continuity

        return QualityAssessment(
            is_acceptable=is_acceptable,
            detection_rate=1.0,
            track_continuity=continuity,
            overall_score=overall_score,
            issues=issues,
            suggestions=suggestions
        )

    def assess_segmentation(self, result: dict[str, Any]) -> QualityAssessment:
        """
        评估分割结果质量

        Args:
            result: 分割结果

        Returns:
            QualityAssessment 对象
        """
        mean_iou = result.get("mean_iou", 0)
        issues = []
        suggestions = []

        if mean_iou < self.iou_threshold:
            issues.append(f"分割 IOU ({mean_iou:.2%}) 低于阈值 ({self.iou_threshold:.2%})")
            suggestions.append("尝试调整点提示或使用更高精度的模型 (vit_h)")

        is_acceptable = mean_iou >= self.iou_threshold
        overall_score = mean_iou

        return QualityAssessment(
            is_acceptable=is_acceptable,
            detection_rate=1.0,
            track_continuity=1.0,
            overall_score=overall_score,
            issues=issues,
            suggestions=suggestions
        )

    def assess(self, tool_name: str, result: dict[str, Any]) -> QualityAssessment:
        """
        根据工具类型自动评估

        Args:
            tool_name: 工具名称
            result: 工具结果

        Returns:
            QualityAssessment 对象
        """
        assessors = {
            "detect": self.assess_detection,
            "track": self.assess_tracking,
            "segment": self.assess_segmentation
        }

        assessor = assessors.get(tool_name)
        if assessor:
            return assessor(result)

        # 默认评估
        return QualityAssessment(
            is_acceptable=True,
            detection_rate=1.0,
            track_continuity=1.0,
            overall_score=1.0,
            issues=[],
            suggestions=[]
        )
