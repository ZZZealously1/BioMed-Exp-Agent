"""
记忆存储层
提供实验经验的写入、检索和统计功能
"""

import json
import os
from typing import Optional
from sqlmodel import SQLModel, create_engine, Session, select

from .models import ExperimentExperience


class MemoryStore:
    """实验经验记忆存储

    使用本地 SQLite 持久化存储，支持：
    - 写入实验经验（reflect 节点调用）
    - 检索相似实验的历史经验（plan 节点调用）
    - 获取某类实验的最佳策略
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # 默认存储在项目根目录的 data/ 下
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            db_dir = os.path.join(base_dir, "data")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "experience.db")

        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        # 自动创建表
        SQLModel.metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------

    def add_experience(self, state) -> ExperimentExperience:
        """将一次实验的状态和结果写入记忆

        Args:
            state: ExperimentState 对象，需包含 experiment_type, species,
                   constraints, current_plan, quality_metrics 等字段

        Returns:
            写入的经验记录
        """
        # 从 state 中提取质量指标
        quality = state.quality_metrics
        detection_rate = quality.detection_rate if quality else 0.0
        track_continuity = quality.track_continuity if quality else 0.0
        quality_score = detection_rate * 0.5 + track_continuity * 0.5

        # 判断是否为一次通过（无修复且质量达标）
        success = (
            state.repair_attempts == 0 and
            detection_rate >= 0.9 and
            track_continuity >= 0.85
        )

        # 提取约束条件
        constraints = state.constraints or {}
        has_low_light = bool(constraints.get("low_light", False))
        target_behavior = constraints.get("target_behavior")

        # 提取视频元数据
        duration = None
        brightness = None
        if state.video_metadata:
            duration = state.video_metadata.duration
            brightness = state.video_metadata.brightness

        # 分析计划特征
        plan_steps = state.current_plan or []
        used_enhance = "enhance_video" in plan_steps
        used_segment = "segment" in plan_steps

        # 尝试提取跟踪器配置（从 tool_results 中找）
        tracker_config = None
        for result in state.tool_results:
            if result.tool_name == "track" and result.output:
                cfg = result.output.get("tracker_config")
                if cfg:
                    tracker_config = json.dumps(cfg)
                    break

        # 获取最后一个结果的质量信息
        last_result = state.get_last_tool_result()
        failure_mode = last_result.failure_mode if last_result else None

        exp = ExperimentExperience(
            experiment_type=state.experiment_type or "unknown",
            species=state.species or "unknown",
            video_duration=duration,
            video_brightness=brightness,
            has_low_light=has_low_light,
            target_behavior=target_behavior,
            plan_steps=json.dumps(plan_steps),
            used_enhance_video=used_enhance,
            used_segment=used_segment,
            tracker_config=tracker_config,
            detection_rate=detection_rate,
            track_continuity=track_continuity,
            quality_score=quality_score,
            success=success,
            repair_attempts=state.repair_attempts,
            failure_mode=failure_mode,
            experiment_id=state.experiment_id,
        )

        with Session(self.engine) as session:
            session.add(exp)
            session.commit()
            session.refresh(exp)

        return exp

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------

    def retrieve_similar(
        self,
        experiment_type: str,
        species: str,
        constraints: Optional[dict] = None,
        top_k: int = 3,
    ) -> list[ExperimentExperience]:
        """检索相似实验的历史经验

        匹配逻辑：
        1. experiment_type + species 精确匹配
        2. has_low_light 精确匹配（低光策略差异大）
        3. 按 quality_score 降序排序

        Args:
            experiment_type: 实验类型
            species: 物种
            constraints: 约束条件字典（如 {"low_light": True}）
            top_k: 返回的最大记录数

        Returns:
            相似经验列表，按质量分数降序排列
        """
        constraints = constraints or {}
        has_low_light = bool(constraints.get("low_light", False))

        with Session(self.engine) as session:
            statement = (
                select(ExperimentExperience)
                .where(ExperimentExperience.experiment_type == experiment_type)
                .where(ExperimentExperience.species == species)
                .where(ExperimentExperience.has_low_light == has_low_light)
                .order_by(ExperimentExperience.quality_score.desc())
                .limit(top_k)
            )
            results = session.exec(statement).all()
            return list(results)

    def get_best_strategy(
        self,
        experiment_type: str,
        species: str,
        constraints: Optional[dict] = None,
        min_quality: float = 0.9,
    ) -> Optional[ExperimentExperience]:
        """获取某类实验的最佳策略

        返回 quality_score >= min_quality 的最高分记录。

        Args:
            experiment_type: 实验类型
            species: 物种
            constraints: 约束条件字典
            min_quality: 最低质量分数阈值

        Returns:
            最佳经验记录，如果没有则返回 None
        """
        similar = self.retrieve_similar(
            experiment_type=experiment_type,
            species=species,
            constraints=constraints,
            top_k=1,
        )
        if similar and similar[0].quality_score >= min_quality:
            return similar[0]
        return None

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self, experiment_type: str, species: str) -> dict:
        """统计某类实验的成功率

        Returns:
            {"total": 总数, "success": 成功数, "success_rate": 成功率,
             "avg_quality": 平均质量分数, "avg_repair": 平均修复次数}
        """
        with Session(self.engine) as session:
            statement = (
                select(ExperimentExperience)
                .where(ExperimentExperience.experiment_type == experiment_type)
                .where(ExperimentExperience.species == species)
            )
            results = session.exec(statement).all()

        if not results:
            return {"total": 0, "success": 0, "success_rate": 0.0, "avg_quality": 0.0, "avg_repair": 0.0}

        total = len(results)
        success_count = sum(1 for r in results if r.success)
        avg_quality = sum(r.quality_score for r in results) / total
        avg_repair = sum(r.repair_attempts for r in results) / total

        return {
            "total": total,
            "success": success_count,
            "success_rate": success_count / total if total > 0 else 0.0,
            "avg_quality": avg_quality,
            "avg_repair": avg_repair,
        }

    def list_all(self, limit: int = 20) -> list[ExperimentExperience]:
        """列出所有经验记录（调试用）"""
        with Session(self.engine) as session:
            statement = select(ExperimentExperience).order_by(ExperimentExperience.created_at.desc()).limit(limit)
            return list(session.exec(statement).all())
