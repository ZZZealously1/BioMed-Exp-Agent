"""
审计日志
记录完整的实验决策链和操作历史
"""

from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class AuditEventType(str, Enum):
    """审计事件类型"""
    # 工作流事件
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"

    # LLM 事件
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"

    # 工具事件
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # 决策事件
    DECISION_MADE = "decision_made"
    CONSTRAINT_VIOLATION = "constraint_violation"
    QUALITY_CHECK = "quality_check"
    REPAIR_TRIGGERED = "repair_triggered"

    # 用户交互
    USER_CONFIRMATION = "user_confirmation"
    USER_FEEDBACK = "user_feedback"


class AuditEvent(BaseModel):
    """审计事件"""
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: AuditEventType
    experiment_id: str
    node_name: str | None = None
    tool_name: str | None = None

    # 事件数据
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    decision: str | None = None
    reason: str | None = None

    # 元数据
    duration_ms: float | None = None
    success: bool = True
    error_message: str | None = None


class ExperimentAudit(BaseModel):
    """实验审计记录"""
    experiment_id: str
    user_request: str
    video_path: str
    experiment_type: str | None = None

    # 时间戳
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None

    # 事件列表
    events: list[AuditEvent] = Field(default_factory=list)

    # 最终状态
    final_status: str = "in_progress"
    final_metrics: dict[str, Any] | None = None
    report_path: str | None = None

    def add_event(self, event: AuditEvent) -> None:
        """添加审计事件"""
        self.events.append(event)

    def finalize(
        self,
        status: str,
        metrics: dict[str, Any] | None = None,
        report_path: str | None = None
    ) -> None:
        """完成审计记录"""
        self.end_time = datetime.now()
        self.final_status = status
        self.final_metrics = metrics
        self.report_path = report_path

    @property
    def duration_seconds(self) -> float | None:
        """计算总耗时"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_decision_chain(self) -> list[dict[str, Any]]:
        """获取决策链"""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "node": e.node_name,
                "decision": e.decision,
                "reason": e.reason
            }
            for e in self.events
            if e.event_type == AuditEventType.DECISION_MADE
        ]

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """获取工具调用记录"""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "tool": e.tool_name,
                "input": e.input_data,
                "output": e.output_data,
                "success": e.success,
                "duration_ms": e.duration_ms
            }
            for e in self.events
            if e.event_type in [AuditEventType.TOOL_CALL, AuditEventType.TOOL_RESULT]
        ]


class AuditLogger:
    """
    审计日志记录器

    记录完整的实验决策链和操作历史
    """

    def __init__(self, log_dir: str | Path = "logs/audit"):
        """
        初始化审计日志器

        Args:
            log_dir: 日志存储目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 当前活跃的审计记录
        self._current_audit: ExperimentAudit | None = None

    def start_experiment(
        self,
        experiment_id: str,
        user_request: str,
        video_path: str,
        experiment_type: str | None = None
    ) -> ExperimentAudit:
        """
        开始新的实验审计

        Args:
            experiment_id: 实验 ID
            user_request: 用户请求
            video_path: 视频路径
            experiment_type: 实验类型

        Returns:
            ExperimentAudit 对象
        """
        self._current_audit = ExperimentAudit(
            experiment_id=experiment_id,
            user_request=user_request,
            video_path=video_path,
            experiment_type=experiment_type
        )

        # 记录开始事件
        self.log_event(
            event_type=AuditEventType.WORKFLOW_START,
            input_data={
                "user_request": user_request,
                "video_path": video_path
            }
        )

        return self._current_audit

    def log_event(
        self,
        event_type: AuditEventType,
        node_name: str | None = None,
        tool_name: str | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        decision: str | None = None,
        reason: str | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error_message: str | None = None
    ) -> AuditEvent:
        """
        记录审计事件

        Args:
            event_type: 事件类型
            node_name: 节点名称
            tool_name: 工具名称
            input_data: 输入数据
            output_data: 输出数据
            decision: 决策内容
            reason: 决策原因
            duration_ms: 耗时 (毫秒)
            success: 是否成功
            error_message: 错误信息

        Returns:
            AuditEvent 对象
        """
        if not self._current_audit:
            raise RuntimeError("没有活跃的实验审计记录")

        event = AuditEvent(
            event_type=event_type,
            experiment_id=self._current_audit.experiment_id,
            node_name=node_name,
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            decision=decision,
            reason=reason,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )

        self._current_audit.add_event(event)
        return event

    def log_tool_call(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error_message: str | None = None
    ) -> AuditEvent:
        """
        记录工具调用

        Args:
            tool_name: 工具名称
            input_data: 输入数据
            output_data: 输出数据
            duration_ms: 耗时
            success: 是否成功
            error_message: 错误信息

        Returns:
            AuditEvent 对象
        """
        return self.log_event(
            event_type=AuditEventType.TOOL_CALL,
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )

    def log_decision(
        self,
        node_name: str,
        decision: str,
        reason: str
    ) -> AuditEvent:
        """
        记录决策

        Args:
            node_name: 节点名称
            decision: 决策内容
            reason: 决策原因

        Returns:
            AuditEvent 对象
        """
        return self.log_event(
            event_type=AuditEventType.DECISION_MADE,
            node_name=node_name,
            decision=decision,
            reason=reason
        )

    def log_quality_check(
        self,
        quality_metrics: dict[str, Any],
        is_acceptable: bool,
        issues: list[str] | None = None
    ) -> AuditEvent:
        """
        记录质量检查

        Args:
            quality_metrics: 质量指标
            is_acceptable: 是否可接受
            issues: 问题列表

        Returns:
            AuditEvent 对象
        """
        return self.log_event(
            event_type=AuditEventType.QUALITY_CHECK,
            output_data=quality_metrics,
            decision="accept" if is_acceptable else "reject",
            reason="; ".join(issues) if issues else None
        )

    def end_experiment(
        self,
        status: str,
        metrics: dict[str, Any] | None = None,
        report_path: str | None = None
    ) -> Path:
        """
        结束实验审计

        Args:
            status: 最终状态
            metrics: 最终指标
            report_path: 报告路径

        Returns:
            审计日志文件路径
        """
        if not self._current_audit:
            raise RuntimeError("没有活跃的实验审计记录")

        # 记录结束事件
        self.log_event(
            event_type=AuditEventType.WORKFLOW_END,
            output_data={"status": status, "metrics": metrics}
        )

        # 完成审计记录
        self._current_audit.finalize(status, metrics, report_path)

        # 保存到文件
        log_path = self._save_audit(self._current_audit)

        self._current_audit = None
        return log_path

    def _save_audit(self, audit: ExperimentAudit) -> Path:
        """保存审计记录到文件"""
        filename = f"{audit.experiment_id}_{audit.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        log_path = self.log_dir / filename

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(audit.model_dump_json(indent=2))

        return log_path

    def load_audit(self, log_path: str | Path) -> ExperimentAudit:
        """加载审计记录"""
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentAudit(**data)

    @property
    def current_audit(self) -> ExperimentAudit | None:
        """获取当前审计记录"""
        return self._current_audit
