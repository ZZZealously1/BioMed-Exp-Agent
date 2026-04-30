"""
约束验证器
验证实验配置和结果是否符合 Protocol 定义
"""

from typing import Any
from pydantic import BaseModel, Field
from enum import Enum


class ConstraintSeverity(str, Enum):
    """约束严重程度"""
    ERROR = "error"      # 硬约束违反
    WARNING = "warning"  # 软约束警告
    INFO = "info"        # 信息提示


class ConstraintViolation(BaseModel):
    """约束违反记录"""
    constraint_name: str
    severity: ConstraintSeverity
    message: str
    condition: str
    actual_value: Any | None = None


class ValidationResult(BaseModel):
    """验证结果"""
    is_valid: bool
    violations: list[ConstraintViolation] = Field(default_factory=list)
    warnings: list[ConstraintViolation] = Field(default_factory=list)
    info: list[ConstraintViolation] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.violations) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class ConstraintValidator:
    """
    约束验证器

    验证实验配置和结果是否符合 Protocol 定义
    """

    def __init__(self, protocol: dict[str, Any]):
        """
        初始化验证器

        Args:
            protocol: Protocol 定义字典
        """
        self.protocol = protocol
        self.hard_constraints = protocol.get("hard_constraints", [])
        self.soft_constraints = protocol.get("soft_constraints", [])

    def validate(
        self,
        parameters: dict[str, Any],
        metrics: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        验证实验配置和结果

        Args:
            parameters: 实验参数
            metrics: 计算指标 (可选)
            quality: 质量指标 (可选)

        Returns:
            ValidationResult 对象
        """
        result = ValidationResult(is_valid=True)

        # 构建验证上下文
        context = {
            "parameters": parameters,
            "metrics": metrics or {},
            "quality": quality or {}
        }

        # 验证硬约束
        for constraint in self.hard_constraints:
            violation = self._check_constraint(constraint, context, ConstraintSeverity.ERROR)
            if violation:
                result.violations.append(violation)
                result.is_valid = False

        # 验证软约束
        for constraint in self.soft_constraints:
            severity = ConstraintSeverity(constraint.get("severity", "warning"))
            violation = self._check_constraint(constraint, context, severity)
            if violation:
                if severity == ConstraintSeverity.WARNING:
                    result.warnings.append(violation)
                else:
                    result.info.append(violation)

        return result

    def validate_parameters(self, parameters: dict[str, Any]) -> ValidationResult:
        """
        仅验证参数配置

        Args:
            parameters: 实验参数

        Returns:
            ValidationResult 对象
        """
        return self.validate(parameters)

    def validate_results(
        self,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        quality: dict[str, Any]
    ) -> ValidationResult:
        """
        验证实验结果

        Args:
            parameters: 实验参数
            metrics: 计算指标
            quality: 质量指标

        Returns:
            ValidationResult 对象
        """
        return self.validate(parameters, metrics, quality)

    def _check_constraint(
        self,
        constraint: dict[str, Any],
        context: dict[str, Any],
        severity: ConstraintSeverity
    ) -> ConstraintViolation | None:
        """
        检查单个约束

        Args:
            constraint: 约束定义
            context: 验证上下文
            severity: 严重程度

        Returns:
            ConstraintViolation 如果违反，否则 None
        """
        condition = constraint.get("condition", "")
        name = constraint.get("name", "unknown")
        message = constraint.get("message", f"约束 {name} 未满足")

        try:
            # 安全评估条件表达式
            result = self._evaluate_condition(condition, context)
            if not result:
                return ConstraintViolation(
                    constraint_name=name,
                    severity=severity,
                    message=message,
                    condition=condition,
                    actual_value=self._extract_actual_value(condition, context)
                )
        except Exception as e:
            return ConstraintViolation(
                constraint_name=name,
                severity=severity,
                message=f"约束评估错误: {str(e)}",
                condition=condition
            )

        return None

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """
        安全评估条件表达式

        Args:
            condition: 条件表达式字符串
            context: 验证上下文

        Returns:
            条件结果
        """
        import re

        # 检查条件是否涉及 quality 或 metrics
        pattern = r"(quality|metrics)\.([a-zA-Z_][a-zA-Z0-9_.]*)"
        quality_metrics_matches = re.findall(pattern, condition)

        # 如果条件涉及 quality 或 metrics，但它们为空，跳过该约束
        for obj_name, _ in quality_metrics_matches:
            obj = context.get(obj_name, {})
            if not obj:  # 空字典表示数据未提供
                return True  # 跳过约束，视为满足

        # 使用受限的 eval 环境
        allowed_names = {
            "parameters": context.get("parameters", {}),
            "metrics": context.get("metrics", {}),
            "quality": context.get("quality", {}),
            "True": True,
            "False": False,
            "None": None,
        }

        # 解析嵌套属性访问 (如 parameters.duration)
        def get_nested_value(obj: dict, path: str) -> Any:
            parts = path.split(".")
            value = obj
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value

        # 替换条件中的属性访问
        attr_pattern = r"(parameters|metrics|quality)\.([a-zA-Z_][a-zA-Z0-9_.]*)"

        def replace_attr(match):
            obj_name = match.group(1)
            attr_path = match.group(2)
            value = get_nested_value(context.get(obj_name, {}), attr_path)
            return repr(value)

        replaced_condition = re.sub(attr_pattern, replace_attr, condition)

        # 安全评估
        try:
            return eval(replaced_condition, {"__builtins__": {}}, allowed_names)
        except:
            return False

    def _extract_actual_value(self, condition: str, context: dict[str, Any]) -> Any:
        """提取条件中涉及的实际值"""
        import re
        pattern = r"(parameters|metrics|quality)\.([a-zA-Z_][a-zA-Z0-9_.]*)"
        matches = re.findall(pattern, condition)

        values = {}
        for obj_name, attr_path in matches:
            obj = context.get(obj_name, {})
            parts = attr_path.split(".")
            value = obj
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            values[f"{obj_name}.{attr_path}"] = value

        return values if len(values) > 1 else list(values.values())[0] if values else None


def validate_experiment(
    protocol_name: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None
) -> ValidationResult:
    """
    便捷函数：验证实验

    Args:
        protocol_name: Protocol 名称
        parameters: 实验参数
        metrics: 计算指标
        quality: 质量指标

    Returns:
        ValidationResult 对象
    """
    from .protocols import load_protocol

    protocol = load_protocol(protocol_name)
    validator = ConstraintValidator(protocol)
    return validator.validate(parameters, metrics, quality)
