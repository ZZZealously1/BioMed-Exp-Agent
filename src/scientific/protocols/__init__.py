"""
Protocol 本体定义
YAML 格式的实验协议定义
"""

from pathlib import Path
import yaml
from typing import Any


PROTOCOLS_DIR = Path(__file__).parent


def load_protocol(name: str) -> dict[str, Any]:
    """
    加载指定的 Protocol

    Args:
        name: Protocol 名称 (如 "epm", "open_field", "epm_v1.0")

    Returns:
        Protocol 定义字典
    """
    # 先尝试精确匹配
    protocol_path = PROTOCOLS_DIR / f"{name}.yaml"

    if not protocol_path.exists():
        # 尝试前缀匹配 (如 "epm" 匹配 "epm_v1.0.yaml")
        matches = sorted(PROTOCOLS_DIR.glob(f"{name}_*.yaml"))
        if matches:
            protocol_path = matches[-1]  # 取最新版本
        else:
            raise FileNotFoundError(f"Protocol 不存在: {name}")

    with open(protocol_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_protocols() -> list[str]:
    """
    列出所有可用的 Protocol

    Returns:
        Protocol 名称列表
    """
    return [p.stem for p in PROTOCOLS_DIR.glob("*.yaml")]


def get_protocol_metadata(name: str) -> dict[str, Any]:
    """
    获取 Protocol 元数据

    Args:
        name: Protocol 名称

    Returns:
        元数据字典
    """
    protocol = load_protocol(name)
    return {
        "name": protocol.get("name"),
        "version": protocol.get("version"),
        "description": protocol.get("description"),
        "species": protocol.get("species", []),
        "metrics": list(protocol.get("metrics", {}).keys())
    }
