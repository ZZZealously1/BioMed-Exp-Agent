"""
MCP 工具模块
包装 YOLO/SORT/SAM 等CV能力为 MCP 标准工具
"""

from .server import create_mcp_server
from .detect import detect_tool, detect_mwm_mouse
from .track import track_tool
from .segment import segment_tool
from .calculate import calculate_tool
from .quality import QualityAssessor

__all__ = [
    "create_mcp_server",
    "detect_tool",
    "detect_mwm_mouse",
    "track_tool",
    "segment_tool",
    "calculate_tool",
    "QualityAssessor"
]
