"""
FastAPI 服务模块
提供 REST API 和 WebSocket 接口
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
