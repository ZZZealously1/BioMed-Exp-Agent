"""
WebSocket 路由
实时推送实验进度和状态
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Any
import json
import asyncio

router = APIRouter()


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        # experiment_id -> list of WebSocket
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: str):
        """接受新连接"""
        await websocket.accept()
        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = []
        self.active_connections[experiment_id].append(websocket)

    def disconnect(self, websocket: WebSocket, experiment_id: str):
        """断开连接"""
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].remove(websocket)
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]

    async def send_message(self, experiment_id: str, message: dict[str, Any]):
        """发送消息到指定实验的所有连接"""
        if experiment_id in self.active_connections:
            for connection in self.active_connections[experiment_id]:
                await connection.send_json(message)

    async def broadcast(self, message: dict[str, Any]):
        """广播消息到所有连接"""
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_json(message)


manager = ConnectionManager()


@router.websocket("/{experiment_id}")
async def experiment_websocket(websocket: WebSocket, experiment_id: str):
    """
    实验进度 WebSocket

    发送消息格式:
    {
        "type": "progress" | "status" | "result" | "error",
        "data": {...}
    }
    """
    await manager.connect(websocket, experiment_id)

    try:
        # 发送连接成功消息
        await websocket.send_json({
            "type": "connected",
            "data": {
                "experiment_id": experiment_id,
                "message": "WebSocket 连接成功"
            }
        })

        while True:
            # 接收客户端消息
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                # 处理客户端请求
                response = await handle_client_message(experiment_id, message)
                if response:
                    await websocket.send_json(response)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "无效的 JSON 格式"}
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)


async def handle_client_message(
    experiment_id: str,
    message: dict[str, Any]
) -> dict[str, Any] | None:
    """
    处理客户端消息

    Args:
        experiment_id: 实验 ID
        message: 客户端消息

    Returns:
        响应消息
    """
    msg_type = message.get("type")

    if msg_type == "ping":
        return {"type": "pong", "data": {"timestamp": message.get("timestamp")}}

    elif msg_type == "subscribe":
        # 订阅实验更新
        return {
            "type": "subscribed",
            "data": {"experiment_id": experiment_id}
        }

    elif msg_type == "get_status":
        # 获取当前状态
        # TODO: 从存储中获取实际状态
        return {
            "type": "status",
            "data": {
                "experiment_id": experiment_id,
                "status": "processing",
                "progress": 0.5
            }
        }

    return None


async def send_progress_update(
    experiment_id: str,
    step: str,
    progress: float,
    details: dict[str, Any] | None = None
):
    """
    发送进度更新

    Args:
        experiment_id: 实验 ID
        step: 当前步骤
        progress: 进度 (0-1)
        details: 详细信息
    """
    await manager.send_message(experiment_id, {
        "type": "progress",
        "data": {
            "experiment_id": experiment_id,
            "step": step,
            "progress": progress,
            "details": details
        }
    })


async def send_status_update(
    experiment_id: str,
    status: str,
    message: str | None = None
):
    """
    发送状态更新

    Args:
        experiment_id: 实验 ID
        status: 新状态
        message: 状态消息
    """
    await manager.send_message(experiment_id, {
        "type": "status",
        "data": {
            "experiment_id": experiment_id,
            "status": status,
            "message": message
        }
    })


async def send_result(
    experiment_id: str,
    metrics: dict[str, Any],
    quality_score: float,
    report_url: str | None = None
):
    """
    发送最终结果

    Args:
        experiment_id: 实验 ID
        metrics: 计算指标
        quality_score: 质量评分
        report_url: 报告 URL
    """
    await manager.send_message(experiment_id, {
        "type": "result",
        "data": {
            "experiment_id": experiment_id,
            "metrics": metrics,
            "quality_score": quality_score,
            "report_url": report_url
        }
    })


async def send_error(
    experiment_id: str,
    error_message: str,
    error_details: dict[str, Any] | None = None
):
    """
    发送错误消息

    Args:
        experiment_id: 实验 ID
        error_message: 错误消息
        error_details: 错误详情
    """
    await manager.send_message(experiment_id, {
        "type": "error",
        "data": {
            "experiment_id": experiment_id,
            "message": error_message,
            "details": error_details
        }
    })


async def request_user_confirmation(
    experiment_id: str,
    prompt: str,
    options: list[str] | None = None
):
    """
    请求用户确认

    Args:
        experiment_id: 实验 ID
        prompt: 提示信息
        options: 选项列表
    """
    await manager.send_message(experiment_id, {
        "type": "confirmation_required",
        "data": {
            "experiment_id": experiment_id,
            "prompt": prompt,
            "options": options
        }
    })
