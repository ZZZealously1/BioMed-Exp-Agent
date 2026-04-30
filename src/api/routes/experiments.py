"""
实验相关 API 路由
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime
import uuid
import os

router = APIRouter()


class ExperimentCreate(BaseModel):
    """创建实验请求"""
    user_request: str
    video_path: Optional[str] = None
    experiment_type: Optional[str] = None
    species: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ExperimentResponse(BaseModel):
    """实验响应"""
    experiment_id: str
    status: str
    message: str
    created_at: datetime


class ExperimentStatus(BaseModel):
    """实验状态"""
    experiment_id: str
    status: str
    current_step: str
    progress: float
    metrics: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class ExperimentResult(BaseModel):
    """实验结果"""
    experiment_id: str
    status: str
    metrics: dict[str, Any]
    quality_score: float
    report_url: Optional[str] = None
    audit_log_url: Optional[str] = None
    completed_at: datetime


# 存储实验状态 (生产环境应使用数据库)
_experiments: dict[str, dict[str, Any]] = {}


@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreate,
    background_tasks: BackgroundTasks
):
    """
    创建新实验

    - **user_request**: 用户的自然语言请求
    - **video_path**: 视频文件路径 (可选，可后续上传)
    - **experiment_type**: 实验类型 (可选，自动识别)
    - **species**: 物种 (可选，自动识别)
    """
    experiment_id = str(uuid.uuid4())

    # 创建实验记录
    experiment = {
        "experiment_id": experiment_id,
        "user_request": request.user_request,
        "video_path": request.video_path,
        "experiment_type": request.experiment_type,
        "species": request.species,
        "parameters": request.parameters or {},
        "status": "pending",
        "created_at": datetime.now(),
        "current_step": "初始化",
        "progress": 0.0
    }

    _experiments[experiment_id] = experiment

    # 如果有视频路径，启动后台处理
    if request.video_path:
        background_tasks.add_task(
            process_experiment,
            experiment_id,
            request.user_request,
            request.video_path
        )

    return ExperimentResponse(
        experiment_id=experiment_id,
        status="pending",
        message="实验已创建，等待处理",
        created_at=experiment["created_at"]
    )


@router.post("/{experiment_id}/upload", response_model=ExperimentStatus)
async def upload_video(
    experiment_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    上传实验视频

    - **experiment_id**: 实验 ID
    - **file**: 视频文件
    """
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="实验不存在")

    experiment = _experiments[experiment_id]

    # 保存视频文件
    upload_dir = os.getenv("UPLOAD_DIR", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, f"{experiment_id}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    experiment["video_path"] = file_path
    experiment["status"] = "processing"

    # 启动后台处理
    if background_tasks:
        background_tasks.add_task(
            process_experiment,
            experiment_id,
            experiment["user_request"],
            file_path
        )

    return ExperimentStatus(
        experiment_id=experiment_id,
        status="processing",
        current_step="视频上传完成",
        progress=0.1
    )


@router.get("/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment_status(experiment_id: str):
    """
    获取实验状态

    - **experiment_id**: 实验 ID
    """
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="实验不存在")

    experiment = _experiments[experiment_id]

    return ExperimentStatus(
        experiment_id=experiment_id,
        status=experiment["status"],
        current_step=experiment.get("current_step", ""),
        progress=experiment.get("progress", 0.0),
        metrics=experiment.get("metrics"),
        error=experiment.get("error")
    )


@router.get("/{experiment_id}/result", response_model=ExperimentResult)
async def get_experiment_result(experiment_id: str):
    """
    获取实验结果

    - **experiment_id**: 实验 ID
    """
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="实验不存在")

    experiment = _experiments[experiment_id]

    if experiment["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"实验尚未完成，当前状态: {experiment['status']}"
        )

    return ExperimentResult(
        experiment_id=experiment_id,
        status=experiment["status"],
        metrics=experiment.get("metrics", {}),
        quality_score=experiment.get("quality_score", 0.0),
        report_url=experiment.get("report_url"),
        audit_log_url=experiment.get("audit_log_url"),
        completed_at=experiment.get("completed_at", datetime.now())
    )


@router.delete("/{experiment_id}")
async def cancel_experiment(experiment_id: str):
    """
    取消实验

    - **experiment_id**: 实验 ID
    """
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail="实验不存在")

    experiment = _experiments[experiment_id]

    if experiment["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"无法取消，实验状态: {experiment['status']}"
        )

    experiment["status"] = "cancelled"

    return {"message": "实验已取消", "experiment_id": experiment_id}


@router.get("/")
async def list_experiments(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """
    列出实验

    - **status**: 按状态过滤 (可选)
    - **limit**: 返回数量限制
    - **offset**: 偏移量
    """
    experiments = list(_experiments.values())

    if status:
        experiments = [e for e in experiments if e["status"] == status]

    total = len(experiments)
    experiments = experiments[offset:offset + limit]

    return {
        "total": total,
        "experiments": [
            {
                "experiment_id": e["experiment_id"],
                "status": e["status"],
                "experiment_type": e.get("experiment_type"),
                "created_at": e["created_at"]
            }
            for e in experiments
        ]
    }


def process_experiment(
    experiment_id: str,
    user_request: str,
    video_path: str
):
    """
    后台处理实验 - 调用真实的 Agent 工作流 (同步版本)
    """
    import asyncio

    experiment = _experiments.get(experiment_id)
    if not experiment:
        return

    try:
        # 更新状态
        experiment["status"] = "processing"
        experiment["current_step"] = "初始化"
        experiment["progress"] = 0.0

        # 导入 Agent 组件
        from ...agent.state import ExperimentState, VideoMetadata
        from ...agent.nodes.plan import plan_node
        from ...agent.nodes.execute import execute_node
        from ...agent.nodes.reflect import reflect_node
        import cv2

        # Step 1: 感知 - 提取视频信息
        experiment["current_step"] = "感知"
        experiment["progress"] = 0.1

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        video_metadata = VideoMetadata(
            path=video_path,
            duration=frame_count / fps if fps > 0 else 0,
            fps=fps,
            width=width,
            height=height,
            brightness=65.0
        )

        # 确定实验类型
        experiment_type = experiment.get("experiment_type") or "open_field"
        species = experiment.get("species") or "mouse"

        # 创建初始状态
        state = ExperimentState(
            user_request=user_request,
            video_path=video_path,
            experiment_type=experiment_type,
            species=species,
            video_metadata=video_metadata,
            current_plan=[],
            current_step=0,
            tool_results=[]
        )

        # Step 2: 规划
        experiment["current_step"] = "规划"
        experiment["progress"] = 0.2

        plan_updates = plan_node(state)
        state = ExperimentState(**{**state.model_dump(), **plan_updates})

        # Step 3: 执行工作流
        experiment["current_step"] = "执行"
        total_steps = len(state.current_plan)

        while not state.is_complete:
            step_progress = 0.3 + (state.current_step / total_steps) * 0.6

            # 更新进度
            current_action = state.current_plan[state.current_step] if state.current_step < len(state.current_plan) else "完成"
            experiment["current_step"] = f"执行: {current_action}"
            experiment["progress"] = round(step_progress, 2)

            # 执行一步
            result = execute_node(state)
            state = ExperimentState(**{**state.model_dump(), **result})

        # Step 4: 反思
        experiment["current_step"] = "反思"
        experiment["progress"] = 0.95

        reflect_updates = reflect_node(state)
        state = ExperimentState(**{**state.model_dump(), **reflect_updates})

        # 完成
        experiment["status"] = "completed"
        experiment["progress"] = 1.0
        experiment["current_step"] = "完成"

        # 提取最终结果（从指标计算工具的结果中提取，而非最后一个可视化工具）
        metric_tools = {
            "calculate_open_field_metrics",
            "calculate_water_maze_metrics",
            "calculate_epm_metrics",
            "calculate_worm_metrics",
            "calculate_zebrafish_metrics",
        }
        metric_result = None
        for result in reversed(state.tool_results):
            if result.tool_name in metric_tools and result.success and result.output:
                metric_result = result
                break

        if metric_result and metric_result.output:
            experiment["metrics"] = metric_result.output.get("metrics", {})
            experiment["interpretation"] = metric_result.output.get("interpretation", {})

        if state.quality_metrics:
            experiment["quality_score"] = (
                state.quality_metrics.detection_rate * 0.5 +
                state.quality_metrics.track_continuity * 0.5
            )
        else:
            experiment["quality_score"] = 0.9

        experiment["completed_at"] = datetime.now()

        # 保存轨迹数据
        experiment["trajectory_data"] = _extract_trajectory_data(state)

        # 提取报告路径
        for result in reversed(state.tool_results):
            if result.tool_name == "generate_report" and result.success and result.output:
                experiment["report_url"] = result.output.get("report_path")
                experiment["report_content"] = result.output.get("report_content", "")
                break

    except Exception as e:
        experiment["status"] = "failed"
        experiment["error"] = str(e)
        experiment["progress"] = 0.0


async def _broadcast_progress(experiment_id: str, state):
    """通过 WebSocket 广播进度"""
    from ..routes.websocket import manager

    message = {
        "type": "progress",
        "experiment_id": experiment_id,
        "step": state.current_step,
        "total_steps": len(state.current_plan),
        "current_action": state.current_plan[state.current_step] if state.current_step < len(state.current_plan) else "完成",
        "is_complete": state.is_complete
    }

    await manager.broadcast(message)


def _extract_trajectory_data(state) -> dict:
    """提取轨迹数据用于可视化"""
    for result in state.tool_results:
        if result.tool_name == "track" and result.output:
            track_history = result.output.get("track_history", {})
            if track_history:
                # 简化轨迹数据
                trajectories = []
                for tid, positions in track_history.items():
                    trajectory = {
                        "track_id": tid,
                        "positions": [
                            {"x": p["x"] + p["w"]/2, "y": p["y"] + p["h"]/2}
                            for p in positions
                        ]
                    }
                    trajectories.append(trajectory)
                return {"trajectories": trajectories}
    return {}
