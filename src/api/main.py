"""
FastAPI 主入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from .routes import experiments, websocket


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("[INFO] BioMed-Exp Agent 启动中...")

    # 初始化数据库连接
    # TODO: 初始化数据库

    yield

    # 关闭时
    print("[INFO] BioMed-Exp Agent 关闭中...")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="BioMed-Exp Agent API",
        description="生物医学实验智能体系统 API",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境需要限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(
        experiments.router,
        prefix="/api/experiments",
        tags=["experiments"]
    )
    app.include_router(
        websocket.router,
        prefix="/ws",
        tags=["websocket"]
    )

    return app


# 默认应用实例
app = create_app()


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "BioMed-Exp Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


def run_cli():
    """CLI 入口"""
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run_cli()
