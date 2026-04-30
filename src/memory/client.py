"""
Mem0 客户端
科学记忆扩展实现
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import os

# Mem0 集成（可选依赖）
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False


class MemoryType(str):
    """记忆类型"""
    WORKING = "working"      # 工作记忆：当前实验上下文
    SHORT_TERM = "short_term"  # 短期记忆：近期案例
    LONG_TERM = "long_term"   # 长期记忆：经验库与文献


class ExperimentMemory(BaseModel):
    """实验记忆"""
    experiment_id: str
    experiment_type: str
    species: str | None = None

    # 实验配置
    parameters: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)

    # 执行过程
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    repair_attempts: list[dict[str, Any]] = Field(default_factory=list)

    # 结果
    final_metrics: dict[str, float] | None = None
    quality_score: float | None = None
    success: bool = True

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)


class MemoryClient:
    """
    记忆客户端

    管理三层记忆：
    - 工作记忆：Redis (当前实验上下文)
    - 短期记忆：ChromaDB (近期案例检索)
    - 长期记忆：PostgreSQL + pgvector (经验库与文献)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        chroma_host: str = "localhost",
        chroma_port: int = 8001,
        postgres_url: str | None = None,
        mem0_api_key: str | None = None
    ):
        """
        初始化记忆客户端

        Args:
            redis_url: Redis 连接 URL
            chroma_host: ChromaDB 主机
            chroma_port: ChromaDB 端口
            postgres_url: PostgreSQL 连接 URL
            mem0_api_key: Mem0 API Key (可选)
        """
        self.redis_url = redis_url or os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.postgres_url = postgres_url or os.getenv("DATABASE_URL")

        # Mem0 客户端 (如果可用)
        self._mem0: Any | None = None
        if MEM0_AVAILABLE and mem0_api_key:
            self._mem0 = Memory(api_key=mem0_api_key)

        # Redis 客户端 (延迟初始化)
        self._redis: Any | None = None

        # ChromaDB 客户端 (延迟初始化)
        self._chroma: Any | None = None

    @property
    def redis(self):
        """获取 Redis 客户端"""
        if self._redis is None:
            import redis
            self._redis = redis.from_url(self.redis_url)
        return self._redis

    @property
    def chroma(self):
        """获取 ChromaDB 客户端"""
        if self._chroma is None:
            import chromadb
            self._chroma = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port
            )
        return self._chroma

    # ==================== 工作记忆 ====================

    def set_working_memory(
        self,
        experiment_id: str,
        context: dict[str, Any],
        ttl: int = 3600
    ) -> None:
        """
        设置工作记忆

        Args:
            experiment_id: 实验 ID
            context: 上下文数据
            ttl: 过期时间 (秒)
        """
        import json
        key = f"working_memory:{experiment_id}"
        self.redis.setex(key, ttl, json.dumps(context, default=str))

    def get_working_memory(self, experiment_id: str) -> dict[str, Any] | None:
        """
        获取工作记忆

        Args:
            experiment_id: 实验 ID

        Returns:
            上下文数据
        """
        import json
        key = f"working_memory:{experiment_id}"
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    def update_working_memory(
        self,
        experiment_id: str,
        updates: dict[str, Any]
    ) -> None:
        """
        更新工作记忆

        Args:
            experiment_id: 实验 ID
            updates: 更新数据
        """
        current = self.get_working_memory(experiment_id) or {}
        current.update(updates)
        self.set_working_memory(experiment_id, current)

    def clear_working_memory(self, experiment_id: str) -> None:
        """清除工作记忆"""
        key = f"working_memory:{experiment_id}"
        self.redis.delete(key)

    # ==================== 短期记忆 ====================

    def store_case(self, memory: ExperimentMemory) -> str:
        """
        存储案例到短期记忆

        Args:
            memory: 实验记忆

        Returns:
            案例 ID
        """
        collection = self.chroma.get_or_create_collection(
            name="experiment_cases",
            metadata={"hnsw:space": "cosine"}
        )

        # 生成案例描述文本
        description = self._generate_case_description(memory)

        collection.add(
            ids=[memory.experiment_id],
            documents=[description],
            metadatas=[{
                "experiment_type": memory.experiment_type,
                "species": memory.species or "unknown",
                "success": memory.success,
                "quality_score": memory.quality_score or 0,
                "created_at": memory.created_at.isoformat(),
                "tags": ",".join(memory.tags)
            }]
        )

        return memory.experiment_id

    def retrieve_similar_cases(
        self,
        query: str,
        experiment_type: str | None = None,
        n_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        检索相似案例

        Args:
            query: 查询文本
            experiment_type: 实验类型过滤
            n_results: 返回数量

        Returns:
            相似案例列表
        """
        collection = self.chroma.get_or_create_collection(
            name="experiment_cases"
        )

        where_filter = None
        if experiment_type:
            where_filter = {"experiment_type": experiment_type}

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        cases = []
        for i in range(len(results["ids"][0])):
            cases.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return cases

    def _generate_case_description(self, memory: ExperimentMemory) -> str:
        """生成案例描述文本"""
        parts = [
            f"实验类型: {memory.experiment_type}",
            f"物种: {memory.species or '未指定'}",
        ]

        if memory.parameters:
            parts.append(f"参数: {memory.parameters}")

        if memory.constraints:
            parts.append(f"约束: {memory.constraints}")

        if memory.repair_attempts:
            parts.append(f"修复尝试: {len(memory.repair_attempts)} 次")

        if memory.final_metrics:
            parts.append(f"结果: {memory.final_metrics}")

        if not memory.success:
            parts.append("状态: 失败")

        return " | ".join(parts)

    # ==================== 长期记忆 ====================

    def store_experience(
        self,
        experience: dict[str, Any],
        category: str = "general"
    ) -> str:
        """
        存储经验到长期记忆

        Args:
            experience: 经验数据
            category: 经验类别

        Returns:
            经验 ID
        """
        # TODO: 实现 PostgreSQL + pgvector 存储
        raise NotImplementedError("长期记忆存储待实现")

    def retrieve_experiences(
        self,
        query: str,
        category: str | None = None,
        n_results: int = 10
    ) -> list[dict[str, Any]]:
        """
        检索相关经验

        Args:
            query: 查询文本
            category: 经验类别过滤
            n_results: 返回数量

        Returns:
            经验列表
        """
        # TODO: 实现 PostgreSQL + pgvector 检索
        raise NotImplementedError("长期记忆检索待实现")

    # ==================== Mem0 集成 ====================

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None
    ) -> str | None:
        """
        使用 Mem0 添加记忆

        Args:
            content: 记忆内容
            user_id: 用户 ID
            metadata: 元数据

        Returns:
            记忆 ID
        """
        if not self._mem0:
            return None

        result = self._mem0.add(
            content,
            user_id=user_id,
            metadata=metadata
        )
        return result.get("id")

    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        使用 Mem0 搜索记忆

        Args:
            query: 查询文本
            user_id: 用户 ID
            limit: 返回数量

        Returns:
            记忆列表
        """
        if not self._mem0:
            return []

        return self._mem0.search(
            query,
            user_id=user_id,
            limit=limit
        )

    # ==================== 经验提取 ====================

    def extract_experience(self, memory: ExperimentMemory) -> dict[str, Any]:
        """
        从实验记忆中提取可复用的经验

        Args:
            memory: 实验记忆

        Returns:
            提取的经验
        """
        experience = {
            "experiment_type": memory.experiment_type,
            "species": memory.species,
            "successful_strategies": [],
            "failed_strategies": [],
            "parameter_insights": {},
            "quality_patterns": {}
        }

        # 分析成功的修复策略
        for attempt in memory.repair_attempts:
            if attempt.get("success"):
                experience["successful_strategies"].append({
                    "trigger": attempt.get("trigger"),
                    "action": attempt.get("action"),
                    "improvement": attempt.get("improvement")
                })
            else:
                experience["failed_strategies"].append({
                    "trigger": attempt.get("trigger"),
                    "action": attempt.get("action")
                })

        # 分析参数影响
        if memory.parameters and memory.quality_score:
            experience["parameter_insights"] = {
                k: v for k, v in memory.parameters.items()
                if isinstance(v, (int, float))
            }
            experience["quality_patterns"]["score"] = memory.quality_score

        return experience
