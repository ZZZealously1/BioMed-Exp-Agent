# 快速开始

## 基本使用

### 1. 启动服务

```bash
# 启动 API 服务
uvicorn src.api.main:app --reload

# 或启动 Web UI
gradio src/ui/app.py
```

### 2. 创建实验

通过 API 创建实验：

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/experiments/",
    json={
        "user_request": "分析这只小鼠在旷场中的焦虑行为",
        "video_path": "/path/to/video.mp4"
    }
)

experiment_id = response.json()["experiment_id"]
print(f"实验 ID: {experiment_id}")
```

### 3. 查询状态

```python
response = httpx.get(
    f"http://localhost:8000/api/experiments/{experiment_id}"
)

status = response.json()
print(f"状态: {status['status']}")
print(f"进度: {status['progress'] * 100}%")
```

### 4. 获取结果

```python
response = httpx.get(
    f"http://localhost:8000/api/experiments/{experiment_id}/result"
)

result = response.json()
print(f"指标: {result['metrics']}")
print(f"质量评分: {result['quality_score']}")
```

## 使用 Python SDK

```python
from src.agent import create_experiment_graph, ExperimentState

# 创建状态
state = ExperimentState(
    user_request="分析小鼠焦虑行为",
    video_path="/path/to/video.mp4"
)

# 运行工作流
graph = create_experiment_graph()
result = await graph.ainvoke(state)

# 查看结果
print(result.metrics)
```

## WebSocket 实时更新

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/experiment_id');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'progress') {
        console.log(`进度: ${data.data.progress * 100}%`);
    } else if (data.type === 'result') {
        console.log('结果:', data.data.metrics);
    }
};
```

## 下一步

- 查看 [API 参考](api_reference.md) 了解完整的 API 文档
- 阅读 [工具开发指南](../notebooks/02_tool_development.ipynb) 学习如何开发自定义工具
