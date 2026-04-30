# API 参考

## 基础 URL

```
http://localhost:8000
```

## 认证

API 使用 JWT Token 认证（待实现）：

```http
Authorization: Bearer <token>
```

---

## 实验端点

### 创建实验

```http
POST /api/experiments/
```

**请求体：**

```json
{
    "user_request": "string",
    "video_path": "string (可选)",
    "experiment_type": "string (可选)",
    "species": "string (可选)",
    "parameters": { } (可选)
}
```

**响应：**

```json
{
    "experiment_id": "uuid",
    "status": "pending",
    "message": "string",
    "created_at": "datetime"
}
```

---

### 上传视频

```http
POST /api/experiments/{experiment_id}/upload
```

**请求：** multipart/form-data

| 字段 | 类型 | 描述 |
|------|------|------|
| file | File | 视频文件 |

**响应：**

```json
{
    "experiment_id": "uuid",
    "status": "processing",
    "current_step": "string",
    "progress": 0.1
}
```

---

### 获取实验状态

```http
GET /api/experiments/{experiment_id}
```

**响应：**

```json
{
    "experiment_id": "uuid",
    "status": "string",
    "current_step": "string",
    "progress": 0.5,
    "metrics": { },
    "error": "string (可选)"
}
```

---

### 获取实验结果

```http
GET /api/experiments/{experiment_id}/result
```

**响应：**

```json
{
    "experiment_id": "uuid",
    "status": "completed",
    "metrics": {
        "center_time_percent": 25.5,
        "total_distance": 4500.0,
        "avg_speed": 7.5
    },
    "quality_score": 0.92,
    "report_url": "string (可选)",
    "audit_log_url": "string (可选)",
    "completed_at": "datetime"
}
```

---

### 取消实验

```http
DELETE /api/experiments/{experiment_id}
```

**响应：**

```json
{
    "message": "实验已取消",
    "experiment_id": "uuid"
}
```

---

### 列出实验

```http
GET /api/experiments/?status=string&limit=20&offset=0
```

**查询参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| status | string | null | 按状态过滤 |
| limit | int | 20 | 返回数量 |
| offset | int | 0 | 偏移量 |

**响应：**

```json
{
    "total": 100,
    "experiments": [
        {
            "experiment_id": "uuid",
            "status": "string",
            "experiment_type": "string",
            "created_at": "datetime"
        }
    ]
}
```

---

## WebSocket 端点

### 实验进度订阅

```
WS /ws/{experiment_id}
```

**消息格式：**

```json
{
    "type": "progress | status | result | error",
    "data": { }
}
```

**客户端消息：**

```json
{
    "type": "ping | subscribe | get_status"
}
```

---

## 错误响应

所有错误响应格式：

```json
{
    "detail": "错误描述"
}
```

**常见状态码：**

| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 201 | 已创建 |
| 400 | 请求错误 |
| 404 | 未找到 |
| 500 | 服务器错误 |
