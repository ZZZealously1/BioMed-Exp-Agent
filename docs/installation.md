# 安装指南

## 系统要求

- Python 3.10+
- uv (推荐) 或 pip
- Docker & Docker Compose
- CUDA 11.8+ (可选，用于 GPU 加速)

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd biomed-exp-agent
```

### 2. 安装 uv (推荐)

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -e ".[dev]"
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置 GLM API：

```bash
# LLM 提供商选择
LLM_PROVIDER=glm

# GLM (智谱 AI) 配置
GLM_API_KEY=your-glm-api-key-here
GLM_MODEL=glm-4-flash
```

获取 GLM API Key: https://open.bigmodel.cn/

### 5. 启动基础设施

```bash
docker-compose up -d
```

### 6. 验证安装

```bash
# 运行测试
uv run pytest tests/

# 启动 API 服务
uv run uvicorn src.api.main:app --reload
```

### 5. 启动基础设施

```bash
docker-compose up -d
```

### 6. 验证安装

```bash
# 运行测试
pytest tests/

# 启动 API 服务
uvicorn src.api.main:app --reload
```

## Docker 部署

### 开发环境

```bash
docker-compose up -d
```

### 生产环境

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 常见问题

### Q: 依赖安装失败？

尝试升级 pip 和 setuptools：

```bash
pip install --upgrade pip setuptools wheel
```

### Q: Docker 服务启动失败？

检查端口是否被占用：

```bash
# Linux/Mac
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :8001  # ChromaDB
```

### Q: GPU 加速不工作？

确保安装了正确的 CUDA 驱动和 PyTorch CUDA 版本：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
