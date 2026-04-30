#!/bin/bash

# BioMed-Exp Agent 开发环境设置脚本

set -e

echo "🚀 设置 BioMed-Exp Agent 开发环境..."

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 版本需要 >= 3.10，当前版本: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python 版本: $PYTHON_VERSION"

# 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv .venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source .venv/bin/activate

# 安装依赖
echo "📥 安装依赖..."
pip install --upgrade pip
pip install -e ".[dev]"

# 复制环境变量配置
if [ ! -f ".env" ]; then
    echo "📝 复制环境变量配置..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件填入 API Keys"
fi

# 启动 Docker 服务
echo "🐳 启动 Docker 服务..."
docker-compose up -d

# 等待服务就绪
echo "⏳ 等待服务就绪..."
sleep 10

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 运行测试
echo "🧪 运行测试..."
pytest tests/ -v --tb=short || true

echo ""
echo "✅ 开发环境设置完成！"
echo ""
echo "下一步："
echo "  1. 编辑 .env 文件填入 API Keys"
echo "  2. 运行 'source .venv/bin/activate' 激活环境"
echo "  3. 运行 'uvicorn src.api.main:app --reload' 启动 API 服务"
echo ""
