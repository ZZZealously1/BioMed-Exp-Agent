#!/bin/bash

# 运行 MCP Inspector 测试工具

set -e

echo "🔍 启动 MCP Inspector..."

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 安装 MCP Inspector (如果未安装)
if ! command -v mcp-inspector &> /dev/null; then
    echo "📦 安装 MCP Inspector..."
    pip install mcp
fi

# 运行 MCP 服务器
echo "🚀 启动 MCP 服务器..."
python -m src.tools.server &

SERVER_PID=$!

# 等待服务器启动
sleep 2

# 运行 Inspector
echo "🔍 运行 MCP Inspector..."
mcp inspect src.tools.server

# 清理
kill $SERVER_PID 2>/dev/null || true

echo "✅ MCP Inspector 测试完成"
