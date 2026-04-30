"""
MCP 服务器入口
定义和启动 MCP 工具服务器
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .detect import detect_tool, detect_handler
from .track import track_tool, track_handler
from .segment import segment_tool, segment_handler
from .calculate import calculate_tool, calculate_handler


# 创建 MCP 服务器实例
server = Server("biomed-tools")


@server.list_tools()
async def list_tools():
    """列出所有可用工具"""
    return [
        detect_tool,
        track_tool,
        segment_tool,
        calculate_tool
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """调用工具"""
    handlers = {
        "detect": detect_handler,
        "track": track_handler,
        "segment": segment_handler,
        "calculate": calculate_handler
    }

    handler = handlers.get(name)
    if not handler:
        return [TextContent(
            type="text",
            text=f"错误: 未知工具 '{name}'"
        )]

    try:
        result = await handler(arguments)
        return [TextContent(
            type="text",
            text=result.model_dump_json()
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"工具执行错误: {str(e)}"
        )]


def create_mcp_server():
    """创建并返回 MCP 服务器"""
    return server


async def run_server():
    """运行 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server())
