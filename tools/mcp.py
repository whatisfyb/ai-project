"""MCP 工具 - 管理 MCP 服务器和调用工具

提供以下工具：
- mcp_list_servers: 列出所有 MCP 服务器
- mcp_connect: 连接 MCP 服务器
- mcp_disconnect: 断开 MCP 服务器
- mcp_list_tools: 列出 MCP 服务器的工具
- mcp_call_tool: 调用 MCP 工具
"""

from typing import Any

from langchain_core.tools import tool

from utils.mcp_client import MCPManager, MCPClient


# 全局 MCP 管理器
_mcp_manager: MCPManager | None = None


def get_mcp_manager() -> MCPManager:
    """获取 MCP 管理器单例"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager


async def init_mcp_from_config_async(background: bool = True) -> dict:
    """从配置文件异步初始化 MCP 服务器（不阻塞主线程）

    Args:
        background: 是否后台连接（True 时不阻塞）

    Returns:
        初始化结果统计
    """
    from utils.config import get_settings_instance

    manager = get_mcp_manager()
    settings = get_settings_instance()

    for server_config in settings.mcp.servers:
        if server_config.get("enabled", True):
            await manager.add_server_async(
                name=server_config["name"],
                command=server_config.get("command"),
                args=server_config.get("args"),
                url=server_config.get("url"),
                env=server_config.get("env"),
                background=background,
            )

    if not background:
        # 等待所有连接完成
        return await manager.wait_for_connections()

    return {"status": "background", "servers": len(settings.mcp.servers)}


def init_mcp_from_config():
    """从配置文件初始化 MCP 服务器（同步，会阻塞）

    Deprecated: 使用 init_mcp_from_config_async 代替
    """
    from utils.config import get_settings_instance

    manager = get_mcp_manager()
    settings = get_settings_instance()

    for server_config in settings.mcp.servers:
        if server_config.get("enabled", True):
            try:
                manager.add_server(
                    name=server_config["name"],
                    command=server_config.get("command"),
                    args=server_config.get("args"),
                    url=server_config.get("url"),
                    env=server_config.get("env"),
                    auto_connect=True,
                )
            except Exception as e:
                import warnings
                warnings.warn(f"无法连接 MCP 服务器 {server_config['name']}: {e}")


# ============ 工具定义 ============

@tool
def mcp_list_servers() -> dict[str, Any]:
    """List all configured MCP servers and their status.

    Returns:
        Dictionary containing list of MCP servers with connection status
    """
    manager = get_mcp_manager()
    servers = manager.list_servers()

    return {
        "servers": servers,
        "total": len(servers),
    }


@tool
def mcp_connect(
    name: str,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Connect to an MCP server.

    Args:
        name: Server name (identifier)
        command: Command to run (for stdio mode), e.g., "npx"
        args: Command arguments, e.g., ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        url: Server URL (for SSE mode)

    Returns:
        Connection result with server info
    """
    manager = get_mcp_manager()

    try:
        client = manager.add_server(
            name=name,
            command=command,
            args=args,
            url=url,
            auto_connect=True,
        )

        return {
            "status": "connected",
            "name": name,
            "server_info": client.get_server_info(),
            "tools_count": len(client._tools),
        }
    except Exception as e:
        return {
            "status": "error",
            "name": name,
            "error": str(e),
        }


@tool
def mcp_disconnect(name: str) -> dict[str, Any]:
    """Disconnect from an MCP server.

    Args:
        name: Server name to disconnect

    Returns:
        Disconnection result
    """
    manager = get_mcp_manager()

    if manager.close_server(name):
        return {
            "status": "disconnected",
            "name": name,
        }
    else:
        return {
            "status": "error",
            "name": name,
            "error": f"Server '{name}' not found",
        }


@tool
def mcp_list_tools(server_name: str | None = None) -> dict[str, Any]:
    """List available tools from MCP server(s).

    Args:
        server_name: Specific server name, or None to list all tools

    Returns:
        Dictionary containing available tools
    """
    manager = get_mcp_manager()

    if server_name:
        client = manager.get_client(server_name)
        if not client:
            return {
                "status": "error",
                "error": f"Server '{server_name}' not found",
            }

        tools = client.list_tools()
        return {
            "server": server_name,
            "tools": tools,
            "count": len(tools),
        }
    else:
        all_tools = []
        for server in manager.list_servers():
            client = manager.get_client(server["name"])
            if client and client._initialized:
                tools = client.list_tools()
                for t in tools:
                    all_tools.append({
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "server": server["name"],
                    })

        return {
            "tools": all_tools,
            "count": len(all_tools),
        }


@tool
def mcp_call_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    server_name: str | None = None,
) -> dict[str, Any]:
    """Call a tool from an MCP server.

    Args:
        tool_name: Name of the tool to call
        arguments: Tool arguments as a dictionary
        server_name: Server name (optional, will search all servers if not specified)

    Returns:
        Tool execution result
    """
    manager = get_mcp_manager()

    # 如果指定了服务器
    if server_name:
        client = manager.get_client(server_name)
        if not client:
            return {
                "status": "error",
                "error": f"Server '{server_name}' not found",
            }

        try:
            result = client.call_tool(tool_name, arguments)
            return {
                "status": "success",
                "tool": tool_name,
                "server": server_name,
                "result": client.get_tool_result_text(result),
            }
        except Exception as e:
            return {
                "status": "error",
                "tool": tool_name,
                "server": server_name,
                "error": str(e),
            }

    # 搜索所有服务器
    for server in manager.list_servers():
        client = manager.get_client(server["name"])
        if client and client._initialized:
            # 检查是否有这个工具
            for t in client._tools:
                if t["name"] == tool_name:
                    try:
                        result = client.call_tool(tool_name, arguments)
                        return {
                            "status": "success",
                            "tool": tool_name,
                            "server": server["name"],
                            "result": client.get_tool_result_text(result),
                        }
                    except Exception as e:
                        return {
                            "status": "error",
                            "tool": tool_name,
                            "server": server["name"],
                            "error": str(e),
                        }

    return {
        "status": "error",
        "error": f"Tool '{tool_name}' not found in any connected server",
    }
