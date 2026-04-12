"""MCP Client - 连接外部 MCP Server

使用官方 mcp SDK 实现 MCP (Model Context Protocol) 客户端，支持：
- stdio 传输（进程管道）
- SSE 传输（HTTP 长连接）
- 工具发现与调用
- 转换为 LangChain 工具
- 异步连接（不阻塞主线程）
"""

import asyncio
import json
import logging
import os
from typing import Any

from langchain_core.tools import Tool

# 抑制 MCP SDK 的 JSON-RPC 解析错误日志（Windows 下 npx 启动时会输出编码信息）
logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)


# ============ 异常类 ============

class MCPError(Exception):
    """MCP 错误"""
    pass


class MCPConnectionError(MCPError):
    """连接错误"""
    pass


class MCPToolError(MCPError):
    """工具调用错误"""
    pass


# ============ MCP Client ============

class MCPClient:
    """MCP Client - 连接外部 MCP Server 并使用其工具

    使用官方 mcp SDK 实现，每次操作创建新连接，适用于工具调用场景。
    支持异步连接，不阻塞主线程。
    """

    def __init__(
        self,
        name: str,
        command: str = None,
        args: list[str] = None,
        url: str = None,
        env: dict = None,
    ):
        """
        Args:
            name: 服务器名称（用于标识）
            command: stdio 模式的命令（如 "npx"）
            args: 命令参数
            url: SSE 模式的 URL
            env: 环境变量
        """
        self.name = name
        self._command = command
        self._args = args or []
        self._url = url
        self._env = env
        self._tools: list[dict] = []
        self._initialized = False
        self._server_info: dict = {}
        self._connecting = False
        self._connect_error: str | None = None

    async def connect_async(self) -> dict:
        """异步连接并初始化

        Returns:
            服务器信息
        """
        self._connecting = True
        self._connect_error = None

        try:
            if self._command:
                # stdio 模式
                from mcp import ClientSession
                from mcp.client.stdio import StdioServerParameters, stdio_client

                env = os.environ.copy()
                if self._env:
                    env.update(self._env)

                server_params = StdioServerParameters(
                    command=self._command,
                    args=self._args,
                    env=env,
                )

                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        result = await session.initialize()
                        self._server_info = {
                            "name": result.serverInfo.name,
                            "version": result.serverInfo.version,
                            "capabilities": result.capabilities.model_dump() if result.capabilities else {},
                        }

                        # 获取工具列表
                        tools_result = await session.list_tools()
                        self._tools = [
                            {
                                "name": t.name,
                                "description": t.description or "",
                                "inputSchema": t.inputSchema if isinstance(t.inputSchema, dict) else (t.inputSchema.model_dump() if t.inputSchema else {}),
                            }
                            for t in tools_result.tools
                        ]
                        self._initialized = True
                        return self._server_info

            elif self._url:
                # SSE 模式
                from mcp import ClientSession
                from mcp.client.sse import sse_client

                async with sse_client(self._url) as (read, write):
                    async with ClientSession(read, write) as session:
                        result = await session.initialize()
                        self._server_info = {
                            "name": result.serverInfo.name,
                            "version": result.serverInfo.version,
                            "capabilities": result.capabilities.model_dump() if result.capabilities else {},
                        }

                        tools_result = await session.list_tools()
                        self._tools = [
                            {
                                "name": t.name,
                                "description": t.description or "",
                                "inputSchema": t.inputSchema if isinstance(t.inputSchema, dict) else (t.inputSchema.model_dump() if t.inputSchema else {}),
                            }
                            for t in tools_result.tools
                        ]
                        self._initialized = True
                        return self._server_info

            else:
                raise MCPConnectionError("必须提供 command 或 url")

        except Exception as e:
            self._connect_error = str(e)
            raise

        finally:
            self._connecting = False

    def connect(self) -> dict:
        """同步连接（向后兼容）

        Returns:
            服务器信息
        """
        return asyncio.run(self.connect_async())

    def list_tools(self) -> list[dict]:
        """获取工具列表（返回 connect 时缓存的结果）"""
        return self._tools

    async def call_tool_async(self, name: str, arguments: dict = None) -> dict:
        """异步调用工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        if self._command:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            env = os.environ.copy()
            if self._env:
                env.update(self._env)

            server_params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=env,
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments or {})
                    return {
                        "content": result.content,
                        "isError": result.isError or False,
                    }

        elif self._url:
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            async with sse_client(self._url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments or {})
                    return {
                        "content": result.content,
                        "isError": result.isError or False,
                    }

        raise MCPConnectionError("必须提供 command 或 url")

    def call_tool(self, name: str, arguments: dict = None) -> dict:
        """同步调用工具（向后兼容）

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        return asyncio.run(self.call_tool_async(name, arguments))

    def get_tool_result_text(self, result: dict) -> str:
        """从结果中提取文本内容"""
        content = result.get("content", [])
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts)

        return str(content)

    def close(self):
        """关闭连接（无状态，不需要操作）"""
        self._initialized = False

    def get_server_info(self) -> dict:
        """获取服务器信息"""
        return self._server_info.copy()

    @property
    def is_connecting(self) -> bool:
        """是否正在连接中"""
        return self._connecting

    @property
    def connect_error(self) -> str | None:
        """连接错误信息"""
        return self._connect_error


# ============ LangChain 工具转换 ============

def mcp_tool_to_langchain(client: MCPClient, tool_info: dict) -> Tool:
    """将 MCP 工具转换为 LangChain Tool

    Args:
        client: MCP Client 实例
        tool_info: MCP 工具信息

    Returns:
        LangChain Tool 实例
    """
    tool_name = tool_info["name"]
    tool_desc = tool_info.get("description", "")
    input_schema = tool_info.get("inputSchema", {})

    def invoke(tool_input: str | dict) -> str:
        # 处理输入
        if isinstance(tool_input, str):
            try:
                arguments = json.loads(tool_input)
            except json.JSONDecodeError:
                arguments = {"input": tool_input}
        else:
            arguments = tool_input

        # 调用工具
        result = client.call_tool(tool_name, arguments)
        return client.get_tool_result_text(result)

    # 构建描述（包含参数说明）
    full_description = tool_desc
    if input_schema:
        props = input_schema.get("properties", {})
        if props:
            param_desc = "\n\n参数:"
            for prop_name, prop_info in props.items():
                param_type = prop_info.get("type", "any")
                param_desc_text = prop_info.get("description", "")
                required = prop_name in input_schema.get("required", [])
                req_mark = "*" if required else ""
                param_desc += f"\n  - {prop_name}{req_mark} ({param_type}): {param_desc_text}"
            full_description += param_desc

    return Tool(
        name=tool_name,
        description=full_description,
        func=invoke,
    )


def convert_all_tools(client: MCPClient) -> list[Tool]:
    """转换 MCP 服务器的所有工具为 LangChain Tools"""
    tools_info = client.list_tools()
    return [mcp_tool_to_langchain(client, t) for t in tools_info]


# ============ MCP 管理器 ============

class MCPManager:
    """MCP 服务器管理器 - 管理多个 MCP 连接

    支持异步初始化，不阻塞主线程。
    """

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, Tool] = {}
        self._connect_tasks: dict[str, asyncio.Task] = {}

    def add_server(
        self,
        name: str,
        command: str = None,
        args: list[str] = None,
        url: str = None,
        env: dict = None,
        auto_connect: bool = True,
    ) -> MCPClient:
        """添加 MCP 服务器（同步，不阻塞）

        Args:
            name: 服务器名称
            command: 命令
            args: 参数
            url: URL
            env: 环境变量
            auto_connect: 是否自动连接（同步模式下立即连接，会阻塞）

        Returns:
            MCPClient 实例
        """
        client = MCPClient(
            name=name,
            command=command,
            args=args,
            url=url,
            env=env,
        )

        self._clients[name] = client

        # 注意：auto_connect=False 时由调用方负责异步连接
        if auto_connect:
            client.connect()

        return client

    async def add_server_async(
        self,
        name: str,
        command: str = None,
        args: list[str] = None,
        url: str = None,
        env: dict = None,
        background: bool = True,
    ) -> MCPClient:
        """异步添加 MCP 服务器

        Args:
            name: 服务器名称
            command: 命令
            args: 参数
            url: URL
            env: 环境变量
            background: 是否后台连接（不阻塞）

        Returns:
            MCPClient 实例
        """
        client = MCPClient(
            name=name,
            command=command,
            args=args,
            url=url,
            env=env,
        )

        self._clients[name] = client

        if background:
            # 后台连接，不阻塞
            task = asyncio.create_task(self._connect_and_log(client))
            self._connect_tasks[name] = task
        else:
            # 等待连接完成
            await client.connect_async()

        return client

    async def _connect_and_log(self, client: MCPClient):
        """后台连接并记录结果"""
        try:
            await client.connect_async()
        except Exception as e:
            # 连接失败，记录错误
            client._connect_error = str(e)

    def get_client(self, name: str) -> MCPClient | None:
        """获取 MCP 客户端"""
        return self._clients.get(name)

    def get_all_tools(self) -> list[Tool]:
        """获取所有工具（转换为 LangChain 格式）"""
        all_tools = []
        for client in self._clients.values():
            if client._initialized:
                tools = convert_all_tools(client)
                all_tools.extend(tools)
                for tool in tools:
                    self._tools[tool.name] = tool
        return all_tools

    def get_tool(self, name: str) -> Tool | None:
        """获取指定工具"""
        return self._tools.get(name)

    def list_servers(self) -> list[dict]:
        """列出所有服务器"""
        return [
            {
                "name": name,
                "connected": client._initialized,
                "connecting": client._connecting,
                "error": client._connect_error,
                "server_info": client.get_server_info(),
                "tools_count": len(client._tools),
            }
            for name, client in self._clients.items()
        ]

    async def wait_for_connections(self, timeout: float = 30.0) -> dict:
        """等待所有后台连接完成

        Args:
            timeout: 超时时间（秒）

        Returns:
            连接结果统计
        """
        if not self._connect_tasks:
            return {"total": 0, "connected": 0, "failed": 0}

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._connect_tasks.values(), return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            pass

        # 统计结果
        connected = sum(1 for c in self._clients.values() if c._initialized)
        failed = sum(1 for c in self._clients.values() if c._connect_error)

        return {
            "total": len(self._clients),
            "connected": connected,
            "failed": failed,
        }

    def close_all(self):
        """关闭所有连接"""
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        self._tools.clear()
        self._connect_tasks.clear()

    def close_server(self, name: str) -> bool:
        """关闭指定服务器"""
        client = self._clients.pop(name, None)
        if client:
            client.close()
            for tool_name in list(self._tools.keys()):
                if tool_name in [t["name"] for t in client._tools]:
                    del self._tools[tool_name]
            # 取消后台连接任务
            if name in self._connect_tasks:
                self._connect_tasks[name].cancel()
                del self._connect_tasks[name]
            return True
        return False
