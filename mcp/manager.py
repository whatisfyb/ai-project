"""MCP Server Manager - manages multiple MCP server connections"""

import threading
from typing import Any

from mcp.client import MCPClient, MCPClientError


class MCPManager:
    """MCP Server Manager - Singleton

    Manages multiple MCP server connections and provides
    a unified interface for tool discovery and invocation.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._clients: dict[str, MCPClient] = {}
        self._tools_cache: dict[str, list] = {}

    # ============ Lifecycle ============

    def initialize(self) -> None:
        """Initialize from config - connect to all enabled servers"""
        from utils.core.config import get_settings_instance

        settings = get_settings_instance()

        # Get MCP config
        mcp_config = getattr(settings, "mcp", None)
        if not mcp_config or not mcp_config.servers:
            return

        for name, server_config in mcp_config.servers.items():
            if not server_config.enabled:
                continue

            self.connect_server(name, {
                "transport": server_config.transport,
                "url": server_config.url,
                "timeout": server_config.timeout,
                "headers": server_config.headers,
            })

    def connect_server(self, name: str, config: dict) -> bool:
        """Connect to a single MCP server

        Args:
            name: Server name
            config: Server configuration

        Returns:
            True if connected successfully
        """
        if name in self._clients:
            return self._clients[name].is_connected

        client = MCPClient(
            name=name,
            transport=config.get("transport", "http"),
            url=config.get("url", ""),
            timeout=config.get("timeout", 30),
            headers=config.get("headers"),
        )

        try:
            if client.connect():
                self._clients[name] = client
                self._tools_cache[name] = client.list_tools()
                return True
        except MCPClientError as e:
            print(f"[MCP] Failed to connect {name}: {e}")

        return False

    def disconnect_server(self, name: str) -> None:
        """Disconnect from a server"""
        if name in self._clients:
            self._clients[name].disconnect()
            del self._clients[name]
        if name in self._tools_cache:
            del self._tools_cache[name]

    def disconnect_all(self) -> None:
        """Disconnect from all servers"""
        for name in list(self._clients.keys()):
            self.disconnect_server(name)

    # ============ Queries ============

    def list_all_tools(self) -> list[tuple[str, Any]]:
        """List all tools from all connected servers

        Returns:
            List of (server_name, tool) tuples
        """
        tools = []
        for server_name, client in self._clients.items():
            for tool in client.list_tools():
                tools.append((server_name, tool))
        return tools

    def list_connected_servers(self) -> list[str]:
        """List connected server names"""
        return list(self._clients.keys())

    def get_client(self, server_name: str) -> MCPClient | None:
        """Get client for a specific server"""
        return self._clients.get(server_name)

    # ============ Tool Invocation ============

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Call a tool on a specific server

        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Result dict with success, content, is_error
        """
        client = self._clients.get(server_name)
        if not client:
            raise MCPClientError(f"Server not connected: {server_name}")

        result = client.call_tool(tool_name, arguments)

        # Extract text content
        if result.content:
            texts = [
                c.get("text", "")
                for c in result.content
                if c.get("type") == "text"
            ]
            return {
                "success": not result.is_error,
                "content": "\n".join(texts) if texts else str(result.content),
                "is_error": result.is_error,
            }

        return {"success": False, "content": "No result", "is_error": True}


# Global singleton
_mcp_manager: MCPManager | None = None


def get_mcp_manager() -> MCPManager:
    """Get MCP manager singleton"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager
