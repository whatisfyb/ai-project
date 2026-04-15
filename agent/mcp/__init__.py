# agent/mcp/__init__.py
"""MCP (Model Context Protocol) integration module

Provides MCP client functionality to connect to external MCP servers
and use their tools as LangChain tools.
"""

from agent.mcp.models import (
    JSONRPCRequest,
    JSONRPCResponse,
    MCPCapabilities,
    InitializeResult,
    MCPToolInputSchema,
    MCPTool,
    ToolCallResult,
)
from agent.mcp.client import MCPClient, MCPClientError
from agent.mcp.manager import MCPManager, get_mcp_manager
from agent.mcp.tools import load_mcp_tools, get_mcp_tool_info, create_mcp_tool

__all__ = [
    # Models
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPCapabilities",
    "InitializeResult",
    "MCPToolInputSchema",
    "MCPTool",
    "ToolCallResult",
    # Client
    "MCPClient",
    "MCPClientError",
    # Manager
    "MCPManager",
    "get_mcp_manager",
    # Tools
    "load_mcp_tools",
    "get_mcp_tool_info",
    "create_mcp_tool",
]
