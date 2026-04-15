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

# Note: Client, Manager, and Tools will be added when implemented
__all__ = [
    # Models
    "JSONRPCRequest",
    "JSONRPCResponse",
    "MCPCapabilities",
    "InitializeResult",
    "MCPToolInputSchema",
    "MCPTool",
    "ToolCallResult",
]
