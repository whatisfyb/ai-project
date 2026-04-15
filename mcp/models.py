"""MCP protocol data models

Based on MCP specification: https://spec.modelcontextprotocol.io/
"""

from typing import Any
from pydantic import BaseModel, Field

# Import JSON-RPC types from common module
from utils.jsonrpc import JSONRPCRequest, JSONRPCResponse


# ============ MCP Capabilities ============

class MCPCapabilities(BaseModel):
    """Server capabilities

    Note: tools, resources, prompts can be bool or dict with additional options
    e.g., {'tools': {'listChanged': True}} or {'tools': True}
    """
    tools: bool | dict[str, Any] = False
    resources: bool | dict[str, Any] = False
    prompts: bool | dict[str, Any] = False

    def has_tools(self) -> bool:
        """Check if server supports tools"""
        return bool(self.tools)

    def has_resources(self) -> bool:
        """Check if server supports resources"""
        return bool(self.resources)

    def has_prompts(self) -> bool:
        """Check if server supports prompts"""
        return bool(self.prompts)


class InitializeResult(BaseModel):
    """Initialize response from MCP server"""
    model_config = {"populate_by_name": True}

    protocol_version: str = Field(alias="protocolVersion")
    capabilities: MCPCapabilities
    server_info: dict[str, str] = Field(alias="serverInfo")


# ============ MCP Tool ============

class MCPToolInputSchema(BaseModel):
    """Tool input parameter schema (JSON Schema)"""
    type: str = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str
    description: str = ""
    input_schema: MCPToolInputSchema = Field(alias="inputSchema")


class ToolCallResult(BaseModel):
    """Tool call result"""
    model_config = {"populate_by_name": True}

    content: list[dict]
    is_error: bool = Field(default=False, alias="isError")
