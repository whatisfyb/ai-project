"""MCP protocol data models

Based on MCP specification: https://spec.modelcontextprotocol.io/
"""

from typing import Any
from pydantic import BaseModel, Field


# ============ JSON-RPC 2.0 ============

class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request"""
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response"""
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any = None
    error: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCResponse":
        """Create from dict"""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error"),
        )

    def is_error(self) -> bool:
        """Check if response is an error"""
        return self.error is not None

    def get_error_message(self) -> str:
        """Get error message"""
        if self.error:
            return self.error.get("message", "Unknown error")
        return ""


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
