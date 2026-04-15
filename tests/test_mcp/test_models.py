"""Tests for MCP data models"""

import pytest
from agent.mcp.models import (
    JSONRPCRequest,
    JSONRPCResponse,
    MCPCapabilities,
    InitializeResult,
    MCPToolInputSchema,
    MCPTool,
    ToolCallResult,
)


class TestJSONRPCRequest:
    def test_create_request(self):
        req = JSONRPCRequest(id=1, method="test", params={"key": "value"})
        assert req.jsonrpc == "2.0"
        assert req.id == 1
        assert req.method == "test"
        assert req.params == {"key": "value"}

    def test_request_without_params(self):
        req = JSONRPCRequest(id=1, method="test")
        assert req.params is None

    def test_request_to_dict(self):
        req = JSONRPCRequest(id=1, method="initialize", params={"protocolVersion": "2024-11-05"})
        d = req.model_dump()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["method"] == "initialize"


class TestJSONRPCResponse:
    def test_success_response(self):
        resp = JSONRPCResponse(id=1, result={"status": "ok"})
        assert resp.jsonrpc == "2.0"
        assert resp.result == {"status": "ok"}
        assert resp.error is None

    def test_error_response(self):
        resp = JSONRPCResponse(id=1, error={"code": -32600, "message": "Invalid Request"})
        assert resp.is_error() is True
        assert resp.get_error_message() == "Invalid Request"

    def test_from_dict(self):
        data = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        resp = JSONRPCResponse.from_dict(data)
        assert resp.id == 1
        assert resp.result == {"tools": []}


class TestMCPCapabilities:
    def test_default_capabilities(self):
        cap = MCPCapabilities()
        assert cap.tools is False
        assert cap.resources is False
        assert cap.prompts is False

    def test_custom_capabilities(self):
        cap = MCPCapabilities(tools=True, resources=True)
        assert cap.tools is True
        assert cap.resources is True


class TestInitializeResult:
    def test_parse_init_result(self):
        data = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": True},
            "serverInfo": {"name": "test-server", "version": "1.0.0"}
        }
        result = InitializeResult(**data)
        assert result.protocol_version == "2024-11-05"
        assert result.capabilities.tools is True


class TestMCPTool:
    def test_parse_tool(self):
        data = {
            "name": "read_file",
            "description": "Read a file",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
        tool = MCPTool(**data)
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.input_schema.type == "object"
        assert "path" in tool.input_schema.properties

    def test_tool_without_description(self):
        data = {"name": "test_tool", "inputSchema": {}}
        tool = MCPTool(**data)
        assert tool.name == "test_tool"
        assert tool.description == ""


class TestToolCallResult:
    def test_success_result(self):
        result = ToolCallResult(
            content=[{"type": "text", "text": "file content"}],
            is_error=False
        )
        assert result.is_error is False
        assert len(result.content) == 1

    def test_error_result(self):
        result = ToolCallResult(
            content=[{"type": "text", "text": "Error: file not found"}],
            is_error=True
        )
        assert result.is_error is True

    def test_parse_from_dict(self):
        data = {
            "content": [{"type": "text", "text": "result"}],
            "isError": True
        }
        result = ToolCallResult(**data)
        assert result.is_error is True
