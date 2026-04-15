"""JSON-RPC 2.0 Protocol Implementation

Based on JSON-RPC 2.0 Specification: https://www.jsonrpc.org/specification

Used by:
- MCP (Model Context Protocol)
- A2A (Agent-to-Agent Protocol)
"""

from typing import Any
from pydantic import BaseModel


# ============ Constants ============

JSONRPC_VERSION = "2.0"


# ============ Error Codes ============

class JSONRPCErrorCodes:
    """JSON-RPC standard error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# ============ Request/Response ============

class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request"""
    jsonrpc: str = JSONRPC_VERSION
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response"""
    jsonrpc: str = JSONRPC_VERSION
    id: int | str | None = None
    result: Any = None
    error: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCResponse":
        """Create from dict"""
        return cls(
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
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


# ============ Helper Functions ============

def make_request(method: str, params: dict | None = None, request_id: int | str | None = None) -> dict:
    """Build JSON-RPC request dict"""
    result = {
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
    }
    if params is not None:
        result["params"] = params
    if request_id is not None:
        result["id"] = request_id
    return result


def make_success_response(request_id: int | str | None, result: Any) -> dict:
    """Build success response dict"""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "result": result,
        "id": request_id,
    }


def make_error_response(
    request_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict:
    """Build error response dict"""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data

    return {
        "jsonrpc": JSONRPC_VERSION,
        "error": error,
        "id": request_id,
    }
