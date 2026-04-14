"""A2A JSON-RPC 协议常量和辅助函数

基于 A2A 协议规范：https://github.com/google/A2A
"""

from typing import Any
from dataclasses import dataclass


# ============ JSON-RPC 2.0 常量 ============

JSONRPC_VERSION = "2.0"


# ============ A2A 方法名 ============

METHOD_MESSAGE_SEND = "message/send"
METHOD_TASKS_GET = "tasks/get"
METHOD_TASKS_CANCEL = "tasks/cancel"
METHOD_TASKS_SUBSCRIBE = "tasks/subscribe"
METHOD_AGENT_GET_CARD = "agent/getCard"


# ============ JSON-RPC 消息结构 ============

@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 请求"""
    jsonrpc: str = JSONRPC_VERSION
    method: str = ""
    params: dict[str, Any] | None = None
    id: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params is not None:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 响应"""
    jsonrpc: str = JSONRPC_VERSION
    result: Any = None
    error: dict | None = None
    id: str | int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCResponse":
        return cls(
            jsonrpc=data.get("jsonrpc", JSONRPC_VERSION),
            result=data.get("result"),
            error=data.get("error"),
            id=data.get("id"),
        )

    def is_error(self) -> bool:
        return self.error is not None

    def get_error_message(self) -> str:
        if self.error:
            return self.error.get("message", "Unknown error")
        return ""


# ============ JSON-RPC 错误码 ============

class JSONRPCErrorCodes:
    """JSON-RPC 标准错误码"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# ============ A2A 请求参数构建 ============

def build_message_send_params(
    task_id: str,
    message: dict,
) -> dict[str, Any]:
    """构建 message/send 方法参数

    Args:
        task_id: Task ID
        message: Message 对象的字典表示

    Returns:
        方法参数
    """
    return {
        "taskId": task_id,
        "message": message,
    }


def build_tasks_get_params(task_id: str) -> dict[str, Any]:
    """构建 tasks/get 方法参数"""
    return {"taskId": task_id}


def build_tasks_cancel_params(task_id: str) -> dict[str, Any]:
    """构建 tasks/cancel 方法参数"""
    return {"taskId": task_id}


# ============ 错误响应构建 ============

def make_error_response(
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """构建错误响应"""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data

    return {
        "jsonrpc": JSONRPC_VERSION,
        "error": error,
        "id": request_id,
    }


def make_success_response(
    request_id: str | int | None,
    result: Any,
) -> dict[str, Any]:
    """构建成功响应"""
    return {
        "jsonrpc": JSONRPC_VERSION,
        "result": result,
        "id": request_id,
    }
