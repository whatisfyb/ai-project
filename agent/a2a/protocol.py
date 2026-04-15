"""A2A JSON-RPC 协议常量和辅助函数

基于 A2A 协议规范：https://github.com/google/A2A
"""

from typing import Any

# Import JSON-RPC types from common module
from utils.jsonrpc import (
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCErrorCodes,
    JSONRPC_VERSION,
    make_error_response as _make_error_response,
    make_success_response as _make_success_response,
)


# ============ A2A 方法名 ============

METHOD_MESSAGE_SEND = "message/send"
METHOD_TASKS_GET = "tasks/get"
METHOD_TASKS_CANCEL = "tasks/cancel"
METHOD_TASKS_SUBSCRIBE = "tasks/subscribe"
METHOD_AGENT_GET_CARD = "agent/getCard"


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


# ============ 响应构建 (re-export from utils.jsonrpc) ============

def make_error_response(
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """构建错误响应"""
    return _make_error_response(request_id, code, message, data)


def make_success_response(
    request_id: str | int | None,
    result: Any,
) -> dict[str, Any]:
    """构建成功响应"""
    return _make_success_response(request_id, result)
