"""A2A Client - 调用外部 Agent 的 HTTP 客户端

遵循 A2A JSON-RPC 2.0 协议，用于调用外部系统的 Agent。
"""

import json
import uuid
from typing import Any

import httpx

from agent.a2a.models import Task, Message, AgentCard
from agent.a2a.protocol import (
    JSONRPC_VERSION,
    METHOD_MESSAGE_SEND,
    METHOD_TASKS_GET,
    METHOD_TASKS_CANCEL,
    METHOD_AGENT_GET_CARD,
    JSONRPCResponse,
    make_success_response,
    make_error_response,
    JSONRPCErrorCodes,
)


class A2AClientError(Exception):
    """A2A Client 错误"""
    pass


class A2AClient:
    """A2A HTTP Client

    用于调用外部 A2A Agent 的客户端。

    使用方式：
        client = A2AClient("http://localhost:8001")

        # 获取 Agent Card
        card = client.get_card()

        # 发送消息
        task = client.message_send(task, message)

        # 查询任务状态
        task = client.tasks_get(task_id)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """初始化 A2A Client

        Args:
            base_url: Agent 的基础 URL（如 http://localhost:8001）
            api_key: 可选的 API Key
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # 构建 headers
        self._headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _build_request(self, method: str, params: dict | None = None) -> dict:
        """构建 JSON-RPC 请求"""
        return {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "params": params,
            "id": str(uuid.uuid4()),
        }

    def _parse_response(self, response_data: dict) -> JSONRPCResponse:
        """解析 JSON-RPC 响应"""
        return JSONRPCResponse.from_dict(response_data)

    def _request(self, method: str, params: dict | None = None) -> Any:
        """发送 JSON-RPC 请求

        Args:
            method: A2A 方法名
            params: 方法参数

        Returns:
            响应结果

        Raises:
            A2AClientError: 请求失败
        """
        request_body = self._build_request(method, params)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/a2a",
                    headers=self._headers,
                    json=request_body,
                )

            if response.status_code != 200:
                raise A2AClientError(f"HTTP error: {response.status_code}")

            response_data = response.json()
            rpc_response = self._parse_response(response_data)

            if rpc_response.is_error():
                raise A2AClientError(rpc_response.get_error_message())

            return rpc_response.result

        except httpx.RequestError as e:
            raise A2AClientError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            raise A2AClientError(f"Invalid JSON response: {e}")

    # ============ A2A 标准方法 ============

    def get_card(self) -> AgentCard:
        """获取 Agent Card

        Returns:
            AgentCard 对象
        """
        result = self._request(METHOD_AGENT_GET_CARD)
        return AgentCard(**result)

    def message_send(
        self,
        task_id: str,
        message: Message,
    ) -> Task:
        """发送消息到 Agent

        Args:
            task_id: Task ID
            message: 消息内容

        Returns:
            更新后的 Task
        """
        params = {
            "taskId": task_id,
            "message": message.model_dump(mode="json"),  # 使用 mode="json" 序列化 datetime
        }
        result = self._request(METHOD_MESSAGE_SEND, params)
        return Task(**result)

    def tasks_get(self, task_id: str) -> Task | None:
        """获取 Task 状态

        Args:
            task_id: Task ID

        Returns:
            Task 对象，如果不存在返回 None
        """
        params = {"taskId": task_id}
        result = self._request(METHOD_TASKS_GET, params)

        if result is None:
            return None

        return Task(**result)

    def tasks_cancel(self, task_id: str) -> Task | None:
        """取消 Task

        Args:
            task_id: Task ID

        Returns:
            取消后的 Task，如果不存在返回 None
        """
        params = {"taskId": task_id}
        result = self._request(METHOD_TASKS_CANCEL, params)

        if result is None:
            return None

        return Task(**result)

    # ============ 便捷方法 ============

    def create_and_send(
        self,
        sender_id: str,
        message: Message,
    ) -> Task:
        """创建 Task 并发送消息

        这是一个便捷方法，自动生成 task_id。

        Args:
            sender_id: 发送方 ID
            message: 消息内容

        Returns:
            创建并更新后的 Task
        """
        task_id = str(uuid.uuid4())
        return self.message_send(task_id, message)


class A2AClientPool:
    """A2A Client 池

    管理多个外部 Agent 的客户端连接。
    """

    def __init__(self):
        self._clients: dict[str, A2AClient] = {}

    def get_client(self, agent_name: str) -> A2AClient | None:
        """获取指定 Agent 的客户端

        Args:
            agent_name: Agent 名称（配置文件中定义）

        Returns:
            A2AClient 实例，如果未配置返回 None
        """
        if agent_name in self._clients:
            return self._clients[agent_name]

        from agent.a2a.config import get_agent_endpoint

        endpoint = get_agent_endpoint(agent_name)
        if not endpoint:
            return None

        client = A2AClient(
            base_url=endpoint.url,
            api_key=endpoint.api_key,
        )
        self._clients[agent_name] = client
        return client

    def list_available_agents(self) -> list[str]:
        """列出所有可用的外部 Agent"""
        from agent.a2a.config import get_a2a_config
        return list(get_a2a_config().agents.keys())


# 全局 Client Pool
_client_pool: A2AClientPool | None = None


def get_client_pool() -> A2AClientPool:
    """获取全局 Client Pool"""
    global _client_pool
    if _client_pool is None:
        _client_pool = A2AClientPool()
    return _client_pool
