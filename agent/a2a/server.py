"""A2A Server - 暴露 HTTP 服务供外部 Agent 调用

遵循 A2A JSON-RPC 2.0 协议，接收外部系统的调用。
"""

import json
import threading
from typing import Callable, Any

from flask import Flask, request, jsonify

from agent.a2a.models import Task, TaskStatus, Message, AgentCard
from agent.a2a.protocol import (
    METHOD_MESSAGE_SEND,
    METHOD_TASKS_GET,
    METHOD_TASKS_CANCEL,
    METHOD_AGENT_GET_CARD,
    make_success_response,
    make_error_response,
    JSONRPCErrorCodes,
)
from agent.a2a.transport import Transport


class A2AServer:
    """A2A HTTP Server

    暴露 A2A JSON-RPC 接口，接收外部 Agent 的调用。

    使用方式：
        server = A2AServer(
            agent_id="my-agent",
            card=agent_card,
            transport=transport,
            port=8001,
        )
        server.start()  # 启动 HTTP 服务
        server.stop()   # 停止服务
    """

    def __init__(
        self,
        agent_id: str,
        card: AgentCard,
        transport: Transport,
        handler: Callable[[Task, Message], None],
        port: int = 8000,
        host: str = "0.0.0.0",
        api_key: str | None = None,
    ):
        """初始化 A2A Server

        Args:
            agent_id: Agent ID
            card: Agent Card
            transport: Transport 实例
            handler: 消息处理函数
            port: 监听端口
            host: 监听地址
            api_key: 可选的 API Key（用于认证）
        """
        self.agent_id = agent_id
        self.card = card
        self.transport = transport
        self.handler = handler
        self.port = port
        self.host = host
        self.api_key = api_key

        self._app: Flask | None = None
        self._server_thread: threading.Thread | None = None
        self._running = False

    def _create_app(self) -> Flask:
        """创建 Flask 应用"""
        app = Flask(__name__)

        @app.route("/a2a", methods=["POST"])
        def handle_a2a():
            return self._handle_request()

        @app.route("/.well-known/agent.json", methods=["GET"])
        def get_agent_card():
            """Agent 发现端点"""
            return jsonify(self.card.model_dump())

        return app

    def _verify_auth(self) -> bool:
        """验证认证"""
        if not self.api_key:
            return True  # 无需认证

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return token == self.api_key

        return False

    def _handle_request(self) -> Any:
        """处理 JSON-RPC 请求"""
        # 验证认证
        if not self._verify_auth():
            return jsonify(make_error_response(
                None,
                JSONRPCErrorCodes.INVALID_REQUEST,
                "Unauthorized",
            )), 401

        # 解析请求
        try:
            data = request.get_json()
            if not data:
                return jsonify(make_error_response(
                    None,
                    JSONRPCErrorCodes.INVALID_REQUEST,
                    "Invalid request body",
                ))
        except Exception:
            return jsonify(make_error_response(
                None,
                JSONRPCErrorCodes.PARSE_ERROR,
                "Parse error",
            ))

        request_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {})

        # 路由到对应方法
        try:
            result = self._dispatch_method(method, params)
            return jsonify(make_success_response(request_id, result))
        except Exception as e:
            return jsonify(make_error_response(
                request_id,
                JSONRPCErrorCodes.INTERNAL_ERROR,
                str(e),
            ))

    def _dispatch_method(self, method: str, params: dict) -> Any:
        """分发方法调用"""
        if method == METHOD_AGENT_GET_CARD:
            return self._handle_get_card()

        elif method == METHOD_MESSAGE_SEND:
            return self._handle_message_send(params)

        elif method == METHOD_TASKS_GET:
            return self._handle_tasks_get(params)

        elif method == METHOD_TASKS_CANCEL:
            return self._handle_tasks_cancel(params)

        else:
            raise ValueError(f"Method not found: {method}")

    def _handle_get_card(self) -> dict:
        """处理 agent/getCard"""
        return self.card.model_dump(mode="json")

    def _handle_message_send(self, params: dict) -> dict:
        """处理 message/send"""
        task_id = params.get("taskId")
        message_data = params.get("message")

        if not task_id or not message_data:
            raise ValueError("Missing taskId or message")

        # 解析 Message
        message = Message(**message_data)

        # 获取或创建 Task
        task = self.transport.tasks_get(task_id)
        if task is None:
            # 创建新 Task（外部调用方作为 sender）
            task = self.transport.create_task(
                sender_id="external",
                receiver_id=self.agent_id,
            )

        # 调用 handler
        self.handler(task, message)

        return task.model_dump(mode="json")

    def _handle_tasks_get(self, params: dict) -> dict | None:
        """处理 tasks/get"""
        task_id = params.get("taskId")
        if not task_id:
            raise ValueError("Missing taskId")

        task = self.transport.tasks_get(task_id)
        if task is None:
            return None

        return task.model_dump(mode="json")

    def _handle_tasks_cancel(self, params: dict) -> dict | None:
        """处理 tasks/cancel"""
        task_id = params.get("taskId")
        if not task_id:
            raise ValueError("Missing taskId")

        task = self.transport.tasks_cancel(task_id)
        if task is None:
            return None

        return task.model_dump(mode="json")

    # ============ 生命周期 ============

    def start(self) -> None:
        """启动服务器（后台线程）"""
        if self._running:
            return

        self._app = self._create_app()
        self._running = True

        # 在后台线程运行
        def run_server():
            self._app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                use_reloader=False,
            )

        self._server_thread = threading.Thread(
            target=run_server,
            name=f"A2AServer-{self.agent_id}",
            daemon=True,
        )
        self._server_thread.start()

    def stop(self) -> None:
        """停止服务器"""
        self._running = False
        # Flask 开发服务器没有优雅停止的方法
        # 生产环境应使用 WSGI 服务器如 gunicorn
