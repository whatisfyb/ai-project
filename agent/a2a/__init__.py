"""A2A (Agent-to-Agent) 通信模块

提供 Agent 间通信能力，基于 Google A2A 协议规范。
"""

from agent.a2a.models import (
    # 核心数据模型
    Task,
    TaskStatus,
    Message,
    MessageRole,
    Part,
    PartType,
    Artifact,
    # Agent 能力声明
    AgentCard,
    AgentCapabilities,
    Skill,
    # Transport
    TransportMessage,
    # 事件
    TaskEvent,
)

from agent.a2a.transport import (
    Transport,
    InMemoryTransport,
    TaskCallback,
    MessageCallback,
    get_transport,
    reset_transport,
)

from agent.a2a.client import (
    A2AClient,
    A2AClientPool,
    A2AClientError,
    get_client_pool,
)

from agent.a2a.server import A2AServer

from agent.a2a.config import (
    A2AConfig,
    AgentEndpoint,
    load_a2a_config,
    get_a2a_config,
    get_agent_endpoint,
)

from agent.a2a.protocol import (
    JSONRPC_VERSION,
    METHOD_MESSAGE_SEND,
    METHOD_TASKS_GET,
    METHOD_TASKS_CANCEL,
    METHOD_AGENT_GET_CARD,
)

from agent.a2a.worker import (
    A2AWorker,
    A2AWorkerPool,
)

from agent.a2a.tools import (
    plan_dispatch,
    worker_list,
)

from agent.a2a.dispatcher import (
    # 结果
    TaskResult,
    TaskResultStatus,
    Inbox,
    # 状态（已弃用，使用 AgentRegistry 生命周期状态）
    MainAgentBusyState,
    # 全局实例
    get_inbox,
    get_agent_state,  # 已弃用
)

__all__ = [
    # Task
    "Task",
    "TaskStatus",
    # Message
    "Message",
    "MessageRole",
    "Part",
    "PartType",
    # Artifact
    "Artifact",
    # Agent
    "AgentCard",
    "AgentCapabilities",
    "Skill",
    # Transport
    "Transport",
    "InMemoryTransport",
    "TransportMessage",
    "TaskCallback",
    "MessageCallback",
    "get_transport",
    "reset_transport",
    # Client
    "A2AClient",
    "A2AClientPool",
    "A2AClientError",
    "get_client_pool",
    # Server
    "A2AServer",
    # Config
    "A2AConfig",
    "AgentEndpoint",
    "load_a2a_config",
    "get_a2a_config",
    "get_agent_endpoint",
    # Protocol
    "JSONRPC_VERSION",
    "METHOD_MESSAGE_SEND",
    "METHOD_TASKS_GET",
    "METHOD_TASKS_CANCEL",
    "METHOD_AGENT_GET_CARD",
    # Worker
    "A2AWorker",
    "A2AWorkerPool",
    # Tools
    "plan_dispatch",
    "worker_list",
    # Dispatcher
    "TaskResult",
    "TaskResultStatus",
    "Inbox",
    "MainAgentBusyState",
    "get_inbox",
    "get_agent_state",
    # Event
    "TaskEvent",
]
