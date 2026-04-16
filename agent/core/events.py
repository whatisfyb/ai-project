"""MainAgent 事件模型

定义事件类型和事件数据结构，用于 MainAgent 事件驱动循环。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    """事件类型"""
    USER_INPUT = "user_input"                     # 用户输入
    INBOX_NOTIFICATION = "inbox_notification"     # Worker/子Agent 结果
    SHUTDOWN = "shutdown"                         # 关闭信号


@dataclass
class AgentEvent:
    """MainAgent 事件"""
    type: EventType
    data: dict = field(default_factory=dict)
    thread_id: str = "default"
    # 完成信号：TUI 等待当前轮处理完成
    on_complete: Optional[Any] = None  # asyncio.Event

    @staticmethod
    def user_input(message: str, thread_id: str = "default", on_complete=None) -> "AgentEvent":
        """创建用户输入事件"""
        return AgentEvent(
            type=EventType.USER_INPUT,
            data={"message": message},
            thread_id=thread_id,
            on_complete=on_complete,
        )

    @staticmethod
    def inbox_notification(
        task_id: str,
        status: str,
        result: str | None = None,
        error: str | None = None,
        thread_id: str = "default"
    ) -> "AgentEvent":
        """创建 inbox 通知事件"""
        return AgentEvent(
            type=EventType.INBOX_NOTIFICATION,
            data={"task_id": task_id, "status": status, "result": result, "error": error},
            thread_id=thread_id,
        )

    @staticmethod
    def shutdown() -> "AgentEvent":
        """创建关闭事件"""
        return AgentEvent(type=EventType.SHUTDOWN)
