"""BaseAgent 基类

定义 Agent 标准接口：
- agent_id: 唯一标识
- agent_type: 类型
- get_card(): 返回 AgentCard
- handle_task(): 处理任务
- get_state()/restore_state(): 状态保存/恢复
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.a2a.models import AgentCard, Task


class BaseAgent(ABC):
    """Agent 基类

    所有 Agent 都应该继承此类，实现标准接口。
    """

    agent_id: str = "unknown"
    agent_type: str = "unknown"

    @abstractmethod
    def get_card(self) -> AgentCard:
        """返回 Agent 能力声明

        Returns:
            AgentCard 描述 Agent 的能力、技能等
        """
        pass

    @abstractmethod
    def handle_task(self, task: Task) -> Any:
        """处理任务

        Args:
            task: A2A Task

        Returns:
            任务结果
        """
        pass

    def get_state(self) -> dict[str, Any]:
        """获取当前状态（用于检查点保存）

        Returns:
            状态字典
        """
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        """恢复状态（从检查点恢复）

        Args:
            state: 之前保存的状态字典
        """
        pass

    def on_interrupt(self) -> None:
        """中断回调

        当 Agent 被中断时调用，可以用于保存进度。
        """
        pass

    def on_idle(self) -> None:
        """空闲回调

        当 Agent 进入空闲状态时调用。
        """
        pass
