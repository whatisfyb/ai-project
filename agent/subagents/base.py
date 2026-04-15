"""子代理基类"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from utils.core.llm import get_llm_model


# 使用 dict 作为 bound，TypedDict 是 dict 的子类型
StateT = TypeVar("StateT", bound=dict)


class BaseSubagent(ABC, Generic[StateT]):
    """子代理基类

    所有子代理继承此类，实现：
    - agent_type: 代理类型标识
    - description: 代理描述（供 main agent 决策使用）
    - tools: 可用工具列表
    - build_graph: 构建状态图
    """

    def __init__(self):
        self.llm = get_llm_model()
        self._graph = None
        self._checkpointer = MemorySaver()

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """代理类型标识"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """代理描述（供 main agent 决策使用）"""
        pass

    @property
    @abstractmethod
    def tools(self) -> list:
        """可用工具列表"""
        pass

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """构建状态图"""
        pass

    @property
    def graph(self) -> StateGraph:
        """获取编译后的图"""
        if self._graph is None:
            graph = self.build_graph()
            self._graph = graph.compile(checkpointer=self._checkpointer)
        return self._graph

    def run(self, input_data: dict, thread_id: str = "default") -> dict[str, Any]:
        """运行代理

        Args:
            input_data: 输入数据
            thread_id: 会话 ID（用于中断恢复）

        Returns:
            执行结果
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(input_data, config)
        return result

    def get_info(self) -> dict[str, Any]:
        """获取代理信息（供 main agent 调用）"""
        return {
            "type": self.agent_type,
            "description": self.description,
            "tools": [t.name for t in self.tools] if self.tools else [],
        }
