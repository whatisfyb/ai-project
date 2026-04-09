"""全局 Agent 注册表

用于管理正在运行的 Agent，支持终止所有 Agent。
通过依赖注入解耦，避免循环依赖。
"""

import threading
from typing import Callable


class AgentRegistry:
    """全局 Agent 注册表，用于管理正在运行的 Agent

    通过依赖注入方式解耦：注册时传入终止函数，而非直接依赖 Executor 类型。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._terminators = {}  # plan_id -> terminate_fn
                    cls._instance._running = False
        return cls._instance

    def register(self, plan_id: str, terminate_fn: Callable[[], None]) -> None:
        """注册一个 Executor 的终止函数

        Args:
            plan_id: Plan ID
            terminate_fn: 终止函数，调用后会停止该 Executor
        """
        with self._lock:
            self._terminators[plan_id] = terminate_fn
            self._running = True

    def unregister(self, plan_id: str) -> None:
        """注销一个 Executor"""
        with self._lock:
            self._terminators.pop(plan_id, None)
            if not self._terminators:
                self._running = False

    def is_running(self) -> bool:
        """检查是否有 Agent 正在运行"""
        with self._lock:
            return self._running

    def get_running_plan_ids(self) -> list[str]:
        """获取所有正在运行的 plan_id"""
        with self._lock:
            return list(self._terminators.keys())

    def terminate_all(self) -> list[str]:
        """终止所有正在运行的 Agent，返回被终止的 plan_id 列表"""
        terminated = []
        with self._lock:
            for plan_id, terminate_fn in self._terminators.items():
                try:
                    terminate_fn()
                    terminated.append(plan_id)
                except Exception:
                    pass
            self._terminators.clear()
            self._running = False
        return terminated

    def get_terminator(self, plan_id: str) -> Callable[[], None] | None:
        """获取指定 plan_id 的终止函数"""
        with self._lock:
            return self._terminators.get(plan_id)


# 全局实例
agent_registry = AgentRegistry()


def terminate() -> list[str]:
    """终止所有正在运行的 Agent

    Returns:
        被终止的 plan_id 列表
    """
    return agent_registry.terminate_all()
