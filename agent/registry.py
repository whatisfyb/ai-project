"""全局 Agent 注册表

用于管理正在运行的 Agent，支持终止所有 Agent。
"""

import threading


class AgentRegistry:
    """全局 Agent 注册表，用于管理正在运行的 Agent"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._executors = {}  # plan_id -> PlanExecutor
                    cls._instance._running = False
        return cls._instance

    def register(self, plan_id: str, executor) -> None:
        """注册一个正在运行的 Executor"""
        with self._lock:
            self._executors[plan_id] = executor
            self._running = True

    def unregister(self, plan_id: str) -> None:
        """注销一个 Executor"""
        with self._lock:
            self._executors.pop(plan_id, None)
            if not self._executors:
                self._running = False

    def is_running(self) -> bool:
        """检查是否有 Agent 正在运行"""
        with self._lock:
            return self._running

    def get_running_plan_ids(self) -> list[str]:
        """获取所有正在运行的 plan_id"""
        with self._lock:
            return list(self._executors.keys())

    def terminate_all(self) -> list[str]:
        """终止所有正在运行的 Agent，返回被终止的 plan_id 列表"""
        terminated = []
        with self._lock:
            for plan_id, executor in self._executors.items():
                try:
                    executor.terminate()
                    terminated.append(plan_id)
                except Exception:
                    pass
            self._executors.clear()
            self._running = False
        return terminated

    def get_executor(self, plan_id: str):
        """获取指定 plan_id 的 executor"""
        with self._lock:
            return self._executors.get(plan_id)


# 全局实例
agent_registry = AgentRegistry()


def terminate() -> list[str]:
    """终止所有正在运行的 Agent

    Returns:
        被终止的 plan_id 列表
    """
    return agent_registry.terminate_all()
