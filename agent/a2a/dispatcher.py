"""任务结果邮箱和状态管理

核心组件：
- TaskResult: 任务执行结果
- Inbox: Worker 结果邮箱（支持 subscribe）
- MainAgentBusyState: MainAgent 忙闲状态（已弃用，使用 Registry 生命周期状态）

注意：Inbox 和 MainAgentBusyState 保留用于向后兼容。
新代码应通过 AgentRegistry 管理 Agent 生命周期。
"""

import queue
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional


# ============ 任务结果 ============


class TaskResultStatus(str, Enum):
    """任务结果状态"""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class TaskResult:
    """单个任务的执行结果"""

    plan_id: str
    task_id: str
    status: TaskResultStatus
    result: Optional[str] = None
    error: Optional[str] = None
    job_id: Optional[str] = None
    worker_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# ============ Inbox 邮箱 ============


class Inbox:
    """Worker 结果邮箱（线程安全）

    Worker 完成任务后将结果写入此邮箱。
    支持 subscribe 机制，结果写入时自动通知订阅者。

    注意：Inbox 是全局共享邮箱，用于 Worker 结果通知。
    Agent 通信应通过 AgentRegistry.send_message() 进行。
    """

    def __init__(self):
        self._queue: queue.Queue[TaskResult] = queue.Queue()
        self._subscribers: list[Callable[[TaskResult], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[TaskResult], None]) -> None:
        """注册结果回调

        Args:
            callback: 回调函数，接收 TaskResult 参数
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[TaskResult], None]) -> None:
        """取消注册结果回调

        Args:
            callback: 要取消的回调函数
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def put(self, result: TaskResult) -> None:
        """写入结果并通知订阅者"""
        self._queue.put(result)

        # 通知所有订阅者
        with self._lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(result)
            except Exception:
                pass

    def get_all(self) -> list[TaskResult]:
        """取出所有结果（非阻塞）"""
        results = []
        while True:
            try:
                result = self._queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results

    def is_empty(self) -> bool:
        """邮箱是否为空"""
        return self._queue.empty()

    def size(self) -> int:
        """邮箱大小"""
        return self._queue.qsize()


# ============ MainAgent 状态（已弃用） ============


class MainAgentBusyState:
    """MainAgent 忙闲状态管理（已弃用）

    请使用 AgentRegistry 的生命周期状态代替：
        from agent.core.registry import get_registry, AgentLifecycleState
        registry = get_registry()
        registry.get_state("main")  # -> AgentLifecycleState.RUNNING / IDLE / PENDING
    """

    def __init__(self):
        self._busy = False
        self._lock = threading.Lock()

    def set_busy(self) -> None:
        """标记为忙碌"""
        with self._lock:
            self._busy = True

    def set_idle(self) -> None:
        """标记为空闲"""
        with self._lock:
            self._busy = False

    def is_busy(self) -> bool:
        """是否忙碌"""
        with self._lock:
            return self._busy

    def is_idle(self) -> bool:
        """是否空闲"""
        with self._lock:
            return not self._busy


# ============ 全局实例 ============

_global_inbox: Optional[Inbox] = None
_global_agent_state: Optional[MainAgentBusyState] = None
_global_lock = threading.Lock()


def get_inbox() -> Inbox:
    """获取全局邮箱"""
    global _global_inbox
    with _global_lock:
        if _global_inbox is None:
            _global_inbox = Inbox()
        return _global_inbox


def get_agent_state() -> MainAgentBusyState:
    """获取全局 Agent 状态（已弃用）

    请使用 AgentRegistry 生命周期状态代替：
        from agent.core.registry import get_registry, AgentLifecycleState
        registry = get_registry()
        state = registry.get_state("main")
    """
    global _global_agent_state
    with _global_lock:
        if _global_agent_state is None:
            _global_agent_state = MainAgentBusyState()
        return _global_agent_state
