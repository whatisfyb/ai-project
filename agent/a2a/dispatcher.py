"""任务结果邮箱和状态管理

核心组件：
- TaskResult: 任务执行结果
- Inbox: Worker 结果邮箱
- MainAgentBusyState: MainAgent 忙闲状态
"""

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


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
    MainAgent 在 idle 状态时从中取出结果并处理。
    """

    def __init__(self):
        self._queue: queue.Queue[TaskResult] = queue.Queue()

    def put(self, result: TaskResult) -> None:
        """写入结果"""
        self._queue.put(result)

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


# ============ MainAgent 状态 ============

class MainAgentBusyState:
    """MainAgent 忙闲状态管理"""

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
    """获取全局 Agent 状态"""
    global _global_agent_state
    with _global_lock:
        if _global_agent_state is None:
            _global_agent_state = MainAgentBusyState()
        return _global_agent_state
