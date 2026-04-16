"""中断信号管理

支持：
- 全局中断（向后兼容）
- 单 Agent 中断
- 检查点保存/恢复
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class InterruptScope(str, Enum):
    """中断范围"""
    GLOBAL = "global"       # 全局中断
    AGENT = "agent"         # 单 Agent 中断


@dataclass
class Checkpoint:
    """检查点数据"""
    agent_id: str
    state: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class SignalManager:
    """信号管理器（单例）

    管理：
    - 全局中断信号
    - 单 Agent 中断信号
    - 检查点存储
    """

    _instance: Optional["SignalManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        # 全局中断事件
        self._global_interrupt = threading.Event()
        # 单 Agent 中断事件
        self._agent_interrupts: dict[str, threading.Event] = {}
        # 检查点存储
        self._checkpoints: dict[str, Checkpoint] = {}
        # 检查点锁
        self._checkpoint_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SignalManager":
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ============ 全局中断 ============

    def set_global_interrupt(self) -> None:
        """设置全局中断信号"""
        self._global_interrupt.set()

    def clear_global_interrupt(self) -> None:
        """清除全局中断信号"""
        self._global_interrupt.clear()

    def is_global_interrupted(self) -> bool:
        """检查全局是否被中断"""
        return self._global_interrupt.is_set()

    # ============ 单 Agent 中断 ============

    def set_agent_interrupt(self, agent_id: str) -> None:
        """设置单 Agent 中断信号"""
        if agent_id not in self._agent_interrupts:
            self._agent_interrupts[agent_id] = threading.Event()
        self._agent_interrupts[agent_id].set()

    def clear_agent_interrupt(self, agent_id: str) -> None:
        """清除单 Agent 中断信号"""
        if agent_id in self._agent_interrupts:
            self._agent_interrupts[agent_id].clear()

    def is_agent_interrupted(self, agent_id: str) -> bool:
        """检查单 Agent 是否被中断"""
        # 如果全局中断，所有 Agent 都被中断
        if self._global_interrupt.is_set():
            return True
        if agent_id in self._agent_interrupts:
            return self._agent_interrupts[agent_id].is_set()
        return False

    # ============ 检查点 ============

    def save_checkpoint(self, agent_id: str, state: dict) -> Checkpoint:
        """保存检查点"""
        checkpoint = Checkpoint(agent_id=agent_id, state=state)
        with self._checkpoint_lock:
            self._checkpoints[agent_id] = checkpoint
        return checkpoint

    def load_checkpoint(self, agent_id: str) -> Optional[Checkpoint]:
        """加载检查点"""
        with self._checkpoint_lock:
            return self._checkpoints.get(agent_id)

    def clear_checkpoint(self, agent_id: str) -> None:
        """清除检查点"""
        with self._checkpoint_lock:
            if agent_id in self._checkpoints:
                del self._checkpoints[agent_id]

    def has_checkpoint(self, agent_id: str) -> bool:
        """检查是否有检查点"""
        with self._checkpoint_lock:
            return agent_id in self._checkpoints


# ============ 全局实例 ============

def _get_manager() -> SignalManager:
    return SignalManager.get_instance()


# ============ 向后兼容接口 ============

def set_interrupt() -> None:
    """设置全局中断信号（向后兼容）"""
    _get_manager().set_global_interrupt()


def clear_interrupt() -> None:
    """清除全局中断信号（向后兼容）"""
    _get_manager().clear_global_interrupt()


def is_interrupted() -> bool:
    """检查全局是否被中断（向后兼容）"""
    return _get_manager().is_global_interrupted()


# ============ 新接口 ============

def set_interrupt_for(agent_id: str) -> None:
    """设置单 Agent 中断信号"""
    _get_manager().set_agent_interrupt(agent_id)


def clear_interrupt_for(agent_id: str) -> None:
    """清除单 Agent 中断信号"""
    _get_manager().clear_agent_interrupt(agent_id)


def is_interrupted_for(agent_id: str) -> bool:
    """检查单 Agent 是否被中断"""
    return _get_manager().is_agent_interrupted(agent_id)


def save_checkpoint(agent_id: str, state: dict) -> Checkpoint:
    """保存检查点"""
    return _get_manager().save_checkpoint(agent_id, state)


def load_checkpoint(agent_id: str) -> Optional[Checkpoint]:
    """加载检查点"""
    return _get_manager().load_checkpoint(agent_id)


def clear_checkpoint(agent_id: str) -> None:
    """清除检查点"""
    _get_manager().clear_checkpoint(agent_id)


def has_checkpoint(agent_id: str) -> bool:
    """检查是否有检查点"""
    return _get_manager().has_checkpoint(agent_id)
