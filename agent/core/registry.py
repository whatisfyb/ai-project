"""统一 Agent Registry

管理：
- Agent 生命周期（pending/running/idle）
- Inbox（消息队列）
- 能力发现（AgentCard）
- 懒加载
- 空闲超时回收
- 终止控制

设计说明：
- 与 Transport 协作，而不是替代它
- Worker 仍然通过 Transport 注册
- Registry 提供统一的生命周期管理
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.a2a.models import AgentCard, Task, Message


class AgentLifecycleState(str, Enum):
    """Agent 生命周期状态"""
    PENDING = "pending"   # 未初始化
    RUNNING = "running"   # 正在运行
    IDLE = "idle"         # 空闲（等待回收）


@dataclass
class AgentEntry:
    """Agent 注册条目"""
    agent_id: str
    agent_type: str
    card: Any  # AgentCard（延迟导入）
    factory: Callable[[], Any] | None = None  # Agent 工厂函数（可选）
    auto_start: bool = False    # 是否自动启动

    # 运行时状态
    state: AgentLifecycleState = AgentLifecycleState.PENDING
    instance: Any = None        # Agent 实例
    thread: threading.Thread | None = None  # 工作线程
    inbox: queue.Queue = field(default_factory=queue.Queue)
    last_active: datetime = field(default_factory=datetime.now)

    # 终止函数（向后兼容）
    terminate_fn: Callable[[], None] | None = None


class AgentRegistry:
    """统一 Agent Registry（单例）

    功能：
    - 管理 Agent 生命周期
    - 管理 Inbox（消息队列）
    - 支持能力发现
    - 支持懒加载
    - 支持空闲超时回收
    - 与 Transport 协作
    """

    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()

    # 空闲超时时间（秒）
    IDLE_TIMEOUT = 300  # 5 分钟

    def __init__(self):
        self._agents: dict[str, AgentEntry] = {}
        self._agents_lock = threading.RLock()
        self._running = False
        self._idle_checker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ============ Agent 注册/注销 ============

    def register(
        self,
        agent_id: str,
        agent_type: str,
        card: AgentCard,
        factory: Callable[[], Any] | None = None,
        auto_start: bool = False,
    ) -> None:
        """注册 Agent

        Args:
            agent_id: Agent 唯一标识
            agent_type: Agent 类型
            card: AgentCard（能力描述）
            factory: Agent 工厂函数（可选，Worker 不需要）
            auto_start: 是否自动启动（Main Agent 需要自动启动）
        """
        with self._agents_lock:
            entry = AgentEntry(
                agent_id=agent_id,
                agent_type=agent_type,
                card=card,
                factory=factory,
                auto_start=auto_start,
            )
            self._agents[agent_id] = entry

            if auto_start and factory:
                self._activate_agent(entry)

    def register_with_terminator(
        self,
        agent_id: str,
        agent_type: str,
        card: AgentCard,
        terminate_fn: Callable[[], None],
    ) -> None:
        """注册 Agent 并提供终止函数（向后兼容）

        Args:
            agent_id: Agent 唯一标识
            agent_type: Agent 类型
            card: AgentCard
            terminate_fn: 终止函数
        """
        with self._agents_lock:
            entry = AgentEntry(
                agent_id=agent_id,
                agent_type=agent_type,
                card=card,
                terminate_fn=terminate_fn,
                state=AgentLifecycleState.RUNNING,
            )
            self._agents[agent_id] = entry

    def unregister(self, agent_id: str) -> bool:
        """注销 Agent

        Args:
            agent_id: Agent 唯一标识

        Returns:
            是否成功注销
        """
        with self._agents_lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return False

            # 如果正在运行，先停止
            if entry.state == AgentLifecycleState.RUNNING:
                self._kill_agent(entry)

            del self._agents[agent_id]
            return True

    # ============ 消息投递 ============

    def send_message(self, target_agent_id: str, task: Task) -> bool:
        """发送消息到目标 Agent（懒加载激活）

        Args:
            target_agent_id: 目标 Agent ID
            task: A2A Task

        Returns:
            是否成功投递
        """
        with self._agents_lock:
            entry = self._agents.get(target_agent_id)
            if entry is None:
                return False

            # 投递到 Inbox
            entry.inbox.put(task)
            entry.last_active = datetime.now()

            # 如果是 pending 状态，激活 Agent
            if entry.state == AgentLifecycleState.PENDING and entry.factory:
                self._activate_agent(entry)
            elif entry.state == AgentLifecycleState.IDLE:
                # 从 idle 恢复到 running
                entry.state = AgentLifecycleState.RUNNING

            return True

    def broadcast(self, task: Task, agent_types: list[str] | None = None) -> list[str]:
        """广播消息到多个 Agent

        Args:
            task: A2A Task
            agent_types: 目标 Agent 类型列表（None 表示所有）

        Returns:
            成功投递的 Agent ID 列表
        """
        dispatched = []
        with self._agents_lock:
            for agent_id, entry in self._agents.items():
                if agent_types is None or entry.agent_type in agent_types:
                    if self.send_message(agent_id, task):
                        dispatched.append(agent_id)
        return dispatched

    # ============ 能力发现 ============

    def find_agents_by_skill(self, skill_name: str) -> list[AgentCard]:
        """按技能查找 Agent

        Args:
            skill_name: 技能名称

        Returns:
            匹配的 AgentCard 列表
        """
        results = []
        with self._agents_lock:
            for entry in self._agents.values():
                if entry.state == AgentLifecycleState.RUNNING:
                    if entry.card.has_skill(skill_name):
                        results.append(entry.card)
        return results

    def list_agents(self) -> list[AgentCard]:
        """列出所有已注册的 Agent"""
        with self._agents_lock:
            return [entry.card for entry in self._agents.values()]

    # ============ 生命周期管理 ============

    def _activate_agent(self, entry: AgentEntry) -> None:
        """激活 Agent（pending → running）"""
        if entry.state != AgentLifecycleState.PENDING or not entry.factory:
            return

        # 创建 Agent 实例
        entry.instance = entry.factory()
        entry.state = AgentLifecycleState.RUNNING
        entry.last_active = datetime.now()

        # 启动工作线程
        entry.thread = threading.Thread(
            target=self._agent_loop,
            args=(entry,),
            daemon=True,
            name=f"agent-{entry.agent_id}",
        )
        entry.thread.start()

    def _agent_loop(self, entry: AgentEntry) -> None:
        """Agent 工作线程主循环"""
        while entry.state == AgentLifecycleState.RUNNING:
            try:
                # 从 Inbox 获取任务（带超时，以便检查状态）
                try:
                    task = entry.inbox.get(timeout=1.0)
                except queue.Empty:
                    # 检查是否应该进入 idle
                    idle_time = (datetime.now() - entry.last_active).total_seconds()
                    if idle_time >= self.IDLE_TIMEOUT and not entry.auto_start:
                        entry.state = AgentLifecycleState.IDLE
                        break
                    continue

                # 执行任务
                if entry.instance and hasattr(entry.instance, 'handle_task'):
                    try:
                        entry.instance.handle_task(task)
                    except Exception as e:
                        print(f"Agent {entry.agent_id} task error: {e}")

                entry.last_active = datetime.now()

            except Exception as e:
                print(f"Agent {entry.agent_id} loop error: {e}")
                break

    def _kill_agent(self, entry: AgentEntry) -> None:
        """终止 Agent（idle → pending）"""
        # 如果有终止函数，调用它
        if entry.terminate_fn:
            try:
                entry.terminate_fn()
            except Exception:
                pass

        entry.state = AgentLifecycleState.PENDING
        entry.instance = None
        # 线程会自动退出（daemon=True）

    # ============ 空闲检查 ============

    def start_idle_checker(self) -> None:
        """启动空闲检查线程"""
        if self._idle_checker_thread is not None:
            return

        self._stop_event.clear()
        self._idle_checker_thread = threading.Thread(
            target=self._idle_checker_loop,
            daemon=True,
            name="idle-checker",
        )
        self._idle_checker_thread.start()

    def stop_idle_checker(self) -> None:
        """停止空闲检查线程"""
        self._stop_event.set()
        if self._idle_checker_thread:
            self._idle_checker_thread.join(timeout=2.0)
            self._idle_checker_thread = None

    def _idle_checker_loop(self) -> None:
        """空闲检查循环"""
        while not self._stop_event.is_set():
            with self._agents_lock:
                for entry in self._agents.values():
                    if entry.state == AgentLifecycleState.IDLE:
                        idle_time = (datetime.now() - entry.last_active).total_seconds()
                        if idle_time >= self.IDLE_TIMEOUT:
                            self._kill_agent(entry)

            # 每 30 秒检查一次
            self._stop_event.wait(30.0)

    # ============ 状态查询 ============

    def get_state(self, agent_id: str) -> AgentLifecycleState | None:
        """获取 Agent 状态"""
        with self._agents_lock:
            entry = self._agents.get(agent_id)
            return entry.state if entry else None

    def get_all_states(self) -> dict[str, AgentLifecycleState]:
        """获取所有 Agent 状态"""
        with self._agents_lock:
            return {aid: e.state for aid, e in self._agents.items()}

    def is_running(self) -> bool:
        """检查是否有 Agent 正在运行"""
        with self._agents_lock:
            return any(e.state == AgentLifecycleState.RUNNING for e in self._agents.values())

    # ============ 终止控制（向后兼容）============

    def terminate_all(self) -> list[str]:
        """终止所有 Agent，返回被终止的 agent_id 列表"""
        terminated = []
        with self._agents_lock:
            for agent_id, entry in self._agents.items():
                if entry.state == AgentLifecycleState.RUNNING:
                    self._kill_agent(entry)
                    terminated.append(agent_id)
        return terminated

    # ============ 旧接口（向后兼容）============

    def register_executor(self, plan_id: str, terminate_fn: Callable[[], None]) -> None:
        """注册 Executor 的终止函数（向后兼容）

        Args:
            plan_id: Plan ID
            terminate_fn: 终止函数
        """
        from agent.a2a.models import AgentCard

        with self._agents_lock:
            entry = AgentEntry(
                agent_id=plan_id,
                agent_type="executor",
                card=AgentCard(
                    id=plan_id,
                    name=f"Executor-{plan_id}",
                    description="Plan Executor",
                ),
                terminate_fn=terminate_fn,
                state=AgentLifecycleState.RUNNING,
            )
            self._agents[plan_id] = entry
            self._running = True

    def unregister_executor(self, plan_id: str) -> None:
        """注销 Executor（向后兼容）"""
        with self._agents_lock:
            if plan_id in self._agents:
                del self._agents[plan_id]
            if not self._agents:
                self._running = False

    def get_terminator(self, plan_id: str) -> Callable[[], None] | None:
        """获取指定 plan_id 的终止函数（向后兼容）"""
        with self._agents_lock:
            entry = self._agents.get(plan_id)
            return entry.terminate_fn if entry else None

    def get_running_plan_ids(self) -> list[str]:
        """获取所有正在运行的 plan_id（向后兼容）"""
        with self._agents_lock:
            return [
                aid for aid, e in self._agents.items()
                if e.agent_type == "executor" and e.state == AgentLifecycleState.RUNNING
            ]


# ============ 全局实例 ============

def get_registry() -> AgentRegistry:
    """获取 Registry 单例"""
    return AgentRegistry.get_instance()


# 向后兼容
agent_registry = AgentRegistry.get_instance()


def terminate() -> list[str]:
    """终止所有 Agent（向后兼容）"""
    return agent_registry.terminate_all()

