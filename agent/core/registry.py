"""统一 Agent Registry

管理：
- Agent Group（同类型多实例动态管理）
- Agent 生命周期（pending/running/idle）
- Inbox（消息队列）
- 中心化路由（dispatch → 智能选择实例）
- 空闲超时回收
- 终止控制

设计说明：
- 发送方通过 dispatch(target_type, task) 投递消息
- Dispatcher 根据路由策略选择目标实例
- 优先空闲实例，无空闲则轮询
- 无实例时动态创建
"""

from __future__ import annotations

import queue
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.a2a.models import AgentCard, Task, Message


class AgentLifecycleState(str, Enum):
    """Agent 生命周期状态"""

    PENDING = "pending"  # 未初始化
    RUNNING = "running"  # 正在运行
    IDLE = "idle"  # 空闲（等待回收）


@dataclass
class AgentEntry:
    """Agent 注册条目"""

    agent_id: str
    agent_type: str
    group_type: str  # 所属组类型（如 "plan", "research"）
    card: Any  # AgentCard（延迟导入）
    factory: Callable[[], Any]  # Agent 工厂函数
    auto_start: bool = False  # 是否自动启动

    # 运行时状态
    state: AgentLifecycleState = AgentLifecycleState.PENDING
    instance: Any = None  # Agent 实例
    thread: threading.Thread | None = None  # 工作线程
    inbox: queue.Queue = field(default_factory=queue.Queue)
    last_active: datetime = field(default_factory=datetime.now)

    # 任务结果回传
    results: dict[str, Any] = field(default_factory=dict)
    result_events: dict[str, threading.Event] = field(default_factory=dict)

    # 终止函数（向后兼容）
    terminate_fn: Callable[[], None] | None = None


@dataclass
class AgentGroup:
    """Agent 组 - 管理同类型多个实例"""

    group_type: str  # 如 "plan", "research", "analysis"
    card: Any  # AgentCard 模板
    factory: Callable[[], Any] | None = None  # Agent 工厂函数
    max_instances: int = 5  # 最大实例数

    # 实例管理
    instances: list[AgentEntry] = field(default_factory=list)
    round_robin_index: int = 0

    def get_idle_instance(self) -> AgentEntry | None:
        """获取空闲实例"""
        for entry in self.instances:
            if entry.state == AgentLifecycleState.IDLE:
                return entry
        return None

    def get_running_instance(self) -> AgentEntry | None:
        """获取运行中的实例（轮询）"""
        if not self.instances:
            return None
        # 过滤出 running 的实例
        running = [e for e in self.instances if e.state == AgentLifecycleState.RUNNING]
        if not running:
            return None
        entry = running[self.round_robin_index % len(running)]
        self.round_robin_index = (self.round_robin_index + 1) % len(running)
        return entry

    def create_instance(self, registry: "AgentRegistry") -> AgentEntry:
        """动态创建新实例"""
        instance_id = f"{self.group_type}-{uuid.uuid4().hex[:6]}"
        entry = AgentEntry(
            agent_id=instance_id,
            agent_type=self.group_type,
            group_type=self.group_type,
            card=self.card,
            factory=self.factory,
        )
        self.instances.append(entry)
        registry._agents[instance_id] = entry
        return entry

    def get_instance_count(self) -> int:
        """获取当前实例数"""
        # 过滤掉已回收的（state=PENDING 且 instance=None）
        active = [
            e
            for e in self.instances
            if e.state != AgentLifecycleState.PENDING or e.instance is not None
        ]
        return len(active)


class AgentRegistry:
    """统一 Agent Registry（单例）

    功能：
    - 管理 Agent Group（同类型多实例）
    - 管理 Agent 生命周期
    - 管理 Inbox（消息队列）
    - 支持中心化路由（dispatch）
    - 支持动态创建
    - 支持空闲超时回收
    - 与 Transport 协作
    """

    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()

    # 空闲超时时间（秒）
    IDLE_TIMEOUT = 300  # 5 分钟

    def __init__(self):
        self._agents: dict[str, AgentEntry] = {}
        self._groups: dict[str, AgentGroup] = {}
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

    # ============ Group 注册 ============

    def register_group(
        self,
        group_type: str,
        card: AgentCard,
        factory: Callable[[], Any] | None = None,
        max_instances: int = 5,
    ) -> None:
        """注册 Agent Group（同类型可多实例）

        Args:
            group_type: 组类型（如 "plan", "research"）
            card: AgentCard 模板
            factory: Agent 工厂函数
            max_instances: 最大实例数
        """
        with self._agents_lock:
            self._groups[group_type] = AgentGroup(
                group_type=group_type,
                card=card,
                factory=factory,
                max_instances=max_instances,
            )

    # ============ 单实例注册（向后兼容）============

    def register(
        self,
        agent_id: str,
        agent_type: str,
        card: AgentCard,
        factory: Callable[[], Any] | None = None,
        auto_start: bool = False,
    ) -> None:
        """注册单个 Agent（向后兼容，用于 MainAgent 等单实例）

        Args:
            agent_id: Agent 唯一标识
            agent_type: Agent 类型
            card: AgentCard（能力描述）
            factory: Agent 工厂函数（可选）
            auto_start: 是否自动启动
        """
        with self._agents_lock:
            entry = AgentEntry(
                agent_id=agent_id,
                agent_type=agent_type,
                group_type=agent_type,
                card=card,
                factory=factory or (lambda: None),
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
        """注册 Agent 并提供终止函数（向后兼容）"""
        with self._agents_lock:
            entry = AgentEntry(
                agent_id=agent_id,
                agent_type=agent_type,
                group_type=agent_type,
                card=card,
                factory=lambda: None,
                terminate_fn=terminate_fn,
                state=AgentLifecycleState.RUNNING,
            )
            self._agents[agent_id] = entry

    def unregister(self, agent_id: str) -> bool:
        """注销 Agent"""
        with self._agents_lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return False

            if entry.state == AgentLifecycleState.RUNNING:
                self._kill_agent(entry)

            del self._agents[agent_id]
            return True

    # ============ 中心化路由 Dispatch ============

    def dispatch(
        self,
        target_type: str,
        task: Task,
        sender_id: str = "main",
    ) -> str | None:
        """中心化路由：投递任务到目标类型的最佳实例

        路由策略：
        1. 优先选择空闲实例
        2. 无空闲则轮询选择运行中实例
        3. 无实例则动态创建（不超过 max_instances）

        Args:
            target_type: 目标 Agent 类型（如 "plan", "research"）
            task: A2A Task（包含 sender_id）
            sender_id: 发送方 ID

        Returns:
            目标实例的 agent_id，失败返回 None
        """
        with self._agents_lock:
            group = self._groups.get(target_type)
            if group is None:
                # 回退：尝试按 agent_id 直接查找
                return self._dispatch_to_agent_id(target_type, task)

            # 1. 优先空闲实例
            entry = group.get_idle_instance()
            if entry:
                entry.state = AgentLifecycleState.RUNNING
                return self._deliver_to_entry(entry, task, sender_id)

            # 2. 轮询运行中实例
            entry = group.get_running_instance()
            if entry:
                return self._deliver_to_entry(entry, task, sender_id)

            # 3. 动态创建新实例
            if group.get_instance_count() < group.max_instances:
                entry = group.create_instance(self)
                return self._deliver_to_entry(entry, task, sender_id)

            # 4. 已达上限，轮询
            entry = group.get_running_instance()
            if entry:
                return self._deliver_to_entry(entry, task, sender_id)

            return None

    def _dispatch_to_agent_id(self, agent_id: str, task: Task) -> str | None:
        """按 agent_id 直接投递（回退路径）"""
        entry = self._agents.get(agent_id)
        if entry is None:
            return None
        return self._deliver_to_entry(entry, task, "main")

    def _deliver_to_entry(self, entry: AgentEntry, task: Task, sender_id: str) -> str:
        """投递任务到 Agent 实例"""
        # 设置 sender_id 到 metadata
        if task.metadata is None:
            task.metadata = {}
        task.metadata["sender_id"] = sender_id

        # 投递到 Inbox
        entry.inbox.put(task)
        entry.last_active = datetime.now()

        # 如果是 pending 状态，激活 Agent
        if entry.state == AgentLifecycleState.PENDING and entry.factory:
            self._activate_agent(entry)
        elif entry.state == AgentLifecycleState.IDLE:
            entry.state = AgentLifecycleState.RUNNING

        return entry.agent_id

    # ============ 旧 send_message（向后兼容）============

    def send_message(self, target_agent_id: str, task: Task) -> bool:
        """发送消息到指定 Agent ID（向后兼容）"""
        with self._agents_lock:
            entry = self._agents.get(target_agent_id)
            if entry is None:
                return False

            entry.inbox.put(task)
            entry.last_active = datetime.now()

            if entry.state == AgentLifecycleState.PENDING and entry.factory:
                self._activate_agent(entry)
            elif entry.state == AgentLifecycleState.IDLE:
                entry.state = AgentLifecycleState.RUNNING

            return True

    def broadcast(self, task: Task, agent_types: list[str] | None = None) -> list[str]:
        """广播消息到多个 Agent"""
        dispatched = []
        with self._agents_lock:
            for agent_id, entry in self._agents.items():
                if agent_types is None or entry.agent_type in agent_types:
                    if self.send_message(agent_id, task):
                        dispatched.append(agent_id)
        return dispatched

    # ============ 能力发现 ============

    def find_agents_by_skill(self, skill_name: str) -> list[AgentCard]:
        """按技能查找 Agent"""
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

    def list_groups(self) -> dict[str, dict[str, Any]]:
        """列出所有 Agent Group 及其实例状态"""
        with self._agents_lock:
            result = {}
            for group_type, group in self._groups.items():
                instances = []
                for entry in group.instances:
                    instances.append(
                        {
                            "agent_id": entry.agent_id,
                            "state": entry.state.value,
                            "last_active": entry.last_active.isoformat(),
                        }
                    )
                result[group_type] = {
                    "group_type": group_type,
                    "instance_count": len(instances),
                    "max_instances": group.max_instances,
                    "instances": instances,
                }
            return result

    # ============ 生命周期管理 ============

    def _activate_agent(self, entry: AgentEntry) -> None:
        """激活 Agent（pending → running）"""
        if entry.state != AgentLifecycleState.PENDING or not entry.factory:
            return

        entry.instance = entry.factory()
        entry.state = AgentLifecycleState.RUNNING
        entry.last_active = datetime.now()

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
                try:
                    task = entry.inbox.get(timeout=1.0)
                except queue.Empty:
                    idle_time = (datetime.now() - entry.last_active).total_seconds()
                    if idle_time >= self.IDLE_TIMEOUT and not entry.auto_start:
                        entry.state = AgentLifecycleState.IDLE
                        if entry.instance and hasattr(entry.instance, "on_idle"):
                            try:
                                entry.instance.on_idle()
                            except Exception:
                                pass
                        break
                    continue

                if entry.instance and hasattr(entry.instance, "handle_task"):
                    try:
                        result = entry.instance.handle_task(task)
                        task_id = getattr(task, "id", str(id(task)))
                        entry.results[task_id] = result
                        if task_id in entry.result_events:
                            entry.result_events[task_id].set()
                        self._notify_inbox(entry, task, result)
                    except Exception as e:
                        task_id = getattr(task, "id", str(id(task)))
                        entry.results[task_id] = {"success": False, "error": str(e)}
                        if task_id in entry.result_events:
                            entry.result_events[task_id].set()

                entry.last_active = datetime.now()

            except Exception as e:
                print(f"Agent {entry.agent_id} loop error: {e}")
                break

    def _notify_inbox(self, entry: AgentEntry, task: Any, result: Any) -> None:
        """将 Agent 执行结果写入全局 Inbox，通知 MainAgent"""
        try:
            from agent.a2a.dispatcher import get_inbox, TaskResult, TaskResultStatus

            inbox = get_inbox()
            task_id = getattr(task, "id", str(id(task)))
            sender_id = (
                getattr(task, "metadata", {}).get("sender_id", "unknown")
                if hasattr(task, "metadata")
                else "unknown"
            )

            if isinstance(result, dict):
                success = result.get("success", True)
                result_text = (
                    result.get("summary") or result.get("result") or str(result)
                )
                error = result.get("error") if not success else None
            else:
                success = True
                result_text = str(result) if result else "completed"
                error = None

            inbox.put(
                TaskResult(
                    plan_id="",
                    task_id=task_id,
                    status=TaskResultStatus.SUCCESS
                    if success
                    else TaskResultStatus.FAILED,
                    result=result_text,
                    error=error,
                    worker_id=entry.agent_id,
                )
            )
        except Exception:
            pass

    def _kill_agent(self, entry: AgentEntry) -> None:
        """终止 Agent（idle → pending）"""
        if entry.terminate_fn:
            try:
                entry.terminate_fn()
            except Exception:
                pass

        entry.state = AgentLifecycleState.PENDING
        entry.instance = None

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
            return any(
                e.state == AgentLifecycleState.RUNNING for e in self._agents.values()
            )

    # ============ 终止控制 ============

    def terminate_all(self) -> list[str]:
        """终止所有 Agent"""
        terminated = []
        with self._agents_lock:
            for agent_id, entry in self._agents.items():
                if entry.state == AgentLifecycleState.RUNNING:
                    self._kill_agent(entry)
                    terminated.append(agent_id)
        return terminated

    # ============ 旧接口（向后兼容）============

    def register_executor(self, plan_id: str, terminate_fn: Callable[[], None]) -> None:
        """注册 Executor 的终止函数（向后兼容）"""
        from agent.a2a.models import AgentCard

        with self._agents_lock:
            entry = AgentEntry(
                agent_id=plan_id,
                agent_type="executor",
                group_type="executor",
                card=AgentCard(
                    id=plan_id,
                    name=f"Executor-{plan_id}",
                    description="Plan Executor",
                ),
                factory=lambda: None,
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
                aid
                for aid, e in self._agents.items()
                if e.agent_type == "executor" and e.state == AgentLifecycleState.RUNNING
            ]


# ============ 全局实例 ============


def get_registry() -> AgentRegistry:
    """获取 Registry 单例"""
    return AgentRegistry.get_instance()


agent_registry = AgentRegistry.get_instance()


def terminate() -> list[str]:
    """终止所有 Agent（向后兼容）"""
    return agent_registry.terminate_all()
