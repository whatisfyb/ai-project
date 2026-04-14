"""A2A Transport 层 - 消息路由

提供 Agent 间通信的传输层实现：
- InMemoryTransport: 同进程内存通信，零开销
"""

import threading
import uuid
from abc import ABC, abstractmethod
from typing import Callable, Any
from datetime import datetime
from weakref import WeakSet

from agent.a2a.models import (
    Task,
    TaskStatus,
    TaskEvent,
    Message,
    AgentCard,
    TransportMessage,
)


# ============ Callback 类型 ============

TaskCallback = Callable[[Task, TaskEvent], None]
MessageCallback = Callable[[TransportMessage], None]


# ============ Transport 抽象基类 ============

class Transport(ABC):
    """Transport 抽象基类

    定义 Agent 间通信的标准接口。
    可扩展为 HTTP、WebSocket、gRPC 等不同传输层。
    """

    @abstractmethod
    def register_agent(self, agent_id: str, card: AgentCard, handler: Callable) -> None:
        """注册 Agent

        Args:
            agent_id: Agent 唯一标识
            card: Agent 能力声明
            handler: 消息处理函数
        """
        pass

    @abstractmethod
    def unregister_agent(self, agent_id: str) -> None:
        """注销 Agent"""
        pass

    @abstractmethod
    def get_agent_card(self, agent_id: str) -> AgentCard | None:
        """获取 Agent Card"""
        pass

    @abstractmethod
    def create_task(self, sender_id: str, receiver_id: str, initial_message: Message | None = None) -> Task:
        """创建 Task

        Args:
            sender_id: 发送方 Agent ID
            receiver_id: 接收方 Agent ID
            initial_message: 初始消息（可选）

        Returns:
            创建的 Task
        """
        pass

    # ============ A2A 标准方法 ============

    @abstractmethod
    def message_send(self, task: Task, message: Message) -> Task:
        """A2A message/send - 发送消息

        Args:
            task: 目标 Task
            message: 消息内容

        Returns:
            更新后的 Task
        """
        pass

    @abstractmethod
    def tasks_get(self, task_id: str) -> Task | None:
        """A2A tasks/get - 获取 Task"""
        pass

    @abstractmethod
    def tasks_cancel(self, task_id: str) -> Task | None:
        """A2A tasks/cancel - 取消 Task"""
        pass

    @abstractmethod
    def tasks_subscribe(self, task_id: str, callback: TaskCallback) -> None:
        """A2A tasks/subscribe - 订阅 Task 事件"""
        pass

    def tasks_unsubscribe(self, task_id: str, callback: TaskCallback) -> None:
        """取消订阅 Task 事件"""
        pass

    # ============ 内部管理方法（非 A2A 标准）============

    @abstractmethod
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """更新 Task 状态（内部方法）"""
        pass

    @abstractmethod
    def list_agents(self) -> list[AgentCard]:
        """列出所有已注册的 Agent"""
        pass


# ============ InMemoryTransport ============

class InMemoryTransport(Transport):
    """内存 Transport - 同进程零开销通信

    特点：
    - 直接内存调用，无序列化开销
    - 线程安全
    - 支持事件订阅
    - 支持 Agent 发现

    使用场景：
    - 同进程内的 Agent 协作
    - MainAgent 与 Workers 通信
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Agent 注册表
        # agent_id -> {"card": AgentCard, "handler": Callable}
        self._agents: dict[str, dict[str, Any]] = {}

        # Task 存储
        # task_id -> Task
        self._tasks: dict[str, Task] = {}

        # 订阅者
        # task_id -> set[TaskCallback]
        self._subscribers: dict[str, set[TaskCallback]] = {}

        # 全局订阅者（监听所有 Task 事件）
        self._global_subscribers: WeakSet[TaskCallback] = WeakSet()

    # ============ Agent 管理 ============

    def register_agent(self, agent_id: str, card: AgentCard, handler: Callable) -> None:
        """注册 Agent"""
        with self._lock:
            self._agents[agent_id] = {
                "card": card,
                "handler": handler,
            }

    def unregister_agent(self, agent_id: str) -> None:
        """注销 Agent"""
        with self._lock:
            self._agents.pop(agent_id, None)

    def get_agent_card(self, agent_id: str) -> AgentCard | None:
        """获取 Agent Card"""
        with self._lock:
            entry = self._agents.get(agent_id)
            return entry["card"] if entry else None

    def list_agents(self) -> list[AgentCard]:
        """列出所有已注册的 Agent"""
        with self._lock:
            return [entry["card"] for entry in self._agents.values()]

    def find_agents_by_skill(self, skill_name: str) -> list[AgentCard]:
        """根据技能查找 Agent"""
        with self._lock:
            return [
                entry["card"]
                for entry in self._agents.values()
                if entry["card"].has_skill(skill_name)
            ]

    # ============ Task 管理 ============

    def create_task(
        self,
        sender_id: str,
        receiver_id: str,
        initial_message: Message | None = None,
        plan_id: str | None = None,
        plantask_id: str | None = None,
    ) -> Task:
        """创建 Task

        Args:
            sender_id: 发送方 Agent ID
            receiver_id: 接收方 Agent ID
            initial_message: 初始消息
            plan_id: 关联的 Plan ID
            plantask_id: 关联的 PlanTask ID

        Returns:
            创建的 Task
        """
        task_id = str(uuid.uuid4())

        task = Task(
            id=task_id,
            status=TaskStatus.PENDING,
            sender_id=sender_id,
            receiver_id=receiver_id,
            plan_id=plan_id,
            plantask_id=plantask_id,
        )

        if initial_message:
            task.add_message(initial_message)

        with self._lock:
            self._tasks[task_id] = task

        # 触发事件
        self._emit_event(task, TaskEvent.CREATED)

        return task

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """更新 Task 状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            old_status = task.status
            task.update_status(status)

        # 触发状态变化事件
        self._emit_event(task, TaskEvent.STATUS_CHANGED)

        # 触发终态事件
        if status == TaskStatus.COMPLETED:
            self._emit_event(task, TaskEvent.COMPLETED)
        elif status == TaskStatus.FAILED:
            self._emit_event(task, TaskEvent.FAILED)

        return True

    def add_task_message(self, task_id: str, message: Message) -> bool:
        """向 Task 添加消息"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.add_message(message)

        self._emit_event(task, TaskEvent.MESSAGE_ADDED)
        return True

    def add_task_artifact(self, task_id: str, artifact_id: str, name: str, content: Any) -> bool:
        """向 Task 添加产物"""
        from agent.a2a.models import Artifact

        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            artifact = Artifact(id=artifact_id, name=name, content=content)
            task.add_artifact(artifact)

        self._emit_event(task, TaskEvent.ARTIFACT_ADDED)
        return True

    # ============ A2A 标准方法实现 ============

    def message_send(self, task: Task, message: Message) -> Task:
        """A2A message/send - 发送消息到目标 Agent

        直接调用目标 Agent 的 handler，无 HTTP 开销。

        Args:
            task: 目标 Task
            message: 消息内容

        Returns:
            更新后的 Task
        """
        receiver_id = task.receiver_id

        with self._lock:
            agent_entry = self._agents.get(receiver_id)
            if not agent_entry:
                raise ValueError(f"Agent not found: {receiver_id}")

            handler = agent_entry["handler"]

            # 添加消息到 Task 历史
            task.add_message(message)

            # 更新状态为 WORKING
            if task.status == TaskStatus.PENDING:
                task.update_status(TaskStatus.WORKING)

        # 触发分发事件
        self._emit_event(task, TaskEvent.DISPATCHED)

        # 直接调用 handler（内存调用）
        # handler 应该是非阻塞的
        handler(task, message)

        return task

    def tasks_get(self, task_id: str) -> Task | None:
        """A2A tasks/get - 获取 Task"""
        with self._lock:
            return self._tasks.get(task_id)

    def tasks_cancel(self, task_id: str) -> Task | None:
        """A2A tasks/cancel - 取消 Task"""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            if task.is_terminal():
                return task  # 已经是终态，无法取消

            task.update_status(TaskStatus.CANCELLED)

        self._emit_event(task, TaskEvent.STATUS_CHANGED)
        return task

    def tasks_subscribe(self, task_id: str, callback: TaskCallback) -> None:
        """A2A tasks/subscribe - 订阅特定 Task 的事件"""
        with self._lock:
            if task_id not in self._subscribers:
                self._subscribers[task_id] = set()
            self._subscribers[task_id].add(callback)

    def tasks_unsubscribe(self, task_id: str, callback: TaskCallback) -> None:
        """取消订阅"""
        with self._lock:
            if task_id in self._subscribers:
                self._subscribers[task_id].discard(callback)

    # ============ 内部管理方法 ============

    def send_message_to_agent(self, agent_id: str, task: Task, message: Message) -> Task:
        """向指定 Agent 发送消息（忽略 task.receiver_id）

        用于需要转发消息的场景。
        """
        with self._lock:
            agent_entry = self._agents.get(agent_id)
            if not agent_entry:
                raise ValueError(f"Agent not found: {agent_id}")

            handler = agent_entry["handler"]
            task.add_message(message)

        handler(task, message)
        return task

    def subscribe_all(self, callback: TaskCallback) -> None:
        """订阅所有 Task 事件"""
        with self._lock:
            self._global_subscribers.add(callback)

    def unsubscribe_all(self, callback: TaskCallback) -> None:
        """取消全局订阅"""
        with self._lock:
            self._global_subscribers.discard(callback)

    # ============ 事件触发 ============

    def _emit_event(self, task: Task, event: TaskEvent) -> None:
        """触发事件（内部方法）"""
        # 获取订阅者（线程安全）
        with self._lock:
            task_callbacks = list(self._subscribers.get(task.id, set()))
            global_callbacks = list(self._global_subscribers)

        # 在锁外调用回调（避免死锁）
        for callback in task_callbacks + global_callbacks:
            try:
                callback(task, event)
            except Exception as e:
                # 回调异常不应影响主流程
                import traceback
                traceback.print_exc()

    # ============ 辅助方法 ============

    def get_pending_tasks_for_agent(self, agent_id: str) -> list[Task]:
        """获取发给指定 Agent 的待处理 Task"""
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.receiver_id == agent_id and task.status == TaskStatus.PENDING
            ]

    def get_working_tasks_for_agent(self, agent_id: str) -> list[Task]:
        """获取指定 Agent 正在处理的 Task"""
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.receiver_id == agent_id and task.status == TaskStatus.WORKING
            ]

    def get_tasks_by_plan(self, plan_id: str) -> list[Task]:
        """获取关联到指定 Plan 的所有 Task"""
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.plan_id == plan_id
            ]

    def clear_completed_tasks(self, max_age_seconds: int = 3600) -> int:
        """清理已完成的 Task

        Args:
            max_age_seconds: 最大存活时间（秒）

        Returns:
            清理的 Task 数量
        """
        now = datetime.now()
        to_remove = []

        with self._lock:
            for task_id, task in self._tasks.items():
                if task.is_terminal():
                    age = (now - task.updated_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                self._subscribers.pop(task_id, None)

        return len(to_remove)


# ============ 全局 Transport 实例 ============

_global_transport: InMemoryTransport | None = None
_global_transport_lock = threading.Lock()


def get_transport() -> InMemoryTransport:
    """获取全局 Transport 实例（单例）"""
    global _global_transport

    with _global_transport_lock:
        if _global_transport is None:
            _global_transport = InMemoryTransport()
        return _global_transport


def reset_transport() -> None:
    """重置全局 Transport（仅用于测试）"""
    global _global_transport

    with _global_transport_lock:
        _global_transport = None
