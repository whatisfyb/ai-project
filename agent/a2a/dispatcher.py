"""任务结果队列和调度监控

核心组件：
- TaskResultQueue: Worker 完成结果队列
- MainAgentState: MainAgent 忙闲状态
- DispatchMonitor: 监控线程，触发 MainAgent 处理结果
- InternalMessageChannel: 内部消息通道
"""

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from enum import Enum


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
    result: str | None = None
    error: str | None = None
    job_id: str | None = None
    worker_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


# ============ 结果队列 ============

class TaskResultQueue:
    """任务结果队列（线程安全）

    Worker 完成任务后将结果写入此队列。
    监控线程从中取出结果并触发 MainAgent 处理。
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
        """队列是否为空"""
        return self._queue.empty()

    def size(self) -> int:
        """队列大小"""
        return self._queue.qsize()


# ============ MainAgent 状态 ============

class MainAgentBusyState:
    """MainAgent 忙闲状态管理

    用于监控线程判断是否可以触发 MainAgent 处理结果。
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


# ============ 内部消息类型 ============

class InternalMessageType(str, Enum):
    """内部消息类型"""
    USER_INPUT = "user_input"           # 用户输入
    TASK_RESULTS = "task_results"       # 任务结果（来自 Worker）
    PLAN_COMPLETED = "plan_completed"   # Plan 完成
    SYSTEM = "system"                   # 系统消息


@dataclass
class InternalMessage:
    """内部消息"""
    type: InternalMessageType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)

    # 任务结果相关
    task_results: list[TaskResult] = field(default_factory=list)

    # Plan 完成相关
    plan_id: str | None = None
    plan_summary: str | None = None


# ============ 内部消息通道 ============

class InternalMessageChannel:
    """内部消息通道

    统一用户输入和系统消息的来源。
    REPL 主循环从此通道获取消息。
    """

    def __init__(self):
        self._queue: queue.Queue[InternalMessage] = queue.Queue()

    def put_user_input(self, content: str) -> None:
        """放入用户输入"""
        self._queue.put(InternalMessage(
            type=InternalMessageType.USER_INPUT,
            content=content,
        ))

    def put_task_results(self, results: list[TaskResult]) -> None:
        """放入任务结果"""
        self._queue.put(InternalMessage(
            type=InternalMessageType.TASK_RESULTS,
            content=f"收到 {len(results)} 个任务结果",
            task_results=results,
        ))

    def put_plan_completed(self, plan_id: str, summary: str) -> None:
        """放入 Plan 完成消息"""
        self._queue.put(InternalMessage(
            type=InternalMessageType.PLAN_COMPLETED,
            content=summary,
            plan_id=plan_id,
            plan_summary=summary,
        ))

    def put_system(self, content: str) -> None:
        """放入系统消息"""
        self._queue.put(InternalMessage(
            type=InternalMessageType.SYSTEM,
            content=content,
        ))

    def get(self, timeout: float | None = None) -> InternalMessage | None:
        """获取消息（可阻塞）"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_nowait(self) -> InternalMessage | None:
        """获取消息（非阻塞）"""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def is_empty(self) -> bool:
        """队列是否为空"""
        return self._queue.empty()


# ============ 调度监控器 ============

class DispatchMonitor:
    """调度监控器

    后台线程，监控任务结果队列和 MainAgent 状态。
    当 MainAgent 空闲且有任务结果时，触发处理。
    """

    def __init__(
        self,
        result_queue: TaskResultQueue,
        agent_state: MainAgentBusyState,
        message_channel: InternalMessageChannel,
        check_interval: float = 0.5,
    ):
        self.result_queue = result_queue
        self.agent_state = agent_state
        self.message_channel = message_channel
        self.check_interval = check_interval

        self._running = False
        self._thread: threading.Thread | None = None

        # 结果累积
        self._pending_results: list[TaskResult] = []

    def start(self) -> None:
        """启动监控"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="DispatchMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """停止监控"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _monitor_loop(self) -> None:
        """监控循环

        当 MainAgent 空闲且有任务结果时：
        1. 打包结果
        2. 构造消息
        3. 调用 MainAgent.chat_async（通过回调）
        """
        import time

        while self._running:
            try:
                # 1. 从队列取出新结果
                new_results = self.result_queue.get_all()
                if new_results:
                    self._pending_results.extend(new_results)

                # 2. 检查是否有待处理结果
                if not self._pending_results:
                    time.sleep(self.check_interval)
                    continue

                # 3. 检查 MainAgent 是否空闲
                if self.agent_state.is_busy():
                    time.sleep(self.check_interval)
                    continue

                # 4. MainAgent 空闲，打包结果并触发回调
                results_to_process = self._pending_results.copy()
                self._pending_results.clear()

                # 触发回调（让 MainAgent 处理）
                self._trigger_callback(results_to_process)

                time.sleep(self.check_interval)

            except Exception as e:
                import traceback
                traceback.print_exc()
                time.sleep(self.check_interval)

    def _trigger_callback(self, results: list) -> None:
        """触发回调，让 MainAgent 处理任务结果

        通过 set_busy + 回调函数的方式，确保线程安全。
        """
        if not results:
            return

        # 标记 MainAgent 为忙碌
        self.agent_state.set_busy()

        try:
            # 调用注册的回调函数
            if self._callback:
                self._callback(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            # 回调结束后，状态由回调函数内部控制
            # 这里不 set_idle，让回调函数控制
            pass

    _callback = None

    def set_callback(self, callback: Callable) -> None:
        """设置回调函数

        Args:
            callback: 回调函数，接收 list[TaskResult] 参数
        """
        self._callback = callback

    def _process_task_results(self, results: list) -> None:
        """处理任务结果

        1. 更新 PlanTask 状态（已在 Worker 中完成）
        2. 检查是否有后续任务就绪
        3. 如果有，分发后续任务
        4. 如果全部完成，写入通知
        """
        from collections import defaultdict
        from store.plan import PlanStore
        from agent.a2a.transport import get_transport
        from agent.a2a.models import TaskStatus, Message, Part

        store = PlanStore()
        transport = get_transport()

        # 按 plan_id 分组
        results_by_plan = defaultdict(list)
        for result in results:
            results_by_plan[result.plan_id].append(result)

        for plan_id, plan_results in results_by_plan.items():
            # 检查后续任务
            pending_tasks = store.get_pending_tasks(plan_id)

            if pending_tasks:
                # 有后续任务，分发
                for plantask in pending_tasks[:2]:  # 限制并发数
                    # 找到可用的 Worker
                    workers = transport.find_agents_by_skill("execute_plantask")
                    if workers:
                        worker = workers[0]
                        # 创建 A2A Task
                        a2a_task = transport.create_task(
                            sender_id="dispatcher",
                            receiver_id=worker.id,
                            initial_message=Message(
                                role="user",
                                parts=[Part.plantask(plan_id=plan_id, task_id=plantask.id)]
                            ),
                            plan_id=plan_id,
                            plantask_id=plantask.id,
                        )
                        # 发送给 Worker
                        transport.send_message(a2a_task, Message(
                            role="user",
                            parts=[Part.plantask(plan_id=plan_id, task_id=plantask.id)]
                        ))

            else:
                # 检查是否全部完成
                all_done = store.check_all_done(plan_id)
                if all_done:
                    # 全部完成，可以写入通知到消息通道
                    plan = store.load_plan(plan_id)
                    if plan and plan.summarized_result:
                        self.message_channel.put_plan_completed(
                            plan_id=plan_id,
                            summary=plan.summarized_result
                        )


# ============ 全局实例 ============

_global_result_queue: TaskResultQueue | None = None
_global_agent_state: MainAgentBusyState | None = None
_global_message_channel: InternalMessageChannel | None = None
_global_dispatch_monitor: DispatchMonitor | None = None

_global_lock = threading.Lock()


def get_result_queue() -> TaskResultQueue:
    """获取全局结果队列"""
    global _global_result_queue
    with _global_lock:
        if _global_result_queue is None:
            _global_result_queue = TaskResultQueue()
        return _global_result_queue


def get_agent_state() -> MainAgentBusyState:
    """获取全局 Agent 状态"""
    global _global_agent_state
    with _global_lock:
        if _global_agent_state is None:
            _global_agent_state = MainAgentBusyState()
        return _global_agent_state


def get_message_channel() -> InternalMessageChannel:
    """获取全局消息通道"""
    global _global_message_channel
    with _global_lock:
        if _global_message_channel is None:
            _global_message_channel = InternalMessageChannel()
        return _global_message_channel


def get_dispatch_monitor() -> DispatchMonitor:
    """获取全局调度监控器"""
    global _global_dispatch_monitor
    if _global_dispatch_monitor is None:
        # 在锁外获取依赖以避免死锁
        result_queue = get_result_queue()
        agent_state = get_agent_state()
        message_channel = get_message_channel()

        with _global_lock:
            if _global_dispatch_monitor is None:
                _global_dispatch_monitor = DispatchMonitor(
                    result_queue=result_queue,
                    agent_state=agent_state,
                    message_channel=message_channel,
                )
    return _global_dispatch_monitor


def init_dispatch_system() -> None:
    """初始化调度系统"""
    get_dispatch_monitor().start()


def shutdown_dispatch_system() -> None:
    """关闭调度系统"""
    global _global_dispatch_monitor
    if _global_dispatch_monitor:
        _global_dispatch_monitor.stop()
