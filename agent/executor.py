"""PlanExecutor - Plan 执行协调器

协调多个 TaskWorker 并行执行任务，提供进度显示。
支持用户中断处理和 Worker 健康检查、自动替换。
"""

import signal
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from agent.plan_store import PlanStore
from agent.models import Task, Plan
from agent.worker import TaskWorker, set_interrupt, clear_interrupt, is_interrupted


class WorkerRegistry:
    """线程安全的 Worker 注册表

    用于 PlanExecutor 监控 Worker 的运行状态。
    """

    def __init__(self):
        self._lock = threading.Lock()
        # worker_id -> {"thread": Thread, "task_id": str | None}
        self._workers: dict[str, dict] = {}

    def register(self, worker_id: str, thread: threading.Thread, task_id: Optional[str] = None) -> None:
        """注册 Worker"""
        with self._lock:
            self._workers[worker_id] = {
                "thread": thread,
                "task_id": task_id,
            }

    def update_task(self, worker_id: str, task_id: Optional[str]) -> None:
        """更新 Worker 当前执行的任务"""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id]["task_id"] = task_id

    def deregister(self, worker_id: str) -> None:
        """注销 Worker"""
        with self._lock:
            self._workers.pop(worker_id, None)

    def get_worker_info(self, worker_id: str) -> Optional[dict]:
        """获取 Worker 信息"""
        with self._lock:
            return self._workers.get(worker_id)

    def get_all_workers(self) -> dict[str, dict]:
        """获取所有 Worker 的信息（副本）"""
        with self._lock:
            return {k: v.copy() for k, v in self._workers.items()}

    def get_worker_ids(self) -> list[str]:
        """获取所有 Worker ID"""
        with self._lock:
            return list(self._workers.keys())


class ProgressTracker:
    """进度追踪与显示"""

    def __init__(self, plan_id: str, store: PlanStore | None = None):
        self.plan_id = plan_id
        self.store = store or PlanStore()

    def render(self) -> Panel:
        """渲染当前进度"""
        plan = self.store.load_plan(self.plan_id)
        if not plan:
            return Panel("Plan 不存在", title="错误")

        lines = [f"[bold]Goal:[/bold] {plan.goal}", ""]

        for task in plan.tasks:
            # 状态图标
            if task.status == "completed":
                icon = "[green]✓[/green]"
            elif task.status == "failed":
                icon = "[red]✗[/red]"
            else:
                # 检查依赖是否满足
                task_status = {t.id: t.status for t in plan.tasks}
                deps_ok = all(
                    task_status.get(d) == "completed" for d in task.dependencies
                )
                if task.claimed_by:
                    icon = "[yellow]⏳[/yellow]"
                elif deps_ok:
                    icon = "[blue]○[/blue]"
                else:
                    icon = "[dim]⏸[/dim]"

            # 任务行
            claimed = f" [[cyan]{task.claimed_by}[/cyan]]" if task.claimed_by else ""
            desc = task.description[:50] + "..." if len(task.description) > 50 else task.description
            lines.append(f"  {icon} [{task.id}] {desc}{claimed}")

        # 状态栏
        total = len(plan.tasks)
        completed = sum(1 for t in plan.tasks if t.status == "completed")
        failed = sum(1 for t in plan.tasks if t.status == "failed")
        progress_bar = self._progress_bar(completed + failed, total)
        status = f"\n[bold]{progress_bar}[/bold] {completed + failed}/{total} 完成"

        return Panel("\n".join(lines), title=f"Plan: {self.plan_id[:20]}...", subtitle=status)

    def _progress_bar(self, done: int, total: int, width: int = 20) -> str:
        """生成进度条"""
        if total == 0:
            return "[" + " " * width + "]"
        filled = int(width * done / total)
        return "[" + "█" * filled + "░" * (width - filled) + "]"


class PlanExecutor:
    """Plan 执行器 (Supervisor 模式)

    协调多个 Worker 并行执行任务。
    监控 Worker 健康状态，自动替换不健康的 Worker。
    Main Agent 不参与执行，只负责协调和汇总。
    """

    def __init__(
        self,
        plan_id: str,
        num_workers: int = 2,
        store: PlanStore | None = None,
        interrupt_event: threading.Event | None = None,
        task_timeout: int = 300,
    ):
        """初始化执行器

        Args:
            plan_id: Plan ID
            num_workers: Worker 数量
            store: PlanStore 实例
            interrupt_event: 外部中断事件，用于从其他线程触发中断
            task_timeout: 任务超时时间（秒）
        """
        self.plan_id = plan_id
        self.num_workers = num_workers
        self.store = store or PlanStore()
        self.tracker = ProgressTracker(plan_id, self.store)
        self.interrupt_event = interrupt_event
        self.task_timeout = task_timeout

        # Worker 注册表
        self.registry = WorkerRegistry()

        # 线程池
        self.executor: ThreadPoolExecutor | None = None
        self.futures: dict[str, Future] = {}  # worker_id -> Future
        self._original_sigint_handler = None

        # Supervisor 配置
        self._poll_interval = 2  # 轮询间隔（秒）
        self._stale_threshold = task_timeout  # 任务超时阈值

    def _setup_interrupt_handler(self):
        """设置中断处理器（仅在主线程中有效，且没有外部 interrupt_event）"""
        def handle_interrupt(signum, frame):
            set_interrupt()  # 设置全局中断标志
            # 如果有外部 interrupt_event，也设置它
            if self.interrupt_event:
                self.interrupt_event.set()
            console = Console()
            console.print("\n[yellow]收到中断信号，正在停止所有 Worker...[/yellow]")

        # 如果有外部 interrupt_event，优先使用它
        # 否则设置 signal 处理（仅在主线程）
        if self.interrupt_event is None and threading.current_thread() is threading.main_thread():
            try:
                self._original_sigint_handler = signal.signal(signal.SIGINT, handle_interrupt)
            except (ValueError, OSError):
                # 信号处理失败（非主线程或其他错误），跳过
                self._original_sigint_handler = None

    def _restore_interrupt_handler(self):
        """恢复原始中断处理器"""
        clear_interrupt()  # 清除全局中断标志
        if self._original_sigint_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            except (ValueError, OSError):
                pass
            self._original_sigint_handler = None

    def run(self) -> dict:
        """执行 Plan (Supervisor 模式)

        启动 Worker 并持续监控其健康状态，自动替换不健康的 Worker。
        等待所有任务完成或被中断，返回结果。

        Returns:
            执行结果
        """
        from agent.registry import agent_registry

        # 清除之前的中断标志
        clear_interrupt()
        self._setup_interrupt_handler()

        # 注册到全局 AgentRegistry
        agent_registry.register(self.plan_id, self)

        try:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self.futures = {}

            # 启动初始 Workers
            self._start_initial_workers()

            # Supervisor 轮询循环
            console = Console()
            with Live(self.tracker.render(), refresh_per_second=2, console=console) as live:
                while not self._all_done() and not is_interrupted():
                    # 检查并替换不健康的 Worker
                    self._check_and_replace_unhealthy_workers()

                    # 回收被遗弃的任务
                    self._reclaim_stale_tasks()

                    # 启动新 Worker 填补空缺
                    self._fill_worker_slots()

                    time.sleep(self._poll_interval)
                    live.update(self.tracker.render())

                # 最终更新
                live.update(self.tracker.render())

            # 如果被中断，释放已领取的任务
            if is_interrupted():
                plan = self.store.load_plan(self.plan_id)
                for task in plan.tasks:
                    if task.claimed_by and task.status == "pending":
                        self.store.release_task(self.plan_id, task.id)

            # 汇总结果（仅在没有被中断且全部完成时）
            plan = self.store.load_plan(self.plan_id)
            interrupted = is_interrupted()
            if not interrupted and self.store.check_all_completed(self.plan_id):
                summarized = self._summarize_results(plan)
                self.store.save_summarized_result(self.plan_id, summarized)
                plan = self.store.load_plan(self.plan_id)

            return {
                "plan_id": self.plan_id,
                "goal": plan.goal,
                "status": plan.status,
                "completed": sum(1 for t in plan.tasks if t.status == "completed"),
                "failed": sum(1 for t in plan.tasks if t.status == "failed"),
                "total": len(plan.tasks),
                "tasks": [
                    {"id": t.id, "status": t.status, "result": t.result}
                    for t in plan.tasks
                ],
                "summarized_result": plan.summarized_result,
                "interrupted": interrupted,
            }

        finally:
            # 从全局 AgentRegistry 注销
            from agent.registry import agent_registry
            agent_registry.unregister(self.plan_id)
            # 关闭所有 Worker
            self._shutdown_all_workers()
            self._restore_interrupt_handler()

    def _start_initial_workers(self) -> None:
        """启动初始的 Worker"""
        for i in range(self.num_workers):
            worker_id = f"worker_{i + 1}"
            self._start_worker(worker_id)

    def _start_worker(self, worker_id: str) -> None:
        """启动单个 Worker"""
        if self.executor is None:
            return

        worker = TaskWorker(
            worker_id=worker_id,
            plan_id=self.plan_id,
            store=self.store,
            registry=self.registry,
        )

        future = self.executor.submit(worker.run)
        self.futures[worker_id] = future

    def _check_and_replace_unhealthy_workers(self) -> None:
        """检查并替换不健康的 Worker"""
        plan = self.store.load_plan(self.plan_id)
        if not plan:
            return

        # 构建 task_id -> worker_id 映射（被占用的任务）
        task_to_worker = {}
        for task in plan.tasks:
            if task.claimed_by and task.status == "pending":
                task_to_worker[task.id] = task.claimed_by

        for worker_id, future in list(self.futures.items()):
            # 检查1：线程已结束
            if future.done():
                # 线程已结束，从注册表移除（Worker 会自动注销）
                if worker_id in self.registry.get_worker_ids():
                    self.registry.deregister(worker_id)
                continue

            # 检查2：检查 Worker 是否还活着
            worker_info = self.registry.get_worker_info(worker_id)
            if not worker_info:
                # Worker 不在注册表中，说明已经异常退出
                self._replace_worker(worker_id)
                continue

            thread = worker_info.get("thread")
            if thread and not thread.is_alive():
                # 线程已死亡
                self._replace_worker(worker_id)

    def _replace_worker(self, worker_id: str) -> None:
        """替换一个不健康的 Worker"""
        console = Console()
        console.print(f"[yellow]Worker {worker_id} 不健康，正在替换...[/yellow]")

        # 注销旧的 Worker
        self.registry.deregister(worker_id)

        # 取消旧的 Future（如果可以）
        if worker_id in self.futures:
            old_future = self.futures[worker_id]
            if not old_future.done():
                old_future.cancel()
            del self.futures[worker_id]

        # 启动新的 Worker
        self._start_worker(worker_id)

    def _reclaim_stale_tasks(self) -> None:
        """回收被遗弃的任务（Worker 已死亡但任务未完成）"""
        plan = self.store.load_plan(self.plan_id)
        if not plan:
            return

        alive_workers = set()
        for worker_id, future in self.futures.items():
            if not future.done():
                alive_workers.add(worker_id)

        for task in plan.tasks:
            if task.status == "pending" and task.claimed_by:
                worker_id = task.claimed_by
                # Worker 已不在活跃列表中，回收任务
                if worker_id not in alive_workers:
                    console = Console()
                    console.print(f"[yellow]任务 {task.id} 被 Worker {worker_id} 遗弃，正在回收...[/yellow]")
                    self.store.release_task(self.plan_id, task.id)

    def _fill_worker_slots(self) -> None:
        """填补空闲的 Worker 槽位"""
        alive_count = sum(1 for f in self.futures.values() if not f.done())

        if alive_count < self.num_workers:
            for i in range(self.num_workers):
                worker_id = f"worker_{i + 1}"
                if worker_id not in self.futures or self.futures[worker_id].done():
                    self._start_worker(worker_id)

    def _shutdown_all_workers(self) -> None:
        """关闭所有 Worker"""
        if self.executor:
            self.executor.shutdown(wait=False)
        self.futures.clear()

    def cancel(self):
        """取消所有 Worker"""
        set_interrupt()  # 设置全局中断标志
        for future in self.futures.values():
            future.cancel()
        if self.executor:
            self.executor.shutdown(wait=False)

    def terminate(self):
        """终止执行（非阻塞，用于信号处理器）

        注意：这是非阻塞调用，仅设置标志。
        实际的清理在 executor.run() 的 finally 块中完成。
        """
        from agent.registry import agent_registry

        # 1. 设置全局中断标志（workers 会在下一次检查时退出）
        set_interrupt()

        # 2. 关闭线程池（不等待）
        if self.executor:
            self.executor.shutdown(wait=False)

        # 3. 清理 futures（不等待 workers）
        self.futures.clear()

        # 4. 释放被占用但未完成的任务
        plan = self.store.load_plan(self.plan_id)
        if plan:
            for task in plan.tasks:
                if task.claimed_by and task.status == "pending":
                    self.store.release_task(self.plan_id, task.id)

        # 5. 从全局注册表移除
        agent_registry.unregister(self.plan_id)

    def _summarize_results(self, plan: Plan) -> str:
        """汇总任务结果

        Args:
            plan: Plan 对象

        Returns:
            汇总结果
        """
        from utils.llm import get_llm_model

        llm = get_llm_model()

        # 构建任务结果摘要
        results_text = []
        for task in plan.tasks:
            results_text.append(f"### {task.id}: {task.description}")
            results_text.append(f"结果: {task.result}")
            results_text.append("")

        prompt = f"""你是一个结果汇总专家。请根据以下任务执行结果生成一份汇总报告。

整体目标: {plan.goal}

任务执行结果:
{chr(10).join(results_text)}

请生成一份简洁的汇总报告，包括：
1. 整体完成情况
2. 主要成果
3. 关键发现或结论

使用中文回答，控制在300字以内。"""

        response = llm.invoke(prompt)
        return response.content

    def _all_done(self) -> bool:
        """检查所有任务是否完成或失败"""
        plan = self.store.load_plan(self.plan_id)
        if not plan:
            return True
        return all(t.status in ("completed", "failed") for t in plan.tasks)
