"""PlanExecutor - Plan 执行协调器

协调多个 TaskWorker 并行执行任务，提供进度显示。
支持用户中断处理。
"""

import signal
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from agent.plan_store import PlanStore
from agent.models import Task, Plan
from agent.worker import TaskWorker, set_interrupt, clear_interrupt, is_interrupted


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
    """Plan 执行器

    协调多个 Worker 并行执行任务。
    Main Agent 不参与执行，只负责协调和汇总。
    """

    def __init__(
        self,
        plan_id: str,
        num_workers: int = 2,
        store: PlanStore | None = None,
        interrupt_event: threading.Event | None = None,
    ):
        """初始化执行器

        Args:
            plan_id: Plan ID
            num_workers: Worker 数量
            store: PlanStore 实例
            interrupt_event: 外部中断事件，用于从其他线程触发中断
        """
        self.plan_id = plan_id
        self.num_workers = num_workers
        self.store = store or PlanStore()
        self.tracker = ProgressTracker(plan_id, self.store)
        self.interrupt_event = interrupt_event

        # 线程池
        self.executor: ThreadPoolExecutor | None = None
        self.futures: list[Future] = []
        self._original_sigint_handler = None

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
        """执行 Plan

        创建 Worker 线程池，等待完成，返回结果。
        如果全部成功，调用 Analysis Agent 汇总。
        支持用户中断（Ctrl+C）。

        Returns:
            执行结果
        """
        # 清除之前的中断标志
        clear_interrupt()
        self._setup_interrupt_handler()

        try:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self.futures = []

            # 创建 Workers
            workers = [
                TaskWorker(
                    worker_id=f"worker_{i + 1}",
                    plan_id=self.plan_id,
                    store=self.store,
                )
                for i in range(self.num_workers)
            ]

            # 提交任务
            for worker in workers:
                future = self.executor.submit(worker.run)
                self.futures.append(future)

            # 等待完成（带进度显示）
            console = Console()
            with Live(self.tracker.render(), refresh_per_second=2, console=console) as live:
                while not self._all_done() and not is_interrupted():
                    time.sleep(0.5)
                    live.update(self.tracker.render())

                # 最终更新
                live.update(self.tracker.render())

            # 收集结果
            worker_results = []
            for future in self.futures:
                try:
                    if future.done():
                        worker_results.append(future.result())
                except Exception as e:
                    pass

            # 关闭线程池
            self.executor.shutdown(wait=False)

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
            self._restore_interrupt_handler()

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

    def cancel(self):
        """取消所有 Worker"""
        set_interrupt()  # 设置全局中断标志
        for future in self.futures:
            future.cancel()
        if self.executor:
            self.executor.shutdown(wait=False)

    def _all_done(self) -> bool:
        """检查所有任务是否完成或失败"""
        plan = self.store.load_plan(self.plan_id)
        if not plan:
            return True
        return all(t.status in ("completed", "failed") for t in plan.tasks)
