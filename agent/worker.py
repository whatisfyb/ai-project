"""TaskWorker - 任务执行器

专门的任务执行器，不是 MainAgent 副本，不会 fork。
支持中断检查。
"""

import threading
import time
from typing import Callable

from agent.plan_store import PlanStore
from agent.models import Task

# 全局中断标志
_interrupt_event = threading.Event()


def set_interrupt():
    """设置中断信号"""
    _interrupt_event.set()


def clear_interrupt():
    """清除中断信号"""
    _interrupt_event.clear()


def is_interrupted() -> bool:
    """检查是否被中断"""
    return _interrupt_event.is_set()


class TaskWorker:
    """任务执行 Worker

    从数据库领取任务，执行，写回结果。
    完成后死亡，不进行总结。

    Worker 不是 MainAgent 副本，不会 fork 自己。
    """

    def __init__(
        self,
        worker_id: str,
        plan_id: str,
        memory: list,
        execute_fn: Callable[[Task, list], str],
        store: PlanStore | None = None,
        timeout: int = 300,
    ):
        """初始化 Worker

        Args:
            worker_id: Worker 标识
            plan_id: Plan ID
            memory: 初始化记忆（复制后独立维护）
            execute_fn: 任务执行函数 (task, memory) -> result
            store: PlanStore 实例
            timeout: 单任务超时时间（秒）
        """
        self.worker_id = worker_id
        self.plan_id = plan_id
        self.memory = memory.copy()  # 复制，独立维护
        self.execute_fn = execute_fn
        self.store = store or PlanStore()
        self.timeout = timeout

    def run(self) -> dict:
        """执行任务循环

        Returns:
            执行结果统计
        """
        results = {"completed": 0, "failed": 0, "tasks": [], "interrupted": False}

        while True:
            # 检查中断
            if is_interrupted():
                results["interrupted"] = True
                break

            # 1. 原子领取任务
            task = self.store.claim_task(self.plan_id, self.worker_id)

            if not task:
                # 没有可执行任务，死亡
                break

            # 2. 执行任务（带超时）
            try:
                result = self._execute_with_timeout(task)
                self.store.update_task_status(
                    self.plan_id, task.id, "completed", result
                )
                results["completed"] += 1
                results["tasks"].append({"id": task.id, "status": "completed"})
            except Exception as e:
                # 检查是否是中断导致的
                if is_interrupted():
                    self.store.release_task(self.plan_id, task.id)
                    results["interrupted"] = True
                    break

                error_msg = f"执行失败: {str(e)}"
                self.store.update_task_status(
                    self.plan_id, task.id, "failed", error_msg
                )
                self.store.release_task(self.plan_id, task.id)
                results["failed"] += 1
                results["tasks"].append({
                    "id": task.id,
                    "status": "failed",
                    "error": str(e)
                })

        return results

    def _execute_with_timeout(self, task: Task) -> str:
        """带超时执行任务

        Args:
            task: 要执行的任务

        Returns:
            执行结果

        Raises:
            TimeoutError: 执行超时
            Exception: 执行失败
        """
        result_container = {"result": None, "error": None}

        def target():
            try:
                # 将任务加入自己的记忆
                self.memory.append({
                    "role": "user",
                    "content": f"请执行以下任务：{task.description}"
                })

                # 执行
                result = self.execute_fn(task, self.memory)

                # 将结果加入自己的记忆
                self.memory.append({
                    "role": "assistant",
                    "content": result
                })

                result_container["result"] = result
            except Exception as e:
                result_container["error"] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            # 超时
            raise TimeoutError(f"任务 {task.id} 执行超时 ({self.timeout}s)")

        if result_container["error"]:
            raise result_container["error"]

        return result_container["result"]
