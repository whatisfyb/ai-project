from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"


class Task(BaseModel):
    id: str = Field(description="任务唯一标识")
    description: str = Field(description="任务描述，说明该任务需要完成的具体工作")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务当前状态，取值：pending(待执行) / completed(已完成) / failed(执行失败)")
    dependencies: list[str] = Field(
        default_factory=list,
        description="前置依赖任务的 ID 列表，为空表示无依赖，可立即执行",
    )
    result: Optional[str] = Field(default=None, description="任务执行结果，成功时由执行器写入")
    error: Optional[str] = Field(default=None, description="失败时的错误信息，由执行器写入")


class Plan(BaseModel):
    id: str = Field(description="计划唯一标识")
    goal: str = Field(description="计划目标，描述用户原始需求，为任务规划提供上下文")
    tasks: list[Task] = Field(default_factory=list, description="计划包含的任务列表")
    status: PlanStatus = Field(
        default=PlanStatus.RUNNING,
        description="计划当前状态，取值：running(进行中，含未完成或失败任务) / completed(所有任务成功完成)",
    )

    def get_ready_tasks(self) -> list[Task]:
        """返回所有可以执行的任务（pending 且依赖全部 completed）。"""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            task
            for task in self.tasks
            if task.status == TaskStatus.PENDING
            and all(dep_id in completed_ids for dep_id in task.dependencies)
        ]

    def check_complete(self) -> bool:
        """检查并更新 Plan 状态，所有任务完成则标记为 completed。"""
        if not self.tasks:
            self.status = PlanStatus.COMPLETED
            return True

        if all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED) for t in self.tasks
        ):
            if all(t.status == TaskStatus.COMPLETED for t in self.tasks):
                self.status = PlanStatus.COMPLETED
                return True
        return False

    def has_circular_dependency(self) -> bool:
        """检测是否存在循环依赖。"""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        task_map = {t.id: t for t in self.tasks}

        def dfs(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            if task_id in task_map:
                for dep_id in task_map[task_id].dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            rec_stack.discard(task_id)
            return False

        return any(
            dfs(task.id) for task in self.tasks if task.id not in visited
        )

    def is_valid_dag(self) -> bool:
        """验证任务图是合法的有向无环图，且所有任务均可达。

        检查项：
        1. 不存在循环依赖（DAG）
        2. 所有依赖目标都存在于 tasks 中（无悬空引用）
        3. 所有任务都能从根节点（无依赖的任务）触达

        Returns:
            验证通过返回 True
        """
        task_ids = {t.id for t in self.tasks}

        # 检查所有依赖是否都指向存在的任务
        for task in self.tasks:
            missing = set(task.dependencies) - task_ids
            if missing:
                raise ValueError(
                    f"Task '{task.id}' 依赖不存在的任务: {missing}"
                )

        # 检查循环依赖
        if self.has_circular_dependency():
            raise ValueError("任务图存在循环依赖")

        # 检查可达性：从根节点 BFS 遍历，验证是否能触达所有任务
        roots = [t.id for t in self.tasks if not t.dependencies]

        # 构建反向邻接表（依赖 → 后继）
        successors: dict[str, list[str]] = {tid: [] for tid in task_ids}
        for task in self.tasks:
            for dep_id in task.dependencies:
                successors[dep_id].append(task.id)

        visited: set[str] = set()
        queue = list(roots)
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for succ in successors.get(current, []):
                if succ not in visited:
                    queue.append(succ)

        unreachable = task_ids - visited
        if unreachable:
            raise ValueError(
                f"以下任务不可达（无法从根节点执行）: {unreachable}"
            )

        return True

    def execution_order(self) -> list[list[str]]:
        """返回按依赖层级分组的执行顺序。

        Returns:
            层级列表，每层是可以并行执行的任务 ID 列表。
            例如 [["task-1"], ["task-2", "task-3"], ["task-4"]]
        """
        task_map = {t.id: t for t in self.tasks}
        completed: set[str] = set()
        remaining = set(task_map.keys())
        levels: list[list[str]] = []

        while remaining:
            # 当前层：所有依赖已完成的就绪任务
            level = sorted([
                tid for tid in remaining
                if all(dep in completed for dep in task_map[tid].dependencies)
            ])
            if not level:
                # 理论上不会到这里（is_valid_dag 已拦截）
                raise ValueError("无法继续推进执行顺序，可能存在循环依赖")
            levels.append(level)
            completed.update(level)
            remaining -= set(level)

        return levels
