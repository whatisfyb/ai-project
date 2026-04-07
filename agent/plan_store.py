"""Plan 持久化存储"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.models import Plan, Task


class PlanRecord(BaseModel):
    """Plan 记录（含元数据）"""
    plan_id: str = Field(description="Plan 唯一标识")
    thread_id: str = Field(description="会话 ID")
    plan: Plan = Field(description="Plan 对象")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")


class PlanStore:
    """Plan 持久化存储

    使用 SQLite 存储 Plan 和 Task 数据，支持：
    - 保存/加载 Plan
    - 更新 Task 状态
    - 中断恢复
    """

    def __init__(self, db_path: str | Path = "data/plans.db"):
        """初始化存储

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summarized_result TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    dependencies TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    claimed_by TEXT,
                    PRIMARY KEY (task_id, plan_id),
                    FOREIGN KEY (plan_id) REFERENCES plans(plan_id)
                )
            """)

            conn.commit()

    def save_plan(self, plan: Plan, thread_id: str = "default") -> str:
        """保存 Plan

        Args:
            plan: Plan 对象
            thread_id: 会话 ID

        Returns:
            plan_id
        """
        # 生成 plan_id
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{thread_id}"
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # 保存 Plan
            conn.execute(
                """
                INSERT OR REPLACE INTO plans (plan_id, thread_id, goal, status, summarized_result, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (plan_id, thread_id, plan.goal, plan.status, plan.summarized_result, now, now)
            )

            # 保存 Tasks
            for task in plan.tasks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks (task_id, plan_id, description, dependencies, status, result, claimed_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (task.id, plan_id, task.description, json.dumps(task.dependencies), task.status, task.result, task.claimed_by)
                )

            conn.commit()

        return plan_id

    def load_plan(self, plan_id: str) -> Plan | None:
        """加载 Plan

        Args:
            plan_id: Plan ID

        Returns:
            Plan 对象，不存在返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            # 查询 Plan
            cursor = conn.execute(
                "SELECT goal, status, summarized_result FROM plans WHERE plan_id = ?",
                (plan_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            goal, status, summarized_result = row

            # 查询 Tasks
            cursor = conn.execute(
                "SELECT task_id, description, dependencies, status, result, claimed_by FROM tasks WHERE plan_id = ?",
                (plan_id,)
            )
            tasks = []
            for task_row in cursor.fetchall():
                task_id, description, dependencies, task_status, result, claimed_by = task_row
                tasks.append(Task(
                    id=task_id,
                    description=description,
                    dependencies=json.loads(dependencies),
                    status=task_status,
                    result=result,
                    claimed_by=claimed_by
                ))

            return Plan(goal=goal, tasks=tasks, status=status, summarized_result=summarized_result)

    def update_task_status(
        self,
        plan_id: str,
        task_id: str,
        status: Literal["pending", "completed", "failed"],
        result: str | None = None
    ) -> bool:
        """更新 Task 状态

        Args:
            plan_id: Plan ID
            task_id: Task ID
            status: 新状态
            result: 任务结果（可选）

        Returns:
            是否成功
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE tasks SET status = ?, result = ? WHERE task_id = ? AND plan_id = ?",
                (status, result, task_id, plan_id)
            )
            conn.execute(
                "UPDATE plans SET updated_at = ? WHERE plan_id = ?",
                (now, plan_id)
            )
            conn.commit()

        return True

    def update_plan_status(
        self,
        plan_id: str,
        status: Literal["pending", "completed", "failed"]
    ) -> bool:
        """更新 Plan 状态

        Args:
            plan_id: Plan ID
            status: 新状态

        Returns:
            是否成功
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE plan_id = ?",
                (status, now, plan_id)
            )
            conn.commit()

        return True

    def list_plans(
        self,
        status: Literal["pending", "completed", "failed"] | None = None,
        thread_id: str | None = None
    ) -> list[PlanRecord]:
        """列出 Plans

        Args:
            status: 按状态过滤
            thread_id: 按会话 ID 过滤

        Returns:
            Plan 列表
        """
        with sqlite3.connect(self.db_path) as conn:
            sql = """
                SELECT plan_id, thread_id, goal, status, created_at, updated_at
                FROM plans
            """
            params = []
            conditions = []

            if status:
                conditions.append("status = ?")
                params.append(status)
            if thread_id:
                conditions.append("thread_id = ?")
                params.append(thread_id)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            sql += " ORDER BY created_at DESC"

            cursor = conn.execute(sql, params)
            records = []

            for row in cursor.fetchall():
                plan_id, thread_id_val, goal, plan_status, created_at, updated_at = row
                plan = self.load_plan(plan_id)
                if plan:
                    records.append(PlanRecord(
                        plan_id=plan_id,
                        thread_id=thread_id_val,
                        plan=plan,
                        created_at=datetime.fromisoformat(created_at),
                        updated_at=datetime.fromisoformat(updated_at)
                    ))

            return records

    def get_pending_tasks(self, plan_id: str) -> list[Task]:
        """获取可执行的待处理 Tasks

        只返回状态为 pending 且依赖已满足的 Tasks

        Args:
            plan_id: Plan ID

        Returns:
            可执行的 Task 列表
        """
        plan = self.load_plan(plan_id)
        if not plan:
            return []

        # 构建 task_id -> status 映射
        task_status = {t.id: t.status for t in plan.tasks}

        pending_tasks = []
        for task in plan.tasks:
            if task.status != "pending":
                continue

            # 检查依赖是否都已完成
            deps_satisfied = all(
                task_status.get(dep) == "completed"
                for dep in task.dependencies
            )

            if deps_satisfied:
                pending_tasks.append(task)

        return pending_tasks

    def delete_plan(self, plan_id: str) -> bool:
        """删除 Plan

        Args:
            plan_id: Plan ID

        Returns:
            是否成功
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks WHERE plan_id = ?", (plan_id,))
            conn.execute("DELETE FROM plans WHERE plan_id = ?", (plan_id,))
            conn.commit()

        return True

    # ============ Worker 分布式方法 ============

    def claim_task(self, plan_id: str, worker_id: str) -> Task | None:
        """原子领取任务

        找到一个可执行的未领取任务，原子地标记为已领取

        Args:
            plan_id: Plan ID
            worker_id: Worker 标识

        Returns:
            领取到的 Task，无可用任务返回 None
        """
        # 获取可执行的任务（依赖已满足 + pending）
        pending_tasks = self.get_pending_tasks(plan_id)

        if not pending_tasks:
            return None

        with sqlite3.connect(self.db_path) as conn:
            # 尝试领取第一个未被领取的任务
            for task in pending_tasks:
                # 原子检查并更新
                cursor = conn.execute(
                    """
                    UPDATE tasks SET claimed_by = ?
                    WHERE task_id = ? AND plan_id = ? AND claimed_by IS NULL
                    """,
                    (worker_id, task.id, plan_id)
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    # 返回领取到的任务
                    task.claimed_by = worker_id
                    return task

            conn.commit()

        return None

    def release_task(self, plan_id: str, task_id: str) -> bool:
        """释放任务（用于失败重试）

        Args:
            plan_id: Plan ID
            task_id: Task ID

        Returns:
            是否成功
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE tasks SET claimed_by = NULL WHERE task_id = ? AND plan_id = ?",
                (task_id, plan_id)
            )
            conn.commit()

        return True

    def check_all_completed(self, plan_id: str) -> bool:
        """检查所有任务是否完成

        Args:
            plan_id: Plan ID

        Returns:
            是否全部完成
        """
        plan = self.load_plan(plan_id)
        if not plan:
            return False

        return all(t.status == "completed" for t in plan.tasks)

    def check_all_done(self, plan_id: str) -> bool:
        """检查所有任务是否结束（完成或失败）

        Args:
            plan_id: Plan ID

        Returns:
            是否全部结束
        """
        plan = self.load_plan(plan_id)
        if not plan:
            return False

        return all(t.status in ("completed", "failed") for t in plan.tasks)

    def save_summarized_result(self, plan_id: str, result: str) -> bool:
        """保存汇总结果

        Args:
            plan_id: Plan ID
            result: 汇总结果

        Returns:
            是否成功
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE plans SET summarized_result = ?, status = 'completed', updated_at = ? WHERE plan_id = ?",
                (result, now, plan_id)
            )
            conn.commit()

        return True

    def get_all_task_results(self, plan_id: str) -> list[dict]:
        """获取所有任务结果

        Args:
            plan_id: Plan ID

        Returns:
            任务结果列表
        """
        plan = self.load_plan(plan_id)
        if not plan:
            return []

        return [
            {"id": t.id, "description": t.description, "result": t.result, "status": t.status}
            for t in plan.tasks
        ]
