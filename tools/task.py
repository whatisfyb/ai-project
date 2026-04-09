"""Task/Plan 管理工具

提供 Plan 和 Task 的查看和修改操作，支持 LLM 动态修改执行计划。

注意：Plan 必须通过 dispatch_agent(subagent_type="Plan") 创建。
      此工具只提供查看和修改功能。

功能：
- Plan: 查看、执行
- Task: 添加、更新、删除、查看
"""

import json
from typing import Any

from langchain_core.tools import tool

from agent.core.models import Task
from store.plan import PlanStore


# ============ Plan 操作 ============

@tool
def plan_get(plan_id: str) -> dict[str, Any]:
    """Get a specific plan with all its tasks.

    Args:
        plan_id: The plan ID to retrieve

    Returns:
        Plan details including all tasks
    """
    store = PlanStore()

    # 获取 plan 记录以获取 thread_id
    from store.plan import PlanRecord
    import sqlite3
    from datetime import datetime

    # 直接查询获取 thread_id
    db_path = store.db_path
    thread_id = None
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT thread_id FROM plans WHERE plan_id = ?",
            (plan_id,)
        )
        row = cursor.fetchone()
        if row:
            thread_id = row[0]

    plan = store.load_plan(plan_id)

    if not plan:
        return {
            "error": f"Plan not found: {plan_id}",
            "plan": None,
        }

    tasks_info = []
    for task in plan.tasks:
        tasks_info.append({
            "id": task.id,
            "description": task.description,
            "dependencies": task.dependencies,
            "status": task.status,
            "result": task.result,
            "claimed_by": task.claimed_by,
        })

    return {
        "plan": {
            "plan_id": plan_id,
            "thread_id": thread_id,
            "goal": plan.goal,
            "status": plan.status,
            "tasks": tasks_info,
            "summarized_result": plan.summarized_result,
        },
        "summary": {
            "total_tasks": len(plan.tasks),
            "completed": sum(1 for t in plan.tasks if t.status == "completed"),
            "pending": sum(1 for t in plan.tasks if t.status == "pending"),
            "failed": sum(1 for t in plan.tasks if t.status == "failed"),
        },
    }


@tool
def plan_execute(plan_id: str, timeout: int = 600) -> dict[str, Any]:
    """Execute a plan using workers.

    This runs the plan in the background with multiple workers.

    Args:
        plan_id: The plan ID to execute
        timeout: Execution timeout in seconds (default: 600 = 10 minutes)

    Returns:
        Execution result or status
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from agent.executor.executor import PlanExecutor

    store = PlanStore()
    plan = store.load_plan(plan_id)

    if not plan:
        return {
            "status": "error",
            "error": f"Plan not found: {plan_id}",
        }

    if not plan.tasks:
        return {
            "status": "error",
            "error": "Plan has no tasks to execute",
        }

    # 检查 plan 是否属于当前会话
    import sqlite3
    current_thread_id = None
    plan_thread_id = None

    try:
        from agent.main.agent import _current_thread_id
        current_thread_id = _current_thread_id.get()
    except:
        pass

    # 获取 plan 的 thread_id
    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.execute(
            "SELECT thread_id FROM plans WHERE plan_id = ?",
            (plan_id,)
        )
        row = cursor.fetchone()
        if row:
            plan_thread_id = row[0]

    # 检查会话归属
    if current_thread_id and plan_thread_id and current_thread_id != plan_thread_id:
        return {
            "status": "error",
            "error": f"Plan {plan_id} 属于会话 {plan_thread_id}，当前会话是 {current_thread_id}。请在正确的会话中执行。",
            "plan_thread_id": plan_thread_id,
            "current_thread_id": current_thread_id,
        }

    pending_tasks = store.get_pending_tasks(plan_id)
    if not pending_tasks:
        return {
            "status": "error",
            "error": "No executable tasks found (all tasks may have dependencies that cannot be satisfied)",
        }

    def _execute():
        executor = PlanExecutor(plan_id=plan_id, num_workers=2)
        return executor.run()

    try:
        with ThreadPoolExecutor(max_workers=1) as t_executor:
            future = t_executor.submit(_execute)
            result = future.result(timeout=timeout)

        return {
            "status": "completed",
            "plan_id": plan_id,
            "goal": result["goal"],
            "completed": result["completed"],
            "failed": result["failed"],
            "total": result["total"],
            "summarized_result": result.get("summarized_result"),
            "tasks": result.get("tasks", []),
        }

    except TimeoutError:
        return {
            "status": "timeout",
            "plan_id": plan_id,
            "error": f"执行超时（{timeout}秒），任务仍在后台运行",
            "goal": plan.goal,
            "interrupted": True,
        }
    except Exception as e:
        return {
            "status": "error",
            "plan_id": plan_id,
            "error": str(e),
        }


# ============ Task 操作 ============

@tool
def task_add(
    plan_id: str,
    task_id: str,
    description: str,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    """Add a new task to an existing plan.

    Args:
        plan_id: The plan ID to add the task to
        task_id: Unique identifier for the new task (e.g., "T5")
        description: Description of what the task does
        dependencies: List of task IDs this task depends on (optional)

    Returns:
        Result of adding the task
    """
    import sqlite3
    from datetime import datetime

    store = PlanStore()
    plan = store.load_plan(plan_id)

    if not plan:
        return {
            "success": False,
            "error": f"Plan not found: {plan_id}",
        }

    existing_ids = {t.id for t in plan.tasks}
    if task_id in existing_ids:
        return {
            "success": False,
            "error": f"Task ID already exists: {task_id}",
        }

    deps = dependencies or []
    for dep in deps:
        if dep not in existing_ids:
            return {
                "success": False,
                "error": f"Dependency task not found: {dep}",
            }

    with sqlite3.connect(store.db_path) as conn:
        now = datetime.now().isoformat()
        conn.execute(
            """
            INSERT INTO tasks (task_id, plan_id, description, dependencies, status, result, claimed_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, plan_id, description, json.dumps(deps), "pending", None, None)
        )
        conn.execute(
            "UPDATE plans SET updated_at = ? WHERE plan_id = ?",
            (now, plan_id)
        )
        conn.commit()

    return {
        "success": True,
        "plan_id": plan_id,
        "task": {
            "id": task_id,
            "description": description,
            "dependencies": deps,
            "status": "pending",
        },
    }


@tool
def task_update(
    plan_id: str,
    task_id: str,
    description: str | None = None,
    dependencies: list[str] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Update an existing task.

    Args:
        plan_id: The plan ID
        task_id: The task ID to update
        description: New description (optional)
        dependencies: New dependencies list (optional)
        status: New status ("pending", "completed", "failed") (optional)

    Returns:
        Result of updating the task
    """
    import sqlite3
    from datetime import datetime

    store = PlanStore()
    plan = store.load_plan(plan_id)

    if not plan:
        return {
            "success": False,
            "error": f"Plan not found: {plan_id}",
        }

    task = None
    for t in plan.tasks:
        if t.id == task_id:
            task = t
            break

    if not task:
        return {
            "success": False,
            "error": f"Task not found: {task_id}",
        }

    if description is not None:
        task.description = description

    if dependencies is not None:
        existing_ids = {t.id for t in plan.tasks} - {task_id}
        for dep in dependencies:
            if dep not in existing_ids:
                return {
                    "success": False,
                    "error": f"Dependency task not found: {dep}",
                }
        task.dependencies = dependencies

    if status is not None:
        if status not in ("pending", "completed", "failed"):
            return {
                "success": False,
                "error": f"Invalid status: {status}. Must be one of: pending, completed, failed",
            }
        task.status = status

    with sqlite3.connect(store.db_path) as conn:
        now = datetime.now().isoformat()
        conn.execute(
            """
            UPDATE tasks SET description = ?, dependencies = ?, status = ?
            WHERE task_id = ? AND plan_id = ?
            """,
            (task.description, json.dumps(task.dependencies), task.status, task_id, plan_id)
        )
        conn.execute(
            "UPDATE plans SET updated_at = ? WHERE plan_id = ?",
            (now, plan_id)
        )
        conn.commit()

    return {
        "success": True,
        "plan_id": plan_id,
        "task": {
            "id": task.id,
            "description": task.description,
            "dependencies": task.dependencies,
            "status": task.status,
        },
    }


@tool
def task_delete(plan_id: str, task_id: str) -> dict[str, Any]:
    """Delete a task from a plan.

    Args:
        plan_id: The plan ID
        task_id: The task ID to delete

    Returns:
        Result of deleting the task
    """
    import sqlite3
    from datetime import datetime

    store = PlanStore()
    plan = store.load_plan(plan_id)

    if not plan:
        return {
            "success": False,
            "error": f"Plan not found: {plan_id}",
        }

    task = None
    for t in plan.tasks:
        if t.id == task_id:
            task = t
            break

    if not task:
        return {
            "success": False,
            "error": f"Task not found: {task_id}",
        }

    depending_tasks = [t.id for t in plan.tasks if task_id in t.dependencies]
    if depending_tasks:
        return {
            "success": False,
            "error": f"Cannot delete task {task_id}. Other tasks depend on it: {depending_tasks}",
        }

    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE task_id = ? AND plan_id = ?", (task_id, plan_id))
        conn.execute(
            "UPDATE plans SET updated_at = ? WHERE plan_id = ?",
            (datetime.now().isoformat(), plan_id)
        )
        conn.commit()

    return {
        "success": True,
        "deleted_task_id": task_id,
        "plan_id": plan_id,
    }


@tool
def task_get(plan_id: str, task_id: str) -> dict[str, Any]:
    """Get details of a specific task.

    Args:
        plan_id: The plan ID
        task_id: The task ID

    Returns:
        Task details
    """
    store = PlanStore()
    plan = store.load_plan(plan_id)

    if not plan:
        return {
            "error": f"Plan not found: {plan_id}",
            "task": None,
        }

    for t in plan.tasks:
        if t.id == task_id:
            return {
                "task": {
                    "id": t.id,
                    "description": t.description,
                    "dependencies": t.dependencies,
                    "status": t.status,
                    "result": t.result,
                    "claimed_by": t.claimed_by,
                },
            }

    return {
        "error": f"Task not found: {task_id}",
        "task": None,
    }
