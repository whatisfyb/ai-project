"""Task/Plan 管理工具 - 统一入口"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Literal

from langchain_core.tools import tool

from agent.core.models import PlanTask
from store.plan import PlanStore


@tool
def task(
    action: Literal["get_plan", "add", "update", "delete", "get"],
    plan_id: str = "",
    task_id: str = "",
    description: str = "",
    dependencies: list[str] = [],
    status: str = "",
) -> dict[str, Any]:
    """Task and Plan management operations.

    Unified tool for managing plans and tasks.

    Args:
        action: Operation to perform:
            - "get_plan": Get plan (uses: plan_id)
            - "add": Add task (uses: plan_id, task_id, description, dependencies)
            - "update": Update task (uses: plan_id, task_id, description, dependencies, status)
            - "delete": Delete task (uses: plan_id, task_id)
            - "get": Get task (uses: plan_id, task_id)
        plan_id: Plan identifier
        task_id: Task identifier
        description: Task description
        dependencies: List of task IDs this task depends on
        status: Task status (pending, completed, failed)

    Returns:
        Operation result.
    """
    try:
        if action == "get_plan":
            return _get_plan(plan_id)
        elif action == "add":
            return _task_add(plan_id, task_id, description, dependencies if dependencies else None)
        elif action == "update":
            return _task_update(
                plan_id,
                task_id,
                description if description else None,
                dependencies if dependencies else None,
                status if status else None,
            )
        elif action == "delete":
            return _task_delete(plan_id, task_id)
        elif action == "get":
            return _task_get(plan_id, task_id)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_plan(plan_id: str) -> dict[str, Any]:
    """获取计划"""
    if not plan_id:
        return {"error": "plan_id is required"}

    store = PlanStore()

    # 获取 thread_id
    thread_id = None
    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.execute("SELECT thread_id FROM plans WHERE plan_id = ?", (plan_id,))
        row = cursor.fetchone()
        if row:
            thread_id = row[0]

    plan = store.load_plan(plan_id)
    if not plan:
        return {"error": f"Plan not found: {plan_id}", "plan": None}

    tasks_info = [{
        "id": t.id,
        "description": t.description,
        "dependencies": t.dependencies,
        "status": t.status,
        "result": t.result,
    } for t in plan.tasks]

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


def _task_add(plan_id: str, task_id: str, description: str, dependencies: list[str] = None) -> dict[str, Any]:
    """添加任务"""
    if not all([plan_id, task_id, description]):
        return {"success": False, "error": "plan_id, task_id, and description are required"}

    store = PlanStore()
    plan = store.load_plan(plan_id)
    if not plan:
        return {"success": False, "error": f"Plan not found: {plan_id}"}

    existing_ids = {t.id for t in plan.tasks}
    if task_id in existing_ids:
        return {"success": False, "error": f"Task ID already exists: {task_id}"}

    deps = dependencies or []
    for dep in deps:
        if dep not in existing_ids:
            return {"success": False, "error": f"Dependency task not found: {dep}"}

    with sqlite3.connect(store.db_path) as conn:
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO tasks (task_id, plan_id, description, dependencies, status, result, claimed_by) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, plan_id, description, json.dumps(deps), "pending", None, None)
        )
        conn.execute("UPDATE plans SET updated_at = ? WHERE plan_id = ?", (now, plan_id))
        conn.commit()

    return {"success": True, "task": {"id": task_id, "description": description, "dependencies": deps, "status": "pending"}}


def _task_update(plan_id: str, task_id: str, description: str = None, dependencies: list[str] = None, status: str = None) -> dict[str, Any]:
    """更新任务"""
    if not all([plan_id, task_id]):
        return {"success": False, "error": "plan_id and task_id are required"}

    store = PlanStore()
    plan = store.load_plan(plan_id)
    if not plan:
        return {"success": False, "error": f"Plan not found: {plan_id}"}

    task = next((t for t in plan.tasks if t.id == task_id), None)
    if not task:
        return {"success": False, "error": f"Task not found: {task_id}"}

    if description is not None:
        task.description = description
    if dependencies is not None:
        existing_ids = {t.id for t in plan.tasks} - {task_id}
        for dep in dependencies:
            if dep not in existing_ids:
                return {"success": False, "error": f"Dependency task not found: {dep}"}
        task.dependencies = dependencies
    if status is not None:
        if status not in ("pending", "completed", "failed"):
            return {"success": False, "error": f"Invalid status: {status}"}
        task.status = status

    with sqlite3.connect(store.db_path) as conn:
        now = datetime.now().isoformat()
        conn.execute(
            "UPDATE tasks SET description = ?, dependencies = ?, status = ? WHERE task_id = ? AND plan_id = ?",
            (task.description, json.dumps(task.dependencies), task.status, task_id, plan_id)
        )
        conn.execute("UPDATE plans SET updated_at = ? WHERE plan_id = ?", (now, plan_id))
        conn.commit()

    return {"success": True, "task": {"id": task.id, "description": task.description, "dependencies": task.dependencies, "status": task.status}}


def _task_delete(plan_id: str, task_id: str) -> dict[str, Any]:
    """删除任务"""
    if not all([plan_id, task_id]):
        return {"success": False, "error": "plan_id and task_id are required"}

    store = PlanStore()
    plan = store.load_plan(plan_id)
    if not plan:
        return {"success": False, "error": f"Plan not found: {plan_id}"}

    task = next((t for t in plan.tasks if t.id == task_id), None)
    if not task:
        return {"success": False, "error": f"Task not found: {task_id}"}

    depending_tasks = [t.id for t in plan.tasks if task_id in t.dependencies]
    if depending_tasks:
        return {"success": False, "error": f"Cannot delete. Tasks depend on it: {depending_tasks}"}

    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE task_id = ? AND plan_id = ?", (task_id, plan_id))
        conn.execute("UPDATE plans SET updated_at = ? WHERE plan_id = ?", (datetime.now().isoformat(), plan_id))
        conn.commit()

    return {"success": True, "deleted_task_id": task_id}


def _task_get(plan_id: str, task_id: str) -> dict[str, Any]:
    """获取任务"""
    if not all([plan_id, task_id]):
        return {"error": "plan_id and task_id are required"}

    store = PlanStore()
    plan = store.load_plan(plan_id)
    if not plan:
        return {"error": f"Plan not found: {plan_id}", "task": None}

    task = next((t for t in plan.tasks if t.id == task_id), None)
    if not task:
        return {"error": f"Task not found: {task_id}", "task": None}

    return {"task": {"id": t.id, "description": t.description, "dependencies": t.dependencies, "status": t.status, "result": t.result}}
