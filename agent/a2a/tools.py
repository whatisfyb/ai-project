"""A2A 工具 - 提供给 MainAgent 使用的 A2A 相关工具

包括：
- plan_dispatch: 非阻塞分发 Plan 给 Worker 执行
- job_status: 查询 A2A Task 状态
- job_list: 列出当前会话的 A2A Task
"""

from typing import Any
from datetime import datetime

from langchain_core.tools import tool

from agent.a2a.models import TaskStatus, TaskEvent, Message, Part
from agent.a2a.transport import InMemoryTransport, get_transport
from store.plan import PlanStore


# ============ 内部辅助函数 ============

def _get_current_thread_id() -> str | None:
    """获取当前会话 ID"""
    try:
        from agent.main.agent import _current_thread_id
        return _current_thread_id.get()
    except:
        return None


def _get_or_create_worker_pool(transport: InMemoryTransport, num_workers: int = 2):
    """获取或创建 Worker Pool

    Worker Pool 是全局单例，按需创建。
    """
    from agent.a2a.worker import A2AWorkerPool

    # 使用模块级变量存储 Worker Pool
    global _worker_pool

    if '_worker_pool' not in globals():
        _worker_pool = None

    if _worker_pool is None:
        _worker_pool = A2AWorkerPool(num_workers=num_workers, transport=transport)
        _worker_pool.start()

    return _worker_pool


def _ensure_workers_started(transport: InMemoryTransport):
    """确保 Workers 已启动"""
    _get_or_create_worker_pool(transport)


# ============ Plan 分发工具 ============

@tool
def plan_dispatch(plan_id: str, num_workers: int = 2) -> dict[str, Any]:
    """Dispatch a plan to A2A workers for non-blocking execution.

    This tool creates A2A Tasks for each PlanTask in the plan and dispatches
    them to available workers. It returns immediately without waiting for
    completion.

    Args:
        plan_id: The plan ID to dispatch
        num_workers: Number of workers to use (default: 2)

    Returns:
        Dispatch result with job IDs for tracking
    """
    transport = get_transport()
    store = PlanStore()

    # 加载 Plan
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

    # 检查会话归属
    current_thread_id = _get_current_thread_id()
    plan_thread_id = None

    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.execute(
            "SELECT thread_id FROM plans WHERE plan_id = ?",
            (plan_id,)
        )
        row = cursor.fetchone()
        if row:
            plan_thread_id = row[0]

    if current_thread_id and plan_thread_id and current_thread_id != plan_thread_id:
        return {
            "status": "error",
            "error": f"Plan {plan_id} 属于会话 {plan_thread_id}，当前会话是 {current_thread_id}。",
        }

    # 获取可执行的待处理任务
    pending_tasks = store.get_pending_tasks(plan_id)
    if not pending_tasks:
        return {
            "status": "error",
            "error": "No executable tasks (all may have unsatisfied dependencies)",
        }

    # 确保 Workers 已启动
    _ensure_workers_started(transport)

    # 获取可用 Worker
    workers = transport.find_agents_by_skill("execute_plantask")
    if not workers:
        return {
            "status": "error",
            "error": "No workers available. Workers may not have started.",
        }

    # 为每个 PlanTask 创建 A2A Task 并分发
    dispatched_jobs = []
    worker_idx = 0

    for plantask in pending_tasks:
        # 轮询选择 Worker
        worker_card = workers[worker_idx % len(workers)]
        worker_id = worker_card.id
        worker_idx += 1

        # 创建 A2A Task
        a2a_task = transport.create_task(
            sender_id="main",
            receiver_id=worker_id,
            initial_message=Message(
                role="user",
                parts=[Part.plantask(plan_id=plan_id, task_id=plantask.id)]
            ),
            plan_id=plan_id,
            plantask_id=plantask.id,
        )

        # 分发给 Worker（非阻塞）
        transport.send_message(a2a_task, Message(
            role="user",
            parts=[Part.plantask(plan_id=plan_id, task_id=plantask.id)]
        ))

        dispatched_jobs.append({
            "job_id": a2a_task.id,
            "task_id": plantask.id,
            "worker_id": worker_id,
        })

    return {
        "status": "dispatched",
        "plan_id": plan_id,
        "goal": plan.goal,
        "dispatched_count": len(dispatched_jobs),
        "jobs": dispatched_jobs,
        "message": f"已分发 {len(dispatched_jobs)} 个任务到 {len(workers)} 个 Worker。使用 job_status 查询进度。",
    }


# ============ Job 状态查询工具 ============

@tool
def job_status(job_id: str) -> dict[str, Any]:
    """Get the status of an A2A Task (job).

    Args:
        job_id: The A2A Task ID returned by plan_dispatch

    Returns:
        Job status and details
    """
    transport = get_transport()

    task = transport.get_task(job_id)
    if not task:
        return {
            "status": "error",
            "error": f"Job not found: {job_id}",
        }

    # 获取最后一条消息
    last_message = None
    if task.history:
        last_msg = task.history[-1]
        last_message = {
            "role": last_msg.role.value,
            "text": last_msg.get_text()[:500] if last_msg.get_text() else None,
        }

    return {
        "job_id": task.id,
        "status": task.status.value,
        "plan_id": task.plan_id,
        "plantask_id": task.plantask_id,
        "sender_id": task.sender_id,
        "receiver_id": task.receiver_id,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "is_terminal": task.is_terminal(),
        "last_message": last_message,
    }


@tool
def job_list(plan_id: str | None = None) -> dict[str, Any]:
    """List A2A Tasks (jobs) for the current session.

    Args:
        plan_id: Filter by plan ID (optional)

    Returns:
        List of jobs with their status
    """
    transport = get_transport()

    if plan_id:
        tasks = transport.get_tasks_by_plan(plan_id)
    else:
        # 获取所有任务
        tasks = []
        # TODO: 按会话过滤，需要 Transport 支持按 sender_id 查询

    jobs_info = []
    for task in tasks:
        jobs_info.append({
            "job_id": task.id,
            "status": task.status.value,
            "plan_id": task.plan_id,
            "plantask_id": task.plantask_id,
            "worker_id": task.receiver_id,
            "updated_at": task.updated_at.isoformat(),
        })

    return {
        "count": len(jobs_info),
        "jobs": jobs_info,
    }


@tool
def job_wait(job_id: str, timeout: int = 60) -> dict[str, Any]:
    """Wait for an A2A Task (job) to complete.

    This is a blocking call that waits until the job reaches a terminal state
    or timeout.

    Args:
        job_id: The A2A Task ID to wait for
        timeout: Maximum wait time in seconds (default: 60)

    Returns:
        Final job status
    """
    import time
    import threading

    transport = get_transport()

    task = transport.get_task(job_id)
    if not task:
        return {
            "status": "error",
            "error": f"Job not found: {job_id}",
        }

    if task.is_terminal():
        # 直接返回状态，不调用 job_status 工具
        return {
            "job_id": task.id,
            "status": task.status.value,
            "plan_id": task.plan_id,
            "plantask_id": task.plantask_id,
            "is_terminal": True,
        }

    # 使用事件等待
    done_event = threading.Event()
    final_task = {"task": task}

    def on_complete(t, event):
        if t.id == job_id and event in (TaskEvent.COMPLETED, TaskEvent.FAILED, TaskEvent.CANCELLED):
            final_task["task"] = t
            done_event.set()

    transport.subscribe_task(job_id, on_complete)

    # 等待完成或超时
    done_event.wait(timeout=timeout)

    transport.unsubscribe_task(job_id, on_complete)

    # 直接返回状态
    final = final_task["task"]
    return {
        "job_id": final.id,
        "status": final.status.value,
        "plan_id": final.plan_id,
        "plantask_id": final.plantask_id,
        "is_terminal": final.is_terminal(),
    }


# ============ Worker 管理工具 ============

@tool
def worker_list() -> dict[str, Any]:
    """List all registered A2A workers.

    Returns:
        List of workers with their status
    """
    transport = get_transport()

    agents = transport.list_agents()

    workers_info = []
    for card in agents:
        workers_info.append({
            "worker_id": card.id,
            "name": card.name,
            "description": card.description,
            "skills": [s.name for s in card.skills],
            "capabilities": {
                "text": card.capabilities.text,
                "files": card.capabilities.files,
                "streaming": card.capabilities.streaming,
            },
        })

    return {
        "count": len(workers_info),
        "workers": workers_info,
    }
