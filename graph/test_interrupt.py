"""Interrupt / Resume 测试 — 使用 mock 精确控制中断时机

运行方式（项目根目录）：
    python graph/test_interrupt.py
"""

import asyncio
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import graph.graph_builder as gb
from langgraph.checkpoint.memory import InMemorySaver

from agents.planner.plan_structure import Plan, Task, TaskStatus
from graph.graph_builder import build_graph, run_initial, resume_graph, request_interrupt

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = type("FakeConsole", (), {"print": __builtins__["print"]})()


# ============================================================
# Mock executor — 通过 durations dict 控制每个任务的延迟
# ============================================================

def make_executor(durations: dict[str, float]):
    """创建一个模拟 exec 函数，捕获外部 durations 字典"""
    async def mock_executor(task, max_iterations=15, context=""):
        dur = durations.get(task.id, 1.0)
        try:
            await asyncio.sleep(dur)
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.error = "任务已取消"
            raise

        task.status = TaskStatus.COMPLETED
        task.result = f"任务 {task.id} 执行完成"
        return task
    return mock_executor


def _run_graph_in_thread(graph, plan, config, results, mock_fn, patch_obj):
    """使用 patch 在后台线程运行 graph"""
    def _run():
        with patch_obj:
            try:
                result = run_initial(graph, plan, config)
                results.append(("success", result))
            except Exception as e:
                results.append(("error", e))

    t = threading.Thread(target=_run)
    t.start()
    return t


# ============================================================
# 测试 1：串行中断 + 恢复
# ============================================================

def test_interrupt_and_resume():
    """4 串行任务：t1(2s) → t2(3s) → ...，在 4s 时中断"""
    durations = {"t1": 2.0, "t2": 3.0, "t3": 2.0, "t4": 2.0}
    mock_fn = make_executor(durations)

    plan = Plan(
        id="interrupt-serial",
        goal="测试中断恢复",
        tasks=[
            Task(id="t1", description="任务1", dependencies=[]),
            Task(id="t2", description="任务2", dependencies=["t1"]),
            Task(id="t3", description="任务3", dependencies=["t2"]),
            Task(id="t4", description="任务4", dependencies=["t3"]),
        ],
    )

    saver = InMemorySaver()
    graph = build_graph(plan, checkpointer=saver)
    config = {"configurable": {"thread_id": "serial-1"}}

    results = []
    patcher = patch.object(gb, "async_execute_task", mock_fn)
    patcher.start()
    worker = _run_graph_in_thread(graph, plan, config, results, mock_fn, patcher)

    # t1(2s) + 部分 t2(3s) → 在 3.5s 时中断
    time.sleep(3.5)
    console.print("  请求中断...")
    request_interrupt(graph, config)

    worker.join(timeout=20)
    assert worker.is_alive() is False, "worker 线程超时"

    state = graph.get_state(config)
    mid_plan: Plan = state.values.get("plan")

    cancelled = [t.id for t in mid_plan.tasks if t.status == TaskStatus.CANCELLED]
    completed = [t.id for t in mid_plan.tasks if t.status == TaskStatus.COMPLETED]
    console.print(f"  中断后 — 完成: {completed}, 取消: {cancelled}")

    assert "t1" in completed, f"t1 应已完成，实际: {completed}"
    assert len(cancelled) >= 1, f"应有被取消任务，实际: {cancelled}"

    patcher.stop()

    # Resume
    console.print("  恢复执行...")
    result = resume_graph(graph, config)
    final_plan: Plan = result["plan"]

    for t in final_plan.tasks:
        assert t.status == TaskStatus.COMPLETED, (
            f"任务 {t.id} 预期 COMPLETED，实际 {t.status.value}"
        )

    assert set(result.get("results", {}).keys()) == {"t1", "t2", "t3", "t4"}
    console.print(f"  最终状态: {[t.id + '=' + t.status.value for t in final_plan.tasks]}")
    console.print("  [bold green]通过[/bold green]")
    return True


# ============================================================
# 测试 2：极短任务 — 中断在任务之间
# ============================================================

def test_interrupt_between_tasks():
    """4 个 0.2s 任务，0.5s 时中断"""
    durations = {"t1": 0.2, "t2": 0.2, "t3": 0.2, "t4": 0.2}
    mock_fn = make_executor(durations)

    plan = Plan(
        id="interrupt-between",
        goal="测试中断在任务之间",
        tasks=[
            Task(id="t1", description="短任务1", dependencies=[]),
            Task(id="t2", description="短任务2", dependencies=["t1"]),
            Task(id="t3", description="短任务3", dependencies=["t2"]),
            Task(id="t4", description="短任务4", dependencies=["t3"]),
        ],
    )

    saver = InMemorySaver()
    graph = build_graph(plan, checkpointer=saver)
    config = {"configurable": {"thread_id": "between-1"}}

    results = []
    patcher = patch.object(gb, "async_execute_task", mock_fn)
    patcher.start()
    worker = _run_graph_in_thread(graph, plan, config, results, mock_fn, patcher)

    time.sleep(0.5)
    request_interrupt(graph, config)
    worker.join(timeout=10)

    state = graph.get_state(config)
    mid_plan: Plan = state.values.get("plan")
    completed = [t.id for t in mid_plan.tasks if t.status == TaskStatus.COMPLETED]
    cancelled = [t.id for t in mid_plan.tasks if t.status == TaskStatus.CANCELLED]
    pending = sum(1 for t in mid_plan.tasks if t.status == TaskStatus.PENDING)
    console.print(f"  中断后 — 完成: {completed}, 取消: {cancelled}, 待执行: {pending}")

    patcher.stop()

    # Resume
    result = resume_graph(graph, config)
    for t in result["plan"].tasks:
        assert t.status == TaskStatus.COMPLETED, f"任务 {t.id} 恢复后未完成"

    assert len(result.get("results", {})) == 4
    console.print("  [bold green]通过[/bold green]")
    return True


# ============================================================
# 测试 3：并行中断 + 恢复
# ============================================================

def test_interrupt_parallel():
    """2 并行任务 (各 3s) + 依赖，在并行阶段中断"""
    durations = {"t-a": 3.0, "t-b": 3.0, "t-c": 2.0, "t-d": 2.0}
    mock_fn = make_executor(durations)

    plan = Plan(
        id="interrupt-parallel",
        goal="测试并行任务中断",
        tasks=[
            Task(id="t-a", description="并行任务A", dependencies=[]),
            Task(id="t-b", description="并行任务B", dependencies=[]),
            Task(id="t-c", description="依赖两者", dependencies=["t-a", "t-b"]),
            Task(id="t-d", description="最后", dependencies=["t-c"]),
        ],
    )

    saver = InMemorySaver()
    graph = build_graph(plan, checkpointer=saver)
    config = {"configurable": {"thread_id": "parallel-1"}}

    results = []
    patcher = patch.object(gb, "async_execute_task", mock_fn)
    patcher.start()
    worker = _run_graph_in_thread(graph, plan, config, results, mock_fn, patcher)

    time.sleep(2)
    request_interrupt(graph, config)
    worker.join(timeout=15)

    state = graph.get_state(config)
    mid_plan: Plan = state.values.get("plan")
    completed = [t.id for t in mid_plan.tasks if t.status == TaskStatus.COMPLETED]
    cancelled = [t.id for t in mid_plan.tasks if t.status == TaskStatus.CANCELLED]
    console.print(f"  并行中断后 — 完成: {completed}, 取消: {cancelled}")

    patcher.stop()

    # Resume
    result = resume_graph(graph, config)
    for t in result["plan"].tasks:
        assert t.status == TaskStatus.COMPLETED

    assert len(result.get("results", {})) == 4
    console.print("  [bold green]通过[/bold green]")
    return True


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    _pass = 0
    _fail = 0

    def _run(fn, name):
        global _pass, _fail
        try:
            fn()
            _pass += 1
        except Exception as e:
            _fail += 1
            console.print(f"  [bold red]失败: {name} — {e}[/bold red]")
            import traceback
            traceback.print_exc()

    console.print("[bold cyan]=== Interrupt / Resume 测试 ===[/bold cyan]")

    _run(test_interrupt_and_resume, "串行中断后恢复")
    _run(test_interrupt_between_tasks, "极短任务中断")
    _run(test_interrupt_parallel, "并行中断后恢复")

    console.print(f"\n[bold cyan]=== 总结 ===[/bold cyan]")
    console.print(f"  {_pass} 通过, {_fail} 失败")

    sys.exit(0 if _fail == 0 else 1)
