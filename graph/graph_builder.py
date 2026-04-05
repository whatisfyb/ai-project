"""根据 Plan 构建 LangGraph 异步执行图（基于 asyncio）

Graph 流程：

    START → router (检查中断标志 + 选就绪任务)
         → parallel_executor (asyncio.as_completed 并发执行本轮所有就绪 task)
         → check_interrupt (取消残留协程 → interrupt 挂起)
         → router (循环：下一批 or synthesize)
         → synthesize (汇总全部 result)
         → END

中断机制：
    外部调用 request_interrupt() → 设置 _cancel_flag_registry + cancel 正在运行的 Task →
    executor 捕捉 CancelledError 写入 CANCELLED → cancel flag 被 graph 内 _check_interrupt 检测到 →
    interrupt() 挂起 → resume 时清除标志，继续调度
"""

import asyncio
import threading
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt, Command

from agents.executor.executor import async_execute_task
from agents.planner.plan_structure import Plan, Task, TaskStatus
from agents.synthesizer.synthesizer import async_synthesize


# ---------------------------------------------------------------------------
# Task registry + cancel flags — cross-thread communication
#
# {config_key: [Task, ...]}           asyncio.Task 列表，每轮 executor 期间有效
# {config_key: threading.Event}       中断标志，set() 后 executor + check_interrupt 可见
# ---------------------------------------------------------------------------

_running_tasks_registry: dict[str, list[asyncio.Task]] = {}
_cancel_events: dict[str, threading.Event] = {}


def _config_key(config: Optional[dict]) -> str:
    """从运行配置中提取唯一标识"""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "__default__")
    return thread_id


def _cancel_running_tasks(config: Optional[dict] = None) -> None:
    """取消指定配置下所有正在运行的任务 + 设置取消标志（阻止后续轮次）。

    由外部（如用户前端）调用。

    Args:
        config: 运行配置，与 request_interrupt 使用的 config 一致
    """
    key = _config_key(config)
    evt = _cancel_events.get(key)
    if evt:
        evt.set()

    for task in _running_tasks_registry.get(key, []):
        if not task.done():
            task.cancel()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class GraphState(dict):
    """Graph 状态"""
    plan: Plan
    results: dict              # {task_id: result_text}
    final_output: str          # Synthesizer 产出
    _cancel_key: str           # 用于在模块级 dict 中查找对应的 threading.Event


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _router_fn(state: dict) -> str:
    """路由函数 — 判断是否还有就绪任务（同步，因为 graph.update_state 可能触发它）

    Returns:
        "executor": 有就绪 task
        "synthesize": 无就绪 task
    """
    ready = state["plan"].get_ready_tasks()
    return "executor" if ready else "synthesize"


async def _parallel_executor(state: dict) -> dict:
    """并行执行节点 — asyncio.as_completed 并发执行所有就绪任务"""
    plan: Plan = state["plan"]
    ready_tasks = plan.get_ready_tasks()
    cancel_key: str = state.get("_cancel_key", "__default__")
    cancel_event = _cancel_events.get(cancel_key)

    if not ready_tasks:
        return {"plan": plan, "results": state.get("results", {})}

    results_dict = dict(state.get("results", {}))

    # 缓存依赖任务映射
    task_map = {t.id: t for t in plan.tasks}

    # 构建前置结果上下文
    def _build_context(task: Task) -> str:
        lines = []
        for dep_id in task.dependencies:
            if dep_id in results_dict:
                dep_task = task_map.get(dep_id)
                dep_desc = dep_task.description if dep_task else dep_id
                lines.append(f"### [{dep_id}] {dep_desc}\n{results_dict[dep_id]}")
            else:
                lines.append(f"### [{dep_id}] 前置任务尚未完成")
        return "\n\n".join(lines)

    # Each asyncio.Task 注册时用同一个 cancel_key
    tasks: list[asyncio.Task] = []
    for t in ready_tasks:
        ctx = _build_context(t) if t.dependencies else ""
        coro = async_execute_task(t, context=ctx)
        tasks.append(asyncio.ensure_future(coro))

    _running_tasks_registry[cancel_key] = tasks

    # as_completed 保证完成一个处理一个
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
        except asyncio.CancelledError:
            pass  # 被外部取消
        except Exception:
            pass

        # 每完成一个任务检查中断
        if cancel_event and cancel_event.is_set():
            for t in tasks:
                if not t.done():
                    t.cancel()
            break

        if isinstance(result, Task) and result.id in task_map:
            task_map[result.id].status = result.status
            task_map[result.id].result = result.result
            task_map[result.id].error = result.error
            results_dict[result.id] = result.result or f"[无结果] {result.error or '任务未执行'}"

    # 清理 registry
    _running_tasks_registry.pop(key, None)

    plan.check_complete()

    return {
        "plan": plan,
        "results": results_dict,
        "final_output": "",
    }


async def _check_interrupt(state: dict) -> dict:
    """检查取消事件 — 被中断则等待协程退完后挂起"""
    cancel_key: str = state.get("_cancel_key", "__default__")
    cancel_event = _cancel_events.get(cancel_key)

    if cancel_event and cancel_event.is_set():
        # 取消残留任务
        for t in _running_tasks_registry.get(cancel_key, []):
            if not t.done():
                t.cancel()
        _running_tasks_registry.pop(cancel_key, None)

        await asyncio.sleep(0.3)
        interrupt("用户请求中断，graph 暂停。恢复后继续执行。")

    return state


async def _synthesize_node(state: dict) -> dict:
    """汇总节点 — 委托给 Synthesizer Agent"""
    plan: Plan = state["plan"]
    results: dict = state.get("results", {})

    if not results:
        return {
            "final_output": f"计划 '{plan.id}' 执行完毕，无输出结果。\n目标: {plan.goal}",
        }

    final = await async_synthesize(plan, results)
    return {"final_output": final}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    plan: Plan,
    checkpointer=None,
) -> CompiledStateGraph:
    """根据 Plan 构建 LangGraph 异步执行图

    Args:
        plan: 已验证的 Plan 实例（调用过 is_valid_dag()）
        checkpointer: 可选的 Checkpointer，传入 SqliteCheckpointer 启用持久化

    Returns:
        编译好的 CompiledStateGraph
    """
    plan.is_valid_dag()

    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("router", lambda state: state)
    workflow.add_node("parallel_executor", _parallel_executor)
    workflow.add_node("check_interrupt", _check_interrupt)
    workflow.add_node("synthesize", _synthesize_node)

    # 连接
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        _router_fn,
        {
            "executor": "parallel_executor",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("parallel_executor", "check_interrupt")
    workflow.add_conditional_edges(
        "check_interrupt",
        _router_fn,
        {
            "executor": "parallel_executor",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("synthesize", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


async def _run_async(graph, initial_value, config: Optional[dict] = None) -> dict:
    """内部异步运行器 — astream + stream_mode=values"""
    invoke_config = config or {}
    last_event = None
    async for event in graph.astream(initial_value, invoke_config, stream_mode="values"):
        last_event = event
    if isinstance(initial_value, Command):
        raise ValueError("Command 必须通过 _run_async_command 调用")
    return last_event


async def _run_async_command(graph, command, config: Optional[dict] = None) -> dict:
    """内部异步运行器 — astream + Command"""
    invoke_config = config or {}
    last_event = None
    async for event in graph.astream(command, invoke_config, stream_mode="values"):
        last_event = event
    return last_event


def run_initial(graph: CompiledStateGraph, plan: Plan, config: Optional[dict] = None) -> dict:
    """首次启动 graph 执行 Plan

    Args:
        graph: 编译好的 graph
        plan: Plan 实例
        config: 可选运行配置 (如 thread_id)

    Returns:
        最后的 state dict（含 final_output）
    """
    key = _config_key(config or {})

    # 创建可序列化的 cancel_key + 模块级 threading.Event
    if key not in _cancel_events:
        _cancel_events[key] = threading.Event()
    # 确保是干净的
    _cancel_events[key].clear()

    initial_state = {
        "plan": plan,
        "results": {},
        "final_output": "",
        "_cancel_key": key,
        "_config": config or {},
    }

    return asyncio.run(_run_async(graph, initial_state, config or {}))


def resume_graph(graph: CompiledStateGraph, config: Optional[dict] = None, resume_value=True) -> dict:
    """从断点恢复 graph

    恢复时自动将被 CANCELLED 状态的任务重置为 PENDING，以便重新执行。

    Args:
        graph: 编译好的 graph
        config: 运行配置 (如 thread_id)
        resume_value: interrupt 恢复值，默认 True

    Returns:
        最后的 state dict
    """
    invoke_config = config or {}
    key = _config_key(invoke_config)

    # 清除旧的取消标志
    if key not in _cancel_events:
        _cancel_events[key] = threading.Event()
    _cancel_events[key].clear()

    # 重置被取消的任务
    current_state = graph.get_state(invoke_config)
    if current_state and current_state.values:
        graph.update_state(invoke_config, {"_cancel_key": key, "_config": invoke_config})
        plan: Plan = current_state.values.get("plan")
        if plan:
            reset_ids = []
            for task in plan.tasks:
                if task.status == TaskStatus.CANCELLED:
                    task.status = TaskStatus.PENDING
                    task.result = None
                    task.error = None
                    reset_ids.append(task.id)
            if reset_ids:
                results_dict = current_state.values.get("results", {})
                for rid in reset_ids:
                    results_dict.pop(rid, None)

    command = Command(resume=resume_value)
    return asyncio.run(_run_async_command(graph, command, invoke_config))


def request_interrupt(graph: CompiledStateGraph, config: Optional[dict] = None) -> None:
    """外部调用：请求中断 graph 执行

    1. set _cancel_event（executor 检测到后取消当前任务，check_interrupt 检测到后挂起）
    2. 向所有正在 await 的 Task 发 CancelledError（中断当前 LLM/工具调用）

    Args:
        graph: 编译好的 graph
        config: 运行配置 (如 thread_id)
    """
    # 从 state 中取出 cancel_key 并 set 对应的事件
    current_state = graph.get_state(config or {})
    if current_state and current_state.values:
        cancel_key = current_state.values.get("_cancel_key", "__default__")
        evt = _cancel_events.get(cancel_key)
        if evt:
            evt.set()

    _cancel_running_tasks(config)
