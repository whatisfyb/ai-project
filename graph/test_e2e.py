"""Graph 端到端测试 — 真实 LLM 调用 + LangSmith 监控

运行方式（项目根目录）：
    python graph/test_e2e.py

前置条件：
    - config.yaml 中 LangSmith 的 api_key 已配置
    - 模型 API 可用
"""

import asyncio
import sys
import time
from pathlib import Path

# 确保项目根在 sys.path 上
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.langsmith import configure_langsmith
from utils.llm import reset_llm_models
from utils.embedding import reset_embedding_model
from agents.planner.plan_structure import Plan, Task, TaskStatus

try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    console = type("FakeConsole", (), {"print": __builtins__["print"]})()
    HAS_RICH = False


# ============================================================
# 测试 1：简单串行 Plan（t1 → t2）
# ============================================================

def test_serial_plan():
    """串行任务：t1 写诗 → t2 赏析"""
    console.print("\n[bold yellow]=== 测试 1：串行计划 ===[/bold yellow]")

    plan = Plan(
        id="e2e-serial",
        goal="测试 graph 串行执行",
        tasks=[
            Task(id="t1", description="用中文写一首关于春天的短诗（不超过4行）", dependencies=[]),
            Task(id="t2", description="给上面这首诗写一段50字以内的赏析", dependencies=["t1"]),
        ],
    )

    console.print(f"  Plan: {plan.goal}")
    console.print(f"  Tasks: {len(plan.tasks)}, 执行层级: {plan.execution_order()}")

    return plan


# ============================================================
# 测试 2：并行 Plan（t1/t2 并发 → t3 依赖两者）
# ============================================================

def test_parallel_plan():
    """并行任务：t-med + t-edu 并发 → t-compare 依赖两者"""
    console.print("\n[bold yellow]=== 测试 2：并行计划 ===[/bold yellow]")

    plan = Plan(
        id="e2e-parallel",
        goal="测试 graph 并行执行",
        tasks=[
            Task(id="t-med", description="总结 AI 在医疗领域的1个最新应用（50字以内）", dependencies=[]),
            Task(id="t-edu", description="总结 AI 在教育领域的1个最新应用（50字以内）", dependencies=[]),
            Task(id="t-compare", description="一句话对比医疗和教育两个领域的 AI 应用差异（30字以内）", dependencies=["t-med", "t-edu"]),
        ],
    )

    console.print(f"  Plan: {plan.goal}")
    console.print(f"  Tasks: {len(plan.tasks)}, 执行层级: {plan.execution_order()}")

    return plan


# ============================================================
# 执行引擎
# ============================================================

def run_e2e(plan, test_desc):
    """构建 graph 并执行，打印结果摘要"""
    t_start = time.perf_counter()

    console.print("\n[bold green]Step 1: 配置 LangSmith...[/bold green]")
    configure_langsmith()
    reset_llm_models()
    reset_embedding_model()

    console.print("[bold green]Step 2: 构建 Graph...[/bold green]")
    from graph.graph_builder import build_graph, run_initial, request_interrupt

    graph = build_graph(plan)
    node_names = list(graph.nodes.keys())
    console.print(f"  节点: {node_names}")

    console.print(f"\n[bold green]Step 3: 执行 {test_desc}...[/bold green]")
    console.print("  (开始运行，请关注 LangSmith 项目查看 trace)\n")

    try:
        result = run_initial(graph, plan)
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        console.print(f"\n  [bold red]执行失败 ({elapsed:.1f}s): {e}[/bold red]")
        return False

    elapsed = time.perf_counter() - t_start

    # 打印结果
    console.print(f"\n[bold]耗时: {elapsed:.1f}s[/bold]")
    console.print(f"[bold]Plan 状态: {result['plan'].status}[/bold]")
    console.print(f"[bold]最终输出:[/bold]")
    console.print(result.get("final_output", "(无输出)"))
    console.print(f"\n[bold]各任务结果:[/bold]")
    for tid, text in result.get("results", {}).items():
        preview = text[:200] + "..." if len(text) > 200 else text
        console.print(f"  [{tid}] {preview}")

    # 检查最终输出非空
    output = result.get("final_output", "")
    if not output.strip():
        console.print("\n  [bold red]最终输出为空![/bold red]")
        return False

    # 检查所有任务状态
    for task in result["plan"].tasks:
        if task.status == TaskStatus.FAILED:
            console.print(f"\n  [bold yellow]警告: [{task.id}] 失败 - {task.error}[/bold yellow]")

    return True


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))  # Ctrl+C 优雅退出

    console.print("[bold cyan]=== Graph 端到端测试 ===[/bold cyan]")

    # 测试 1：串行
    plan1 = test_serial_plan()
    ok1 = run_e2e(plan1, "串行任务")

    # 测试 2：并行
    plan2 = test_parallel_plan()
    ok2 = run_e2e(plan2, "并行任务")

    console.print(f"\n[bold cyan]=== 测试总结 ===[/bold cyan]")
    console.print(f"  串行任务: {'通过' if ok1 else '失败'}")
    console.print(f"  并行任务: {'通过' if ok2 else '失败'}")
    console.print(f"\n  LangSmith 项目: research_agent")
    console.print(f"  请前往 https://smith.langchain.com 查看 trace 详情")

    sys.exit(0 if (ok1 and ok2) else 1)
