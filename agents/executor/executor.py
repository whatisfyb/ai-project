"""执行器模块 — 接收 Task 并自主调用工具完成"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path (needed for __main__ execution)
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agents.planner.plan_structure import Task, TaskStatus
from utils.llm import get_llm_model

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "executor_prompt.txt"
_EXECUTOR_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

from tools import (
    tavily_search,
    tavily_extract,
    firecrawl_scrape,
    firecrawl_crawl,
    firecrawl_map,
    arxiv_search,
    arxiv_get_by_id,
    arxiv_download_pdf,
)

_ALL_TOOLS = [
    tavily_search,
    tavily_extract,
    firecrawl_scrape,
    firecrawl_crawl,
    firecrawl_map,
    arxiv_search,
    arxiv_get_by_id,
    arxiv_download_pdf,
]

# tool name -> function
_TOOL_MAP = {t.name: t for t in _ALL_TOOLS}

_MAX_TOOL_CALLS = 15


def _build_prompt(task: Task, context: str = "") -> str:
    """构建 prompt 输入"""
    base = _EXECUTOR_PROMPT.format(
        task_instruction=f"任务 ID: {task.id}\n任务描述: {task.description}"
    )
    if context:
        return base + f"\n\n## 前置任务执行结果\n\n{context}"
    return base


def execute_task(
    task: Task,
    llm: Optional[ChatOpenAI] = None,
    max_iterations: int = _MAX_TOOL_CALLS,
    context: str = "",
) -> Task:
    """执行单个任务

    Args:
        task: 待执行的 Task 实例
        llm: 可选的 LLM 实例，不提供则用默认模型
        max_iterations: 最大工具调用轮次，防止无限循环
        context: 可选的前置任务执行结果上下文，注入到 prompt 中

    Returns:
        更新后的 Task 实例，result 或 error 已写入
    """
    if llm is None:
        llm = get_llm_model()

    llm_with_tools = llm.bind_tools(_ALL_TOOLS)
    instruction = _build_prompt(task, context)

    messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=f"请完成以下任务:\n{task.description}"),
    ]

    for _ in range(max_iterations):
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"LLM 调用失败: {str(e)}"
            return task

        messages.append(response)

        # 检查是否有 tool_call
        if not response.tool_calls:
            # 没有 tool call 说明模型已给出最终回复
            if response.content and response.content.strip():
                task.result = response.content
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.FAILED
                task.error = "模型返回空结果，未调用工具也未给出回复"
            return task

        # 执行所有 tool call
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")

            if tool_name not in _TOOL_MAP:
                result_content = {"error": f"未知工具: {tool_name}"}
            else:
                try:
                    tool_fn = _TOOL_MAP[tool_name]
                    result_content = tool_fn.invoke(tool_args)
                except Exception as e:
                    result_content = {"error": f"工具调用失败 ({tool_name}): {str(e)}"}

            messages.append(ToolMessage(
                content=str(result_content),
                tool_call_id=tool_id,
            ))

    # 超出循环次数
    task.status = TaskStatus.FAILED
    task.error = f"超出最大工具调用轮次 ({max_iterations})，任务未完成"
    return task


# ---------------------------------------------------------------------------
# Async variant — used by graph_builder's asyncio.gather
# ---------------------------------------------------------------------------


async def async_execute_task(
    task: Task,
    max_iterations: int = _MAX_TOOL_CALLS,
    context: str = "",
) -> Task:
    """异步版本的 execute_task，支持取消（CancelledError）。

    Args:
        task: 待执行的 Task 实例
        max_iterations: 最大工具调用轮次
        context: 前置任务执行结果上下文

    Returns:
        更新后的 Task 实例

    Raises:
        asyncio.CancelledError: 任务被取消时抛出
    """
    llm = get_llm_model()
    llm_with_tools = llm.bind_tools(_ALL_TOOLS)
    instruction = _build_prompt(task, context)

    messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=f"请完成以下任务:\n{task.description}"),
    ]

    for _ in range(max_iterations):
        try:
            response = await llm_with_tools.ainvoke(messages)
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.error = "任务已取消"
            raise
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"LLM 调用失败: {str(e)}"
            return task

        messages.append(response)

        if not response.tool_calls:
            if response.content and response.content.strip():
                task.result = response.content
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.FAILED
                task.error = "模型返回空结果，未调用工具也未给出回复"
            return task

        # 并发执行所有工具调用（同步工具通过 to_thread 包装）
        tool_coroutines = []
        tool_meta = []  # (tool_name, tool_id) 对应每个 coroutine

        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id", "")
            tool_meta.append((tool_name, tool_id))

            if tool_name not in _TOOL_MAP:
                # 未知工具，直接返回 error coroutine
                async def _unknown_tool(name=tool_name):
                    return {"error": f"未知工具: {name}"}
                tool_coroutines.append(_unknown_tool())
            else:
                tool_fn = _TOOL_MAP[tool_name]
                tool_coroutines.append(
                    asyncio.to_thread(tool_fn.invoke, tool_args)
                )

        results = await asyncio.gather(*tool_coroutines, return_exceptions=True)

        for (tool_name, tool_id), result_or_exc in zip(tool_meta, results):
            if isinstance(result_or_exc, Exception):
                result_content = {"error": f"工具调用失败 ({tool_name}): {result_or_exc}"}
            else:
                result_content = result_or_exc

            messages.append(ToolMessage(
                content=str(result_content),
                tool_call_id=tool_id,
            ))

    task.status = TaskStatus.FAILED
    task.error = f"超出最大工具调用轮次 ({max_iterations})，任务未完成"
    return task


# ---------------------------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich import print as rprint
    from rich.console import Console

    from utils.llm import reset_llm_models
    from utils.embedding import reset_embedding_model

    console = Console()

    demo_tasks = [
        Task(
            id="task-search-ai",
            description="搜索2024年人工智能领域的最新进展，总结3个重要趋势",
        ),
        Task(
            id="task-arxiv-llm",
            description="在arXiv上搜索关于大语言模型(LLM)安全的最新论文，列出3篇有影响力的论文，包括标题、作者和摘要",
        ),
    ]

    for i, t in enumerate(demo_tasks, 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]示例 {i}: {t.description}[/bold cyan]")
        console.print(f"{'='*60}")

        reset_llm_models()
        reset_embedding_model()

        result = execute_task(t)

        console.print(f"\n  [bold]Task ID:[/bold]  {result.id}")
        console.print(f"  [bold]Status:[/bold]   {result.status}")
        if result.result:
            preview = result.result[:500] + "..." if len(result.result) > 500 else result.result
            console.print(f"  [bold]Result:[/bold]   {preview}")
        if result.error:
            console.print(f"  [bold red]Error:[/bold red]    {result.error}")
