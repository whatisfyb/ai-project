"""Synthesizer Agent — 将多个任务的执行结果整合为最终报告"""

import asyncio
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI

from agents.planner.plan_structure import Plan
from utils.llm import get_llm_model

# ---------------------------------------------------------------------------
# Prompt (loaded from synthesizer_prompt.txt)
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "synthesizer_prompt.txt"
_SYNTHESIZER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def async_synthesize(
    plan: Plan,
    results: dict[str, str],
    llm: Optional["ChatOpenAI"] = None,
) -> str:
    """异步版本的 synthesize

    Args:
        plan: Plan 实例
        results: {task_id: result_text} 执行结果字典
        llm: 可选 LLM 实例

    Returns:
        格式化后的最终报告文本
    """
    if llm is None:
        llm = get_llm_model()

    task_result_lines = []
    for task in plan.tasks:
        res = results.get(task.id, task.result or "[无结果]")
        task_result_lines.append(f"### [{task.id}] {task.description}")
        task_result_lines.append(f"状态: {task.status}")
        task_result_lines.append(f"结果:\n{res}\n")

    task_results = "\n\n".join(task_result_lines)

    prompt = _SYNTHESIZER_PROMPT.format(
        plan_goal=plan.goal,
        task_results=task_results,
    )

    response = await llm.ainvoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def synthesize(
    plan: Plan,
    results: dict[str, str],
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """将任务执行结果整合为最终报告

    Args:
        plan: Plan 实例，提供目标和任务结构
        results: {task_id: result_text} 执行结果字典
        llm: 可选的 LLM 实例

    Returns:
        格式化后的最终报告文本
    """
    if llm is None:
        llm = get_llm_model()

    # 构建 prompt 上下文
    task_result_lines = []
    for task in plan.tasks:
        res = results.get(task.id, task.result or "[无结果]")
        task_result_lines.append(f"### [{task.id}] {task.description}")
        task_result_lines.append(f"状态: {task.status}")
        task_result_lines.append(f"结果:\n{res}\n")

    task_results = "\n\n".join(task_result_lines)

    prompt = _SYNTHESIZER_PROMPT.format(
        plan_goal=plan.goal,
        task_results=task_results,
    )

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)
