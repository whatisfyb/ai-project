"""Reviewer Agent — 宽松审核计划或执行结果"""

import json
import re
from pathlib import Path
from typing import Optional, Union

from langchain_openai import ChatOpenAI

from agents.planner.plan_structure import Plan, Task, TaskStatus
from utils.llm import get_llm_model

# ---------------------------------------------------------------------------
# Prompt (loaded from reviewer_prompt.txt)
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent / "reviewer_prompt.txt"
_REVIEWER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Review result type
# ---------------------------------------------------------------------------


class ReviewResult:
    """审查结果"""

    def __init__(
        self,
        review_type: str,
        passed: bool,
        summary: str,
        details: list[dict],
        overall_comment: str,
    ):
        self.review_type = review_type  # "plan" or "result"
        self.passed = passed
        self.summary = summary
        self.details = details
        self.overall_comment = overall_comment

    def __repr__(self) -> str:
        icon = "✅" if self.passed else "❌"
        return f"ReviewResult({icon} {self.summary})"


# ---------------------------------------------------------------------------
# JSON parsing (same as planner)
# ---------------------------------------------------------------------------

def _parse_json_from_text(text: str) -> dict:
    """从 LLM 输出中提取 JSON"""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从输出中解析出合法 JSON: {text[:300]}...")


# ---------------------------------------------------------------------------
# Helper: build review content
# ---------------------------------------------------------------------------

def _build_review_plan(plan: Plan) -> str:
    """构建计划审查的上下文"""
    lines = [
        f"审查模式: 模式 A — 审查计划",
        f"计划 ID: {plan.id}",
        f"计划目标: {plan.goal}",
        f"任务数量: {len(plan.tasks)}",
        "",
        "任务详情:",
    ]
    for t in plan.tasks:
        deps = ", ".join(t.dependencies) if t.dependencies else "无"
        lines.append(f"  [{t.id}] {t.description} (依赖: {deps})")
    return "\n".join(lines)


def _build_review_task(task: Task) -> str:
    """构建单个任务执行结果审查的上下文"""
    lines = [
        f"审查模式: 模式 B — 审查执行结果",
        f"任务 ID: {task.id}",
        f"任务描述: {task.description}",
        f"任务状态: {task.status}",
    ]
    if task.result:
        lines.append(f"\n执行结果:\n{task.result}")
    if task.error:
        lines.append(f"\n错误信息:\n{task.error}")
    if not task.result and not task.error:
        lines.append("\n无执行结果或错误信息")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def review_plan(
    plan: Plan,
    llm: Optional[ChatOpenAI] = None,
) -> ReviewResult:
    """审查一个计划

    Args:
        plan: 待审查的 Plan 实例
        llm: 可选的 LLM 实例

    Returns:
        ReviewResult 实例
    """
    if llm is None:
        llm = get_llm_model()

    context = _build_review_plan(plan)
    instruction = _REVIEWER_PROMPT.format(review_instruction=context)

    response = llm.invoke(instruction)
    content = response.content if hasattr(response, "content") else str(response)

    try:
        data = _parse_json_from_text(content)
        return ReviewResult(
            review_type=data.get("review_type", "plan"),
            passed=data.get("passed", True),
            summary=data.get("summary", ""),
            details=data.get("details", []),
            overall_comment=data.get("overall_comment", ""),
        )
    except Exception:
        # JSON 解析失败时，返回默认通过
        return ReviewResult(
            review_type="plan",
            passed=True,
            summary="无法解析 LLM 响应为结构化审查结果，默认放行",
            details=[],
            overall_comment=content[:500],
        )


def review_task_result(
    task: Task,
    llm: Optional[ChatOpenAI] = None,
) -> ReviewResult:
    """审查单个任务的执行结果

    Args:
        task: 待审查的 Task 实例（result/error 已填充）
        llm: 可选的 LLM 实例

    Returns:
        ReviewResult 实例
    """
    if llm is None:
        llm = get_llm_model()

    context = _build_review_task(task)
    instruction = _REVIEWER_PROMPT.format(review_instruction=context)

    response = llm.invoke(instruction)
    content = response.content if hasattr(response, "content") else str(response)

    try:
        data = _parse_json_from_text(content)
        return ReviewResult(
            review_type=data.get("review_type", "result"),
            passed=data.get("passed", True),
            summary=data.get("summary", ""),
            details=data.get("details", []),
            overall_comment=data.get("overall_comment", ""),
        )
    except Exception:
        return ReviewResult(
            review_type="result",
            passed=True,
            summary="无法解析 LLM 响应为结构化审查结果，默认放行",
            details=[],
            overall_comment=content[:500],
        )


def review_all_tasks(
    plan: Plan,
    llm: Optional[ChatOpenAI] = None,
) -> list[tuple[str, ReviewResult]]:
    """批量审查计划中所有任务的执行结果

    Args:
        plan: Plan 实例（所有 task 应有 result 或 error）
        llm: 可选的 LLM 实例

    Returns:
        (task_id, ReviewResult) 列表
    """
    results = []
    for task in plan.tasks:
        result = review_task_result(task, llm)
        results.append((task.id, result))
    return results


# ---------------------------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich.console import Console

    from utils.llm import reset_llm_models
    from utils.embedding import reset_embedding_model

    console = Console()

    # Demo 1: Review a Plan
    console.print("\n[bold cyan]=== 示例 1: 审查计划 ===[/bold cyan]")

    demo_plan = Plan(
        id="demo-article",
        goal="写一篇关于人工智能在科研中应用的文章",
        tasks=[
            Task(id="task-1", description="搜集AI在科研领域的最新应用案例", dependencies=[]),
            Task(id="task-2", description="撰写文章大纲", dependencies=["task-1"]),
            Task(id="task-3", description="完成全文撰写", dependencies=["task-2"]),
            Task(id="task-4", description="校对和发布", dependencies=["task-2"]),  # 故意设置的异常：task-4 只依赖 task-2，跳过了 task-3
        ],
    )

    reset_llm_models()
    reset_embedding_model()

    review = review_plan(demo_plan)
    console.print(f"\n  [bold]审查类型:[/bold] {review.review_type}")
    console.print(f"  [bold]通过:[/bold] {review.passed}")
    console.print(f"  [bold]总结:[/bold] {review.summary}")
    for d in review.details:
        console.print(f"    [{d['item_id']}] {d['status']}: {d['comment']}")
    console.print(f"  [bold]整体意见:[/bold] {review.overall_comment}")

    # Demo 2: Review a Task Result
    console.print("\n[bold cyan]=== 示例 2: 审查执行结果 ===[/bold cyan]")

    demo_task = Task(
        id="task-search",
        description="搜索2024年AI在药物研发领域的应用",
        status=TaskStatus.COMPLETED,
        result="2024年AI在药物研发领域主要应用于三个方面：(1)AlphaFold3预测蛋白质结构，加速靶点发现；(2)生成式AI模型设计新型分子结构，如Insilico Medicine的Pharma.AI平台；(3)机器学习预测药物毒性和药代动力学性质，降低临床试验失败率。其中AlphaFold3于2024年5月发布，可预测蛋白质-配体相互作用。",
    )

    reset_llm_models()
    reset_embedding_model()

    review = review_task_result(demo_task)
    console.print(f"\n  [bold]审查类型:[/bold] {review.review_type}")
    console.print(f"  [bold]通过:[/bold] {review.passed}")
    console.print(f"  [bold]总结:[/bold] {review.summary}")
    for d in review.details:
        console.print(f"    [{d['item_id']}] {d['status']}: {d['comment']}")
    console.print(f"  [bold]整体意见:[/bold] {review.overall_comment}")