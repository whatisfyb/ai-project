"""Agent Tool - 用于派发任务给子代理

使用 @tool 装饰器封装，提供统一的子代理调用接口。
"""

from typing import Any, Literal, TYPE_CHECKING
from langchain_core.tools import tool

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from agent.subagents import PlanAgent, ResearchAgent, AnalysisAgent


def _get_subagent_classes():
    """延迟加载子代理类，避免循环导入"""
    from agent.subagents import PlanAgent, ResearchAgent, AnalysisAgent
    return {
        "Plan": PlanAgent,
        "Research": ResearchAgent,
        "Analysis": AnalysisAgent,
    }

_SUBAGENT_DESCRIPTIONS = {
    "Plan": "规划代理，用于分析复杂需求并拆解为可执行的子任务",
    "Research": "研究代理，用于搜索信息、下载论文、使用RAG知识库",
    "Analysis": "分析代理，用于数据分析、报告生成、可视化建议",
    "ExecutePlan": "执行代理，用于执行复杂的多步骤任务（自动拆解并并行执行）",
}


def _get_subagent_description(subagent_type: str) -> str:
    """获取子代理描述"""
    return _SUBAGENT_DESCRIPTIONS.get(subagent_type, "未知代理类型")


def _run_subagent(subagent_type: str, prompt: str) -> dict[str, Any]:
    """运行子代理的内部实现"""
    subagent_classes = _get_subagent_classes()

    if subagent_type == "ExecutePlan":
        # 特殊处理：执行计划
        return _run_execute_plan(prompt)

    if subagent_type not in subagent_classes:
        return {
            "status": "error",
            "error": f"未知的子代理类型: {subagent_type}",
        }

    agent_class = subagent_classes[subagent_type]
    agent = agent_class()

    # 根据代理类型传递正确的参数
    # PlanAgent.run(task=...), ResearchAgent.run(query=...), AnalysisAgent.run(task=...)
    if subagent_type == "Research":
        result = agent.run(query=prompt)
    else:  # Plan 和 Analysis 都用 task 参数
        result = agent.run(task=prompt)

    return result


def _run_execute_plan(prompt: str) -> dict[str, Any]:
    """执行复杂任务：生成计划 → 执行 → 汇总

    Args:
        prompt: 任务描述

    Returns:
        执行结果
    """
    from agent.subagents import PlanAgent
    from agent.executor import PlanExecutor, execute_task_with_llm

    # 1. 生成计划
    plan_agent = PlanAgent()
    plan, plan_id = plan_agent.run(task=prompt)

    # 2. 执行计划
    executor = PlanExecutor(
        plan_id=plan_id,
        memory=[{"role": "user", "content": prompt}],
        execute_fn=execute_task_with_llm,
        num_workers=2,
    )
    result = executor.run()

    # 3. 返回结果
    return {
        "plan_id": plan_id,
        "goal": result["goal"],
        "completed": result["completed"],
        "failed": result["failed"],
        "total": result["total"],
        "summarized_result": result.get("summarized_result"),
        "tasks": result["tasks"],
    }


def _generate_summary(subagent_type: str, result: dict) -> str:
    """生成结果摘要"""
    if subagent_type == "Plan":
        steps = result.get("steps", [])
        return f"生成了 {len(steps)} 个执行步骤"
    elif subagent_type == "Research":
        search_count = len(result.get("search_results", []))
        paper_count = len(result.get("papers", []))
        return f"完成研究：{search_count} 个搜索结果，{paper_count} 篇论文"
    elif subagent_type == "Analysis":
        file_path = result.get("file_path")
        if file_path:
            return f"分析完成，报告已保存到 {file_path}"
        return "分析完成"
    elif subagent_type == "ExecutePlan":
        completed = result.get("completed", 0)
        failed = result.get("failed", 0)
        total = result.get("total", 0)
        return f"执行完成：{completed}/{total} 成功，{failed} 失败"
    return "任务完成"


@tool
def dispatch_agent(
    subagent_type: Literal["Plan", "Research", "Analysis", "ExecutePlan"],
    prompt: str,
) -> dict[str, Any]:
    """Dispatch a task to a specialized subagent for execution.

    Use this tool when you need to delegate work to specialized agents:
    - Plan: For breaking down complex tasks into executable steps
    - Research: For searching information, downloading papers, using RAG knowledge base
    - Analysis: For data analysis, report generation, visualization suggestions
    - ExecutePlan: For executing complex multi-step tasks (auto breaks down and runs in parallel)

    Args:
        subagent_type: Type of subagent to use (Plan/Research/Analysis/ExecutePlan)
        prompt: Clear task description of what needs to be done

    Returns:
        Dictionary containing execution results with status, summary, and data
    """
    try:
        result = _run_subagent(subagent_type, prompt)
        summary = _generate_summary(subagent_type, result)
        return {
            "status": "completed",
            "subagent_type": subagent_type,
            "prompt": prompt,
            "result": result,
            "summary": summary,
        }
    except Exception as e:
        return {
            "status": "error",
            "subagent_type": subagent_type,
            "prompt": prompt,
            "error": str(e),
        }


@tool
def list_subagents() -> dict[str, Any]:
    """List all available subagents with their descriptions.

    Returns:
        Dictionary containing all subagent types and their descriptions
    """
    return {
        "subagents": [
            {"type": k, "description": v}
            for k, v in _SUBAGENT_DESCRIPTIONS.items()
        ]
    }
