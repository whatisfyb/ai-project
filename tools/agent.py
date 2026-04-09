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
}


def _get_subagent_description(subagent_type: str) -> str:
    """获取子代理描述"""
    return _SUBAGENT_DESCRIPTIONS.get(subagent_type, "未知代理类型")


def _get_current_thread_id() -> str:
    """获取当前 thread_id（从 context variable）"""
    from agent.main_agent import _current_thread_id
    return _current_thread_id.get()


def _run_subagent(subagent_type: str, prompt: str) -> dict[str, Any]:
    """运行子代理的内部实现"""
    subagent_classes = _get_subagent_classes()

    if subagent_type not in subagent_classes:
        return {
            "status": "error",
            "error": f"未知的子代理类型: {subagent_type}",
        }

    # 获取当前会话的 thread_id
    thread_id = _get_current_thread_id()

    agent_class = subagent_classes[subagent_type]
    agent = agent_class()

    # 根据代理类型传递正确的参数
    # 所有代理都接收 thread_id 参数
    if subagent_type == "Research":
        result = agent.run(query=prompt, thread_id=thread_id)
    else:  # Plan 和 Analysis 都用 task 参数
        result = agent.run(task=prompt, thread_id=thread_id)

    # 统一转换为 dict 格式
    if isinstance(result, tuple):
        plan, plan_id = result
        return {
            "plan": plan,
            "plan_id": plan_id,
        }
    return result


def _generate_summary(subagent_type: str, result: dict) -> str:
    """生成结果摘要"""
    if subagent_type == "Plan":
        plan = result.get("plan")
        if plan:
            return f"生成了 {len(plan.tasks)} 个执行步骤"
        return "生成了执行计划"
    elif subagent_type == "Research":
        search_count = len(result.get("search_results", []))
        paper_count = len(result.get("papers", []))
        return f"完成研究：{search_count} 个搜索结果，{paper_count} 篇论文"
    elif subagent_type == "Analysis":
        file_path = result.get("file_path")
        if file_path:
            return f"分析完成，报告已保存到 {file_path}"
        return "分析完成"
    return "任务完成"


@tool
def dispatch_agent(
    subagent_type: Literal["Plan", "Research", "Analysis"],
    prompt: str,
) -> dict[str, Any]:
    """Dispatch a task to a specialized subagent for execution.

    Use this tool when you need to delegate work to specialized agents:
    - Plan: For breaking down complex tasks into executable steps
    - Research: For searching information, downloading papers, using RAG knowledge base
    - Analysis: For data analysis, report generation, visualization suggestions

    For managing and executing plans, use the plan_* tools (plan_list, plan_get, plan_execute, etc.)

    Args:
        subagent_type: Type of subagent to use (Plan/Research/Analysis)
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
