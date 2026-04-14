"""Agent 工具 - 统一入口"""

from typing import Any, Literal

from langchain_core.tools import tool


_SUBAGENT_DESCRIPTIONS = {
    "Plan": "规划代理，分析复杂需求并拆解为子任务",
    "Research": "研究代理，搜索信息、下载论文、使用知识库",
    "Analysis": "分析代理，数据分析、报告生成",
}


def _get_subagent_classes():
    """延迟加载子代理类"""
    from agent.subagents import PlanAgent, ResearchAgent, AnalysisAgent
    return {"Plan": PlanAgent, "Research": ResearchAgent, "Analysis": AnalysisAgent}


def _get_current_thread_id() -> str:
    """获取当前 thread_id"""
    from agent.main.agent import _current_thread_id
    return _current_thread_id.get()


def _run_subagent(subagent_type: str, prompt: str) -> dict[str, Any]:
    """运行子代理"""
    subagent_classes = _get_subagent_classes()

    if subagent_type not in subagent_classes:
        return {"status": "error", "error": f"Unknown subagent: {subagent_type}"}

    thread_id = _get_current_thread_id()
    agent = subagent_classes[subagent_type]()

    if subagent_type == "Research":
        result = agent.run(query=prompt, thread_id=thread_id)
    else:
        result = agent.run(task=prompt, thread_id=thread_id)

    if isinstance(result, tuple):
        plan, plan_id = result
        return {"plan": plan, "plan_id": plan_id}
    return result


def _generate_summary(subagent_type: str, result: dict) -> str:
    """生成结果摘要"""
    if subagent_type == "Plan":
        plan = result.get("plan")
        return f"生成 {len(plan.tasks)} 个步骤" if plan else "生成计划"
    elif subagent_type == "Research":
        return f"完成研究：{len(result.get('search_results', []))} 搜索, {len(result.get('papers', []))} 论文"
    elif subagent_type == "Analysis":
        return f"分析完成: {result.get('file_path', '')}" if result.get('file_path') else "分析完成"
    return "完成"


@tool
def agent(
    action: Literal["dispatch", "list"],
    subagent_type: str = "",
    prompt: str = "",
) -> dict[str, Any]:
    """Agent operations for task delegation.

    Unified tool for dispatching tasks to subagents.

    Args:
        action: Operation to perform:
            - "dispatch": Dispatch task to subagent (uses: subagent_type, prompt)
            - "list": List all available subagents
        subagent_type: Type of subagent for dispatch (Plan, Research, Analysis)
        prompt: Task prompt for dispatch

    Returns:
        Operation result.
    """
    try:
        if action == "dispatch":
            return _dispatch(subagent_type, prompt)
        elif action == "list":
            return _list_subagents()
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _dispatch(subagent_type: str, prompt: str) -> dict[str, Any]:
    """分发任务"""
    if not subagent_type:
        return {"status": "error", "error": "subagent_type is required"}
    if not prompt:
        return {"status": "error", "error": "prompt is required"}

    result = _run_subagent(subagent_type, prompt)
    summary = _generate_summary(subagent_type, result)

    return {
        "status": "completed",
        "subagent_type": subagent_type,
        "result": result,
        "summary": summary,
    }


def _list_subagents() -> dict[str, Any]:
    """列出子代理"""
    return {
        "subagents": [{"type": k, "description": v} for k, v in _SUBAGENT_DESCRIPTIONS.items()]
    }
