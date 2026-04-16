"""Agent 工具 - 统一入口

支持：
- dispatch: 分发任务到子代理
- list: 列出可用子代理
"""

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
        return {"success": False, "error": f"Unknown subagent: {subagent_type}"}

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
        return (
            f"分析完成: {result.get('file_path', '')}"
            if result.get("file_path")
            else "分析完成"
        )
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
            return {"success": False, "error": f"Unknown action: {action}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _dispatch(subagent_type: str, prompt: str) -> dict[str, Any]:
    """分发任务"""
    if not subagent_type:
        return {"success": False, "error": "subagent_type is required"}
    if not prompt:
        return {"success": False, "error": "prompt is required"}

    result = _run_subagent(subagent_type, prompt)
    summary = _generate_summary(subagent_type, result)

    return {
        "success": True,
        "subagent_type": subagent_type,
        "result": result,
        "summary": summary,
    }


def _list_subagents() -> dict[str, Any]:
    """列出子代理"""
    return {
        "success": True,
        "subagents": [
            {"type": k, "description": v} for k, v in _SUBAGENT_DESCRIPTIONS.items()
        ],
    }


# ============ Registry 集成工具 ============


@tool
def agent_list() -> dict[str, Any]:
    """List all registered agents in the registry.

    Returns information about all agents including their state and capabilities.
    """
    from agent.core.registry import get_registry, AgentLifecycleState

    registry = get_registry()
    agents = registry.list_agents()
    states = registry.get_all_states()

    agent_info = []
    for card in agents:
        state = states.get(card.id, AgentLifecycleState.PENDING)
        agent_info.append(
            {
                "agent_id": card.id,
                "name": card.name,
                "description": card.description,
                "state": state.value,
                "skills": [s.name for s in card.skills],
            }
        )

    return {
        "success": True,
        "count": len(agent_info),
        "agents": agent_info,
    }


@tool
def agent_status(agent_id: str) -> dict[str, Any]:
    """Get the status of a specific agent.

    Args:
        agent_id: The agent ID to check

    Returns:
        Agent status and details
    """
    from agent.core.registry import get_registry, AgentLifecycleState
    from agent.core.signals import has_checkpoint

    registry = get_registry()

    state = registry.get_state(agent_id)
    if state is None:
        return {"success": False, "error": f"Agent not found: {agent_id}"}

    agents = registry.list_agents()
    card = next((a for a in agents if a.id == agent_id), None)

    return {
        "success": True,
        "agent_id": agent_id,
        "state": state.value,
        "has_checkpoint": has_checkpoint(agent_id),
        "card": {
            "name": card.name if card else "Unknown",
            "description": card.description if card else "",
            "skills": [s.name for s in card.skills] if card else [],
        }
        if card
        else None,
    }


@tool
def agent_dispatch(
    agent_type: str,
    prompt: str,
    wait: bool = True,
    timeout: int = 60,
) -> dict[str, Any]:
    """Dispatch a task to an agent.

    Args:
        agent_type: Type of agent (Plan, Research, Analysis)
        prompt: Task prompt
        wait: Whether to wait for completion (default: True)
        timeout: Timeout in seconds when waiting (default: 60)

    Returns:
        Dispatch result
    """
    import time
    import threading
    from agent.core.registry import get_registry
    from agent.a2a.models import Task, Message

    registry = get_registry()

    # 查找 Agent
    agent_map = {
        "Plan": "plan-agent",
        "Research": "research-agent",
        "Analysis": "analysis-agent",
    }

    agent_id = agent_map.get(agent_type)
    if not agent_id:
        return {"success": False, "error": f"Unknown agent type: {agent_type}"}

    state = registry.get_state(agent_id)
    if state is None:
        return {"success": False, "error": f"Agent not registered: {agent_type}"}

    # 如果 Agent 处于 pending 状态，通过 Registry 激活（懒加载）
    from agent.core.registry import AgentLifecycleState

    if state == AgentLifecycleState.PENDING:
        # 通过 send_message 激活 Agent
        task = Task(
            id=f"task-{agent_type.lower()}-{int(time.time())}",
            sender_id="main",
            receiver_id=agent_id,
            history=[Message.user_text(prompt)],
            metadata={"thread_id": _get_current_thread_id()},
        )
        sent = registry.send_message(agent_id, task)
        if not sent:
            return {
                "success": False,
                "error": f"Failed to activate agent: {agent_type}",
            }

    # 运行子代理（直接调用，因为 subagents 目前是同步执行的）
    def run_subagent():
        subagent_classes = _get_subagent_classes()
        if agent_type not in subagent_classes:
            return {"success": False, "error": f"Unknown subagent: {agent_type}"}

        thread_id = _get_current_thread_id()
        agent = subagent_classes[agent_type]()

        if agent_type == "Research":
            result = agent.run(query=prompt, thread_id=thread_id)
        else:
            result = agent.run(task=prompt, thread_id=thread_id)

        if isinstance(result, tuple):
            plan, plan_id = result
            return {
                "success": True,
                "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan,
                "plan_id": plan_id,
            }
        return {"success": True, "result": result}

    if not wait:
        # 异步执行
        result_holder = {"result": None, "done": False}

        def run_async():
            result_holder["result"] = run_subagent()
            result_holder["done"] = True

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        return {
            "success": True,
            "agent_type": agent_type,
            "status": "dispatched",
            "message": f"Task dispatched to {agent_type}. Check status later.",
        }

    # 同步等待
    return run_subagent()
