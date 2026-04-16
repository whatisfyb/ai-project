"""Agent 初始化模块

集中管理所有 Agent 的注册和启动：
- Main Agent 自动启动
- 其他 Agent 懒加载
- Worker Pool 管理
"""

import threading
from typing import Optional

from agent.core.registry import get_registry, AgentLifecycleState
from agent.a2a.models import AgentCard, AgentCapabilities, Skill


def bootstrap_agents(num_workers: int = 2) -> None:
    """初始化所有 Agent

    Args:
        num_workers: Worker 数量
    """
    registry = get_registry()

    # 注册 Main Agent
    _register_main_agent(registry)

    # 注册 Subagents（懒加载）
    _register_subagents(registry)

    # 注册 Worker Pool
    _register_workers(registry, num_workers)

    # 启动空闲检查线程
    registry.start_idle_checker()


def _register_main_agent(registry) -> None:
    """注册 Main Agent"""
    from agent.main.agent import MainAgent

    main_agent = MainAgent()

    registry.register(
        agent_id="main",
        agent_type="main",
        card=main_agent.get_card(),
        factory=lambda: main_agent,
        auto_start=True,  # Main Agent 自动启动
    )


def _register_subagents(registry) -> None:
    """注册子代理（懒加载）"""
    # Plan Agent
    plan_card = AgentCard(
        id="plan-agent",
        name="Plan Agent",
        description="规划代理，分析复杂需求并拆解为子任务",
        capabilities=AgentCapabilities(text=True),
        skills=[
            Skill(name="plan", description="生成执行计划"),
            Skill(name="decompose", description="拆解复杂任务"),
        ],
    )
    registry.register(
        agent_id="plan-agent",
        agent_type="subagent",
        card=plan_card,
        factory=_create_plan_agent,
        auto_start=False,
    )

    # Research Agent
    research_card = AgentCard(
        id="research-agent",
        name="Research Agent",
        description="研究代理，搜索信息、下载论文、使用知识库",
        capabilities=AgentCapabilities(text=True, files=True),
        skills=[
            Skill(name="research", description="研究分析"),
            Skill(name="search", description="网络搜索"),
            Skill(name="paper_kb", description="论文知识库"),
        ],
    )
    registry.register(
        agent_id="research-agent",
        agent_type="subagent",
        card=research_card,
        factory=_create_research_agent,
        auto_start=False,
    )

    # Analysis Agent
    analysis_card = AgentCard(
        id="analysis-agent",
        name="Analysis Agent",
        description="分析代理，数据分析、报告生成",
        capabilities=AgentCapabilities(text=True, files=True),
        skills=[
            Skill(name="analysis", description="数据分析"),
            Skill(name="report", description="报告生成"),
        ],
    )
    registry.register(
        agent_id="analysis-agent",
        agent_type="subagent",
        card=analysis_card,
        factory=_create_analysis_agent,
        auto_start=False,
    )


def _create_plan_agent():
    """创建 Plan Agent 实例"""
    from agent.subagents import PlanAgent

    return PlanAgent()


def _create_research_agent():
    """创建 Research Agent 实例"""
    from agent.subagents import ResearchAgent

    return ResearchAgent()


def _create_analysis_agent():
    """创建 Analysis Agent 实例"""
    from agent.subagents import AnalysisAgent

    return AnalysisAgent()


def _register_workers(registry, num_workers: int = 2) -> None:
    """注册 Worker Pool"""
    from agent.a2a.worker import A2AWorker
    from agent.a2a.transport import get_transport

    transport = get_transport()

    for i in range(num_workers):
        worker_id = f"worker-{i + 1}"
        worker = A2AWorker(worker_id=worker_id, transport=transport)

        # 先启动 Worker（注册到 Transport）
        worker.start()

        # 再注册到 Registry（标记为 RUNNING）
        registry.register_with_terminator(
            agent_id=worker_id,
            agent_type="worker",
            card=worker.get_card(),
            terminate_fn=worker.stop_nowait,
        )


def shutdown_agents() -> None:
    """关闭所有 Agent"""
    registry = get_registry()
    registry.stop_idle_checker()
    registry.terminate_all()


# 模块级变量跟踪初始化状态
_initialized = False
_init_lock = threading.Lock()


def ensure_initialized(num_workers: int = 2) -> bool:
    """确保 Agent 系统已初始化

    Args:
        num_workers: Worker 数量

    Returns:
        是否首次初始化
    """
    global _initialized

    with _init_lock:
        if _initialized:
            return False

        bootstrap_agents(num_workers)
        _initialized = True
        return True


def is_initialized() -> bool:
    """检查是否已初始化"""
    return _initialized
