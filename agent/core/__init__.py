"""Core 模块 - 数据模型、信号、注册表、事件"""

from agent.core.models import PlanTask, Plan, MainAgentState
from agent.core.signals import (
    set_interrupt, clear_interrupt, is_interrupted,
    set_interrupt_for, clear_interrupt_for, is_interrupted_for,
    save_checkpoint, load_checkpoint, clear_checkpoint, has_checkpoint,
)
from agent.core.registry import (
    AgentRegistry, AgentLifecycleState, get_registry,
    agent_registry, terminate,
)
from agent.core.events import AgentEvent, EventType
from agent.core.base_agent import BaseAgent

__all__ = [
    # Models
    "PlanTask",
    "Plan",
    "MainAgentState",
    # Signals
    "set_interrupt",
    "clear_interrupt",
    "is_interrupted",
    "set_interrupt_for",
    "clear_interrupt_for",
    "is_interrupted_for",
    "save_checkpoint",
    "load_checkpoint",
    "clear_checkpoint",
    "has_checkpoint",
    # Registry
    "AgentRegistry",
    "AgentLifecycleState",
    "get_registry",
    "agent_registry",
    "terminate",
    # Events
    "AgentEvent",
    "EventType",
    # BaseAgent
    "BaseAgent",
]
