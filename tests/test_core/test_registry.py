"""Registry 单元测试"""

import pytest

from agent.core.registry import (
    AgentRegistry,
    AgentLifecycleState,
    get_registry,
    terminate,
)
from agent.a2a.models import AgentCard, AgentCapabilities, Skill


class TestAgentRegistry:
    """AgentRegistry 测试"""

    def test_singleton(self):
        """测试单例模式"""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_register_agent(self):
        """测试注册 Agent"""
        registry = get_registry()

        card = AgentCard(
            id="test-register",
            name="Test",
            description="Test agent",
            capabilities=AgentCapabilities(),
        )

        registry.register(
            agent_id="test-register",
            agent_type="test",
            card=card,
            factory=lambda: None,
        )

        state = registry.get_state("test-register")
        assert state == AgentLifecycleState.PENDING

        # 清理
        registry.unregister("test-register")

    def test_unregister_agent(self):
        """测试注销 Agent"""
        registry = get_registry()

        card = AgentCard(
            id="test-unregister",
            name="Test",
            description="Test agent",
            capabilities=AgentCapabilities(),
        )

        registry.register(
            agent_id="test-unregister",
            agent_type="test",
            card=card,
        )

        result = registry.unregister("test-unregister")
        assert result is True

        state = registry.get_state("test-unregister")
        assert state is None

    def test_list_agents(self):
        """测试列出 Agent"""
        registry = get_registry()

        card = AgentCard(
            id="test-list",
            name="Test",
            description="Test agent",
            capabilities=AgentCapabilities(),
            skills=[Skill(name="test_skill", description="Test")],
        )

        registry.register(
            agent_id="test-list",
            agent_type="test",
            card=card,
        )

        agents = registry.list_agents()
        agent_ids = [a.id for a in agents]
        assert "test-list" in agent_ids

        # 清理
        registry.unregister("test-list")

    def test_find_agents_by_skill(self):
        """测试按技能查找 Agent"""
        registry = get_registry()

        card = AgentCard(
            id="test-skill",
            name="Test",
            description="Test agent",
            capabilities=AgentCapabilities(),
            skills=[Skill(name="unique_skill_123", description="Unique skill")],
        )

        # 注册并设置为 running
        registry.register(
            agent_id="test-skill",
            agent_type="test",
            card=card,
        )

        # 手动设置状态为 running
        with registry._agents_lock:
            if "test-skill" in registry._agents:
                registry._agents["test-skill"].state = AgentLifecycleState.RUNNING

        found = registry.find_agents_by_skill("unique_skill_123")
        assert len(found) == 1
        assert found[0].id == "test-skill"

        # 清理
        registry.unregister("test-skill")

    def test_backward_compatible_register_executor(self):
        """测试向后兼容的 register_executor"""
        registry = get_registry()

        called = []
        registry.register_executor("test-plan-id", lambda: called.append(True))

        assert registry.is_running() is True

        # 终止
        terminated = terminate()
        assert "test-plan-id" in terminated
        assert called == [True]


class TestAgentLifecycleState:
    """AgentLifecycleState 测试"""

    def test_state_values(self):
        """测试状态值"""
        assert AgentLifecycleState.PENDING.value == "pending"
        assert AgentLifecycleState.RUNNING.value == "running"
        assert AgentLifecycleState.IDLE.value == "idle"
