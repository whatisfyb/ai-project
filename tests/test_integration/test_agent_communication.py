"""Agent 通信集成测试

测试：
- Registry Agent 注册与激活
- Inbox subscribe 机制
- MainAgent 事件模型
- BaseAgent 接口
- Agent 懒加载机制
"""

import pytest
import threading
import queue

from agent.core.registry import AgentRegistry, AgentLifecycleState, get_registry
from agent.core.signals import (
    set_interrupt,
    clear_interrupt,
    is_interrupted,
    set_interrupt_for,
    clear_interrupt_for,
    is_interrupted_for,
    save_checkpoint,
    load_checkpoint,
    clear_checkpoint,
    has_checkpoint,
)
from agent.core.events import AgentEvent, EventType
from agent.core.base_agent import BaseAgent
from agent.a2a.dispatcher import Inbox, TaskResult, TaskResultStatus
from agent.a2a.models import AgentCard, AgentCapabilities, Skill, Task, Message


# ============ Helper: 测试用 Agent ============


class MockAgent(BaseAgent):
    agent_id = "mock-agent"
    agent_type = "mock"

    def __init__(self):
        self.last_task = None
        self._state_data = {}
        self.handled_tasks = []

    def get_card(self) -> AgentCard:
        return AgentCard(
            id=self.agent_id,
            name="Mock Agent",
            description="Test mock agent",
            capabilities=AgentCapabilities(text=True),
            skills=[Skill(name="mock_skill", description="Mock skill")],
        )

    def handle_task(self, task) -> str:
        self.last_task = task
        self.handled_tasks.append(task)
        return "mock_result"

    def get_state(self) -> dict:
        return dict(self._state_data)

    def restore_state(self, state: dict) -> None:
        self._state_data = dict(state)


# ============ Registry 测试 ============


class TestRegistryIntegration:
    """Registry 集成测试"""

    def setup_method(self):
        """每个测试前重置 Registry"""
        self.registry = AgentRegistry()
        self.registry._agents.clear()

    def test_register_and_get_state(self):
        """测试注册并查询状态"""
        agent = MockAgent()
        self.registry.register(
            agent_id="mock-agent",
            agent_type="mock",
            card=agent.get_card(),
            factory=lambda: agent,
        )
        assert self.registry.get_state("mock-agent") == AgentLifecycleState.PENDING

    def test_send_message_activates_agent(self):
        """测试通过 send_message 激活 Agent"""
        agent = MockAgent()
        self.registry.register(
            agent_id="mock-agent",
            agent_type="mock",
            card=agent.get_card(),
            factory=lambda: agent,
        )
        assert self.registry.get_state("mock-agent") == AgentLifecycleState.PENDING

        task = Task(
            id="test-task-1",
            sender_id="main",
            receiver_id="mock-agent",
            history=[Message.user_text("hello")],
        )

        result = self.registry.send_message("mock-agent", task)
        assert result is True
        assert self.registry.get_state("mock-agent") == AgentLifecycleState.RUNNING

        # 等待 Agent 线程启动并处理任务
        import time

        time.sleep(0.5)

    def test_find_agents_by_skill(self):
        """测试按技能查找 Agent"""
        agent = MockAgent()
        self.registry.register(
            agent_id="mock-agent",
            agent_type="mock",
            card=agent.get_card(),
            factory=lambda: agent,
        )

        # PENDING 状态不可发现
        found = self.registry.find_agents_by_skill("mock_skill")
        assert len(found) == 0

        # RUNNING 状态可发现
        self.registry._agents["mock-agent"].state = AgentLifecycleState.RUNNING
        found = self.registry.find_agents_by_skill("mock_skill")
        assert len(found) == 1
        assert found[0].id == "mock-agent"

    def test_broadcast(self):
        """测试广播消息"""
        agent1 = MockAgent()
        agent2 = MockAgent()
        agent2.agent_id = "mock-agent-2"

        self.registry.register(
            agent_id="mock-agent",
            agent_type="mock",
            card=agent1.get_card(),
            factory=lambda: agent1,
        )
        self.registry.register(
            agent_id="mock-agent-2",
            agent_type="mock",
            card=agent2.get_card(),
            factory=lambda: agent2,
        )

        task = Task(
            id="broadcast-test",
            sender_id="main",
            receiver_id="mock-agent",
            history=[Message.user_text("broadcast")],
        )

        dispatched = self.registry.broadcast(task, agent_types=["mock"])
        assert len(dispatched) == 2

    def test_auto_start(self):
        """测试 auto_start 自动启动"""
        agent = MockAgent()
        self.registry.register(
            agent_id="mock-agent",
            agent_type="mock",
            card=agent.get_card(),
            factory=lambda: agent,
            auto_start=True,
        )
        # auto_start=True 会自动调用 _activate_agent
        assert self.registry.get_state("mock-agent") == AgentLifecycleState.RUNNING

        # 清理
        self.registry.unregister("mock-agent")

    def test_unregister(self):
        """测试注销 Agent"""
        agent = MockAgent()
        self.registry.register(
            agent_id="mock-agent",
            agent_type="mock",
            card=agent.get_card(),
        )
        assert self.registry.get_state("mock-agent") is not None

        result = self.registry.unregister("mock-agent")
        assert result is True
        assert self.registry.get_state("mock-agent") is None


# ============ Inbox 测试 ============


class TestInboxIntegration:
    """Inbox subscribe 机制集成测试"""

    def test_inbox_subscribe_and_put(self):
        """测试 Inbox subscribe 和 put"""
        inbox = Inbox()
        received = []

        inbox.subscribe(lambda r: received.append(r))

        result = TaskResult(
            plan_id="plan-1",
            task_id="task-1",
            status=TaskResultStatus.SUCCESS,
            result="done",
        )
        inbox.put(result)

        assert len(received) == 1
        assert received[0].plan_id == "plan-1"
        assert received[0].status == TaskResultStatus.SUCCESS

    def test_inbox_get_all(self):
        """测试 Inbox get_all"""
        inbox = Inbox()

        inbox.put(
            TaskResult(plan_id="p1", task_id="t1", status=TaskResultStatus.SUCCESS)
        )
        inbox.put(
            TaskResult(plan_id="p2", task_id="t2", status=TaskResultStatus.FAILED)
        )

        results = inbox.get_all()
        assert len(results) == 2

    def test_inbox_unsubscribe(self):
        """测试 Inbox unsubscribe"""
        inbox = Inbox()
        received = []

        callback = lambda r: received.append(r)
        inbox.subscribe(callback)
        inbox.unsubscribe(callback)

        inbox.put(
            TaskResult(plan_id="p1", task_id="t1", status=TaskResultStatus.SUCCESS)
        )
        assert len(received) == 0


# ============ Events 测试 ============


class TestEventsIntegration:
    """事件模型集成测试"""

    def test_user_input_event(self):
        """测试 USER_INPUT 事件创建"""
        event = AgentEvent.user_input("hello", "thread-1")
        assert event.type == EventType.USER_INPUT
        assert event.data["message"] == "hello"
        assert event.thread_id == "thread-1"

    def test_inbox_notification_event(self):
        """测试 INBOX_NOTIFICATION 事件创建"""
        event = AgentEvent.inbox_notification(
            task_id="task-1",
            status="success",
            result="done",
        )
        assert event.type == EventType.INBOX_NOTIFICATION
        assert event.data["task_id"] == "task-1"
        assert event.data["status"] == "success"

    def test_shutdown_event(self):
        """测试 SHUTDOWN 事件创建"""
        event = AgentEvent.shutdown()
        assert event.type == EventType.SHUTDOWN

    def test_event_with_on_complete(self):
        """测试事件带 on_complete 回调"""
        import asyncio

        on_complete = asyncio.Event()
        event = AgentEvent.user_input("hello", on_complete=on_complete)
        assert event.on_complete is not None


# ============ BaseAgent 接口测试 ============


class TestBaseAgentIntegration:
    """BaseAgent 接口测试"""

    def test_mock_agent_interface(self):
        """测试 MockAgent 实现了 BaseAgent 接口"""
        agent = MockAgent()

        assert agent.agent_id == "mock-agent"
        assert agent.agent_type == "mock"

        card = agent.get_card()
        assert card.id == "mock-agent"
        assert card.name == "Mock Agent"

    def test_get_state_restore_state(self):
        """测试状态保存和恢复"""
        agent = MockAgent()
        agent._state_data = {"step": 5, "data": "test"}

        state = agent.get_state()
        assert state["step"] == 5

        new_agent = MockAgent()
        new_agent.restore_state(state)
        assert new_agent._state_data["step"] == 5

    def test_default_state_methods(self):
        """测试 BaseAgent 默认方法返回空状态"""
        agent = MockAgent()
        # MockAgent 的 _state_data 初始为空 dict
        state = agent.get_state()
        assert isinstance(state, dict)

        # restore_state 不报错
        agent.restore_state({"test": "data"})
        assert agent._state_data["test"] == "data"


# ============ Signals 跨 Agent 测试 ============


class TestSignalsIntegration:
    """Signals 跨 Agent 集成测试"""

    def setup_method(self):
        clear_interrupt()
        clear_interrupt_for("agent-a")
        clear_interrupt_for("agent-b")

    def test_interrupt_does_not_affect_other_agents(self):
        """测试中断 A 不影响 B"""
        set_interrupt_for("agent-a")
        assert is_interrupted_for("agent-a") is True
        assert is_interrupted_for("agent-b") is False

        clear_interrupt_for("agent-a")

    def test_global_interrupt_overrides_per_agent(self):
        """测试全局中断优先于单 Agent 中断"""
        assert is_interrupted_for("agent-a") is False

        set_interrupt()
        assert is_interrupted_for("agent-a") is True

        clear_interrupt()

    def test_checkpoint_with_agent_state(self):
        """测试检查点与 Agent 状态"""
        agent = MockAgent()
        agent._state_data = {"step": 10, "context": "processing"}

        save_checkpoint(agent.agent_id, agent.get_state())

        assert has_checkpoint(agent.agent_id) is True

        loaded = load_checkpoint(agent.agent_id)
        assert loaded.state["step"] == 10

        clear_checkpoint(agent.agent_id)
