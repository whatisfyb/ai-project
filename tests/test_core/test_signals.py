"""Signals 单元测试"""

import pytest

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


class TestGlobalInterrupt:
    """全局中断测试"""

    def test_set_and_clear_interrupt(self):
        """测试设置和清除中断"""
        clear_interrupt()
        assert is_interrupted() is False

        set_interrupt()
        assert is_interrupted() is True

        clear_interrupt()
        assert is_interrupted() is False

    def test_interrupt_affects_all_agents(self):
        """测试全局中断影响所有 Agent"""
        clear_interrupt()

        # 全局中断时，所有 Agent 都应该被中断
        set_interrupt()
        assert is_interrupted_for("agent-1") is True
        assert is_interrupted_for("agent-2") is True

        clear_interrupt()


class TestPerAgentInterrupt:
    """单 Agent 中断测试"""

    def test_set_and_clear_agent_interrupt(self):
        """测试设置和清除单 Agent 中断"""
        clear_interrupt()
        clear_interrupt_for("test-agent-1")

        assert is_interrupted_for("test-agent-1") is False

        set_interrupt_for("test-agent-1")
        assert is_interrupted_for("test-agent-1") is True

        clear_interrupt_for("test-agent-1")
        assert is_interrupted_for("test-agent-1") is False

    def test_different_agents_independent(self):
        """测试不同 Agent 的中断是独立的"""
        clear_interrupt()
        clear_interrupt_for("agent-a")
        clear_interrupt_for("agent-b")

        set_interrupt_for("agent-a")
        assert is_interrupted_for("agent-a") is True
        assert is_interrupted_for("agent-b") is False

        clear_interrupt_for("agent-a")


class TestCheckpoint:
    """检查点测试"""

    def test_save_and_load_checkpoint(self):
        """测试保存和加载检查点"""
        agent_id = "test-checkpoint-agent"

        # 清理
        clear_checkpoint(agent_id)

        # 保存检查点
        state = {"step": 5, "data": "test"}
        checkpoint = save_checkpoint(agent_id, state)

        assert checkpoint.agent_id == agent_id
        assert checkpoint.state["step"] == 5

        # 加载检查点
        loaded = load_checkpoint(agent_id)
        assert loaded is not None
        assert loaded.state["step"] == 5
        assert loaded.state["data"] == "test"

        # 清理
        clear_checkpoint(agent_id)

    def test_has_checkpoint(self):
        """测试检查点存在性"""
        agent_id = "test-has-checkpoint"

        clear_checkpoint(agent_id)
        assert has_checkpoint(agent_id) is False

        save_checkpoint(agent_id, {"data": "test"})
        assert has_checkpoint(agent_id) is True

        clear_checkpoint(agent_id)
        assert has_checkpoint(agent_id) is False

    def test_load_nonexistent_checkpoint(self):
        """测试加载不存在的检查点"""
        checkpoint = load_checkpoint("nonexistent-agent-id")
        assert checkpoint is None

    def test_overwrite_checkpoint(self):
        """测试覆盖检查点"""
        agent_id = "test-overwrite-checkpoint"

        save_checkpoint(agent_id, {"version": 1})
        save_checkpoint(agent_id, {"version": 2})

        loaded = load_checkpoint(agent_id)
        assert loaded.state["version"] == 2

        clear_checkpoint(agent_id)
