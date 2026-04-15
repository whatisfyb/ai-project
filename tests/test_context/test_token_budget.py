"""Token 预算分割测试"""

import pytest
from utils.context.compact import (
    find_split_point,
    find_split_point_by_tokens,
    _adjust_split_for_tool_chain,
)
from utils.context.token_counter import count_messages_tokens


class TestFindSplitPointByTokens:
    """测试基于 Token 预算的分割"""

    def test_empty_messages(self):
        """测试空消息列表"""
        to_summarize, to_keep = find_split_point_by_tokens([], 1000)
        assert to_summarize == []
        assert to_keep == []

    def test_single_message_under_budget(self):
        """测试单条消息在预算内"""
        messages = [{"role": "user", "content": "Hello"}]
        to_summarize, to_keep = find_split_point_by_tokens(messages, 1000)
        assert to_summarize == []
        assert to_keep == messages

    def test_messages_under_budget(self):
        """测试多条消息在预算内"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        to_summarize, to_keep = find_split_point_by_tokens(messages, 10000)
        assert to_summarize == []
        assert to_keep == messages

    def test_messages_over_budget(self):
        """测试消息超过预算"""
        # 创建一些较长的消息
        messages = [
            {"role": "user", "content": "A" * 2000},
            {"role": "assistant", "content": "B" * 2000},
            {"role": "user", "content": "C" * 2000},
            {"role": "assistant", "content": "D" * 2000},
        ]

        # 预算只够保留最后一条消息
        # 2000 字符约 500 tokens
        token_budget = 600  # 只够保留约一条消息

        to_summarize, to_keep = find_split_point_by_tokens(messages, token_budget)

        # 应该有消息被压缩
        assert len(to_summarize) > 0
        assert len(to_keep) > 0
        assert len(to_summarize) + len(to_keep) == len(messages)

    def test_exact_budget_boundary(self):
        """测试恰好达到预算边界"""
        messages = [
            {"role": "user", "content": "Short"},
            {"role": "assistant", "content": "Also short"},
        ]

        # 计算总 token
        total = count_messages_tokens(messages)

        # 预算刚好等于总 token
        to_summarize, to_keep = find_split_point_by_tokens(messages, total)
        assert to_summarize == []
        assert to_keep == messages


class TestAdjustSplitForToolChain:
    """测试工具调用链的分割调整"""

    def test_split_at_user_message(self):
        """测试分割点是用户消息（无需调整）"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine"},
        ]
        # 分割点在第 3 条消息（user）
        adjusted = _adjust_split_for_tool_chain(messages, 2)
        assert adjusted == 2

    def test_split_at_tool_message(self):
        """测试分割点是工具消息（需要调整到对应的 AIMessage）"""
        messages = [
            {"role": "user", "content": "Read file"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "read"}]},
            },
            {
                "role": "tool",
                "content": "file content",
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
            {"role": "user", "content": "Thanks"},
        ]
        # 分割点在第 3 条消息（tool），应该调整到第 2 条（assistant）
        adjusted = _adjust_split_for_tool_chain(messages, 2)
        assert adjusted == 1  # 调整到 AIMessage

    def test_split_at_assistant_with_tools(self):
        """测试分割点是带工具调用的 AIMessage"""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "bash"}]},
            },
            {
                "role": "tool",
                "content": "output",
                "metadata": {"tool_call_id": "call_1", "name": "bash"},
            },
        ]
        # 分割点在第 2 条消息（assistant with tools）
        # 应该保持不变，因为后面的 tool 会被包含在 to_keep 中
        adjusted = _adjust_split_for_tool_chain(messages, 1)
        assert adjusted == 1

    def test_split_before_orphan_tool(self):
        """测试分割点前有孤立的 ToolMessage

        当分割点前一条消息是 tool 时，需要确保 tool 和对应的 assistant 一起保留。
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "read"}]},
            },
            {
                "role": "tool",
                "content": "content",
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
            {"role": "user", "content": "Next request"},
        ]
        # 分割点在第 4 条消息（index 3）
        # 前一条消息（index 2）是 tool
        # 需要找到对应的 assistant（index 1）
        adjusted = _adjust_split_for_tool_chain(messages, 3)
        # 应该调整到 assistant 的位置，这样 tool 和 assistant 都在 to_keep 中
        assert adjusted == 1  # 调整到 AIMessage


class TestFindSplitPointComparison:
    """对比两种分割模式"""

    def test_token_budget_vs_message_count_short_messages(self):
        """测试短消息：Token 预算会保留更多消息"""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "OK"},
            {"role": "assistant", "content": "Sure"},
            {"role": "user", "content": "Thanks"},
        ]

        # 消息数模式：保留最近 2 条
        _, to_keep_count = find_split_point(messages, keep_recent=2)
        assert len(to_keep_count) == 2

        # Token 预算模式：预算足够保留所有消息
        _, to_keep_tokens = find_split_point_by_tokens(messages, 1000)
        assert len(to_keep_tokens) == 5  # 所有消息都在预算内

    def test_token_budget_vs_message_count_long_messages(self):
        """测试长消息：Token 预算会保留更少消息"""
        messages = [
            {"role": "user", "content": "A" * 5000},
            {"role": "assistant", "content": "B" * 5000},
            {"role": "user", "content": "C" * 5000},
            {"role": "assistant", "content": "D" * 5000},
            {"role": "user", "content": "E" * 5000},
        ]

        # 消息数模式：保留最近 2 条
        _, to_keep_count = find_split_point(messages, keep_recent=2)
        assert len(to_keep_count) == 2

        # Token 预算模式：预算只够约 1 条消息
        # 5000 字符约 1250 tokens
        token_budget = 1500
        _, to_keep_tokens = find_split_point_by_tokens(messages, token_budget)
        # 应该只保留最后 1 条
        assert len(to_keep_tokens) >= 1
        assert len(to_keep_tokens) <= 2

    def test_token_budget_respects_tool_chain(self):
        """测试 Token 预算模式也会尊重工具调用链"""
        messages = [
            {"role": "user", "content": "Read"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "read"}]},
            },
            {
                "role": "tool",
                "content": "X" * 10000,  # 很长的内容
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
            {"role": "user", "content": "Thanks"},
        ]

        # 使用很小的预算
        token_budget = 50
        _, to_keep = find_split_point_by_tokens(messages, token_budget)

        # 即使预算很小，也不应该分割工具调用链
        # to_keep 应该从一条完整的消息开始
        if to_keep:
            first_kept = to_keep[0]
            # 第一条保留的消息不应该是 tool（需要有对应的 assistant）
            if first_kept["role"] == "tool":
                # 如果是 tool，前面必须有对应的 assistant
                assert any(
                    m["role"] == "assistant" and
                    "tool_calls" in m.get("metadata", {})
                    for m in to_keep
                )
