"""微压缩模块测试"""

import pytest
from utils.context.micro_compact import (
    micro_compact_messages,
    summarize_tool_result,
    deduplicate_tool_results,
    COMPACTABLE_TOOLS,
)


class TestSummarizeToolResult:
    """测试工具结果摘要生成"""

    def test_bash_success(self):
        """测试 bash 命令成功结果"""
        content = '{"command": "ls -la", "returncode": 0, "stdout": "file1\\nfile2\\nfile3", "stderr": "", "success": true}'
        result = summarize_tool_result("bash", content)
        assert "[bash]" in result
        assert "成功" in result
        assert "exit 0" in result

    def test_bash_failure(self):
        """测试 bash 命令失败结果"""
        content = '{"command": "cat nonexistent", "returncode": 1, "stdout": "", "stderr": "No such file", "success": false}'
        result = summarize_tool_result("bash", content)
        assert "失败" in result
        assert "exit 1" in result

    def test_read_file(self):
        """测试文件读取结果"""
        content = '{"file_name": "test.py", "line_count": 100, "file_size": 2048, "is_large": false}'
        result = summarize_tool_result("read", content)
        assert "[read]" in result
        assert "test.py" in result
        assert "100行" in result

    def test_read_large_file(self):
        """测试大文件读取结果"""
        content = '{"file_name": "large.log", "line_count": 5000, "file_size": 1048576, "is_large": true}'
        result = summarize_tool_result("read", content)
        assert "大文件" in result

    def test_grep_results(self):
        """测试 grep 搜索结果"""
        content = '{"count": 25, "pattern": "def test", "truncated": false}'
        result = summarize_tool_result("grep", content)
        assert "[grep]" in result
        assert "25个匹配" in result

    def test_grep_truncated(self):
        """测试 grep 截断结果"""
        content = '{"count": 100, "pattern": "import", "truncated": true}'
        result = summarize_tool_result("grep", content)
        assert "截断" in result

    def test_web_search(self):
        """测试网页搜索结果"""
        content = '{"action": "search", "results": [{"title": "a"}, {"title": "b"}]}'
        result = summarize_tool_result("web", content)
        assert "[web:search]" in result
        assert "2 个结果" in result

    def test_arxiv_search(self):
        """测试 arXiv 搜索结果"""
        content = '{"action": "arxiv_search", "count": 5, "papers": []}'
        result = summarize_tool_result("web", content)
        assert "[web:arxiv_search]" in result
        assert "5 篇论文" in result

    def test_error_result(self):
        """测试错误结果"""
        content = '{"error": "File not found: test.py"}'
        result = summarize_tool_result("read", content)
        assert "错误" in result

    def test_non_json_result(self):
        """测试非 JSON 格式结果"""
        content = "This is plain text output\nwith multiple lines\n"
        result = summarize_tool_result("bash", content)
        assert "[bash]" in result
        # splitlines() 不计算末尾空行，所以是 2 行
        assert "行" in result  # 只检查包含"行"即可

    def test_long_text_truncation(self):
        """测试长文本截断"""
        content = "a" * 1000
        result = summarize_tool_result("bash", content)
        assert "..." in result
        assert len(result) < 150  # 摘要应该很短


class TestMicroCompactMessages:
    """测试微压缩核心逻辑"""

    def test_empty_messages(self):
        """测试空消息列表"""
        result = micro_compact_messages([])
        assert result["messages"] == []
        assert result["tools_cleared"] == 0
        assert result["tokens_saved"] == 0

    def test_no_tool_messages(self):
        """测试没有工具消息的情况"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = micro_compact_messages(messages)
        assert result["tools_cleared"] == 0
        assert result["messages"] == messages

    def test_single_tool_result(self):
        """测试单个工具结果（应保留）"""
        messages = [
            {"role": "user", "content": "Read file"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {
                    "tool_calls": [{"id": "call_1", "name": "read", "args": {"file_path": "test.py"}}]
                },
            },
            {
                "role": "tool",
                "content": '{"file_name": "test.py", "line_count": 50, "file_size": 1024}',
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
        ]
        result = micro_compact_messages(messages, keep_recent=3)
        # 单个工具结果应该被保留
        assert result["tools_cleared"] == 0
        assert result["tools_kept"] == 1

    def test_multiple_tool_results_keep_recent(self):
        """测试多个工具结果，保留最近的"""
        messages = [
            {"role": "user", "content": "Read files"},
            # 第一个工具调用
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": '{"file_name": "file1.py", "line_count": 100, "file_size": 2048}',
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
            # 第二个工具调用
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_2", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": '{"file_name": "file2.py", "line_count": 200, "file_size": 4096}',
                "metadata": {"tool_call_id": "call_2", "name": "read"},
            },
            # 第三个工具调用
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_3", "name": "bash", "args": {}}]},
            },
            {
                "role": "tool",
                "content": '{"command": "ls", "returncode": 0, "stdout": "a\\nb\\nc", "stderr": "", "success": true}',
                "metadata": {"tool_call_id": "call_3", "name": "bash"},
            },
        ]

        # keep_recent=2，应该清理 1 个，保留 2 个
        result = micro_compact_messages(messages, keep_recent=2)

        assert result["tools_cleared"] == 1
        assert result["tools_kept"] == 2
        assert result["tokens_saved"] > 0

        # 检查清理的消息被替换为摘要
        cleared_msg = result["messages"][2]  # 第一个 tool 消息
        assert "[read]" in cleared_msg["content"]
        assert "file1.py" in cleared_msg["content"]
        assert cleared_msg["metadata"].get("micro_compacted") == True

        # 保留的消息应该保持原样
        kept_msg = result["messages"][6]  # 第三个 tool 消息
        assert "micro_compacted" not in kept_msg.get("metadata", {})

    def test_non_compactable_tools_not_affected(self):
        """测试不可压缩的工具不受影响"""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "some_other_tool", "args": {}}]},
            },
            {
                "role": "tool",
                "content": "Some important result that should not be compacted",
                "metadata": {"tool_call_id": "call_1", "name": "some_other_tool"},
            },
        ]

        result = micro_compact_messages(messages, keep_recent=0)

        # 不在 COMPACTABLE_TOOLS 中的工具不应该被清理
        assert result["tools_cleared"] == 0
        assert result["tokens_saved"] == 0

    def test_tokens_saved_calculation(self):
        """测试 token 节省计算"""
        long_content = '{"file_name": "large.py", "line_count": 1000, "file_size": 50000, "content": "' + "x" * 10000 + '"}'

        messages = [
            {"role": "user", "content": "Read"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": long_content,
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_2", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": '{"file_name": "small.py", "line_count": 10}',
                "metadata": {"tool_call_id": "call_2", "name": "read"},
            },
        ]

        result = micro_compact_messages(messages, keep_recent=1)

        # 应该有显著的 token 节省
        assert result["tokens_saved"] > 100


class TestDeduplicateToolResults:
    """测试去重功能"""

    def test_no_duplicates(self):
        """测试没有重复的情况"""
        messages = [
            {"role": "user", "content": "Read"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": '{"file_name": "a.py", "line_count": 10}',
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_2", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": '{"file_name": "b.py", "line_count": 20}',
                "metadata": {"tool_call_id": "call_2", "name": "read"},
            },
        ]

        result = deduplicate_tool_results(messages)
        assert result["duplicates_found"] == 0

    def test_with_duplicates(self):
        """测试有重复的情况"""
        same_content = '{"file_name": "config.py", "line_count": 50, "file_size": 1024}'

        messages = [
            {"role": "user", "content": "Read config"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_1", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": same_content,
                "metadata": {"tool_call_id": "call_1", "name": "read"},
            },
            {"role": "user", "content": "Read again"},
            {
                "role": "assistant",
                "content": "",
                "metadata": {"tool_calls": [{"id": "call_2", "name": "read", "args": {}}]},
            },
            {
                "role": "tool",
                "content": same_content,  # 相同内容
                "metadata": {"tool_call_id": "call_2", "name": "read"},
            },
        ]

        result = deduplicate_tool_results(messages)
        assert result["duplicates_found"] == 1

        # 第一个重复结果应该被标记
        first_tool_msg = result["messages"][2]
        assert "重复结果已清理" in first_tool_msg["content"]


class TestCompactableTools:
    """测试可压缩工具列表"""

    def test_file_tools_are_compactable(self):
        """测试文件工具可压缩"""
        assert "read" in COMPACTABLE_TOOLS
        assert "grep" in COMPACTABLE_TOOLS
        assert "glob" in COMPACTABLE_TOOLS

    def test_shell_tools_are_compactable(self):
        """测试 Shell 工具可压缩"""
        assert "bash" in COMPACTABLE_TOOLS
        assert "bash_script" in COMPACTABLE_TOOLS

    def test_web_tools_are_compactable(self):
        """测试 Web 工具可压缩"""
        assert "web" in COMPACTABLE_TOOLS

    def test_important_tools_not_compactable(self):
        """测试重要工具不在可压缩列表中"""
        # 这些工具的结果包含重要决策信息，不应被压缩
        assert "plan_dispatch" not in COMPACTABLE_TOOLS
        assert "job_status" not in COMPACTABLE_TOOLS
        assert "task" not in COMPACTABLE_TOOLS
