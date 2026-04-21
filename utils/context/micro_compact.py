"""微压缩模块 - 清理旧的工具结果以节省 token

微压缩是一种轻量级的上下文压缩方式，不调用 LLM，直接清理旧的工具结果。

策略：
1. 保护尾部（最近的消息）
2. 清理超出保护范围的旧工具结果
3. 用信息性摘要替换完整内容

参考：
- Hermes Agent 的工具输出修剪
- Claude Code 的微压缩机制
"""

import json
import re
from typing import Any

from utils.context.token_counter import count_tokens


# ============ 配置 ============

# 可压缩的工具类型（结果往往很长，价值随时间降低）
COMPACTABLE_TOOLS = {
    # 文件操作
    "read",
    "grep",
    "grep_count",
    "glob",
    "glob_list",
    # Shell 命令
    "bash",
    "bash_script",
    # Web 操作
    "web",  # 统一入口，内部有 search/fetch/scrape 等
    "tavily_search",
    "tavily_extract",
    "firecrawl_scrape",
    "firecrawl_crawl",
}

# 清理占位符
CLEARED_PLACEHOLDER = "[旧工具结果已清理]"


# ============ 工具结果摘要生成 ============

def summarize_tool_result(tool_name: str, content: str) -> str:
    """生成工具结果的一行摘要

    Args:
        tool_name: 工具名称
        content: 工具结果内容

    Returns:
        一行摘要文本
    """
    # 尝试解析 JSON 格式的结果
    try:
        if content.strip().startswith("{"):
            data = json.loads(content)
            return _summarize_json_result(tool_name, data)
    except (json.JSONDecodeError, AttributeError):
        pass

    # 非 JSON 格式，生成通用摘要
    return _summarize_text_result(tool_name, content)


def _summarize_json_result(tool_name: str, data: dict) -> str:
    """根据工具类型生成 JSON 结果的摘要"""

    # bash/shell 命令
    if tool_name in ("bash", "bash_script"):
        cmd = data.get("command", "?")[:50]
        code = data.get("returncode", "?")
        stdout_lines = len(data.get("stdout", "").splitlines())
        stderr_lines = len(data.get("stderr", "").splitlines())
        status = "成功" if data.get("success") else "失败"
        return f"[{tool_name}] {status}: `{cmd}` → exit {code}, {stdout_lines}行输出, {stderr_lines}行错误"

    # read 文件读取
    if tool_name == "read":
        if data.get("error"):
            return f"[read] 错误: {data['error'][:50]}"
        file_name = data.get("file_name", "?")
        line_count = data.get("line_count", 0)
        file_size = data.get("file_size", 0)
        is_large = data.get("is_large", False)
        large_tag = " (大文件)" if is_large else ""
        return f"[read] {file_name}: {line_count}行, {file_size}字节{large_tag}"

    # grep 搜索
    if tool_name in ("grep", "grep_count"):
        if data.get("error"):
            return f"[{tool_name}] 错误: {data['error'][:50]}"
        count = data.get("count", 0)
        pattern = data.get("pattern", "?")[:30]
        truncated = " (截断)" if data.get("truncated") else ""
        return f"[{tool_name}] 搜索 '{pattern}': {count}个匹配{truncated}"

    # glob 文件列表
    if tool_name in ("glob", "glob_list"):
        if data.get("error"):
            return f"[{tool_name}] 错误: {data['error'][:50]}"
        count = data.get("count", 0)
        pattern = data.get("pattern", "?")[:30]
        return f"[{tool_name}] 模式 '{pattern}': {count}个文件"

    # web 统一工具
    if tool_name == "web":
        action = data.get("action", "?")
        if data.get("error"):
            return f"[web:{action}] 错误: {data['error'][:50]}"
        if action == "search":
            results = len(data.get("results", []))
            return f"[web:search] 找到 {results} 个结果"
        if action in ("fetch", "scrape"):
            url = data.get("url", "?")[:40]
            return f"[web:{action}] {url}"
        if action == "arxiv_search":
            count = data.get("count", 0)
            return f"[web:arxiv_search] 找到 {count} 篇论文"
        return f"[web:{action}] 完成"

    # 通用摘要
    keys = list(data.keys())[:5]
    return f"[{tool_name}] 结果包含: {', '.join(keys)}"


def _summarize_text_result(tool_name: str, content: str) -> str:
    """生成文本结果的通用摘要"""
    lines = content.splitlines()
    line_count = len(lines)
    char_count = len(content)

    # 截取前 50 个字符作为预览
    preview = content[:50].replace("\n", " ")
    if len(content) > 50:
        preview += "..."

    return f"[{tool_name}] {line_count}行, {char_count}字符: {preview}"


# ============ 微压缩核心逻辑 ============

def micro_compact_messages(
    messages: list[dict],
    keep_recent: int = 10,
) -> dict[str, Any]:
    """微压缩消息列表 - 清理旧的工具结果

    Args:
        messages: 消息列表（dict 格式）
        keep_recent: 保留最近 N 个工具结果（默认 5）

    Returns:
        {
            'messages': 压缩后的消息列表,
            'tools_cleared': 清理的工具数,
            'tools_kept': 保留的工具数,
            'tokens_saved': 节省的 token 数,
        }
    """
    if not messages:
        return {
            "messages": messages,
            "tools_cleared": 0,
            "tools_kept": 0,
            "tokens_saved": 0,
        }

    # 1. 收集所有可压缩的工具的 tool_call_id
    tool_call_ids = _collect_compactable_tool_ids(messages)

    if not tool_call_ids:
        return {
            "messages": messages,
            "tools_cleared": 0,
            "tools_kept": 0,
            "tokens_saved": 0,
        }

    # 2. 确定保留和清理的 ID
    all_ids = list(tool_call_ids.keys())
    keep_count = min(keep_recent, len(all_ids))

    # 保留最近的 N 个
    keep_ids = set(all_ids[-keep_count:]) if keep_count > 0 else set()
    clear_ids = set(all_ids[:-keep_count]) if keep_count > 0 and keep_count < len(all_ids) else set()

    if not clear_ids:
        return {
            "messages": messages,
            "tools_cleared": 0,
            "tools_kept": len(keep_ids),
            "tokens_saved": 0,
        }

    # 3. 执行压缩
    tokens_before = 0
    tokens_after = 0
    tools_cleared = 0
    tools_kept = 0

    result_messages = []

    for msg in messages:
        role = msg.get("role")

        if role == "tool":
            metadata = msg.get("metadata") or {}
            tool_call_id = metadata.get("tool_call_id")
            tool_name = metadata.get("name", "unknown")
            content = msg.get("content", "")

            if tool_call_id in clear_ids:
                # 计算节省的 token
                tokens_before += count_tokens(content)

                # 生成摘要
                summary = summarize_tool_result(tool_name, content)
                tokens_after += count_tokens(summary)

                # 替换内容
                new_msg = {
                    **msg,
                    "content": summary,
                    "metadata": {
                        **metadata,
                        "micro_compacted": True,
                    },
                }
                result_messages.append(new_msg)
                tools_cleared += 1

            elif tool_call_id in keep_ids:
                # 保留完整内容
                result_messages.append(msg)
                tools_kept += 1
            else:
                # 不在可压缩列表中的工具，保留原样
                result_messages.append(msg)
        else:
            # 非工具消息，保留原样
            result_messages.append(msg)

    return {
        "messages": result_messages,
        "tools_cleared": tools_cleared,
        "tools_kept": tools_kept,
        "tokens_saved": tokens_before - tokens_after,
    }


def _collect_compactable_tool_ids(messages: list[dict]) -> dict[str, str]:
    """收集所有可压缩工具的 tool_call_id

    Args:
        messages: 消息列表

    Returns:
        {tool_call_id: tool_name} 字典，按遇到顺序排列
    """
    tool_ids = {}

    for msg in messages:
        role = msg.get("role")

        # 从 assistant 消息中提取 tool_calls
        if role == "assistant":
            metadata = msg.get("metadata") or {}
            tool_calls = metadata.get("tool_calls", [])

            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_id = tc.get("id", "")

                if tool_name in COMPACTABLE_TOOLS and tool_id:
                    tool_ids[tool_id] = tool_name

    return tool_ids


# ============ 高级功能：去重 ============

def deduplicate_tool_results(messages: list[dict]) -> dict[str, Any]:
    """去重相同的工具调用结果

    当多次调用相同工具获取相同结果时，只保留最新的一份完整内容，
    旧的用摘要替换。

    Args:
        messages: 消息列表

    Returns:
        {
            'messages': 处理后的消息列表,
            'duplicates_found': 发现的重复数,
        }
    """
    # 记录每个工具调用的签名 -> (所有 ID 列表，按顺序)
    signatures: dict[str, list[str]] = {}
    duplicates_found = 0

    # 第一遍：收集签名和 ID（按顺序）
    for msg in messages:
        if msg.get("role") != "tool":
            continue

        metadata = msg.get("metadata") or {}
        tool_name = metadata.get("name", "unknown")
        tool_call_id = metadata.get("tool_call_id", "")
        content = msg.get("content", "")

        if not tool_call_id:
            continue

        # 生成签名（工具名 + 内容哈希）
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        signature = f"{tool_name}:{content_hash}"

        if signature in signatures:
            signatures[signature].append(tool_call_id)
            duplicates_found += 1
        else:
            signatures[signature] = [tool_call_id]

    if duplicates_found == 0:
        return {
            "messages": messages,
            "duplicates_found": 0,
        }

    # 第二遍：确定要清理的 ID（保留最后一个，清理前面的）
    duplicate_ids = set()
    for ids in signatures.values():
        if len(ids) > 1:
            # 保留最后一个（最新的），清理前面的
            duplicate_ids.update(ids[:-1])

    result_messages = []
    for msg in messages:
        if msg.get("role") == "tool":
            metadata = msg.get("metadata") or {}
            tool_call_id = metadata.get("tool_call_id")

            if tool_call_id in duplicate_ids:
                tool_name = metadata.get("name", "unknown")
                content = msg.get("content", "")
                summary = summarize_tool_result(tool_name, content)

                new_msg = {
                    **msg,
                    "content": f"[重复结果已清理] {summary}",
                    "metadata": {
                        **metadata,
                        "deduplicated": True,
                    },
                }
                result_messages.append(new_msg)
                continue

        result_messages.append(msg)

    return {
        "messages": result_messages,
        "duplicates_found": duplicates_found,
    }
