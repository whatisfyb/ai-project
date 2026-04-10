"""上下文压缩模块

使用 LLM 生成对话摘要，压缩早期消息以节省上下文空间。
"""

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from utils.llm import get_llm_model
from utils.token_counter import count_messages_tokens


# 压缩提示词
COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents
2. Key Technical Concepts: List important technical concepts, technologies, and frameworks discussed
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created
4. Errors and Fixes: List all errors encountered and how they were fixed
5. Pending Tasks: Outline any pending tasks
6. Current Work: Describe what was being worked on most recently

Please provide your summary in a clear, structured format. Focus on technical details that would be essential for continuing the work without losing context.

Respond with TEXT ONLY. Do NOT call any tools."""

# 压缩后的用户提示
COMPACT_USER_PROMPT = """This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

{summary}

Recent messages are preserved verbatim. Continue from where the conversation left off."""


def group_messages_by_round(messages: list[dict]) -> list[list[dict]]:
    """按 API 轮次分组消息

    每轮 = user → assistant → tool_calls → tool_results
    确保工具调用链完整

    Args:
        messages: 消息列表（dict 格式）

    Returns:
        分组后的消息列表
    """
    groups = []
    current = []

    for msg in messages:
        role = msg['role']

        # 新的一轮从 user 或 assistant 开始
        if role == 'user' and current:
            groups.append(current)
            current = [msg]
        elif role == 'assistant' and current:
            # 检查是否是新的 assistant 消息（不同的 tool_calls）
            metadata = msg.get('metadata') or {}
            has_tools = 'tool_calls' in metadata
            # 如果当前组以 tool 结尾，且这是新的 assistant，开始新组
            if current[-1]['role'] == 'tool' and not has_tools:
                groups.append(current)
                current = [msg]
            else:
                current.append(msg)
        else:
            current.append(msg)

    if current:
        groups.append(current)

    return groups


def find_split_point(
    messages: list[dict],
    keep_recent: int = 10,
) -> tuple[list[dict], list[dict]]:
    """找到分割点，保留最近的消息

    确保工具调用链完整：
    - ToolMessage 必须跟在有 tool_calls 的 AIMessage 后面
    - 不能在工具调用链中间分割

    Args:
        messages: 消息列表
        keep_recent: 保留最近 N 条消息

    Returns:
        (to_summarize, to_keep) 分割后的消息
    """
    if len(messages) <= keep_recent:
        return [], messages

    # 从后往前找到合适的分割点
    split_idx = len(messages) - keep_recent

    # 检查分割点是否在工具调用链中间
    # 如果 split_idx 指向 ToolMessage，需要往前找到 AIMessage
    while split_idx < len(messages):
        msg = messages[split_idx]
        if msg['role'] == 'tool':
            # 找到对应的 AIMessage（有 tool_calls）
            for i in range(split_idx - 1, -1, -1):
                m = messages[i]
                if m['role'] == 'assistant':
                    metadata = m.get('metadata') or {}
                    if 'tool_calls' in metadata:
                        split_idx = i
                        break
            break
        elif msg['role'] == 'assistant':
            # 检查是否有 tool_calls（后面可能有 ToolMessage）
            metadata = msg.get('metadata') or {}
            if 'tool_calls' in metadata:
                # 这条消息后面有 ToolMessage，不能在这里分割
                split_idx += 1
            else:
                break
        else:
            break

    return messages[:split_idx], messages[split_idx:]


async def generate_summary(
    messages: list[dict],
    llm=None,
) -> str:
    """生成对话摘要

    Args:
        messages: 要摘要的消息列表
        llm: LLM 实例（可选）

    Returns:
        摘要文本
    """
    if llm is None:
        llm = get_llm_model()

    # 构建消息
    langchain_messages = []

    # 添加系统提示
    langchain_messages.append(SystemMessage(content=COMPACT_PROMPT))

    # 转换消息格式
    for msg in messages:
        role = msg['role']
        content = msg.get('content') or ''

        if role == 'user':
            langchain_messages.append(HumanMessage(content=content))
        elif role == 'assistant':
            # 只保留文本内容，忽略 tool_calls（摘要不需要）
            langchain_messages.append(AIMessage(content=content))
        elif role == 'tool':
            # 工具返回结果
            tool_name = (msg.get('metadata') or {}).get('name', 'tool')
            langchain_messages.append(HumanMessage(content=f"[Tool {tool_name}]: {content}"))

    # 调用 LLM
    response = await llm.ainvoke(langchain_messages)
    return response.content


def create_summary_message(summary: str) -> dict:
    """创建摘要消息

    Args:
        summary: 摘要文本

    Returns:
        摘要消息（dict 格式）
    """
    return {
        'role': 'user',
        'content': COMPACT_USER_PROMPT.format(summary=summary),
        'metadata': {'is_compact_summary': True},
    }


async def compact_messages(
    messages: list[dict],
    keep_recent: int = 10,
    llm=None,
) -> dict[str, Any]:
    """压缩消息列表

    Args:
        messages: 消息列表
        keep_recent: 保留最近 N 条消息
        llm: LLM 实例

    Returns:
        {
            'compact_messages': 压缩后的消息列表,
            'summary': 摘要文本,
            'tokens_before': 压缩前 token 数,
            'tokens_after': 压缩后 token 数,
            'messages_removed': 移除的消息数,
        }
    """
    if len(messages) <= keep_recent:
        return {
            'compact_messages': messages,
            'summary': None,
            'tokens_before': count_messages_tokens(messages),
            'tokens_after': count_messages_tokens(messages),
            'messages_removed': 0,
        }

    # 分割消息
    to_summarize, to_keep = find_split_point(messages, keep_recent)

    if not to_summarize:
        return {
            'compact_messages': to_keep,
            'summary': None,
            'tokens_before': count_messages_tokens(messages),
            'tokens_after': count_messages_tokens(to_keep),
            'messages_removed': 0,
        }

    # 计算压缩前 token 数
    tokens_before = count_messages_tokens(messages)

    # 生成摘要
    summary = await generate_summary(to_summarize, llm)

    # 创建摘要消息
    summary_msg = create_summary_message(summary)

    # 组合压缩后的消息
    compacted = [summary_msg] + to_keep

    # 计算压缩后 token 数
    tokens_after = count_messages_tokens(compacted)

    return {
        'compact_messages': compacted,
        'summary': summary,
        'tokens_before': tokens_before,
        'tokens_after': tokens_after,
        'messages_removed': len(to_summarize),
    }
