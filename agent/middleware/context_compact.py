"""上下文压缩相关节点"""
import asyncio

from agent.core.models import MainAgentState
from utils.token_counter import count_messages_tokens, count_tokens
from utils.config import get_settings_instance
from utils.compact import compact_messages


# 模块级 checkpointer 注册
_checkpointer = None


def set_checkpointer(checkpointer) -> None:
    """注册 checkpointer（由 MainAgent 初始化时调用）"""
    global _checkpointer
    _checkpointer = checkpointer


def should_auto_compact(total_tokens: int, context_window: int) -> bool:
    """检查是否需要自动压缩

    Args:
        total_tokens: 当前 token 总数
        context_window: 模型上下文窗口大小

    Returns:
        是否需要压缩
    """
    settings = get_settings_instance()
    compact_settings = settings.compact

    if not compact_settings.auto_enabled:
        return False

    # 计算有效上下文窗口
    buffer = min(compact_settings.buffer_tokens, context_window // 4)
    output_reserve = min(compact_settings.output_reserve, context_window // 4)

    effective_window = context_window - output_reserve - buffer

    if effective_window <= 0:
        effective_window = context_window // 2

    # 计算阈值
    threshold = effective_window * compact_settings.threshold_pct

    return total_tokens >= threshold


async def compact_session(thread_id: str, context_window: int) -> dict:
    """压缩会话上下文

    Args:
        thread_id: 会话 ID
        context_window: 模型上下文窗口大小

    Returns:
        压缩结果，包含摘要和 token 统计
    """
    from store.session import SessionStore
    from utils.llm import get_llm_model

    session_store = SessionStore()
    messages = session_store.get_messages(thread_id)

    # 从配置读取 keep_recent
    settings = get_settings_instance()
    keep_recent = settings.compact.keep_recent

    # 确保 keep_recent 不超过消息数的一半，且至少保留 2 条
    keep_recent = min(keep_recent, max(2, len(messages) // 2))

    if len(messages) <= keep_recent:
        return {
            "success": False,
            "message": f"消息数量不足 (当前 {len(messages)} 条，需要 > {keep_recent} 条)",
            "tokens_before": count_messages_tokens(messages),
            "tokens_after": count_messages_tokens(messages),
        }

    # 执行压缩
    llm = get_llm_model()
    result = await compact_messages(messages, keep_recent, llm)

    if result["messages_removed"] == 0:
        return {
            "success": False,
            "message": "消息数量不足，无需压缩",
            "tokens_before": result["tokens_before"],
            "tokens_after": result["tokens_after"],
        }

    # 更新 session_store：清空旧消息，添加压缩后的消息
    session_store.clear_messages(thread_id)
    compacted = result["compact_messages"]
    for msg in compacted:
        role = msg["role"]
        content = msg.get("content", "")
        metadata = msg.get("metadata")
        session_store.add_message(thread_id, role, content, metadata)

    # 清除 LangGraph checkpointer 的状态（下次 chat 会从 session_store 恢复）
    if _checkpointer and hasattr(_checkpointer, "delete_thread"):
        _checkpointer.delete_thread(thread_id)

    # 刷新 token 计量：重新计算压缩后的 token 数并写回 session_store
    compacted_msgs = session_store.get_messages(thread_id)
    messages_tokens = count_messages_tokens(compacted_msgs)
    total_tokens = messages_tokens
    session_store.update_session_tokens(thread_id, total_tokens)

    return {
        "success": True,
        "summary": result["summary"],
        "tokens_before": result["tokens_before"],
        "tokens_after": result["tokens_after"],
        "messages_removed": result["messages_removed"],
        "messages_kept": len(compacted),
        "compacted_messages": compacted,
    }


async def check_token_node(state: MainAgentState, context_window: int) -> dict:
    """Token 检查节点 - 检查 token 是否超过阈值，超过则压缩

    放在 finish_section 的最后一个节点，token_count_node 之后执行。

    Args:
        state: MainAgentState 状态（含 thread_id, session_id）
        context_window: 模型上下文窗口大小

    Returns:
        压缩结果或空字典
    """
    from store.session import SessionStore

    thread_id = state.get("thread_id")
    session_id = state.get("session_id")
    if not thread_id:
        return {}

    session_store = SessionStore()
    session = session_store.get_session(session_id or thread_id)
    if not session:
        return {}

    total_tokens = session.get("total_tokens", 0)

    if not should_auto_compact(total_tokens, context_window):
        return {}

    # 执行压缩
    result = await compact_session(session_id or thread_id, context_window)

    if result.get("success"):
        # 将 dict 格式转换为 LangChain Message 对象
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        compacted_messages = []
        for msg in result["compacted_messages"]:
            role = msg.get("role")
            content = msg.get("content", "")
            metadata = msg.get("metadata") or {}

            if role == "user":
                compacted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                tool_calls = metadata.get("tool_calls")
                if tool_calls:
                    compacted_messages.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    compacted_messages.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = metadata.get("tool_call_id")
                tool_name = metadata.get("name")
                if tool_call_id:
                    compacted_messages.append(ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    ))

        return {"messages": compacted_messages}

    return {}
