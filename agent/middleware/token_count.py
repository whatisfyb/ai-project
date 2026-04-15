"""Token 计量相关节点"""
from agent.core.models import MainAgentState
from utils.context.token_counter import count_messages_tokens, count_tokens


def recalculate_session_tokens(thread_id: str) -> int:
    """重新计算会话的 token 数（启动时调用）

    Args:
        thread_id: 会话 ID

    Returns:
        计算后的 token 总数
    """
    from store.session import SessionStore
    from agent.main.prompts import MAIN_AGENT_PROMPT

    session_store = SessionStore()
    messages = session_store.get_messages(thread_id)

    # 计算 system prompt token
    system_tokens = count_tokens(MAIN_AGENT_PROMPT)

    # 计算 messages token
    messages_tokens = count_messages_tokens(messages)

    total_tokens = system_tokens + messages_tokens

    # 更新 session_store
    session_store.update_session_tokens(thread_id, total_tokens)

    return total_tokens


async def token_count_node(state: MainAgentState) -> dict:
    """Token 计量节点 - 统计本次 system prompt + messages 的 token 数，存入 session_store

    放在 finish_section 的最后一个节点，在 graph 执行完后统计。

    Args:
        state: MainAgentState 状态（含 thread_id, session_id）

    Returns:
        空字典（不更新状态，仅副作用）
    """
    from store.session import SessionStore

    thread_id = state.get("thread_id")
    session_id = state.get("session_id")
    if not thread_id:
        return {}

    session_store = SessionStore()

    messages = state.get("messages", [])
    memory_context = state.get("memory_context")

    # 构建 system prompt 内容
    from agent.main.prompts import MAIN_AGENT_PROMPT
    system_content = MAIN_AGENT_PROMPT
    if memory_context:
        system_content = f"{memory_context}\n\n---\n\n{MAIN_AGENT_PROMPT}"

    # 计算 system prompt token
    system_tokens = count_tokens(system_content)

    # 计算 messages token
    messages_tokens = count_messages_tokens(messages)

    total_tokens = system_tokens + messages_tokens

    # 存入 session_store
    session_store.update_session_tokens(session_id or thread_id, total_tokens)

    return {}
