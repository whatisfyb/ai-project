"""上下文压缩相关节点"""
import asyncio
import threading
from datetime import datetime
from typing import Optional

from agent.core.models import MainAgentState
from utils.context.token_counter import count_messages_tokens, count_tokens
from utils.core.config import get_settings_instance
from utils.context.compact import compact_messages


# 模块级 checkpointer 注册
_checkpointer = None


def set_checkpointer(checkpointer) -> None:
    """注册 checkpointer（由 MainAgent 初始化时调用）"""
    global _checkpointer
    _checkpointer = checkpointer


# ============ 断路器状态 ============

class CircuitBreakerState:
    """断路器状态（线程安全）"""

    def __init__(
        self,
        min_savings_pct: float = 0.10,
        consecutive_threshold: int = 2,
        reset_after_seconds: float = 300.0,  # 5 分钟后重置
    ):
        self.min_savings_pct = min_savings_pct
        self.consecutive_threshold = consecutive_threshold
        self.reset_after_seconds = reset_after_seconds

        self._recent_savings: list[tuple[float, datetime]] = []
        self._lock = threading.Lock()
        self._triggered = False
        self._triggered_at: Optional[datetime] = None

    def record_savings(self, savings_pct: float) -> None:
        """记录压缩节省比例"""
        with self._lock:
            self._recent_savings.append((savings_pct, datetime.now()))
            # 只保留最近的 10 次记录
            if len(self._recent_savings) > 10:
                self._recent_savings = self._recent_savings[-10:]

            # 检查是否触发断路器
            if len(self._recent_savings) >= self.consecutive_threshold:
                recent = [s[0] for s in self._recent_savings[-self.consecutive_threshold:]]
                if all(s < self.min_savings_pct for s in recent):
                    self._triggered = True
                    self._triggered_at = datetime.now()

    def should_skip_compact(self) -> bool:
        """是否应该跳过压缩（断路器是否触发）"""
        with self._lock:
            if not self._triggered:
                return False

            # 检查是否应该重置
            if self._triggered_at:
                elapsed = (datetime.now() - self._triggered_at).total_seconds()
                if elapsed >= self.reset_after_seconds:
                    self._reset()
                    return False

            return True

    def _reset(self) -> None:
        """重置断路器"""
        self._triggered = False
        self._triggered_at = None
        self._recent_savings = []

    def reset(self) -> None:
        """手动重置断路器（线程安全）"""
        with self._lock:
            self._reset()

    def get_status(self) -> dict:
        """获取断路器状态"""
        with self._lock:
            return {
                "triggered": self._triggered,
                "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
                "recent_savings": [s[0] for s in self._recent_savings],
                "consecutive_count": len([s for s in self._recent_savings if s[0] < self.min_savings_pct]),
            }


# 全局断路器实例
_circuit_breaker: Optional[CircuitBreakerState] = None


def get_circuit_breaker() -> CircuitBreakerState:
    """获取断路器实例"""
    global _circuit_breaker
    if _circuit_breaker is None:
        settings = get_settings_instance()
        compact_settings = settings.compact
        _circuit_breaker = CircuitBreakerState(
            min_savings_pct=compact_settings.circuit_breaker_min_savings,
            consecutive_threshold=compact_settings.circuit_breaker_consecutive,
            reset_after_seconds=compact_settings.circuit_breaker_reset_seconds,
        )
    return _circuit_breaker


def reset_circuit_breaker() -> None:
    """重置断路器"""
    global _circuit_breaker
    if _circuit_breaker:
        _circuit_breaker.reset()


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

    # 断路器检查
    if compact_settings.circuit_breaker_enabled:
        circuit_breaker = get_circuit_breaker()
        if circuit_breaker.should_skip_compact():
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
    from utils.core.llm import get_llm_model

    session_store = SessionStore()
    messages = session_store.get_messages(thread_id)

    # 从配置读取压缩参数
    settings = get_settings_instance()
    compact_settings = settings.compact

    keep_recent = compact_settings.keep_recent
    micro_enabled = compact_settings.micro_compact_enabled
    micro_keep = compact_settings.micro_compact_keep_recent
    use_token_budget = compact_settings.use_token_budget
    tail_budget_ratio = compact_settings.tail_budget_ratio

    # 计算 token 预算
    buffer = min(compact_settings.buffer_tokens, context_window // 4)
    output_reserve = min(compact_settings.output_reserve, context_window // 4)
    effective_window = context_window - output_reserve - buffer

    # 尾部 token 预算 = 有效窗口 * 尾部预算比例
    token_budget = int(effective_window * tail_budget_ratio)

    # 检查是否有足够的消息需要压缩
    if use_token_budget:
        # Token 预算模式：检查当前消息是否超过 token 预算
        current_tokens = count_messages_tokens(messages)
        if current_tokens <= token_budget:
            return {
                "success": False,
                "message": f"消息 token 数不足 (当前 {current_tokens:,}，预算 {token_budget:,})",
                "tokens_before": current_tokens,
                "tokens_after": current_tokens,
            }
    else:
        # 消息数模式：检查消息数量
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
    result = await compact_messages(
        messages,
        keep_recent,
        llm,
        enable_micro_compact=micro_enabled,
        micro_compact_keep_recent=micro_keep,
        use_token_budget=use_token_budget,
        token_budget=token_budget if use_token_budget else None,
    )

    if result["messages_removed"] == 0:
        return {
            "success": False,
            "message": "消息数量不足，无需压缩",
            "tokens_before": result["tokens_before"],
            "tokens_after": result["tokens_after"],
        }

    # 计算压缩节省比例并记录到断路器
    tokens_before = result["tokens_before"]
    tokens_after = result["tokens_after"]
    savings_pct = (tokens_before - tokens_after) / tokens_before if tokens_before > 0 else 0

    if compact_settings.circuit_breaker_enabled:
        circuit_breaker = get_circuit_breaker()
        circuit_breaker.record_savings(savings_pct)

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
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "savings_pct": savings_pct,
        "messages_removed": result["messages_removed"],
        "messages_kept": result.get("messages_kept", len(compacted)),
        "compacted_messages": compacted,
        "micro_compact": result.get("micro_compact"),
        "split_mode": result.get("split_mode"),
    }


async def check_token_node(state: MainAgentState, context_window: int) -> dict:
    """Token 检查节点 - 检查 token 是否超过阈值，超过则压缩

    放在 finish_section 的最后一个节点，token_count_node 之后执行。

    Args:
        state: MainAgentState 状态（含 thread_id）
        context_window: 模型上下文窗口大小

    Returns:
        压缩结果或空字典
    """
    from store.session import SessionStore

    thread_id = state.get("thread_id")
    if not thread_id:
        return {}

    session_store = SessionStore()
    session = session_store.get_session(thread_id)
    if not session:
        return {}

    total_tokens = session.get("total_tokens", 0)

    if not should_auto_compact(total_tokens, context_window):
        return {}

    # 执行压缩
    result = await compact_session(thread_id, context_window)

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
