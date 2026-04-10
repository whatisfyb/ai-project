"""Main Agent - 智能路由主代理"""

import asyncio
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from utils.llm import get_llm_model
from utils.config import get_default_model_config, get_settings_instance
from utils.token_counter import TokenCounter, count_messages_tokens
from utils.compact import compact_messages
from agent.core.models import MainAgentState
from agent.core.signals import is_interrupted
from agent.main.tools import get_main_agent_tools
from agent.main.prompts import MAIN_AGENT_PROMPT
from store.session import SessionStore


# Context variable 用于在工具调用时传递 thread_id
_current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="default")


class MainAgent:
    """Main Agent - 智能路由主代理

    职责：
    - 理解用户意图
    - 路由到子代理或直接调用工具
    - 整合结果并生成响应
    """

    def __init__(self, context_window: int | None = None):
        """初始化 Main Agent

        Args:
            context_window: 模型上下文窗口大小（token 数），不填则从配置读取
        """
        self.llm = get_llm_model()

        # 从配置读取 context_window
        if context_window is None:
            model_config = get_default_model_config()
            context_window = model_config.context_window

        self.context_window = context_window
        self._init_tools()
        self._graph = None
        self._checkpointer = None
        self._token_counters: dict[str, TokenCounter] = {}  # thread_id -> TokenCounter

    def _init_tools(self):
        """初始化工具"""
        self.tools = get_main_agent_tools()
        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def _reason_node(self, state: MainAgentState) -> dict:
        """推理节点 - 分析用户输入并决定下一步"""
        messages = state["messages"]
        current_task = state.get("current_task")

        # 构建系统提示
        system_msg = SystemMessage(content=MAIN_AGENT_PROMPT)

        # 构建用户消息
        if current_task:
            user_msg = HumanMessage(content=current_task)
            all_messages = [system_msg] + messages + [user_msg]
        else:
            all_messages = [system_msg] + messages

        # 异步调用 LLM
        response = await self.llm_with_tools.ainvoke(all_messages)

        return {"messages": [response]}

    def _route_decision(self, state: MainAgentState) -> str:
        """路由决策 - 决定下一步执行什么"""
        messages = state["messages"]
        if not messages:
            return "end"

        last_message = messages[-1]

        # 检查是否有工具调用（支持字典和消息对象）
        if isinstance(last_message, dict):
            has_tool_calls = "tool_calls" in last_message and last_message["tool_calls"]
        else:
            has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        if has_tool_calls:
            return "tools"

        return "end"

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(MainAgentState)

        # 添加节点
        graph.add_node("reason", self._reason_node)
        graph.add_node("tools", ToolNode(self.tools))

        # 添加条件边
        graph.add_edge(START, "reason")
        graph.add_conditional_edges(
            "reason",
            self._route_decision,
            {
                "tools": "tools",
                "end": END,
            },
        )

        # 工具执行后返回推理
        graph.add_edge("tools", "reason")

        return graph

    @property
    def graph(self):
        """获取编译后的图"""
        if self._graph is None:
            graph = self.build_graph()
            self._graph = graph.compile(checkpointer=self.checkpointer)
        return self._graph

    @property
    def checkpointer(self):
        """获取检查点存储（延迟初始化，共享实例）

        注意：使用 MemorySaver 进行会话内的状态管理。
        持久化通过 session_store 单独实现。
        """
        if self._checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            self._checkpointer = MemorySaver()
        return self._checkpointer

    @property
    def session_store(self):
        """获取会话存储（延迟初始化）"""
        if not hasattr(self, '_session_store') or self._session_store is None:
            self._session_store = SessionStore()
        return self._session_store

    def get_token_counter(self, thread_id: str) -> TokenCounter:
        """获取会话的 TokenCounter（从历史消息计算上下文 token）"""
        if thread_id not in self._token_counters:
            counter = TokenCounter(
                context_window=self.context_window,
                warning_threshold=0.8,
            )
            self._token_counters[thread_id] = counter
        return self._token_counters[thread_id]

    def refresh_token_counter(self, thread_id: str) -> TokenCounter:
        """刷新会话的 TokenCounter（根据历史消息重新计算）"""
        counter = self.get_token_counter(thread_id)
        counter.reset()

        # 从历史消息计算 token
        history = self.session_store.get_messages(thread_id)
        if history:
            for msg in history:
                content = msg.get("content") or ""
                if content:
                    counter.add_message(content)

        return counter

    def get_token_status(self, thread_id: str = "default") -> dict:
        """获取会话的 token 状态

        Returns:
            包含 token 统计信息的字典
        """
        counter = self.get_token_counter(thread_id)
        return counter.get_status()

    async def compact_session(
        self,
        thread_id: str,
        keep_recent: int | None = None,
    ) -> dict[str, Any]:
        """压缩会话上下文

        Args:
            thread_id: 会话 ID
            keep_recent: 保留最近 N 条消息（None 则从配置读取）

        Returns:
            压缩结果，包含摘要和 token 统计
        """
        # 获取当前消息
        messages = self.session_store.get_messages(thread_id)

        # 从配置读取 keep_recent
        if keep_recent is None:
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
        result = await compact_messages(messages, keep_recent, self.llm)

        if result["messages_removed"] == 0:
            return {
                "success": False,
                "message": "消息数量不足，无需压缩",
                "tokens_before": result["tokens_before"],
                "tokens_after": result["tokens_after"],
            }

        # 更新 session_store：清空旧消息，添加压缩后的消息
        self.session_store.clear_messages(thread_id)
        for msg in result["compact_messages"]:
            role = msg["role"]
            content = msg.get("content", "")
            metadata = msg.get("metadata")
            self.session_store.add_message(thread_id, role, content, metadata)

        # 清除 LangGraph checkpointer 的状态（下次 chat 会从 session_store 恢复）
        config = {"configurable": {"thread_id": thread_id}}
        if hasattr(self.checkpointer, 'delete_thread'):
            self.checkpointer.delete_thread(thread_id)

        # 刷新 token 计数器
        self.refresh_token_counter(thread_id)

        return {
            "success": True,
            "summary": result["summary"],
            "tokens_before": result["tokens_before"],
            "tokens_after": result["tokens_after"],
            "messages_removed": result["messages_removed"],
            "messages_kept": len(result["compact_messages"]),
        }

    def _should_auto_compact(self, token_counter: TokenCounter) -> bool:
        """检查是否需要自动压缩

        Args:
            token_counter: Token 计数器

        Returns:
            是否需要压缩
        """
        settings = get_settings_instance()
        compact_settings = settings.compact

        if not compact_settings.auto_enabled:
            return False

        # 计算有效上下文窗口
        # 确保 buffer 和 reserve 不超过 context_window
        buffer = min(compact_settings.buffer_tokens, self.context_window // 4)
        output_reserve = min(compact_settings.output_reserve, self.context_window // 4)

        effective_window = self.context_window - output_reserve - buffer

        # 确保有效窗口为正数
        if effective_window <= 0:
            effective_window = self.context_window // 2

        # 计算阈值
        threshold = effective_window * compact_settings.threshold_pct

        return token_counter.total_tokens >= threshold

    async def _auto_compact_if_needed(
        self,
        thread_id: str,
        on_compact=None,
    ) -> dict[str, Any] | None:
        """如果需要则自动压缩

        Args:
            thread_id: 会话 ID
            on_compact: 压缩时的回调函数 on_compact(result)

        Returns:
            压缩结果，如果未压缩则返回 None
        """
        token_counter = self.get_token_counter(thread_id)

        if not self._should_auto_compact(token_counter):
            return None

        settings = get_settings_instance()
        compact_settings = settings.compact

        # 执行压缩
        result = await self.compact_session(
            thread_id,
            keep_recent=compact_settings.keep_recent,
        )

        if result["success"] and on_compact:
            on_compact(result)

        return result

    async def chat_async(self, message: str, thread_id: str = "default", on_token=None) -> dict[str, Any]:
        """与 Main Agent 对话（streaming 模式）

        Args:
            message: 用户消息
            thread_id: 会话 ID
            on_token: 回调函数，每收到一个 token 就调用 on_token(token)

        Returns:
            响应结果，包含 messages 和 token_status
        """
        # 设置当前 thread_id 到 context（供工具调用时使用）
        _current_thread_id.set(thread_id)

        graph = self.graph

        # 确保会话存在
        self.session_store.get_or_create_session(thread_id)

        # 刷新 token 计数器（从历史消息重新计算）
        token_counter = self.refresh_token_counter(thread_id)

        # 自动压缩检查
        auto_compact_result = None
        if self._should_auto_compact(token_counter):
            # 执行自动压缩
            auto_compact_result = await self.compact_session(thread_id)
            if auto_compact_result["success"]:
                # 压缩成功，重新刷新计数器
                token_counter = self.refresh_token_counter(thread_id)

        # 记录对话前的消息数量（用于后续同步）
        pre_msg_count = len(self.session_store.get_messages(thread_id))

        # 计算输入消息的 token 并添加到计数器
        input_tokens = count_messages_tokens([HumanMessage(content=message)])
        token_counter.add_message(message)

        # 从 checkpointer 获取历史消息
        config = {"configurable": {"thread_id": thread_id}}
        existing_state = graph.get_state(config)
        existing_messages = list(existing_state.values.get("messages", [])) if existing_state else []

        # 如果 LangGraph 没有历史（重启后），从 session_store 恢复
        if not existing_messages:
            from langchain_core.messages import ToolMessage
            history = self.session_store.get_messages(thread_id)
            # 排除刚添加的用户消息
            history = [h for h in history if h['role'] != 'user' or h['content'] != message]
            for h in history:
                metadata = h.get('metadata') or {}
                role = h['role']
                content = h['content'] or ""

                if role == 'user':
                    existing_messages.append(HumanMessage(content=content))
                elif role == 'tool':
                    # ToolMessage 需要 tool_call_id
                    tool_call_id = metadata.get('tool_call_id')
                    tool_name = metadata.get('name')
                    if tool_call_id:
                        existing_messages.append(ToolMessage(
                            content=content,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        ))
                else:
                    # assistant - 检查是否有 tool_calls
                    tool_calls = metadata.get('tool_calls')
                    if tool_calls:
                        # 有工具调用，需要创建带 tool_calls 的 AIMessage
                        existing_messages.append(AIMessage(
                            content=content,
                            tool_calls=tool_calls,
                        ))
                    else:
                        existing_messages.append(AIMessage(content=content))

        # 追加新消息
        input_data = {
            "messages": existing_messages + [HumanMessage(content=message)],
            "current_task": None,
            "memory_context": None,
            "subagent_results": {},
        }

        accumulated_response = ""

        # 使用 astream() 进行 streaming
        # 注意：astream 内部 yield 不频繁，需要用 wait_for 加超时来检查中断
        try:
            async for chunk in graph.astream(
                input_data,
                config,
                stream_mode=["messages", "updates"],
                version="v2",
            ):
                # 每次收到 chunk 后检查中断
                if is_interrupted():
                    # 被中断了，同步已生成的消息到 session_store
                    final_state = graph.get_state(config)
                    if final_state and final_state.values:
                        final_messages = final_state.values.get("messages", [])
                        new_messages = final_messages[pre_msg_count:]
                        if new_messages:
                            self.session_store.add_messages_batch(thread_id, new_messages)
                            output_tokens = count_messages_tokens(new_messages)
                            token_counter.add_messages(new_messages)
                        existing_messages = list(final_messages)
                    existing_messages.append(
                        AIMessage(content="[系统消息] 用户中断了执行。")
                    )
                    return {
                        "messages": existing_messages,
                        "token_status": token_counter.get_status(),
                    }

                # version="v2" 返回 dict 格式: {"type": ..., "data": ..., "ns": ...}
                if chunk.get("type") == "messages":
                    msg, _ = chunk.get("data", {})
                    if isinstance(msg, AIMessageChunk) and msg.content:
                        accumulated_response += msg.content
                        if on_token:
                            on_token(msg.content)
                elif chunk.get("type") == "updates":
                    # 节点更新，不做特殊处理
                    pass

        except asyncio.CancelledError:
            # 被取消，同步已生成的消息到 session_store
            final_state = graph.get_state(config)
            if final_state and final_state.values:
                final_messages = final_state.values.get("messages", [])
                # 获取新增的消息（对话前有 pre_msg_count 条）
                new_messages = final_messages[pre_msg_count:]
                if new_messages:
                    self.session_store.add_messages_batch(thread_id, new_messages)
            raise

        # 正常结束，同步完整消息链到 session_store
        # 从 graph 最终状态获取所有消息，只同步新增的部分
        final_state = graph.get_state(config)
        if final_state and final_state.values:
            final_messages = final_state.values.get("messages", [])
            # 获取新增的消息（对话前有 pre_msg_count 条）
            new_messages = final_messages[pre_msg_count:]
            if new_messages:
                self.session_store.add_messages_batch(thread_id, new_messages)
                # 计算输出消息的 token
                output_tokens = count_messages_tokens(new_messages)
                token_counter.add_messages(new_messages)
            # 更新 existing_messages 用于返回
            existing_messages = list(final_messages)

        # 更新 session_store 的 token 统计
        self.session_store.update_session_tokens(thread_id, token_counter.total_tokens)

        result = {
            "messages": existing_messages,
            "token_status": token_counter.get_status(),
        }

        # 如果发生了自动压缩，添加到结果中
        if auto_compact_result and auto_compact_result.get("success"):
            result["auto_compact"] = auto_compact_result

        return result

    def chat(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        """与 Main Agent 对话（兼容模式，内部调用 async 版本）

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Returns:
            响应结果
        """
        try:
            loop = asyncio.get_running_loop()
            # 已经在 running loop 中，创建 task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.chat_async(message, thread_id))
                return future.result()
        except RuntimeError:
            # 没有 running loop，可以直接用 asyncio.run
            return asyncio.run(self.chat_async(message, thread_id))

    def get_response(self, result: dict[str, Any]) -> str:
        """从结果中提取响应文本"""
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                return last.content
        return ""


def create_main_agent() -> MainAgent:
    """创建 Main Agent 实例"""
    return MainAgent()
