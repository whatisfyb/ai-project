"""Main Agent - 智能路由主代理"""

import asyncio
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langgraph.graph import StateGraph, END, START

from utils.llm import get_llm_model
from utils.config import get_default_model_config, get_settings_instance
from agent.core.models import MainAgentState
from agent.core.signals import is_interrupted
from agent.main.tools import get_main_agent_tools
from agent.main.prompts import MAIN_AGENT_PROMPT
from agent.middleware.long_term_memory import load_memory_node, memory_check_node
from agent.middleware.token_count import token_count_node
from agent.middleware.context_compact import check_token_node, set_checkpointer
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

    def _init_tools(self):
        """初始化工具"""
        self.tools = get_main_agent_tools()
        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def _reason_node(self, state: MainAgentState) -> dict:
        """推理节点 - 分析用户输入并决定下一步"""
        messages = state["messages"]
        current_task = state.get("current_task")
        memory_context = state.get("memory_context")

        # 构建系统提示
        system_content = MAIN_AGENT_PROMPT

        # 如果有记忆上下文，添加到系统提示
        if memory_context:
            system_content = f"{memory_context}\n\n---\n\n{MAIN_AGENT_PROMPT}"

        system_msg = SystemMessage(content=system_content)

        # 构建用户消息
        if current_task:
            user_msg = HumanMessage(content=current_task)
            all_messages = [system_msg] + messages + [user_msg]
        else:
            all_messages = [system_msg] + messages

        # 异步调用 LLM
        response = await self.llm_with_tools.ainvoke(all_messages)

        return {"messages": messages + [response]}

    async def _tools_node(self, state: MainAgentState) -> dict:
        """工具执行节点 - 执行工具调用并追加消息"""
        from langgraph.prebuilt import ToolNode

        tool_node = ToolNode(self.tools)
        result = await tool_node.ainvoke(state)

        # result 是 {"messages": [tool_messages]}，需要追加到现有消息
        return {"messages": state["messages"] + result["messages"]}

    async def _check_token_node_wrapper(self, state: MainAgentState) -> dict:
        """Token 检查节点包装器"""
        return await check_token_node(state, self.context_window)

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

    # ============ Start Section 节点 ============

    # ============ Finish Section 节点 ============

    def _build_finish_section(self) -> StateGraph:
        """构建 finish_section 子图 - 对话结束后的处理逻辑

        当前包含：
        - memory_check: 检查并保存记忆

        后续可扩展：
        - 日志记录
        - 状态清理
        - 汇总报告
        """
        from langgraph.graph import StateGraph as SubStateGraph

        finish_graph = SubStateGraph(MainAgentState)

        # 添加节点
        finish_graph.add_node("memory_check", memory_check_node)
        finish_graph.add_node("token_count", token_count_node)
        finish_graph.add_node("check_token", self._check_token_node_wrapper)

        # 定义边：并行执行 memory_check 和 token_count -> check_token
        finish_graph.add_edge(START, "memory_check")
        finish_graph.add_edge(START, "token_count")
        finish_graph.add_edge("token_count", "check_token")
        finish_graph.add_edge("memory_check", END)
        finish_graph.add_edge("check_token", END)

        return finish_graph

    def _build_start_section(self) -> StateGraph:
        """构建 start_section 子图 - 对话开始前的处理逻辑

        当前包含：
        - load_memory: 加载长期记忆到上下文

        后续可扩展：
        - 权限检查
        - 会话初始化
        - 动态工具加载
        """
        from langgraph.graph import StateGraph as SubStateGraph

        start_graph = SubStateGraph(MainAgentState)

        # 添加节点
        start_graph.add_node("load_memory", load_memory_node)

        # 定义边
        start_graph.add_edge(START, "load_memory")
        start_graph.add_edge("load_memory", END)

        return start_graph

    def build_graph(self) -> StateGraph:
        """构建状态图

        完整结构：

        START -> start_section -> reason -> (tools or finish_section)
                                          ↓Yes        ↓No
                                        reason   finish_section
                                          ↓             ↓
                                        ...循环...      END

        start_section:  对话前处理（可扩展）
        main_loop:      ReAct 循环（reason <-> tools）
        finish_section: 对话后处理（memory_check 等）
        """
        graph = StateGraph(MainAgentState)

        # ============ 构建子图 ============
        start_section = self._build_start_section().compile()
        finish_section = self._build_finish_section().compile()

        # ============ 添加节点 ============
        graph.add_node("start_section", start_section)
        graph.add_node("reason", self._reason_node)
        graph.add_node("tools", self._tools_node)
        graph.add_node("finish_section", finish_section)

        # ============ 添加边 ============
        # START -> start_section -> reason
        graph.add_edge(START, "start_section")
        graph.add_edge("start_section", "reason")

        # reason 的条件路由
        graph.add_conditional_edges(
            "reason",
            self._route_decision,
            {
                "tools": "tools",
                "end": "finish_section",  # 无工具调用进入 finish_section
            },
        )

        # tools 执行后返回 reason（ReAct 循环）
        graph.add_edge("tools", "reason")

        # finish_section -> END
        graph.add_edge("finish_section", END)

        return graph

    @property
    def graph(self):
        """获取编译后的图"""
        if self._graph is None:
            graph = self.build_graph()
            self._graph = graph.compile(checkpointer=self.checkpointer)
            set_checkpointer(self.checkpointer)
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

    async def chat_async(self, message: str, thread_id: str = "default", on_token=None) -> dict[str, Any]:
        """与 Main Agent 对话（streaming 模式）

        Args:
            message: 用户消息
            thread_id: 会话 ID
            on_token: 回调函数，每收到一个 token 就调用 on_token(token)

        Returns:
            响应结果，包含 messages
        """
        # 设置当前 thread_id 到 context（供工具调用时使用）
        _current_thread_id.set(thread_id)

        graph = self.graph

        # 确保会话存在
        self.session_store.get_or_create_session(thread_id)

        # 记录对话前的消息数量（用于后续同步）
        pre_msg_count = len(self.session_store.get_messages(thread_id))

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
            "thread_id": thread_id,
            "session_id": thread_id,
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

                        # 处理未完成的 tool_calls：保留 AIMessage 内容，清空 tool_calls
                        if final_messages:
                            last_msg = final_messages[-1]
                            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                                # 有未完成的工具调用，清空 tool_calls 保留 content
                                interrupted_content = last_msg.content or ""
                                final_messages[-1] = AIMessage(
                                    content=interrupted_content + "\n\n[系统消息] 用户中断了执行，工具调用已取消。"
                                    if interrupted_content else "[系统消息] 用户中断了执行，工具调用已取消。"
                                )

                        new_messages = final_messages[pre_msg_count:]
                        if new_messages:
                            self.session_store.add_messages_batch(thread_id, new_messages)
                        existing_messages = list(final_messages)
                    return {
                        "messages": existing_messages,
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

                # 处理未完成的 tool_calls：保留 AIMessage 内容，清空 tool_calls
                if final_messages:
                    last_msg = final_messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        interrupted_content = last_msg.content or ""
                        final_messages[-1] = AIMessage(
                            content=interrupted_content + "\n\n[系统消息] 执行被取消，工具调用已终止。"
                            if interrupted_content else "[系统消息] 执行被取消，工具调用已终止。"
                        )

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
            # 更新 existing_messages 用于返回
            existing_messages = list(final_messages)

        result = {
            "messages": existing_messages,
        }

        return result


def create_main_agent() -> MainAgent:
    """创建 Main Agent 实例"""
    return MainAgent()
