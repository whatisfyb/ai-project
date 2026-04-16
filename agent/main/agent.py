"""Main Agent - 智能路由主代理（事件驱动架构）"""

import asyncio
from contextvars import ContextVar
from typing import Any, Callable

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
)
from langgraph.graph import StateGraph, END, START

from utils.core.llm import get_llm_model
from utils.core.config import get_default_model_config
from agent.core.models import MainAgentState
from agent.core.signals import is_interrupted, save_checkpoint, load_checkpoint
from agent.core.events import AgentEvent, EventType
from agent.core.base_agent import BaseAgent
from agent.a2a.models import AgentCard, AgentCapabilities, Skill
from agent.main.tools import get_main_agent_tools
from agent.main.prompts import MAIN_AGENT_PROMPT
from agent.middleware.long_term_memory import load_memory_node, memory_check_node
from agent.middleware.token_count import token_count_node
from agent.middleware.context_compact import check_token_node
from store.session import SessionStore


# Context variable 用于在工具调用时传递 thread_id
_current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="default")


class MainAgent(BaseAgent):
    """Main Agent - 智能路由主代理（事件驱动架构）

    职责：
    - 理解用户意图
    - 路由到子代理或直接调用工具
    - 整合结果并生成响应
    - 响应 Inbox 通知（Worker 结果）

    架构：
    - 4 个独立 Section：start, reason, tools, finish
    - 事件驱动循环：asyncio.Queue 处理 USER_INPUT 和 INBOX_NOTIFICATION
    """

    agent_id = "main"
    agent_type = "main"

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

        # 编译四个独立 Section
        self.start_section = self._build_start_section().compile()
        self.reason_section = self._build_reason_section().compile()
        self.tools_section = self._build_tools_section().compile()
        self.finish_section = self._build_finish_section().compile()

        # 事件循环相关
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_state: MainAgentState | None = None
        self._state_lock = asyncio.Lock()

        # 流式输出回调
        self._on_token: Callable[[str], None] | None = None

    def _init_tools(self):
        """初始化工具"""
        self.tools = get_main_agent_tools()
        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        # 缓存 ToolNode 避免循环中重复创建
        from langgraph.prebuilt import ToolNode

        self._tool_node = ToolNode(self.tools)

    # ============ BaseAgent 接口实现 ============

    def get_card(self) -> AgentCard:
        """返回 Agent 能力声明"""
        return AgentCard(
            id=self.agent_id,
            name="Main Agent",
            description="智能路由主代理，负责理解用户意图、路由到子代理、整合结果",
            capabilities=AgentCapabilities(
                text=True,
                files=True,
                streaming=True,
                push_notifications=True,
            ),
            skills=[
                Skill(name="chat", description="与用户对话"),
                Skill(name="route", description="路由任务到子代理"),
                Skill(name="tool_call", description="调用工具"),
            ],
        )

    def handle_task(self, task) -> Any:
        """处理 A2A Task"""
        # MainAgent 通过事件循环处理任务，不直接调用
        if self._running:
            self.send_event_sync(
                AgentEvent.user_input(
                    message=task.history[0].get_text() if task.history else "",
                    thread_id=task.metadata.get("thread_id", "default"),
                )
            )

    # ============ Section 构建方法 ============

    def _build_start_section(self) -> StateGraph:
        """构建 start_section 子图 - 对话开始前的处理逻辑

        包含：
        - load_memory: 加载长期记忆到上下文
        """
        from langgraph.graph import StateGraph as SubStateGraph

        start_graph = SubStateGraph(MainAgentState)
        start_graph.add_node("load_memory", load_memory_node)
        start_graph.add_edge(START, "load_memory")
        start_graph.add_edge("load_memory", END)
        return start_graph

    def _build_reason_section(self) -> StateGraph:
        """构建 reason_section 子图 - LLM 推理"""
        from langgraph.graph import StateGraph as SubStateGraph

        graph = SubStateGraph(MainAgentState)
        graph.add_node("reason", self._reason_node)
        graph.add_edge(START, "reason")
        graph.add_edge("reason", END)
        return graph

    def _build_tools_section(self) -> StateGraph:
        """构建 tools_section 子图 - 工具执行"""
        from langgraph.graph import StateGraph as SubStateGraph

        graph = SubStateGraph(MainAgentState)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        return graph

    def _build_finish_section(self) -> StateGraph:
        """构建 finish_section 子图 - 对话结束后的处理逻辑

        包含：
        - memory_check: 检查并保存记忆
        - token_count: 统计 token
        - check_token: 检查是否需要压缩
        - sync_state: 同步状态到 SessionStore
        """
        from langgraph.graph import StateGraph as SubStateGraph

        finish_graph = SubStateGraph(MainAgentState)

        finish_graph.add_node("memory_check", memory_check_node)
        finish_graph.add_node("token_count", token_count_node)
        finish_graph.add_node("check_token", self._check_token_node_wrapper)
        finish_graph.add_node("sync_state", self._sync_state_node)

        # memory_check 和 token_count 并行 -> check_token -> sync_state
        finish_graph.add_edge(START, "memory_check")
        finish_graph.add_edge(START, "token_count")
        finish_graph.add_edge("token_count", "check_token")
        finish_graph.add_edge("check_token", "sync_state")
        finish_graph.add_edge("memory_check", "sync_state")
        finish_graph.add_edge("sync_state", END)

        return finish_graph

    # ============ 节点方法 ============

    async def _reason_node(self, state: MainAgentState) -> dict:
        """推理节点 - 分析用户输入并决定下一步

        注意：用户消息已由 _inject_event() 或 _run_without_loop() 注入到 state["messages"] 中，
        此节点不再重复添加用户消息。
        """
        messages = state["messages"]
        memory_context = state.get("memory_context")

        # 构建系统提示
        system_content = MAIN_AGENT_PROMPT

        # 如果有记忆上下文，添加到系统提示
        if memory_context:
            system_content = f"{memory_context}\n\n---\n\n{MAIN_AGENT_PROMPT}"

        system_msg = SystemMessage(content=system_content)

        # 构建 LLM 输入消息（用户消息已在 messages 中）
        all_messages = [system_msg] + messages

        # 流式调用（支持 on_token 回调）
        if self._on_token:
            accumulated = AIMessageChunk(content="")
            async for chunk in self.llm_with_tools.astream(all_messages):
                if is_interrupted():
                    break
                accumulated += chunk
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    self._on_token(chunk.content)
            response = accumulated
        else:
            response = await self.llm_with_tools.ainvoke(all_messages)

        return {"messages": messages + [response]}

    async def _tools_node(self, state: MainAgentState) -> dict:
        """工具执行节点 - 执行工具调用并追加消息"""
        result = await self._tool_node.ainvoke(state)

        # result 是 {"messages": [tool_messages]}，需要追加到现有消息
        return {"messages": state["messages"] + result["messages"]}

    async def _check_token_node_wrapper(self, state: MainAgentState) -> dict:
        """Token 检查节点包装器"""
        return await check_token_node(state, self.context_window)

    async def _sync_state_node(self, state: MainAgentState) -> dict:
        """同步状态到 SessionStore"""
        thread_id = state.get("thread_id", "default")
        messages = state.get("messages", [])
        if messages:
            self.session_store.add_messages_batch(thread_id, messages)
        return {}

    # ============ 事件驱动循环 ============

    async def run_loop(self, thread_id: str = "default"):
        """MainAgent 事件驱动主循环

        Args:
            thread_id: 会话 ID
        """
        self._running = True
        self._loop = asyncio.get_running_loop()
        first_trigger = True

        # 初始化 state（包括恢复历史消息）
        self._current_state = self._make_initial_state(thread_id)

        # 恢复历史消息
        history = self.session_store.get_messages(thread_id)
        if history:
            self._current_state["messages"] = self._restore_messages(history)

        # 注册 inbox 监听
        self._register_inbox_listener()

        while self._running:
            # 1. 等待事件（关键：这里可以响应外部事件！）
            event = await self._event_queue.get()

            if event.type == EventType.SHUTDOWN:
                break

            # 2. 首次触发走 start_section
            if first_trigger:
                self._current_state = await self.start_section.ainvoke(
                    self._current_state
                )
                first_trigger = False

            # 3. 注入事件到 state
            self._inject_event(event)

            # 4. reason → tools ReAct 循环
            while True:
                # 检查中断
                if is_interrupted():
                    break

                self._current_state = await self.reason_section.ainvoke(
                    self._current_state
                )

                if self._has_tool_calls(self._current_state):
                    self._current_state = await self.tools_section.ainvoke(
                        self._current_state
                    )
                else:
                    break

            # 5. finish_section
            self._current_state = await self.finish_section.ainvoke(self._current_state)

            # 6. 通知完成
            if event.on_complete:
                event.on_complete.set()

        self._running = False

    def _make_initial_state(self, thread_id: str) -> MainAgentState:
        """创建初始状态"""
        return MainAgentState(
            messages=[],
            current_task=None,
            memory_context=None,
            thread_id=thread_id,
            event_type="",
            inbox_results=[],
        )

    def _inject_event(self, event: AgentEvent):
        """将事件注入到当前 state"""
        if event.type == EventType.USER_INPUT:
            message = event.data["message"]
            self._current_state["messages"].append(HumanMessage(content=message))
            self._current_state["current_task"] = message
            self._current_state["event_type"] = "user_input"
            self._current_state["inbox_results"] = []

        elif event.type == EventType.INBOX_NOTIFICATION:
            # 格式化 inbox 结果并注入
            task_id = event.data.get("task_id", "")
            status = event.data.get("status", "")
            result = event.data.get("result")
            error = event.data.get("error")

            summary = self._format_inbox_result(task_id, status, result, error)
            self._current_state["messages"].append(SystemMessage(content=summary))
            self._current_state["inbox_results"].append(
                {
                    "task_id": task_id,
                    "status": status,
                    "result": result,
                    "error": error,
                }
            )
            self._current_state["current_task"] = None
            self._current_state["event_type"] = "inbox_notification"

    def _has_tool_calls(self, state: MainAgentState) -> bool:
        """检查最后一条消息是否有工具调用"""
        messages = state.get("messages", [])
        if not messages:
            return False
        last = messages[-1]
        if isinstance(last, dict):
            return bool(last.get("tool_calls"))
        return hasattr(last, "tool_calls") and bool(last.tool_calls)

    def _format_inbox_result(
        self, task_id: str, status: str, result: str | None, error: str | None
    ) -> str:
        """格式化单个 inbox 结果"""
        parts = ["[Worker 结果通知]"]
        status_text = "成功" if status == "success" else "失败"
        line = f"任务 {task_id}: {status_text}"
        if result:
            line += f" → {result[:200]}"
        elif error:
            line += f" → 错误: {error[:100]}"
        parts.append(line)
        parts.append("\n请分析结果并决定下一步。")
        return "\n".join(parts)

    def _register_inbox_listener(self):
        """注册 inbox 事件监听器"""
        try:
            from agent.a2a.dispatcher import get_inbox

            def on_inbox_result(task_result):
                """Worker 结果回调（从任意线程调用）"""
                event = AgentEvent.inbox_notification(
                    task_id=task_result.task_id,
                    status=task_result.status.value
                    if hasattr(task_result.status, "value")
                    else str(task_result.status),
                    result=task_result.result
                    if hasattr(task_result, "result")
                    else None,
                    error=task_result.error if hasattr(task_result, "error") else None,
                    thread_id=self._current_state.get("thread_id", "default")
                    if self._current_state
                    else "default",
                )
                self.send_event_sync(event)

            inbox = get_inbox()
            inbox.subscribe(on_inbox_result)
        except Exception:
            # Inbox 可能还未初始化，忽略
            pass

    # ============ 事件投递方法 ============

    async def send_event(
        self, event: AgentEvent, on_token: Callable[[str], None] | None = None
    ):
        """投递事件到事件队列

        Args:
            event: 事件对象
            on_token: 流式输出回调
        """
        self._on_token = on_token
        await self._event_queue.put(event)

    def send_event_sync(self, event: AgentEvent):
        """从非 async 上下文投递事件（线程安全）"""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._event_queue.put(event), self._loop)

    async def shutdown(self):
        """关闭事件循环"""
        self._running = False
        await self._event_queue.put(AgentEvent.shutdown())

    @property
    def is_running(self) -> bool:
        return self._running

    # ============ 兼容接口 ============

    @property
    def session_store(self):
        """获取会话存储（延迟初始化）"""
        if not hasattr(self, "_session_store") or self._session_store is None:
            self._session_store = SessionStore()
        return self._session_store

    async def chat_async(
        self, message: str, thread_id: str = "default", on_token=None
    ) -> dict[str, Any]:
        """与 Main Agent 对话（兼容旧接口，内部使用事件循环）

        如果事件循环未启动，会话模式（不启动循环，直接执行）。
        如果事件循环已启动，投递事件并等待完成。

        Args:
            message: 用户消息
            thread_id: 会话 ID
            on_token: 回调函数，每收到一个 token 就调用 on_token(token)

        Returns:
            响应结果，包含 messages
        """
        # 设置当前 thread_id 到 context（供工具调用时使用）
        _current_thread_id.set(thread_id)

        # 确保 session 存在
        self.session_store.get_or_create_session(thread_id)

        # 恢复历史消息
        if (
            self._current_state is None
            or self._current_state.get("thread_id") != thread_id
        ):
            self._current_state = self._make_initial_state(thread_id)
            history = self.session_store.get_messages(thread_id)
            if history:
                self._current_state["messages"] = self._restore_messages(history)

        # 如果事件循环未启动，使用简化流程
        if not self._running:
            return await self._run_without_loop(message, thread_id, on_token)

        # 创建完成信号
        on_complete = asyncio.Event()

        # 投递事件
        event = AgentEvent.user_input(message, thread_id, on_complete)
        await self.send_event(event, on_token=on_token)

        # 等待处理完成
        await on_complete.wait()

        return {
            "messages": self._current_state.get("messages", [])
            if self._current_state
            else []
        }

    async def _run_without_loop(
        self, message: str, thread_id: str, on_token=None
    ) -> dict[str, Any]:
        """不启动事件循环的简化执行流程（向后兼容）"""
        self._on_token = on_token

        # 注入用户消息
        self._current_state["messages"].append(HumanMessage(content=message))
        self._current_state["current_task"] = message
        self._current_state["event_type"] = "user_input"

        # start_section
        self._current_state = await self.start_section.ainvoke(self._current_state)

        # ReAct 循环
        while True:
            if is_interrupted():
                break

            self._current_state = await self.reason_section.ainvoke(self._current_state)

            if self._has_tool_calls(self._current_state):
                self._current_state = await self.tools_section.ainvoke(
                    self._current_state
                )
            else:
                break

        # finish_section
        self._current_state = await self.finish_section.ainvoke(self._current_state)

        return {
            "messages": self._current_state.get("messages", [])
            if self._current_state
            else []
        }

    def _restore_messages(self, history: list[dict]) -> list:
        """从 session_store 恢复 LangChain 消息对象"""
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        result = []
        for h in history:
            metadata = h.get("metadata") or {}
            role = h["role"]
            content = h["content"] or ""

            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "tool":
                tool_call_id = metadata.get("tool_call_id")
                tool_name = metadata.get("name")
                if tool_call_id:
                    result.append(
                        ToolMessage(
                            content=content, tool_call_id=tool_call_id, name=tool_name
                        )
                    )
            else:
                tool_calls = metadata.get("tool_calls")
                if tool_calls:
                    result.append(AIMessage(content=content, tool_calls=tool_calls))
                else:
                    result.append(AIMessage(content=content))
        return result

    # ============ BaseAgent 状态保存/恢复 ============

    def get_state(self) -> dict[str, Any]:
        """获取当前状态（用于检查点保存）"""
        if self._current_state is None:
            return {}
        return {
            "messages": [
                {
                    "role": "user"
                    if hasattr(m, "type") and m.type == "human"
                    else "assistant",
                    "content": str(m.content),
                }
                for m in self._current_state.get("messages", [])
            ],
            "thread_id": self._current_state.get("thread_id", "default"),
            "current_task": self._current_state.get("current_task"),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """恢复状态（从检查点恢复）"""
        if not state:
            return
        thread_id = state.get("thread_id", "default")
        self._current_state = self._make_initial_state(thread_id)
        if "messages" in state:
            for m in state["messages"]:
                if m["role"] == "user":
                    self._current_state["messages"].append(
                        HumanMessage(content=m["content"])
                    )
                else:
                    self._current_state["messages"].append(
                        AIMessage(content=m["content"])
                    )
        if "current_task" in state:
            self._current_state["current_task"] = state["current_task"]

    def on_interrupt(self) -> None:
        """中断回调 - 保存检查点"""
        if self._current_state:
            save_checkpoint(self.agent_id, self.get_state())


def create_main_agent() -> MainAgent:
    """创建 Main Agent 实例"""
    return MainAgent()
