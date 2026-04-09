"""Main Agent - 智能路由主代理"""

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from utils.llm import get_llm_model
from agent.state import MainAgentState


# 系统提示词
MAIN_AGENT_PROMPT = """你是一个智能助手，能够理解用户需求并选择最合适的方式完成任务。

## 可用工具

### 搜索和信息工具
- `tavily_search` - 网络搜索
- `tavily_extract` - URL内容提取
- `arxiv_search` - arXiv论文搜索
- `arxiv_download_pdf` - 下载arXiv论文PDF

### 子代理分发工具
- `dispatch_agent` - 分发任务给专门的子代理执行
  - subagent_type="Plan": 用于复杂任务的拆解和规划
  - subagent_type="Research": 用于信息搜索、论文查找、知识库检索
  - subagent_type="Analysis": 用于数据分析、报告生成

- `execute_plan` - 执行已有的计划（需要 plan_id）。如果之前的计划被中断，用户想继续时，用之前的 plan_id 调用此工具恢复执行。

- `list_subagents` - 列出所有可用的子代理

### Skills 技能工具
- `list_skills` - 列出所有可用的 AI 技能（如天气查询等）
- `load_skills` - 加载指定技能到当前上下文（需要技能名称列表）
- `skill_call` - 调用已加载的技能执行实际任务
  - name: 技能名称（如 "weather"）
  - args_json: 技能参数的 JSON 字符串（如 \'{"city": "北京", "days": 3}\'）

## 决策规则

1. **简单对话**（问候、闲聊）：直接回复，不要调用任何工具
2. **简单搜索**（"搜索XXX"）：使用 tavily_search
3. **复杂研究**（"研究XXX领域"、"帮我调研XXX"）：使用 dispatch_agent，subagent_type="Research"
4. **任务规划**（"帮我规划XXX"、"如何完成XXX"）：使用 dispatch_agent，subagent_type="Plan"
5. **复杂任务执行**（"帮我写一个爬虫"、"帮我开发XXX"）：
   - 先用 dispatch_agent，subagent_type="Plan" 生成计划，获取 plan_id
   - 再用 execute_plan，plan_id=<上一步返回的plan_id> 执行计划
6. **数据分析**（"分析XXX数据"、"生成报告"）：使用 dispatch_agent，subagent_type="Analysis"
7. **恢复中断的任务**（"继续"、"继续执行"、"接着做"）：
   - 从对话历史中找到最近的 plan_id
   - 使用 execute_plan(plan_id) 恢复执行

重要：对于简单问候和闲聊，直接回复用户，不要调用任何工具！
"""


def _is_interrupted() -> bool:
    """检查是否被中断"""
    from agent.worker import is_interrupted as _is_interrupted
    return _is_interrupted()


class MainAgent:
    """Main Agent - 智能路由主代理

    职责：
    - 理解用户意图
    - 路由到子代理或直接调用工具
    - 整合结果并生成响应
    """

    def __init__(self):
        self.llm = get_llm_model()
        self._init_tools()
        self._graph = None
        self._checkpointer = None

    def _init_tools(self):
        """初始化工具"""
        from tools.tavily import tavily_search, tavily_extract
        from tools.arxiv_search import arxiv_search, arxiv_download_pdf
        from tools.agent import dispatch_agent, list_subagents, execute_plan
        from tools.skills_manager import load_skills, list_skills, skill_call

        self.tools = [
            tavily_search,
            tavily_extract,
            arxiv_search,
            arxiv_download_pdf,
            dispatch_agent,
            execute_plan,
            list_subagents,
            load_skills,
            list_skills,
            skill_call,
        ]

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
        """获取检查点存储（延迟初始化，共享实例）"""
        if self._checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            self._checkpointer = MemorySaver()
        return self._checkpointer

    async def chat_async(self, message: str, thread_id: str = "default", on_token=None) -> dict[str, Any]:
        """与 Main Agent 对话（streaming 模式）

        Args:
            message: 用户消息
            thread_id: 会话 ID
            on_token: 回调函数，每收到一个 token 就调用 on_token(token)

        Returns:
            响应结果，包含 messages
        """
        from langchain_core.messages import HumanMessage

        graph = self.graph

        # 从 checkpointer 获取历史消息
        config = {"configurable": {"thread_id": thread_id}}
        existing_state = graph.get_state(config)
        existing_messages = list(existing_state.values.get("messages", [])) if existing_state else []

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
                if _is_interrupted():
                    # 被中断了，保存已收到的内容作为助手消息
                    if accumulated_response:
                        existing_messages.append(AIMessage(content=accumulated_response))
                    existing_messages.append(
                        AIMessage(content="[系统消息] 用户中断了执行。")
                    )
                    return {"messages": existing_messages}

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
            # 被取消，保存部分输出
            if accumulated_response:
                existing_messages.append(AIMessage(content=accumulated_response))
            existing_messages.append(
                AIMessage(content="[系统消息] 用户中断了执行。")
            )
            raise

        # 正常结束，保存最终响应
        if accumulated_response:
            existing_messages.append(AIMessage(content=accumulated_response))

        return {"messages": existing_messages}

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


# ============ REPL 入口 ============

# 模块级别的中断标志（用于 signal handler 和 async 函数之间共享）
_repl_interrupted = False


def run_repl():
    """运行 REPL 交互（同步入口，内部启动 async 循环）"""
    import asyncio
    import signal
    from agent.registry import agent_registry, terminate
    from agent.worker import set_interrupt, is_interrupted, clear_interrupt

    global _repl_interrupted

    def handle_interrupt(signum, frame):
        """REPL 级别的 Ctrl+C 处理"""
        global _repl_interrupted
        _repl_interrupted = True
        set_interrupt()

    # 保存原始处理器并注册新的
    _old_sigint_handler = signal.signal(signal.SIGINT, handle_interrupt)

    try:
        asyncio.run(_run_repl_async())
    finally:
        signal.signal(signal.SIGINT, _old_sigint_handler)


async def _run_repl_async():
    """异步 REPL 循环"""
    global _repl_interrupted
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from agent.registry import agent_registry, terminate
    from agent.worker import is_interrupted, clear_interrupt

    console = Console()
    agent = create_main_agent()

    console.print(Panel.fit(
        "[bold green]Main Agent 已启动[/bold green]\n"
        "输入消息与 Agent 对话\n\n"
        "[dim]命令：[/dim]\n"
        "  [cyan]/exit[/cyan]  - 退出\n"
        "  [cyan]/clear[/cyan]  - 重置会话\n"
        "  [cyan]/status[/cyan]  - 查看状态\n"
        "  [cyan]Ctrl+C[/cyan]   - 终止当前任务",
        title="Main Agent"
    ))

    thread_id = "default"

    while True:
        # 检查是否被中断
        if _repl_interrupted:
            _repl_interrupted = False
            clear_interrupt()
            if agent_registry.is_running():
                terminate()
                console.print("[yellow]任务已终止[/yellow]\n")
                console.print("[dim]请输入新任务或 /exit 退出[/dim]\n")
            else:
                # 没有任务在运行，Ctrl+C 只是放弃当前输入，继续等待
                console.print("[dim]已取消输入[/dim]\n")
            continue

        # 异步等待用户输入（使用 asyncio.to_thread 避免阻塞事件循环）
        try:
            user_input = await asyncio.to_thread(
                lambda: console.input("[bold blue]You:[/bold blue] ").strip()
            )
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C 或 Ctrl+D 在等待输入时
            user_input = ""

        if not user_input:
            # Ctrl+C 会导致 user_input 为空，这不算正常输入，继续循环即可
            continue

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            console.print("[bold yellow]再见！[/bold yellow]")
            break

        if user_input.lower() == "/clear":
            thread_id = f"session_{int(__import__('time').time())}"
            console.print("[bold green]会话已重置[/bold green]\n")
            continue

        if user_input.lower() == "/status":
            running = agent_registry.is_running()
            plan_ids = agent_registry.get_running_plan_ids()
            if running:
                console.print(f"[cyan]正在运行的任务: {plan_ids}[/cyan]\n")
            else:
                console.print("[dim]没有正在运行的任务[/dim]\n")
            continue

        # 调用 Agent（async streaming 模式）
        console.print("\n[bold green]Agent:[/bold green] ")
        accumulated = []

        def on_token(token):
            accumulated.append(token)
            console.print(token, end="")  # 实时打印 token

        try:
            result = await agent.chat_async(user_input, thread_id, on_token=on_token)
        except Exception as e:
            console.print(f"\n[bold red]错误:[/bold red] {e}\n")
            continue

        console.print()  # 换行

        # 如果被中断了，打印提示
        if is_interrupted():
            console.print("[yellow]\n[执行被中断，已保存部分输出][/yellow]\n")
            clear_interrupt()

        console.print()


if __name__ == "__main__":
    run_repl()
