"""Main Agent - 智能路由主代理"""

import asyncio
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from utils.llm import get_llm_model
from agent.state import MainAgentState


# Context variable 用于在工具调用时传递 thread_id
_current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="default")


# 系统提示词
MAIN_AGENT_PROMPT = """你是一个智能助手，能够理解用户需求并选择最合适的方式完成任务。

## 可用工具

### 文件操作工具
- `read` - 读取文件内容（支持行偏移和限制，大文件自动截断）
- `write` - 创建/覆盖文件
- `append` - 追加内容到文件
- `edit` - 精准编辑（搜索替换）
- `edit_regex` - 正则表达式编辑

### Shell 命令工具
- `bash` - 执行 Shell 命令
- `bash_script` - 执行多行脚本

### 文件查找工具
- `glob` - 查找匹配 glob 模式的文件（如 `**/*.py`）
- `glob_list` - 多模式文件查找

### 搜索和信息工具
- `web_search` - 网络搜索
- `web_fetch` - 从URL提取内容
- `web_scrape` - 抓取网页（Firecrawl，支持 markdown/html）
- `web_crawl` - 爬取整个网站
- `web_map` - 发现网站所有URL
- `arxiv_search` - arXiv学术论文搜索
- `arxiv_get_by_id` - 根据ID获取arXiv论文
- `arxiv_download_pdf` - 下载arXiv论文PDF
- `grep` - 在文件中搜索内容（基于 ripgrep，支持正则表达式、文件类型过滤）
- `grep_count` - 统计匹配次数（比 grep 更快）

### 子代理分发工具
- `dispatch_agent` - 分发任务给专门的子代理执行
  - subagent_type="Plan": 用于复杂任务的拆解和规划
  - subagent_type="Research": 用于信息搜索、论文查找、知识库检索
  - subagent_type="Analysis": 用于数据分析、报告生成

- `list_subagents` - 列出所有可用的子代理

### Plan/Task 管理工具
- `plan_get` - 获取指定计划的详细信息
- `plan_execute` - 执行指定计划
- `task_add` - 添加任务到计划
- `task_update` - 更新任务（描述、依赖、状态）
- `task_delete` - 删除任务
- `task_get` - 获取任务详情

### Skills 技能工具
- `list_skills` - 列出所有可用的 AI 技能（如天气查询等）
- `load_skills` - 加载指定技能到当前上下文（需要技能名称列表）
- `skill_call` - 调用已加载的技能执行实际任务
  - name: 技能名称（如 "weather"）
  - args_json: 技能参数的 JSON 字符串（如 \'{"city": "北京", "days": 3}\'）

## 决策规则

1. **简单对话**（问候、闲聊）：直接回复，不要调用任何工具
2. **简单搜索**（"搜索XXX"）：使用 web_search
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
        from tools.web import (
            web_search, web_fetch, web_scrape, web_crawl, web_map,
            arxiv_search, arxiv_get_by_id, arxiv_download_pdf,
        )
        from tools.agent import dispatch_agent, list_subagents
        from tools.skills import load_skills, list_skills, skill_call
        from tools.task import (
            plan_get, plan_execute,
            task_add, task_update, task_delete, task_get,
        )
        from tools.grep import grep, grep_count
        from tools.read import read
        from tools.write import write, append
        from tools.edit import edit, edit_regex
        from tools.bash import bash, bash_script
        from tools.glob import glob, glob_list

        self.tools = [
            # Web 搜索和内容提取
            web_search,
            web_fetch,
            web_scrape,
            web_crawl,
            web_map,
            # arXiv 学术搜索
            arxiv_search,
            arxiv_get_by_id,
            arxiv_download_pdf,
            # 子代理
            dispatch_agent,
            list_subagents,
            # Plan/Task 管理
            plan_get,
            plan_execute,
            task_add,
            task_update,
            task_delete,
            task_get,
            # Skills
            load_skills,
            list_skills,
            skill_call,
            # 文件搜索
            grep,
            grep_count,
            # 文件操作
            read,
            write,
            append,
            edit,
            edit_regex,
            # Shell
            bash,
            bash_script,
            # 文件查找
            glob,
            glob_list,
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
            from agent.session_store import SessionStore
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
        from langchain_core.messages import HumanMessage

        # 设置当前 thread_id 到 context（供工具调用时使用）
        _current_thread_id.set(thread_id)

        graph = self.graph

        # 确保会话存在
        self.session_store.get_or_create_session(thread_id)

        # 保存用户消息
        self.session_store.add_message(thread_id, "user", message)

        # 从 checkpointer 获取历史消息
        config = {"configurable": {"thread_id": thread_id}}
        existing_state = graph.get_state(config)
        existing_messages = list(existing_state.values.get("messages", [])) if existing_state else []

        # 如果 LangGraph 没有历史（重启后），从 session_store 恢复
        if not existing_messages:
            history = self.session_store.get_messages(thread_id)
            # 排除刚添加的用户消息
            history = [h for h in history if h['role'] != 'user' or h['content'] != message]
            for h in history:
                if h['role'] == 'user':
                    existing_messages.append(HumanMessage(content=h['content']))
                else:
                    existing_messages.append(AIMessage(content=h['content']))

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
                        self.session_store.add_message(thread_id, "assistant", accumulated_response)
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
                self.session_store.add_message(thread_id, "assistant", accumulated_response)
            existing_messages.append(
                AIMessage(content="[系统消息] 用户中断了执行。")
            )
            raise

        # 正常结束，保存最终响应
        if accumulated_response:
            existing_messages.append(AIMessage(content=accumulated_response))
            self.session_store.add_message(thread_id, "assistant", accumulated_response)

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
    from rich.table import Table
    from agent.registry import agent_registry, terminate
    from agent.worker import is_interrupted, clear_interrupt
    from agent.session_store import SessionStore

    console = Console()
    agent = create_main_agent()
    session_store = SessionStore()

    console.print(Panel.fit(
        "[bold green]Main Agent 已启动[/bold green]\n"
        "输入消息与 Agent 对话\n\n"
        "[dim]命令：[/dim]\n"
        "  [cyan]/exit[/cyan]      - 退出\n"
        "  [cyan]/clear[/cyan]     - 重置会话\n"
        "  [cyan]/status[/cyan]    - 查看状态\n"
        "  [cyan]/sessions[/cyan]  - 列出所有会话\n"
        "  [cyan]/resume <id>[/cyan] - 切换到指定会话\n"
        "  [cyan]/history[/cyan]   - 查看当前会话历史\n"
        "  [cyan]Ctrl+C[/cyan]     - 终止当前任务",
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

        # 显示当前会话
        current_session = session_store.get_session(thread_id)
        session_title = current_session.get("title", thread_id) if current_session else thread_id

        # 异步等待用户输入（使用 asyncio.to_thread 避免阻塞事件循环）
        try:
            user_input = await asyncio.to_thread(
                lambda: console.input(f"[bold blue]You[/bold blue] [dim]({session_title})[/dim]: ").strip()
            )
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C 或 Ctrl+D 在等待输入时
            user_input = ""

        if not user_input:
            # Ctrl+C 会导致 user_input 为空，这不算正常输入，继续循环即可
            continue

        if not user_input:
            continue

        # 处理命令
        cmd_lower = user_input.lower()

        if cmd_lower == "/exit":
            console.print("[bold yellow]再见！[/bold yellow]")
            break

        if cmd_lower == "/clear":
            thread_id = f"session_{int(__import__('time').time())}"
            console.print("[bold green]会话已重置[/bold green]\n")
            continue

        if cmd_lower == "/status":
            running = agent_registry.is_running()
            plan_ids = agent_registry.get_running_plan_ids()
            if running:
                console.print(f"[cyan]正在运行的任务: {plan_ids}[/cyan]\n")
            else:
                console.print("[dim]没有正在运行的任务[/dim]\n")
            continue

        if cmd_lower == "/sessions":
            sessions = session_store.list_sessions(limit=20)
            if not sessions:
                console.print("[dim]没有历史会话[/dim]\n")
                continue

            table = Table(title="历史会话")
            table.add_column("ID", style="cyan")
            table.add_column("标题")
            table.add_column("消息数", justify="right")
            table.add_column("更新时间")

            for s in sessions:
                # 格式化时间
                try:
                    dt = __import__('datetime').datetime.fromisoformat(s['updated_at'])
                    time_str = dt.strftime('%m-%d %H:%M')
                except:
                    time_str = s['updated_at'][:16]

                # 标记当前会话
                is_current = " [green]*[/green]" if s['session_id'] == thread_id else ""
                table.add_row(
                    s['session_id'] + is_current,
                    s['title'][:30],
                    str(s['message_count']),
                    time_str
                )

            console.print(table)
            console.print("[dim]使用 /resume <session_id> 切换会话[/dim]\n")
            continue

        if cmd_lower.startswith("/resume"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("[red]用法: /resume <session_id>[/red]\n")
                continue

            new_session_id = parts[1].strip()
            session = session_store.get_session(new_session_id)
            if not session:
                console.print(f"[red]会话不存在: {new_session_id}[/red]\n")
                continue

            thread_id = new_session_id

            # 回放历史对话
            messages = session_store.get_messages(thread_id)
            if messages:
                console.print(f"\n[bold green]已切换到会话: {session['title']}[/bold green]")
                console.print(f"[dim]历史消息 ({len(messages)} 条):[/dim]\n")

                for msg in messages:
                    role = "用户" if msg['role'] == 'user' else "助手"
                    role_color = "blue" if msg['role'] == 'user' else "green"
                    content = msg['content'] or ""
                    # 截断显示
                    if len(content) > 300:
                        content = content[:300] + "..."

                    console.print(f"[{role_color}]{role}[/{role_color}]: {content}\n")
            else:
                console.print(f"[bold green]已切换到会话: {session['title']}[/bold green] (无历史消息)\n")
            continue

        if cmd_lower == "/history":
            messages = session_store.get_messages(thread_id)
            if not messages:
                console.print("[dim]当前会话没有历史消息[/dim]\n")
                continue

            console.print(f"\n[bold]当前会话历史 ({len(messages)} 条消息):[/bold]\n")

            for msg in messages:
                role = "用户" if msg['role'] == 'user' else "助手"
                role_color = "blue" if msg['role'] == 'user' else "green"
                content = msg['content'] or ""

                # 时间
                try:
                    dt = __import__('datetime').datetime.fromisoformat(msg['timestamp'])
                    time_str = dt.strftime('%H:%M')
                except:
                    time_str = ""

                console.print(f"[dim][{time_str}][/dim] [{role_color}]{role}[/{role_color}]: {content}\n")
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
