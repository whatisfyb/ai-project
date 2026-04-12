"""REPL 交互循环"""

import asyncio
import signal
import threading

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.main.agent import create_main_agent
from agent.main.commands import CommandContext, execute_command
from agent.core.registry import terminate
from agent.core.signals import set_interrupt, clear_interrupt, is_interrupted
from store.session import SessionStore


# 模块级别的中断标志（用于 signal handler 和 async 函数之间共享）
_repl_interrupted = False


def _show_history(console: Console, session_store: SessionStore, thread_id: str, show_timestamp: bool = False):
    """显示会话历史消息

    Args:
        console: Rich Console
        session_store: 会话存储
        thread_id: 会话 ID
        show_timestamp: 是否显示时间戳
    """
    messages = session_store.get_messages(thread_id)
    if not messages:
        return False

    session = session_store.get_session(thread_id)
    session_title = session.get("title", thread_id) if session else thread_id

    console.print(f"\n[bold green]会话: {session_title}[/bold green]")
    console.print(f"[dim]历史消息 ({len(messages)} 条):[/dim]\n")

    for msg in messages:
        role = msg['role']
        content = msg['content'] or ""
        metadata = msg.get('metadata') or {}

        # 根据角色设置显示
        if role == 'user':
            role_display = "用户"
            role_color = "blue"
        elif role == 'tool':
            # 工具返回消息
            tool_name = metadata.get('name', 'tool')
            role_display = f"工具[{tool_name}]"
            role_color = "yellow"
        else:
            # assistant - 检查是否有工具调用
            tool_calls = metadata.get('tool_calls')
            if tool_calls:
                tool_names = [tc.get('name', '?') for tc in tool_calls]
                role_display = f"助手[调用: {', '.join(tool_names)}]"
            else:
                role_display = "助手"
            role_color = "green"

        # 截断显示
        if len(content) > 300:
            content = content[:300] + "..."

        if show_timestamp:
            try:
                dt = __import__('datetime').datetime.fromisoformat(msg['timestamp'])
                time_str = dt.strftime('%H:%M')
            except:
                time_str = ""
            console.print(f"[dim][{time_str}][/dim] [{role_color}]{role_display}[/{role_color}]: {content}\n")
        else:
            console.print(f"[{role_color}]{role_display}[/{role_color}]: {content}\n")

    return True


def run_repl():
    """运行 REPL 交互（同步入口，内部启动 async 循环）"""
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

    console = Console()
    agent = create_main_agent()
    session_store = SessionStore()

    # 异步初始化 MCP 服务器（后台连接，不阻塞）
    try:
        from tools.mcp import init_mcp_from_config_async, get_mcp_manager
        await init_mcp_from_config_async(background=True)
        manager = get_mcp_manager()
        servers = manager.list_servers()
        connecting_count = sum(1 for s in servers if s.get("connecting"))
        if connecting_count > 0:
            console.print(f"[dim]正在后台连接 {connecting_count} 个 MCP 服务器...[/dim]\n")
    except Exception as e:
        console.print(f"[dim]MCP 初始化: {e}[/dim]\n")

    console.print(Panel.fit(
        "[bold green]Main Agent 已启动[/bold green]\n"
        "输入消息与 Agent 对话\n\n"
        "[dim]命令：[/dim]\n"
        "  [cyan]/new[/cyan]         - 创建并进入新会话\n"
        "  [cyan]/delete <id>[/cyan] - 删除指定会话\n"
        "  [cyan]/sessions[/cyan]    - 列出所有会话\n"
        "  [cyan]/resume <id>[/cyan] - 切换到指定会话\n"
        "  [cyan]/history[/cyan]     - 查看当前会话历史\n"
        "  [cyan]/compact[/cyan]     - 压缩上下文\n"
        "  [cyan]/status[/cyan]      - 查看运行状态\n"
        "  [cyan]/help[/cyan]        - 显示帮助\n"
        "  [cyan]/exit[/cyan]        - 退出\n"
        "  [cyan]Ctrl+C[/cyan]       - 终止当前任务",
        title="Main Agent"
    ))

    # 启动时进入最后一次会话（如果没有则创建新会话）
    sessions = session_store.list_sessions(limit=1)
    if sessions:
        thread_id = sessions[0]['session_id']
        # 重新计算 token 数（配置可能已变更）
        from agent.middleware.token_count import recalculate_session_tokens
        recalculate_session_tokens(thread_id)
        _show_history(console, session_store, thread_id, show_timestamp=False)
    else:
        thread_id = f"session_{int(__import__('time').time())}"
        session_store.create_session(thread_id)
        console.print("[dim]已创建新会话[/dim]\n")

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

        # 显示当前会话和 token 状态
        current_session = session_store.get_session(thread_id)
        session_title = current_session.get("title", thread_id) if current_session else thread_id
        total_tokens = current_session.get("total_tokens", 0) if current_session else 0

        # 从配置获取上下文状态
        from utils.config import get_default_model_config, check_context_status
        model_config = get_default_model_config()
        context_status = check_context_status(total_tokens, model_config)

        # 计算用量百分比
        usage_ratio = round(context_status.percent_used, 1)
        token_info = f"[dim]({usage_ratio}%)[/dim]"

        # 异步等待用户输入（使用 asyncio.to_thread 避免阻塞事件循环）
        try:
            user_input = await asyncio.to_thread(
                lambda: console.input(f"[bold blue]You[/bold blue] [dim]({session_title})[/dim] {token_info}: ").strip()
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
        if user_input.startswith("/"):
            ctx = CommandContext(
                console=console,
                session_store=session_store,
                thread_id=thread_id,
                agent=agent,
            )
            result = await execute_command(user_input, ctx)

            if result.error:
                console.print(f"[red]{result.error}[/red]\n")
                continue

            if result.should_exit:
                break

            if result.new_thread_id:
                thread_id = result.new_thread_id

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
