"""命令注册与执行"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable, Union, Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from store.session import SessionStore

if TYPE_CHECKING:
    from agent.main.agent import MainAgent


@dataclass
class CommandContext:
    """命令执行的上下文"""
    console: Console
    session_store: SessionStore
    thread_id: str
    agent: "MainAgent"


@dataclass
class CommandResult:
    """命令执行结果"""
    new_thread_id: Optional[str] = None
    should_exit: bool = False
    error: Optional[str] = None


# 命令处理器签名（支持同步和异步）
CommandHandler = Callable[[str, CommandContext], Union[CommandResult, Awaitable[CommandResult]]]

# 命令注册表
_commands: dict[str, CommandHandler] = {}


def register_command(name: str, handler: CommandHandler) -> None:
    """注册命令处理器"""
    _commands[name.lower()] = handler


def get_command(name: str) -> CommandHandler | None:
    """获取命令处理器"""
    return _commands.get(name.lower())


def list_commands() -> list[str]:
    """列出所有已注册的命令"""
    return list(_commands.keys())


async def execute_command(input_str: str, ctx: CommandContext) -> CommandResult:
    """执行命令

    Args:
        input_str: 用户输入（以 / 开头）
        ctx: 命令执行上下文

    Returns:
        CommandResult: 命令执行结果
    """
    parts = input_str.split(maxsplit=1)
    cmd_name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    handler = get_command(cmd_name)
    if handler is None:
        return CommandResult(error=f"未定义的指令: {cmd_name}")

    result = handler(args, ctx)
    if asyncio.iscoroutine(result):
        result = await result
    return result


# ============ 内置命令实现 ============

def _cmd_exit(args: str, ctx: CommandContext) -> CommandResult:
    """退出 REPL"""
    ctx.console.print("[bold yellow]再见！[/bold yellow]")
    return CommandResult(should_exit=True)


def _cmd_new(args: str, ctx: CommandContext) -> CommandResult:
    """创建新会话"""
    import time
    new_thread_id = f"session_{int(time.time())}"
    ctx.session_store.create_session(new_thread_id)
    ctx.console.print("[bold green]已创建并进入新会话[/bold green]\n")
    return CommandResult(new_thread_id=new_thread_id)


def _cmd_delete(args: str, ctx: CommandContext) -> CommandResult:
    """删除指定会话"""
    if not args:
        ctx.console.print("[red]用法: /delete <session_id>[/red]\n")
        return CommandResult()

    del_session_id = args.strip()

    if del_session_id == ctx.thread_id:
        ctx.console.print("[red]不能删除当前会话，请先切换到其他会话[/red]\n")
        return CommandResult()

    session = ctx.session_store.get_session(del_session_id)
    if not session:
        ctx.console.print(f"[red]会话不存在: {del_session_id}[/red]\n")
        return CommandResult()

    ctx.session_store.delete_session(del_session_id)
    ctx.console.print(f"[yellow]已删除会话: {session['title']}[/yellow]\n")
    return CommandResult()


def _cmd_status(args: str, ctx: CommandContext) -> CommandResult:
    """查看运行状态"""
    from agent.core.registry import agent_registry
    running = agent_registry.is_running()
    plan_ids = agent_registry.get_running_plan_ids()
    if running:
        ctx.console.print(f"[cyan]正在运行的任务: {plan_ids}[/cyan]\n")
    else:
        ctx.console.print("[dim]没有正在运行的任务[/dim]\n")
    return CommandResult()


def _cmd_help(args: str, ctx: CommandContext) -> CommandResult:
    """显示帮助"""
    ctx.console.print(Panel.fit(
        "[bold green]Main Agent 帮助[/bold green]\n\n"
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
        title="帮助"
    ))
    ctx.console.print()
    return CommandResult()


def _cmd_sessions(args: str, ctx: CommandContext) -> CommandResult:
    """列出所有会话"""
    sessions = ctx.session_store.list_sessions(limit=20)
    if not sessions:
        ctx.console.print("[dim]没有历史会话[/dim]\n")
        return CommandResult()

    table = Table(title="历史会话")
    table.add_column("ID", style="cyan")
    table.add_column("标题")
    table.add_column("消息数", justify="right")
    table.add_column("更新时间")

    for s in sessions:
        try:
            dt = __import__('datetime').datetime.fromisoformat(s['updated_at'])
            time_str = dt.strftime('%m-%d %H:%M')
        except:
            time_str = s['updated_at'][:16]

        is_current = " [green]*[/green]" if s['session_id'] == ctx.thread_id else ""
        table.add_row(
            s['session_id'] + is_current,
            s['title'][:30],
            str(s['message_count']),
            time_str
        )

    ctx.console.print(table)
    ctx.console.print("[dim]使用 /resume <session_id> 切换会话[/dim]\n")
    return CommandResult()


def _cmd_resume(args: str, ctx: CommandContext) -> CommandResult:
    """切换到指定会话"""
    from agent.main.repl import _show_history

    if not args:
        ctx.console.print("[red]用法: /resume <session_id>[/red]\n")
        return CommandResult()

    new_session_id = args.strip()
    session = ctx.session_store.get_session(new_session_id)
    if not session:
        ctx.console.print(f"[red]会话不存在: {new_session_id}[/red]\n")
        return CommandResult()

    if not _show_history(ctx.console, ctx.session_store, new_session_id, show_timestamp=False):
        ctx.console.print(f"[bold green]已切换到会话: {session['title']}[/bold green] (无历史消息)\n")

    return CommandResult(new_thread_id=new_session_id)


def _cmd_history(args: str, ctx: CommandContext) -> CommandResult:
    """查看当前会话历史"""
    from agent.main.repl import _show_history

    if not _show_history(ctx.console, ctx.session_store, ctx.thread_id, show_timestamp=True):
        ctx.console.print("[dim]当前会话没有历史消息[/dim]\n")
    return CommandResult()


async def _cmd_compact(args: str, ctx: CommandContext) -> CommandResult:
    """压缩上下文"""
    from agent.middleware.context_compact import compact_session

    session = ctx.session_store.get_session(ctx.thread_id)
    total_tokens = session.get("total_tokens", 0) if session else 0
    ctx.console.print(f"[dim]压缩前: {total_tokens:,} tokens[/dim]")
    ctx.console.print("[cyan]正在生成摘要...[/cyan]")

    try:
        result = await compact_session(ctx.thread_id, ctx.agent.context_window)

        if result["success"]:
            saved = result["tokens_before"] - result["tokens_after"]
            ctx.console.print(f"\n[bold green]✓ 压缩完成[/bold green]")
            ctx.console.print(f"  - 移除消息: {result['messages_removed']} 条")
            ctx.console.print(f"  - 保留消息: {result['messages_kept']} 条")
            ctx.console.print(f"  - Token: {result['tokens_before']:,} → {result['tokens_after']:,} (节省 {saved:,})")
            ctx.console.print(f"\n[dim]摘要:[/dim]")
            summary = result.get("summary", "")
            if len(summary) > 500:
                summary = summary[:500] + "..."
            ctx.console.print(Panel(summary, border_style="dim"))
        else:
            ctx.console.print(f"[yellow]{result.get('message', '压缩失败')}[/yellow]\n")
    except Exception as e:
        ctx.console.print(f"[red]压缩失败: {e}[/red]\n")

    return CommandResult()


# 注册内置命令
register_command("/exit", _cmd_exit)
register_command("/new", _cmd_new)
register_command("/delete", _cmd_delete)
register_command("/status", _cmd_status)
register_command("/help", _cmd_help)
register_command("/sessions", _cmd_sessions)
register_command("/resume", _cmd_resume)
register_command("/history", _cmd_history)
register_command("/compact", _cmd_compact)
