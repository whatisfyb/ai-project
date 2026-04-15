"""TUI 命令处理

为 TUI 提供命令执行逻辑，直接操作 TUI 组件。
"""

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.main.tui import MainAgentTUI


class CommandHandler:
    """TUI 命令处理器"""

    def __init__(self, app: "MainAgentTUI"):
        self.app = app
        self.session_store = app.session_store
        self.thread_id = app.thread_id
        self.agent = app.agent

    def update_thread_id(self, thread_id: str):
        """更新当前会话 ID"""
        self.thread_id = thread_id

    def _append(self, text: str):
        """追加文本到聊天区"""
        self.app._append_chat(text)

    def _update_status(self):
        """更新状态栏"""
        self.app._update_status()

    # ============ 命令实现 ============

    def cmd_exit(self, args: str) -> dict:
        """退出"""
        self._append("再见！")
        return {"should_exit": True}

    def cmd_new(self, args: str) -> dict:
        """创建新会话"""
        new_thread_id = f"session_{int(time.time())}"
        self.session_store.create_session(new_thread_id)
        self._append("已创建并进入新会话")
        return {"new_thread_id": new_thread_id}

    def cmd_delete(self, args: str) -> dict:
        """删除指定会话"""
        if not args:
            self._append("用法: /delete <session_id>")
            return {}

        del_session_id = args.strip()

        if del_session_id == self.thread_id:
            self._append("不能删除当前会话，请先切换到其他会话")
            return {}

        session = self.session_store.get_session(del_session_id)
        if not session:
            self._append(f"会话不存在: {del_session_id}")
            return {}

        self.session_store.delete_session(del_session_id)
        self._append(f"已删除会话: {session['title']}")
        return {}

    def cmd_status(self, args: str) -> dict:
        """查看运行状态"""
        from agent.core.registry import agent_registry
        running = agent_registry.is_running()
        plan_ids = agent_registry.get_running_plan_ids()
        if running:
            self._append(f"正在运行的任务: {plan_ids}")
        else:
            self._append("没有正在运行的任务")
        return {}

    def cmd_help(self, args: str) -> dict:
        """显示帮助"""
        help_text = """Main Agent 帮助

命令：
  /new           - 创建并进入新会话
  /delete <id>   - 删除指定会话
  /sessions      - 列出所有会话
  /resume <id>   - 切换到指定会话
  /history       - 查看当前会话历史
  /compact       - 压缩上下文
  /status        - 查看运行状态
  /help          - 显示帮助
  /exit          - 退出
  Ctrl+C         - 终止当前任务"""
        self._append(help_text)
        return {}

    def cmd_sessions(self, args: str) -> dict:
        """列出所有会话"""
        sessions = self.session_store.list_sessions(limit=20)
        if not sessions:
            self._append("没有历史会话")
            return {}

        lines = ["历史会话:"]
        for s in sessions:
            try:
                dt = __import__('datetime').datetime.fromisoformat(s['updated_at'])
                time_str = dt.strftime('%m-%d %H:%M')
            except:
                time_str = s['updated_at'][:16]

            is_current = " *" if s['session_id'] == self.thread_id else ""
            lines.append(f"  {s['session_id']}{is_current} - {s['title'][:30]} ({s['message_count']}条, {time_str})")

        lines.append("\n使用 /resume <session_id> 切换会话")
        self._append("\n".join(lines))
        return {}

    def cmd_resume(self, args: str) -> dict:
        """切换到指定会话"""
        if not args:
            self._append("用法: /resume <session_id>")
            return {}

        new_session_id = args.strip()
        session = self.session_store.get_session(new_session_id)
        if not session:
            self._append(f"会话不存在: {new_session_id}")
            return {}

        # 显示历史
        messages = self.session_store.get_messages(new_session_id)
        if messages:
            self._append(f"\n会话: {session['title']}")
            self._append(f"历史消息 ({len(messages)} 条):")
            for msg in messages[:10]:
                role = msg['role']
                content = (msg.get('content') or "")[:100]
                if role == 'user':
                    self._append(f"  用户: {content}")
                elif role == 'assistant':
                    self._append(f"  助手: {content}")
        else:
            self._append(f"已切换到会话: {session['title']} (无历史消息)")

        return {"new_thread_id": new_session_id}

    def cmd_history(self, args: str) -> dict:
        """查看当前会话历史"""
        messages = self.session_store.get_messages(self.thread_id)
        if not messages:
            self._append("当前会话没有历史消息")
            return {}

        session = self.session_store.get_session(self.thread_id)
        title = session.get("title", self.thread_id) if session else self.thread_id

        self._append(f"\n会话: {title}")
        self._append(f"历史消息 ({len(messages)} 条):")

        for msg in messages:
            role = msg['role']
            content = msg.get('content') or ""
            metadata = msg.get('metadata') or {}

            if role == 'user':
                self._append(f"  用户: {content[:200]}")
            elif role == 'tool':
                tool_name = metadata.get('name', 'tool')
                self._append(f"  工具[{tool_name}]: {content[:100]}")
            else:
                tool_calls = metadata.get('tool_calls')
                if tool_calls:
                    tool_names = [tc.get('name', '?') for tc in tool_calls]
                    self._append(f"  助手[调用: {', '.join(tool_names)}]: {content[:100]}")
                else:
                    self._append(f"  助手: {content[:200]}")

        return {}

    async def cmd_compact(self, args: str) -> dict:
        """压缩上下文"""
        from agent.middleware.context_compact import compact_session

        session = self.session_store.get_session(self.thread_id)
        total_tokens = session.get("total_tokens", 0) if session else 0
        self._append(f"压缩前: {total_tokens:,} tokens")
        self._append("正在压缩...")

        try:
            result = await compact_session(self.thread_id, self.agent.context_window)

            if result["success"]:
                saved = result["tokens_before"] - result["tokens_after"]
                self._append(f"✓ 压缩完成")
                self._append(f"  - 移除消息: {result['messages_removed']} 条")
                self._append(f"  - 保留消息: {result['messages_kept']} 条")
                self._append(f"  - Token: {result['tokens_before']:,} → {result['tokens_after']:,} (节省 {saved:,})")

                # 显示微压缩统计
                micro = result.get("micro_compact")
                if micro and micro.get("tools_cleared", 0) > 0:
                    self._append(f"  - 微压缩: 清理 {micro['tools_cleared']} 个旧工具结果, 节省 {micro['tokens_saved']:,} tokens")

                summary = result.get("summary", "")
                if summary and len(summary) > 100:
                    summary = summary[:100] + "..."
                    self._append(f"\n摘要预览: {summary}")
            else:
                self._append(result.get('message', '压缩失败'))
        except Exception as e:
            self._append(f"压缩失败: {e}")

        return {}


# 命令映射
COMMANDS = {
    "/exit": "cmd_exit",
    "/new": "cmd_new",
    "/delete": "cmd_delete",
    "/status": "cmd_status",
    "/help": "cmd_help",
    "/sessions": "cmd_sessions",
    "/resume": "cmd_resume",
    "/history": "cmd_history",
    "/compact": "cmd_compact",
}


async def execute_tui_command(cmd: str, handler: CommandHandler) -> dict:
    """执行 TUI 命令"""
    parts = cmd.split(maxsplit=1)
    cmd_name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    method_name = COMMANDS.get(cmd_name)
    if not method_name:
        handler._append(f"未定义的指令: {cmd_name}")
        return {"error": f"未定义的指令: {cmd_name}"}

    method = getattr(handler, method_name, None)
    if not method:
        handler._append(f"命令未实现: {cmd_name}")
        return {"error": f"命令未实现: {cmd_name}"}

    import asyncio
    result = method(args)
    if asyncio.iscoroutine(result):
        result = await result

    return result if result else {}
