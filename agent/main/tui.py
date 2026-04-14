"""Textual TUI for Main Agent"""

import asyncio
import threading
from datetime import datetime
from typing import Optional
from collections import OrderedDict

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Static, Input, RichLog, Button
from textual.reactive import reactive

from agent.main.agent import create_main_agent
from agent.main.command import CommandHandler, execute_tui_command
from agent.core.registry import agent_registry, terminate
from agent.core.signals import set_interrupt, clear_interrupt, is_interrupted
from store.session import SessionStore
from agent.a2a.dispatcher import get_inbox, get_agent_state, TaskResultStatus


# ============ Worker 状态追踪 ============

class WorkerStatusTracker:
    """Worker 状态追踪器（全局单例）"""

    _instance: Optional["WorkerStatusTracker"] = None

    def __init__(self):
        self._workers: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._callbacks: list = []

    @classmethod
    def get_instance(cls) -> "WorkerStatusTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_running(self, worker_id: str, task_id: str, description: str, plan_id: str = ""):
        with self._lock:
            self._workers[worker_id] = {
                "status": "running",
                "task_id": task_id,
                "description": description,
                "plan_id": plan_id,
                "started_at": datetime.now(),
            }
        self._notify_callbacks()

    def set_done(self, worker_id: str, success: bool = True):
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id]["status"] = "done" if success else "failed"
                self._workers[worker_id]["completed_at"] = datetime.now()
        self._notify_callbacks()

    def clear_done(self, worker_id: str):
        with self._lock:
            if worker_id in self._workers and self._workers[worker_id]["status"] in ("done", "failed"):
                del self._workers[worker_id]
        self._notify_callbacks()

    def get_all(self) -> dict:
        with self._lock:
            return dict(self._workers)

    def subscribe(self, callback):
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        for cb in self._callbacks:
            try:
                cb()
            except:
                pass


def get_worker_tracker() -> WorkerStatusTracker:
    return WorkerStatusTracker.get_instance()


# ============ TUI App ============

class MainAgentTUI(App):
    """Main Agent TUI"""

    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-log {
        height: 1fr;
        border: solid green;
        margin: 1;
        overflow-x: hidden;
        overflow-y: auto;
    }

    #worker-panel {
        height: auto;
        display: none;
        margin: 0 1;
        padding: 1;
        background: $surface;
        border: solid yellow;
    }

    #input-row {
        height: 3;
        margin: 1;
    }

    #input {
        width: 1fr;
    }

    #action-btn {
        width: 8;
        margin-left: 1;
    }

    #status {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "interrupt", "中断"),
        ("ctrl+l", "clear", "清屏"),
        ("ctrl+d", "quit", "退出"),
    ]

    # Reactive 状态
    session_title = reactive("")
    token_percent = reactive(0.0)
    inbox_count = reactive(0)
    running_count = reactive(0)

    def __init__(self):
        super().__init__()
        self.agent = create_main_agent()
        self.session_store = SessionStore()
        self.inbox = get_inbox()
        self.agent_state = get_agent_state()
        self.worker_tracker = get_worker_tracker()
        self.thread_id: str = ""
        self._command_handler: Optional[CommandHandler] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield RichLog(id="chat-log", wrap=True)
        yield Static(id="worker-panel")
        with Horizontal(id="input-row"):
            yield Input(placeholder="输入消息，/help 查看命令...", id="input")
            yield Button("发送", id="action-btn", variant="primary")
        yield Static(" idle │ inbox: 0 │ workers: 0", id="status")

    def on_mount(self) -> None:
        """启动时初始化"""
        sessions = self.session_store.list_sessions(limit=1)
        if sessions:
            self.thread_id = sessions[0]["session_id"]
            from agent.middleware.token_count import recalculate_session_tokens
            recalculate_session_tokens(self.thread_id)
            self._show_history()
        else:
            import time
            self.thread_id = f"session_{int(time.time())}"
            self.session_store.create_session(self.thread_id)

        self._update_status()
        self.worker_tracker.subscribe(self._on_worker_change)
        asyncio.create_task(self._background_checker())
        asyncio.create_task(self._init_mcp())

        # 初始化命令处理器
        self._command_handler = CommandHandler(self)

        self.query_one("#input", Input).focus()

    async def _init_mcp(self):
        try:
            from tools.mcp import init_mcp_from_config_async
            await init_mcp_from_config_async(background=True)
        except:
            pass

    async def _background_checker(self):
        while True:
            await asyncio.sleep(0.5)
            try:
                self.inbox_count = self.inbox.size()
                workers = self.worker_tracker.get_all()
                self.running_count = sum(1 for w in workers.values() if w["status"] == "running")
                self._update_status()
                self._update_worker_panel()

                if self.agent_state.is_idle() and not self.inbox.is_empty():
                    results = self.inbox.get_all()
                    if results:
                        await self._handle_inbox(results)
            except:
                pass

    def _on_worker_change(self):
        self.call_from_thread(self._update_worker_panel)

    def _update_status(self):
        session = self.session_store.get_session(self.thread_id)
        title = (session.get("title", self.thread_id) if session else self.thread_id)[:15]
        tokens = session.get("total_tokens", 0) if session else 0

        from utils.config import get_default_model_config, check_context_status
        model_config = get_default_model_config()
        context_status = check_context_status(tokens, model_config)

        self.session_title = title
        self.token_percent = round(context_status.percent_used, 1)

        busy = self.is_busy
        status_text = f" {'[yellow]busy[/yellow]' if busy else '[green]idle[/green]'} │ inbox: {self.inbox_count} │ workers: {self.running_count} │ {self.session_title} ({self.token_percent}%)"
        self.query_one("#status", Static).update(status_text)

    def _update_worker_panel(self):
        workers = self.worker_tracker.get_all()
        panel = self.query_one("#worker-panel", Static)

        now = datetime.now()
        for wid in list(workers.keys()):
            w = workers[wid]
            if w["status"] in ("done", "failed"):
                elapsed = (now - w.get("completed_at", now)).total_seconds()
                if elapsed > 3:
                    self.worker_tracker.clear_done(wid)

        workers = self.worker_tracker.get_all()

        if not workers:
            panel.display = False
            return

        lines = []
        for wid, w in workers.items():
            icon = {"running": "[yellow]●[/]", "done": "[green]✓[/]", "failed": "[red]✗[/]"}.get(w["status"], "○")
            desc = w.get("description", w.get("task_id", "?"))[:40]
            lines.append(f" {icon} [{wid}] {desc}")

        panel.update("\n".join(lines))
        panel.display = True

    def _append_chat(self, text: str):
        """追加文本到聊天区"""
        try:
            log = self.query_one("#chat-log", RichLog)
            log.write(text)
        except:
            pass

    def _clear_chat(self):
        """清空聊天"""
        try:
            self.query_one("#chat-log", RichLog).clear()
        except:
            pass

    def _show_history(self):
        messages = self.session_store.get_messages(self.thread_id)
        for msg in messages:
            role = msg["role"]
            content = msg.get("content") or ""
            if role == "user":
                self._append_chat(f"You: {content}")
            elif role == "assistant":
                self._append_chat(f"Agent: {content[:500]}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "input":
            return

        text = event.value.strip()
        if not text:
            return

        event.input.value = ""
        self._append_chat(f"You: {text}")

        if text.startswith("/"):
            asyncio.create_task(self._handle_command(text))
        else:
            asyncio.create_task(self._call_agent(text))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮点击处理"""
        if event.button.id != "action-btn":
            return

        if self.is_busy:
            # 中断
            self._do_interrupt()
        else:
            # 发送输入框内容
            input_widget = self.query_one("#input", Input)
            text = input_widget.value.strip()
            if not text:
                return

            input_widget.value = ""
            self._append_chat(f"You: {text}")

            if text.startswith("/"):
                asyncio.create_task(self._handle_command(text))
            else:
                asyncio.create_task(self._call_agent(text))

    @property
    def is_busy(self) -> bool:
        """是否正在运行"""
        return not self.agent_state.is_idle()

    def _update_action_button(self):
        """更新按钮状态"""
        try:
            btn = self.query_one("#action-btn", Button)
            if self.is_busy:
                btn.label = "中断"
                btn.variant = "error"
            else:
                btn.label = "发送"
                btn.variant = "primary"
        except:
            pass

    def _do_interrupt(self):
        """执行中断"""
        # 1. 设置中断信号（MainAgent.chat_async 会检查）
        set_interrupt()

        # 2. 终止 Plan 执行器（如果有）
        if agent_registry.is_running():
            terminated = terminate()
            if terminated:
                self._append_chat(f"已终止任务: {', '.join(terminated)}")

        self._append_chat("已中断")
        self.agent_state.set_idle()
        self._update_status()
        self._update_action_button()

    async def _handle_command(self, cmd: str):
        """处理命令"""
        try:
            result = await execute_tui_command(cmd, self._command_handler)

            if result.get("should_exit"):
                self.exit()

            new_thread_id = result.get("new_thread_id")
            if new_thread_id:
                self.thread_id = new_thread_id
                self._command_handler.update_thread_id(new_thread_id)
                self._update_status()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._append_chat(f"错误: {e}")

    async def _call_agent(self, text: str):
        self._append_chat("Agent: ")

        self.agent_state.set_busy()
        self._update_status()
        self._update_action_button()

        def on_token(token):
            # 直接写入，实现流式效果
            try:
                log = self.query_one("#chat-log", RichLog)
                log.write(token)
            except:
                pass

        try:
            await self.agent.chat_async(text, self.thread_id, on_token=on_token)
        except Exception as e:
            self._append_chat(f"错误: {e}")
        finally:
            # 清除中断信号
            clear_interrupt()
            self.agent_state.set_idle()
            self._update_status()
            self._update_action_button()

    async def _handle_inbox(self, results):
        self._append_chat("收到 Worker 任务结果...")

        lines = ["以下任务已完成："]
        for r in results:
            icon = "✓" if r.status == TaskResultStatus.SUCCESS else "✗"
            lines.append(f"  {icon} {r.task_id}: {(r.result or r.error or '?')[:100]}")
        self._append_chat("\n".join(lines))

        prompt = "\n".join(lines) + "\n\n请分析结果并决定下一步。"

        self.agent_state.set_busy()
        self._update_status()
        self._update_action_button()

        def on_token(token):
            try:
                log = self.query_one("#chat-log", RichLog)
                log.write(token)
            except:
                pass

        try:
            await self.agent.chat_async(prompt, self.thread_id, on_token=on_token)
        finally:
            clear_interrupt()
            self.agent_state.set_idle()
            self._update_status()
            self._update_action_button()

    def action_interrupt(self):
        if agent_registry.is_running():
            terminate()
            self._append_chat("已中断")
        clear_interrupt()

    def action_clear(self):
        self._clear_chat()


def run_tui():
    app = MainAgentTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
