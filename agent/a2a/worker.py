"""A2A Worker Agent - 基于 A2A 协议的任务执行器

Worker 作为 A2A Agent，接收 MainAgent 分发的任务并异步执行。
支持：
- 注册到 Transport
- 接收 A2A Task
- 执行 PlanTask
- 状态回传
- 继承 BaseAgent 标准接口
"""

import queue
import threading
import uuid
from typing import Any, Callable
from datetime import datetime

from agent.a2a.models import (
    Task,
    TaskStatus,
    TaskEvent,
    Message,
    MessageRole,
    Part,
    PartType,
    Artifact,
    AgentCard,
    AgentCapabilities,
    Skill,
)
from agent.a2a.transport import InMemoryTransport, TaskCallback
from agent.core.signals import is_interrupted, save_checkpoint
from agent.core.base_agent import BaseAgent


# ============ Worker 工具获取 ============


def _get_default_tools():
    """获取 Worker 默认可用工具

    Worker 可用的工具集：
    - web: Web 搜索、抓取、arXiv
    - paper_kb: 论文知识库
    - read: 文件读取
    - write: 文件写入
    - grep: 文件内容搜索
    - glob: 文件搜索
    - bash: Shell 命令执行
    """
    from tools.web import web
    from tools.paper_kb import paper_kb
    from tools.read import read
    from tools.write import write, append
    from tools.edit import edit, edit_regex
    from tools.grep import grep, grep_count
    from tools.glob import glob, glob_list
    from tools.bash import bash, bash_script

    return [
        # Web 和知识库
        web,
        paper_kb,
        # 文件操作
        read,
        write,
        append,
        edit,
        edit_regex,
        # 搜索
        grep,
        grep_count,
        glob,
        glob_list,
        # Shell
        bash,
        bash_script,
    ]


# ============ A2A Worker Agent ============


class A2AWorker(BaseAgent):
    """A2A Worker Agent

    作为 A2A Agent 注册到 Transport，接收并执行任务。
    继承 BaseAgent 标准接口，支持 Registry 生命周期管理。

    特点：
    - 异步执行：任务放入队列，后台线程处理
    - 非阻塞：handler 立即返回，不阻塞调用方
    - 状态回传：执行完成后更新 Task 状态
    - 支持订阅：MainAgent 可订阅任务完成事件
    - 状态保存/恢复：支持检查点

    使用方式：
        transport = InMemoryTransport()
        worker = A2AWorker(worker_id="worker-1", transport=transport)
        worker.start()

        # 之后 MainAgent 可以通过 transport.send_message() 发送任务
    """

    agent_type = "worker"

    def __init__(
        self,
        worker_id: str | None = None,
        transport: InMemoryTransport | None = None,
        tools: list | None = None,
        max_iterations: int = 10,
        task_timeout: int = 300,
    ):
        """初始化 Worker

        Args:
            worker_id: Worker 唯一标识，默认自动生成
            transport: Transport 实例，默认使用全局实例
            tools: 可用工具列表，默认使用内置工具
            max_iterations: LLM 工具调用最大迭代次数
            task_timeout: 单任务超时时间（秒）
        """
        self.agent_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.worker_id = self.agent_id
        self.transport = transport or self._get_default_transport()
        self.tools = tools or _get_default_tools()
        self.max_iterations = max_iterations
        self.task_timeout = task_timeout

        # 任务队列
        self._task_queue: queue.Queue[tuple[Task, Message] | None] = queue.Queue()

        # 工作线程
        self._worker_thread: threading.Thread | None = None
        self._running = False

        # 当前执行状态
        self._current_task: Task | None = None
        self._completed_count = 0
        self._failed_count = 0

        # Agent Card
        self._card = self._build_card()

    def _get_default_transport(self) -> InMemoryTransport:
        """获取默认 Transport"""
        from agent.a2a.transport import get_transport

        return get_transport()

    def _build_card(self) -> AgentCard:
        """构建 Agent Card"""
        skills = [
            # 核心能力
            Skill(name="execute_plantask", description="执行 PlanTask 任务"),
            # Web 和知识库
            Skill(name="web", description="Web 搜索、抓取、arXiv 论文搜索"),
            Skill(name="paper_kb", description="论文知识库查询"),
            # 文件操作
            Skill(name="read", description="读取文件内容"),
            Skill(name="write", description="写入文件"),
            Skill(name="edit", description="编辑文件"),
            # 搜索
            Skill(name="grep", description="文件内容搜索"),
            Skill(name="glob", description="文件名搜索"),
            # Shell
            Skill(name="bash", description="执行 Shell 命令"),
        ]

        return AgentCard(
            id=self.worker_id,
            name=f"Worker-{self.worker_id}",
            description="任务执行器，接收并执行 A2A Task，支持 Web 搜索、文件操作、Shell 命令等",
            capabilities=AgentCapabilities(
                text=True,
                files=True,
                streaming=False,
                push_notifications=True,  # 支持状态推送
            ),
            skills=skills,
        )

    @property
    def card(self) -> AgentCard:
        """获取 Agent Card"""
        return self._card

    @property
    def is_running(self) -> bool:
        """Worker 是否正在运行"""
        return (
            self._running
            and self._worker_thread is not None
            and self._worker_thread.is_alive()
        )

    @property
    def current_task(self) -> Task | None:
        """当前正在执行的任务"""
        return self._current_task

    @property
    def stats(self) -> dict:
        """执行统计"""
        return {
            "worker_id": self.worker_id,
            "running": self.is_running,
            "completed": self._completed_count,
            "failed": self._failed_count,
            "current_task": self._current_task.id if self._current_task else None,
        }

    # ============ BaseAgent 接口实现 ============

    def get_card(self) -> AgentCard:
        """返回 Agent 能力声明"""
        return self._card

    def handle_task(self, task: Task) -> Any:
        """处理 A2A Task

        Args:
            task: A2A Task 对象

        Returns:
            任务执行结果
        """
        if not task.history:
            return {"success": False, "error": "Empty task"}

        message = task.history[0]
        self._handle_message(task, message)
        return {"success": True, "task_id": task.id}

    def get_state(self) -> dict[str, Any]:
        """获取当前状态（用于检查点保存）"""
        return {
            "worker_id": self.worker_id,
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            "current_task_id": self._current_task.id if self._current_task else None,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """恢复状态（从检查点恢复）"""
        self._completed_count = state.get("completed_count", 0)
        self._failed_count = state.get("failed_count", 0)

    def on_interrupt(self) -> None:
        """中断回调 - 保存检查点"""
        state = self.get_state()
        save_checkpoint(self.agent_id, state)

    # ============ 生命周期 ============

    def start(self) -> None:
        """启动 Worker

        注册到 Transport 并启动工作线程。
        """
        if self._running:
            return

        self._running = True

        # 注册到 Transport
        self.transport.register_agent(
            agent_id=self.worker_id,
            card=self._card,
            handler=self._handle_message,
        )

        # 启动工作线程
        self._worker_thread = threading.Thread(
            target=self._work_loop,
            name=f"A2AWorker-{self.worker_id}",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """停止 Worker

        等待当前任务完成后退出。
        """
        if not self._running:
            return

        self._running = False

        # 发送终止信号
        self._task_queue.put(None)

        # 等待线程结束
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)

        # 从 Transport 注销
        self.transport.unregister_agent(self.worker_id)

    def stop_nowait(self) -> None:
        """立即停止（不等待当前任务）"""
        self._running = False
        self._task_queue.put(None)
        self.transport.unregister_agent(self.worker_id)

    # ============ 消息处理 ============

    def _handle_message(self, task: Task, message: Message) -> None:
        """Transport 消息处理器

        这是注册到 Transport 的 handler，会被 send_message() 调用。
        handler 应该是非阻塞的，所以将任务放入队列。
        """
        if not self._running:
            # Worker 未启动，拒绝任务
            self.transport.update_task_status(task.id, TaskStatus.REJECTED)
            return

        # 放入队列，异步处理
        self._task_queue.put((task, message))

    # ============ 工作循环 ============

    def _work_loop(self) -> None:
        """工作线程主循环"""
        while self._running:
            # 检查中断
            if is_interrupted():
                break

            try:
                # 等待任务
                item = self._task_queue.get(timeout=1)

                if item is None:
                    # 终止信号
                    break

                task, message = item

                # 再次检查中断（任务开始前）
                if is_interrupted():
                    break

                # 执行任务
                self._execute_task(task, message)

            except queue.Empty:
                # 超时，继续等待
                continue
            except Exception as e:
                # 意外错误
                import traceback

                traceback.print_exc()

    def _execute_task(self, task: Task, message: Message) -> None:
        """执行单个任务"""
        self._current_task = task

        # 执行结果（用于 finally 中报告）
        final_status: TaskStatus = TaskStatus.WORKING
        final_result: str | None = None
        final_error: str | None = None

        # 获取任务描述（用于 UI 显示）
        plantask_info = self._extract_plantask_info(task, message)
        task_description = ""
        if plantask_info:
            # 从 PlanStore 获取任务描述
            from store.plan import PlanStore

            store = PlanStore()
            plan_id = plantask_info.get("plan_id", "")
            task_id = plantask_info.get("task_id", "")
            if plan_id and task_id:
                plan = store.load_plan(plan_id)
                if plan:
                    for t in plan.tasks:
                        if t.id == task_id:
                            task_description = t.description
                            break

        # 报告状态：开始运行
        from agent.main.tui import get_worker_tracker

        tracker = get_worker_tracker()
        tracker.set_running(
            worker_id=self.worker_id,
            task_id=task.plantask_id or task.id[:8],
            description=task_description or "执行任务",
            plan_id=task.plan_id or "",
        )

        try:
            # 更新状态为 WORKING
            self.transport.update_task_status(task.id, TaskStatus.WORKING)

            if plantask_info:
                # 执行 PlanTask
                result = self._execute_plantask(plantask_info, task)
            else:
                # 执行普通文本任务
                result = self._execute_text_task(message, task)

            # 检查是否被中断
            if is_interrupted():
                # 被中断，更新状态
                self.transport.update_task_status(task.id, TaskStatus.CANCELLED)
                final_status = TaskStatus.CANCELLED
                final_result = result
                final_error = "用户中断"
                return

            # 添加结果消息
            result_msg = Message.agent_text(result)
            self.transport.add_task_message(task.id, result_msg)

            # 更新状态为完成
            self.transport.update_task_status(task.id, TaskStatus.COMPLETED)
            self._completed_count += 1

            final_status = TaskStatus.COMPLETED
            final_result = result

        except Exception as e:
            # 执行失败
            error_msg = f"执行失败: {str(e)}"
            self.transport.add_task_message(task.id, Message.agent_text(error_msg))
            self.transport.update_task_status(task.id, TaskStatus.FAILED)
            self._failed_count += 1

            final_status = TaskStatus.FAILED
            final_error = str(e)

        finally:
            self._current_task = None

            # 无论如何都报告任务结果到 inbox
            self._report_task_result(task, final_status, final_result, final_error)

            # 报告状态：完成/失败
            tracker.set_done(
                self.worker_id, success=(final_status == TaskStatus.COMPLETED)
            )

    def _report_task_result(
        self,
        task: Task,
        status: TaskStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """报告任务结果到邮箱"""
        from agent.a2a.dispatcher import (
            get_inbox,
            TaskResult,
            TaskResultStatus,
        )

        inbox = get_inbox()

        task_result = TaskResult(
            plan_id=task.plan_id,
            task_id=task.plantask_id,
            status=TaskResultStatus.SUCCESS
            if status == TaskStatus.COMPLETED
            else TaskResultStatus.FAILED,
            result=result,
            error=error,
            job_id=task.id,
            worker_id=self.worker_id,
        )

        inbox.put(task_result)

    def _extract_plantask_info(self, task: Task, message: Message) -> dict | None:
        """从 Task/Message 中提取 PlanTask 信息"""
        # 优先从 Task 元数据获取
        if task.plan_id and task.plantask_id:
            return {
                "plan_id": task.plan_id,
                "task_id": task.plantask_id,
            }

        # 从 Message parts 中查找
        for part in message.parts:
            if part.type == PartType.PLANTASK:
                return part.content

        return None

    def _execute_plantask(self, plantask_info: dict, a2a_task: Task) -> str:
        """执行 PlanTask

        Args:
            plantask_info: {"plan_id": str, "task_id": str}
            a2a_task: A2A Task

        Returns:
            执行结果
        """
        from store.plan import PlanStore
        from utils.core.llm import get_llm_model

        plan_id = plantask_info["plan_id"]
        task_id = plantask_info["task_id"]

        # 从数据库加载 PlanTask
        store = PlanStore()
        plan = store.load_plan(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        # 查找对应的 PlanTask
        plantask = None
        for t in plan.tasks:
            if t.id == task_id:
                plantask = t
                break

        if not plantask:
            raise ValueError(f"PlanTask not found: {task_id}")

        # 领取任务（原子操作）
        claimed = store.claim_task(plan_id, self.worker_id)
        if not claimed or claimed.id != task_id:
            # 任务已被其他 Worker 领取
            return f"任务 {task_id} 已被其他 Worker 处理"

        # 构建 LLM 调用
        llm = get_llm_model()
        llm_with_tools = llm.bind_tools(self.tools)

        task_prompt = f"""你是一个任务执行者。请完成以下任务：

任务描述：{plantask.description}

可用工具：
- web: Web 搜索、抓取网页、arXiv 论文搜索
- paper_kb: 论文知识库查询
- read: 读取文件内容
- write/append: 写入文件
- edit/edit_regex: 编辑文件
- grep/grep_count: 文件内容搜索
- glob/glob_list: 文件名搜索
- bash/bash_script: 执行 Shell 命令

请根据任务需要选择合适的工具执行。使用中文回答。"""

        messages = [{"role": "user", "content": task_prompt}]

        # 多轮工具调用循环
        for _ in range(self.max_iterations):
            # 检查中断
            if is_interrupted():
                store.update_task_status(plan_id, task_id, "pending", None)  # 重置状态
                return "任务被中断"

            response = llm_with_tools.invoke(messages)

            if not hasattr(response, "tool_calls") or not response.tool_calls:
                # 没有工具调用，返回结果
                result = response.content

                # 保存结果到数据库
                store.update_task_status(plan_id, task_id, "completed", result)

                return result

            # 执行工具调用
            for tool_call in response.tool_calls:
                # 检查中断
                if is_interrupted():
                    store.update_task_status(plan_id, task_id, "pending", None)
                    return "任务被中断"

                tool_name = tool_call["name"]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    tool_result = f"错误：未找到工具 {tool_name}"
                else:
                    try:
                        tool_result = tool.invoke(tool_args)
                        if isinstance(tool_result, dict):
                            import json

                            tool_result = json.dumps(
                                tool_result, ensure_ascii=False, indent=2
                            )
                        elif not isinstance(tool_result, str):
                            tool_result = str(tool_result)
                    except Exception as e:
                        tool_result = f"工具执行错误: {str(e)}"

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content
                        if hasattr(response, "content")
                        else "",
                        "tool_calls": [tool_call],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "name": tool_name,
                        "tool_call_id": tool_call.get("id"),
                    }
                )

        # 达到最大迭代次数
        result = "任务执行达到最大迭代次数"
        store.update_task_status(plan_id, task_id, "completed", result)
        return result

    def _execute_text_task(self, message: Message, a2a_task: Task) -> str:
        """执行普通文本任务（非 PlanTask）"""
        from utils.core.llm import get_llm_model

        llm = get_llm_model()
        llm_with_tools = llm.bind_tools(self.tools)

        messages = [{"role": "user", "content": message.get_text()}]

        for _ in range(self.max_iterations):
            # 检查中断
            if is_interrupted():
                return "任务被中断"

            response = llm_with_tools.invoke(messages)

            if not hasattr(response, "tool_calls") or not response.tool_calls:
                return response.content

            for tool_call in response.tool_calls:
                # 检查中断
                if is_interrupted():
                    return "任务被中断"

                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool = next((t for t in self.tools if t.name == tool_name), None)
                if tool:
                    try:
                        tool_result = tool.invoke(tool_args)
                        if isinstance(tool_result, dict):
                            import json

                            tool_result = json.dumps(
                                tool_result, ensure_ascii=False, indent=2
                            )
                        elif not isinstance(tool_result, str):
                            tool_result = str(tool_result)
                    except Exception as e:
                        tool_result = f"工具执行错误: {str(e)}"
                else:
                    tool_result = f"未找到工具: {tool_name}"

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [tool_call],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "name": tool_name,
                        "tool_call_id": tool_call.get("id"),
                    }
                )

        return "任务执行达到最大迭代次数"


# ============ Worker Pool ============


class A2AWorkerPool:
    """A2A Worker 池

    管理多个 A2AWorker 实例，提供统一的启动/停止接口。
    """

    def __init__(
        self,
        num_workers: int = 2,
        transport: InMemoryTransport | None = None,
        tools: list | None = None,
    ):
        """初始化 Worker Pool

        Args:
            num_workers: Worker 数量
            transport: Transport 实例
            tools: 工具列表
        """
        self.num_workers = num_workers
        self.transport = transport or self._get_default_transport()
        self.tools = tools

        self._workers: list[A2AWorker] = []

    def _get_default_transport(self) -> InMemoryTransport:
        from agent.a2a.transport import get_transport

        return get_transport()

    def start(self) -> None:
        """启动所有 Worker"""
        for i in range(self.num_workers):
            worker = A2AWorker(
                worker_id=f"worker-{i + 1}",
                transport=self.transport,
                tools=self.tools,
            )
            worker.start()
            self._workers.append(worker)

    def stop(self) -> None:
        """停止所有 Worker"""
        for worker in self._workers:
            worker.stop()
        self._workers.clear()

    def get_stats(self) -> list[dict]:
        """获取所有 Worker 状态"""
        return [w.stats for w in self._workers]

    def get_available_workers(self) -> list[A2AWorker]:
        """获取空闲的 Worker"""
        return [w for w in self._workers if w.current_task is None]

    @property
    def workers(self) -> list[A2AWorker]:
        """所有 Worker"""
        return self._workers.copy()
