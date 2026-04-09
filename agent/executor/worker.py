"""TaskWorker - 任务执行器

专门的任务执行器，不是 MainAgent 副本，不会 fork。
支持中断检查和工具调用。
"""

import threading
from typing import Optional

from agent.core.models import Task
from agent.core.signals import is_interrupted
from store.plan import PlanStore
from utils.llm import get_llm_model


def _get_worker_tools():
    """获取 Worker 可用的工具列表"""
    from tools.web import web_search, web_fetch
    from tools.web import arxiv_search, arxiv_download_pdf
    from tools.web import web_scrape, web_crawl, web_map

    return [
        web_search,
        web_fetch,
        arxiv_search,
        arxiv_download_pdf,
        web_scrape,
        web_crawl,
        web_map,
    ]


class TaskWorker:
    """任务执行 Worker

    从数据库领取任务，执行，写回结果。
    完成后死亡，不进行总结。

    Worker 不是 MainAgent 副本，不会 fork 自己。
    支持注册到 WorkerRegistry 以便 Supervisor 监控。
    """

    def __init__(
        self,
        worker_id: str,
        plan_id: str,
        store: PlanStore | None = None,
        registry=None,
        timeout: int = 300,
    ):
        """初始化 Worker

        Args:
            worker_id: Worker 标识
            plan_id: Plan ID
            store: PlanStore 实例
            registry: WorkerRegistry 实例，用于注册自己
            timeout: 单任务超时时间（秒）
        """
        self.worker_id = worker_id
        self.plan_id = plan_id
        self.store = store or PlanStore()
        self.registry = registry
        self.timeout = timeout

    def run(self) -> dict:
        """执行任务循环

        Returns:
            执行结果统计
        """
        # 注册到 WorkerRegistry
        if self.registry:
            self.registry.register(self.worker_id, threading.current_thread())

        results = {"completed": 0, "failed": 0, "tasks": [], "interrupted": False}

        try:
            while True:
                # 检查中断
                if is_interrupted():
                    results["interrupted"] = True
                    break

                # 1. 原子领取任务
                task = self.store.claim_task(self.plan_id, self.worker_id)

                if not task:
                    # 没有可执行任务，死亡
                    break

                # 2. 更新注册表中的任务信息
                if self.registry:
                    self.registry.update_task(self.worker_id, task.id)

                # 3. 执行任务（带超时）
                try:
                    result = self._execute_with_timeout(task)
                    self.store.update_task_status(
                        self.plan_id, task.id, "completed", result
                    )
                    results["completed"] += 1
                    results["tasks"].append({"id": task.id, "status": "completed"})
                except Exception as e:
                    # 检查是否是中断导致的
                    if is_interrupted():
                        self.store.release_task(self.plan_id, task.id)
                        results["interrupted"] = True
                        break

                    error_msg = f"执行失败: {str(e)}"
                    self.store.update_task_status(
                        self.plan_id, task.id, "failed", error_msg
                    )
                    self.store.release_task(self.plan_id, task.id)
                    results["failed"] += 1
                    results["tasks"].append({
                        "id": task.id,
                        "status": "failed",
                        "error": str(e)
                    })
        finally:
            # 注销自己
            if self.registry:
                self.registry.deregister(self.worker_id)

        return results

    def _execute_with_timeout(self, task: Task) -> str:
        """带超时执行任务

        支持多轮工具调用，直到 LLM 返回文本响应。

        Args:
            task: 要执行的任务

        Returns:
            执行结果

        Raises:
            TimeoutError: 执行超时
            Exception: 执行失败
        """
        result_container = {"result": None, "error": None}

        def target():
            try:
                # 初始化 LLM 和工具
                llm = get_llm_model()
                tools = _get_worker_tools()
                llm_with_tools = llm.bind_tools(tools)

                # 构建任务提示
                task_prompt = f"""你是一个任务执行者。请完成以下任务：

任务描述：{task.description}

请直接执行任务并返回结果。如果需要搜索信息、获取数据等，可以使用工具辅助。
使用中文回答。"""

                # 使用本地消息列表（不跨任务共享）
                messages = [
                    {"role": "user", "content": task_prompt}
                ]

                # 多轮工具调用循环
                max_iterations = 10
                for _ in range(max_iterations):
                    # 调用 LLM
                    response = llm_with_tools.invoke(messages)

                    # 检查是否有工具调用
                    if not hasattr(response, "tool_calls") or not response.tool_calls:
                        # 没有工具调用，直接返回响应
                        messages.append({"role": "assistant", "content": response.content})
                        result_container["result"] = response.content
                        return

                    # 有工具调用，执行工具
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        # 查找工具
                        tool = next((t for t in tools if t.name == tool_name), None)
                        if not tool:
                            tool_result = f"错误：未找到工具 {tool_name}"
                        else:
                            try:
                                tool_result = tool.invoke(tool_args)
                                # 转换为字符串
                                if isinstance(tool_result, dict):
                                    import json
                                    tool_result = json.dumps(tool_result, ensure_ascii=False, indent=2)
                                elif not isinstance(tool_result, str):
                                    tool_result = str(tool_result)
                            except Exception as e:
                                tool_result = f"工具执行错误: {str(e)}"

                        # 添加工具调用和结果到消息列表
                        messages.append({
                            "role": "assistant",
                            "content": response.content if hasattr(response, "content") else "",
                            "tool_calls": [tool_call],
                        })
                        messages.append({
                            "role": "tool",
                            "content": tool_result,
                            "name": tool_name,
                            "tool_call_id": tool_call.get("id"),
                        })

                # 达到最大迭代次数
                result_container["result"] = "任务执行达到最大迭代次数，请稍后重试。"

            except Exception as e:
                result_container["error"] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            # 超时
            raise TimeoutError(f"任务 {task.id} 执行超时 ({self.timeout}s)")

        if result_container["error"]:
            raise result_container["error"]

        return result_container["result"]
