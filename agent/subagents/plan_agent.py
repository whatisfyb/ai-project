"""Plan Agent - 复杂任务拆解"""

import json
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START

from agent.models import Plan, Task
from agent.subagents.base import BaseSubagent
from agent.plan_store import PlanStore


# ============ Agent 状态 ============

class PlanAgentState(TypedDict):
    """Plan Agent 状态"""
    task: str  # 原始任务
    plan: Plan | None  # 生成的计划
    plan_id: str | None  # 持久化 ID


class PlanAgent(BaseSubagent[PlanAgentState]):
    """Plan Agent - 复杂任务拆解

    职责：
    - 分析复杂需求
    - 拆解为可执行的子任务
    - 生成执行计划
    - 识别依赖关系
    - 持久化存储
    """

    def __init__(self):
        super().__init__()
        self.store = PlanStore()

    @property
    def agent_type(self) -> str:
        return "Plan"

    @property
    def description(self) -> str:
        return "规划代理，用于分析复杂需求并拆解为可执行的子任务。适用于需要多步骤完成的复杂任务。"

    @property
    def tools(self) -> list:
        return []

    def _plan_node(self, state: PlanAgentState) -> dict:
        """生成计划节点"""
        task = state["task"]
        thread_id = state.get("thread_id", "default")

        # 获取 JSON schema 用于提示词
        schema = Plan.model_json_schema()

        prompt = f"""你是一个任务规划专家。请分析以下任务，将其拆解为可执行的子任务。

任务：{task}

请严格按照以下 JSON Schema 输出结果（只输出 JSON，不要其他内容）：

```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

要求：
1. 拆分为多个有依赖关系的子任务
2. 每个任务要有唯一 ID（如 T1, T2, T3）
3. 明确任务之间的依赖关系（某任务依赖哪些其他任务完成后才能执行）
4. 没有依赖的任务可以并行执行
5. 使用中文描述
6. 只输出 JSON，不要输出任何其他内容"""

        response = self.llm.invoke(prompt)
        content = response.content

        # 解析 JSON
        try:
            # 提取 JSON 块
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            data = json.loads(json_str)
            plan = Plan.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            # 解析失败，创建默认计划
            plan = Plan(
                goal=task,
                tasks=[Task(id="T1", description=task, dependencies=[], status="pending")],
                status="pending"
            )

        # 持久化存储
        plan_id = self.store.save_plan(plan, thread_id)

        return {"plan": plan, "plan_id": plan_id}

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(PlanAgentState)

        # 添加节点
        graph.add_node("plan", self._plan_node)

        # 添加边
        graph.add_edge(START, "plan")
        graph.add_edge("plan", END)

        return graph

    def run(self, task: str, thread_id: str = "default") -> tuple[Plan, str]:
        """运行 Plan Agent

        Args:
            task: 待规划的任务
            thread_id: 会话 ID

        Returns:
            (Plan 对象, plan_id)
        """
        input_data = {
            "task": task,
            "plan": None,
            "plan_id": None,
        }
        result = super().run(input_data, thread_id)
        return result.get("plan"), result.get("plan_id")

    # ============ 恢复与管理方法 ============

    def resume_plan(self, plan_id: str) -> Plan | None:
        """恢复 Plan

        Args:
            plan_id: Plan ID

        Returns:
            Plan 对象，不存在返回 None
        """
        return self.store.load_plan(plan_id)

    def get_pending_tasks(self, plan_id: str) -> list[Task]:
        """获取可执行的待处理 Tasks

        Args:
            plan_id: Plan ID

        Returns:
            可执行的 Task 列表（依赖已满足）
        """
        return self.store.get_pending_tasks(plan_id)

    def complete_task(self, plan_id: str, task_id: str, result: str | None = None) -> bool:
        """完成 Task

        Args:
            plan_id: Plan ID
            task_id: Task ID
            result: 任务执行结果

        Returns:
            是否成功
        """
        return self.store.update_task_status(plan_id, task_id, "completed", result)

    def fail_task(self, plan_id: str, task_id: str, result: str | None = None) -> bool:
        """Task 失败

        Args:
            plan_id: Plan ID
            task_id: Task ID
            result: 失败原因

        Returns:
            是否成功
        """
        return self.store.update_task_status(plan_id, task_id, "failed", result)

    def list_plans(
        self,
        status: Literal["pending", "completed", "failed"] | None = None
    ) -> list:
        """列出 Plans

        Args:
            status: 按状态过滤

        Returns:
            Plan 列表
        """
        return self.store.list_plans(status=status)
