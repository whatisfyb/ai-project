"""Plan Agent - 复杂任务拆解"""

import json
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START

from agent.core.models import Plan, PlanTask
from agent.subagents.base import BaseSubagent
from store.plan import PlanStore


# ============ Agent 状态 ============

class PlanAgentState(TypedDict):
    """Plan Agent 状态"""
    task: str  # 原始任务
    plan: Plan | None  # 生成的计划
    plan_id: str | None  # 持久化 ID
    thread_id: str  # 会话 ID


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
        thread_id = state["thread_id"]

        prompt = f"""你是一个任务规划专家，负责将复杂需求拆解为可执行的并行任务计划。

## 任务
{task}

## 你的职责

深度分析需求后，识别：
1. **可并行执行的部分** - 哪些子任务相互独立，可以同时执行？
2. **串行依赖链** - 哪些任务必须按顺序执行？
3. **合理的任务粒度** - 每个任务应该足够独立、可端到端完成

## 任务拆解原则

1. **独立性**：每个任务应该是自包含的，能独立执行并产生有意义的结果
2. **可并行性**：没有依赖的任务应能并行执行
3. **清晰描述**：description 要明确说明"做什么"，而不是"怎么做"
4. **最小依赖**：只声明真正必要的依赖

## 示例

用户需求："帮我研究transformer和bert的区别"
✅ 正确拆解：
```json
{{
  "goal": "分析transformer和bert的区别",
  "tasks": [
    {{"id": "T1", "description": "在arxiv搜索transformer相关论文", "dependencies": []}},
    {{"id": "T2", "description": "用网页搜索bert的含义和原理", "dependencies": []}},
    {{"id": "T3", "description": "对比transformer和bert的区别并总结", "dependencies": ["T1", "T2"]}}
  ]
}}
```

## 输出格式

**只输出JSON，不要任何解释**：
```json
{{
  "goal": "整体目标描述",
  "tasks": [
    {{"id": "T1", "description": "任务描述", "dependencies": []}},
    {{"id": "T2", "description": "任务描述", "dependencies": ["T1"]}}
  ]
}}
```"""

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
                tasks=[PlanTask(id="T1", description=task, dependencies=[], status="pending")],
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
            "thread_id": thread_id,
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

    def get_pending_tasks(self, plan_id: str) -> list[PlanTask]:
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

    def claim_task(self, plan_id: str, worker_id: str) -> PlanTask | None:
        """原子领取任务

        Args:
            plan_id: Plan ID
            worker_id: Worker 标识

        Returns:
            领取到的 Task，无可用任务返回 None
        """
        return self.store.claim_task(plan_id, worker_id)

    def release_task(self, plan_id: str, task_id: str) -> bool:
        """释放任务（用于失败重试）

        Args:
            plan_id: Plan ID
            task_id: Task ID

        Returns:
            是否成功
        """
        return self.store.release_task(plan_id, task_id)

    def check_all_completed(self, plan_id: str) -> bool:
        """检查所有任务是否完成

        Args:
            plan_id: Plan ID

        Returns:
            是否全部完成
        """
        return self.store.check_all_completed(plan_id)

    def check_all_done(self, plan_id: str) -> bool:
        """检查所有任务是否结束（完成或失败）

        Args:
            plan_id: Plan ID

        Returns:
            是否全部结束
        """
        return self.store.check_all_done(plan_id)

    def save_summarized_result(self, plan_id: str, result: str) -> bool:
        """保存汇总结果

        Args:
            plan_id: Plan ID
            result: 汇总结果

        Returns:
            是否成功
        """
        return self.store.save_summarized_result(plan_id, result)

    def get_all_task_results(self, plan_id: str) -> list[dict]:
        """获取所有任务结果

        Args:
            plan_id: Plan ID

        Returns:
            任务结果列表
        """
        return self.store.get_all_task_results(plan_id)

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

if __name__ == "__main__":
    agent = PlanAgent()
    plan, plan_id = agent.run("我要你创建一个plan：T1与T2并行运行 T1 上arxiv查找3篇transformer论文 T2 用网页搜索搜索bert的含义 T3 前两个任务执行完毕后 总结transformer和Bert的区别")
    print(f"生成的计划 ID: {plan_id}")
    print("计划内容:")
    for task in plan.tasks:
        print(f"- {task.id}: {task.description} (依赖: {task.dependencies})")
