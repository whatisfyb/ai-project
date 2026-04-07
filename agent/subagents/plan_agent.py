"""Plan Agent - 复杂任务拆解"""

import json
from typing import Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START

from agent.subagents.base import BaseSubagent


# ============ Pydantic 模型 ============

class Task(BaseModel):
    """单个任务"""
    id: str = Field(description="任务唯一标识，如 T1, T2, T3")
    description: str = Field(description="任务详细描述")
    dependencies: list[str] = Field(default=[], description="依赖的任务 ID 列表")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending", description="任务状态"
    )


class Plan(BaseModel):
    """执行计划"""
    goal: str = Field(description="整体目标")
    tasks: list[Task] = Field(description="任务列表")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending", description="计划状态"
    )


# ============ Agent 状态 ============

class PlanAgentState(TypedDict):
    """Plan Agent 状态"""
    task: str  # 原始任务
    plan: Plan | None  # 生成的计划


class PlanAgent(BaseSubagent[PlanAgentState]):
    """Plan Agent - 复杂任务拆解

    职责：
    - 分析复杂需求
    - 拆解为可执行的子任务
    - 生成执行计划
    - 识别依赖关系
    """

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

        return {"plan": plan}

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(PlanAgentState)

        # 添加节点
        graph.add_node("plan", self._plan_node)

        # 添加边
        graph.add_edge(START, "plan")
        graph.add_edge("plan", END)

        return graph

    def run(self, task: str, thread_id: str = "default") -> Plan:
        """运行 Plan Agent

        Args:
            task: 待规划的任务
            thread_id: 会话 ID

        Returns:
            Plan 对象
        """
        input_data = {
            "task": task,
            "plan": None,
        }
        result = super().run(input_data, thread_id)
        return result.get("plan")
