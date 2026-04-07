"""Plan Agent - 复杂任务拆解"""

from typing import Annotated
from typing_extensions import TypedDict
import operator

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START

from agent.subagents.base import BaseSubagent


class PlanAgentState(TypedDict):
    """Plan Agent 状态"""
    task: str  # 原始任务
    analysis: str | None  # 任务分析结果
    context: Annotated[list[str], operator.add]  # 收集的上下文
    plan: str | None  # 生成的计划
    steps: list[dict] | None  # 拆解的步骤


@tool
def analyze_task(task: str) -> str:
    """分析任务，识别关键需求和约束

    Args:
        task: 待分析的任务描述

    Returns:
        任务分析结果
    """
    # 这个工具主要由 LLM 调用，实际逻辑在节点中实现
    return f"分析任务: {task}"


@tool
def breakdown_task(task: str, context: str) -> list[dict]:
    """将复杂任务拆解为可执行的子任务

    Args:
        task: 任务描述
        context: 相关上下文信息

    Returns:
        拆解后的子任务列表
    """
    # 这个工具主要由 LLM 调用，实际逻辑在节点中实现
    return [{"step": 1, "task": task}]


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
        return [analyze_task, breakdown_task]

    def _analyze_node(self, state: PlanAgentState) -> dict:
        """分析任务节点"""
        task = state["task"]

        prompt = f"""你是一个任务分析专家。请分析以下任务：

任务：{task}

请从以下角度分析：
1. 任务目标：这个任务要达成什么？
2. 关键需求：完成这个任务需要什么？
3. 潜在挑战：可能遇到什么困难？
4. 依赖关系：是否有前置条件？

请用简洁的中文回答。"""

        response = self.llm.invoke(prompt)
        return {"analysis": response.content}

    def _plan_node(self, state: PlanAgentState) -> dict:
        """生成计划节点"""
        task = state["task"]
        analysis = state["analysis"] or ""
        context = "\n".join(state["context"]) if state["context"] else "无"

        prompt = f"""你是一个规划专家。请根据以下信息制定执行计划：

任务：{task}

分析结果：
{analysis}

上下文信息：
{context}

请制定详细的执行计划，格式如下：

## 执行计划

### 步骤 1: [步骤名称]
- 描述：[具体要做什么]
- 输出：[预期产出]
- 依赖：[是否有前置步骤]

### 步骤 2: ...
...

### 总结
- 总步骤数：X
- 预估复杂度：低/中/高
- 关键风险：...
"""

        response = self.llm.invoke(prompt)
        return {"plan": response.content}

    def _breakdown_node(self, state: PlanAgentState) -> dict:
        """拆解任务节点"""
        plan = state["plan"] or ""

        prompt = f"""请将以下计划转换为结构化的步骤列表（JSON 格式）：

{plan}

输出格式：
```json
{{
  "steps": [
    {{"step": 1, "name": "步骤名称", "description": "描述", "dependencies": []}},
    ...
  ]
}}
```

只输出 JSON，不要其他内容。"""

        response = self.llm.invoke(prompt)
        content = response.content

        # 尝试解析 JSON
        import json
        steps = []
        try:
            # 提取 JSON 块
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            data = json.loads(json_str)
            steps = data.get("steps", [])
        except (json.JSONDecodeError, KeyError):
            # 如果解析失败，创建默认步骤
            steps = [{"step": 1, "name": "执行计划", "description": plan}]

        return {"steps": steps}

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(PlanAgentState)

        # 添加节点
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("breakdown", self._breakdown_node)

        # 添加边
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "plan")
        graph.add_edge("plan", "breakdown")
        graph.add_edge("breakdown", END)

        return graph

    def run(self, task: str, thread_id: str = "default") -> dict:
        """运行 Plan Agent

        Args:
            task: 待规划的任务
            thread_id: 会话 ID

        Returns:
            包含 analysis, plan, steps 的结果
        """
        input_data = {
            "task": task,
            "analysis": None,
            "context": [],
            "plan": None,
            "steps": None,
        }
        result = super().run(input_data, thread_id)
        return result
