"""Analysis Agent - 数据分析与报告生成"""

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START

from agent.subagents.base import BaseSubagent
from agent.core.models import Plan, PlanTask


class AnalysisAgentState(TypedDict):
    """Analysis Agent 状态"""
    task: str  # 分析任务
    data_context: Annotated[list[str], operator.add]  # 数据上下文
    analysis_result: str | None  # 分析结果
    report: str | None  # 生成的报告
    file_path: str | None  # 输出文件路径


class AnalysisAgent(BaseSubagent[AnalysisAgentState]):
    """Analysis Agent - 数据分析与报告生成

    职责：
    - 数据分析
    - 生成分析报告
    - 可视化建议
    """

    @property
    def agent_type(self) -> str:
        return "Analysis"

    @property
    def description(self) -> str:
        return (
            "分析代理，用于数据分析、报告生成、可视化建议。"
            "适用于需要处理数据、生成报告的任务。"
        )

    @property
    def tools(self) -> list:
        # Analysis agent 主要用 LLM 进行分析，工具可选
        return []

    def _collect_data_node(self, state: AnalysisAgentState) -> dict:
        """收集数据节点"""
        task = state["task"]

        prompt = f"""你是一个数据分析专家。用户需要分析的任务是：{task}

请分析这个任务需要什么类型的数据：
1. 数据来源（文件、数据库、API 等）
2. 数据格式（CSV、JSON、文本等）
3. 分析方法（统计分析、文本分析、比较分析等）
4. 预期输出（报告、图表、结论等）

请简要描述。"""

        response = self.llm.invoke(prompt)
        return {"data_context": [response.content]}

    def _analyze_node(self, state: AnalysisAgentState) -> dict:
        """分析处理节点"""
        task = state["task"]
        data_context = "\n".join(state["data_context"]) if state["data_context"] else ""

        prompt = f"""你是一个数据分析专家。请根据以下信息进行分析：

任务：{task}

数据上下文：
{data_context}

请执行分析并给出：
1. 分析方法说明
2. 主要发现
3. 数据洞察
4. 结论

如果缺少具体数据，请说明需要什么数据，并给出分析方法建议。"""

        response = self.llm.invoke(prompt)
        return {"analysis_result": response.content}

    def _report_node(self, state: AnalysisAgentState) -> dict:
        """生成报告节点"""
        task = state["task"]
        analysis_result = state["analysis_result"] or ""
        data_context = "\n".join(state["data_context"]) if state["data_context"] else ""

        prompt = f"""你是一个报告撰写专家。请根据以下分析结果生成一份正式的分析报告：

任务：{task}

分析结果：
{analysis_result}

数据上下文：
{data_context}

请生成格式化的分析报告，包括：

---
# 分析报告

## 1. 背景
[任务背景说明]

## 2. 方法
[分析方法说明]

## 3. 发现
[主要发现]

## 4. 结论
[结论与建议]

## 5. 可视化建议
[推荐的图表类型和说明]

---
生成时间：[当前日期]
"""

        response = self.llm.invoke(prompt)
        return {"report": response.content}

    def _save_node(self, state: AnalysisAgentState) -> dict:
        """保存报告节点"""
        report = state["report"]
        task = state["task"]

        if not report:
            return {"file_path": None}

        # 生成文件名
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in task[:30])
        filename = f"analysis_report_{safe_name}_{timestamp}.md"

        # 保存到 data 目录
        from pathlib import Path
        output_dir = Path("data/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        output_path.write_text(report, encoding="utf-8")

        return {"file_path": str(output_path)}

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(AnalysisAgentState)

        # 添加节点
        graph.add_node("collect", self._collect_data_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("report", self._report_node)
        graph.add_node("save", self._save_node)

        # 添加边
        graph.add_edge(START, "collect")
        graph.add_edge("collect", "analyze")
        graph.add_edge("analyze", "report")
        graph.add_edge("report", "save")
        graph.add_edge("save", END)

        return graph

    def run(self, task: str, thread_id: str = "default") -> dict:
        """运行 Analysis Agent

        Args:
            task: 分析任务描述
            thread_id: 会话 ID

        Returns:
            包含 analysis_result, report, file_path 的结果
        """
        input_data = {
            "task": task,
            "data_context": [],
            "analysis_result": None,
            "report": None,
            "file_path": None,
        }
        result = super().run(input_data, thread_id)
        return result
