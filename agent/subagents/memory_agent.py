"""Memory Agent - 记忆分析代理

分析用户输入，决定是否需要创建/更新/删除长期记忆。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils.llm import get_llm_model
from store.long_term_memory_persistency import (
    LongTermMemoryStore,
    Memory,
    MemoryType,
    MEMORY_TYPES,
    get_memory_store,
)


# ============ 记忆操作工具 ============

class MemoryModifyInput(BaseModel):
    """记忆修改工具输入"""
    action: Literal["create", "update", "delete", "list", "search"] = Field(
        description="操作类型: create(创建), update(更新), delete(删除), list(列出), search(搜索)"
    )
    name: str | None = Field(
        default=None,
        description="记忆名称（create/update/delete 时必需）"
    )
    description: str | None = Field(
        default=None,
        description="记忆描述（create/update 时必需）"
    )
    type: MemoryType | None = Field(
        default=None,
        description="记忆类型: user, feedback, project, reference（create/update 时必需）"
    )
    content: str | None = Field(
        default=None,
        description="记忆内容（create/update 时必需）"
    )
    query: str | None = Field(
        default=None,
        description="搜索关键词（search 时使用）"
    )


@tool
def long_term_memory_modify(
    action: str,
    name: str | None = None,
    description: str | None = None,
    type: str | None = None,
    content: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """长期记忆修改工具，支持记忆的增删改查操作。

    Args:
        action: 操作类型 - create(创建), update(更新), delete(删除), list(列出), search(搜索)
        name: 记忆名称（create/update/delete 时必需）
        description: 记忆描述（create/update 时必需）
        type: 记忆类型 - user, feedback, project, reference（create/update 时必需）
        content: 记忆内容（create/update 时必需）
        query: 搜索关键词（search 时使用）

    Returns:
        操作结果
    """
    store = get_memory_store()

    try:
        if action == "create":
            if not name or not description or not type or not content:
                return {"success": False, "error": "创建记忆需要 name, description, type, content"}

            if type not in MEMORY_TYPES:
                return {"success": False, "error": f"无效的类型: {type}，有效类型: {MEMORY_TYPES}"}

            # 检查是否已存在
            if store.memory_exists(name):
                # 已存在则更新
                memory = Memory(
                    name=name,
                    description=description,
                    type=type,
                    content=content,
                )
                store.update(name, memory)
                return {"success": True, "action": "updated", "name": name}

            memory = Memory(
                name=name,
                description=description,
                type=type,
                content=content,
            )
            path = store.create(memory)
            return {"success": True, "action": "created", "name": name, "path": path}

        elif action == "update":
            if not name:
                return {"success": False, "error": "更新记忆需要 name"}

            existing = store.read(name)
            if not existing:
                return {"success": False, "error": f"记忆不存在: {name}"}

            # 合并更新
            updated_memory = Memory(
                name=name,
                description=description or existing.description,
                type=type if type else existing.type,
                content=content or existing.content,
            )
            success = store.update(name, updated_memory)
            return {"success": success, "action": "updated", "name": name}

        elif action == "delete":
            if not name:
                return {"success": False, "error": "删除记忆需要 name"}

            success = store.delete(name)
            return {"success": success, "action": "deleted", "name": name}

        elif action == "list":
            memories = store.list()
            return {
                "success": True,
                "action": "listed",
                "count": len(memories),
                "memories": [
                    {
                        "name": m.filename.replace(".md", ""),
                        "type": m.type,
                        "description": m.description,
                    }
                    for m in memories
                ],
            }

        elif action == "search":
            if not query:
                return {"success": False, "error": "搜索需要 query"}

            results = store.search(query)
            return {
                "success": True,
                "action": "searched",
                "query": query,
                "count": len(results),
                "memories": [
                    {
                        "name": m.name,
                        "type": m.type,
                        "description": m.description,
                        "content_preview": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                    }
                    for m in results
                ],
            }

        else:
            return {"success": False, "error": f"未知操作: {action}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ Agent 状态 ============

class MemoryAgentState(TypedDict):
    """记忆分析 Agent 状态"""
    messages: list
    user_input: str
    action_taken: bool


# ============ Agent 提示词 ============

MEMORY_ANALYZE_PROMPT = """你是一个记忆管理助手，负责分析用户输入并决定是否需要修改长期记忆。

## 记忆类型说明

1. **user** - 用户信息：角色、技能、偏好、背景
2. **feedback** - 反馈指导：用户告诉你要做或不做的事情
3. **project** - 项目信息：截止日期、计划、进度、决策
4. **reference** - 外部引用：文档链接、看板地址、追踪系统

## 何时创建/更新记忆

- 用户明确说"记得"、"记住"
- 用户透露自己的角色、技能、偏好
- 用户纠正你的行为或给出指导
- 用户提到项目截止日期、计划等
- 用户提到外部系统的位置

## 何时不创建记忆

- 简单的问答对话
- 临时的任务描述
- 可以从代码中推断的信息
- 已经有相同内容的记忆

## 记忆内容格式

对于 feedback 和 project 类型，使用以下格式：
```
[事实/规则]

**Why:** [原因]
**How to apply:** [如何应用]
```

## 当前记忆列表

{memory_list}

## 操作指南

1. 先用 list 或 search 查看现有记忆
2. 判断是否需要 create、update 还是 delete
3. 执行相应操作

请分析以下用户输入，决定是否需要修改记忆："""


# ============ Memory Agent ============

class MemoryAgent:
    """记忆分析 Agent

    分析用户输入，决定是否需要创建/更新/删除长期记忆。
    """

    def __init__(self):
        self.llm = get_llm_model()
        self.tools = [long_term_memory_modify]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self._graph = None

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(MemoryAgentState)

        # 分析节点
        async def analyze_node(state: MemoryAgentState) -> dict:
            user_input = state["user_input"]
            store = get_memory_store()

            # 获取当前记忆列表
            memories = store.list()
            memory_list = "\n".join([
                f"- [{m.type}] {m.filename.replace('.md', '')}: {m.description}"
                for m in memories
            ]) if memories else "（暂无记忆）"

            # 构建系统提示
            system_msg = SystemMessage(
                content=MEMORY_ANALYZE_PROMPT.format(memory_list=memory_list)
            )
            user_msg = HumanMessage(content=f"用户输入：{user_input}")

            response = await self.llm_with_tools.ainvoke([system_msg, user_msg])
            return {"messages": [response]}

        # 工具执行节点 - 执行工具并标记 action_taken
        async def tools_node(state: MemoryAgentState) -> dict:
            from langchain_core.messages import ToolMessage

            messages = state["messages"]
            last_message = messages[-1]

            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return {"action_taken": False}

            # 执行工具调用
            tool_node_executor = ToolNode(self.tools)
            result = await tool_node_executor.ainvoke({"messages": messages})
            new_messages = result.get("messages", [])

            return {
                "messages": new_messages,
                "action_taken": True,  # 标记已执行工具
            }

        # 路由决策
        def should_use_tools(state: MemoryAgentState) -> str:
            messages = state["messages"]
            if not messages:
                return "end"

            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            return "end"

        # 添加节点和边
        graph.add_node("analyze", analyze_node)
        graph.add_node("tools", tools_node)

        graph.add_edge(START, "analyze")
        graph.add_conditional_edges(
            "analyze",
            should_use_tools,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "analyze")

        return graph

    @property
    def graph(self):
        """获取编译后的图"""
        if self._graph is None:
            g = self.build_graph()
            self._graph = g.compile()
        return self._graph

    @property
    def agent_type(self) -> str:
        """代理类型标识"""
        return "Memory"

    @property
    def description(self) -> str:
        """代理描述"""
        return "记忆管理代理，分析用户输入并管理长期记忆的增删改查。"

    @property
    def available_tools(self) -> list:
        """可用工具列表"""
        return [t.name for t in self.tools]

    def get_info(self) -> dict[str, Any]:
        """获取代理信息"""
        return {
            "type": self.agent_type,
            "description": self.description,
            "tools": self.available_tools,
        }

    async def run_async(self, user_input: str) -> dict[str, Any]:
        """异步运行记忆分析

        Args:
            user_input: 用户输入

        Returns:
            分析结果
        """
        input_data = {
            "messages": [],
            "user_input": user_input,
            "action_taken": False,
        }

        try:
            result = await self.graph.ainvoke(input_data)
            return {
                "success": True,
                "action_taken": result.get("action_taken", False),
                "messages": result.get("messages", []),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_taken": False,
            }

    def run(self, user_input: str) -> dict[str, Any]:
        """同步运行记忆分析"""
        import asyncio
        return asyncio.run(self.run_async(user_input))


# ============ 便捷函数 ============

_memory_agent: MemoryAgent | None = None


def get_memory_agent() -> MemoryAgent:
    """获取全局 Memory Agent 实例"""
    global _memory_agent
    if _memory_agent is None:
        _memory_agent = MemoryAgent()
    return _memory_agent
