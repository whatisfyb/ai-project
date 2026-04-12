"""Memory Agent - 记忆分析代理

分析用户输入，决定是否需要创建/更新/删除长期记忆。
采用 ReAct 模式，支持多步操作，最多 3 次迭代。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from pydantic import Field
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from agent.subagents.base import BaseSubagent
from store.long_term_memory_persistency import (
    MemoryType,
    MEMORY_TYPES,
    get_memory_store,
)


# ============ 工具输入模型 ============

class CreateMemoryInput(BaseModel):
    """创建记忆的参数"""
    action: Literal["create"] = Field(description="操作类型：创建记忆")
    name: str = Field(description="记忆名称，用于文件名，如 'user_preference_spicy'")
    description: str = Field(description="记忆的一行描述，用于判断相关性")
    type: MemoryType = Field(description="记忆类型：user(用户信息), feedback(反馈指导), project(项目信息), reference(外部引用)")
    content: str = Field(description="记忆的详细内容")


class UpdateMemoryInput(BaseModel):
    """更新记忆的参数"""
    action: Literal["update"] = Field(description="操作类型：更新记忆")
    name: str = Field(description="要更新的记忆名称")
    description: str | None = Field(default=None, description="新的记忆描述")
    type: MemoryType | None = Field(default=None, description="新的记忆类型")
    content: str | None = Field(default=None, description="新的记忆内容")


class DeleteMemoryInput(BaseModel):
    """删除记忆的参数"""
    action: Literal["delete"] = Field(description="操作类型：删除记忆")
    name: str = Field(description="要删除的记忆名称")


class ListMemoryInput(BaseModel):
    """列出记忆的参数"""
    action: Literal["list"] = Field(description="操作类型：列出所有记忆")


class SearchMemoryInput(BaseModel):
    """搜索记忆的参数"""
    action: Literal["search"] = Field(description="操作类型：搜索记忆")
    query: str = Field(description="搜索关键词")


# ============ 记忆操作工具 ============

@tool
def create_memory(
    name: str,
    description: str,
    type: MemoryType,
    content: str,
) -> dict[str, Any]:
    """创建一个新的长期记忆。

    Args:
        name: 记忆名称，用于文件名，如 'user_preference_spicy'
        description: 记忆的一行描述，用于判断相关性
        type: 记忆类型 - user(用户信息), feedback(反馈指导), project(项目信息), reference(外部引用)
        content: 记忆的详细内容

    Returns:
        操作结果
    """
    store = get_memory_store()

    try:
        if type not in MEMORY_TYPES:
            return {"success": False, "error": f"无效的类型: {type}，有效类型: {MEMORY_TYPES}"}

        # 检查是否已存在
        if store.memory_exists(name):
            # 已存在则更新
            from store.long_term_memory_persistency import Memory
            memory = Memory(
                name=name,
                description=description,
                type=type,
                content=content,
            )
            store.update(name, memory)
            return {"success": True, "action": "updated", "name": name, "type": type}

        from store.long_term_memory_persistency import Memory
        memory = Memory(
            name=name,
            description=description,
            type=type,
            content=content,
        )
        path = store.create(memory)
        return {"success": True, "action": "created", "name": name, "type": type, "path": path}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def update_memory(
    name: str = Field(description="要更新的记忆名称，如 user_name, user_role"),
    content: str = Field(description="新的记忆内容，必须填写具体的文字内容，不能为空"),
    description: str | None = Field(default=None, description="新的记忆描述（可选）"),
    type: MemoryType | None = Field(default=None, description="新的记忆类型（可选）"),
) -> dict[str, Any]:
    """更新一个已存在的长期记忆。必须提供 name 和 content 参数。

    Args:
        name: 要更新的记忆名称
        content: 新的记忆内容（必填）
        description: 新的记忆描述（可选，不填则保持原值）
        type: 新的记忆类型（可选，不填则保持原值）

    Returns:
        操作结果
    """
    store = get_memory_store()

    try:
        from store.long_term_memory_persistency import Memory
        existing = store.read(name)
        if not existing:
            return {"success": False, "error": f"记忆不存在: {name}"}

        # 合并更新
        updated_memory = Memory(
            name=name,
            description=description or existing.description,
            type=type if type else existing.type,
            content=content,
        )
        success = store.update(name, updated_memory)
        return {"success": success, "action": "updated", "name": name, "type": updated_memory.type}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def delete_memory(name: str) -> dict[str, Any]:
    """删除一个长期记忆。

    Args:
        name: 要删除的记忆名称

    Returns:
        操作结果
    """
    store = get_memory_store()

    try:
        success = store.delete(name)
        return {"success": success, "action": "deleted", "name": name}

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def list_memories() -> dict[str, Any]:
    """列出所有长期记忆。

    Returns:
        记忆列表
    """
    store = get_memory_store()

    try:
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

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def search_memories(query: str) -> dict[str, Any]:
    """搜索长期记忆。

    Args:
        query: 搜索关键词

    Returns:
        匹配的记忆列表
    """
    store = get_memory_store()

    try:
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

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ 旧版兼容工具（保留向后兼容） ============

@tool
def long_term_memory_modify(
    action: Literal["create", "update", "delete", "list", "search"],
    name: str | None = None,
    description: str | None = None,
    type: MemoryType | None = None,
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
                from store.long_term_memory_persistency import Memory
                memory = Memory(
                    name=name,
                    description=description,
                    type=type,
                    content=content,
                )
                store.update(name, memory)
                return {"success": True, "action": "updated", "name": name, "type": type}

            from store.long_term_memory_persistency import Memory
            memory = Memory(
                name=name,
                description=description,
                type=type,
                content=content,
            )
            path = store.create(memory)
            return {"success": True, "action": "created", "name": name, "type": type, "path": path}

        elif action == "update":
            if not name:
                return {"success": False, "error": "更新记忆需要 name"}

            from store.long_term_memory_persistency import Memory
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
            return {"success": success, "action": "updated", "name": name, "type": updated_memory.type}

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
    iteration: int
    actions: list[dict]


# ============ Agent 提示词 ============

MEMORY_ANALYZE_PROMPT = """你是一个记忆管理助手，负责分析用户输入并决定是否需要修改长期记忆。

## 当前记忆列表

{memory_list}

## 记忆类型说明

1. **user** - 用户信息：角色、技能、偏好、背景
2. **feedback** - 反馈指导：用户告诉你要做或不做的事情
3. **project** - 项目信息：截止日期、计划、进度、决策
4. **reference** - 外部引用：文档链接、看板地址、追踪系统

## ⚠️ 极其重要：工具参数必须填写实际值

**绝对不要传 null 或 None 作为参数值！**

错误示例（禁止）：
- content: None  ❌
- name: null  ❌
- type: null  ❌

正确示例：
- content: "用户名字是方炎彬"  ✅
- name: "user_name"  ✅
- type: "user"  ✅

## 操作流程（必须严格遵循）

**第一步：判断用户意图**
- 用户是否要求"记住"、"记得"、纠正信息、透露偏好？
- 如果不是，直接回复"无需修改记忆"，不要调用任何工具。

**第二步：检查现有记忆**
- 查看上面的"当前记忆列表"
- 判断用户要更新的记忆是否已存在

**第三步：执行操作**
- 如果记忆已存在且需要修改 → 调用 `update_memory`，必须填写 content 参数
- 如果记忆不存在需要创建 → 调用 `create_memory`，必须填写所有参数
- 如果需要删除 → 调用 `delete_memory`

## 工具调用示例

创建新记忆（所有参数必填）：
```
create_memory(
    name="user_preference_theme",
    description="用户的主题偏好",
    type="user",
    content="用户喜欢深色主题"
)
```

更新已有记忆（name 和 content 必填）：
```
update_memory(
    name="user_name",
    content="用户名字是方炎彬"
)
```

## 重要规则

1. **参数必须有实际值**：禁止传 null/None，必须填写具体的字符串
2. **name 参数必须具体**：使用 snake_case 格式，如 `user_name`, `project_deadline`
3. **content 参数必须填写具体内容**：不要留空，必须写出记忆的具体内容
4. **不要猜测用户信息**：只记录用户明确说的内容
5. **一次只操作一个记忆**：如果需要操作多个记忆，分多次调用

请分析以下用户输入，决定是否需要修改记忆："""


# ============ 常量 ============

MAX_ITERATIONS = 3


# ============ Memory Agent ============

class MemoryAgent(BaseSubagent[MemoryAgentState]):
    """记忆分析 Agent

    分析用户输入，决定是否需要创建/更新/删除长期记忆。
    采用 ReAct 模式，支持多步操作。
    """

    def __init__(self):
        super().__init__()
        self._tool_node = None
        self._compiled_graph = None  # 覆盖父类的 _graph

    @property
    def agent_type(self) -> str:
        return "Memory"

    @property
    def description(self) -> str:
        return "记忆管理代理，分析用户输入并管理长期记忆的增删改查。"

    @property
    def tools(self) -> list:
        return [create_memory, update_memory, delete_memory, list_memories, search_memories]

    @property
    def graph(self):
        """获取编译后的图（不使用 checkpointer，因为 Memory Agent 不需要持久化状态）"""
        if self._compiled_graph is None:
            g = self.build_graph()
            self._compiled_graph = g.compile()
        return self._compiled_graph

    def _get_tool_node(self) -> ToolNode:
        """获取工具节点实例（延迟初始化）"""
        if self._tool_node is None:
            self._tool_node = ToolNode(self.tools)
        return self._tool_node

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(MemoryAgentState)

        # 分析节点
        async def analyze_node(state: MemoryAgentState) -> dict:
            user_input = state["user_input"]
            iteration = state["iteration"]

            # 获取当前记忆列表
            store = get_memory_store()
            memories = store.list()
            memory_list = "\n".join([
                f"- [{m.type}] {m.filename.replace('.md', '')}: {m.description}"
                for m in memories
            ]) if memories else "（暂无记忆）"

            # 构建消息
            if iteration == 0:
                # 首次分析，添加系统提示
                system_msg = SystemMessage(
                    content=MEMORY_ANALYZE_PROMPT.format(memory_list=memory_list)
                )
                user_msg = HumanMessage(content=f"用户输入：{user_input}")
                messages = [system_msg, user_msg]
            else:
                # 后续迭代，只添加用户消息
                messages = state["messages"]

            # 调用 LLM
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke(messages)

            return {
                "messages": messages + [response],
                "iteration": iteration + 1,
            }

        # 工具执行节点
        async def tools_node(state: MemoryAgentState) -> dict:
            messages = state["messages"]
            actions = state["actions"]

            last_message = messages[-1]
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return {"actions": actions}

            # 执行工具
            tool_node = self._get_tool_node()
            result = await tool_node.ainvoke({"messages": messages})
            new_messages = result.get("messages", [])

            # 提取操作结果
            new_actions = []
            for msg in new_messages:
                if isinstance(msg, ToolMessage):
                    # 从 tool_calls 获取操作信息
                    tool_call = None
                    for tc in last_message.tool_calls:
                        if tc["id"] == msg.tool_call_id:
                            tool_call = tc
                            break

                    if tool_call:
                        action_record = {
                            "tool": "long_term_memory_modify",
                            "args": tool_call["args"],
                        }
                        # 解析工具返回结果
                        if hasattr(msg, "content"):
                            import json
                            try:
                                result_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                                if isinstance(result_data, dict):
                                    action_record["result"] = result_data
                            except (json.JSONDecodeError, TypeError):
                                action_record["result"] = {"raw": msg.content}

                        new_actions.append(action_record)

            return {
                "messages": messages + new_messages,
                "actions": actions + new_actions,
            }

        # 路由决策
        def should_continue(state: MemoryAgentState) -> str:
            messages = state["messages"]
            iteration = state["iteration"]

            # 达到最大迭代次数
            if iteration >= MAX_ITERATIONS:
                return "end"

            # 检查是否有工具调用
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
            should_continue,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "analyze")

        return graph

    async def run_async(self, user_input: str) -> dict[str, Any]:
        """异步运行记忆分析

        Args:
            user_input: 用户输入

        Returns:
            分析结果，包含 success 和 actions
        """
        input_data: MemoryAgentState = {
            "messages": [],
            "user_input": user_input,
            "iteration": 0,
            "actions": [],
        }

        try:
            result = await self.graph.ainvoke(input_data)

            # 提取成功的操作
            successful_actions = []
            for action in result.get("actions", []):
                action_result = action.get("result", {})
                if action_result.get("success"):
                    successful_actions.append({
                        "action": action_result.get("action"),
                        "name": action_result.get("name"),
                        "type": action_result.get("type"),
                    })

            return {
                "success": True,
                "actions": successful_actions,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "actions": [],
            }

    def run(self, user_input: str) -> dict[str, Any]:
        """同步运行记忆分析"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 已有事件循环，创建新线程运行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.run_async(user_input))
                return future.result()
        else:
            return asyncio.run(self.run_async(user_input))


# ============ 便捷函数 ============

_memory_agent: MemoryAgent | None = None


def get_memory_agent() -> MemoryAgent:
    """获取全局 Memory Agent 实例"""
    global _memory_agent
    if _memory_agent is None:
        _memory_agent = MemoryAgent()
    return _memory_agent


if __name__ == "__main__":
    import asyncio
    agent = get_memory_agent()

    # 测试异步运行
    async def test():
        result = await agent.run_async("现在我是方炎彬 不是小明了 记住我的角色和技能")
        print(result)

    asyncio.run(test())
