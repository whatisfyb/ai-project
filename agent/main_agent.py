"""Main Agent - 智能路由主代理"""

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from utils.llm import get_llm_model
from agent.state import MainAgentState


# 系统提示词
MAIN_AGENT_PROMPT = """你是一个智能助手，能够理解用户需求并选择最合适的方式完成任务。

## 可用工具

### 搜索和信息工具
- `tavily_search` - 网络搜索
- `tavily_extract` - URL内容提取
- `arxiv_search` - arXiv论文搜索
- `arxiv_download_pdf` - 下载arXiv论文PDF

### 子代理分发工具
- `dispatch_agent` - 分发任务给专门的子代理执行
  - subagent_type="Plan": 用于复杂任务的拆解和规划
  - subagent_type="Research": 用于信息搜索、论文查找、知识库检索
  - subagent_type="Analysis": 用于数据分析、报告生成
  - subagent_type="ExecutePlan": 用于执行复杂的多步骤任务（会自动拆解并并行执行）

- `list_subagents` - 列出所有可用的子代理

## 决策规则

1. **简单对话**（问候、闲聊）：直接回复，不要调用任何工具
2. **简单搜索**（"搜索XXX"）：使用 tavily_search
3. **复杂研究**（"研究XXX领域"、"帮我调研XXX"）：使用 dispatch_agent，subagent_type="Research"
4. **任务规划**（"帮我规划XXX"、"如何完成XXX"）：使用 dispatch_agent，subagent_type="Plan"
5. **复杂任务执行**（"帮我写一个爬虫"、"帮我开发XXX"）：使用 dispatch_agent，subagent_type="ExecutePlan"
6. **数据分析**（"分析XXX数据"、"生成报告"）：使用 dispatch_agent，subagent_type="Analysis"

重要：对于简单问候和闲聊，直接回复用户，不要调用任何工具！
"""


class MainAgent:
    """Main Agent - 智能路由主代理

    职责：
    - 理解用户意图
    - 路由到子代理或直接调用工具
    - 整合结果并生成响应
    """

    def __init__(self):
        self.llm = get_llm_model()
        self._init_tools()
        self._graph = None
        self._checkpointer = None

    def _init_tools(self):
        """初始化工具"""
        from tools.tavily import tavily_search, tavily_extract
        from tools.arxiv_search import arxiv_search, arxiv_download_pdf
        from tools.agent import dispatch_agent, list_subagents
        from tools.skills_manager import load_skills, list_skills

        self.tools = [
            tavily_search,
            tavily_extract,
            arxiv_search,
            arxiv_download_pdf,
            dispatch_agent,
            list_subagents,
            load_skills,
            list_skills,
        ]

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _reason_node(self, state: MainAgentState) -> dict:
        """推理节点 - 分析用户输入并决定下一步"""
        messages = state["messages"]
        current_task = state.get("current_task")

        # 构建系统提示
        system_msg = SystemMessage(content=MAIN_AGENT_PROMPT)

        # 构建用户消息
        if current_task:
            user_msg = HumanMessage(content=current_task)
            all_messages = [system_msg] + messages + [user_msg]
        else:
            all_messages = [system_msg] + messages

        # 调用 LLM 进行推理
        response = self.llm_with_tools.invoke(all_messages)

        return {"messages": [response]}

    def _route_decision(self, state: MainAgentState) -> str:
        """路由决策 - 决定下一步执行什么"""
        messages = state["messages"]
        if not messages:
            return "end"

        last_message = messages[-1]

        # 检查是否有工具调用（支持字典和消息对象）
        if isinstance(last_message, dict):
            has_tool_calls = "tool_calls" in last_message and last_message["tool_calls"]
        else:
            has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        if has_tool_calls:
            return "tools"

        return "end"

    def build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(MainAgentState)

        # 添加节点
        graph.add_node("reason", self._reason_node)
        graph.add_node("tools", ToolNode(self.tools))

        # 添加条件边
        graph.add_edge(START, "reason")
        graph.add_conditional_edges(
            "reason",
            self._route_decision,
            {
                "tools": "tools",
                "end": END,
            },
        )

        # 工具执行后返回推理
        graph.add_edge("tools", "reason")

        return graph

    @property
    def graph(self):
        """获取编译后的图"""
        if self._graph is None:
            graph = self.build_graph()
            self._graph = graph.compile(checkpointer=self.checkpointer)
        return self._graph

    @property
    def checkpointer(self):
        """获取检查点存储（延迟初始化，共享实例）"""
        if self._checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            self._checkpointer = MemorySaver()
        return self._checkpointer

    def chat(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        """与 Main Agent 对话

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Returns:
            响应结果
        """
        from langchain_core.messages import HumanMessage

        # 使用共享的图
        graph = self.graph

        # 从 checkpointer 获取历史消息
        config = {"configurable": {"thread_id": thread_id}}
        existing_state = graph.get_state(config)
        existing_messages = list(existing_state.values.get("messages", [])) if existing_state else []

        # 追加新消息
        input_data = {
            "messages": existing_messages + [HumanMessage(content=message)],
            "current_task": None,
            "memory_context": None,
            "subagent_results": {},
        }

        result = graph.invoke(input_data, config)
        return result

    def get_response(self, result: dict[str, Any]) -> str:
        """从结果中提取响应文本"""
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                return last.content
        return ""


def create_main_agent() -> MainAgent:
    """创建 Main Agent 实例"""
    return MainAgent()


# 简单的 REPL 入口
def run_repl():
    """运行 REPL 交互"""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    agent = create_main_agent()

    console.print("[bold green]Main Agent 已启动[/bold green]")
    console.print("输入消息与 Agent 对话，输入 /exit 退出\n")

    thread_id = "default"

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/exit":
                console.print("[bold yellow]再见！[/bold yellow]")
                break

            if user_input.lower() == "/clear":
                thread_id = f"session_{int(__import__('time').time())}"
                console.print("[bold green]会话已重置[/bold green]\n")
                continue

            # 调用 Agent
            result = agent.chat(user_input, thread_id)
            response = agent.get_response(result)

            # 显示响应
            console.print("\n[bold green]Agent:[/bold green]")
            console.print(Markdown(response))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[bold yellow]再见！[/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]错误:[/bold red] {e}\n")


if __name__ == "__main__":
    run_repl()
