"""状态定义"""

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


class MainAgentState(TypedDict):
    """Main Agent 状态"""
    messages: Annotated[list[dict], operator.add]  # 消息历史
    current_task: str | None  # 当前任务
    memory_context: str | None  # 记忆上下文
    subagent_results: dict[str, Any]  # 子代理结果
