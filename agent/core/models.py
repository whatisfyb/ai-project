"""数据模型定义"""

from typing import Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class PlanTask(BaseModel):
    """Plan 中的执行单元"""
    id: str = Field(description="任务唯一标识，如 T1, T2, T3")
    description: str = Field(description="任务详细描述")
    dependencies: list[str] = Field(default=[], description="依赖的任务 ID 列表")
    status: Literal["pending", "completed", "failed"] = Field(
        default="pending", description="任务状态"
    )
    result: str | None = Field(default=None, description="任务执行结果")
    claimed_by: str | None = Field(default=None, description="领取该任务的 Worker ID")


class Plan(BaseModel):
    """执行计划"""
    goal: str = Field(description="整体目标")
    tasks: list[PlanTask] = Field(description="任务列表")
    status: Literal["pending", "completed", "failed"] = Field(
        default="pending", description="计划状态"
    )
    summarized_result: str | None = Field(default=None, description="最终汇总结果")


class MainAgentState(TypedDict):
    """Main Agent 状态"""
    messages: list[dict]  # 消息历史
    current_task: str | None  # 当前任务
    memory_context: str | None  # 记忆上下文
    thread_id: str  # 会话 ID
    # 事件驱动相关字段
    event_type: str  # 触发事件类型 ("user_input" | "inbox_notification")
    inbox_results: list[dict]  # inbox 通知的结果列表
