"""Agent 数据模型"""

from typing import Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    """单个任务"""
    id: str = Field(description="任务唯一标识，如 T1, T2, T3")
    description: str = Field(description="任务详细描述")
    dependencies: list[str] = Field(default=[], description="依赖的任务 ID 列表")
    status: Literal["pending", "completed", "failed"] = Field(
        default="pending", description="任务状态"
    )
    result: str | None = Field(default=None, description="任务执行结果")


class Plan(BaseModel):
    """执行计划"""
    goal: str = Field(description="整体目标")
    tasks: list[Task] = Field(description="任务列表")
    status: Literal["pending", "completed", "failed"] = Field(
        default="pending", description="计划状态"
    )
