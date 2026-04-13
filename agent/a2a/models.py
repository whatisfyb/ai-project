"""A2A (Agent-to-Agent) 协议数据模型

基于 Google A2A 协议规范，用于 Agent 间通信。
参考: https://github.com/google/A2A

核心概念：
- AgentCard: Agent 能力声明，用于发现和匹配
- Task: Agent 间传递的工作单元，有独立生命周期
- Message: 通信消息，包含角色和内容部分
- Artifact: 任务产生的产物（文件、数据等）
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


# ============ Task 状态 ============

class TaskStatus(str, Enum):
    """A2A Task 状态机

    状态流转：
    pending → working → completed
                      ↘ failed
                      ↘ cancelled
                      ↘ rejected
    """
    PENDING = "pending"      # 已创建，等待处理
    WORKING = "working"      # 正在处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 执行失败
    CANCELLED = "cancelled"  # 用户取消
    REJECTED = "rejected"    # Agent 拒绝处理


# ============ Message 部分 ============

class PartType(str, Enum):
    """Message Part 类型"""
    TEXT = "text"           # 文本内容
    FILE = "file"           # 文件引用
    DATA = "data"           # 结构化数据
    PLANTASK = "plantask"   # PlanTask 引用（项目特有）


class Part(BaseModel):
    """Message 的组成部分

    A2A 消息由多个 Part 组成，每个 Part 可以是不同类型的内容。
    """
    type: PartType = Field(description="内容类型")
    content: Any = Field(description="内容数据")
    metadata: dict[str, Any] | None = Field(default=None, description="元数据")

    @classmethod
    def text(cls, text: str, metadata: dict | None = None) -> "Part":
        """创建文本 Part"""
        return cls(type=PartType.TEXT, content=text, metadata=metadata)

    @classmethod
    def file(cls, url: str, mime_type: str | None = None, name: str | None = None) -> "Part":
        """创建文件 Part"""
        return cls(
            type=PartType.FILE,
            content={"url": url, "mime_type": mime_type, "name": name}
        )

    @classmethod
    def data(cls, data: dict | list, schema: dict | None = None) -> "Part":
        """创建数据 Part"""
        return cls(type=PartType.DATA, content=data, metadata={"schema": schema} if schema else None)

    @classmethod
    def plantask(cls, plan_id: str, task_id: str) -> "Part":
        """创建 PlanTask 引用 Part（项目特有）"""
        return cls(
            type=PartType.PLANTASK,
            content={"plan_id": plan_id, "task_id": task_id}
        )


# ============ Message ============

class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"       # 用户/调用方
    AGENT = "agent"     # Agent/被调用方


class Message(BaseModel):
    """A2A Message - Agent 间通信的基本单元

    消息包含角色和多个内容部分，支持文本、文件、结构化数据等。
    """
    role: MessageRole = Field(description="消息角色")
    parts: list[Part] = Field(default_factory=list, description="消息内容部分")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    @classmethod
    def user_text(cls, text: str) -> "Message":
        """创建用户文本消息"""
        return cls(role=MessageRole.USER, parts=[Part.text(text)])

    @classmethod
    def agent_text(cls, text: str) -> "Message":
        """创建 Agent 文本消息"""
        return cls(role=MessageRole.AGENT, parts=[Part.text(text)])

    def get_text(self) -> str:
        """获取所有文本内容"""
        texts = [p.content for p in self.parts if p.type == PartType.TEXT]
        return "\n".join(texts)


# ============ Artifact ============

class Artifact(BaseModel):
    """A2A Artifact - 任务产生的产物

    任务执行过程中产生的文件、数据等产物。
    """
    id: str = Field(description="产物唯一标识")
    name: str = Field(description="产物名称")
    mime_type: str | None = Field(default=None, description="MIME 类型")
    content: Any = Field(description="产物内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


# ============ A2A Task ============

class Task(BaseModel):
    """A2A Task - Agent 间传递的工作单元

    Task 是 A2A 协议的核心概念，代表一个独立的工作单元。
    它有自己的生命周期、状态机和消息历史。

    与 PlanTask 的关系：
    - PlanTask 是 Plan 的静态执行单元，持久化在 SQLite
    - A2A Task 是动态的通信单元，在 Agent 间流转
    - 一个 A2A Task 可以对应一个或多个 PlanTask 的执行
    """
    id: str = Field(description="Task 唯一标识")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="当前状态")
    history: list[Message] = Field(default_factory=list, description="消息历史")
    artifacts: list[Artifact] = Field(default_factory=list, description="产生的产物")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

    # 关联信息
    sender_id: str | None = Field(default=None, description="发送方 Agent ID")
    receiver_id: str | None = Field(default=None, description="接收方 Agent ID")

    # PlanTask 关联（项目特有）
    plan_id: str | None = Field(default=None, description="关联的 Plan ID")
    plantask_id: str | None = Field(default=None, description="关联的 PlanTask ID")

    def add_message(self, message: Message) -> None:
        """添加消息到历史"""
        self.history.append(message)
        self.updated_at = datetime.now()

    def add_artifact(self, artifact: Artifact) -> None:
        """添加产物"""
        self.artifacts.append(artifact)
        self.updated_at = datetime.now()

    def update_status(self, status: TaskStatus) -> None:
        """更新状态"""
        self.status = status
        self.updated_at = datetime.now()

    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.REJECTED,
        )


# ============ Agent Card ============

class Skill(BaseModel):
    """Agent 技能描述"""
    name: str = Field(description="技能名称")
    description: str = Field(description="技能描述")
    input_schema: dict | None = Field(default=None, description="输入参数 Schema")
    output_schema: dict | None = Field(default=None, description="输出参数 Schema")


class AgentCapabilities(BaseModel):
    """Agent 能力标志"""
    text: bool = Field(default=True, description="支持文本")
    files: bool = Field(default=False, description="支持文件")
    streaming: bool = Field(default=False, description="支持流式输出")
    push_notifications: bool = Field(default=False, description="支持推送通知")


class AgentCard(BaseModel):
    """A2A Agent Card - Agent 能力声明

    用于 Agent 发现和能力匹配。每个 Agent 都应该提供自己的 Card。
    类似于 Web 服务 /.well-known/agent.json 的概念。
    """
    id: str = Field(description="Agent 唯一标识")
    name: str = Field(description="Agent 名称")
    description: str = Field(description="Agent 描述")
    version: str = Field(default="1.0.0", description="版本号")
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities, description="能力标志")
    skills: list[Skill] = Field(default_factory=list, description="技能列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="扩展元数据")

    def has_skill(self, skill_name: str) -> bool:
        """检查是否拥有指定技能"""
        return any(s.name == skill_name for s in self.skills)


# ============ Transport 消息 ============

class TransportMessage(BaseModel):
    """Transport 层消息封装

    用于 Transport 层传递的内部消息格式。
    """
    task: Task = Field(description="关联的 A2A Task")
    message: Message = Field(description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


# ============ 事件类型 ============

class TaskEvent(str, Enum):
    """Task 事件类型"""
    CREATED = "created"           # Task 创建
    DISPATCHED = "dispatched"     # Task 已分发
    STATUS_CHANGED = "status_changed"  # 状态变化
    MESSAGE_ADDED = "message_added"    # 新消息
    ARTIFACT_ADDED = "artifact_added"  # 新产物
    COMPLETED = "completed"       # Task 完成
    FAILED = "failed"             # Task 失败
