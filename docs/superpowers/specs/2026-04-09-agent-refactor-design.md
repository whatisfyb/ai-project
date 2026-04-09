# Agent 目录重构设计文档

## 概述

对 `agent/` 目录进行深度重构，按职责分层组织代码，优化模块边界和依赖关系。

## 当前问题

| 问题 | 说明 |
|------|------|
| main_agent.py 职责过多 | 包含 MainAgent、REPL、系统提示词、上下文管理，文件 630+ 行 |
| 中断信号管理分散 | `_interrupt_event` 在 worker.py，但 main_agent.py 也引用 |
| 存储层位置不清 | session_store.py 和 plan_store.py 放在 agent/ 下不合理 |
| 工具初始化重复 | main_agent.py 和 worker.py 都有类似逻辑 |
| 缺乏分层 | 核心、存储、执行器混在一起 |
| 循环依赖风险 | registry 被 main 和 executor 共同依赖 |

## 重构目标

1. 按职责分层，目录结构清晰
2. 拆分大文件，单一职责
3. 存储层提升到项目级别
4. 解耦循环依赖
5. 统一数据文件路径到 `.data/`

## 新目录结构

```
agent/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── models.py       # Task, Plan, MainAgentState
│   ├── signals.py      # 中断信号管理
│   └── registry.py     # AgentRegistry（依赖注入解耦）
├── main/
│   ├── __init__.py
│   ├── agent.py        # MainAgent 类
│   ├── tools.py        # 工具注册
│   ├── prompts.py      # 系统提示词
│   └── repl.py         # REPL 循环
├── executor/
│   ├── __init__.py
│   ├── executor.py     # PlanExecutor, WorkerRegistry, ProgressTracker
│   └── worker.py       # TaskWorker
└── subagents/
    ├── __init__.py
    ├── base.py
    ├── plan_agent.py
    ├── research_agent.py
    └── analysis_agent.py

store/
├── __init__.py
├── session.py          # SessionStore
└── plan.py             # PlanStore

.data/                  # 运行时数据（.gitignore）
├── sessions.db
└── plans.db
```

## 文件迁移映射

| 原文件 | 新文件 | 说明 |
|--------|--------|------|
| agent/models.py | agent/core/models.py | 移动 |
| agent/state.py | agent/core/models.py | 合并到 models.py |
| agent/worker.py（中断部分） | agent/core/signals.py | 抽出独立模块 |
| agent/registry.py | agent/core/registry.py | 重写接口 |
| agent/main_agent.py（MainAgent类） | agent/main/agent.py | 拆分 |
| agent/main_agent.py（工具初始化） | agent/main/tools.py | 拆分 |
| agent/main_agent.py（提示词） | agent/main/prompts.py | 拆分 |
| agent/main_agent.py（REPL） | agent/main/repl.py | 拆分 |
| agent/executor.py | agent/executor/executor.py | 移动 |
| agent/worker.py（TaskWorker类） | agent/executor/worker.py | 移动 |
| agent/session_store.py | store/session.py | 移动 |
| agent/plan_store.py | store/plan.py | 移动 |

## 模块详细设计

### 1. agent/core/models.py

**职责**：定义所有数据模型

```python
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
import operator
from pydantic import BaseModel, Field


class Task(BaseModel):
    """单个任务"""
    id: str = Field(description="任务唯一标识")
    description: str = Field(description="任务详细描述")
    dependencies: list[str] = Field(default=[], description="依赖的任务 ID 列表")
    status: Literal["pending", "completed", "failed"] = Field(default="pending")
    result: str | None = Field(default=None)
    claimed_by: str | None = Field(default=None)


class Plan(BaseModel):
    """执行计划"""
    goal: str = Field(description="整体目标")
    tasks: list[Task] = Field(description="任务列表")
    status: Literal["pending", "completed", "failed"] = Field(default="pending")
    summarized_result: str | None = Field(default=None)


class MainAgentState(TypedDict):
    """Main Agent 状态"""
    messages: Annotated[list[dict], operator.add]
    current_task: str | None
    memory_context: str | None
    subagent_results: dict[str, Any]
```

**依赖**：无

---

### 2. agent/core/signals.py

**职责**：全局中断信号管理

```python
import threading

_interrupt_event = threading.Event()


def set_interrupt():
    """设置中断信号"""
    _interrupt_event.set()


def clear_interrupt():
    """清除中断信号"""
    _interrupt_event.clear()


def is_interrupted() -> bool:
    """检查是否被中断"""
    return _interrupt_event.is_set()
```

**依赖**：无

---

### 3. agent/core/registry.py

**职责**：管理运行中的 Executor，通过依赖注入解耦

```python
import threading
from typing import Callable

class AgentRegistry:
    """全局 Agent 注册表"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._terminators = {}  # plan_id -> terminate_fn
                    cls._instance._running = False
        return cls._instance

    def register(self, plan_id: str, terminate_fn: Callable[[], None]) -> None:
        """注册一个 Executor 的终止函数"""
        with self._lock:
            self._terminators[plan_id] = terminate_fn
            self._running = True

    def unregister(self, plan_id: str) -> None:
        """注销一个 Executor"""
        with self._lock:
            self._terminators.pop(plan_id, None)
            if not self._terminators:
                self._running = False

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_running_plan_ids(self) -> list[str]:
        with self._lock:
            return list(self._terminators.keys())

    def terminate_all(self) -> list[str]:
        """终止所有 Executor"""
        terminated = []
        with self._lock:
            for plan_id, terminate_fn in self._terminators.items():
                try:
                    terminate_fn()
                    terminated.append(plan_id)
                except Exception:
                    pass
            self._terminators.clear()
            self._running = False
        return terminated


agent_registry = AgentRegistry()


def terminate() -> list[str]:
    return agent_registry.terminate_all()
```

**依赖**：无（通过回调解耦）

---

### 4. agent/main/agent.py

**职责**：MainAgent 核心逻辑

**导入关系**：
```python
from agent.core.models import MainAgentState
from agent.core.signals import is_interrupted
from agent.main.tools import get_main_agent_tools
from agent.main.prompts import MAIN_AGENT_PROMPT
from store.session import SessionStore
```

**导出**：`MainAgent`, `create_main_agent`

---

### 5. agent/main/tools.py

**职责**：工具注册与初始化

```python
def get_main_agent_tools() -> list:
    """获取 Main Agent 可用的工具列表"""
    from tools.web import (...)
    from tools.agent import (...)
    from tools.skills import (...)
    from tools.task import (...)
    from tools.grep import (...)
    from tools.read import (...)
    from tools.write import (...)
    from tools.edit import (...)
    from tools.bash import (...)
    from tools.glob import (...)

    return [...]
```

**导出**：`get_main_agent_tools`

---

### 6. agent/main/prompts.py

**职责**：系统提示词定义

**导出**：`MAIN_AGENT_PROMPT`

---

### 7. agent/main/repl.py

**职责**：REPL 交互循环

**导入关系**：
```python
from agent.main.agent import create_main_agent
from agent.core.registry import agent_registry, terminate
from agent.core.signals import set_interrupt, clear_interrupt, is_interrupted
from store.session import SessionStore
```

**导出**：`run_repl`

---

### 8. agent/executor/executor.py

**职责**：Plan 执行协调器

**导入关系**：
```python
from agent.core.models import Task, Plan
from agent.core.signals import set_interrupt, clear_interrupt, is_interrupted
from agent.core.registry import agent_registry
from agent.executor.worker import TaskWorker
from store.plan import PlanStore
```

**导出**：`PlanExecutor`, `WorkerRegistry`, `ProgressTracker`

---

### 9. agent/executor/worker.py

**职责**：任务执行 Worker

**导入关系**：
```python
from agent.core.models import Task
from agent.core.signals import is_interrupted
from store.plan import PlanStore
```

**导出**：`TaskWorker`

---

### 10. store/session.py

**职责**：会话存储

**数据库路径**：`.data/sessions.db`

**导入关系**：
```python
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent.parent / ".data" / "sessions.db"
```

**导出**：`SessionStore`

---

### 11. store/plan.py

**职责**：Plan 存储

**数据库路径**：`.data/plans.db`

**导入关系**：
```python
from agent.core.models import Plan, Task

DEFAULT_DB_PATH = Path(__file__).parent.parent / ".data" / "plans.db"
```

**导出**：`PlanStore`, `PlanRecord`

---

## 依赖关系图

```
store/plan.py ──────┐
                    │
                    ▼
agent/core/models.py ◄────── agent/core/signals.py
        │                           │
        │                           │
        ▼                           ▼
agent/core/registry.py         agent/executor/worker.py
        │                           │
        │                           │
        ▼                           ▼
agent/executor/executor.py ◄───────┘
        │
        │
        ▼
agent/main/agent.py ◄─── agent/main/tools.py
        │                 agent/main/prompts.py
        │
        ▼
agent/main/repl.py
```

**特点**：
- 单向依赖，无循环
- core 层无外部依赖
- store 只依赖 core/models

---

## 导出设计

### agent/__init__.py

```python
from agent.core.models import MainAgentState, Task, Plan
from agent.main.agent import MainAgent, create_main_agent
from agent.main.repl import run_repl

__all__ = [
    "MainAgentState",
    "Task",
    "Plan",
    "MainAgent",
    "create_main_agent",
    "run_repl",
]
```

### agent/core/__init__.py

```python
from agent.core.models import Task, Plan, MainAgentState
from agent.core.signals import set_interrupt, clear_interrupt, is_interrupted
from agent.core.registry import agent_registry, terminate

__all__ = [
    "Task",
    "Plan",
    "MainAgentState",
    "set_interrupt",
    "clear_interrupt",
    "is_interrupted",
    "agent_registry",
    "terminate",
]
```

### agent/main/__init__.py

```python
from agent.main.agent import MainAgent, create_main_agent
from agent.main.repl import run_repl

__all__ = [
    "MainAgent",
    "create_main_agent",
    "run_repl",
]
```

### agent/executor/__init__.py

```python
from agent.executor.executor import PlanExecutor, WorkerRegistry, ProgressTracker
from agent.executor.worker import TaskWorker

__all__ = [
    "PlanExecutor",
    "WorkerRegistry",
    "ProgressTracker",
    "TaskWorker",
]
```

### store/__init__.py

```python
from store.session import SessionStore
from store.plan import PlanStore, PlanRecord

__all__ = [
    "SessionStore",
    "PlanStore",
    "PlanRecord",
]
```

---

## .gitignore 更新

```gitignore
# 运行时数据
.data/
```

---

## 实施步骤

### 阶段一：创建新结构

1. 创建目录结构
2. 创建 store/ 目录
3. 更新 .gitignore

### 阶段二：迁移 core 层

1. 创建 agent/core/models.py（合并 models.py + state.py）
2. 创建 agent/core/signals.py
3. 创建 agent/core/registry.py（重写）
4. 创建 agent/core/__init__.py

### 阶段三：迁移 store 层

1. 创建 store/session.py
2. 创建 store/plan.py
3. 创建 store/__init__.py

### 阶段四：迁移 executor 层

1. 创建 agent/executor/executor.py
2. 创建 agent/executor/worker.py
3. 创建 agent/executor/__init__.py

### 阶段五：迁移 main 层

1. 创建 agent/main/prompts.py
2. 创建 agent/main/tools.py
3. 创建 agent/main/agent.py
4. 创建 agent/main/repl.py
5. 创建 agent/main/__init__.py

### 阶段六：迁移 subagents 层

1. 移动 agent/subagents/ 内容，更新导入

### 阶段七：更新顶层导出

1. 更新 agent/__init__.py

### 阶段八：清理旧文件

1. 删除旧的 agent/*.py 文件
2. 删除 agent/models.py, agent/state.py 等

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 导入路径变更导致遗漏 | 全局搜索旧路径，确保全部更新 |
| 运行时发现遗漏的依赖 | 逐步迁移，每层完成后测试 |
| 数据库路径变更 | 确保新路径自动创建目录 |

---

## 验收标准

1. 目录结构符合设计
2. 所有导入路径正确
3. 运行 `python -m agent.main.repl` 可正常启动
4. 多会话管理正常
5. Plan 执行和中断功能正常
6. 子代理调用正常
