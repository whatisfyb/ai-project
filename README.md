# AI Agent Harness

基于 LangChain/LangGraph 构建的多 Agent 协作框架，支持智能任务路由、并行执行、多会话管理与工具扩展。
- Arthur：fangyanbin

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户终端 (TUI)                         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      AgentRegistry                           │
│         生命周期 + Inbox + 能力发现 + 终止控制                 │
└─────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Main Agent   │     │  Plan Agent   │     │    Worker     │
│  (事件驱动)    │     │  (懒加载)     │     │  (并行执行)   │
├───────────────┤     ├───────────────┤     ├───────────────┤
│ Inbox: Queue  │     │ Inbox: Queue  │     │ Inbox: Queue  │
│ State: running│     │ State: pending│     │ State: running│
│ Event Loop    │     │ idle→自动回收  │     │ idle→自动回收  │
└───────────────┘     └───────────────┘     └───────────────┘
         │                     │                     
         ├──────────┬──────────┤                     
         ▼          ▼          ▼                     
┌──────────────┐ ┌──────────────────┐    
│Research Agent│ │ Analysis Agent   │    
│ (懒加载)     │ │  (懒加载)        │    
└──────────────┘ └──────────────────┘    
```

### MainAgent 事件驱动架构

```
┌──────────────────────────────────────────────┐
│           MainAgent Event Loop               │
│                                              │
│   ┌─────────────────┐                        │
│   │  Wait Event     │◄─────────────────────┐ │
│   │  (asyncio.Queue)│                      │ │
│   └────────┬────────┘                      │ │
│            ↓                                │ │
│   ┌─────────────────┐  (仅首次)             │ │
│   │  start_section  │                      │ │
│   └────────┬────────┘                      │ │
│            ↓                                │ │
│   ┌─────────────────┐    有工具调用         │ │
│   │ reason_section   │──────────────┐       │ │
│   └────────┬────────┘              ↓       │ │
│            ↓ 无工具          ┌────────────┐│ │
│   ┌─────────────────┐       │tools_section││ │
│   │ finish_section  │       └──────┬─────┘│ │
│   └────────┬────────┘              │      │ │
│            ↓                       └──→reason_section
│       (回到 Wait)                         │ │
│            └──────────────────────────────┘ │
└──────────────────────────────────────────────┘

事件类型：
- USER_INPUT: 用户输入
- INBOX_NOTIFICATION: Worker/子Agent 结果
- SHUTDOWN: 关闭信号
```

### Agent 状态机

```
┌─────────┐    首次消息    ┌─────────┐    inbox空    ┌─────────┐
│ pending │ ────────────▶ │ running │ ───────────▶ │  idle   │
└─────────┘               └─────────┘              └─────────┘
     ▲                          │                       │
     │                          │ 收到新消息             │ idle 5分钟
     │                          ▼                       │
     │                     ┌─────────┐                  │
     └──────────────────── │ running │ ◀────────────────┘
                           └─────────┘
                                
特殊：Main Agent 无超时限制，永不死亡
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangChain, LangGraph |
| LLM | OpenAI API 兼容接口 |
| 向量存储 | ChromaDB |
| 数据持久化 | SQLite |
| 终端 UI | Textual (TUI) |
| 数据验证 | Pydantic |

## 核心特性

### 1. 统一 Agent Registry

所有 Agent（Main、Subagent、Worker）通过 `AgentRegistry` 统一管理：
- **生命周期管理**: pending → running → idle → pending（自动回收）
- **Inbox 消息队列**: 每个 Agent 拥有独立的 Inbox
- **能力发现**: 通过 `AgentCard.skills` 按技能查找 Agent
- **懒加载**: Subagent 注册时为 pending，首次消息到达时自动激活
- **空闲回收**: 5 分钟无任务的 Agent 自动回收，节省资源

### 2. 事件驱动 MainAgent

MainAgent 重构为事件驱动架构：
- 4 个独立 Section: start, reason, tools, finish
- `asyncio.Queue` 事件循环，可在推理间隙响应外部事件
- Worker 完成任务时自动通过 `Inbox.subscribe()` 通知 MainAgent
- 无需等待整个 ReAct 循环完成

### 3. BaseAgent 标准接口

所有 Agent 继承 `BaseAgent`，实现标准接口：
- `get_card()`: 返回 AgentCard（能力声明）
- `handle_task(task)`: 处理 A2A Task
- `get_state()` / `restore_state()`: 状态保存/恢复（检查点）
- `on_interrupt()`: 中断回调

### 4. 中断与恢复

- **全局中断**: `set_interrupt()` / `is_interrupted()`（向后兼容）
- **单 Agent 中断**: `set_interrupt_for(agent_id)` / `is_interrupted_for(agent_id)`
- **检查点保存**: `save_checkpoint(agent_id, state)` / `load_checkpoint(agent_id)`

### 5. 多 Agent 协作

| Agent | 职责 | 场景 |
|-------|------|------|
| Plan Agent | 任务拆解与规划 | 复杂需求分解为可并行执行的子任务 |
| Research Agent | 信息收集与研究 | 网络搜索、论文检索、知识库查询 |
| Analysis Agent | 数据分析与报告 | 生成分析报告、可视化建议 |
| Worker | 并行任务执行 | 执行 PlanTask，支持工具调用 |

### 6. 并行任务执行

- **Worker Pool**: 多 Worker 并行执行独立任务
- **依赖感知**: 自动识别任务依赖关系，串行执行有依赖的任务
- **健康检查**: 自动检测并替换不健康的 Worker
- **中断恢复**: 支持 Ctrl+C 中断，保存状态后可恢复执行

### 7. 多会话管理

- 会话隔离：每个会话独立的对话历史
- 会话持久化：SQLite 存储会话元数据和消息
- 事件驱动：MainAgent 通过 `chat_async()` 向后兼容旧接口

### 8. 工具扩展

内置工具：
- **Web**: 搜索、网页抓取、爬虫
- **arXiv**: 论文搜索、PDF 下载
- **File**: 读写、编辑、搜索
- **Shell**: 命令执行
- **RAG**: 向量检索

扩展机制：
```python
# tools/custom.py
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """工具描述"""
    return "result"
```

## 目录结构

```
agent/
├── core/               # 核心抽象
│   ├── models.py       # 数据模型 (MainAgentState, Plan, PlanTask)
│   ├── signals.py      # 信号管理（全局中断 + 单 Agent 中断 + 检查点）
│   ├── registry.py      # AgentRegistry（生命周期 + Inbox + 能力发现）
│   ├── events.py       # 事件模型（USER_INPUT, INBOX_NOTIFICATION, SHUTDOWN）
│   └── base_agent.py   # BaseAgent 基类
├── main/               # 主代理（事件驱动架构）
│   ├── agent.py        # MainAgent + 4 Section + Event Loop
│   ├── tools.py        # 工具注册
│   ├── prompts.py      # 系统提示词
│   ├── tui.py          # Textual TUI
│   └── command.py      # 命令处理
├── a2a/                # Agent-to-Agent 协议
│   ├── models.py       # A2A 数据模型
│   ├── transport.py    # Transport 层
│   ├── worker.py       # A2AWorker (继承 BaseAgent)
│   ├── dispatcher.py   # Inbox + TaskResult
│   └── tools.py        # plan_dispatch, job_status, etc.
├── subagents/          # 子代理（懒加载）
│   ├── base.py         # BaseSubagent 基类
│   ├── plan_agent.py   # Plan Agent (继承 BaseAgent)
│   ├── research_agent.py # Research Agent (继承 BaseAgent)
│   └── analysis_agent.py # Analysis Agent (继承 BaseAgent)
├── middleware/          # 中间件
│   ├── context_compact.py # 上下文压缩
│   ├── token_count.py  # Token 计数
│   └── long_term_memory.py # 长期记忆
├── bootstrap.py        # Agent 初始化（注册 + 启动）
store/                  # 持久化层
├── session.py          # 会话存储
└── plan.py             # 计划存储
tools/                  # 工具层
├── agent.py            # agent, agent_dispatch, agent_list, agent_status
└── web.py              # Web 搜索
skills/                 # 技能扩展
tests/                  # 测试
├── test_core/          # 核心测试（registry, signals）
└── test_integration/   # 集成测试（agent_communication）
```

## 快速开始

```bash
# 安装依赖
uv sync

# 配置 LLM (config.yaml)
llm:
  models:
    - name: OpenAI
      model: gpt-4
      api_key: sk-xxx
      base_url: https://api.openai.com/v1

# 启动
python main.py
```

## 命令

| 命令 | 说明 |
|------|------|
| `/new` | 创建新会话 |
| `/sessions` | 列出所有会话 |
| `/resume <id>` | 切换到指定会话 |
| `/history` | 查看当前会话历史 |
| `/status` | 查看运行状态 |
| `/exit` | 退出 |

## API 设计

### AgentRegistry

```python
from agent.core.registry import get_registry, AgentLifecycleState

registry = get_registry()

# 注册 Agent
registry.register(
    agent_id="plan-agent",
    agent_type="subagent",
    card=plan_card,
    factory=lambda: PlanAgent(),
    auto_start=False,  # 懒加载
)

# 发送消息（激活 Agent）
registry.send_message("plan-agent", task)

# 按技能查找
agents = registry.find_agents_by_skill("plan")

# 查询状态
state = registry.get_state("plan-agent")  # -> AgentLifecycleState.PENDING
```

### Signals

```python
from agent.core.signals import (
    set_interrupt, is_interrupted,           # 全局中断（向后兼容）
    set_interrupt_for, is_interrupted_for,   # 单 Agent 中断
    save_checkpoint, load_checkpoint,        # 检查点
)

# 全局中断
set_interrupt()
assert is_interrupted() is True

# 单 Agent 中断
set_interrupt_for("plan-agent")
assert is_interrupted_for("plan-agent") is True

# 检查点
save_checkpoint("plan-agent", {"step": 3})
cp = load_checkpoint("plan-agent")
```

### Events

```python
from agent.core.events import AgentEvent, EventType

# 用户输入事件
event = AgentEvent.user_input("hello", "thread-1")

# Inbox 通知事件
event = AgentEvent.inbox_notification(
    task_id="task-1", status="success", result="done"
)

# 关闭事件
event = AgentEvent.shutdown()
```