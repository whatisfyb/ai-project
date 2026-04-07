# Main Agent 设计文档

## 概述

构建一个直接与用户交互的主代理（Main Agent），能够智能路由任务到子代理或直接调用工具，支持长期记忆。

## 需求总结

| 维度 | 决策 |
|------|------|
| **定位** | 直接与用户交互的主代理，智能路由任务 |
| **交互方式** | 终端命令行 REPL |
| **决策机制** | LLM 自行决策 |
| **子代理** | Plan、Research、Analysis + Fork 机制 |
| **记忆** | 长期记忆（向量存储持久化） |
| **工具** | 复用现有全部工具 |

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户终端                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Main Agent                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    REPL Loop                         │    │
│  │  用户输入 → 意图理解 → 路由决策 → 执行 → 响应         │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│         ┌────────────────────┼────────────────────┐         │
│         ▼                    ▼                    ▼         │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │
│  │   Plan      │      │  Research   │      │  Analysis   │  │
│  │  (LangGraph)│      │ (LangGraph) │      │ (LangGraph) │  │
│  └─────────────┘      └─────────────┘      └─────────────┘  │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Fork Worker (共享上下文)                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 工具层 (Tools)                        │    │
│  │  tavily | arxiv | firecrawl | vector_store | ...    │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              记忆层 (Memory)                          │    │
│  │              长期记忆 (VectorStore)                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块设计

### 1. REPL Loop

**职责**：管理用户交互循环

**功能**：
- 接收用户输入
- 显示 Agent 响应（支持流式输出）
- 管理会话生命周期（开始/结束/中断）
- 处理特殊命令（如 `/exit`, `/clear`, `/help`）

**实现文件**：`main_agent/repl.py`

### 2. Main Agent（ReAct 模式）

**职责**：智能路由和任务执行

**核心流程**：
```
Thought → Action → Observation → Thought → ...
```

**能力**：
- 理解用户意图
- 选择合适的工具或子代理
- 执行并收集结果
- 生成响应

**实现文件**：`main_agent/agent.py`

### 3. 子代理（LangGraph 状态机）

#### 3.1 Plan Agent

**职责**：复杂任务拆解

**能力**：
- 分析复杂需求
- 拆解为可执行的子任务
- 生成执行计划
- 识别依赖关系

**可用工具**：Read, Grep, Glob（只读）

**状态流转**：
```
接收任务 → 分析需求 → 探索代码库 → 生成计划 → 输出
```

**实现文件**：`main_agent/subagents/plan_agent.py`

#### 3.2 Research Agent

**职责**：信息收集与研究

**能力**：
- 网络搜索
- arXiv 论文搜索与下载
- RAG 知识库检索
- 信息整合

**可用工具**：tavily_search, arxiv_search, firecrawl, vector_store, retriever

**状态流转**：
```
接收任务 → 确定搜索策略 → 并行搜索 → 整合结果 → 输出
```

**实现文件**：`main_agent/subagents/research_agent.py`

#### 3.3 Analysis Agent

**职责**：数据分析与报告生成

**能力**：
- 数据分析
- 生成分析报告
- 可视化建议

**可用工具**：Read, Write, vector_store, retriever

**状态流转**：
```
接收任务 → 收集数据 → 分析处理 → 生成报告 → 输出
```

**实现文件**：`main_agent/subagents/analysis_agent.py`

### 4. Fork 机制

**职责**：共享上下文的并行任务执行

**能力**：
- 复制主代理状态
- 独立执行子任务
- 结果回传主代理

**实现文件**：`main_agent/fork.py`

### 5. 工具层

**职责**：统一工具接口

**现有工具集成**：
| 工具 | 文件 | 功能 |
|------|------|------|
| tavily_search | `tools/tavily.py` | 网络搜索 |
| tavily_extract | `tools/tavily.py` | URL 内容提取 |
| arxiv_search | `tools/arxiv_search.py` | arXiv 论文搜索 |
| firecrawl | `tools/firecrawl.py` | 网页爬取 |
| vector_store | `utils/vector_store.py` | 向量存储 |
| retriever | `utils/retriever.py` | RAG 检索 |

**实现文件**：`main_agent/tools.py`

### 6. 记忆层

**职责**：长期记忆存储和检索

**功能**：
- 存储对话历史
- 存储用户偏好
- 语义检索相关记忆
- 记忆压缩与摘要

**存储内容**：
- 用户消息和 Agent 响应
- 重要决策和结论
- 用户偏好设置

**实现文件**：`main_agent/memory.py`

## 实现步骤

### 阶段一：基础框架（Day 1）

#### Step 1.1：创建项目结构

```
main_agent/
├── __init__.py
├── agent.py          # Main Agent 核心逻辑
├── repl.py           # REPL 交互循环
├── state.py          # 状态定义
├── tools.py          # 工具注册与封装
├── memory.py         # 记忆管理
├── fork.py           # Fork 机制
├── prompts.py        # 系统提示词
└── subagents/
    ├── __init__.py
    ├── base.py       # 子代理基类
    ├── plan_agent.py
    ├── research_agent.py
    └── analysis_agent.py
```

#### Step 1.2：实现状态定义 (`state.py`)

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class MainAgentState(TypedDict):
    """Main Agent 状态"""
    messages: Annotated[list, operator.add]  # 消息历史
    current_task: str | None                 # 当前任务
    memory_context: str | None               # 记忆上下文
    subagent_results: dict                   # 子代理结果
```

#### Step 1.3：实现工具注册 (`tools.py`)

- 封装现有工具为 LangChain Tool 格式
- 注册所有可用工具
- 提供工具描述供 LLM 决策

### 阶段二：Main Agent 核心（Day 2）

#### Step 2.1：实现系统提示词 (`prompts.py`)

- Main Agent 角色定义
- 可用工具和子代理描述
- 决策规则

#### Step 2.2：实现 Main Agent (`agent.py`)

- ReAct 循环实现
- 工具调用逻辑
- 子代理派发逻辑
- 响应生成

#### Step 2.3：实现 REPL (`repl.py`)

- 用户输入处理
- 流式输出显示
- 特殊命令处理
- 会话管理

### 阶段三：子代理实现（Day 3-4）

#### Step 3.1：实现子代理基类 (`subagents/base.py`)

- LangGraph 状态机基类
- 通用节点和边
- 结果返回格式

#### Step 3.2：实现 Plan Agent (`subagents/plan_agent.py`)

- 状态定义
- 节点实现（分析、探索、规划）
- 图构建

#### Step 3.3：实现 Research Agent (`subagents/research_agent.py`)

- 状态定义
- 节点实现（搜索、下载、RAG）
- 图构建

#### Step 3.4：实现 Analysis Agent (`subagents/analysis_agent.py`)

- 状态定义
- 节点实现（收集、分析、报告）
- 图构建

### 阶段四：记忆系统（Day 5）

#### Step 4.1：实现记忆管理 (`memory.py`)

- 对话历史存储
- 语义检索
- 记忆压缩

#### Step 4.2：集成到 Main Agent

- 会话开始时加载相关记忆
- 会话结束时保存记忆

### 阶段五：Fork 机制（Day 6）

#### Step 5.1：实现 Fork (`fork.py`)

- 状态复制
- 独立执行
- 结果回传

### 阶段六：集成测试（Day 7）

#### Step 6.1：端到端测试

- REPL 交互测试
- 子代理调用测试
- 记忆持久化测试

#### Step 6.2：入口文件

```python
# main.py
from main_agent.repl import run_repl

if __name__ == "__main__":
    run_repl()
```

## 文件清单

| 文件路径 | 职责 |
|----------|------|
| `main_agent/__init__.py` | 模块初始化 |
| `main_agent/state.py` | 状态定义 |
| `main_agent/prompts.py` | 系统提示词 |
| `main_agent/tools.py` | 工具注册与封装 |
| `main_agent/memory.py` | 记忆管理 |
| `main_agent/agent.py` | Main Agent 核心逻辑 |
| `main_agent/repl.py` | REPL 交互循环 |
| `main_agent/fork.py` | Fork 机制 |
| `main_agent/subagents/__init__.py` | 子代理模块初始化 |
| `main_agent/subagents/base.py` | 子代理基类 |
| `main_agent/subagents/plan_agent.py` | Plan 子代理 |
| `main_agent/subagents/research_agent.py` | Research 子代理 |
| `main_agent/subagents/analysis_agent.py` | Analysis 子代理 |
| `main.py` | 入口文件 |

## 技术栈

- **LangChain** - 工具封装和 Agent 框架
- **LangGraph** - 子代理状态机
- **ChromaDB** - 向量存储（记忆系统）
- **Rich** - 终端美化输出

## 后续扩展

- [ ] 添加更多子代理（Code、Test 等）
- [ ] 支持多模态输入（图片、文件）
- [ ] Web UI 界面
- [ ] Agent 性能监控
