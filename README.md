# AI Agent Harness

基于 LangChain/LangGraph 构建的多 Agent 协作框架，支持智能任务路由、并行执行、多会话管理与工具扩展。

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户终端 (REPL)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Main Agent                              │
│              (ReAct 推理模式 + 智能路由)                       │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Plan Agent  │      │Research Agent│     │Analysis Agent│
│ (LangGraph) │      │ (LangGraph)  │     │ (LangGraph)  │
└─────────────┘      └─────────────┘      └─────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Executor (Worker Pool)                     │
│              并行任务执行 + 健康检查 + 中断恢复                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Tools Layer                           │
│    Web Search | arXiv | Firecrawl | RAG | File | Shell      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Store Layer                             │
│         Session Store | Plan Store (SQLite 持久化)           │
└─────────────────────────────────────────────────────────────┘
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangChain, LangGraph |
| LLM | OpenAI API 兼容接口 |
| 向量存储 | ChromaDB |
| 数据持久化 | SQLite |
| 终端 UI | Rich |
| 数据验证 | Pydantic |

## 核心特性

### 1. 智能路由 (Main Agent)

基于 ReAct 推理模式，自动判断用户意图并路由到合适的处理方式：
- 简单对话 → 直接回复
- 信息搜索 → 调用搜索工具
- 复杂任务 → 分发到子代理

### 2. 多 Agent 协作

| Agent | 职责 | 场景 |
|-------|------|------|
| Plan Agent | 任务拆解与规划 | 复杂需求分解为可并行执行的子任务 |
| Research Agent | 信息收集与研究 | 网络搜索、论文检索、知识库查询 |
| Analysis Agent | 数据分析与报告 | 生成分析报告、可视化建议 |

### 3. 并行任务执行

- **Worker Pool**: 多 Worker 并行执行独立任务
- **依赖感知**: 自动识别任务依赖关系，串行执行有依赖的任务
- **健康检查**: 自动检测并替换不健康的 Worker
- **中断恢复**: 支持 Ctrl+C 中断，保存状态后可恢复执行

### 4. 多会话管理

- 会话隔离：每个会话独立的对话历史
- 会话持久化：SQLite 存储会话元数据和消息
- 工具调用上下文：完整保存 AIMessage(tool_calls) + ToolMessage 链

### 5. 工具扩展

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
├── core/           # 核心抽象
│   ├── models.py   # 数据模型 (Task, Plan, State)
│   ├── signals.py  # 中断信号管理
│   └── registry.py # Agent 注册表
├── main/           # 主代理
│   ├── agent.py    # MainAgent 实现
│   ├── tools.py    # 工具注册
│   ├── prompts.py  # 系统提示词
│   └── repl.py     # 终端交互
├── executor/       # 执行器
│   ├── executor.py # PlanExecutor + WorkerPool
│   └── worker.py   # TaskWorker
└── subagents/      # 子代理
    ├── plan_agent.py
    ├── research_agent.py
    └── analysis_agent.py

store/              # 持久化层
├── session.py      # 会话存储
└── plan.py         # 计划存储

tools/              # 工具层
skills/             # 技能扩展
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

---

## 项目版本

**AI Agent Harness 智能代理协作框架**  
LangChain | LangGraph | SQLite | ChromaDB | Rich  
2025.04 - 至今

基于 LangGraph 状态机与多 Agent 协作构建的智能任务执行框架，支持复杂任务的自动拆解、并行执行与中断恢复。系统围绕 Main Agent（智能路由）、Plan Agent（任务规划）、Research Agent（信息收集）、Analysis Agent（数据分析）等核心代理进行能力拆分，覆盖用户对话、工具调用、任务执行、会话管理等全流程场景。

**工作内容：**

针对复杂任务的自动拆解与执行，基于 LangGraph 状态机构建 Plan Agent，采用依赖拓扑排序实现任务并行度分析，结合 Worker Pool 与 SQLite 原子锁，任务执行成功率提升至 95%+

针对多轮对话的工具调用上下文恢复问题，设计 AIMessage(tool_calls) + ToolMessage 完整消息链持久化方案，基于 SQLite + JSON 序列化实现会话级状态保存，对话上下文恢复准确率达 100%

针对 LLM 推理过程中的用户中断需求，采用 Threading.Event 实现全局中断信号机制，结合 AgentRegistry 依赖注入模式，实现跨组件的优雅中断与状态保存

针对多样化外部工具集成需求，基于 LangChain Tool 抽象设计统一工具接口，支持 Web 搜索、arXiv 论文、Firecrawl 爬虫、RAG 检索等 20+ 工具的即插即用扩展
