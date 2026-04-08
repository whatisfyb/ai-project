# Plan Agent 执行器设计文档

## 概述

Plan Agent 用于将复杂任务拆解为可执行的子任务，支持持久化、中断恢复、并行执行。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Main Agent                            │
│  - messages: [用户对话历史...]                            │
│  - 发现复杂任务 → 调用 Plan Agent                         │
│  - 创建 PlanExecutor → 等待完成 → 汇总                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Plan Agent       │ → 生成 Plan (保存到数据库)
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   PlanExecutor      │ → 创建 Worker 线程池
              └─────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Worker 1 │   │ Worker 2 │   │ Worker 3 │
    │ (线程)   │   │ (线程)   │   │ (线程)   │
    └──────────┘   └──────────┘   └──────────┘
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                   SQLite 数据库
              (Plan + Tasks + 状态 + 结果)
                         │
                         ▼
              ┌─────────────────────┐
              │  Analysis Agent     │ → 汇总结果（Main Agent 调用）
              └─────────────────────┘
```

## 核心原则

**Worker 不是 MainAgent 副本！**

- ❌ 错误：fork MainAgent → 会继续 fork，无限递归
- ✅ 正确：创建专门的 Worker 线程 → 只执行任务，不 fork

Worker 是一个轻量的任务执行器：
- 有独立的 memory（初始化时复制）
- 有执行函数 execute_fn
- 不会创建新的 Plan 或 fork

## 数据模型

### Task

```python
class Task(BaseModel):
    id: str                          # 任务唯一标识 (T1, T2, T3...)
    description: str                 # 任务详细描述
    dependencies: list[str]          # 依赖的任务 ID 列表
    status: Literal["pending", "completed", "failed"]  # 任务状态
    result: str | None               # 任务执行结果
    claimed_by: str | None           # 领取该任务的 Worker ID
```

### Plan

```python
class Plan(BaseModel):
    goal: str                        # 整体目标
    tasks: list[Task]                # 任务列表
    status: Literal["pending", "completed", "failed"]  # 计划状态
    summarized_result: str | None    # 最终汇总结果
```

## 核心流程

### 1. 生成 Plan

```python
plan_agent = PlanAgent()
plan, plan_id = plan_agent.run(task="帮我写一个爬虫")
# Plan 自动保存到数据库
```

### 2. 并行执行

```python
# Main Agent 创建执行器，等待完成
executor = PlanExecutor(
    plan_id=plan_id,
    memory=main_agent.messages,  # 初始化记忆
    execute_fn=execute_task,     # 任务执行函数
    num_workers=2,               # 创建 2 个 Worker 线程
)
result = executor.run()  # 阻塞等待全部完成
```

### 3. Worker 工作流程

**Worker 是专门的任务执行器，不会 fork 自己！**

```python
class Worker:
    """专门的任务执行器，不会 fork"""

    def __init__(self, worker_id, plan_id, memory, execute_fn):
        self.worker_id = worker_id
        self.plan_id = plan_id
        self.memory = memory.copy()  # 初始化：复制原记忆
        self.execute_fn = execute_fn  # 执行函数，不会 fork

    def run(self):
        while True:
            # 1. 原子领取任务
            task = store.claim_task(plan_id, worker_id)

            if not task:
                # 没有可执行任务，死亡（不总结）
                break

            # 2. 执行任务（维护自己的记忆）
            try:
                # 将任务加入自己的记忆
                self.memory.append({"role": "user", "content": task.description})

                result = self.execute_fn(task, self.memory)

                # 将结果加入自己的记忆
                self.memory.append({"role": "assistant", "content": result})

                # 写回数据库
                store.update_task_status(plan_id, task.id, "completed", result)
            except Exception as e:
                store.update_task_status(plan_id, task.id, "failed", str(e))
                store.release_task(plan_id, task.id)

        # Worker 结束，不进行总结
```

### 4. 汇总结果（Main Agent 负责）

```python
# PlanExecutor.run() 返回后，Main Agent 调用 Analysis Agent
def run(self) -> dict:
    # 1. 创建并启动所有 Worker 线程
    futures = [self.executor.submit(worker.run) for worker in workers]

    # 2. 等待所有 Worker 完成（带进度显示）
    with Live(self.tracker.render()):
        for future in futures:
            future.result()

    # 3. 检查结果
    plan = self.store.load_plan(self.plan_id)
    if self.store.check_all_completed(self.plan_id):
        # 4. Main Agent 调用 Analysis Agent 汇总
        results = self.store.get_all_task_results(self.plan_id)
        summary = analysis_agent.analyze(results)
        self.store.save_summarized_result(self.plan_id, summary)

    return {
        "plan_id": self.plan_id,
        "status": plan.status,
        "summarized_result": plan.summarized_result,
    }
```

## 关键设计决策

### 1. Fork 实现：线程池

- **原因**：LLM 调用是 I/O 密集型，多线程足够
- **实现**：`ThreadPoolExecutor` 管理 Worker 线程
- **优势**：
  - 支持统一取消（用户中断时）
  - 超时控制
  - 资源回收

```python
class PlanExecutor:
    def __init__(self, ...):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures: list[Future] = []

    def run(self):
        # 提交所有 worker
        for i in range(self.num_workers):
            worker = Worker(...)
            future = self.executor.submit(worker.run)
            self.futures.append(future)

        # 等待所有 worker 完成（带进度显示）
        for future in self.futures:
            future.result()

    def cancel(self):
        """取消所有 Worker"""
        for future in self.futures:
            future.cancel()
        self.executor.shutdown(wait=False)
```

### 2. Worker 数量

- **默认**：2（纯 Worker 线程，Main Agent 不参与执行）
- **配置**：可通过参数调整
- **原则**：Main Agent 只负责协调和汇总，不执行具体任务

### 3. 记忆传递

```python
# Fork 时：初始化记忆 = 原记忆 + 后续独立记忆
worker = Worker(
    memory=main_agent.memory.copy(),  # 初始化时复制原记忆
    ...
)

# Worker 执行过程中维护自己的记忆
def execute_task(task, memory):
    # 执行任务，追加到自己的 memory
    memory.append({"role": "user", "content": task.description})
    response = llm.invoke(memory)
    memory.append({"role": "assistant", "content": response})
    return response
```

- **初始化**：复制原 agent 的记忆作为上下文
- **后续**：Worker 维护独立记忆，不再共享
- **结果**：只通过 Task.result 写回数据库

### 4. 任务领取：原子操作

```python
def claim_task(plan_id, worker_id) -> Task | None:
    # 原子地检查并更新
    UPDATE tasks SET claimed_by = ?
    WHERE task_id = ? AND plan_id = ? AND claimed_by IS NULL
```

- 防止多个 Worker 抢同一个任务
- SQLite 的事务保证原子性

### 5. 总结权：Main Agent 负责

- **Worker**：只执行任务，完成后死亡，不总结
- **Main Agent**：等待所有 Worker 完成，自己调用 Analysis Agent 总结
- **避免竞争**：Worker 不关心总结，职责分明

### 6. 错误处理

- **失败不重试**：直接标记 `failed`，记录错误信息
- **超时处理**：单个任务超时 5 分钟，超时视为失败
- **Plan 继续**：部分任务失败不影响其他任务执行

### 7. 数据库：SQLite

- 轻量级，无需额外服务
- 事务支持，保证原子操作
- 文件存储，便于持久化

### 8. 用户中断处理

- **触发方式**：用户按 Ctrl+C（SIGINT）
- **处理流程**：
  1. 捕获 SIGINT 信号
  2. 设置全局中断标志
  3. Worker 检测到中断，释放已领取的任务
  4. 关闭线程池，退出执行
  5. 恢复原始信号处理器

- **状态保留**：已完成的任务结果保留在数据库，可后续恢复

```python
# 中断相关函数
from agent.worker import set_interrupt, clear_interrupt, is_interrupted

# 设置中断（通常由信号处理器自动调用）
set_interrupt()

# 检查是否被中断
if is_interrupted():
    # Worker 停止工作

# 清除中断标志（执行前自动调用）
clear_interrupt()
```

## 进度显示

```
Plan: 编写一个爬虫
├── ✓ T1: 分析网站结构
├── ✓ T2: 配置环境  
├── ⏳ T3: 编写爬虫 [main_agent]
├── ⏸ T4: 测试爬虫 (等待 T3)
└── ⏸ T5: 编写文档 (等待 T4)

[████████░░░░░░░░░░░░] 2/5 完成
```

**状态图标**：
- `✓` 已完成（绿色）
- `✗` 失败（红色）
- `⏳` 执行中（黄色）
- `○` 可执行（蓝色）
- `⏸` 等待依赖（灰色）

## 文件结构

```
agent/
├── models.py           # Task, Plan 数据模型
├── plan_store.py       # Plan 持久化存储
├── worker.py           # TaskWorker, PlanExecutor
├── main_agent.py       # 主 Agent（协调）
└── subagents/
    ├── plan_agent.py   # Plan 生成
    └── analysis_agent.py  # 结果汇总
```

## API 接口

### PlanStore

| 方法 | 说明 |
|------|------|
| `save_plan(plan, thread_id)` | 保存 Plan，返回 plan_id |
| `load_plan(plan_id)` | 加载 Plan |
| `claim_task(plan_id, worker_id)` | 原子领取任务 |
| `release_task(plan_id, task_id)` | 释放任务 |
| `update_task_status(plan_id, task_id, status, result)` | 更新任务状态 |
| `check_all_completed(plan_id)` | 检查是否全部完成 |
| `get_all_task_results(plan_id)` | 获取所有任务结果 |
| `save_summarized_result(plan_id, result)` | 保存汇总结果 |

### PlanExecutor

```python
executor = PlanExecutor(
    plan_id: str,
    memory: list,
    execute_fn: Callable[[Task, list], str],
    num_workers: int = 1,
)
result = executor.run()
```

### Worker

```python
worker = Worker(
    worker_id: str,
    plan_id: str,
    memory: list,              # 初始化记忆（复制原 agent 记忆）
    execute_fn: Callable[[Task, list], str],
)

# Worker 运行时维护独立的 memory
# - 初始化：memory = 原记忆副本
# - 执行中：追加任务和结果到自己的 memory
# - 不共享：不修改原 agent 的 memory
result = worker.run()
```

### PlanExecutor（线程池管理）

```python
executor = PlanExecutor(
    plan_id: str,
    memory: list,
    execute_fn: Callable[[Task, list], str],
    num_workers: int = 1,
)

# 执行
result = executor.run()

# 取消（用户中断时）
executor.cancel()  # 取消所有 Worker
```

## 待实现

- [x] Task/Plan 数据模型 ✅ 已测试
- [x] PlanStore 持久化存储 ✅ 已测试
- [x] TaskWorker 任务执行器 ✅ 已测试
- [x] PlanExecutor 协调器（线程池管理、取消支持）✅ 已测试
- [x] 进度显示 ✅ 已测试
- [x] 集成到 MainAgent ✅ 已测试
- [x] execute_fn 实现 ✅ 已测试
- [x] 集成汇总功能 ✅ 已测试
- [x] 用户中断处理 ✅ 已测试

---

## 实现步骤

### 步骤 1：更新 TaskWorker 类 ✅ 完成

**目标**：
- TaskWorker 是专门的任务执行器（不是 MainAgent 副本）
- 独立记忆维护（初始化时复制，后续独立）
- 任务领取 → 执行 → 写回结果
- 完成后死亡，不总结

**验证方式**：
```python
# Mock 执行函数
def mock_execute(task, memory):
    return f"执行结果: {task.description}"

# 创建 Plan
plan = Plan(goal="测试", tasks=[Task(id="T1", description="测试任务")])
plan_id = store.save_plan(plan)

# 创建 Worker
worker = TaskWorker(
    worker_id="test_worker",
    plan_id=plan_id,
    memory=[{"role": "user", "content": "初始记忆"}],
    execute_fn=mock_execute,
)
result = worker.run()

# 验证
assert store.load_plan(plan_id).tasks[0].status == "completed"
assert store.load_plan(plan_id).tasks[0].result == "执行结果: 测试任务"
```

---

### 步骤 2：更新 PlanExecutor ✅ 完成

**目标**：
- 线程池管理多个 TaskWorker
- 进度实时显示
- 支持取消

**验证方式**：
```python
# 创建多任务 Plan
plan = Plan(
    goal="测试",
    tasks=[
        Task(id="T1", description="任务1", dependencies=[]),
        Task(id="T2", description="任务2", dependencies=["T1"]),
        Task(id="T3", description="任务3", dependencies=["T1"]),
    ]
)
plan_id = store.save_plan(plan)

# 执行
executor = PlanExecutor(
    plan_id=plan_id,
    memory=[],
    execute_fn=mock_execute,
    num_workers=2,
)
result = executor.run()

# 验证
assert result["completed"] == 3
assert store.check_all_completed(plan_id)
```

---

### 步骤 3：实现 execute_fn ✅ 完成

**目标**：
- 真实调用 LLM 执行任务
- 将任务描述和记忆传给 LLM
- 返回执行结果

**验证方式**：
```python
def real_execute(task, memory):
    llm = get_llm_model()
    memory.append({"role": "user", "content": task.description})
    response = llm.invoke(memory)
    memory.append({"role": "assistant", "content": response.content})
    return response.content

executor = PlanExecutor(
    plan_id=plan_id,
    memory=main_agent.messages,
    execute_fn=real_execute,
    num_workers=2,
)
result = executor.run()

# 验证：每个 task 都有真实的 LLM 输出
for task in store.load_plan(plan_id).tasks:
    assert task.result is not None
    assert len(task.result) > 0
```

---

### 步骤 4：集成 Analysis Agent ✅ 完成

**目标**：
- PlanExecutor 完成后调用 Analysis Agent 汇总
- 保存 summarized_result

**验证方式**：
```python
result = executor.run()

# 验证
plan = store.load_plan(plan_id)
assert plan.summarized_result is not None
assert plan.status == "completed"
```

---

### 步骤 5：集成到 MainAgent ✅ 完成

**目标**：
- MainAgent 识别复杂任务
- 调用 PlanAgent 生成 Plan
- 创建 PlanExecutor 执行
- 返回汇总结果给用户

**验证方式**：
```python
main_agent = MainAgent()
result = main_agent.chat("帮我写一个爬虫，抓取某网站的文章")

# 验证
assert "summarized_result" in result
assert len(result["summarized_result"]) > 0
```
