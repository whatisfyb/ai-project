# Graph Builder Async Redesign

## 目标

将 `graph_builder` 从线程并行改为 `asyncio` 协程，实现真正的中断能力和可取消的任务执行。

## 动机

当前设计使用 `ThreadPoolExecutor` + `as_completed` 并行运行任务，但 Python 线程无法被外部终止，导致中断逻辑只能等待本轮任务全部跑完。asyncio 的 `Task.cancel()` 可以在 await 点投掷 `CancelledError`，让正在执行的 LLM 调用被真正打断。

## 改造范围

### 需要修改的文件

| 文件 | 变化 |
|------|------|
| `graph/graph_builder.py` | 核心改造：sync → async |
| `agents/executor/executor.py` | 新增 `async_execute_task` |
| `agents/synthesizer/synthesizer.py` | 新增 `async_synthesize` |
| `agents/planner/plan_structure.py` | `TaskStatus` 新增 `CANCELLED` 值 |

## 架构设计

### graph_builder.py 结构变化

并行执行从 `ThreadPoolExecutor` 改为 `asyncio.as_completed`，使用模块级 `_running_tasks_registry` 管理正在运行的 `asyncio.Task` 实例，供外部中断函数 cancel。

### executor.py 新增异步函数

新增 `async_execute_task`，与 `execute_task` 并存：

- LLM 调用：`await llm_with_tools.ainvoke(messages)`
- 工具调用：`asyncio.to_thread(tool_fn.invoke, tool_args)` 包装同步调用
- 多个工具调用用 `asyncio.gather(*tool_coroutines, return_exceptions=True)` 并发
- 捕获 `CancelledError` → 标记 `CANCELLED` → re-raise

### synthesizer.py 新增异步函数

新增 `async_synthesize`，LLM 调用改用 `await llm.ainvoke(prompt)`。

### plan_structure.py 状态扩展

`TaskStatus` 新增 `CANCELLED = "cancelled"`。`check_complete()` 将其视为终结状态。当依赖是 `CANCELLED` 时，下游任务的依赖视为未完成，不会就绪。

### 中断逻辑

外部调用 `request_interrupt(graph, config)`：
1. `graph.update_state` 设置 `interrupt_requested = True`
2. `_cancel_running_tasks(config)` 对每个正在运行的 `asyncio.Task` 调用 `.cancel()`

正在运行的协程在最近的 await 点抛出 `CancelledError`，被 `async_execute_task` 捕获后标记为 `CANCELLED` 并重新抛出。`_parallel_executor` 检测到取消标志后立即退出本轮循环。下一轮进入 `_check_interrupt` 时清除 flag 并调用 `interrupt()` 挂起。

恢复时 `resume_graph` 自动将 CANCELLED 任务重置为 PENDING，清除对应 results 条目，然后 `Command(resume)` 继续调度。

## 错误处理

- `CancelledError`：标记 `CANCELLED`，不重试
- 其他异常：标记 `FAILED`，记录 error，继续执行其他任务
- `as_completed` 循环中异常被 `except Exception: continue` 吞掉，不影响其他任务

## 中断后恢复

- `interrupt_requested` 在 `_check_interrupt` 中自动设为 False
- `resume_graph` 将 CANCELLED → PENDING + 清除 result/error + 清除 results 字典条目
- COMPLETED 的任务自然跳过
- FAILED 的任务保持 FAILED，不会被重新执行

## 向后兼容性

- 保留原有 `execute_task`（同步），`synthesize`（同步）
- 仅新增 `async_execute_task`，`async_synthesize`
- `graph_builder` 的公共 API 不变：`build_graph()` 对外仍提供 `run_initial()`, `resume_graph()`, `request_interrupt()`
- `run_initial`/`resume_graph` 内部改用 `graph.astream()` + `asyncio.run()` 包装，因为节点是 `async def`
