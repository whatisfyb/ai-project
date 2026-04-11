"""长期记忆相关节点"""
import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from agent.core.models import MainAgentState
from agent.subagents.memory_agent import get_memory_agent


# ============ 常量定义 ============

# 触发记忆检查的关键词
MEMORY_KEYWORDS = {
    # 记忆保存
    "remember": ["记得", "记住", "别忘了", "别忘了", "记住这点", "记住这个"],
    # 用户信息
    "user_info": ["我是", "我在", "我的", "我从事", "我负责", "我擅长"],
    # 反馈纠正
    "feedback": ["不要", "别", "停止", "不要这样", "换个方式", "下次注意", "以后"],
    # 项目信息
    "project": ["截止", "deadline", "计划", "安排", "进度", "发布", "上线"],
    # 外部引用
    "reference": ["看板", "dashboard", "文档在", "链接是", "地址是", "追踪在", "记录在"],
    # 记忆删除
    "forget": ["忘了", "忘记", "删除记忆", "不需要记住", "不用记"],
}


# ============ 关键词检测 ============

def detect_memory_keywords(text: str) -> tuple[bool, str | None, list[str]]:
    """检测文本中是否包含记忆相关关键词

    Args:
        text: 用户输入文本

    Returns:
        (是否包含关键词, 关键词类别, 发现的关键词列表)
    """
    found_keywords = []
    found_category = None

    for category, keywords in MEMORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                found_keywords.append(keyword)
                if found_category is None:
                    found_category = category

    has_keywords = len(found_keywords) > 0
    return has_keywords, found_category, found_keywords


# ============ 检查节点 ============

async def check_memory_generation_node_async(state: MainAgentState) -> dict[str, Any]:
    """记忆生成检查节点（异步版本）

    作为 Main Agent graph 的一个节点，检查用户输入是否包含记忆关键词。
    如果包含，异步启动记忆分析 Agent。

    Args:
        state: MainAgentState 状态

    Returns:
        更新后的状态
    """
    # 从状态中获取用户输入
    messages = state.get("messages", [])
    if not messages:
        return {
            "memory_checked": True,
            "should_generate_memory": False,
            "memory_result": None,
        }

    # 获取最后一条用户消息
    user_input = _extract_user_input(messages)

    if not user_input:
        return {
            "memory_checked": True,
            "should_generate_memory": False,
            "memory_result": None,
        }

    # 检测关键词
    has_keywords, category, keywords = detect_memory_keywords(user_input)

    if not has_keywords:
        return {
            "memory_checked": True,
            "should_generate_memory": False,
            "keyword_category": None,
            "keywords_found": [],
            "memory_result": None,
        }

    # 异步运行记忆分析
    try:
        agent = get_memory_agent()
        result = await agent.run_async(user_input)

        return {
            "memory_checked": True,
            "should_generate_memory": True,
            "keyword_category": category,
            "keywords_found": keywords,
            "memory_result": result,
        }
    except Exception as e:
        return {
            "memory_checked": True,
            "should_generate_memory": True,
            "keyword_category": category,
            "keywords_found": keywords,
            "memory_result": {"success": False, "error": str(e)},
        }


# ============ 辅助函数 ============

def _extract_user_input(messages: list) -> str:
    """从消息列表中提取最后一条用户输入

    Args:
        messages: 消息列表

    Returns:
        用户输入文本
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ============ 节点定义 ============

async def load_memory_node(state: MainAgentState) -> dict:
    """加载记忆节点 - 从长期记忆中加载相关记忆到上下文"""
    from store.long_term_memory_persistency import get_memory_store, memory_age_days

    store = get_memory_store()
    memories = store.list()

    if not memories:
        return {"memory_context": None}

    # 构建记忆上下文
    memory_lines = []
    for m in memories[:20]:  # 最多加载 20 条记忆
        # 获取完整记忆内容
        full_memory = store.read(m.filename.replace(".md", ""))
        if full_memory:
            age = memory_age_days(m.mtime_ms)
            age_text = "今天" if age == 0 else f"{age}天前"
            memory_lines.append(
                f"- [{m.type}] {m.description} ({age_text}):\n  {full_memory.content}"
            )

    if memory_lines:
        memory_context = f"""# 用户记忆

以下是关于用户的长期记忆，请在对话中参考这些信息：

{chr(10).join(memory_lines)}

注意：记忆是时间点快照，可能已过时。涉及代码/文件引用时请验证当前状态。
"""
    else:
        memory_context = None

    return {"memory_context": memory_context}


async def memory_check_node(state: MainAgentState) -> dict:
    """记忆检查节点 - 检查用户输入是否需要保存记忆（不阻塞主线程）"""
    # fire-and-forget，不等待结果
    asyncio.create_task(check_memory_generation_node_async(state))
    return {}


# ============ 便捷函数 ============

def run_memory_check(user_input: str) -> dict[str, Any]:
    """运行记忆检查（便捷函数）

    Args:
        user_input: 用户输入

    Returns:
        检查结果
    """
    has_keywords, category, keywords = detect_memory_keywords(user_input)

    if not has_keywords:
        return {
            "should_check": False,
            "category": None,
            "keywords": [],
            "result": None,
        }

    # 运行记忆分析
    agent = get_memory_agent()
    result = asyncio.run(agent.run_async(user_input))

    return {
        "should_check": True,
        "category": category,
        "keywords": keywords,
        "result": result,
    }


async def run_memory_check_async(user_input: str) -> dict[str, Any]:
    """运行记忆检查（异步版本）

    Args:
        user_input: 用户输入

    Returns:
        检查结果
    """
    has_keywords, category, keywords = detect_memory_keywords(user_input)

    if not has_keywords:
        return {
            "should_check": False,
            "category": None,
            "keywords": [],
            "result": None,
        }

    # 运行记忆分析
    agent = get_memory_agent()
    result = await agent.run_async(user_input)

    return {
        "should_check": True,
        "category": category,
        "keywords": keywords,
        "result": result,
    }
