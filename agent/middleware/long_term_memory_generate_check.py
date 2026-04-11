"""长期记忆生成检查节点

检查用户输入是否包含记忆相关关键词，如果包含则异步启动 Memory Agent 分析。
"""

from __future__ import annotations

import asyncio
from typing import Any, Annotated
import operator

from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict

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


# ============ 检查节点状态扩展 ============

class MemoryCheckState(TypedDict):
    """记忆检查状态（扩展 MainAgentState）"""
    messages: Annotated[list, operator.add]
    current_task: str | None
    memory_context: str | None
    subagent_results: dict[str, Any]
    # 记忆检查相关
    memory_checked: bool
    should_generate_memory: bool
    keyword_category: str | None
    keywords_found: list[str]
    memory_result: dict[str, Any] | None


# ============ 检查节点 ============

def check_memory_generation_node(state: MainAgentState) -> dict[str, Any]:
    """记忆生成检查节点（同步版本）

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

    # 尝试运行记忆分析
    try:
        try:
            loop = asyncio.get_running_loop()
            # 已经在异步上下文中，创建任务
            agent = get_memory_agent()
            future = asyncio.create_task(agent.run_async(user_input))

            return {
                "memory_checked": True,
                "should_generate_memory": True,
                "keyword_category": category,
                "keywords_found": keywords,
                "memory_result": {"status": "pending", "future": future},
            }
        except RuntimeError:
            # 没有运行的事件循环，同步执行
            agent = get_memory_agent()
            result = asyncio.run(agent.run_async(user_input))

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
