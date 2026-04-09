"""Search Skill - 搜索网页获取信息"""

import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.web import web_search


def run(query: str, max_results: int = 5, **kwargs) -> dict[str, Any]:
    """执行搜索

    Args:
        query: 搜索关键词
        max_results: 最大返回结果数，默认 5
        **kwargs: 额外参数

    Returns:
        搜索结果字典
    """
    if not query or not query.strip():
        return {
            "success": False,
            "error": "搜索关键词不能为空",
            "results": [],
        }

    try:
        # 调用 Tavily 搜索
        result = tavily_search.invoke({
            "query": query,
            "max_results": max_results,
        })

        return {
            "success": True,
            "query": query,
            "results": result.get("results", []),
            "answer": result.get("answer", ""),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": [],
        }
