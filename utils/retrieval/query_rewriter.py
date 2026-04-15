"""查询改写模块 - 扩展查询词提升召回率"""

import re
from typing import Optional
from langchain_core.language_models import BaseChatModel


def get_default_llm() -> BaseChatModel:
    """获取默认 LLM"""
    from utils.llm import get_llm_model
    return get_llm_model()


def _clean_llm_response(content: str) -> str:
    """清理 LLM 返回内容，移除思考过程"""
    lines = content.strip().split('\n')
    result_lines = []

    # 从后往前找中文内容
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # 检查是否像查询词（包含中文，较短）
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in stripped)
        is_short_result = len(stripped) < 50 and len(stripped) >= 2

        if has_chinese and is_short_result:
            result_lines.insert(0, stripped)
        elif result_lines:
            # 已经找到结果了，遇到非结果行就停止
            break

    return '\n'.join(result_lines) if result_lines else content


def expand_query(
    query: str,
    llm: Optional[BaseChatModel] = None,
    n_expansions: int = 3,
) -> list[str]:
    """扩展查询词

    Args:
        query: 原始查询
        llm: 语言模型（可选）
        n_expansions: 扩展数量

    Returns:
        扩展后的查询列表（包含原始查询）
    """
    if llm is None:
        llm = get_default_llm()

    prompt = f"""将以下搜索查询扩展为 {n_expansions} 个不同的表达方式。
每行一个，只输出查询词，不要其他内容。

查询：{query}"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)

        # 清理思考过程
        content = _clean_llm_response(content)

        # 解析结果：取非空行
        expansions = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if len(line) >= 2:
                expansions.append(line)

        # 去重，保留原始查询
        result = [query]
        for q in expansions[:n_expansions]:
            if q not in result:
                result.append(q)

        return result
    except Exception as e:
        # 扩展失败，返回原始查询
        return [query]


def rewrite_query(
    query: str,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """改写查询（生成更清晰的搜索表达）

    Args:
        query: 原始查询
        llm: 语言模型

    Returns:
        改写后的查询
    """
    if llm is None:
        llm = get_default_llm()

    prompt = f"""请将以下搜索查询改写为更适合文档检索的表达。

要求：
1. 保留核心意图
2. 补充可能缺失的关键词
3. 移除无关词汇
4. 只输出改写后的查询，不要解释

原始查询：{query}

改写后："""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        content = _clean_llm_response(content)
        return content.strip()
    except Exception:
        return query


class QueryRewriter:
    """查询改写器"""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm or get_default_llm()

    def expand(self, query: str, n: int = 3) -> list[str]:
        """扩展查询"""
        return expand_query(query, self.llm, n)

    def rewrite(self, query: str) -> str:
        """改写查询"""
        return rewrite_query(query, self.llm)

    def expand_and_rewrite(self, query: str, n: int = 3) -> list[str]:
        """改写 + 扩展"""
        # 先改写
        rewritten = self.rewrite(query)
        # 再扩展
        expansions = self.expand(rewritten, n)
        return expansions
