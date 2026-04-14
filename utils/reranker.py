"""重排序模块 - Cross-Encoder 模型对检索结果精排"""

from typing import Optional
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document


# 默认模型
DEFAULT_MODEL = "BAAI/bge-reranker-base"

# 模型实例缓存
_reranker_model: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """获取重排序模型（懒加载）"""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(DEFAULT_MODEL)
    return _reranker_model


def reset_reranker():
    """重置模型实例"""
    global _reranker_model
    _reranker_model = None


def rerank(
    query: str,
    documents: list[Document],
    top_k: Optional[int] = None,
) -> list[Document]:
    """对文档列表进行重排序

    Args:
        query: 查询文本
        documents: 待重排序的文档列表
        top_k: 返回的文档数量，默认返回全部（按新排序）

    Returns:
        重排序后的文档列表
    """
    if not documents:
        return []

    if len(documents) == 1:
        return documents

    # 构建 query-doc pairs
    pairs = [(query, doc.page_content) for doc in documents]

    # 计算相关性分数
    model = get_reranker()
    scores = model.predict(pairs)

    # 按分数降序排列
    scored_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 取 top_k 或全部
    if top_k is not None:
        scored_docs = scored_docs[:top_k]

    return [doc for doc, score in scored_docs]


def rerank_with_scores(
    query: str,
    documents: list[Document],
    top_k: Optional[int] = None,
) -> list[tuple[Document, float]]:
    """对文档列表进行重排序，返回带分数的结果

    Args:
        query: 查询文本
        documents: 待重排序的文档列表
        top_k: 返回的文档数量

    Returns:
        (Document, score) 元组列表，按分数降序
    """
    if not documents:
        return []

    if len(documents) == 1:
        return [(documents[0], 1.0)]

    # 构建 query-doc pairs
    pairs = [(query, doc.page_content) for doc in documents]

    # 计算相关性分数
    model = get_reranker()
    scores = model.predict(pairs)

    # 按分数降序排列
    scored_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 取 top_k 或全部
    if top_k is not None:
        scored_docs = scored_docs[:top_k]

    return scored_docs
