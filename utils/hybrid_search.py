"""混合检索模块 - 向量检索 + BM25 + RRF 融合"""

import asyncio
from typing import Optional
from langchain_core.documents import Document

from utils.vector_store import VectorStore
from utils.bm25 import BM25Index, get_bm25_index


# RRF 融合参数
RRF_K = 60


def rrf_fusion(
    vector_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    top_k: int = 10,
) -> list[tuple[Document, float]]:
    """RRF 融合算法

    Args:
        vector_results: 向量检索结果 [(Document, score), ...]
        bm25_results: BM25 检索结果 [(Document, score), ...]
        top_k: 返回数量

    Returns:
        融合后的结果 [(Document, rrf_score), ...]
    """
    # 文档 ID → 融合分数
    doc_scores: dict[str, tuple[Document, float]] = {}

    # 处理向量检索结果
    for rank, (doc, _) in enumerate(vector_results):
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        rrf_score = 1 / (RRF_K + rank + 1)

        if doc_id not in doc_scores:
            doc_scores[doc_id] = (doc, rrf_score)
        else:
            # 同一文档，累加分数
            old_doc, old_score = doc_scores[doc_id]
            doc_scores[doc_id] = (old_doc, old_score + rrf_score)

    # 处理 BM25 检索结果
    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        rrf_score = 1 / (RRF_K + rank + 1)

        if doc_id not in doc_scores:
            doc_scores[doc_id] = (doc, rrf_score)
        else:
            # 同一文档，累加分数
            old_doc, old_score = doc_scores[doc_id]
            doc_scores[doc_id] = (old_doc, old_score + rrf_score)

    # 按融合分数排序
    sorted_results = sorted(
        doc_scores.values(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results[:top_k]


class HybridSearcher:
    """混合检索器（并发执行向量检索和 BM25 检索）"""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: Optional[BM25Index] = None,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """初始化混合检索器

        Args:
            vector_store: 向量存储
            bm25_index: BM25 索引（可选，默认使用全局实例）
            vector_weight: 向量检索权重（暂未使用，RRF 不需要）
            bm25_weight: BM25 检索权重（暂未使用，RRF 不需要）
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index or get_bm25_index()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
        use_bm25: bool = True,
    ) -> list[tuple[Document, float]]:
        """混合检索（协程并发执行）"""
        # 如果不用 BM25，直接返回向量结果
        if not use_bm25:
            return self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter
            )

        # 协程并发执行
        return asyncio.run(self._search_async(query, k, filter))

    async def _search_async(
        self,
        query: str,
        k: int,
        filter: Optional[dict],
    ) -> list[tuple[Document, float]]:
        """异步并发检索"""
        loop = asyncio.get_event_loop()
        vector_task = loop.run_in_executor(
            None,
            self.vector_store.similarity_search_with_score,
            query, k, filter
        )
        bm25_task = loop.run_in_executor(
            None,
            self.bm25_index.search,
            query, k
        )
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

        if not bm25_results:
            return vector_results
        if not vector_results:
            return bm25_results
        return rrf_fusion(vector_results, bm25_results, top_k=k)

    def build_bm25_index(self):
        """从向量库构建 BM25 索引"""
        # 获取向量库中的所有文档
        # ChromaDB 不支持直接获取所有文档，用空查询获取大量结果
        all_docs = self.vector_store.similarity_search("", k=10000)

        if all_docs:
            # 为文档生成 ID
            doc_ids = []
            for i, doc in enumerate(all_docs):
                doc_id = doc.metadata.get("id", f"doc_{i}")
                if "id" not in doc.metadata:
                    doc.metadata["id"] = doc_id
                doc_ids.append(doc_id)

            # 构建 BM25 索引
            self.bm25_index.build(all_docs, doc_ids)
            self.bm25_index.save()
