"""混合检索模块 - 向量检索 + 全文检索 + RRF 融合"""

import asyncio
from typing import Optional
from langchain_core.documents import Document

from utils.retrieval.vector_store import VectorStore
from utils.retrieval.whoosh_index import WhooshIndex, get_whoosh_index


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
        bm25_results: 全文检索结果 [(Document, score), ...]
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

    # 处理全文检索结果
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
    sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

    return sorted_results[:top_k]


class HybridSearcher:
    """混合检索器（并发执行向量检索和全文检索）"""

    def __init__(
        self,
        vector_store: VectorStore,
        whoosh_index: Optional[WhooshIndex] = None,
    ):
        """初始化混合检索器

        Args:
            vector_store: 向量存储
            whoosh_index: Whoosh 索引（可选，默认使用全局实例）
        """
        self.vector_store = vector_store
        self.whoosh_index = whoosh_index or get_whoosh_index()

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
        use_bm25: bool = True,
    ) -> list[tuple[Document, float]]:
        """混合检索（协程并发执行）

        Args:
            query: 查询文本
            k: 返回数量
            filter: 元数据过滤条件
            use_bm25: 是否使用全文检索
        """
        if not use_bm25:
            return self.vector_store.similarity_search_with_score(
                query, k=k, filter=filter
            )

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
            None, self.vector_store.similarity_search_with_score, query, k, filter
        )
        whoosh_filter = self._to_whoosh_filter(filter) if filter else None
        whoosh_task = loop.run_in_executor(
            None, self.whoosh_index.search, query, k, whoosh_filter
        )
        vector_results, whoosh_results = await asyncio.gather(vector_task, whoosh_task)

        if not whoosh_results:
            return vector_results
        if not vector_results:
            return whoosh_results
        return rrf_fusion(vector_results, whoosh_results, top_k=k)

    @staticmethod
    def _to_whoosh_filter(filter_dict: dict) -> dict:
        """将 ChromaDB 风格 filter 转换为 Whoosh filter

        ChromaDB: {"section": "abstract"}, {"$and": [{"section": "abstract"}, {"year": {"$gte": 2020}}]}
        Whoosh: {"section": "abstract", "year_min": 2020}
        """
        whoosh_filter = {}

        if "$and" in filter_dict:
            for cond in filter_dict["$and"]:
                for key, val in cond.items():
                    if key == "year" and isinstance(val, dict):
                        if "$gte" in val:
                            whoosh_filter["year_min"] = val["$gte"]
                        if "$lte" in val:
                            whoosh_filter["year_max"] = val["$lte"]
                    elif key == "authors" and isinstance(val, dict):
                        whoosh_filter["authors"] = val.get("$contains", str(val))
                    elif key == "keywords" and isinstance(val, dict):
                        whoosh_filter["keywords"] = val.get("$contains", str(val))
                    else:
                        whoosh_filter[key] = val
        else:
            for key, val in filter_dict.items():
                if key == "year" and isinstance(val, dict):
                    if "$gte" in val:
                        whoosh_filter["year_min"] = val["$gte"]
                    if "$lte" in val:
                        whoosh_filter["year_max"] = val["$lte"]
                elif key == "authors" and isinstance(val, dict):
                    whoosh_filter["authors"] = val.get("$contains", str(val))
                elif key == "keywords" and isinstance(val, dict):
                    whoosh_filter["keywords"] = val.get("$contains", str(val))
                else:
                    whoosh_filter[key] = val

        return whoosh_filter

    def add_documents(
        self,
        documents: list[Document],
        doc_ids: Optional[list[str]] = None,
    ):
        """增量添加文档到全文索引

        Args:
            documents: 文档列表
            doc_ids: 文档 ID 列表
        """
        self.whoosh_index.add_documents(documents, doc_ids)

    def delete_documents(self, doc_ids: list[str]):
        """从全文索引删除文档

        Args:
            doc_ids: 文档 ID 列表
        """
        self.whoosh_index.delete_documents(doc_ids)

    def build_index(self):
        """从向量库构建全文索引（全量同步）"""
        all_docs = self.vector_store.similarity_search("", k=10000)

        if all_docs:
            doc_ids = []
            for i, doc in enumerate(all_docs):
                doc_id = doc.metadata.get("id", f"doc_{i}")
                if "id" not in doc.metadata:
                    doc.metadata["id"] = doc_id
                doc_ids.append(doc_id)

            self.whoosh_index.add_documents(all_docs, doc_ids)
