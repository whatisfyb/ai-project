"""检索器"""

from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from utils.vector_store import VectorStore


class Retriever(BaseRetriever):
    """基于 VectorStore 的检索器"""

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 5,
        filter: Optional[dict] = None,
    ):
        """初始化检索器

        Args:
            vector_store: VectorStore 实例
            k: 返回的相关文档数量
            filter: 元数据过滤条件
        """
        self.vector_store = vector_store
        self.k = k
        self.filter = filter

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """获取相关文档

        Args:
            query: 查询文本
            run_manager: 回调管理器

        Returns:
            相关的 Document 列表
        """
        return self.vector_store.similarity_search(
            query=query,
            k=self.k,
            filter=self.filter,
        )

    def get_relevant_documents_with_score(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:
        """获取相关文档（带相似度分数）

        Args:
            query: 查询文本

        Returns:
            (Document, score) 元组列表
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=self.k,
            filter=self.filter,
        )


def create_retriever(
    collection_name: str = "default",
    persist_dir: Optional[str] = None,
    k: int = 5,
    filter: Optional[dict] = None,
) -> Retriever:
    """创建检索器

    Args:
        collection_name: collection 名称
        persist_dir: 持久化目录
        k: 返回数量
        filter: 元数据过滤条件

    Returns:
        Retriever 实例
    """
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )
    return Retriever(
        vector_store=vector_store,
        k=k,
        filter=filter,
    )


class MultiRetriever:
    """多路检索器 - 支持多种检索方式"""

    def __init__(self, retrievers: list[Retriever]):
        """初始化多路检索器

        Args:
            retrievers: Retriever 列表
        """
        self.retrievers = retrievers

    def retrieve(self, query: str) -> list[Document]:
        """多路检索并合并结果

        Args:
            query: 查询文本

        Returns:
            合并去重后的 Document 列表
        """
        all_docs = []
        seen_ids = set()

        for retriever in self.retrievers:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)

        return all_docs

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """多路检索并返回分数

        Args:
            query: 查询文本

        Returns:
            (Document, min_score) 元组列表
        """
        doc_scores: dict[str, tuple[Document, float]] = {}

        for retriever in self.retrievers:
            docs_with_scores = retriever.get_relevant_documents_with_score(query)
            for doc, score in docs_with_scores:
                doc_id = doc.metadata.get("id", doc.page_content[:50])
                if doc_id not in doc_scores or score < doc_scores[doc_id][1]:
                    doc_scores[doc_id] = (doc, score)

        return sorted(doc_scores.values(), key=lambda x: x[1])