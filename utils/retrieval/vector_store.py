"""向量存储 - ChromaDB"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from utils.core.embedding import embed_texts, get_embedding_dimension


class VectorStore:
    """ChromaDB 向量存储"""

    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: Optional[str] = None,
    ):
        """初始化向量存储

        Args:
            collection_name: collection 名称
            persist_dir: 持久化目录，不提供则用内存模式
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        if persist_dir:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client()

        self._collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(
        self,
        texts: list[str],
        ids: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
    ) -> list[str]:
        """添加文本到向量库

        Args:
            texts: 文本列表
            ids: ID 列表，不提供则自动生成
            metadatas: 元数据列表

        Returns:
            生成的 ID 列表
        """
        # 生成 embeddings
        embeddings = embed_texts(texts)

        # 生成 IDs
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in texts]

        # 存储
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas if metadatas else None,
        )

        return ids

    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """添加 LangChain Document 到向量库

        Args:
            documents: Document 列表
            ids: ID 列表

        Returns:
            生成的 ID 列表
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, ids, metadatas)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """相似度搜索

        Args:
            query: 查询文本
            k: 返回数量
            filter: 元数据过滤条件

        Returns:
            相关的 Document 列表
        """
        query_embedding = embed_texts([query])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                doc = Document(page_content=doc_text, metadata=metadata or {})
                documents.append(doc)

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """相似度搜索（带分数）

        Args:
            query: 查询文本
            k: 返回数量
            filter: 元数据过滤条件

        Returns:
            (Document, score) 元组列表，分数越低越相似
        """
        query_embedding = embed_texts([query])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
        )

        documents_with_score = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                doc = Document(page_content=doc_text, metadata=metadata or {})
                score = results["distances"][0][i] if results["distances"] else 0.0
                documents_with_score.append((doc, score))

        return documents_with_score

    def delete(self, ids: list[str]) -> None:
        """删除向量

        Args:
            ids: 要删除的 ID 列表
        """
        self._collection.delete(ids=ids)

    def delete_by_metadata(self, key: str, value: str) -> int:
        """按元数据条件删除文档

        Args:
            key: 元数据键
            value: 元数据值

        Returns:
            删除的文档数量
        """
        results = self._collection.get(
            where={key: value},
            include=["metadatas"],
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
        return len(results["ids"]) if results["ids"] else 0

    def delete_collection(self) -> None:
        """删除整个 collection"""
        self.client.delete_collection(self.collection_name)

    def count(self) -> int:
        """获取向量数量"""
        return self._collection.count()

    def exists(self, ids: list[str]) -> list[bool]:
        """检查 ID 是否存在

        Args:
            ids: ID 列表

        Returns:
            存在状态列表
        """
        try:
            self._collection.get(ids=ids)
            return [True] * len(ids)
        except Exception:
            return [False] * len(ids)
