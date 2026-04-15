"""BM25 检索模块 - 关键词检索"""

import json
import pickle
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# 默认索引存储路径
DEFAULT_INDEX_PATH = Path(".data/bm25_index")


def _tokenize_chinese(text: str) -> list[str]:
    """中文分词（简单按字符分割 + 英文按空格）

    生产环境建议用 jieba 分词，这里保持轻量
    """
    tokens = []
    word = ""

    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            # 中文字符：单独作为一个 token
            if word:
                tokens.append(word.lower())
                word = ""
            tokens.append(char)
        elif char.isalnum():
            # 英文/数字：累积成词
            word += char
        else:
            # 其他字符：分隔符
            if word:
                tokens.append(word.lower())
                word = ""

    if word:
        tokens.append(word.lower())

    return tokens


class BM25Index:
    """BM25 索引管理"""

    def __init__(self, index_path: Path | str = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # 索引数据
        self._bm25: Optional[BM25Okapi] = None
        self._documents: list[Document] = []
        self._doc_ids: list[str] = []  # 文档 ID 列表，用于去重

    def build(self, documents: list[Document], doc_ids: Optional[list[str]] = None):
        """构建 BM25 索引

        Args:
            documents: 文档列表
            doc_ids: 文档 ID 列表（可选）
        """
        self._documents = documents
        self._doc_ids = doc_ids or [str(i) for i in range(len(documents))]

        # 分词
        corpus = [_tokenize_chinese(doc.page_content) for doc in documents]

        # 构建 BM25 索引
        self._bm25 = BM25Okapi(corpus)

    def add_documents(
        self,
        documents: list[Document],
        doc_ids: Optional[list[str]] = None,
    ):
        """增量添加文档（重建索引）

        Args:
            documents: 新增文档列表
            doc_ids: 新增文档 ID 列表
        """
        new_ids = doc_ids or [str(len(self._documents) + i) for i in range(len(documents))]

        # 检查重复
        for i, doc_id in enumerate(new_ids):
            if doc_id not in self._doc_ids:
                self._documents.append(documents[i])
                self._doc_ids.append(doc_id)

        # 重建索引
        if self._documents:
            corpus = [_tokenize_chinese(doc.page_content) for doc in self._documents]
            self._bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """BM25 检索

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            (Document, score) 元组列表
        """
        if not self._bm25 or not self._documents:
            return []

        # 查询分词
        query_tokens = _tokenize_chinese(query)

        # BM25 检索
        scores = self._bm25.get_scores(query_tokens)

        # 取 Top-K
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return [(self._documents[i], scores[i]) for i in top_indices if scores[i] > 0]

    def save(self):
        """保存索引到磁盘"""
        if not self._bm25:
            return

        # 保存 BM25 索引
        with open(self.index_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._bm25, f)

        # 保存文档（JSON 格式）
        docs_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in self._documents
        ]
        with open(self.index_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"documents": docs_data, "doc_ids": self._doc_ids}, f, ensure_ascii=False)

    def load(self) -> bool:
        """从磁盘加载索引

        Returns:
            是否加载成功
        """
        pkl_path = self.index_path.with_suffix(".pkl")
        json_path = self.index_path.with_suffix(".json")

        if not pkl_path.exists() or not json_path.exists():
            return False

        try:
            # 加载 BM25 索引
            with open(pkl_path, "rb") as f:
                self._bm25 = pickle.load(f)

            # 加载文档
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in data["documents"]
            ]
            self._doc_ids = data.get("doc_ids", [str(i) for i in range(len(self._documents))])

            return True
        except Exception:
            return False

    def count(self) -> int:
        """返回索引文档数量"""
        return len(self._documents)

    def clear(self):
        """清空索引"""
        self._bm25 = None
        self._documents = []
        self._doc_ids = []

        # 删除磁盘文件
        pkl_path = self.index_path.with_suffix(".pkl")
        json_path = self.index_path.with_suffix(".json")
        if pkl_path.exists():
            pkl_path.unlink()
        if json_path.exists():
            json_path.unlink()


# 全局 BM25 索引实例（懒加载）
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """获取 BM25 索引实例"""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


def reset_bm25_index():
    """重置 BM25 索引"""
    global _bm25_index
    _bm25_index = None
