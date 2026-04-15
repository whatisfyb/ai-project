"""Whoosh 全文索引模块 - 支持增量更新的 BM25 检索"""

import jieba
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from whoosh.index import create_in, exists_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser
from whoosh.analysis import Tokenizer, Token


# 默认索引路径
DEFAULT_INDEX_PATH = Path(".data/whoosh_index")


# jieba 分词器
class JiebaTokenizer(Tokenizer):
    """jieba 分词器"""

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, mode='', **kwargs):
        for word in jieba.cut(value):
            word = word.strip()
            if word:
                token = Token()
                token.text = word
                token.pos = start_pos
                token.startchar = start_char
                token.endchar = start_char + len(word)
                start_char += len(word)
                start_pos += 1
                yield token


# 索引 schema
SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),  # 文档 ID（唯一）
    content=TEXT(stored=True, analyzer=JiebaTokenizer()),  # 中文分词
)


class WhooshIndex:
    """Whoosh 全文索引"""

    def __init__(self, index_path: Path | str = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = None

    def _get_index(self):
        """获取索引实例"""
        if self._index is not None:
            return self._index

        # Whoosh 需要字符串路径
        index_path_str = str(self.index_path.absolute())

        if exists_in(index_path_str):
            self._index = open_dir(index_path_str)
        else:
            self._index = create_in(index_path_str, SCHEMA)
        return self._index

    def add_documents(
        self,
        documents: list[Document],
        doc_ids: Optional[list[str]] = None,
    ):
        """增量添加文档

        Args:
            documents: 文档列表
            doc_ids: 文档 ID 列表
        """
        index = self._get_index()
        writer = index.writer()

        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else doc.metadata.get("id", f"doc_{i}")
            content = doc.page_content

            # 添加文档（自动覆盖相同 ID）
            writer.update_document(
                doc_id=doc_id,
                content=content,
            )

        writer.commit()

    def delete_documents(self, doc_ids: list[str]):
        """删除文档

        Args:
            doc_ids: 要删除的文档 ID 列表
        """
        index = self._get_index()
        writer = index.writer()

        for doc_id in doc_ids:
            writer.delete_by_term("doc_id", doc_id)

        writer.commit()

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """全文检索

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            (Document, score) 元组列表
        """
        index = self._get_index()
        searcher = index.searcher()

        # 解析查询
        parser = QueryParser("content", index.schema)
        q = parser.parse(query)

        # 搜索
        results = searcher.search(q, limit=k)

        # 转换结果
        documents = []
        for hit in results:
            doc = Document(
                page_content=hit["content"],
                metadata={"id": hit["doc_id"]},
            )
            # Whoosh 分数归一化到 0-1
            score = hit.score / 10.0 if hit.score else 0.0
            documents.append((doc, score))

        return documents

    def count(self) -> int:
        """返回索引文档数量"""
        index = self._get_index()
        return index.doc_count()

    def clear(self):
        """清空索引"""
        self._index = None
        # 删除索引目录内容，但保留目录本身
        import shutil
        if self.index_path.exists():
            for item in self.index_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            self.index_path.mkdir(parents=True, exist_ok=True)
        # 重建空索引
        self._index = create_in(str(self.index_path.absolute()), SCHEMA)


# 全局索引实例
_whoosh_index: Optional[WhooshIndex] = None


def get_whoosh_index() -> WhooshIndex:
    """获取 Whoosh 索引实例"""
    global _whoosh_index
    if _whoosh_index is None:
        _whoosh_index = WhooshIndex()
    return _whoosh_index


def reset_whoosh_index():
    """重置索引实例"""
    global _whoosh_index
    _whoosh_index = None
