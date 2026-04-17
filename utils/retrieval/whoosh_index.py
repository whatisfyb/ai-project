"""Whoosh 全文索引模块 - 支持增量更新的 BM25 检索"""

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import jieba
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from whoosh.index import create_in, exists_in, open_dir
from whoosh.fields import Schema, TEXT, ID, KEYWORD, NUMERIC, STORED
from whoosh.qparser import QueryParser
from whoosh.analysis import Tokenizer, Token


# 默认索引路径
DEFAULT_INDEX_PATH = Path(".data/whoosh_index")


# jieba 分词器
class JiebaTokenizer(Tokenizer):
    """jieba 分词器"""

    def __call__(
        self,
        value,
        positions=False,
        chars=False,
        keeporiginal=False,
        removestops=True,
        start_pos=0,
        start_char=0,
        mode="",
        **kwargs,
    ):
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


SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),
    content=TEXT(stored=True, analyzer=JiebaTokenizer()),
    paper_id=ID(stored=True),
    section=ID(stored=True),
    authors=TEXT(stored=False, analyzer=JiebaTokenizer()),
    keywords=KEYWORD(stored=True, commas=True),
    year=NUMERIC(stored=True, numtype=int),
)


class WhooshIndex:
    """Whoosh 全文索引"""

    def __init__(self, index_path: Path | str = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = None

    def _get_index(self):
        """获取索引实例，自动处理 schema 迁移"""
        if self._index is not None:
            return self._index

        index_path_str = str(self.index_path.absolute())

        if exists_in(index_path_str):
            self._index = open_dir(index_path_str)
            if not self._check_schema(self._index):
                import shutil

                if self.index_path.exists():
                    for item in self.index_path.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                self._index = create_in(index_path_str, SCHEMA)
        else:
            self._index = create_in(index_path_str, SCHEMA)
        return self._index

    @staticmethod
    def _check_schema(index) -> bool:
        """检查索引 schema 是否包含必需字段"""
        schema = index.schema
        required_fields = {"paper_id", "section", "year"}
        return required_fields.issubset(schema.names())

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
            meta = doc.metadata or {}

            authors_str = meta.get("authors", "")
            if isinstance(authors_str, list):
                authors_str = ",".join(authors_str)

            keywords_str = meta.get("keywords", "")
            if isinstance(keywords_str, list):
                keywords_str = ",".join(keywords_str)

            year_val = meta.get("year")
            year_int = int(year_val) if year_val and str(year_val).isdigit() else None

            writer.update_document(
                doc_id=doc_id,
                content=content,
                paper_id=meta.get("paper_id", ""),
                section=meta.get("section", ""),
                authors=authors_str,
                keywords=keywords_str,
                year=year_int,
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

    def delete_by_prefix(self, field: str, prefix: str):
        """按字段前缀删除文档

        Args:
            field: 字段名
            prefix: 前缀值
        """
        from whoosh.query import Prefix

        index = self._get_index()
        writer = index.writer()
        writer.delete_by_query(Prefix(field, prefix))
        writer.commit()

    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """全文检索

        Args:
            query: 查询文本
            k: 返回数量
            filter: 元数据过滤条件，支持 paper_id, section, authors, keywords, year, year_min, year_max

        Returns:
            (Document, score) 元组列表
        """
        index = self._get_index()
        searcher = index.searcher()

        parser = QueryParser("content", index.schema)
        text_query = parser.parse(query)

        if filter:
            from whoosh.query import Term, NumericRange, And as WAnd, Or as WOr

            filter_nodes = []
            for key, val in filter.items():
                if key == "paper_id":
                    filter_nodes.append(Term("paper_id", str(val)))
                elif key == "section":
                    filter_nodes.append(Term("section", str(val)))
                elif key == "authors":
                    filter_nodes.append(Term("authors", str(val)))
                elif key == "keywords":
                    for kw in str(val).split(","):
                        kw = kw.strip()
                        if kw:
                            filter_nodes.append(Term("keywords", kw))
                elif key == "year":
                    filter_nodes.append(Term("year", int(val)))
                elif key == "year_min":
                    filter_nodes.append(NumericRange("year", int(val), None))
                elif key == "year_max":
                    filter_nodes.append(NumericRange("year", None, int(val)))

            if filter_nodes:
                from whoosh.query import And as WAnd

                combined_filter = WAnd(filter_nodes)
                from whoosh.search import Searcher

                final_query = (
                    WAnd([text_query, combined_filter]) if filter_nodes else text_query
                )
            else:
                final_query = text_query
        else:
            final_query = text_query

        results = searcher.search(final_query, limit=k)

        documents = []
        for hit in results:
            meta = {"id": hit["doc_id"]}
            if "paper_id" in hit:
                meta["paper_id"] = hit["paper_id"]
            if "section" in hit:
                meta["section"] = hit["section"]
            if "keywords" in hit:
                meta["keywords"] = hit["keywords"]
            if "year" in hit:
                meta["year"] = hit["year"]

            doc = Document(
                page_content=hit["content"],
                metadata=meta,
            )
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
