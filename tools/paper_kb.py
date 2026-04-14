"""论文知识库工具 - 统一入口

合并了检索、列表、统计、入库等功能。
"""

from typing import Any, Optional, Literal
from langchain_core.tools import tool

from utils.vector_store import VectorStore
from utils.reranker import rerank_with_scores
from utils.hybrid_search import HybridSearcher
from utils.whoosh_index import get_whoosh_index


# 默认配置
DEFAULT_COLLECTION = "papers"
DEFAULT_PERSIST_DIR = ".data/chroma"

# 重排序阈值
RERANK_MIN_CANDIDATES = 3


def _get_paper_store() -> VectorStore:
    return VectorStore(collection_name=DEFAULT_COLLECTION, persist_dir=DEFAULT_PERSIST_DIR)


def _get_hybrid_searcher() -> HybridSearcher:
    return HybridSearcher(_get_paper_store(), get_whoosh_index())


def _merge_results(results: list[tuple], top_k: int) -> list[tuple]:
    """去重融合"""
    doc_scores: dict[str, tuple] = {}
    for doc, score in results:
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        if doc_id not in doc_scores or score > doc_scores[doc_id][1]:
            doc_scores[doc_id] = (doc, score)
    return sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)[:top_k]


def _format_doc(doc, score) -> dict:
    """格式化文档结果"""
    meta = doc.metadata or {}
    authors_val = meta.get("authors", "")
    keywords_val = meta.get("keywords", "")

    authors = authors_val.split(",") if isinstance(authors_val, str) and authors_val else []
    keywords = keywords_val.split(",") if isinstance(keywords_val, str) and keywords_val else []

    return {
        "paper_id": meta.get("paper_id", "unknown"),
        "title": meta.get("title", ""),
        "authors": authors,
        "keywords": keywords,
        "year": meta.get("year"),
        "section": meta.get("section", ""),
        "content": doc.page_content,
        "relevance_score": round(float(score), 4) if score else None,
    }


@tool
def paper_kb(
    action: Literal["search", "list", "stats", "ingest", "ingest_status", "ingest_cancel"],
    query: str = "",
    top_k: int = 5,
    section: str = "",
    author: str = "",
    keyword: str = "",
    year_min: int = 0,
    year_max: int = 0,
    limit: int = 20,
    pdf_paths: list[str] = [],
    task_id: str = "",
    rerank: bool = True,
    hybrid: bool = True,
) -> dict[str, Any]:
    """Paper knowledge base operations.

    Unified tool for searching, listing, and managing academic papers in the knowledge base.

    Args:
        action: Operation to perform:
            - "search": Search papers (uses: query, top_k, section, author, keyword, year_min, year_max, rerank, hybrid)
            - "list": List papers (uses: limit, author, keyword, year_min, year_max)
            - "stats": Get statistics (no args needed)
            - "ingest": Ingest PDFs (uses: pdf_paths)
            - "ingest_status": Check status (uses: task_id)
            - "ingest_cancel": Cancel task (uses: task_id)
        query: Search query
        top_k: Max results (default: 5)
        section: Filter by section
        author: Filter by author
        keyword: Filter by keyword
        year_min: Min year filter
        year_max: Max year filter
        limit: Max papers for list (default: 20)
        pdf_paths: PDF file paths for ingest
        task_id: Task ID for status/cancel
        rerank: Enable reranking (default: True)
        hybrid: Enable hybrid search (default: True)

    Returns:
        Dictionary containing operation results.
    """
    try:
        if action == "search":
            return _action_search(
                query=query,
                top_k=top_k,
                section=section if section else None,
                author=author if author else None,
                keyword=keyword if keyword else None,
                year_min=year_min if year_min > 0 else None,
                year_max=year_max if year_max > 0 else None,
                rerank=rerank,
                hybrid=hybrid,
            )
        elif action == "list":
            return _action_list(
                limit=limit,
                author=author if author else None,
                keyword=keyword if keyword else None,
                year_min=year_min if year_min > 0 else None,
                year_max=year_max if year_max > 0 else None,
            )
        elif action == "stats":
            return _action_stats()
        elif action == "ingest":
            return _action_ingest(pdf_paths=pdf_paths)
        elif action == "ingest_status":
            return _action_ingest_status(task_id=task_id)
        elif action == "ingest_cancel":
            return _action_ingest_cancel(task_id=task_id)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _action_search(
    query: str = "",
    top_k: int = 5,
    section: Optional[str] = None,
    author: Optional[str] = None,
    keyword: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    rerank: bool = True,
    hybrid: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """搜索论文"""
    if not query:
        return {"status": "error", "error": "query is required for search action"}

    # 构建过滤条件
    filter_dict = None
    filter_conditions = []
    if section:
        filter_conditions.append({"section": section})
    if author:
        filter_conditions.append({"authors": {"$contains": author}})
    if keyword:
        filter_conditions.append({"keywords": {"$contains": keyword}})
    if year_min is not None:
        filter_conditions.append({"year": {"$gte": year_min}})
    if year_max is not None:
        filter_conditions.append({"year": {"$lte": year_max}})
    if len(filter_conditions) == 1:
        filter_dict = filter_conditions[0]
    elif len(filter_conditions) > 1:
        filter_dict = {"$and": filter_conditions}

    # 执行检索
    if hybrid and not filter_dict:
        searcher = _get_hybrid_searcher()
        results = searcher.search(query, k=top_k, use_bm25=True)
    else:
        store = _get_paper_store()
        results = store.similarity_search_with_score(query, k=top_k, filter=filter_dict)

    # 重排序
    if rerank and len(results) >= RERANK_MIN_CANDIDATES:
        documents = [doc for doc, _ in results]
        results = rerank_with_scores(query, documents, top_k=top_k)

    return {
        "status": "success",
        "query": query,
        "total_results": len(results),
        "results": [_format_doc(doc, score) for doc, score in results],
    }


def _action_list(
    limit: int = 20,
    author: Optional[str] = None,
    keyword: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    **kwargs,
) -> dict[str, Any]:
    """列出论文"""
    store = _get_paper_store()
    results = store.similarity_search("", k=limit * 3)

    # 去重收集论文
    papers = {}
    for doc in results:
        meta = doc.metadata or {}
        paper_id = meta.get("paper_id", "unknown")
        if paper_id not in papers:
            authors_val = meta.get("authors", "")
            keywords_val = meta.get("keywords", "")
            authors = authors_val.split(",") if isinstance(authors_val, str) and authors_val else []
            keywords = keywords_val.split(",") if isinstance(keywords_val, str) and keywords_val else []
            papers[paper_id] = {
                "paper_id": paper_id,
                "title": meta.get("title", ""),
                "authors": authors,
                "keywords": keywords,
                "year": meta.get("year"),
                "sections": [],
            }
        section = meta.get("section", "")
        if section and section not in papers[paper_id]["sections"]:
            papers[paper_id]["sections"].append(section)

    # 过滤
    paper_list = list(papers.values())
    if author:
        paper_list = [p for p in paper_list if any(author.lower() in a.lower() for a in p.get("authors", []))]
    if keyword:
        paper_list = [p for p in paper_list if any(keyword.lower() in k.lower() for k in p.get("keywords", []))]
    if year_min is not None:
        paper_list = [p for p in paper_list if p.get("year") and p["year"] >= year_min]
    if year_max is not None:
        paper_list = [p for p in paper_list if p.get("year") and p["year"] <= year_max]

    return {"status": "success", "total_papers": len(paper_list[:limit]), "papers": paper_list[:limit]}


def _action_stats(**kwargs) -> dict[str, Any]:
    """统计信息"""
    store = _get_paper_store()
    total_chunks = store.count()
    results = store.similarity_search("", k=1000)

    paper_ids = set()
    sections = {}
    years = {}
    for doc in results:
        meta = doc.metadata or {}
        if meta.get("paper_id"):
            paper_ids.add(meta["paper_id"])
        if meta.get("section"):
            sections[meta["section"]] = sections.get(meta["section"], 0) + 1
        if meta.get("year"):
            years[meta["year"]] = years.get(meta["year"], 0) + 1

    return {
        "status": "success",
        "total_papers": len(paper_ids),
        "total_chunks": total_chunks,
        "sections_count": sections,
        "years_count": dict(sorted(years.items())),
    }


def _action_ingest(pdf_paths: list[str] = None, **kwargs) -> dict[str, Any]:
    """入库论文"""
    if not pdf_paths:
        return {"status": "error", "error": "pdf_paths is required"}

    from pathlib import Path
    import threading
    import time
    import hashlib
    from store.ingest_task import IngestTaskStore
    from utils.paper_parser import PaperParser

    # 验证文件
    valid_paths = [p for p in pdf_paths if Path(p).exists() and p.lower().endswith(".pdf")]
    if not valid_paths:
        return {"status": "error", "error": "No valid PDF files"}

    # 创建任务
    task_id = f"ingest_{int(time.time())}_{hashlib.md5('|'.join(sorted(valid_paths)).encode()).hexdigest()[:8]}"
    ingest_store = IngestTaskStore()
    ingest_store.create_task(task_id, len(valid_paths))

    # 后台执行
    def run_task():
        task = ingest_store.get_task(task_id)
        if not task:
            return
        task.status = "running"
        ingest_store.update_task(task)
        parser = PaperParser(_get_paper_store())
        try:
            for pdf_path in valid_paths:
                task = ingest_store.get_task(task_id)
                if task.status == "interrupted":
                    break
                try:
                    sections = parser.parse_pdf(pdf_path)
                    paper_id = Path(pdf_path).stem[:50]
                    for s in sections:
                        s.metadata["paper_id"] = paper_id
                    if sections:
                        from langchain_core.documents import Document
                        docs = [Document(page_content=s.content, metadata=s.metadata) for s in sections]
                        ids = [f"{paper_id}_{s.section_type}" for s in sections]
                        _get_paper_store().add_documents(docs, ids=ids)
                        title = sections[0].metadata.get("title", Path(pdf_path).stem)
                        ingest_store.add_paper_result(task_id, task, pdf_path, True, paper_id, title, [s.section_type for s in sections])
                    else:
                        ingest_store.add_paper_result(task_id, task, pdf_path, False, error="No content extracted")
                except Exception as e:
                    ingest_store.add_paper_result(task_id, task, pdf_path, False, error=str(e))
            task = ingest_store.get_task(task_id)
            if task.status != "interrupted":
                task.status = "completed"
                ingest_store.update_task(task)
        except Exception as e:
            task = ingest_store.get_task(task_id)
            task.status = "failed"
            task.error = str(e)
            ingest_store.update_task(task)

    thread = threading.Thread(target=run_task, daemon=True)
    thread.start()

    return {
        "status": "started",
        "task_id": task_id,
        "total_papers": len(valid_paths),
        "message": f"Ingestion started. Use paper_kb(action='ingest_status', task_id='{task_id}') to check progress.",
    }


def _action_ingest_status(task_id: str = None, **kwargs) -> dict[str, Any]:
    """查询入库状态"""
    if not task_id:
        return {"status": "error", "error": "task_id is required"}

    from store.ingest_task import IngestTaskStore
    ingest_store = IngestTaskStore()
    task = ingest_store.get_task(task_id)

    if not task:
        return {"status": "error", "error": f"Task not found: {task_id}"}

    progress = round(task.processed_papers / task.total_papers * 100, 1) if task.total_papers > 0 else 0

    return {
        "status": "success",
        "task_id": task.task_id,
        "task_status": task.status,
        "progress": f"{progress}%",
        "total_papers": task.total_papers,
        "processed_papers": task.processed_papers,
        "succeeded_papers": task.succeeded_papers,
        "failed_papers": task.failed_papers,
        "results": task.results[-5:] if task.results else [],
        "error": task.error,
    }


def _action_ingest_cancel(task_id: str = None, **kwargs) -> dict[str, Any]:
    """取消入库任务"""
    if not task_id:
        return {"status": "error", "error": "task_id is required"}

    from store.ingest_task import IngestTaskStore
    ingest_store = IngestTaskStore()
    task = ingest_store.get_task(task_id)

    if not task:
        return {"status": "error", "error": f"Task not found: {task_id}"}

    if task.status not in ("pending", "running"):
        return {"status": "error", "error": f"Task is {task.status}, cannot cancel"}

    task.status = "interrupted"
    ingest_store.update_task(task)

    return {
        "status": "success",
        "task_id": task_id,
        "message": "Task cancelled",
        "processed_papers": task.processed_papers,
    }
