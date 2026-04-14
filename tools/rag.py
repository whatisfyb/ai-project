"""RAG 工具 - 论文知识库检索

提供基于向量数据库的论文内容检索能力。
"""

from typing import Any, Optional, Literal
from langchain_core.tools import tool

from utils.vector_store import VectorStore
from utils.reranker import rerank_with_scores
from utils.hybrid_search import HybridSearcher, rrf_fusion
from utils.whoosh_index import get_whoosh_index


# 默认的 papers collection
DEFAULT_COLLECTION = "papers"
DEFAULT_PERSIST_DIR = ".data/chroma"


def _get_paper_store() -> VectorStore:
    """获取论文向量存储"""
    return VectorStore(
        collection_name=DEFAULT_COLLECTION,
        persist_dir=DEFAULT_PERSIST_DIR,
    )


def _get_hybrid_searcher() -> HybridSearcher:
    """获取混合检索器"""
    vector_store = _get_paper_store()
    whoosh_index = get_whoosh_index()
    return HybridSearcher(vector_store, whoosh_index)


# 重排序触发阈值：候选结果 >= 此值时才触发重排序
RERANK_MIN_CANDIDATES = 3


def _merge_multi_query_results(
    results: list[tuple],
    top_k: int,
) -> list[tuple]:
    """融合多查询结果（按文档 ID 去重，保留最高分）

    Args:
        results: 所有查询的结果列表
        top_k: 返回数量

    Returns:
        去重后的结果列表
    """
    doc_scores: dict[str, tuple] = {}

    for doc, score in results:
        doc_id = doc.metadata.get("id", doc.page_content[:50])

        if doc_id not in doc_scores:
            doc_scores[doc_id] = (doc, score)
        else:
            # 保留最高分
            _, old_score = doc_scores[doc_id]
            if score > old_score:
                doc_scores[doc_id] = (doc, score)

    # 按分数排序
    sorted_results = sorted(
        doc_scores.values(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results[:top_k]


@tool
def paper_search(
    query: str,
    top_k: int = 5,
    section: Optional[Literal["abstract", "introduction", "conclusion"]] = None,
    author: Optional[str] = None,
    keyword: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    rerank: bool = True,
    hybrid: bool = True,
    expand_query: bool = False,
) -> dict[str, Any]:
    """Search for relevant content from the paper knowledge base.

    Use this tool to retrieve information from previously indexed academic papers.
    Supports filtering by section type, author, keyword, and publication year.
    Uses hybrid search (vector + BM25) for better recall, and cross-encoder reranking for precision.

    Args:
        query: The search query describing what information you need
        top_k: Maximum number of results to return (default: 5)
        section: Filter by paper section - "abstract", "introduction", or "conclusion"
        author: Filter by author name (partial match)
        keyword: Filter by keyword (partial match)
        year_min: Minimum publication year (inclusive)
        year_max: Maximum publication year (inclusive)
        rerank: Whether to rerank results using cross-encoder (default: True)
        hybrid: Whether to use hybrid search (vector + BM25) (default: True)
        expand_query: Whether to expand query using LLM for better recall (default: False)

    Returns:
        Dictionary containing search results with paper metadata and relevant content
    """
    try:
        # 查询扩展
        queries = [query]
        if expand_query:
            from utils.query_rewriter import expand_query as do_expand
            queries = do_expand(query, n_expansions=2)

        # 构建元数据过滤条件
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

        # 合并过滤条件
        filter_dict = None
        if filter_conditions:
            if len(filter_conditions) == 1:
                filter_dict = filter_conditions[0]
            else:
                filter_dict = {"$and": filter_conditions}

        # 执行检索
        if hybrid and not filter_dict:
            # 混合检索（不支持元数据过滤）
            searcher = _get_hybrid_searcher()

            if len(queries) == 1:
                # 单查询
                results = searcher.search(query, k=top_k, use_bm25=True)
            else:
                # 多查询：分别检索，融合结果
                all_results = []
                for q in queries:
                    all_results.extend(searcher.search(q, k=top_k * 2, use_bm25=True))

                # 按文档 ID 去重融合
                results = _merge_multi_query_results(all_results, top_k)
        else:
            # 纯向量检索（支持元数据过滤）
            store = _get_paper_store()
            results = store.similarity_search_with_score(query, k=top_k, filter=filter_dict)

        # 重排序：候选结果 >= 阈值时触发
        if rerank and len(results) >= RERANK_MIN_CANDIDATES:
            documents = [doc for doc, _ in results]
            reranked = rerank_with_scores(query, documents, top_k=top_k)
            # 用重排序分数替换原来的向量距离
            results = [(doc, score) for doc, score in reranked]

        # 格式化结果
        formatted_results = []
        for doc, score in results:
            meta = doc.metadata or {}
            # authors 和 keywords 可能是逗号分隔的字符串，也可能是列表
            authors_val = meta.get("authors", "")
            keywords_val = meta.get("keywords", "")

            if isinstance(authors_val, str):
                authors = authors_val.split(",") if authors_val else []
            elif isinstance(authors_val, list):
                authors = authors_val
            else:
                authors = []

            if isinstance(keywords_val, str):
                keywords = keywords_val.split(",") if keywords_val else []
            elif isinstance(keywords_val, list):
                keywords = keywords_val
            else:
                keywords = []

            formatted_results.append({
                "paper_id": meta.get("paper_id", "unknown"),
                "title": meta.get("title", ""),
                "authors": authors,
                "keywords": keywords,
                "year": meta.get("year"),
                "source": meta.get("source", ""),
                "section": meta.get("section", ""),
                "content": doc.page_content,
                "relevance_score": round(float(score), 4) if score else None,
            })

        return {
            "status": "success",
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "results": [],
        }


@tool
def paper_list(
    limit: int = 20,
    author: Optional[str] = None,
    keyword: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> dict[str, Any]:
    """List papers in the knowledge base with optional filters.

    Use this tool to see what papers are available in the knowledge base
    before searching for specific content.

    Args:
        limit: Maximum number of papers to list (default: 20)
        author: Filter by author name (partial match)
        keyword: Filter by keyword (partial match)
        year_min: Minimum publication year (inclusive)
        year_max: Maximum publication year (inclusive)

    Returns:
        Dictionary containing list of papers with their metadata
    """
    try:
        store = _get_paper_store()

        # 使用空查询获取所有文档（按相关性排序无意义，但可以获取元数据）
        # ChromaDB 不直接支持 "get all"，我们用一个通用查询
        results = store.similarity_search("", k=limit * 3)  # 获取更多以去重

        # 去重并收集论文元数据
        papers = {}
        for doc in results:
            meta = doc.metadata or {}
            paper_id = meta.get("paper_id", "unknown")

            if paper_id not in papers:
                # authors 和 keywords 可能是字符串或列表
                authors_val = meta.get("authors", "")
                keywords_val = meta.get("keywords", "")

                if isinstance(authors_val, str):
                    authors = authors_val.split(",") if authors_val else []
                elif isinstance(authors_val, list):
                    authors = authors_val
                else:
                    authors = []

                if isinstance(keywords_val, str):
                    keywords = keywords_val.split(",") if keywords_val else []
                elif isinstance(keywords_val, list):
                    keywords = keywords_val
                else:
                    keywords = []

                paper_info = {
                    "paper_id": paper_id,
                    "title": meta.get("title", ""),
                    "authors": authors,
                    "keywords": keywords,
                    "year": meta.get("year"),
                    "source": meta.get("source", ""),
                    "sections": [],
                }
                papers[paper_id] = paper_info

            # 添加章节信息
            section = meta.get("section", "")
            if section and section not in papers[paper_id]["sections"]:
                papers[paper_id]["sections"].append(section)

        # 转为列表
        paper_list = list(papers.values())

        # 应用过滤
        if author:
            paper_list = [
                p for p in paper_list
                if any(author.lower() in a.lower() for a in p.get("authors", []))
            ]

        if keyword:
            paper_list = [
                p for p in paper_list
                if any(keyword.lower() in k.lower() for k in p.get("keywords", []))
            ]

        if year_min is not None:
            paper_list = [p for p in paper_list if p.get("year") and p["year"] >= year_min]

        if year_max is not None:
            paper_list = [p for p in paper_list if p.get("year") and p["year"] <= year_max]

        # 限制数量
        paper_list = paper_list[:limit]

        return {
            "status": "success",
            "total_papers": len(paper_list),
            "papers": paper_list,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "papers": [],
        }


@tool
def paper_stats() -> dict[str, Any]:
    """Get statistics about the paper knowledge base.

    Returns:
        Dictionary containing statistics like total papers, total chunks, etc.
    """
    try:
        store = _get_paper_store()
        total_chunks = store.count()

        # 获取所有文档来统计论文数
        results = store.similarity_search("", k=1000)
        paper_ids = set()
        sections = {}
        years = {}

        for doc in results:
            meta = doc.metadata or {}
            paper_id = meta.get("paper_id")
            if paper_id:
                paper_ids.add(paper_id)

            section = meta.get("section", "")
            if section:
                sections[section] = sections.get(section, 0) + 1

            year = meta.get("year")
            if year:
                years[year] = years.get(year, 0) + 1

        return {
            "status": "success",
            "total_papers": len(paper_ids),
            "total_chunks": total_chunks,
            "sections_count": sections,
            "years_count": dict(sorted(years.items())),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


@tool
def paper_build_index() -> dict[str, Any]:
    """Build full-text index from the vector store for hybrid search.

    Call this after ingesting new papers to enable full-text + vector hybrid search.
    The index is persisted to disk and loaded automatically on next search.

    Returns:
        Dictionary containing build status and document count
    """
    try:
        searcher = _get_hybrid_searcher()
        searcher.build_index()

        whoosh_index = get_whoosh_index()
        doc_count = whoosh_index.count()

        return {
            "status": "success",
            "message": f"全文索引构建完成，共 {doc_count} 篇文档",
            "document_count": doc_count,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ============ 论文入库工具 ============

import threading
import time
import hashlib
from pathlib import Path

from store.ingest_task import IngestTaskStore, IngestTask
from utils.paper_parser import (
    PaperParser, PaperMeta, validate_paper_format, validate_and_parse
)


def _get_ingest_store() -> IngestTaskStore:
    """获取入库任务存储"""
    return IngestTaskStore()


def _generate_task_id(pdf_paths: list[str]) -> str:
    """根据 PDF 路径生成任务 ID"""
    combined = "|".join(sorted(pdf_paths))
    hash_val = hashlib.md5(combined.encode()).hexdigest()[:8]
    return f"ingest_{int(time.time())}_{hash_val}"


def _generate_paper_id(file_path: str) -> str:
    """根据文件路径生成论文 ID"""
    stem = Path(file_path).stem
    # 清理非法字符
    clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)
    return clean[:50] or f"paper_{int(time.time())}"


def _run_ingest_task(
    task_id: str,
    pdf_paths: list[str],
    store: VectorStore,
    ingest_store: IngestTaskStore,
):
    """后台执行入库任务"""
    task = ingest_store.get_task(task_id)
    if not task:
        return

    task.status = "running"
    ingest_store.update_task(task)

    parser = PaperParser(store)

    try:
        for pdf_path in pdf_paths:
            # 检查中断
            task = ingest_store.get_task(task_id)
            if task.status == "interrupted":
                break

            # 检查是否已入库（根据文件路径去重）
            existing = store.similarity_search("", k=1000)
            file_exists = any(
                pdf_path in (doc.metadata.get("file_path", "") or "")
                for doc in existing
            )

            if file_exists:
                ingest_store.add_paper_result(
                    task_id, task,
                    file_path=pdf_path,
                    success=False,
                    error="论文已在知识库中（路径重复）",
                )
                continue

            # 验证论文格式
            validation = validate_paper_format(pdf_path)

            if not validation.is_valid:
                ingest_store.add_paper_result(
                    task_id, task,
                    file_path=pdf_path,
                    success=False,
                    error=validation.message,
                )
                continue

            # 解析并入库
            paper_id = _generate_paper_id(pdf_path)

            try:
                # 不传 meta，让 parse_pdf 自动提取元数据
                sections = parser.parse_pdf(pdf_path)

                # 更新 paper_id（自动生成的可能不准确）
                for section in sections:
                    section.metadata["paper_id"] = paper_id

                if not sections:
                    ingest_store.add_paper_result(
                        task_id, task,
                        file_path=pdf_path,
                        success=False,
                        error="无法提取论文内容",
                    )
                    continue

                # 入库
                documents = []
                ids = []
                for section in sections:
                    from langchain_core.documents import Document
                    section.metadata["file_path"] = pdf_path
                    documents.append(Document(
                        page_content=section.content,
                        metadata=section.metadata,
                    ))
                    ids.append(f"{paper_id}_{section.section_type}")

                store.add_documents(documents, ids=ids)

                # 获取标题
                title = sections[0].metadata.get("title", Path(pdf_path).stem)

                ingest_store.add_paper_result(
                    task_id, task,
                    file_path=pdf_path,
                    success=True,
                    paper_id=paper_id,
                    title=title,
                    sections=[s.section_type for s in sections],
                )

            except Exception as e:
                ingest_store.add_paper_result(
                    task_id, task,
                    file_path=pdf_path,
                    success=False,
                    error=str(e),
                )

        # 标记完成
        task = ingest_store.get_task(task_id)
        if task.status != "interrupted":
            task.status = "completed"
            ingest_store.update_task(task)

    except Exception as e:
        task = ingest_store.get_task(task_id)
        task.status = "failed"
        task.error = str(e)
        ingest_store.update_task(task)


@tool
def paper_ingest(
    pdf_paths: list[str],
) -> dict[str, Any]:
    """Ingest PDF papers into the knowledge base asynchronously.

    Validates that each PDF is a proper academic paper (must have abstract, introduction, conclusion).
    Papers that don't meet the criteria will be skipped.
    Runs in background - use paper_ingest_status to check progress.

    Args:
        pdf_paths: List of PDF file paths to ingest

    Returns:
        Dictionary containing task_id and initial status
    """
    if not pdf_paths:
        return {
            "status": "error",
            "error": "PDF 路径列表不能为空",
        }

    # 验证文件存在
    valid_paths = []
    invalid_paths = []

    for path in pdf_paths:
        if Path(path).exists() and path.lower().endswith(".pdf"):
            valid_paths.append(path)
        else:
            invalid_paths.append(path)

    if not valid_paths:
        return {
            "status": "error",
            "error": "没有有效的 PDF 文件",
            "invalid_paths": invalid_paths,
        }

    # 去重（根据文件路径）
    unique_paths = list(dict.fromkeys(valid_paths))

    # 创建任务
    task_id = _generate_task_id(unique_paths)
    ingest_store = _get_ingest_store()
    ingest_store.create_task(task_id, len(unique_paths))

    # 启动后台任务
    store = _get_paper_store()
    thread = threading.Thread(
        target=_run_ingest_task,
        args=(task_id, unique_paths, store, ingest_store),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "task_id": task_id,
        "total_papers": len(unique_paths),
        "invalid_paths": invalid_paths,
        "message": f"入库任务已启动，共 {len(unique_paths)} 篇论文。使用 paper_ingest_status(task_id='{task_id}') 查询进度。",
    }


@tool
def paper_ingest_status(task_id: str) -> dict[str, Any]:
    """Check the status of a paper ingestion task.

    Args:
        task_id: The task ID returned by paper_ingest

    Returns:
        Dictionary containing task status and progress
    """
    ingest_store = _get_ingest_store()
    task = ingest_store.get_task(task_id)

    if not task:
        return {
            "status": "error",
            "error": f"任务不存在: {task_id}",
        }

    # 计算进度百分比
    progress = 0
    if task.total_papers > 0:
        progress = round(task.processed_papers / task.total_papers * 100, 1)

    return {
        "status": "success",
        "task_id": task.task_id,
        "task_status": task.status,
        "progress": f"{progress}%",
        "total_papers": task.total_papers,
        "processed_papers": task.processed_papers,
        "succeeded_papers": task.succeeded_papers,
        "failed_papers": task.failed_papers,
        "results": task.results[-5:] if task.results else [],  # 最近5条结果
        "error": task.error,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


@tool
def paper_ingest_list(
    status: Optional[str] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """List paper ingestion tasks.

    Args:
        status: Filter by status (pending, running, completed, failed, interrupted)
        limit: Maximum number of tasks to return

    Returns:
        Dictionary containing list of tasks
    """
    ingest_store = _get_ingest_store()
    tasks = ingest_store.list_tasks(status=status, limit=limit)

    return {
        "status": "success",
        "total_tasks": len(tasks),
        "tasks": [
            {
                "task_id": t.task_id,
                "status": t.status,
                "total_papers": t.total_papers,
                "processed_papers": t.processed_papers,
                "succeeded_papers": t.succeeded_papers,
                "failed_papers": t.failed_papers,
                "created_at": t.created_at,
                "updated_at": t.updated_at,
            }
            for t in tasks
        ],
    }


@tool
def paper_ingest_cancel(task_id: str) -> dict[str, Any]:
    """Cancel a running paper ingestion task.

    Args:
        task_id: The task ID to cancel

    Returns:
        Dictionary containing cancellation status
    """
    ingest_store = _get_ingest_store()
    task = ingest_store.get_task(task_id)

    if not task:
        return {
            "status": "error",
            "error": f"任务不存在: {task_id}",
        }

    if task.status not in ("pending", "running"):
        return {
            "status": "error",
            "error": f"任务状态为 {task.status}，无法取消",
        }

    task.status = "interrupted"
    ingest_store.update_task(task)

    return {
        "status": "success",
        "task_id": task_id,
        "message": "任务已标记为中断",
        "processed_papers": task.processed_papers,
        "succeeded_papers": task.succeeded_papers,
    }
