"""Web 工具 - 统一入口

合并了 Tavily、Firecrawl、arXiv 的所有工具。
"""

import re
from pathlib import Path
from typing import Any, Literal

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.tools import tool

from utils.core.config import Settings


# ============ 客户端获取 ============

def _get_tavily_client():
    """获取 Tavily 客户端"""
    settings = Settings()
    api_key = settings.tavily.api_key
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=api_key)
    except ImportError:
        return None


def _get_firecrawl_client():
    """获取 Firecrawl 客户端"""
    settings = Settings()
    api_key = settings.firecrawl.api_key
    if not api_key:
        return None
    try:
        from firecrawl import FirecrawlApp
        return FirecrawlApp(api_key=api_key)
    except ImportError:
        return None


def _pydantic_to_dict(obj):
    """Pydantic 对象转 dict"""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif isinstance(obj, list):
        return [_pydantic_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _pydantic_to_dict(v) for k, v in obj.items()}
    return obj


# ============ 统一工具 ============

@tool
def web(
    action: Literal["search", "fetch", "scrape", "crawl", "map", "arxiv_search", "arxiv_get", "arxiv_download"],
    query: str = "",
    url: str = "",
    urls: list[str] = [],
    max_results: int = 5,
    formats: list[str] = ["markdown"],
    max_depth: int = 1,
    limit: int = 10,
    search: str = "",
    arxiv_id: str = "",
    save_dir: str = "./downloads/papers",
) -> dict[str, Any]:
    """Web operations: search, fetch, scrape, crawl, and arXiv.

    Unified tool for web search, content extraction, and academic paper operations.

    Args:
        action: Operation to perform:
            - "search": Web search (uses: query, max_results)
            - "fetch": Extract URLs content (uses: urls)
            - "scrape": Scrape single URL (uses: url, formats)
            - "crawl": Crawl website (uses: url, max_depth, limit)
            - "map": Discover URLs (uses: url, search)
            - "arxiv_search": Search arXiv (uses: query, max_results)
            - "arxiv_get": Get arXiv paper (uses: arxiv_id)
            - "arxiv_download": Download arXiv PDF (uses: arxiv_id, save_dir)
        query: Search query (for search/arxiv_search)
        url: Single URL (for scrape/crawl/map)
        urls: List of URLs (for fetch)
        max_results: Max results for search (default: 5)
        formats: Output formats for scrape (default: ["markdown"])
        max_depth: Crawl depth (default: 1)
        limit: Crawl limit (default: 10)
        search: Search pattern for map
        arxiv_id: arXiv paper ID
        save_dir: Directory to save PDF (default: "./downloads/papers")

    Returns:
        Operation result with success status and data.
    """
    try:
        if action == "search":
            return _web_search(query, max_results)
        elif action == "fetch":
            return _web_fetch(urls)
        elif action == "scrape":
            return _web_scrape(url, formats)
        elif action == "crawl":
            return _web_crawl(url, max_depth, limit)
        elif action == "map":
            return _web_map(url, search if search else None)
        elif action == "arxiv_search":
            return _arxiv_search(query, max_results)
        elif action == "arxiv_get":
            return _arxiv_get(arxiv_id)
        elif action == "arxiv_download":
            return _arxiv_download(arxiv_id, save_dir)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ Tavily 实现 ============

def _web_search(query: str, max_results: int) -> dict[str, Any]:
    """网络搜索"""
    if not query:
        return {"success": False, "error": "query is required"}

    client = _get_tavily_client()
    if not client:
        return {"success": False, "error": "Tavily API key not configured"}

    try:
        results = client.search(query=query, max_results=max_results, include_answer=True)
        return results
    except Exception as e:
        return {"success": False, "error": str(e)}


def _web_fetch(urls: list[str]) -> dict[str, Any]:
    """提取 URL 内容"""
    if not urls:
        return {"success": False, "error": "urls is required"}

    client = _get_tavily_client()
    if not client:
        return {"success": False, "error": "Tavily API key not configured"}

    try:
        results = client.extract(urls=urls)
        return results
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ Firecrawl 实现 ============

def _web_scrape(url: str, formats: list[str]) -> dict[str, Any]:
    """抓取网页"""
    if not url:
        return {"success": False, "error": "url is required"}

    client = _get_firecrawl_client()
    if not client:
        return {"success": False, "error": "Firecrawl API key not configured"}

    try:
        result = client.scrape(url=url, formats=formats)
        return _pydantic_to_dict(result)
    except Exception as e:
        return {"success": False, "error": str(e)}


def _web_crawl(url: str, max_depth: int, limit: int) -> dict[str, Any]:
    """爬取网站"""
    if not url:
        return {"success": False, "error": "url is required"}

    client = _get_firecrawl_client()
    if not client:
        return {"success": False, "error": "Firecrawl API key not configured"}

    try:
        result = client.crawl(url=url, max_discovery_depth=max_depth, limit=limit)
        return _pydantic_to_dict(result)
    except Exception as e:
        return {"success": False, "error": str(e)}


def _web_map(url: str, search: str = None) -> dict[str, Any]:
    """发现网站 URL"""
    if not url:
        return {"success": False, "error": "url is required"}

    client = _get_firecrawl_client()
    if not client:
        return {"success": False, "error": "Firecrawl API key not configured"}

    try:
        if search:
            result = client.map(url=url, search=search)
        else:
            result = client.map(url=url)
        return _pydantic_to_dict(result)
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ arXiv 实现 ============

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _arxiv_search(query: str, max_results: int) -> dict[str, Any]:
    """搜索 arXiv"""
    if not query:
        return {"success": False, "error": "query is required"}

    base_url = "https://export.arxiv.org/api/query"
    params = {"search_query": query, "start": 0, "max_results": max_results, "sortBy": "relevance", "sortOrder": "descending"}

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()

        papers = []
        entries = re.split(r'<entry>', response.text)[1:]

        for entry_xml in entries:
            def extract(tag):
                match = re.search(rf'<{tag}>(.*?)</{tag}>', entry_xml, re.DOTALL)
                return match.group(1).strip() if match else ""

            authors = re.findall(r'<author>.*?<name>(.*?)</name>.*?</author>', entry_xml, re.DOTALL)
            authors = [a.strip() for a in authors]
            categories = re.findall(r'<category term="([^"]+)"', entry_xml)
            full_id = extract("id")
            arxiv_id = re.search(r'([0-9]+\.[0-9]+)', full_id)
            arxiv_id = arxiv_id.group(1) if arxiv_id else full_id

            papers.append({
                "source": "arxiv",
                "paper_id": arxiv_id,
                "title": extract("title").replace("\n", " ").strip(),
                "authors": authors,
                "abstract": extract("summary").replace("\n", " ").strip(),
                "published": extract("published"),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "categories": categories,
            })

        return {"success": True, "count": len(papers), "papers": papers}

    except Exception as e:
        return {"success": False, "error": str(e)}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _arxiv_get(arxiv_id: str) -> dict[str, Any]:
    """获取 arXiv 论文"""
    if not arxiv_id:
        return {"success": False, "error": "arxiv_id is required"}

    clean_id = re.sub(r'v[0-9]+$', '', arxiv_id)
    base_url = "https://export.arxiv.org/api/query"

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(base_url, params={"id_list": clean_id, "max_results": 1})
            response.raise_for_status()

        entries = re.split(r'<entry>', response.text)[1:]
        if not entries:
            return {"success": False, "error": f"Paper {arxiv_id} not found"}

        entry_xml = entries[0]

        def extract(tag):
            match = re.search(rf'<{tag}>(.*?)</{tag}>', entry_xml, re.DOTALL)
            return match.group(1).strip() if match else ""

        authors = re.findall(r'<author>.*?<name>(.*?)</name>.*?</author>', entry_xml, re.DOTALL)
        authors = [a.strip() for a in authors]
        categories = re.findall(r'<category term="([^"]+)"', entry_xml)
        full_id = extract("id")
        final_id = re.search(r'([0-9]+\.[0-9]+)', full_id)
        final_id = final_id.group(1) if final_id else clean_id

        return {
            "success": True,
            "source": "arxiv",
            "paper_id": final_id,
            "title": extract("title").replace("\n", " ").strip(),
            "authors": authors,
            "abstract": extract("summary").replace("\n", " ").strip(),
            "published": extract("published"),
            "url": f"https://arxiv.org/abs/{final_id}",
            "pdf_url": f"https://arxiv.org/pdf/{final_id}.pdf",
            "categories": categories,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _arxiv_download(arxiv_id: str, save_dir: str) -> dict[str, Any]:
    """下载 arXiv PDF"""
    if not arxiv_id:
        return {"success": False, "error": "arxiv_id is required"}

    clean_id = re.sub(r'v[0-9]+$', '', arxiv_id)
    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{clean_id}.pdf"

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            response = client.get(pdf_url)
            response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        return {
            "success": True,
            "arxiv_id": clean_id,
            "pdf_url": pdf_url,
            "saved_path": str(file_path.absolute()),
            "file_size": file_path.stat().st_size,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
