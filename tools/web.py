"""Web 工具聚合 - 网络搜索和内容提取

聚合了以下来源的工具：
- Tavily: 网络搜索和 URL 内容提取
- Firecrawl: 网页抓取和爬取
- arXiv: 学术论文搜索和下载
"""

from typing import Any
from langchain_core.tools import tool

from utils.config import Settings


# ============ Tavily 工具 ============

@tool
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for information.

    Uses Tavily for general web search with excellent results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary containing search results with title, url, content, and score
    """
    settings = Settings()
    api_key = settings.tavily.api_key

    if not api_key:
        return {"error": "TAVILY_API_KEY not configured in config.yaml"}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        results = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )
        return results
    except ImportError:
        return {"error": "tavily-python not installed"}
    except Exception as e:
        return {"error": str(e)}


@tool
def web_fetch(urls: list[str]) -> dict[str, Any]:
    """Extract content from specific URLs.

    Uses Tavily to extract and summarize content from web pages.

    Args:
        urls: List of URLs to extract content from

    Returns:
        Dictionary containing extracted content for each URL
    """
    settings = Settings()
    api_key = settings.tavily.api_key

    if not api_key:
        return {"error": "TAVILY_API_KEY not configured in config.yaml"}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        results = client.extract(urls=urls)
        return results
    except ImportError:
        return {"error": "tavily-python not installed"}
    except Exception as e:
        return {"error": str(e)}


# ============ Firecrawl 工具 ============

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


@tool
def web_scrape(url: str, formats: list[str] | None = None) -> dict[str, Any]:
    """Scrape a single URL and extract content using Firecrawl.

    Args:
        url: URL to scrape
        formats: Content formats to extract (e.g., ["markdown", "html"]). Default: ["markdown"]

    Returns:
        Dictionary containing scraped content with markdown/html
    """
    if formats is None:
        formats = ["markdown"]

    client = _get_firecrawl_client()
    if client is None:
        return {"error": "FIRECRAWL_API_KEY not configured in config.yaml"}

    try:
        result = client.scrape(url=url, formats=formats)
        return result
    except Exception as e:
        return {"error": str(e)}


@tool
def web_crawl(url: str, max_depth: int = 1, limit: int = 10) -> dict[str, Any]:
    """Crawl a website starting from URL using Firecrawl.

    Args:
        url: Root URL to start crawling
        max_depth: Maximum crawl depth (default: 1)
        limit: Maximum number of pages to crawl (default: 10)

    Returns:
        Dictionary containing crawled pages with their content
    """
    client = _get_firecrawl_client()
    if client is None:
        return {"error": "FIRECRAWL_API_KEY not configured in config.yaml"}

    try:
        result = client.crawl(url=url, max_depth=max_depth, limit=limit)
        return result
    except Exception as e:
        return {"error": str(e)}


@tool
def web_map(url: str, search: str | None = None) -> dict[str, Any]:
    """Map a website to discover all indexed URLs using Firecrawl.

    Args:
        url: Root URL to map
        search: Optional search query to filter results

    Returns:
        Dictionary containing list of discovered URLs
    """
    client = _get_firecrawl_client()
    if client is None:
        return {"error": "FIRECRAWL_API_KEY not configured in config.yaml"}

    try:
        if search:
            result = client.search(url=url, query=search)
        else:
            result = client.map(url=url)
        return result
    except Exception as e:
        return {"error": str(e)}


# ============ arXiv 工具 ============

import re
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def arxiv_search(query: str, max_results: int = 5) -> dict:
    """Search arXiv for academic papers.

    Uses the arXiv Atom API (free, no API key required).

    Args:
        query: Search query string (supports Lucene syntax)
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary containing list of papers with metadata
    """
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()

        content = response.text
        papers = []
        entries = re.split(r'<entry>', content)[1:]

        for entry_xml in entries:
            def extract(tag):
                match = re.search(rf'<{tag}>(.*?)</{tag}>', entry_xml, re.DOTALL)
                return match.group(1).strip() if match else ""

            authors = []
            author_matches = re.findall(r'<author>.*?<name>(.*?)</name>.*?</author>', entry_xml, re.DOTALL)
            for a in author_matches:
                authors.append(a.strip())

            categories = re.findall(r"<category term=\"([^\"]+)\"", entry_xml)
            full_id = extract("id")
            arxiv_id_match = re.search(r'([0-9]+\.[0-9]+)', full_id)
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else full_id

            papers.append({
                "source": "arxiv",
                "paper_id": arxiv_id,
                "title": extract("title").replace("\n", " ").strip(),
                "authors": authors,
                "abstract": extract("summary").replace("\n", " ").strip(),
                "published": extract("published"),
                "updated": extract("updated"),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "categories": categories,
            })

        return {
            "success": True,
            "query": query,
            "count": len(papers),
            "papers": papers,
        }

    except httpx.HTTPError as e:
        return {"success": False, "error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to search arXiv: {str(e)}"}


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def arxiv_get_by_id(arxiv_id: str) -> dict:
    """Get a specific arXiv paper by its ID.

    Args:
        arxiv_id: arXiv ID (e.g., "2301.12345" or "2301.12345v2")

    Returns:
        Dictionary containing paper metadata
    """
    clean_id = re.sub(r'v[0-9]+$', '', arxiv_id)

    base_url = "https://export.arxiv.org/api/query"
    params = {"id_list": clean_id, "max_results": 1}

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()

        content = response.text
        entries = re.split(r'<entry>', content)[1:]

        if not entries:
            return {"success": False, "error": f"Paper {arxiv_id} not found"}

        entry_xml = entries[0]

        def extract(tag):
            match = re.search(rf'<{tag}>(.*?)</{tag}>', entry_xml, re.DOTALL)
            return match.group(1).strip() if match else ""

        authors = []
        author_matches = re.findall(r'<author>.*?<name>(.*?)</name>.*?</author>', entry_xml, re.DOTALL)
        for a in author_matches:
            authors.append(a.strip())

        categories = re.findall(r"<category term=\"([^\"]+)\"", entry_xml)
        full_id = extract("id")
        final_id_match = re.search(r'([0-9]+\.[0-9]+)', full_id)
        final_id = final_id_match.group(1) if final_id_match else clean_id

        return {
            "success": True,
            "source": "arxiv",
            "paper_id": final_id,
            "title": extract("title").replace("\n", " ").strip(),
            "authors": authors,
            "abstract": extract("summary").replace("\n", " ").strip(),
            "published": extract("published"),
            "updated": extract("updated"),
            "url": f"https://arxiv.org/abs/{final_id}",
            "pdf_url": f"https://arxiv.org/pdf/{final_id}.pdf",
            "categories": categories,
        }

    except httpx.HTTPError as e:
        return {"success": False, "error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to get arXiv paper: {str(e)}"}


@tool
def arxiv_download_pdf(arxiv_id: str, save_dir: str = "./downloads/papers") -> dict:
    """Download arXiv paper PDF.

    Args:
        arxiv_id: arXiv ID (e.g., "2301.12345")
        save_dir: Directory to save the PDF

    Returns:
        Dictionary containing download result with file path
    """
    from pathlib import Path

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

    except httpx.HTTPError as e:
        return {"success": False, "error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to download PDF: {str(e)}"}
