"""Firecrawl Web Scraping & Crawling 工具"""

from typing import Any
from langchain_core.tools import tool

from utils.config import Settings


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
def firecrawl_scrape(url: str, formats: list[str] | None = None) -> dict[str, Any]:
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
        result = client.scrape_url(
            url=url,
            formats=formats,
        )
        return result
    except Exception as e:
        return {"error": str(e)}


@tool
def firecrawl_crawl(
    url: str,
    max_depth: int = 1,
    limit: int = 10,
) -> dict[str, Any]:
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
        result = client.crawl_url(
            url=url,
            max_depth=max_depth,
            limit=limit,
        )
        return result
    except Exception as e:
        return {"error": str(e)}


@tool
def firecrawl_map(url: str, search: str | None = None) -> dict[str, Any]:
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
            result = client.search_url(url=url, query=search)
        else:
            result = client.map_url(url=url)
        return result
    except Exception as e:
        return {"error": str(e)}
