"""Tavily Web Search 工具"""

from typing import Any
from langchain_core.tools import tool

from utils.config import Settings


@tool
def tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web using Tavily.

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
def tavily_extract(urls: list[str]) -> dict[str, Any]:
    """Extract content from specific URLs using Tavily.

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
