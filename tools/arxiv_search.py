"""arXiv 搜索工具 - 搜索、获取、下载 arXiv 论文"""

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ArxivPaper:
    """arXiv 论文结构"""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str  # ISO date
    updated: str    # ISO date
    url: str
    pdf_url: str
    categories: list[str]


def _parse_arxiv_id_from_url(url: str) -> Optional[str]:
    """从 URL 中提取 arXiv ID"""
    # 支持多种 URL 格式
    patterns = [
        r'arxiv\.org/abs/([0-9]+\.[0-9]+)',
        r'arxiv\.org/pdf/([0-9]+\.[0-9]+)',
        r'arxiv\.org/abs/([0-9]+\.[0-9]+v[0-9]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1).rstrip('v')
    return None


def _parse_atom_entry(entry: dict) -> ArxivPaper:
    """解析 ATOM entry 为 ArxivPaper"""
    # 提取 ID（如 2301.12345）
    full_id = entry.get("id", "")
    arxiv_id_match = re.search(r'([0-9]+\.[0-9]+)', full_id)
    arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else full_id

    # 提取作者
    authors = []
    if isinstance(entry.get("author"), list):
        authors = [a.get("name", "") for a in entry["author"]]
    elif entry.get("author"):
        authors = [entry["author"].get("name", "")]

    # 提取分类
    categories = []
    if isinstance(entry.get("category"), list):
        categories = [c.get("term", "") for c in entry["category"]]
    elif entry.get("category"):
        categories = [entry["category"].get("term", "")]

    # PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=entry.get("title", "").replace("\n", " ").strip(),
        authors=authors,
        abstract=entry.get("summary", "").replace("\n", " ").strip(),
        published=entry.get("published", ""),
        updated=entry.get("updated", ""),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf_url,
        categories=categories,
    )


@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def arxiv_search(query: str, max_results: int = 5) -> dict:
    """Search arXiv for papers.

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

        # 解析 XML 响应 (arXiv API 返回 ATOM XML)
        # 简单解析：用正则提取关键字段
        content = response.text

        papers = []
        # 分割各个 entry
        entries = re.split(r'<entry>', content)[1:]

        for entry_xml in entries:
            # 提取字段
            def extract(tag):
                match = re.search(rf'<{tag}>(.*?)</{tag}>', entry_xml, re.DOTALL)
                return match.group(1).strip() if match else ""

            def extract_list(tag):
                matches = re.findall(rf'<{tag}>(.*?)</{tag}>', entry_xml, re.DOTALL)
                return matches

            # 解析作者（可能有多个）
            authors = []
            author_matches = re.findall(r'<author>.*?<name>(.*?)</name>.*?</author>', entry_xml, re.DOTALL)
            for a in author_matches:
                authors.append(a.strip())

            # 解析分类
            categories = re.findall(r"<category term=\"([^\"]+)\"", entry_xml)

            # 提取 ID
            full_id = extract("id")
            arxiv_id_match = re.search(r'([0-9]+\.[0-9]+)', full_id)
            arxiv_id = arxiv_id_match.group(1) if arxiv_id_match else full_id

            paper = {
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
            }
            papers.append(paper)

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
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def arxiv_get_by_id(arxiv_id: str) -> dict:
    """Get a specific arXiv paper by its ID.

    Args:
        arxiv_id: arXiv ID (e.g., "2301.12345" or "2301.12345v2")

    Returns:
        Dictionary containing paper metadata
    """
    # 清理 ID（移除 v 版本号用于 API 查询）
    clean_id = re.sub(r'v[0-9]+$', '', arxiv_id)

    base_url = "https://export.arxiv.org/api/query"
    params = {
        "id_list": clean_id,
        "max_results": 1,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(base_url, params=params)
            response.raise_for_status()

        content = response.text
        entries = re.split(r'<entry>', content)[1:]

        if not entries:
            return {"success": False, "error": f"Paper {arxiv_id} not found"}

        entry_xml = entries[0]

        # 提取字段
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
    # 清理 ID
    clean_id = re.sub(r'v[0-9]+$', '', arxiv_id)
    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 文件名：arxiv_id.pdf
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
