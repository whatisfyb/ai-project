"""Tools for LLM/Agent
"""

from tools.web import (
    web_search, web_fetch, web_scrape, web_crawl, web_map,
    arxiv_search, arxiv_get_by_id, arxiv_download_pdf,
)
from tools.skills import load_skills, list_skills, get_skill, clear_skills_cache, skill_call
from tools.agent import dispatch_agent, list_subagents
from tools.task import (
    plan_get, plan_execute,
    task_add, task_update, task_delete, task_get,
)

__all__ = [
    # Web search/fetch (Tavily + Firecrawl)
    "web_search",
    "web_fetch",
    "web_scrape",
    "web_crawl",
    "web_map",
    # arXiv
    "arxiv_search",
    "arxiv_get_by_id",
    "arxiv_download_pdf",
    # Skills
    "load_skills",
    "list_skills",
    "skill_call",
    "get_skill",
    "clear_skills_cache",
    # Agent
    "dispatch_agent",
    "list_subagents",
    # Plan/Task management
    "plan_get",
    "plan_execute",
    "task_add",
    "task_update",
    "task_delete",
    "task_get",
]
