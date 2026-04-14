"""Tools for LLM/Agent"""

# 合并后的核心工具
from tools.web import web
from tools.agent import agent
from tools.paper_kb import paper_kb
from tools.task import task

# Skills
from tools.skills import load_skills, list_skills, get_skill, clear_skills_cache, skill_call

__all__ = [
    # Web (merged: search, fetch, scrape, crawl, map, arxiv)
    "web",
    # Agent (merged: dispatch, list)
    "agent",
    # Paper KB (merged: search, list, stats, ingest)
    "paper_kb",
    # Task (merged: get_plan, add, update, delete, get)
    "task",
    # Skills
    "load_skills",
    "list_skills",
    "skill_call",
    "get_skill",
    "clear_skills_cache",
]
