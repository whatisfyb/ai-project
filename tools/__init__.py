"""Tools for LLM/Agent"""

# 合并后的核心工具
from tools.web import web
from tools.agent import agent
from tools.paper_kb import paper_kb
from tools.task import task

# Skills
from tools.skills import load_skills, list_skills, get_skill, clear_skills_cache, skill_call

# 文件操作工具
from tools.bash import bash, bash_script
from tools.edit import edit, edit_regex
from tools.glob import glob, glob_list
from tools.grep import grep, grep_count
from tools.read import read
from tools.write import write, append

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
    # File operations
    "bash",
    "bash_script",
    "edit",
    "edit_regex",
    "glob",
    "glob_list",
    "grep",
    "grep_count",
    "read",
    "write",
    "append",
]
