"""工具注册与初始化"""


def get_main_agent_tools() -> list:
    """获取 Main Agent 可用的工具列表"""
    from tools.web import (
        web_search, web_fetch, web_scrape, web_crawl, web_map,
        arxiv_search, arxiv_get_by_id, arxiv_download_pdf,
    )
    from tools.agent import dispatch_agent, list_subagents
    from tools.skills import load_skills, list_skills, skill_call
    from tools.task import (
        plan_get, plan_execute,
        task_add, task_update, task_delete, task_get,
    )
    from tools.grep import grep, grep_count
    from tools.read import read
    from tools.write import write, append
    from tools.edit import edit, edit_regex
    from tools.bash import bash, bash_script
    from tools.glob import glob, glob_list

    return [
        # Web 搜索和内容提取
        web_search,
        web_fetch,
        web_scrape,
        web_crawl,
        web_map,
        # arXiv 学术搜索
        arxiv_search,
        arxiv_get_by_id,
        arxiv_download_pdf,
        # 子代理
        dispatch_agent,
        list_subagents,
        # Plan/Task 管理
        plan_get,
        plan_execute,
        task_add,
        task_update,
        task_delete,
        task_get,
        # Skills
        load_skills,
        list_skills,
        skill_call,
        # 文件搜索
        grep,
        grep_count,
        # 文件操作
        read,
        write,
        append,
        edit,
        edit_regex,
        # Shell
        bash,
        bash_script,
        # 文件查找
        glob,
        glob_list,
    ]
