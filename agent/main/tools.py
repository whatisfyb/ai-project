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
    from tools.rag import (
        paper_search, paper_list, paper_stats,
        paper_ingest, paper_ingest_status, paper_ingest_list, paper_ingest_cancel,
    )
    from tools.mcp import (
        mcp_list_servers, mcp_connect, mcp_disconnect,
        mcp_list_tools, mcp_call_tool,
    )
    from agent.a2a.tools import (
        plan_dispatch, job_status, job_list, job_wait, worker_list,
    )

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
        # 论文知识库 RAG
        paper_search,
        paper_list,
        paper_stats,
        # 论文入库
        paper_ingest,
        paper_ingest_status,
        paper_ingest_list,
        paper_ingest_cancel,
        # 子代理
        dispatch_agent,
        list_subagents,
        # Plan/Task 管理
        plan_get,
        # plan_execute,
        task_add,
        task_update,
        task_delete,
        task_get,
        # A2A Worker 分发
        plan_dispatch,
        job_status,
        job_list,
        job_wait,
        worker_list,
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
        # MCP 工具
        mcp_list_servers,
        mcp_connect,
        mcp_disconnect,
        mcp_list_tools,
        mcp_call_tool,
    ]
