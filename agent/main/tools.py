"""工具注册与初始化"""

from agent.mcp.tools import load_mcp_tools


def get_main_agent_tools() -> list:
    """获取 Main Agent 可用的工具列表

    合并后的工具：
    - web: Web 搜索、抓取、arXiv（原 8 个）
    - agent: 子代理分发（原 2 个）
    - paper_kb: 论文知识库（原 7 个）
    - task: 任务管理（原 4 个）
    """
    # 合并后的核心工具
    from tools.web import web
    from tools.agent import agent
    from tools.paper_kb import paper_kb
    from tools.task import task

    # A2A 工具
    from agent.a2a.tools import (
        plan_dispatch, job_status, job_list, job_wait, worker_list,
    )
    # Skills 工具
    from tools.skills import load_skills, list_skills, skill_call
    # 文件操作工具
    from tools.grep import grep, grep_count
    from tools.read import read
    from tools.write import write, append
    from tools.edit import edit, edit_regex
    from tools.bash import bash, bash_script
    from tools.glob import glob, glob_list

    # MCP 工具
    mcp_tools = load_mcp_tools()

    return [
        # === 核心工具（合并后）===
        web,
        agent,
        paper_kb,
        task,

        # === A2A Worker ===
        plan_dispatch,
        job_status,
        job_list,
        job_wait,
        worker_list,

        # === Skills ===
        load_skills,
        list_skills,
        skill_call,

        # === 文件操作 ===
        grep,
        grep_count,
        read,
        write,
        append,
        edit,
        edit_regex,
        bash,
        bash_script,
        glob,
        glob_list,

        # === MCP 工具 ===
        *mcp_tools,
    ]
