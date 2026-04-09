"""Bash 工具 - Shell 命令执行

支持：
- 跨平台（Windows PowerShell/cmd, Unix bash/zsh）
- 超时控制
- 工作目录设置
"""

import os
import subprocess
import shlex
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


# Windows 使用的 shell
WINDOWS_SHELL = "powershell"

# 默认超时（秒）
DEFAULT_TIMEOUT = 30


def _get_shell() -> str:
    """获取默认 shell"""
    import platform
    if platform.system() == "Windows":
        return WINDOWS_SHELL
    return "/bin/bash"


def _get_cwd() -> Path:
    """获取默认工作目录（项目根目录）"""
    return Path(__file__).parent.parent


def _bash_impl(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: str | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bash 命令执行的内部实现（供 bash 和 bash_script 共用）"""
    try:
        # 确定工作目录
        if cwd:
            work_dir = Path(cwd)
        else:
            work_dir = _get_cwd()

        # 构建环境变量
        env = os.environ.copy()
        if environment:
            env.update(environment)

        # Windows 使用 shell=True
        import platform
        is_windows = platform.system() == "Windows"

        if is_windows:
            # Windows: 使用 PowerShell
            if WINDOWS_SHELL == "powershell":
                cmd_to_run = ["powershell", "-NoProfile", "-Command", command]
            else:
                cmd_to_run = ["cmd", "/c", command]
            shell_used = WINDOWS_SHELL
        else:
            # Unix: 使用 bash -c
            cmd_to_run = ["/bin/bash", "-c", command]
            shell_used = "bash"

        # 执行命令
        result = subprocess.run(
            cmd_to_run,
            capture_output=True,
            text=True,
            cwd=str(work_dir),
            env=env,
            timeout=timeout,
            shell=False,  # 已在命令中指定 shell
        )

        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "shell": shell_used,
            "cwd": str(work_dir),
            "duration": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "success": False,
            "timeout": True,
            "duration": timeout,
        }
    except FileNotFoundError as e:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command not found: {str(e)}",
            "success": False,
            "error": "Command not found",
        }
    except Exception as e:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "error": str(e),
        }


@tool
def bash(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: str | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a shell command and return its output.

    Supports bash, zsh, PowerShell, and cmd on Windows.

    Args:
        command: The shell command to execute
        timeout: Maximum execution time in seconds (default: 30)
        cwd: Working directory for the command (default: project root)
        environment: Additional environment variables to set

    Returns:
        Dictionary containing command output and metadata
    """
    return _bash_impl(command=command, timeout=timeout, cwd=cwd, environment=environment)


@tool
def bash_script(
    script: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Execute a multi-line shell script.

    Args:
        script: Multi-line shell script to execute
        timeout: Maximum execution time in seconds (default: 30)
        cwd: Working directory for the script (default: project root)

    Returns:
        Dictionary containing script output and metadata
    """
    # 将多行脚本合并为单行命令，用分号分隔
    lines = script.strip().split("\n")
    # 过滤空行和注释
    commands = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    command = "; ".join(commands)
    return _bash_impl(command=command, timeout=timeout, cwd=cwd)
