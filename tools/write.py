"""Write 工具 - 文件写入

支持：
- 创建新文件
- 覆盖已有文件（可配置备份）
- 自动创建父目录
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


def _ensure_dir(path: Path) -> None:
    """确保目录存在"""
    path.parent.mkdir(parents=True, exist_ok=True)


def _create_backup(path: Path) -> str | None:
    """创建备份文件"""
    if not path.exists():
        return None

    backup_dir = path.parent / ".backup"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.name}.{timestamp}.bak"
    backup_path = backup_dir / backup_name

    shutil.copy2(path, backup_path)
    return str(backup_path)


@tool
def write(
    file_path: str,
    content: str,
    create_backup: bool = False,
    ensure_dir: bool = True,
) -> dict[str, Any]:
    """Write content to a file, creating it if it doesn't exist.

    Args:
        file_path: Path to the file to write (absolute or relative to project root)
        content: Content to write to the file
        create_backup: Whether to create a backup before overwriting (default: False)
        ensure_dir: Whether to automatically create parent directories (default: True)

    Returns:
        Dictionary containing operation result and metadata
    """
    try:
        path = Path(file_path)

        # 相对路径转为绝对路径
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / path

        # 检查是否是目录
        if path.exists() and path.is_dir():
            return {
                "error": f"Path is a directory, not a file: {file_path}",
                "success": False,
            }

        # 创建备份
        backup_path = None
        if create_backup and path.exists():
            backup_path = _create_backup(path)

        # 确保父目录存在
        if ensure_dir:
            _ensure_dir(path)

        # 写入文件
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        # 获取文件信息
        file_size = path.stat().st_size
        line_count = content.count("\n") + 1 if content else 0

        result = {
            "success": True,
            "file_path": str(path),
            "file_name": path.name,
            "file_size": file_size,
            "line_count": line_count,
            "bytes_written": len(content.encode("utf-8")),
        }

        if backup_path:
            result["backup"] = backup_path

        return result

    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to write file: {str(e)}",
        }


@tool
def append(
    file_path: str,
    content: str,
    create_if_not_exists: bool = True,
) -> dict[str, Any]:
    """Append content to the end of a file.

    Args:
        file_path: Path to the file to append to
        content: Content to append to the file
        create_if_not_exists: Create the file if it doesn't exist (default: True)

    Returns:
        Dictionary containing operation result and metadata
    """
    try:
        path = Path(file_path)

        # 相对路径转为绝对路径
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / path

        # 检查文件是否存在
        if not path.exists():
            if not create_if_not_exists:
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}",
                }
            # 创建新文件
            path.parent.mkdir(parents=True, exist_ok=True)

        # 追加内容
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)

        file_size = path.stat().st_size
        bytes_appended = len(content.encode("utf-8"))

        return {
            "success": True,
            "file_path": str(path),
            "file_size": file_size,
            "bytes_appended": bytes_appended,
        }

    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to append to file: {str(e)}",
        }
