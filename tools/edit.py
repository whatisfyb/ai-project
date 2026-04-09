"""Edit 工具 - 精准文件编辑

基于搜索替换的精准编辑，支持：
- 单个替换和批量替换
- 行号定位编辑
- 替换计数
"""

import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


@tool
def edit(
    file_path: str,
    old_string: str,
    new_string: str = "",
    replace_all: bool = False,
    create_backup: bool = False,
) -> dict[str, Any]:
    """Edit a file by replacing text.

    Uses search and replace to make precise edits. The old_string must match
    exactly (including whitespace and newlines).

    Args:
        file_path: Path to the file to edit (absolute or relative to project root)
        old_string: The exact text to search for and replace
        new_string: The replacement text (default: empty string to delete)
        replace_all: Replace all occurrences, not just the first one (default: False)
        create_backup: Create a backup before editing (default: False)

    Returns:
        Dictionary containing edit result and metadata
    """
    try:
        path = Path(file_path)

        # 相对路径转为绝对路径
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / path

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        if not path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}",
            }

        # 读取原内容
        with open(path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # 创建备份
        backup_path = None
        if create_backup:
            import shutil
            from datetime import datetime
            backup_dir = path.parent / ".backup"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.name}.{timestamp}.bak"
            backup_path = backup_dir / backup_name
            shutil.copy2(path, backup_path)

        # 执行替换
        if replace_all:
            new_content, count = original_content.replace(old_string, new_string), original_content.count(old_string)
        else:
            if old_string in original_content:
                new_content = original_content.replace(old_string, new_string, 1)
                count = 1
            else:
                return {
                    "success": False,
                    "error": "old_string not found in file",
                    "searched_for": old_string[:100] + ("..." if len(old_string) > 100 else ""),
                }

        # 写入修改后的内容
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        # 计算统计
        original_lines = original_content.count("\n") + 1
        new_lines = new_content.count("\n") + 1

        return {
            "success": True,
            "file_path": str(path),
            "replacements": count,
            "original_size": len(original_content.encode("utf-8")),
            "new_size": len(new_content.encode("utf-8")),
            "original_lines": original_lines,
            "new_lines": new_lines,
            "backup": str(backup_path) if backup_path else None,
        }

    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to edit file: {str(e)}",
        }


@tool
def edit_regex(
    file_path: str,
    pattern: str,
    replacement: str,
    replace_all: bool = True,
    case_sensitive: bool = True,
    create_backup: bool = False,
) -> dict[str, Any]:
    """Edit a file using regex pattern matching and replacement.

    Args:
        file_path: Path to the file to edit
        pattern: Regex pattern to search for
        replacement: Replacement string (supports backreferences like \\1, \\2)
        replace_all: Replace all matches (default: True)
        case_sensitive: Whether the search is case-sensitive (default: True)
        create_backup: Create a backup before editing (default: False)

    Returns:
        Dictionary containing edit result and metadata
    """
    try:
        path = Path(file_path)

        if not path.is_absolute():
            project_root = Path(__file__).parent.parent
            path = project_root / path

        if not path.exists() or not path.is_file():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        with open(path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # 创建备份
        backup_path = None
        if create_backup:
            import shutil
            from datetime import datetime
            backup_dir = path.parent / ".backup"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.name}.{timestamp}.bak"
            backup_path = backup_dir / backup_name
            shutil.copy2(path, backup_path)

        # 编译正则
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        # 执行替换
        if replace_all:
            new_content, count = regex.subn(replacement, original_content)
        else:
            new_content, count = regex.subn(replacement, original_content, count=1)

        if count == 0:
            return {
                "success": False,
                "error": "Pattern not found in file",
                "searched_for": pattern,
            }

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return {
            "success": True,
            "file_path": str(path),
            "replacements": count,
            "original_size": len(original_content.encode("utf-8")),
            "new_size": len(new_content.encode("utf-8")),
            "backup": str(backup_path) if backup_path else None,
        }

    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {str(e)}",
        }
    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied: {file_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to edit file: {str(e)}",
        }
