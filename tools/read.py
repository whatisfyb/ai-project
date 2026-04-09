"""Read 工具 - 文件读取

支持：
- 绝对路径和相对路径
- 行偏移和行数限制
- 大文件自动截断
- 二进制文件检测
"""

import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


# 大文件阈值（超过此大小只读取前 N 行）
LARGE_FILE_THRESHOLD = 1024 * 1024  # 1MB
LARGE_FILE_PREVIEW_LINES = 500


@tool
def read(
    file_path: str,
    offset: int = 0,
    limit: int | None = None,
    show_lines: bool = True,
) -> dict[str, Any]:
    """Read the contents of a file.

    Supports absolute and relative paths, line offsets, and limits.
    Large files (>1MB) are automatically truncated with a preview.

    Args:
        file_path: Path to the file to read (absolute or relative to project root)
        offset: Line number to start reading from (0-indexed, default: 0)
        limit: Maximum number of lines to read (default: None = all lines)
        show_lines: Whether to include line numbers in output (default: True)

    Returns:
        Dictionary containing file contents and metadata
    """
    try:
        path = Path(file_path)

        # 相对路径转为绝对路径
        if not path.is_absolute():
            # 假设相对于项目根目录
            project_root = Path(__file__).parent.parent
            path = project_root / path

        # 检查路径是否存在
        if not path.exists():
            return {
                "error": f"File not found: {file_path}",
                "content": None,
                "exists": False,
            }

        # 检查是否是文件
        if not path.is_file():
            if path.is_dir():
                return {
                    "error": f"Path is a directory, not a file: {file_path}",
                    "content": None,
                    "is_directory": True,
                }
            return {
                "error": f"Path is not a file: {file_path}",
                "content": None,
            }

        # 获取文件大小
        file_size = path.stat().st_size

        # 检测是否是大文件
        is_large = file_size > LARGE_FILE_THRESHOLD
        was_truncated = False

        if is_large:
            # 大文件只读取预览
            content_parts = []
            line_count = 0
            preview_limit = LARGE_FILE_PREVIEW_LINES + offset
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line_count < offset:
                        line_count += 1
                        continue
                    if line_count >= preview_limit:
                        was_truncated = True
                        break
                    content_parts.append(line.rstrip("\n"))
                    line_count += 1
            content = "\n".join(content_parts)
        else:
            # 普通文件直接读取
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if offset > 0:
                    # 跳过前面的行
                    for _ in range(offset):
                        f.readline()
                if limit is not None:
                    content_parts = []
                    for i in range(limit):
                        line = f.readline()
                        if not line:
                            break
                        content_parts.append(line.rstrip("\n"))
                    content = "\n".join(content_parts)
                else:
                    content = f.read().rstrip("\n")

        # 构建带行号的内容
        if show_lines and content:
            lines = content.split("\n")
            max_line_num = offset + len(lines)
            num_width = len(str(max_line_num))
            numbered_lines = [
                f"{str(offset + i + 1).rjust(num_width)} │ {line}"
                for i, line in enumerate(lines)
            ]
            display_content = "\n".join(numbered_lines)
        else:
            display_content = content

        result = {
            "content": content,
            "display_content": display_content,
            "file_path": str(path),
            "file_name": path.name,
            "file_size": file_size,
            "line_count": len(content.split("\n")) if content else 0,
            "offset": offset,
            "limit": limit,
            "is_large": is_large,
            "was_truncated": was_truncated,
            "exists": True,
        }

        if was_truncated:
            result["warning"] = (
                f"File is large ({file_size} bytes). "
                f"Showing lines {offset + 1}-{offset + LARGE_FILE_PREVIEW_LINES}. "
                f"Use offset={offset + LARGE_FILE_PREVIEW_LINES} to read more."
            )

        return result

    except PermissionError:
        return {
            "error": f"Permission denied: {file_path}",
            "content": None,
            "exists": False,
        }
    except UnicodeDecodeError:
        return {
            "error": f"File is binary, not text: {file_path}",
            "content": None,
            "is_binary": True,
            "exists": True,
        }
    except Exception as e:
        return {
            "error": f"Failed to read file: {str(e)}",
            "content": None,
            "exists": False,
        }
