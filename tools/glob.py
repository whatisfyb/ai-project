"""Glob 工具 - 文件模式匹配

支持：
- 标准 glob 模式（**/*.py, *.txt 等）
- 递归搜索
- 文件类型过滤
- 排序（按修改时间）
"""

import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


# 默认排除的目录
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    "coverage",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".DS_Store",
}

# 默认排除的文件模式
DEFAULT_EXCLUDE_FILES = {
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.bin",
    "*.dat",
    "*.ico",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.bmp",
    "*.svg",
    "*.eot",
    "*.otf",
    "*.ttf",
    "*.woff",
    "*.woff2",
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.chunk.js",
    "*.map",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}


def _is_excluded(path: Path, exclude_dirs: set, exclude_files: set) -> bool:
    """检查路径是否应该被排除"""
    # 检查目录
    for part in path.parts:
        if part in exclude_dirs:
            return True

    # 检查文件名
    for pattern in exclude_files:
        if path.match(pattern):
            return True

    return False


def _glob_impl(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    max_results: int = 100,
    include_files_only: bool = True,
    sort_by: str = "name",
) -> dict[str, Any]:
    """Glob 模式匹配的内部实现"""
    try:
        base_path = Path(path)

        # 相对路径转为绝对路径
        if not base_path.is_absolute():
            project_root = Path(__file__).parent.parent
            base_path = project_root / base_path

        if not base_path.exists():
            return {
                "error": f"Path does not exist: {path}",
                "files": [],
            }

        # 使用 pathlib 的 glob
        if recursive:
            matches = base_path.glob(pattern)
        else:
            matches = base_path.glob(pattern)

        # 过滤和收集结果
        files = []
        for match in matches:
            if include_files_only and match.is_dir():
                continue
            if _is_excluded(match, DEFAULT_EXCLUDE_DIRS, DEFAULT_EXCLUDE_FILES):
                continue
            files.append(match)
            if len(files) >= max_results:
                break

        # 排序
        if sort_by == "modified":
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        elif sort_by == "size":
            files.sort(key=lambda p: p.stat().st_size, reverse=True)
        else:  # name
            files.sort(key=lambda p: str(p).lower())

        # 构建结果
        result_files = []
        for f in files:
            stat = f.stat()
            result_files.append({
                "path": str(f),
                "name": f.name,
                "relative_path": str(f.relative_to(base_path)) if f.is_relative_to(base_path) else str(f),
                "is_dir": f.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })

        return {
            "files": result_files,
            "count": len(result_files),
            "pattern": pattern,
            "path": str(base_path),
            "recursive": recursive,
            "truncated": len(files) >= max_results,
        }

    except Exception as e:
        return {
            "error": f"Glob failed: {str(e)}",
            "files": [],
        }


@tool
def glob(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    max_results: int = 100,
    include_files_only: bool = True,
    sort_by: str = "name",
) -> dict[str, Any]:
    """Find files matching a glob pattern.

    Supports standard glob patterns like **/*.py, *.txt, src/**/*.js

    Args:
        pattern: Glob pattern to match (e.g., "**/*.py", "*.txt", "src/**/*.js")
        path: Base directory to search in (default: current directory)
        recursive: Whether to search recursively (default: True)
        max_results: Maximum number of results to return (default: 100)
        include_files_only: Only return files, not directories (default: True)
        sort_by: Sort results by "name", "modified", or "size" (default: "name")

    Returns:
        Dictionary containing matched files and metadata
    """
    return _glob_impl(
        pattern=pattern,
        path=path,
        recursive=recursive,
        max_results=max_results,
        include_files_only=include_files_only,
        sort_by=sort_by,
    )


@tool
def glob_list(
    patterns: list[str],
    path: str = ".",
    recursive: bool = True,
    max_results: int = 100,
) -> dict[str, Any]:
    """Find files matching multiple glob patterns.

    Equivalent to running glob for each pattern and combining results.

    Args:
        patterns: List of glob patterns to match
        path: Base directory to search in (default: current directory)
        recursive: Whether to search recursively (default: True)
        max_results: Maximum number of results to return (default: 100)

    Returns:
        Dictionary containing matched files grouped by pattern
    """
    results = {}
    all_files = {}

    for pattern in patterns:
        result = _glob_impl(
            pattern=pattern,
            path=path,
            recursive=recursive,
            max_results=max_results,
        )
        results[pattern] = result.get("files", [])
        for f in result.get("files", []):
            all_files[f["path"]] = f

        if len(all_files) >= max_results:
            break

    # 去重并转换为列表
    unique_files = list(all_files.values())

    return {
        "files": unique_files,
        "count": len(unique_files),
        "patterns": patterns,
        "path": str(path),
        "truncated": len(unique_files) >= max_results,
        "by_pattern": {p: len(r) for p, r in results.items()},
    }
