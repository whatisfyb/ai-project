"""Grep 工具 - 文件内容搜索

基于 ripgrep (rg) 实现，支持：
- 正则表达式搜索
- 多文件类型过滤
- 行号和上下文显示
- 限制返回数量避免输出过长
"""

import subprocess
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


# 支持的文件类型
SUPPORTED_EXTENSIONS = {
    "py": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "tsx": "TypeScript React",
    "jsx": "JavaScript React",
    "java": "Java",
    "c": "C",
    "cpp": "C++",
    "cc": "C++",
    "h": "C Header",
    "hpp": "C++ Header",
    "cs": "C#",
    "go": "Go",
    "rs": "Rust",
    "rb": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "kt": "Kotlin",
    "scala": "Scala",
    "vue": "Vue",
    "svelte": "Svelte",
    "html": "HTML",
    "htm": "HTML",
    "css": "CSS",
    "scss": "SCSS",
    "sass": "Sass",
    "less": "Less",
    "json": "JSON",
    "yaml": "YAML",
    "yml": "YAML",
    "xml": "XML",
    "toml": "TOML",
    "ini": "INI",
    "cfg": "Config",
    "conf": "Config",
    "md": "Markdown",
    "txt": "Text",
    "log": "Log",
    "sql": "SQL",
    "sh": "Shell",
    "bash": "Bash",
    "zsh": "Zsh",
    "fish": "Fish",
    "ps1": "PowerShell",
    "bat": "Batch",
    "cmd": "Batch",
    "dockerfile": "Dockerfile",
    "makefile": "Makefile",
    "cmake": "CMake",
    "r": "R",
    "lua": "Lua",
    "perl": "Perl",
    "pl": "Perl",
    "pyw": "Python (no console)",
    "pyx": "Cython",
    "d": "D",
    "ex": "Elixir",
    "exs": "Elixir",
    "erl": "Erlang",
    "hs": "Haskell",
    "ml": "OCaml",
    "fs": "F#",
    "clj": "Clojure",
    "cljs": "ClojureScript",
    "groovy": "Groovy",
    "gradle": "Gradle",
    "tf": "Terraform",
    "tfvars": "Terraform",
    "v": "Verilog",
    "vhd": "VHDL",
    "asm": "Assembly",
    "s": "Assembly",
    "rkt": "Racket",
    "scm": "Scheme",
    "jl": "Julia",
    "nim": "Nim",
    "zig": "Zig",
    "v": "V",
    "mod": "Module (Nim)",
}

# 默认排除的目录
DEFAULT_EXCLUDE_DIRS = [
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
]


def _build_file_type_filter(include_types: list[str] | None = None) -> list[str]:
    """构建 ripgrep 文件类型参数"""
    if not include_types:
        return []

    flags = []
    for t in include_types:
        # 支持扩展名或类型名
        if t.startswith("."):
            flags.extend(["--type", t[1:]])
        else:
            flags.extend(["--type", t])
    return flags


def _build_extension_filter(extensions: list[str] | None = None) -> list[str]:
    """构建扩展名过滤参数"""
    if not extensions:
        return []

    # -g 支持 glob 模式
    patterns = []
    for ext in extensions:
        if not ext.startswith("."):
            ext = "." + ext
        patterns.append(f"*{ext}")

    # 合并为一个 glob 模式
    return ["-g", "*.{" + ",".join(ext.lstrip(".") for ext in extensions) + "}"]


def _build_exclude_filter(exclude_patterns: list[str] | None = None) -> list[str]:
    """构建排除参数"""
    flags = []
    exclude_patterns = exclude_patterns or []

    # 添加默认排除目录
    for d in DEFAULT_EXCLUDE_DIRS:
        flags.extend(["--exclude-dir", d])

    # 添加用户指定的排除模式
    for pattern in exclude_patterns:
        flags.extend(["--exclude-dir", pattern])

    # 排除常见不想搜索的文件
    common_exclude = [
        "*.min.js",
        "*.min.css",
        "*.bundle.js",
        "*.chunk.js",
        "*.map",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
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
        "*.ico",
        "*.eot",
        "*.otf",
        "*.ttf",
        "*.woff",
        "*.woff2",
    ]
    for pattern in common_exclude:
        flags.extend(["--exclude", pattern])

    return flags


def _python_grep(
    pattern: str,
    path: str = ".",
    include_types: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    case_sensitive: bool = False,
    whole_word: bool = False,
    context_lines: int = 0,
    max_results: int = 50,
    include_line_numbers: bool = True,
) -> dict[str, Any]:
    """Python 原生 grep 实现（备选，当 ripgrep 不可用时使用）"""
    import re

    search_path = Path(path)
    if not search_path.exists():
        return {"error": f"Path does not exist: {path}", "matches": []}

    # 编译正则
    flags = 0 if case_sensitive else re.IGNORECASE
    if whole_word:
        pattern = r"\b" + re.escape(pattern) + r"\b"
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"error": f"Invalid regex: {e}", "matches": []}

    # 确定要搜索的扩展名
    if include_types:
        extensions = set()
        for t in include_types:
            if t.startswith("."):
                extensions.add(t.lower())
            elif t in SUPPORTED_EXTENSIONS:
                extensions.add("." + t.lower())
            else:
                # 可能是类型名如 "python"
                for ext, name in SUPPORTED_EXTENSIONS.items():
                    if name.lower() == t.lower():
                        extensions.add("." + ext.lower())
                        break
    else:
        extensions = None  # 表示不限制扩展名

    # 排除的目录
    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    if exclude_patterns:
        exclude_set.update(exclude_patterns)

    matches = []
    try:
        for file_path in search_path.rglob("*"):
            if file_path.is_dir():
                # 检查是否应该排除
                if file_path.name in exclude_set:
                    continue
                continue

            # 检查扩展名过滤
            if extensions is not None:
                if file_path.suffix.lower() not in extensions:
                    continue

            # 检查是否在排除目录内
            parts = file_path.parts
            if any(p in exclude_set for p in parts):
                continue

            # 跳过二进制文件
            if file_path.stat().st_size > 5 * 1024 * 1024:  # > 5MB
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue

            # 搜索
            for line_no, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    match_info = {
                        "path": str(file_path),
                        "line_number": line_no,
                        "content": line.rstrip("\n"),
                    }
                    matches.append(match_info)
                    if len(matches) >= max_results:
                        break

            if len(matches) >= max_results:
                break

    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "matches": []}

    return {
        "matches": matches,
        "count": len(matches),
        "truncated": len(matches) >= max_results,
        "pattern": pattern,
        "path": str(search_path),
        "include_types": include_types,
        "engine": "python",
    }


def _ripgrep(
    pattern: str,
    path: str = ".",
    include_types: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    case_sensitive: bool = False,
    whole_word: bool = False,
    context_lines: int = 0,
    max_results: int = 50,
    include_line_numbers: bool = True,
) -> dict[str, Any]:
    """基于 ripgrep 的实现"""
    search_path = Path(path)
    if not search_path.exists():
        return {"error": f"Path does not exist: {path}", "matches": []}

    cmd = ["rg", "--json"]

    if case_sensitive:
        cmd.append("--case-sensitive")
    else:
        cmd.append("--ignore-case")

    if whole_word:
        cmd.append("--word-regexp")

    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])

    if include_line_numbers:
        cmd.append("--line-number")
    else:
        cmd.append("--no-line-number")

    cmd.extend(_build_file_type_filter(include_types))
    cmd.extend(_build_extension_filter(include_types))
    cmd.extend(_build_exclude_filter(exclude_patterns))

    cmd.extend(["--max-count", str(max_results)])
    cmd.append(pattern)
    cmd.append(str(search_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        matches = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = eval(line)
                if entry.get("type") == "match":
                    data = entry.get("data", {})
                    match_info = {
                        "path": data.get("path", {}).get("text", ""),
                        "line_number": data.get("line_number", 0),
                        "content": data.get("lines", {}).get("text", "").rstrip("\n"),
                    }
                    if context_lines > 0 and "context" in data:
                        match_info["context"] = data["context"]
                    matches.append(match_info)
            except (ValueError, SyntaxError, KeyError):
                continue

        return {
            "matches": matches,
            "count": len(matches),
            "truncated": result.stdout.count("\n") >= max_results * 2,
            "pattern": pattern,
            "path": str(search_path),
            "include_types": include_types,
            "engine": "ripgrep",
        }

    except subprocess.TimeoutExpired:
        return {"error": "Search timed out after 30 seconds", "matches": [], "engine": "ripgrep"}
    except FileNotFoundError:
        return {"error": "ripgrep not found, using Python fallback", "matches": [], "engine": "ripgrep"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "matches": [], "engine": "ripgrep"}


@tool
def grep(
    pattern: str,
    path: str = ".",
    include_types: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    case_sensitive: bool = False,
    whole_word: bool = False,
    context_lines: int = 0,
    max_results: int = 50,
    include_line_numbers: bool = True,
) -> dict[str, Any]:
    """Search for a pattern in files.

    Supports searching through code and text files. Does NOT support binary
    files like PDF, Word documents, or images.

    Args:
        pattern: Regex pattern to search for (supports regex syntax)
        path: Directory path to search in (default: current directory)
        include_types: List of file types/extensions to search.
            Examples: ["py", "js", "json", ".txt"] or ["python", "javascript"]
        exclude_patterns: List of directory names to exclude (default exclusions apply)
        case_sensitive: Whether search is case-sensitive (default: False)
        whole_word: Match whole words only (default: False)
        context_lines: Number of context lines before/after match (default: 0)
        max_results: Maximum number of results to return (default: 50)
        include_line_numbers: Whether to include line numbers (default: True)

    Returns:
        Dictionary containing search results with matches and metadata
    """
    # 优先使用 ripgrep
    result = _ripgrep(
        pattern=pattern,
        path=path,
        include_types=include_types,
        exclude_patterns=exclude_patterns,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        context_lines=context_lines,
        max_results=max_results,
        include_line_numbers=include_line_numbers,
    )

    # 如果 ripgrep 不可用，使用 Python 实现
    if result.get("engine") == "ripgrep" and "ripgrep not found" in result.get("error", ""):
        result = _python_grep(
            pattern=pattern,
            path=path,
            include_types=include_types,
            exclude_patterns=exclude_patterns,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            context_lines=context_lines,
            max_results=max_results,
            include_line_numbers=include_line_numbers,
        )

    return result


def _python_grep_count(
    pattern: str,
    path: str = ".",
    include_types: list[str] | None = None,
) -> dict[str, Any]:
    """Python 原生 grep_count 实现"""
    import re

    search_path = Path(path)
    if not search_path.exists():
        return {"error": f"Path does not exist: {path}", "counts": []}

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Invalid regex: {e}", "counts": []}

    # 确定要搜索的扩展名
    if include_types:
        extensions = set()
        for t in include_types:
            if t.startswith("."):
                extensions.add(t.lower())
            elif t in SUPPORTED_EXTENSIONS:
                extensions.add("." + t.lower())
            else:
                for ext, name in SUPPORTED_EXTENSIONS.items():
                    if name.lower() == t.lower():
                        extensions.add("." + ext.lower())
                        break
    else:
        extensions = None

    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    counts = {}
    file_count = 0

    try:
        for file_path in search_path.rglob("*"):
            if file_path.is_dir():
                if file_path.name in exclude_set:
                    continue
                continue

            if extensions is not None:
                if file_path.suffix.lower() not in extensions:
                    continue

            parts = file_path.parts
            if any(p in exclude_set for p in parts):
                continue

            if file_path.stat().st_size > 5 * 1024 * 1024:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                continue

            count = len(regex.findall(content))
            if count > 0:
                counts[str(file_path)] = count
                file_count += count

            # 限制文件数量避免太慢
            if len(counts) >= 1000:
                break

    except Exception as e:
        return {"error": f"Count failed: {str(e)}", "counts": []}

    return {
        "counts": [{"path": k, "count": v} for k, v in counts.items()],
        "total": file_count,
        "pattern": pattern,
        "path": str(search_path),
        "engine": "python",
    }


def _ripgrep_count(
    pattern: str,
    path: str = ".",
    include_types: list[str] | None = None,
) -> dict[str, Any]:
    """基于 ripgrep 的 count 实现"""
    search_path = Path(path)
    if not search_path.exists():
        return {"error": f"Path does not exist: {path}", "counts": []}

    cmd = ["rg", "--json", "--count"]

    if include_types:
        cmd.extend(_build_file_type_filter(include_types))
        cmd.extend(_build_extension_filter(include_types))

    cmd.extend(_build_exclude_filter(None))
    cmd.append(pattern)
    cmd.append(str(search_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        counts = []
        seen_paths = set()
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = eval(line)
                if entry.get("type") == "match":
                    data = entry.get("data", {})
                    path_text = data.get("path", {}).get("text", "")
                    if path_text not in seen_paths:
                        seen_paths.add(path_text)
                        counts.append({"path": path_text, "count": data.get("count", 0)})
            except (ValueError, SyntaxError, KeyError):
                continue

        return {
            "counts": counts,
            "total": sum(c["count"] for c in counts),
            "pattern": pattern,
            "path": str(search_path),
            "engine": "ripgrep",
        }

    except subprocess.TimeoutExpired:
        return {"error": "Search timed out", "counts": [], "engine": "ripgrep"}
    except FileNotFoundError:
        return {"error": "ripgrep not found", "counts": [], "engine": "ripgrep"}
    except Exception as e:
        return {"error": str(e), "counts": [], "engine": "ripgrep"}


@tool
def grep_count(
    pattern: str,
    path: str = ".",
    include_types: list[str] | None = None,
) -> dict[str, Any]:
    """Count the number of matches for a pattern in files.

    Faster than full grep when you only need the count, not the actual matches.

    Args:
        pattern: Regex pattern to search for
        path: Directory path to search in (default: current directory)
        include_types: List of file types/extensions to search

    Returns:
        Dictionary containing count per file
    """
    # 优先使用 ripgrep
    result = _ripgrep_count(pattern, path, include_types)

    if result.get("engine") == "ripgrep" and "ripgrep not found" in result.get("error", ""):
        result = _python_grep_count(pattern, path, include_types)

    return result
