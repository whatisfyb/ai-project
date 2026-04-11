"""长期记忆持久化 - 基于 Markdown 文件的记忆系统

参考 Claude Code 的记忆系统设计：
- 记忆类型: user, feedback, project, reference
- 每个记忆是独立的 .md 文件，使用 YAML frontmatter
- MEMORY.md 作为索引文件

存储路径: .data/memory/
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import re

from pydantic import BaseModel, Field


# ============ 数据模型 ============

MemoryType = Literal["user", "feedback", "project", "reference"]

MEMORY_TYPES: list[MemoryType] = ["user", "feedback", "project", "reference"]


class Memory(BaseModel):
    """记忆模型"""
    name: str = Field(description="记忆名称，用于文件名")
    description: str = Field(description="一行描述，用于判断相关性")
    type: MemoryType = Field(description="记忆类型")
    content: str = Field(description="记忆内容")
    file_path: str | None = Field(default=None, description="文件路径")
    mtime_ms: int | None = Field(default=None, description="最后修改时间戳（毫秒）")


class MemoryHeader(BaseModel):
    """记忆头部信息（用于列表和索引）"""
    filename: str = Field(description="文件名")
    file_path: str = Field(description="文件完整路径")
    mtime_ms: int = Field(description="最后修改时间戳（毫秒）")
    description: str | None = Field(default=None, description="描述")
    type: MemoryType | None = Field(default=None, description="类型")


# ============ 常量 ============

DEFAULT_MEMORY_DIR = Path(__file__).parent.parent / ".data" / "memory"
ENTRYPOINT_NAME = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000
FRONTMATTER_MAX_LINES = 30


# ============ 工具函数 ============

def parse_memory_type(raw: str | None) -> MemoryType | None:
    """解析记忆类型"""
    if raw is None or raw not in MEMORY_TYPES:
        return None
    return raw


def memory_age_days(mtime_ms: int) -> int:
    """计算记忆年龄（天数）"""
    import time
    return max(0, int((time.time() * 1000 - mtime_ms) / 86_400_000))


def memory_age(mtime_ms: int) -> str:
    """人类可读的记忆年龄"""
    d = memory_age_days(mtime_ms)
    if d == 0:
        return "today"
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


def memory_freshness_text(mtime_ms: int) -> str:
    """记忆新鲜度提示（超过1天的显示警告）"""
    d = memory_age_days(mtime_ms)
    if d <= 1:
        return ""
    return (
        f"This memory is {d} days old. "
        "Memories are point-in-time observations, not live state — "
        "claims about code behavior or file:line citations may be outdated. "
        "Verify against current code before asserting as fact."
    )


# ============ Frontmatter 解析 ============

def parse_frontmatter(content: str) -> dict[str, Any]:
    """解析 YAML frontmatter"""
    if not content.startswith("---"):
        return {}

    # 查找结束标记
    end_match = re.search(r'\n---\s*\n', content[3:])
    if not end_match:
        return {}

    frontmatter_str = content[3:end_match.end() + 2].strip()
    frontmatter = {}

    # 简单的 YAML 解析（不支持嵌套）
    for line in frontmatter_str.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # 移除引号
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            frontmatter[key] = value

    return frontmatter


def build_frontmatter(name: str, description: str, type: MemoryType) -> str:
    """构建 YAML frontmatter"""
    return f"""---
name: {name}
description: {description}
type: {type}
---

"""


# ============ 持久化类 ============

class LongTermMemoryStore:
    """长期记忆存储管理器"""

    def __init__(self, memory_dir: Path | str | None = None):
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.entrypoint_path = self.memory_dir / ENTRYPOINT_NAME

    # ============ CRUD 操作 ============

    def create(self, memory: Memory) -> str:
        """创建新记忆

        Args:
            memory: 记忆对象

        Returns:
            创建的文件路径
        """
        # 生成文件名
        filename = self._sanitize_filename(memory.name)
        file_path = self.memory_dir / filename

        # 构建文件内容
        frontmatter = build_frontmatter(
            name=memory.name,
            description=memory.description,
            type=memory.type
        )
        full_content = frontmatter + memory.content

        # 写入文件
        file_path.write_text(full_content, encoding="utf-8")

        # 更新索引
        self._update_entrypoint()

        return str(file_path)

    def read(self, name: str) -> Memory | None:
        """读取记忆

        Args:
            name: 记忆名称

        Returns:
            记忆对象，不存在返回 None
        """
        filename = self._sanitize_filename(name)
        file_path = self.memory_dir / filename

        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8")
        mtime_ms = int(file_path.stat().st_mtime * 1000)

        # 解析 frontmatter
        frontmatter = parse_frontmatter(content)

        # 提取正文（移除 frontmatter）
        body = self._extract_body(content)

        return Memory(
            name=frontmatter.get("name", name),
            description=frontmatter.get("description", ""),
            type=parse_memory_type(frontmatter.get("type")) or "user",
            content=body,
            file_path=str(file_path),
            mtime_ms=mtime_ms,
        )

    def update(self, name: str, memory: Memory) -> bool:
        """更新记忆

        Args:
            name: 原记忆名称
            memory: 新的记忆内容

        Returns:
            是否成功
        """
        filename = self._sanitize_filename(name)
        file_path = self.memory_dir / filename

        if not file_path.exists():
            return False

        # 如果名称改变，需要重命名文件
        new_filename = self._sanitize_filename(memory.name)
        if new_filename != filename:
            new_file_path = self.memory_dir / new_filename
            file_path.rename(new_file_path)
            file_path = new_file_path

        # 构建文件内容
        frontmatter = build_frontmatter(
            name=memory.name,
            description=memory.description,
            type=memory.type
        )
        full_content = frontmatter + memory.content

        # 写入文件
        file_path.write_text(full_content, encoding="utf-8")

        # 更新索引
        self._update_entrypoint()

        return True

    def delete(self, name: str) -> bool:
        """删除记忆

        Args:
            name: 记忆名称

        Returns:
            是否成功
        """
        filename = self._sanitize_filename(name)
        file_path = self.memory_dir / filename

        if not file_path.exists():
            return False

        file_path.unlink()

        # 更新索引
        self._update_entrypoint()

        return True

    def list(self, type: MemoryType | None = None) -> list[MemoryHeader]:
        """列出所有记忆

        Args:
            type: 按类型过滤（可选）

        Returns:
            记忆头列表，按修改时间倒序
        """
        headers = []

        for file_path in self.memory_dir.glob("*.md"):
            # 跳过 MEMORY.md
            if file_path.name == ENTRYPOINT_NAME:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                mtime_ms = int(file_path.stat().st_mtime * 1000)

                # 解析 frontmatter
                frontmatter = parse_frontmatter(content)

                header = MemoryHeader(
                    filename=file_path.name,
                    file_path=str(file_path),
                    mtime_ms=mtime_ms,
                    description=frontmatter.get("description"),
                    type=parse_memory_type(frontmatter.get("type")),
                )

                # 类型过滤
                if type is None or header.type == type:
                    headers.append(header)

            except Exception:
                # 跳过解析失败的文件
                continue

        # 按修改时间倒序
        headers.sort(key=lambda h: h.mtime_ms, reverse=True)

        return headers

    def search(self, query: str, type: MemoryType | None = None) -> list[Memory]:
        """搜索记忆内容

        Args:
            query: 搜索关键词
            type: 按类型过滤（可选）

        Returns:
            匹配的记忆列表
        """
        results = []
        query_lower = query.lower()

        for header in self.list(type=type):
            memory = self.read(header.filename.replace(".md", ""))
            if memory:
                # 在名称、描述、内容中搜索
                if (query_lower in memory.name.lower() or
                    query_lower in memory.description.lower() or
                    query_lower in memory.content.lower()):
                    results.append(memory)

        return results

    # ============ 索引管理 ============

    def _update_entrypoint(self) -> None:
        """更新 MEMORY.md 索引文件"""
        headers = self.list()

        lines = []
        for header in headers:
            # 格式: - [Title](file.md) — one-line hook
            type_tag = f"[{header.type}] " if header.type else ""
            ts = datetime.fromtimestamp(header.mtime_ms / 1000).isoformat()
            desc = header.description or ""
            line = f"- {type_tag}[{header.filename.replace('.md', '')}]({header.filename}) ({ts}): {desc}"
            lines.append(line)

        # 截断检查
        if len(lines) > MAX_ENTRYPOINT_LINES:
            lines = lines[:MAX_ENTRYPOINT_LINES]

        content = "\n".join(lines)

        # 字节限制
        if len(content.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
            content = content[:MAX_ENTRYPOINT_BYTES]
            # 确保在行边界截断
            last_newline = content.rfind("\n")
            if last_newline > 0:
                content = content[:last_newline]

        self.entrypoint_path.write_text(content, encoding="utf-8")

    def get_entrypoint_content(self) -> str:
        """获取 MEMORY.md 索引内容"""
        if not self.entrypoint_path.exists():
            return ""

        content = self.entrypoint_path.read_text(encoding="utf-8")

        # 截断检查
        lines = content.split("\n")
        if len(lines) > MAX_ENTRYPOINT_LINES:
            lines = lines[:MAX_ENTRYPOINT_LINES]
            content = "\n".join(lines)

        if len(content.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
            content = content[:MAX_ENTRYPOINT_BYTES]
            last_newline = content.rfind("\n")
            if last_newline > 0:
                content = content[:last_newline]

        return content

    def get_memory_manifest(self) -> str:
        """获取记忆清单（用于 prompt）"""
        headers = self.list()
        lines = []

        for h in headers:
            type_tag = f"[{h.type}] " if h.type else ""
            ts = datetime.fromtimestamp(h.mtime_ms / 1000).isoformat()
            if h.description:
                lines.append(f"- {type_tag}{h.filename} ({ts}): {h.description}")
            else:
                lines.append(f"- {type_tag}{h.filename} ({ts})")

        return "\n".join(lines)

    # ============ 工具方法 ============

    def _sanitize_filename(self, name: str) -> str:
        """清理文件名"""
        # 替换特殊字符
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", name)
        # 移除多余空格
        safe_name = re.sub(r'\s+', "_", safe_name.strip())
        # 添加扩展名
        if not safe_name.endswith(".md"):
            safe_name += ".md"
        return safe_name

    def _extract_body(self, content: str) -> str:
        """提取正文（移除 frontmatter）"""
        if not content.startswith("---"):
            return content.strip()

        # 查找结束标记
        match = re.search(r'\n---\s*\n', content[3:])
        if not match:
            return content.strip()

        return content[3 + match.end():].strip()

    def memory_exists(self, name: str) -> bool:
        """检查记忆是否存在"""
        filename = self._sanitize_filename(name)
        return (self.memory_dir / filename).exists()

    def get_memory_count(self) -> int:
        """获取记忆数量"""
        return len(self.list())

    def clear_all(self) -> int:
        """清空所有记忆

        Returns:
            删除的记忆数量
        """
        count = 0
        for file_path in self.memory_dir.glob("*.md"):
            file_path.unlink()
            count += 1
        return count


# ============ 便捷函数 ============

_memory_store: LongTermMemoryStore | None = None


def get_memory_store() -> LongTermMemoryStore:
    """获取全局记忆存储实例"""
    global _memory_store
    if _memory_store is None:
        _memory_store = LongTermMemoryStore()
    return _memory_store


def create_memory(name: str, description: str, type: MemoryType, content: str) -> str:
    """创建记忆（便捷函数）"""
    store = get_memory_store()
    memory = Memory(
        name=name,
        description=description,
        type=type,
        content=content,
    )
    return store.create(memory)


def read_memory(name: str) -> Memory | None:
    """读取记忆（便捷函数）"""
    return get_memory_store().read(name)


def update_memory(name: str, memory: Memory) -> bool:
    """更新记忆（便捷函数）"""
    return get_memory_store().update(name, memory)


def delete_memory(name: str) -> bool:
    """删除记忆（便捷函数）"""
    return get_memory_store().delete(name)


def list_memories(type: MemoryType | None = None) -> list[MemoryHeader]:
    """列出记忆（便捷函数）"""
    return get_memory_store().list(type=type)


def search_memories(query: str, type: MemoryType | None = None) -> list[Memory]:
    """搜索记忆（便捷函数）"""
    return get_memory_store().search(query, type=type)
