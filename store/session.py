"""Session 管理 - 持久化对话历史

提供：
- 会话的创建、列表、切换
- 消息的存储和检索
- 与 LangGraph checkpointer 同步
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# 默认数据库路径
DEFAULT_DB_PATH = Path(__file__).parent.parent / ".data" / "sessions.db"


class SessionStore:
    """会话存储管理器"""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            # 会话元数据表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0
                )
            """)

            # 消息表（用于查看历史）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES session_metadata(session_id)
                )
            """)

            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp)
            """)

            conn.commit()

    # ============ 会话管理 ============

    def create_session(self, session_id: str, title: str = "") -> dict[str, Any]:
        """创建新会话"""
        now = datetime.now().isoformat()
        if not title:
            title = f"会话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO session_metadata (session_id, title, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, 0)
                """,
                (session_id, title, now, now)
            )
            conn.commit()

        return {
            "session_id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """获取会话信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT session_id, title, created_at, updated_at, message_count
                FROM session_metadata WHERE session_id = ?
                """,
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "session_id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "message_count": row[4],
                }
        return None

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """列出所有会话（按更新时间倒序）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT session_id, title, created_at, updated_at, message_count
                FROM session_metadata
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,)
            )
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "session_id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "message_count": row[4],
                })
            return sessions

    def update_session_title(self, session_id: str, title: str) -> bool:
        """更新会话标题"""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE session_metadata SET title = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (title, now, session_id)
            )
            conn.commit()
        return True

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
            conn.commit()
        return True

    # ============ 消息管理 ============

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """添加消息到会话"""
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, metadata_json)
            )
            message_id = cursor.lastrowid

            # 更新会话的消息计数和更新时间
            conn.execute(
                """
                UPDATE session_metadata
                SET message_count = message_count + 1, updated_at = ?
                WHERE session_id = ?
                """,
                (now, session_id)
            )

            conn.commit()

        return message_id

    def add_messages_batch(
        self,
        session_id: str,
        messages: list[BaseMessage],
    ) -> int:
        """批量添加消息（从 LangGraph messages 同步）"""
        now = datetime.now().isoformat()
        count = 0

        with sqlite3.connect(self.db_path) as conn:
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content or ""

                # 提取 metadata
                metadata = {}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    metadata["tool_calls"] = [
                        {"name": tc.get("name"), "args": tc.get("args")}
                        for tc in msg.tool_calls
                    ]

                metadata_json = json.dumps(metadata) if metadata else None

                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, role, content, now, metadata_json)
                )
                count += 1

            # 更新会话统计
            if count > 0:
                conn.execute(
                    """
                    UPDATE session_metadata
                    SET message_count = message_count + ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (count, now, session_id)
                )

            conn.commit()

        return count

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """获取会话的所有消息"""
        with sqlite3.connect(self.db_path) as conn:
            sql = """
                SELECT id, role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """
            if limit:
                sql += f" LIMIT {limit}"

            cursor = conn.execute(sql, (session_id,))
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                })
            return messages

    def get_last_n_messages(
        self,
        session_id: str,
        n: int = 10,
    ) -> list[dict[str, Any]]:
        """获取最近 N 条消息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, n)
            )
            messages = []
            for row in reversed(cursor.fetchall()):  # 反转顺序
                messages.append({
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                })
            return messages

    def clear_messages(self, session_id: str) -> bool:
        """清空会话的所有消息"""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                """
                UPDATE session_metadata
                SET message_count = 0, updated_at = ?
                WHERE session_id = ?
                """,
                (now, session_id)
            )
            conn.commit()
        return True

    # ============ 工具方法 ============

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return self.get_session(session_id) is not None

    def get_or_create_session(self, session_id: str, title: str = "") -> dict[str, Any]:
        """获取或创建会话"""
        session = self.get_session(session_id)
        if session:
            return session
        return self.create_session(session_id, title)

    def format_messages_for_display(
        self,
        messages: list[dict[str, Any]],
        include_metadata: bool = False,
    ) -> str:
        """格式化消息用于显示"""
        lines = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            timestamp = msg.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M")
                except:
                    time_str = ""
            else:
                time_str = ""

            content = msg["content"] or ""
            # 截断过长的内容
            if len(content) > 200:
                content = content[:200] + "..."

            line = f"[{time_str}] {role}: {content}"
            if include_metadata and msg.get("metadata"):
                metadata = msg["metadata"]
                if "tool_calls" in metadata:
                    tools = [tc["name"] for tc in metadata["tool_calls"] if tc.get("name")]
                    if tools:
                        line += f" [工具: {', '.join(tools)}]"

            lines.append(line)

        return "\n".join(lines)
