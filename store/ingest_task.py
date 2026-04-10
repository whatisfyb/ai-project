"""论文入库任务存储

管理论文入库任务的状态，支持：
- 任务创建和状态更新
- 中断恢复
- 进度查询
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field


# 默认数据库路径
DEFAULT_DB_PATH = Path(__file__).parent.parent / ".data" / "ingest_tasks.db"


@dataclass
class IngestTask:
    """入库任务"""
    task_id: str
    status: Literal["pending", "running", "completed", "failed", "interrupted"]
    total_papers: int = 0
    processed_papers: int = 0
    succeeded_papers: int = 0
    failed_papers: int = 0
    results: list[dict] = field(default_factory=list)  # 每篇论文的处理结果
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class IngestTaskStore:
    """入库任务存储"""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    total_papers INTEGER DEFAULT 0,
                    processed_papers INTEGER DEFAULT 0,
                    succeeded_papers INTEGER DEFAULT 0,
                    failed_papers INTEGER DEFAULT 0,
                    results TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def create_task(self, task_id: str, total_papers: int) -> IngestTask:
        """创建新任务"""
        now = datetime.now().isoformat()
        task = IngestTask(
            task_id=task_id,
            status="pending",
            total_papers=total_papers,
            created_at=now,
            updated_at=now,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO ingest_tasks
                (task_id, status, total_papers, processed_papers, succeeded_papers, failed_papers, results, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, task.status, task.total_papers, 0, 0, 0, "[]", None, now, now)
            )
            conn.commit()

        return task

    def get_task(self, task_id: str) -> Optional[IngestTask]:
        """获取任务"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT task_id, status, total_papers, processed_papers, succeeded_papers, failed_papers, results, error, created_at, updated_at
                FROM ingest_tasks WHERE task_id = ?
                """,
                (task_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            return IngestTask(
                task_id=row[0],
                status=row[1],
                total_papers=row[2],
                processed_papers=row[3],
                succeeded_papers=row[4],
                failed_papers=row[5],
                results=json.loads(row[6]) if row[6] else [],
                error=row[7],
                created_at=row[8],
                updated_at=row[9],
            )

    def update_task(self, task: IngestTask) -> None:
        """更新任务状态"""
        task.updated_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE ingest_tasks
                SET status = ?, processed_papers = ?, succeeded_papers = ?, failed_papers = ?, results = ?, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    task.status,
                    task.processed_papers,
                    task.succeeded_papers,
                    task.failed_papers,
                    json.dumps(task.results, ensure_ascii=False),
                    task.error,
                    task.updated_at,
                    task.task_id,
                )
            )
            conn.commit()

    def add_paper_result(
        self,
        task_id: str,
        task: IngestTask,
        file_path: str,
        success: bool,
        paper_id: Optional[str] = None,
        title: Optional[str] = None,
        sections: Optional[list[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """添加单篇论文的处理结果"""
        result = {
            "file_path": file_path,
            "success": success,
            "paper_id": paper_id,
            "title": title,
            "sections": sections,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

        task.results.append(result)
        task.processed_papers += 1

        if success:
            task.succeeded_papers += 1
        else:
            task.failed_papers += 1

        self.update_task(task)

    def list_tasks(
        self,
        status: Optional[Literal["pending", "running", "completed", "failed", "interrupted"]] = None,
        limit: int = 20,
    ) -> list[IngestTask]:
        """列出任务"""
        with sqlite3.connect(self.db_path) as conn:
            sql = "SELECT task_id, status, total_papers, processed_papers, succeeded_papers, failed_papers, results, error, created_at, updated_at FROM ingest_tasks"
            params = []

            if status:
                sql += " WHERE status = ?"
                params.append(status)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            tasks = []

            for row in cursor.fetchall():
                tasks.append(IngestTask(
                    task_id=row[0],
                    status=row[1],
                    total_papers=row[2],
                    processed_papers=row[3],
                    succeeded_papers=row[4],
                    failed_papers=row[5],
                    results=json.loads(row[6]) if row[6] else [],
                    error=row[7],
                    created_at=row[8],
                    updated_at=row[9],
                ))

            return tasks

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM ingest_tasks WHERE task_id = ?", (task_id,))
            conn.commit()
        return True
