"""轻量 SQLite 状态仓库。

GeoAgent 第一版不引入 ORM：状态表很少、字段主要是结构化 JSON，直接使用
sqlite3 便于在 CLI、FastAPI 和并行 SubAgent 中共享同一份事实记录。每次操作
使用独立连接，SQLite 的 WAL 模式负责读写并发；GIS 计算本身不在数据库事务里。
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from app.core.models import (
    ApprovalRequest,
    ApprovalStatus,
    Artifact,
    Checkpoint,
    Conversation,
    ConversationMemory,
    ConversationMemoryEntry,
    Dataset,
    MemoryItem,
    Message,
    Run,
    SubTask,
    Task,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
    ToolStatus,
    TraceEvent,
    User,
    UserProfile,
    UserSession,
    WorkingMemory,
    utc_now,
)

T = TypeVar("T")

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    user_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_memories (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    run_id TEXT,
    dataset_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    goal TEXT,
    status TEXT,
    result TEXT,
    created_at TEXT,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS working_memories (
    task_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subtasks (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    task_id TEXT,
    parent_run_id TEXT,
    agent_id TEXT,
    status TEXT,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    turn_count INTEGER,
    tool_call_count INTEGER,
    metadata_json TEXT,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    status TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT,
    created_by_run_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_lineage (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    operation TEXT NOT NULL,
    input_dataset_ids_json TEXT NOT NULL,
    output_dataset_id TEXT NOT NULL,
    tool_call_id TEXT,
    parameters_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT,
    run_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT,
    scope TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_user_id, scope, memory_key)
);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    email TEXT UNIQUE,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT
);
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS planning_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trace_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at);
CREATE INDEX IF NOT EXISTS idx_working_memories_conversation ON working_memories(conversation_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_lineage_output ON dataset_lineage(output_dataset_id);
CREATE INDEX IF NOT EXISTS idx_planning_sessions_user ON planning_sessions(user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_planning_sessions_conversation ON planning_sessions(conversation_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_approvals_user ON approvals(user_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(source_run_id, status);
"""


class StateStore:
    """保存 GeoAgent 的可恢复事实和派生索引。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            self._migrate_schema(db)
            self._ensure_message_fts(db)
            db.commit()

    @staticmethod
    def _ensure_message_fts(db: sqlite3.Connection) -> None:
        try:
            db.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    message_id UNINDEXED, conversation_id UNINDEXED, role UNINDEXED, content,
                    tokenize='trigram'
                )"""
            )
            db.execute(
                """INSERT INTO messages_fts(message_id,conversation_id,role,content)
                SELECT m.id,m.conversation_id,m.role,m.content FROM messages m
                WHERE NOT EXISTS (SELECT 1 FROM messages_fts f WHERE f.message_id=m.id)"""
            )
        except sqlite3.OperationalError as exc:
            if "fts5" not in str(exc).casefold() and "trigram" not in str(exc).casefold():
                raise

    @staticmethod
    def _migrate_schema(db: sqlite3.Connection) -> None:
        """为已有本地 SQLite 增加字段，不删除旧业务数据。"""

        conversation_columns = {row[1] for row in db.execute("PRAGMA table_info(conversations)").fetchall()}
        if "user_id" not in conversation_columns:
            db.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
        message_columns = {row[1] for row in db.execute("PRAGMA table_info(messages)").fetchall()}
        if "dataset_ids_json" not in message_columns:
            db.execute("ALTER TABLE messages ADD COLUMN dataset_ids_json TEXT NOT NULL DEFAULT '[]'")
        StateStore._add_columns(
            db,
            "tasks",
            {
                "conversation_id": "TEXT",
                "goal": "TEXT",
                "status": "TEXT",
                "result": "TEXT",
                "created_at": "TEXT",
            },
        )
        StateStore._add_columns(
            db,
            "runs",
            {
                "conversation_id": "TEXT",
                "task_id": "TEXT",
                "parent_run_id": "TEXT",
                "agent_id": "TEXT",
                "status": "TEXT",
                "started_at": "TEXT",
                "finished_at": "TEXT",
                "error": "TEXT",
                "turn_count": "INTEGER",
                "tool_call_count": "INTEGER",
                "metadata_json": "TEXT",
            },
        )
        StateStore._add_columns(db, "tool_calls", {"status": "TEXT", "updated_at": "TEXT"})
        StateStore._add_columns(db, "datasets", {"owner_user_id": "TEXT", "created_by_run_id": "TEXT"})
        StateStore._add_columns(db, "artifacts", {"owner_user_id": "TEXT", "run_id": "TEXT"})
        memory_columns = {row[1] for row in db.execute("PRAGMA table_info(memories)").fetchall()}
        if "owner_user_id" not in memory_columns:
            db.execute("ALTER TABLE memories RENAME TO memories_legacy")
            db.execute(
                """CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT,
                scope TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_user_id, scope, memory_key)
                )"""
            )
            db.execute(
                """INSERT INTO memories(id,owner_user_id,scope,memory_key,payload_json,updated_at)
                SELECT id,NULL,scope,memory_key,payload_json,updated_at FROM memories_legacy"""
            )
            db.execute("DROP TABLE memories_legacy")
        StateStore._backfill_core_columns(db)
        db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_user_id, scope, updated_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_runs_conversation_updated ON runs(conversation_id, updated_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id, updated_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_conversation_updated ON tasks(conversation_id, updated_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_datasets_owner ON datasets(owner_user_id, created_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_owner_run ON artifacts(owner_user_id, run_id, created_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_run_status ON tool_calls(run_id, status, updated_at)")
        for row in db.execute("SELECT id, payload_json FROM runs").fetchall():
            payload = json.loads(row[1])
            if "replan_count" not in payload:
                continue
            payload.pop("replan_count", None)
            db.execute(
                "UPDATE runs SET payload_json=? WHERE id=?",
                (json.dumps(payload, ensure_ascii=False, separators=(",", ":")), row[0]),
            )

    @staticmethod
    def _add_columns(db: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in columns.items():
            if name not in existing:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @staticmethod
    def _backfill_core_columns(db: sqlite3.Connection) -> None:
        for row in db.execute("SELECT id, payload_json FROM tasks").fetchall():
            payload = json.loads(row[1])
            db.execute(
                """UPDATE tasks SET conversation_id=?, goal=?, status=?, result=?, created_at=?
                WHERE id=? AND (conversation_id IS NULL OR goal IS NULL OR status IS NULL OR created_at IS NULL)""",
                (payload.get("conversation_id"), payload.get("goal"), payload.get("status", "PENDING"), payload.get("result"), payload.get("created_at"), row[0]),
            )
        for row in db.execute("SELECT id, payload_json FROM runs").fetchall():
            payload = json.loads(row[1])
            db.execute(
                """UPDATE runs SET conversation_id=?, task_id=?, parent_run_id=?, agent_id=?, status=?,
                started_at=?, finished_at=?, error=?, turn_count=?, tool_call_count=?, metadata_json=?
                WHERE id=? AND (conversation_id IS NULL OR agent_id IS NULL OR status IS NULL OR metadata_json IS NULL)""",
                (
                    payload.get("conversation_id"),
                    payload.get("task_id"),
                    payload.get("parent_run_id"),
                    payload.get("agent_id", "main"),
                    payload.get("status", "CREATED"),
                    payload.get("started_at"),
                    payload.get("finished_at"),
                    payload.get("error"),
                    payload.get("turn_count", 0),
                    payload.get("tool_call_count", 0),
                    json.dumps(payload.get("metadata") or {}, ensure_ascii=False, separators=(",", ":")),
                    row[0],
                ),
            )
        for row in db.execute("SELECT id, payload_json FROM datasets").fetchall():
            payload = json.loads(row[1])
            db.execute(
                "UPDATE datasets SET owner_user_id=?, created_by_run_id=? WHERE id=? AND (owner_user_id IS NULL AND created_by_run_id IS NULL)",
                (payload.get("owner_user_id"), payload.get("created_by_run_id"), row[0]),
            )
        for row in db.execute("SELECT id, payload_json FROM artifacts").fetchall():
            payload = json.loads(row[1])
            db.execute(
                "UPDATE artifacts SET owner_user_id=?, run_id=? WHERE id=? AND (owner_user_id IS NULL AND run_id IS NULL)",
                (payload.get("owner_user_id"), payload.get("run_id"), row[0]),
            )
        for row in db.execute("SELECT id, result_json FROM tool_calls WHERE status IS NULL").fetchall():
            status = ToolExecutionStatus.COMPLETED.value if row[1] else ToolExecutionStatus.PENDING.value
            db.execute("UPDATE tool_calls SET status=?, updated_at=created_at WHERE id=?", (status, row[0]))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database_path, timeout=30, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA busy_timeout = 30000")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """提供跨实体原子提交边界；业务层不得自行开启第二个连接。"""

        with self._connect() as db:
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise

    @staticmethod
    def _json(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def _model(model_type: type[T], value: str) -> T:
        return model_type.model_validate_json(value)  # type: ignore[attr-defined]

    def save_user(self, user: User) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO users(id,username,email,payload_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?)""",
                (user.id, user.username, user.email, user.model_dump_json(), user.created_at.isoformat(), user.updated_at.isoformat()),
            )
            db.commit()

    def count_users(self) -> int:
        with self._connect() as db:
            row = db.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0])

    def get_user(self, user_id: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM users WHERE id=?", (user_id,)).fetchone()
        return self._model(User, row[0]) if row else None

    def get_user_by_username(self, username: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM users WHERE username=?", (username.casefold(),)).fetchone()
        return self._model(User, row[0]) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM users WHERE email=?", (email.casefold(),)).fetchone()
        return self._model(User, row[0]) if row else None

    def save_session(self, session: UserSession) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO sessions(id,user_id,token_hash,payload_json,expires_at,created_at,last_seen_at)
                VALUES(?,?,?,?,?,?,?)""",
                (session.id, session.user_id, session.token_hash, session.model_dump_json(), session.expires_at.isoformat(), session.created_at.isoformat(), session.last_seen_at.isoformat() if session.last_seen_at else None),
            )
            db.commit()

    def get_session(self, token_hash: str) -> UserSession | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM sessions WHERE token_hash=?", (token_hash,)).fetchone()
        return self._model(UserSession, row[0]) if row else None

    def touch_session(self, session_id: str, timestamp) -> None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row:
                session = self._model(UserSession, row[0]).model_copy(update={"last_seen_at": timestamp})
                db.execute("UPDATE sessions SET payload_json=?, last_seen_at=? WHERE id=?", (session.model_dump_json(), timestamp.isoformat(), session_id))
                db.commit()

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            db.commit()
        return cursor.rowcount > 0

    def delete_session_by_token(self, token_hash: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            db.commit()
        return cursor.rowcount > 0

    def save_user_profile(self, profile: UserProfile) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO user_profiles(user_id,payload_json,updated_at) VALUES(?,?,?)
                ON CONFLICT(user_id) DO UPDATE SET payload_json=excluded.payload_json,
                updated_at=excluded.updated_at""",
                (profile.user_id, profile.model_dump_json(), profile.updated_at.isoformat()),
            )
            db.commit()

    def get_user_profile(self, user_id: str) -> UserProfile | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
        return self._model(UserProfile, row[0]) if row else None

    def upsert_conversation(self, conversation_id: str, title: str, timestamp: str, user_id: str | None = None) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO conversations(id,title,user_id,created_at,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                user_id=COALESCE(conversations.user_id, excluded.user_id), updated_at=excluded.updated_at""",
                (conversation_id, title, user_id, timestamp, timestamp),
            )
            db.commit()

    def create_conversation(self, title: str = "新对话", *, user_id: str | None = None) -> Conversation:
        conversation = Conversation(title=title, user_id=user_id)
        self.upsert_conversation(conversation.id, conversation.title, conversation.created_at.isoformat(), user_id)
        return conversation

    def list_conversations(self, limit: int = 50, *, user_id: str | None = None) -> list[Conversation]:
        query = "SELECT * FROM conversations"
        args: tuple[Any, ...] = ()
        if user_id is not None:
            query += " WHERE user_id=?"
            args = (user_id,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        args += (max(1, limit),)
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [Conversation.model_validate(dict(row)) for row in rows]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return Conversation.model_validate(dict(row)) if row else None

    def get_conversation_for_user(self, conversation_id: str, user_id: str) -> Conversation | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone()
        return Conversation.model_validate(dict(row)) if row else None

    def delete_conversation(self, conversation_id: str, *, user_id: str | None = None) -> bool:
        with self._connect() as db:
            query = "SELECT id FROM conversations WHERE id=?"
            args: tuple[str, ...] = (conversation_id,)
            if user_id is not None:
                query += " AND user_id=?"
                args += (user_id,)
            if db.execute(query, args).fetchone() is None:
                return False

            run_ids, task_ids = self._conversation_run_and_task_ids(db, conversation_id)
            self._preserve_user_resources(db, run_ids)
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                run_args = tuple(run_ids)
                db.execute(f"DELETE FROM trace_events WHERE run_id IN ({placeholders})", run_args)
                db.execute(f"DELETE FROM checkpoints WHERE run_id IN ({placeholders})", run_args)
                db.execute(f"DELETE FROM tool_calls WHERE run_id IN ({placeholders})", run_args)
                db.execute(f"DELETE FROM dataset_lineage WHERE run_id IN ({placeholders})", run_args)
                db.execute(
                    f"DELETE FROM approvals WHERE source_run_id IN ({placeholders}) "
                    f"OR json_extract(payload_json, '$.continuation_run_id') IN ({placeholders})",
                    (*run_args, *run_args),
                )
                db.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", run_args)
            db.execute("DELETE FROM approvals WHERE json_extract(payload_json, '$.conversation_id')=?", (conversation_id,))
            if _has_message_fts(db):
                db.execute("DELETE FROM messages_fts WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM conversation_memories WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM planning_sessions WHERE conversation_id=?", (conversation_id,))
            db.execute("DELETE FROM working_memories WHERE conversation_id=?", (conversation_id,))
            if task_ids:
                placeholders = ",".join("?" for _ in task_ids)
                task_args = tuple(task_ids)
                db.execute(f"DELETE FROM subtasks WHERE task_id IN ({placeholders})", task_args)
                db.execute(f"DELETE FROM working_memories WHERE task_id IN ({placeholders})", task_args)
                db.execute(f"DELETE FROM tasks WHERE id IN ({placeholders})", task_args)
            cursor = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            db.commit()
        return cursor.rowcount > 0

    def _conversation_run_and_task_ids(self, db: sqlite3.Connection, conversation_id: str) -> tuple[list[str], list[str]]:
        run_ids = {
            row[0]
            for row in db.execute(
                "SELECT id FROM runs WHERE conversation_id=?",
                (conversation_id,),
            ).fetchall()
        }
        while run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            child_ids = {
                row[0]
                for row in db.execute(
                    f"SELECT id FROM runs WHERE parent_run_id IN ({placeholders})",
                    tuple(run_ids),
                ).fetchall()
            }
            if child_ids.issubset(run_ids):
                break
            run_ids.update(child_ids)
        task_ids = {
            row[0]
            for row in db.execute(
                "SELECT id FROM tasks WHERE conversation_id=?",
                (conversation_id,),
            ).fetchall()
        }
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            task_ids.update(
                row[0]
                for row in db.execute(
                    f"SELECT DISTINCT task_id FROM runs "
                    f"WHERE id IN ({placeholders}) AND task_id IS NOT NULL",
                    tuple(run_ids),
                ).fetchall()
            )
        return list(run_ids), list(task_ids)

    def _preserve_user_resources(self, db: sqlite3.Connection, run_ids: list[str]) -> None:
        """删除会话事实前解除其派生资源的 Run 引用，不触碰实体文件。"""

        if not run_ids:
            return
        run_id_set = set(run_ids)
        for row in db.execute("SELECT id,payload_json FROM datasets").fetchall():
            dataset = self._model(Dataset, row[1])
            if dataset.created_by_run_id in run_id_set:
                preserved = dataset.model_copy(update={"created_by_run_id": None})
                db.execute("UPDATE datasets SET created_by_run_id=NULL, payload_json=? WHERE id=?", (preserved.model_dump_json(), row[0]))
        for row in db.execute("SELECT id,payload_json FROM artifacts").fetchall():
            artifact = self._model(Artifact, row[1])
            if artifact.run_id in run_id_set:
                preserved = artifact.model_copy(update={"run_id": None})
                db.execute("UPDATE artifacts SET run_id=NULL, payload_json=? WHERE id=?", (preserved.model_dump_json(), row[0]))

    def save_conversation_memory(self, memory: ConversationMemory) -> None:
        """写结构化记忆时保留并发产生的摘要游标与摘要内容。"""

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload_json FROM conversation_memories WHERE conversation_id=?",
                (memory.conversation_id,),
            ).fetchone()
            if row is not None:
                current = self._model(ConversationMemory, row[0])
                memory = memory.model_copy(
                    update={
                        "summary": current.summary,
                        "summary_version": current.summary_version,
                        "summarized_through_message_id": current.summarized_through_message_id,
                        "summary_updated_at": current.summary_updated_at,
                        "key_facts": _merge_conversation_entries(current.key_facts, memory.key_facts),
                        "decisions": _merge_conversation_entries(current.decisions, memory.decisions),
                        "important_references": _merge_conversation_entries(current.important_references, memory.important_references),
                    }
                )
            db.execute(
                """INSERT INTO conversation_memories(conversation_id,user_id,payload_json,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(conversation_id) DO UPDATE SET user_id=excluded.user_id,
                payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
                (memory.conversation_id, memory.user_id, memory.model_dump_json(), memory.updated_at.isoformat()),
            )
            db.commit()

    def commit_conversation_summary(
        self,
        *,
        conversation_id: str,
        user_id: str,
        expected_version: int,
        expected_through_message_id: str | None,
        through_message_id: str,
        summary: str,
        key_facts: list[ConversationMemoryEntry],
        decisions: list[ConversationMemoryEntry],
        important_references: list[ConversationMemoryEntry],
        unresolved_topics: list[ConversationMemoryEntry],
    ) -> bool:
        """CAS 摘要提交；与普通结构化记忆更新串行合并，避免互相覆盖。"""

        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload_json FROM conversation_memories WHERE conversation_id=? AND user_id=?",
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                db.rollback()
                return False
            current = self._model(ConversationMemory, row[0])
            if (
                current.summary_version != expected_version
                or current.summarized_through_message_id != expected_through_message_id
            ):
                db.rollback()
                return False
            updated = current.model_copy(
                update={
                    "summary": summary,
                    "key_facts": _merge_conversation_entries(current.key_facts, key_facts),
                    "decisions": _merge_conversation_entries(current.decisions, decisions),
                    "important_references": _merge_conversation_entries(current.important_references, important_references),
                    "unresolved_topics": _merge_conversation_entries(current.unresolved_topics, unresolved_topics),
                    "summary_version": current.summary_version + 1,
                    "summarized_through_message_id": through_message_id,
                    "summary_updated_at": now,
                    "updated_at": now,
                }
            )
            db.execute(
                "UPDATE conversation_memories SET payload_json=?,updated_at=? WHERE conversation_id=? AND user_id=?",
                (updated.model_dump_json(), now.isoformat(), conversation_id, user_id),
            )
            db.commit()
            return True

    def get_conversation_memory_for_user(self, conversation_id: str, user_id: str) -> ConversationMemory | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM conversation_memories WHERE conversation_id=? AND user_id=?", (conversation_id, user_id)).fetchone()
        return self._model(ConversationMemory, row[0]) if row else None

    def save_message(self, message: Message) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO messages
                (id,conversation_id,role,content,run_id,dataset_ids_json,created_at) VALUES(?,?,?,?,?,?,?)""",
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.run_id,
                    self._json(list(dict.fromkeys(message.dataset_ids))),
                    message.created_at.isoformat(),
                ),
            )
            if _has_message_fts(db):
                db.execute("DELETE FROM messages_fts WHERE message_id=?", (message.id,))
                db.execute(
                    "INSERT INTO messages_fts(message_id,conversation_id,role,content) VALUES(?,?,?,?)",
                    (message.id, message.conversation_id, message.role, message.content),
                )
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (message.created_at.isoformat(), message.conversation_id))
            db.commit()

    def list_messages(self, conversation_id: str, limit: int = 100) -> list[Message]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT * FROM (
                    SELECT rowid AS _rowid, * FROM messages
                    WHERE conversation_id=?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                ) ORDER BY created_at ASC, _rowid ASC""",
                (conversation_id, max(1, limit)),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def list_messages_after(self, conversation_id: str, message_id: str | None) -> list[Message]:
        """按 SQLite 持久化顺序返回摘要游标之后的原始消息。"""

        with self._connect() as db:
            if message_id is None:
                cursor_rowid = 0
            else:
                cursor = db.execute(
                    "SELECT rowid FROM messages WHERE id=? AND conversation_id=?",
                    (message_id, conversation_id),
                ).fetchone()
                if cursor is None:
                    return []
                cursor_rowid = int(cursor[0])
            rows = db.execute(
                "SELECT rowid AS _rowid,* FROM messages WHERE conversation_id=? AND rowid>? ORDER BY rowid",
                (conversation_id, cursor_rowid),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def search_messages(
        self,
        conversation_id: str,
        query: str,
        *,
        limit: int = 5,
        exclude_message_ids: set[str] | None = None,
    ) -> list[Message]:
        """限定单个会话的全文检索；FTS 不可用或无命中时回退到 LIKE。"""

        text = re.sub(r"\s+", " ", query).strip()
        if not text:
            return []
        excluded = sorted(exclude_message_ids or ())
        exclusion_sql = f" AND m.id NOT IN ({','.join('?' for _ in excluded)})" if excluded else ""
        with self._connect() as db:
            rows = []
            if _has_message_fts(db) and len(text) >= 3:
                terms = _trigram_terms(text)
                match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
                if match_query:
                    try:
                        rows = db.execute(
                            f"""SELECT m.rowid AS _rowid,m.* FROM messages_fts f
                            JOIN messages m ON m.id=f.message_id
                            WHERE messages_fts MATCH ? AND f.conversation_id=?{exclusion_sql}
                            ORDER BY bm25(messages_fts),m.rowid DESC LIMIT ?""",
                            (match_query, conversation_id, *excluded, max(1, limit)),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = []
            if not rows:
                rows = db.execute(
                    f"""SELECT rowid AS _rowid,* FROM messages m
                    WHERE conversation_id=? AND content LIKE ?{exclusion_sql}
                    ORDER BY rowid DESC LIMIT ?""",
                    (conversation_id, f"%{text}%", *excluded, max(1, limit)),
                ).fetchall()
        messages = [self._message_from_row(row) for row in rows]
        messages.reverse()
        return messages

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        payload = dict(row)
        payload.pop("_rowid", None)
        raw_dataset_ids = payload.pop("dataset_ids_json", "[]")
        try:
            dataset_ids = json.loads(raw_dataset_ids)
        except (TypeError, json.JSONDecodeError):
            dataset_ids = []
        payload["dataset_ids"] = [item for item in dataset_ids if isinstance(item, str)] if isinstance(dataset_ids, list) else []
        return Message.model_validate(payload)

    def save_task(self, task: Task) -> None:
        with self._connect() as db:
            self._save_task(db, task)
            db.commit()

    @staticmethod
    def _save_task(db: sqlite3.Connection, task: Task) -> None:
        db.execute(
            """INSERT OR REPLACE INTO tasks
            (id,conversation_id,goal,status,result,created_at,payload_json,updated_at)
            VALUES(?,?,?,?,?,?,?,?)""",
            (task.id, task.conversation_id, task.goal, task.status.value, task.result, task.created_at.isoformat(), task.model_dump_json(), task.updated_at.isoformat()),
        )

    def save_run_and_task(self, run: Run, task: Task | None = None) -> None:
        """原子保存 Run 及其对应 Task，禁止状态转换出现半提交。"""

        with self.transaction() as db:
            self._save_run(db, run)
            if task is not None:
                self._save_task(db, task)

    def get_task(self, task_id: str) -> Task | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._model(Task, row[0]) if row else None

    def list_tasks(self, conversation_id: str | None = None, limit: int = 50) -> list[Task]:
        query = "SELECT payload_json FROM tasks"
        args: tuple[Any, ...] = ()
        if conversation_id:
            query += " WHERE conversation_id=?"
            args = (conversation_id,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        args += (max(1, limit),)
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [self._model(Task, row[0]) for row in rows]

    def save_working_memory(self, memory: WorkingMemory) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO working_memories(task_id,conversation_id,payload_json,updated_at) VALUES(?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET conversation_id=excluded.conversation_id,
                payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
                (memory.task_id, memory.conversation_id, memory.model_dump_json(), memory.updated_at.isoformat()),
            )
            db.commit()

    def get_working_memory(self, task_id: str) -> WorkingMemory | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM working_memories WHERE task_id=?", (task_id,)).fetchone()
        return self._model(WorkingMemory, row[0]) if row else None

    def save_subtask(self, task_id: str, subtask: SubTask) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO subtasks(id,task_id,payload_json) VALUES(?,?,?)",
                (subtask.id, task_id, subtask.model_dump_json()),
            )
            db.commit()

    def get_subtask(self, task_id: str, subtask_id: str) -> SubTask | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM subtasks WHERE task_id=? AND id=?", (task_id, subtask_id)).fetchone()
        return self._model(SubTask, row[0]) if row else None

    def save_run(self, run: Run) -> None:
        with self._connect() as db:
            self._save_run(db, run)
            db.commit()

    @staticmethod
    def _save_run(db: sqlite3.Connection, run: Run) -> None:
        db.execute(
            """INSERT OR REPLACE INTO runs
            (id,conversation_id,task_id,parent_run_id,agent_id,status,started_at,finished_at,error,
             turn_count,tool_call_count,metadata_json,payload_json,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run.id,
                run.conversation_id,
                run.task_id,
                run.parent_run_id,
                run.agent_id,
                run.status.value,
                run.started_at.isoformat() if run.started_at else None,
                run.finished_at.isoformat() if run.finished_at else None,
                run.error,
                run.turn_count,
                run.tool_call_count,
                StateStore._json(run.metadata),
                run.model_dump_json(),
                utc_now().isoformat(),
            ),
        )

    def get_run(self, run_id: str) -> Run | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._model(Run, row[0]) if row else None

    def list_child_runs(self, parent_run_id: str) -> list[Run]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM runs WHERE parent_run_id=? ORDER BY updated_at", (parent_run_id,)).fetchall()
        return [self._model(Run, row[0]) for row in rows]

    def run_belongs_to_user(self, run_id: str, user_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                """SELECT 1 FROM runs r JOIN conversations c
                ON r.conversation_id=c.id
                WHERE r.id=? AND c.user_id=?""",
                (run_id, user_id),
            ).fetchone()
        return row is not None

    def user_id_for_run(self, run_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT c.user_id FROM runs r JOIN conversations c
                ON r.conversation_id=c.id
                WHERE r.id=?""",
                (run_id,),
            ).fetchone()
        return row[0] if row else None

    def list_runs(self, limit: int = 50, *, user_id: str | None = None) -> list[Run]:
        query = "SELECT r.payload_json FROM runs r"
        args: tuple[Any, ...] = ()
        if user_id is not None:
            query += " JOIN conversations c ON r.conversation_id=c.id WHERE c.user_id=?"
            args = (user_id,)
        query += " ORDER BY r.updated_at DESC LIMIT ?"
        args += (max(1, limit),)
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [self._model(Run, row[0]) for row in rows]

    def list_runs_for_conversation(self, conversation_id: str, limit: int = 10000) -> list[Run]:
        """读取单个 Conversation 的运行记录，供生命周期保护和按需详情使用。"""

        with self._connect() as db:
            rows = db.execute(
                """SELECT payload_json FROM runs
                WHERE conversation_id=?
                ORDER BY updated_at DESC LIMIT ?""",
                (conversation_id, max(1, limit)),
            ).fetchall()
        return [self._model(Run, row[0]) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        return bool(self.delete_runs([run_id]))

    def delete_runs(self, run_ids: list[str]) -> list[str]:
        deleted: list[str] = []
        task_ids: set[str] = set()
        with self._connect() as db:
            for run_id in dict.fromkeys(run_ids):
                row = db.execute("SELECT payload_json FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    continue
                run = self._model(Run, row[0])
                self._preserve_user_resources(db, [run_id])
                db.execute("DELETE FROM trace_events WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM checkpoints WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM tool_calls WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM dataset_lineage WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM approvals WHERE source_run_id=? OR json_extract(payload_json, '$.continuation_run_id')=?", (run_id, run_id))
                if run.task_id:
                    task_ids.add(run.task_id)
                db.execute("DELETE FROM runs WHERE id=?", (run_id,))
                deleted.append(run_id)
            for task_id in task_ids:
                remaining = db.execute("SELECT 1 FROM runs WHERE task_id=? LIMIT 1", (task_id,)).fetchone()
                if remaining is None:
                    db.execute("DELETE FROM subtasks WHERE task_id=?", (task_id,))
                    db.execute("DELETE FROM working_memories WHERE task_id=?", (task_id,))
                    db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            db.commit()
        return deleted

    def save_approval(self, approval: ApprovalRequest) -> None:
        with self._connect() as db:
            self._save_approval(db, approval)
            db.commit()

    @staticmethod
    def _save_approval(db: sqlite3.Connection, approval: ApprovalRequest) -> None:
        db.execute(
            """INSERT OR REPLACE INTO approvals(id,user_id,source_run_id,status,payload_json,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?)""",
            (approval.id, approval.user_id, approval.source_run_id, approval.status.value, approval.model_dump_json(), approval.created_at.isoformat(), utc_now().isoformat()),
        )

    def save_approval_and_run(self, approval: ApprovalRequest, run: Run, task: Task | None = None) -> None:
        """原子保存审批决定及其对应的 Run/Task 状态。"""

        with self.transaction() as db:
            self._save_approval(db, approval)
            self._save_run(db, run)
            if task is not None:
                self._save_task(db, task)

    def get_approval(self, approval_id: str, *, user_id: str | None = None) -> ApprovalRequest | None:
        query = "SELECT payload_json FROM approvals WHERE id=?"
        args: tuple[Any, ...] = (approval_id,)
        if user_id is not None:
            query += " AND user_id=?"
            args += (user_id,)
        with self._connect() as db:
            row = db.execute(query, args).fetchone()
        return self._model(ApprovalRequest, row[0]) if row else None

    def list_approvals(self, user_id: str, *, status: ApprovalStatus | None = None, limit: int = 50) -> list[ApprovalRequest]:
        query = "SELECT payload_json FROM approvals WHERE user_id=?"
        args: list[Any] = [user_id]
        if status is not None:
            query += " AND status=?"
            args.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, limit))
        with self._connect() as db:
            rows = db.execute(query, tuple(args)).fetchall()
        return [self._model(ApprovalRequest, row[0]) for row in rows]

    def update_approval(self, approval: ApprovalRequest) -> None:
        self.save_approval(approval)

    def find_pending_approval(self, *, user_id: str, source_run_id: str, tool_name: str, argument_fingerprint: str) -> ApprovalRequest | None:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload_json FROM approvals WHERE user_id=? AND source_run_id=? AND status=? ORDER BY created_at DESC",
                (user_id, source_run_id, ApprovalStatus.PENDING.value),
            ).fetchall()
        for row in rows:
            approval = self._model(ApprovalRequest, row[0])
            if approval.tool_name == tool_name and approval.argument_fingerprint == argument_fingerprint:
                return approval
        return None

    def consume_approval(
        self,
        *,
        approval_id: str,
        user_id: str,
        run_id: str | None = None,
        continuation_run_id: str | None = None,
        tool_name: str,
        argument_fingerprint: str,
    ) -> ApprovalRequest | None:
        now = utc_now()
        expected_run_id = run_id or continuation_run_id
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM approvals WHERE id=? AND user_id=?", (approval_id, user_id)).fetchone()
            if row is None:
                return None
            approval = self._model(ApprovalRequest, row[0])
            if approval.status is not ApprovalStatus.APPROVED or expected_run_id is None or approval.source_run_id != expected_run_id and approval.continuation_run_id != expected_run_id or approval.tool_name != tool_name or approval.argument_fingerprint != argument_fingerprint:
                return None
            consumed = approval.model_copy(update={"status": ApprovalStatus.CONSUMED, "consumed_at": now})
            db.execute("UPDATE approvals SET status=?,payload_json=?,updated_at=? WHERE id=? AND status=?", (consumed.status.value, consumed.model_dump_json(), now.isoformat(), approval_id, ApprovalStatus.APPROVED.value))
            if db.total_changes != 1:
                return None
            db.commit()
        return consumed

    def save_tool_call(self, call: ToolCall, result: ToolResult | None = None) -> None:
        status = ToolExecutionStatus.COMPLETED.value if result is not None else ToolExecutionStatus.PENDING.value
        timestamp = utc_now().isoformat()
        with self._connect() as db:
            db.execute(
                """INSERT INTO tool_calls
                (id,run_id,name,arguments_json,status,result_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                run_id=excluded.run_id,
                name=excluded.name,
                arguments_json=excluded.arguments_json,
                status=excluded.status,
                result_json=COALESCE(excluded.result_json, tool_calls.result_json),
                updated_at=excluded.updated_at""",
                (call.id, call.run_id, call.name, self._json(call.arguments), status, result.model_dump_json() if result else None, timestamp, timestamp),
            )
            db.commit()

    def get_tool_call(self, call_id: str) -> tuple[ToolExecutionStatus, ToolResult | None] | None:
        with self._connect() as db:
            row = db.execute("SELECT status,result_json FROM tool_calls WHERE id=?", (call_id,)).fetchone()
        if row is None:
            return None
        result = self._model(ToolResult, row[1]) if row[1] else None
        return ToolExecutionStatus(row[0]), result

    def mark_tool_call_running(self, call: ToolCall) -> bool:
        timestamp = utc_now().isoformat()
        with self._connect() as db:
            row = db.execute("SELECT status FROM tool_calls WHERE id=?", (call.id,)).fetchone()
            if row is None:
                db.execute(
                    """INSERT INTO tool_calls(id,run_id,name,arguments_json,status,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?)""",
                    (call.id, call.run_id, call.name, self._json(call.arguments), ToolExecutionStatus.RUNNING.value, timestamp, timestamp),
                )
                db.commit()
                return True
            current = ToolExecutionStatus(row[0])
            if current is ToolExecutionStatus.RUNNING:
                return False
            if current is ToolExecutionStatus.COMPLETED:
                return False
            db.execute(
                "UPDATE tool_calls SET status=?, run_id=?, name=?, arguments_json=?, updated_at=? WHERE id=?",
                (ToolExecutionStatus.RUNNING.value, call.run_id, call.name, self._json(call.arguments), timestamp, call.id),
            )
            db.commit()
            return True

    def save_tool_call_result(self, call_id: str, result: ToolResult) -> None:
        status = (
            ToolExecutionStatus.BLOCKED
            if result.status is ToolStatus.BLOCKED
            else ToolExecutionStatus.COMPLETED
            if result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}
            else ToolExecutionStatus.FAILED
        )
        with self._connect() as db:
            db.execute("UPDATE tool_calls SET status=?, result_json=?, updated_at=? WHERE id=?", (status.value, result.model_dump_json(), utc_now().isoformat(), call_id))
            db.commit()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO checkpoints(id,run_id,phase,payload_json,created_at) VALUES(?,?,?,?,?)",
                (checkpoint.id, checkpoint.run_id, checkpoint.phase, checkpoint.model_dump_json(), checkpoint.created_at.isoformat()),
            )
            db.commit()

    def latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload_json FROM checkpoints WHERE run_id=? ORDER BY created_at DESC LIMIT 1", (run_id,)
            ).fetchone()
        return self._model(Checkpoint, row[0]) if row else None

    def save_dataset(self, dataset: Dataset) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO datasets(id,owner_user_id,created_by_run_id,payload_json,created_at) VALUES(?,?,?,?,?)",
                (dataset.id, dataset.owner_user_id, dataset.created_by_run_id, dataset.model_dump_json(), dataset.created_at.isoformat()),
            )
            db.commit()

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return self._model(Dataset, row[0]) if row else None

    def get_dataset_for_user(self, dataset_id: str, user_id: str) -> Dataset | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM datasets WHERE id=? AND (owner_user_id IS NULL OR owner_user_id=?)", (dataset_id, user_id)).fetchone()
        return self._model(Dataset, row[0]) if row else None

    def list_datasets(self, kind: str | None = None) -> list[Dataset]:
        query = "SELECT payload_json FROM datasets"
        args: tuple[Any, ...] = ()
        if kind:
            query += " WHERE json_extract(payload_json, '$.kind')=?"
            args = (kind,)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [self._model(Dataset, row[0]) for row in rows]

    def list_datasets_for_user(self, user_id: str, kind: str | None = None) -> list[Dataset]:
        query = "SELECT payload_json FROM datasets WHERE (owner_user_id IS NULL OR owner_user_id=?)"
        args: list[Any] = [user_id]
        if kind:
            query += " AND json_extract(payload_json, '$.kind')=?"
            args.append(kind)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, tuple(args)).fetchall()
        return [self._model(Dataset, row[0]) for row in rows]

    def save_lineage(
        self,
        *,
        lineage_id: str,
        run_id: str | None,
        operation: str,
        input_dataset_ids: list[str],
        output_dataset_id: str,
        tool_call_id: str | None,
        parameters: dict[str, Any],
        created_at: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO dataset_lineage
                (id,run_id,operation,input_dataset_ids_json,output_dataset_id,tool_call_id,parameters_json,created_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (lineage_id, run_id, operation, self._json(input_dataset_ids), output_dataset_id, tool_call_id, self._json(parameters), created_at),
            )
            db.commit()

    def list_lineage(self, output_dataset_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM dataset_lineage"
        args: tuple[Any, ...] = ()
        if output_dataset_id:
            query += " WHERE output_dataset_id=?"
            args = (output_dataset_id,)
        query += " ORDER BY created_at"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "operation": row["operation"],
                "input_dataset_ids": json.loads(row["input_dataset_ids_json"]),
                "output_dataset_id": row["output_dataset_id"],
                "tool_call_id": row["tool_call_id"],
                "parameters": json.loads(row["parameters_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_artifact(self, artifact: Artifact) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO artifacts(id,owner_user_id,run_id,payload_json,created_at) VALUES(?,?,?,?,?)",
                (artifact.id, artifact.owner_user_id, artifact.run_id, artifact.model_dump_json(), artifact.created_at.isoformat()),
            )
            db.commit()

    def list_artifacts(self, run_id: str | None = None) -> list[Artifact]:
        with self._connect() as db:
            if run_id is None:
                rows = db.execute("SELECT payload_json FROM artifacts ORDER BY created_at DESC").fetchall()
            else:
                rows = db.execute("SELECT payload_json FROM artifacts WHERE run_id=? ORDER BY created_at DESC", (run_id,)).fetchall()
        return [self._model(Artifact, row[0]) for row in rows]

    def list_artifacts_for_user(self, user_id: str, run_id: str | None = None) -> list[Artifact]:
        query = "SELECT payload_json FROM artifacts WHERE (owner_user_id IS NULL OR owner_user_id=?)"
        args: list[Any] = [user_id]
        if run_id is not None:
            query += " AND run_id=?"
            args.append(run_id)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, tuple(args)).fetchall()
        return [self._model(Artifact, row[0]) for row in rows]

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        return self._model(Artifact, row[0]) if row else None

    def get_artifact_for_user(self, artifact_id: str, user_id: str) -> Artifact | None:
        with self._connect() as db:
            row = db.execute("SELECT payload_json FROM artifacts WHERE id=? AND (owner_user_id IS NULL OR owner_user_id=?)", (artifact_id, user_id)).fetchone()
        return self._model(Artifact, row[0]) if row else None

    def save_memory(self, memory: MemoryItem) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO memories(id,owner_user_id,scope,memory_key,payload_json,updated_at) VALUES(?,?,?,?,?,?)
                ON CONFLICT(owner_user_id,scope,memory_key) DO UPDATE SET id=excluded.id,
                payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (memory.id, memory.owner_user_id, memory.scope, memory.key, memory.model_dump_json(), memory.updated_at.isoformat()),
            )
            db.commit()

    def list_memories(self, scope: str = "project", *, owner_user_id: str | None = None) -> list[MemoryItem]:
        query = "SELECT payload_json FROM memories WHERE scope=?"
        args: tuple[Any, ...] = (scope,)
        if owner_user_id is not None:
            query += " AND owner_user_id=?"
            args += (owner_user_id,)
        query += " ORDER BY updated_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        return [self._model(MemoryItem, row[0]) for row in rows]

    def record_event(self, event: TraceEvent) -> TraceEvent:
        with self._connect() as db:
            row = db.execute("SELECT COALESCE(MAX(sequence), -1) FROM trace_events WHERE run_id=?", (event.run_id,)).fetchone()
            sequence = max(event.sequence, int(row[0]) + 1)
            event = event.model_copy(update={"sequence": sequence})
            db.execute(
                "INSERT OR IGNORE INTO trace_events(id,run_id,sequence,payload_json) VALUES(?,?,?,?)",
                (event.id, event.run_id, event.sequence, event.model_dump_json()),
            )
            db.commit()
        return event

    def list_events(self, run_id: str) -> list[TraceEvent]:
        with self._connect() as db:
            rows = db.execute("SELECT payload_json FROM trace_events WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()
        return [self._model(TraceEvent, row[0]) for row in rows]

def _has_message_fts(db: sqlite3.Connection) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone() is not None


def _trigram_terms(text: str, *, limit: int = 48) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"[\w]+", text, flags=re.UNICODE):
        if len(word) < 3:
            continue
        for index in range(len(word) - 2):
            term = word[index : index + 3]
            if term not in terms:
                terms.append(term)
            if len(terms) >= limit:
                return terms
    return terms


def _merge_conversation_entries(
    current: list[ConversationMemoryEntry],
    incoming: list[ConversationMemoryEntry],
    *,
    limit: int = 20,
) -> list[ConversationMemoryEntry]:
    result = list(current)
    seen = {
        (item.reference_type, item.reference_id)
        if item.reference_type and item.reference_id
        else (item.content.casefold(), item.source_message_id)
        for item in result
    }
    for item in incoming:
        key = (
            (item.reference_type, item.reference_id)
            if item.reference_type and item.reference_id
            else (item.content.casefold(), item.source_message_id)
        )
        if item.content.strip() and key not in seen:
            result.append(item)
            seen.add(key)
    return result[-limit:]


__all__ = ["StateStore"]
