"""独立知识 SQLite：原文片段、来源和向量一起版本化，事务发布。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class KnowledgeStore:
    def __init__(self, directory: Path) -> None:
        self.path = directory / "knowledge.sqlite3"

    @contextmanager
    def connect(self, *, write: bool = False):
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path if write else self.path.as_uri() + "?mode=ro", uri=not write, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    INSERT OR IGNORE INTO metadata VALUES ('revision', '0');
                    CREATE TABLE IF NOT EXISTS collections (
                        name TEXT PRIMARY KEY, imported_at TEXT NOT NULL, documents INTEGER NOT NULL);
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY, collection TEXT NOT NULL, payload TEXT NOT NULL,
                        content_hash TEXT NOT NULL, embedding_key TEXT, vector BLOB);
                    CREATE INDEX IF NOT EXISTS chunks_collection ON chunks(collection);
                """)
            with db:
                yield db
        finally:
            db.close()

    def revision(self) -> int:
        if not self.path.exists():
            return 0
        with self.connect() as db:
            return int(db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])

    def snapshot(self) -> tuple[int, list[dict]]:
        if not self.path.exists():
            return 0, []
        with self.connect() as db:
            db.execute("BEGIN")
            revision = int(db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])
            rows = db.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        return revision, [{**json.loads(row["payload"]), "embedding_key": row["embedding_key"],
                           "vector": np.frombuffer(row["vector"], dtype="<f4").copy() if row["vector"] else None}
                          for row in rows]

    def replace_collection(self, name: str, chunks: list[dict]) -> dict:
        """只有解析全部成功后才替换这个来源集合；其他集合不变。相同正文保留向量。"""
        with self.connect(write=True) as db:
            db.execute("BEGIN IMMEDIATE")
            old = {row["id"]: row for row in db.execute("SELECT * FROM chunks WHERE collection=?", (name,))}
            prepared = {item["id"]: (item, json.dumps(item, ensure_ascii=False, sort_keys=True)) for item in chunks}
            changed = sum(identifier not in old or old[identifier]["payload"] != payload
                          for identifier, (_, payload) in prepared.items())
            removed = set(old) - prepared.keys()
            if not changed and not removed:
                return {"chunks": len(chunks), "changed": 0, "removed": 0}
            for identifier, (item, payload) in prepared.items():
                previous = old.get(identifier)
                same_text = previous is not None and previous["content_hash"] == item["content_hash"]
                db.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
                           (identifier, name, payload, item["content_hash"],
                            previous["embedding_key"] if same_text else None,
                            previous["vector"] if same_text else None))
            db.executemany("DELETE FROM chunks WHERE id=?", ((identifier,) for identifier in removed))
            db.execute("INSERT OR REPLACE INTO collections VALUES (?,?,?)",
                       (name, datetime.now(UTC).isoformat(), len({item["document_id"] for item in chunks})))
            db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
        return {"chunks": len(chunks), "changed": changed, "removed": len(removed)}

    def publish_vectors(self, revision: int, key: str, vectors: dict[str, np.ndarray]) -> None:
        """网络请求不占数据库锁；发布时检查导入版本，防止给已变更正文套上旧向量。"""
        with self.connect(write=True) as db:
            db.execute("BEGIN IMMEDIATE")
            if int(db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0]) != revision:
                raise ValueError("资料在嵌入期间已更新，请重新执行 embed；旧索引未被覆盖。")
            db.executemany("UPDATE chunks SET embedding_key=?,vector=? WHERE id=?",
                           ((key, np.asarray(vector, dtype="<f4").tobytes(), identifier)
                            for identifier, vector in vectors.items()))
            if vectors:
                db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
